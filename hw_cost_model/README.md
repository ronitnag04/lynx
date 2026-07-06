# ProtoAcc hardware cost model

Anchored, per-submodule hardware cost predictor for the ProtoAcc
Serializer and Deserializer. Ground-truth data comes from **Yosys
cell counts** (not Sky130 area) -- deterministic, PDK-free, ~3-4 min
per config. Consumed by
[`ml_model/optimize_hw_config.py`](../ml_model/optimize_hw_config.py)
and
[`ml_model/exhaustive_search.py`](../ml_model/exhaustive_search.py) as
the second (and third) objective in the Pareto search.

## The cost is a 2-vector

Yosys reports `Number of cells:` per module, but firtool emits every
Chisel `Mem` / `SyncReadMem` as an explicit
`ram_<D>x<W>` register-file module. Yosys leaves those as flip-flops,
so a naive `total cells` count overweights deep queues by ~40x
relative to real SRAM macros. We split the cost:

| axis | source |
|---|---|
| `logic_cells` | transitive primitive-cell count *excluding* `ram_<D>x<W>` submodules |
| `ram_bits`    | Σ D·W across every `ram_<D>x<W>` reachable from the top |

Downstream tools can either optimize both directly (3-objective NSGA)
or collapse to a scalar via a user-selected weight κ:

    scalar_cost = logic_cells + κ · ram_bits

Default **κ = 0.15** gate-equivalents per SRAM bit -- roughly what a
compiled SRAM macro would cost per bit in a modern std-cell library.
It is **not** baked into the fitted model; every consumer passes κ at
prediction time.

## Files

- [`defaults.py`](defaults.py) -- parameter names, sweep grids, and
  defaults. Single source of truth shared with the ML model.
- [`structural_features.py`](structural_features.py) -- RTL-derived
  features per config. Every feature comes from a specific Chisel line
  (documented in the file); each feature is annotated with the
  submodule bucket it belongs to.
- [`hw_cost_model.py`](hw_cost_model.py) -- two model classes and the
  top-level API:
  - `StructuralCostModel` -- untrained bit-count sum used before any
    Yosys data exists. Monotone in every knob by construction.
  - `TrainedCostModel` -- per-bucket, per-label (`logic_cells` +
    `ram_bits`) additive predictor fit from a Yosys sweep. Loads/saves
    as a joblib file.
  - `hardware_cost_2vec(cfgs, side)` -- returns `(N, 2)` array of
    `(logic_cells, ram_bits)` predictions. Preferred for
    multi-objective search.
  - `hardware_cost(cfgs, side, kappa=0.15)` -- scalar wrapper
    returning `logic_cells + kappa * ram_bits`.
  - `use_trained_model(side, path)` -- register once at CLI startup;
    all subsequent `hardware_cost*` calls on that side use the trained
    model.
- [`parse_yosys_stat.py`](parse_yosys_stat.py) -- parses a Yosys
  `synth.log`, walks the module hierarchy (with cycle-safe
  memoization), attributes every reachable cell to a bucket, splits
  `ram_<D>x<W>` cells into `ram_bits`, and appends one row to a CSV
  under `flock`. Runnable as `python -m hw_cost_model.parse_yosys_stat`.
- [`fit_from_yosys.py`](fit_from_yosys.py) -- fits a
  `TrainedCostModel` from the CSV. Two independent NNLS fits per
  bucket (one for `logic_cells`, one for `ram_bits`); non-negative
  coefficients guarantee monotonicity, which is essential for correct
  Pareto search. Prints per-bucket + top-level MAPE and Spearman ρ on
  a 10 % held-out split. `--kappa` is used only for a scalar
  diagnostic; not baked into the model.
- [`run_yosys_sweep.sh`](run_yosys_sweep.sh) -- GNU-parallel driver
  that:
  1. Pulls `generated-src/` from `s3://ronitnag04-lynx/verilator_build_files/<cls>.zip`
     if the cached zip exists (huge time saver on random sweeps we've
     already elaborated). Falls back to a local `make verilog` on cache
     miss.
  2. Runs `sv2v -DSYNTHESIS` on the .sv files.
  3. Runs Yosys (`read_verilog` + `hierarchy -top` + `synth` + `stat`).
  4. Parses the resulting `synth.log` and appends one row per config to
     the output CSV under `flock`.

## The bucket structure

Buckets **partition** the module hierarchy -- every module belongs to
exactly one bucket. The parser enforces this by treating every other
bucket's top-module names as "stop-at" nodes during the reachability
walk.

### DES buckets (12)

| bucket | Yosys tops | driven by knobs |
|---|---|---|
| `top`      | `ProtoAccel` | (RoCC glue only) |
| `cr`       | `CommandRouter` | `des_cr_rocc_commands` |
| `dth`      | `DescriptorTableHandler` | `des_dth_l1_reqs`, `des_dth_fd_reqs`, `des_dth_fd_resps` |
| `fh`       | `FieldHandler` | -- |
| `fw`       | `FixedWriter` | `des_fw_l1_reqs` |
| `hbw`      | `HasBitsWriter` | -- (fixed depth 10) |
| `ml`       | `MemLoader` | `des_ml_buf_info_q`, `des_ml_load_info_q` |
| `descr_l1` | `L1MemHelper` (base) | `des_top_descriptor_reqs` |
| `ml_l1`    | `L1MemHelper_1` | `des_top_memloader_reqs` |
| `hbw_l1`   | `L1MemHelper_2` | -- (default 32 outstanding) |
| `fw_l1`    | `L1MemHelperWriteFast` | -- (default 4 outstanding) |
| `tlb`      | `DTLB_2`, `PMAChecker`, `PMPChecker_s3`, `OptimizationBarrier_TLBEntryData` | -- (fixed) |
| `tl_glue`  | `TLBuffer*`, `TLXbar*`, `TLWidthWidget*`, `Repeater*` (prefix-matched) | -- (fixed) |

The three plain `L1MemHelper*` instances are ordered by firtool in
Chisel-source order (`mem_descr`, `mem_memloader`, `mem_hasbits`), so
`L1MemHelper` → descr, `L1MemHelper_1` → ml, `L1MemHelper_2` → hbw. The
write-fast variant (`mem_fixedwriter`) is a different module class.

### SER buckets (11)

| bucket | Yosys tops | driven by knobs |
|---|---|---|
| `top`      | `ProtoAccelSerializer` | (RoCC glue only) |
| `cr`       | `CommandRouterSerializer` | `ser_cr_rocc_commands` |
| `dth`      | `SerDescriptorTableHandler` | `ser_dth_hasbits_reqs`, `..._descriptor_reqs`, `..._reg_resps`, `..._reqs_meta`, `..._fh_outputs` |
| `mfh`      | `MultiFieldHandler`, `FieldDispatchRouter`, `MemWriteArbiter` | `ser_field_handlers` |
| `fh`       | `SerFieldHandler` | `ser_field_handlers` |
| `mw`       | `SerMemwriter` | `ser_mw_write_input`, `..._inject`, `..._ptrs` |
| `varint`   | `CombinationalVarintEncode` | `ser_field_handlers` (3 per handler + 1 in mw) |
| `fh_l1`    | (per-handler `L1MemHelper` -- see note below) | `ser_field_handlers` |
| `mw_l1`    | `L1MemHelperWriteFast` | -- (fixed) |
| `tlb`      | (same as DES) | -- |
| `tl_glue`  | (same as DES) | -- |

The SER side instantiates `ser_field_handlers` many `L1MemHelper`
instances; the parser assigns each one to `fh_l1` (verified once by
inspecting the response-FIFO-vec size on a first synth log).

## Sampling plan

Reuse `gen_protoacc_sweep_configs.py` (already exists) as the sole
source of Scala configs. For the training set (per side, ~300 configs
recommended):

- **Defaults** (1 row): baseline.
- **Corner mass** (2 rows): all-min and all-max on the side.
- **OFAT** (~40 rows): `-t ofat --emit <side>` -- every value of every
  knob one at a time; freezes the other side at defaults.
- **Latin hypercube** (~200 rows): `-t random -n 200 --emit <side>`.
- **Two-axis stress** (10-20 hand-picked): configs that push 2-3 axes
  to extremes; the Pareto-corner regime.
- **Held-out random** (~30 rows): `-t random -n 30 -s <different seed>`,
  never seen by the fit; used to report the final generalization number.

The S3 build cache means random-sample sweeps we've already run for
Verilator experiments come "free" (zip pull is much faster than
elaboration). OFAT and stress corners will mostly be cache misses.

## Full pipeline

```bash
conda activate /home/ubuntu/lynx-chipyard/.conda-env

# 1. Generate the sweep configs (fills chipyard/config/*.scala + sweep_configs.csv).
cd generators/protoacc/software/verilator-bench
python3 gen_protoacc_sweep_configs.py --emit des -t random -n 200 -s 42
python3 gen_protoacc_sweep_configs.py --emit ser -t random -n 200 -s 42
# ... plus OFAT / corners as above.

# 2. Run the Yosys sweep. Uses the same S3 cache as run_sweep.sh.
cd ../lynx/hw_cost_model
./run_yosys_sweep.sh \
    --side des \
    --sweep-csv ../../verilator-bench/sweep_configs.csv \
    --output    yosys_sweep_results_des.csv \
    --workers 32 --pull-s3-builds --push-s3-builds

./run_yosys_sweep.sh \
    --side ser \
    --sweep-csv ../../verilator-bench/sweep_configs.csv \
    --output    yosys_sweep_results_ser.csv \
    --workers 32 --pull-s3-builds --push-s3-builds

# 3. Fit.
python3 -m hw_cost_model.fit_from_yosys \
    --input yosys_sweep_results_des.csv \
    --side  des \
    --output checkpoints/des_cost_model.joblib \
    --metrics-json checkpoints/des_cost_metrics.json \
    --kappa 0.15

python3 -m hw_cost_model.fit_from_yosys \
    --input yosys_sweep_results_ser.csv \
    --side  ser \
    --output checkpoints/ser_cost_model.joblib \
    --metrics-json checkpoints/ser_cost_metrics.json \
    --kappa 0.15

# 4. Point the Pareto search at the trained model.
cd ../ml_model
python3 optimize_hw_config.py --side des \
    --hw-cost-model ../hw_cost_model/checkpoints/des_cost_model.joblib \
    --num-objectives 3 \
    ...
```

Success criteria (target for the fitter output):

- Top-level MAPE < 5 % on the LHS held-out set (both labels).
- Top-level MAPE < 10 % on the two-axis stress set.
- Spearman ρ > 0.97 on both sets.
- Every bucket coefficient ≥ 0 (guaranteed by NNLS) → per-axis
  monotonicity of the total prediction.

## The two-mode contract

Every downstream caller uses the same entry points:

```python
from hw_cost_model import (
    hardware_cost, hardware_cost_2vec, use_trained_model, DEFAULT_KAPPA,
)

# Mode 1 (default): structural RTL-derived predictor.
c_scalar = hardware_cost([cfg], side="ser")                # 1-D array
c_vec    = hardware_cost_2vec([cfg], side="ser")            # (N, 2) array

# Mode 2: load a trained Yosys-fit model once, then use as normal.
use_trained_model("ser", "checkpoints/ser_cost_model.joblib")
c_scalar = hardware_cost([cfg], side="ser", kappa=DEFAULT_KAPPA)
c_vec    = hardware_cost_2vec([cfg], side="ser")
```

The [`ml_model/util.py`](../ml_model/util.py) file remains as a
compatibility shim so scripts that do `from util import hardware_cost`
keep working.

## Sanity check

Per-axis monotonicity of the structural predictor (must always hold):

```bash
cd generators/protoacc/software/lynx
python3 -c "
from hw_cost_model import hardware_cost, DEFAULT_CONFIG_BY_SIDE, PARAM_VALUES_BY_SIDE
for side in ('des','ser'):
    d = DEFAULT_CONFIG_BY_SIDE[side]
    for k, vs in PARAM_VALUES_BY_SIDE[side].items():
        prev = None
        for v in vs:
            c = float(hardware_cost([dict(d, **{k:v})], side)[0])
            assert prev is None or c >= prev - 1e-9, (side, k, v, c, prev)
            prev = c
print('OK')"
```

That invariant should hold after every future edit here.
