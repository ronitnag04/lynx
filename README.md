# LYNX: Learned AnalYtical Neural Models for AcXelerators

## Directory layout

| Path                    | Purpose |
|-------------------------|---------|
| `HyperProtoBench/`      | Google's HyperProtoBench source (`.proto` schemas + `.inc` runtime data). Source for benches 0–5. |
| `fleetbench/`           | Google's fleetbench submodule. The 20 `fleetbench/proto/Message<N>.proto` schemas (with per-field runtime values in `access_message<N>.cc`) are the source for synthetic benches, combined 5-at-a-time by `verilator-bench/gen_synth_proto.py`. |
| `fleetbench_runtime.py` | Parses fleetbench `access_message<N>.cc` files into the `{msg: {field: [lengths...]}}` shape `proto_to_accel.py` expects. Used by the synthetic-bench generator. |
| `analytical_model/`     | Protobuf schema analyzer → feature vectors. See `analytical_model/README.md`. `protobuf_analyzer.py` handles HPB (benches 0–5); `analyze_synth.py` extends the same analysis JSON with fleetbench-derived synthetic benches (bench6+). Both feed `extract_features.py`. |
| `build_training_dataset.py` | Joins Verilator sweep results (one CSV from Chipyard `generators/protoacc/software/verilator-bench/run_sweep.sh`) with the analytical-feature vector per bench, writing one enriched CSV via `--output-csv` (default `ml_model/data/training_data.csv`) for `ml_model/train.py`. |
| `ml_model/`             | PyTorch training code. Reads `training_data.csv` produced by `build_training_dataset.py`. |
| `sample_protoacc_model/`| Older end-to-end sample built against placeholder throughput data. |

## Training data pipeline

```
  HyperProtoBench/bench[0-5]/     fleetbench/proto/Message[0-19].proto
     .proto + .inc                   + access_message<N>.cc
           │                                    │
           │                        gen_synth_proto.py
           │                   (combines 5 random messages
           │                    per synthetic bench, produces
           │                    gen/synth/bench<N>/*)
           ▼                                    │
      proto_to_accel.py  ◄─── --runtime-lengths ┘
      (→ gen/bench<N>_{descriptors.h,data.c})
                │                                       ProtoAcc RTL
                │                                            │
                ▼                                            ▼
        analytical_model/                       verilator-bench/run_sweep.sh
        ├── protobuf_analyzer.py   (HPB 0-5)   │ (drives each HW-sweep Scala
        ├── analyze_synth.py       (synth 6+)  │  config through the Verilator
        └── extract_features.py                │  simulator, one row per
               │                               │  config×bench. Auto-picks up
               ▼                               │  synth benches via build/)
       extracted_features.json           hw_sweep_results.csv
                       \                       /
                        \                     /
                         ▼                   ▼
                    build_training_dataset.py
                              │
                              ▼
                     training_data.csv  ──► ml_model/train.py
```

## Quick start: produce a training dataset

Assumes a Chipyard checkout with `generators/protoacc/software/verilator-bench`
(this repo is normally the **`software/lynx`** submodule next to that directory).
Set `CHIPYARD_ROOT` to your Chipyard tree if scripts cannot infer it. The
command block below works end-to-end from the verilator-bench directory; it
covers both the stock HPB benches (0–5) and any number of synthetic
fleetbench-derived benches (6+).

`SYNTH_COUNT` controls how many synthetic benches are added; setting it to 0
reproduces the historical HPB-only flow. Keep `SYNTH_COUNT` / `SYNTH_SEED`
identical across the `make synth` / `make gen` / `make bench` invocations so
the intermediate files line up.

```bash
# 0. From Chipyard root: move into the verilator-bench tree.
cd generators/protoacc/software/verilator-bench

# 1. Build descriptor/data sources and ELFs for all benches (HPB + synth).
#    With SYNTH_COUNT=0 this is the historical 6-bench HPB-only pipeline.
#    Each stage is internally parallelized; gen_batch is the path to use
#    when you're regenerating everything (3m for 4000 benches), while
#    `make gen -j$(nproc)` is the right choice for incremental rebuilds.
make synth     SYNTH_COUNT=20 SYNTH_SEED=42  # generates gen/synth/bench6..bench25/
make gen_batch SYNTH_COUNT=20 SYNTH_SEED=42  # emits gen/bench<N>_{descriptors.h,data.c}
make -j$(nproc) SYNTH_COUNT=20 SYNTH_SEED=42 bench   # builds build/bench<N>_{ser,des}.riscv

# 2. Refresh the analytical features JSON. protobuf_analyzer handles HPB
#    benches 0-5; analyze_synth appends bench6+ (parallel, ~3m at 4000
#    benches on a 16-core box). Both go to the same JSON and feed the
#    shared extract_features.py so every bench ends up in
#    extracted_features.json with the same column layout.
cd ../lynx/analytical_model
python3 protobuf_analyzer.py --verilator-bench-profile \
    --output protobuf_analysis_verilator_bench.json
python3 analyze_synth.py \
    --synth-root ../../verilator-bench/gen/synth \
    --in-json protobuf_analysis_verilator_bench.json \
    --out-json protobuf_analysis_verilator_bench.json
python3 extract_features.py \
    --input protobuf_analysis_verilator_bench.json \
    --output extracted_features.json

# 3. Generate sweep configs + run the sweep. run_sweep.sh auto-discovers the
#    bench list from build/bench*_ser.riscv, so synthetic benches are
#    included by default (no separate --benches flag needed).
cd "${CHIPYARD_ROOT:?}"
python3 generators/protoacc/software/verilator-bench/gen_protoacc_sweep_configs.py \
    --emit both -t random -n 32 -s 42
bash generators/protoacc/software/verilator-bench/run_sweep.sh \
    --output /tmp/sweep.csv

# 4. Join sweep results with analytical features → enriched training CSV.
#    --output-csv defaults to ml_model/data/training_data.csv; set it
#    explicitly to keep multiple training runs side by side.
cd generators/protoacc/software/lynx
python3 build_training_dataset.py \
    --sweep-csv /tmp/sweep.csv \
    --output-csv ./ml_model/data/training_data.csv

# 5. Train (example): paths depend on how ml_model/load_data expects tensors —
#    see ml_model/README.md or train.py --help.
cd ml_model
python3 train.py --help
```

### Adding / changing synthetic benches later

Re-running `make synth` with the same `SYNTH_SEED` is deterministic: bench<N>
always selects the same 5-message tuple. To expand the catalog without
renumbering existing benches, bump `SYNTH_COUNT` (keeps `SYNTH_START=6` and
the same seed → existing bench6..benchK stay identical, new ids appended).
To produce a completely different catalog for OOD experiments, change
`SYNTH_SEED`. Either way, re-run steps 1–4 end to end; `analyze_synth.py`
updates the JSON in place and `run_sweep.sh` resumes where it left off
based on the `(config_name, bench, op)` keys already in the CSV.

**Uniqueness.** Subsets are drawn by deterministically shuffling the full
space of C(|pool|, `SYNTH_N_MESSAGES`) combinations (15,504 for the default
20-message pool × 5-msg benches) and taking the first `SYNTH_COUNT` entries.
No duplicate subsets across the catalog; if `SYNTH_COUNT` exceeds that space
the generator errors out rather than silently duplicating benches. Grow the
pool (`--available-ids`), shrink `--n-messages`, or lower `--count` to stay
under the cap.

**Parallelism at scale.** Every stage of synth-bench generation is
parallelized so 4k-bench catalogs are tractable on a single build box.
Measured wall times on a 16-core host:

| Stage                         | Script / target              | 200 benches | 4000 benches |
|-------------------------------|------------------------------|-------------|--------------|
| 1. Combine protos + runtime   | `make synth`                 | 0.27 s      | 0.87 s       |
| 2. `proto_to_accel.py` batch  | `make gen_batch`             | 10 s        | **~3 min**   |
| 3. Compile RISC-V ELFs        | `make -j$(nproc) bench`      | —           | ~5–10 min    |
| 4. Analytical features        | `analyze_synth.py` (default) | 9 s         | **~3 min**   |

Stages 2 and 4 dominate. Both accept `--workers N` (default `nproc/2`; pass
`--workers 1` for deterministic stdout order during debugging). Stage 2's
``make gen`` (per-bench pattern rules, used when only a few benches changed)
is slower at large counts than ``make gen_batch`` (single Python process +
worker pool) because each bench otherwise pays ~40 ms of interpreter startup.

## Training CSV schema

Produced by `build_training_dataset.py` at `--output-csv` (default
`ml_model/data/training_data.csv`). `ml_model/train.py` reads this CSV
directly — there is no intermediate `.npy` step.

| Column group               | Examples                                             | Role      |
|----------------------------|------------------------------------------------------|-----------|
| Identifiers                | `config_name`, `side`, `bench`, `op`, `iters`, `cycles`, `bytes`, `wall_s` | carry-along; dropped from the feature matrix |
| Hardware knobs (19 cols)   | `des_top_descriptor_reqs`, `ser_field_handlers`, …   | input features |
| Analytical features (59 cols) | `feat_total_size_bytes`, `feat_avg_depth`, `feat_size_bytes_distribution_0..9`, `feat_depth_counter_list_0..14`, … | input features |
| Labels                     | `throughput_bytes_per_sec` (raw), `throughput_gbits_per_sec` (8/1e9 scaled) | target |

Feature axis: 19 hardware knobs + 59 analytical features = 78 input columns.
Throughput is reported in bytes/sec at the Verilator config's nominal clock
(assumed 1 GHz); `throughput_gbits_per_sec` is the version `train.py`
regresses on.
