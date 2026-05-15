# LYNX: Learned AnalYtical Neural Models for AcXelerators

## Directory layout

| Path                    | Purpose |
|-------------------------|---------|
| `HyperProtoBench/`      | Google's HyperProtoBench source (`.proto` schemas + `.inc` runtime data). Source for benches 0–5. |
| `analytical_model/`     | Protobuf schema analyzer → feature vectors. See details below. |
| `build_training_dataset.py` | Joins Verilator sweep results (one CSV from Chipyard `generators/protoacc/software/verilator-bench/run_sweep.sh`) with the analytical-feature vector per bench, writing one enriched CSV via `--output-csv` (default `ml_model/data/training_data.csv`) for `ml_model/train.py`. |
| `ml_model/`             | PyTorch training code. Reads `training_data.csv` produced by `build_training_dataset.py`. See details below. |

## analytical_model/

Extracts features from protobuf schemas to characterize workload complexity.

| File                    | Purpose |
|-------------------------|---------|
| `protobuf_analyzer.py`  | Analyzes HyperProtoBench schemas, produces `protobuf_analysis.json` with ProtoBuf message metadata |
| `extract_features.py`   | Converts raw analysis JSON into feature vectors, outputs `extracted_features.json` |
| `plot_features.py`      | Visualizes feature distributions across benchmarks |

## ml_model/

PyTorch-based throughput prediction models.

| File                    | Purpose |
|-------------------------|---------|
| `train.py`              | Main training script. Trains neural network on hardware config + analytical features → throughput |
| `model.py`              | PyTorch model architecture definition |
| `exhaustive_search.py`  | Brute-force search over hardware config space for comparison |
| `util.py`               | Shared utilities for data loading and preprocessing |

## Training data pipeline

```
  HyperProtoBench/bench[0-5]/
     .proto + .inc
           │
           ▼
      proto_to_accel.py
      (→ gen/bench<N>_{descriptors.h,data.c})
                │                                       ProtoAcc RTL
                │                                            │
                ▼                                            ▼
        analytical_model/                       verilator-bench/run_sweep.sh
        protobuf_analyzer.py                   (drives each HW config through
               │                                the Verilator simulator,
               ▼                                one row per config×bench)
        protobuf_analysis.json                         │
               │                                       ▼
        extract_features.py                   hw_sweep_results.csv
               │                                       │
               ▼                                       │
       extracted_features.json                        │
                       \                              /
                        \                            /
                         ▼                          ▼
                    build_training_dataset.py
                              │
                              ▼
                     training_data.csv  ──► ml_model/train.py
```

## Quick start: produce a training dataset

Assumes a Chipyard checkout with `generators/protoacc/software/verilator-bench`
(this repo is normally the **`software/lynx`** submodule next to that directory).
Set `CHIPYARD_ROOT` to your Chipyard tree if scripts cannot infer it. Use the LYNX
Chipyard repo: [https://github.com/ronitnag04/lynx-chipyard.git](https://github.com/ronitnag04/lynx-chipyard.git).

```bash
# 0. From Chipyard root: move into the verilator-bench tree.
cd generators/protoacc/software/verilator-bench

# 1. Build descriptor/data sources and benchmark binaries.
make gen -j$(nproc)    # emits gen/bench<N>_{descriptors.h,data.c}
make -j$(nproc) bench  # builds build/bench<N>_{ser,des}.riscv

# 2. Extract analytical features from HyperProtoBench schemas.
cd ../lynx/analytical_model
python3 protobuf_analyzer.py --verilator-bench-profile \
    --output protobuf_analysis.json
python3 extract_features.py \
    --input protobuf_analysis.json \
    --output extracted_features.json

# 3. Generate sweep configs and run the hardware sweep.
cd "${CHIPYARD_ROOT:?}"
python3 generators/protoacc/software/verilator-bench/gen_protoacc_sweep_configs.py \
    --emit both -t random -n 32 -s 42
bash generators/protoacc/software/verilator-bench/run_sweep.sh \
    --output /tmp/sweep.csv

# 4. Join sweep results with analytical features → training CSV.
cd generators/protoacc/software/lynx
python3 build_training_dataset.py \
    --sweep-csv /tmp/sweep.csv \
    --output-csv ./ml_model/data/training_data.csv

# 5. Train the model.
cd ml_model
python3 train.py --help
```

## Training CSV schema

Produced by `build_training_dataset.py` at `--output-csv` (default
`ml_model/data/training_data.csv`). `ml_model/train.py` reads this CSV
directly — there is no intermediate `.npy` step.

| Column group               | Examples                                             | Role      |
|----------------------------|------------------------------------------------------|-----------|
| Identifiers                | `config_name`, `side`, `bench`, `op`, `iters`, `cycles`, `bytes`, `wall_s` | simulation metadata; dropped from the feature matrix |
| Hardware knobs (19 cols)   | `des_top_descriptor_reqs`, `ser_field_handlers`, …   | input features |
| Analytical features (59 cols) | `feat_total_size_bytes`, `feat_avg_depth`, `feat_size_bytes_distribution_0..9`, `feat_depth_counter_list_0..5`, … | input features |
| Labels                     | `throughput_bytes_per_sec` (raw), `throughput_gbits_per_sec` (8/1e9 scaled) | target |

Feature axis: 19 hardware knobs + 50 analytical features = 69 input columns.
Throughput is reported in bytes/sec at the Verilator config's nominal clock
(assumed 1 GHz); `throughput_gbits_per_sec` is the version `train.py`
regresses on.
