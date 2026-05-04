# LYNX: Learned AnalYtical Neural Models for AcXelerators

## Directory layout

| Path                    | Purpose |
|-------------------------|---------|
| `HyperProtoBench/`      | Google's HyperProtoBench source (`.proto` schemas + `.inc` runtime data). |
| `analytical_model/`     | Protobuf schema analyzer → feature vectors. See `analytical_model/README.md`. Supports `--verilator-bench-profile` to match what the Verilator-simulated benches actually exercise. |
| `build_training_dataset.py` | Joins Verilator sweep results (one CSV from `hyperscale-grpc-chipyard/.../verilator-bench/run_sweep.py`) with the analytical-feature vector per bench, producing a training CSV plus optional `X.npy` / `y.npy` tensors for `ml_model/train.py`. |
| `ml_model/`             | PyTorch training code. Reads `X.npy` / `y.npy`. |
| `sample_protoacc_model/`| Older end-to-end sample built against placeholder throughput data. |

## Training data pipeline

```
                    .proto + .inc                     ProtoAcc RTL
                         │                                │
                         ▼                                ▼
  analytical_model/                    verilator-bench/run_sweep.py
  ├── protobuf_analyzer.py  ◄──────────┤ (drives each HW-sweep Scala config
  └── extract_features.py              │  through the Verilator simulator,
         │                             │  writing one row per config×bench)
         ▼                             ▼
  extracted_features.json     hw_sweep_results.csv
                  \                   /
                   \                 /
                    ▼               ▼
               build_training_dataset.py
                         │
                         ▼
                training_dataset.csv  ──► ml_model/train.py
                     + X.npy, y.npy
```

## Quick start: produce a training dataset for one benchmark

Assumes Chipyard is set up at `/home/ec2-user/hyperscale-grpc-chipyard` and
the Verilator sim + bench ELFs have been built
(see `hyperscale-grpc-chipyard/generators/protoacc/software/verilator-bench/STATUS.md`).

```bash
# 1. Refresh analytical features (bench-scoped, so they match the Verilator benches):
cd /home/ec2-user/lynx/analytical_model
python3 protobuf_analyzer.py --verilator-bench-profile \
    --output protobuf_analysis_verilator_bench.json
python3 extract_features.py \
    --input protobuf_analysis_verilator_bench.json \
    --output extracted_features.json

# 2. Generate + simulate a sweep across the ProtoAcc hardware parameter space
#    for ONE benchmark (here: bench1 serializer). The run appends one row per
#    sample config to the CSV; re-running resumes at the first missing row.
python3 /home/ec2-user/hyperscale-grpc-chipyard/generators/protoacc/software/gen_protoacc_sweep_configs.py \
    --emit both -t random -n 32 -s 42
python3 /home/ec2-user/hyperscale-grpc-chipyard/generators/protoacc/software/verilator-bench/run_sweep.py \
    --bench bench1 --op ser --side ser \
    --output /tmp/bench1_ser_sweep.csv

# 3. Join sweep results with analytical features:
python3 /home/ec2-user/lynx/build_training_dataset.py \
    --sweep-csv /tmp/bench1_ser_sweep.csv \
    --output    /tmp/bench1_ser_training.csv \
    --npy-dir   /tmp/bench1_ser_training_npy

# 4. Train (example):
cd /home/ec2-user/lynx/ml_model
python3 train.py \
    --features /tmp/bench1_ser_training_npy/X.npy \
    --labels   /tmp/bench1_ser_training_npy/y.npy
```

## Training CSV schema

Produced by `build_training_dataset.py`:

| Column group               | Examples                                             | Role      |
|----------------------------|------------------------------------------------------|-----------|
| Identifiers                | `config_name`, `side`, `bench`, `op`, `iters`, `cycles`, `bytes`, `wall_s` | carry-along; dropped from `X.npy` |
| Hardware knobs (19 cols)   | `des_top_descriptor_reqs`, `ser_field_handlers`, …   | input features (X) |
| Analytical features (59 cols) | `feat_total_size_bytes`, `feat_avg_depth`, `feat_size_bytes_distribution_0..9`, `feat_depth_counter_list_0..14`, … | input features (X) |
| Label                      | `throughput_bytes_per_sec`                           | target (y) |

`X.npy` columns are hardware knobs (19) then analytical features (59) = 78 total.
`y.npy` is `(N, 1)` with throughput in bytes/sec at the Verilator config's nominal clock (assumed 1 GHz).
`schema.json` in the `--npy-dir` records the exact column order so `predict.py` can keep the same X layout.
