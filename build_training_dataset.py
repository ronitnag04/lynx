#!/usr/bin/env python3
"""
build_training_dataset.py — join Verilator sweep results with protobuf
analytical features into per-side training datasets for the Lynx ML model.

Inputs:
  --sweep-csv CSV      From
                       /home/ec2-user/hyperscale-grpc-chipyard/generators/
                       protoacc/software/verilator-bench/run_sweep.sh — one row
                       per (hw_config x bench x op) with cycle/byte/throughput
                       and the hardware parameter knobs. Rows are tagged with
                       op=ser|des; this script splits by op.
  --features JSON      Default
                       /home/ec2-user/lynx/analytical_model/extracted_features.json
                       (produced by analytical_model/extract_features.py).
                       Keyed on bench name (``bench0``..``bench5``).

Output (matches sample_protoacc_model/generate_dataset.py layout, which is
what ml_model/train.py expects):
  <output-base-dir>/serializer_dataset/
      train_features.npy  train_labels.npy
      test_features.npy   test_labels.npy
  <output-base-dir>/deserializer_dataset/
      train_features.npy  train_labels.npy
      test_features.npy   test_labels.npy

Feature layout (per row, matching the sample generator):
  [ hardware knobs for this side ]  ++  [ flattened analytical features ]
  * serializer rows use only ``ser_*`` knobs;
  * deserializer rows use only ``des_*`` knobs.
  Labels are throughput_bytes_per_sec, shape (N, 1).

Typical usage:
  python3 build_training_dataset.py \\
      --sweep-csv /tmp/combined_sweep.csv \\
      --output-base-dir /home/ec2-user/lynx/ml_model/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np


DEFAULT_FEATURES = Path("/home/ec2-user/lynx/analytical_model/extracted_features.json")
DEFAULT_OUTPUT_BASE = Path("/home/ec2-user/lynx/ml_model/data")

# Feature keys we know about (keep in sync with analytical_model/extract_features.py).
SCALAR_FEATURES = [
    "total_size_bytes", "num_messages",
    "min_size_bytes", "max_size_bytes", "avg_size_bytes",
    "min_total_fields", "max_total_fields", "avg_total_fields",
    "min_nested_message_count", "max_nested_message_count", "avg_nested_message_count",
    "min_depth", "max_depth", "avg_depth",
]
LIST_FEATURES = [
    "size_bytes_distribution",           # 10 bins
    "total_fields_distribution",         # 10 bins
    "nested_message_count_distribution", # 10 bins
    "depth_counter_list",                # 15 bins
]

# Sweep CSV metadata/identifier columns (everything here is neither a hardware
# knob nor the label).
ID_COLUMNS = ["config_name", "side", "bench", "op", "iters", "cycles", "bytes", "wall_s"]
LABEL_COLUMN = "throughput_bytes_per_sec"

# Map op value in sweep CSV -> (dataset dir name, knob prefix).
OP_TO_DATASET: Dict[str, Tuple[str, str]] = {
    "ser": ("serializer_dataset", "ser_"),
    "des": ("deserializer_dataset", "des_"),
}


def flatten_features(features: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Flatten per-bench feature dict into (bench_name -> {feat_name: value}).

    Returns the stable list of feature column names. Benches whose feature
    JSON lacks a known key get NaN for that column; downstream drops NaN rows.
    """
    list_lengths: Dict[str, int] = {k: 0 for k in LIST_FEATURES}
    for _, fv in features.items():
        for k in LIST_FEATURES:
            v = fv.get(k)
            if isinstance(v, list):
                list_lengths[k] = max(list_lengths[k], len(v))

    column_order: List[str] = []
    for k in SCALAR_FEATURES:
        column_order.append(f"feat_{k}")
    for k in LIST_FEATURES:
        for i in range(list_lengths[k]):
            column_order.append(f"feat_{k}_{i}")

    out: Dict[str, Dict[str, float]] = {}
    for bench_name, fv in features.items():
        row: Dict[str, float] = {}
        for k in SCALAR_FEATURES:
            row[f"feat_{k}"] = float(fv.get(k, float("nan")))
        for k in LIST_FEATURES:
            arr = fv.get(k) or []
            for i in range(list_lengths[k]):
                row[f"feat_{k}_{i}"] = float(arr[i]) if i < len(arr) else float("nan")
        out[bench_name] = row
    return out, column_order


def build_side_dataset(sweep_df: pd.DataFrame,
                        feat_rows: Dict[str, Dict[str, float]],
                        feat_columns: List[str],
                        knob_prefix: str,
                        ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, x_columns) for one side of the sweep.

    Selects only hardware knob columns starting with ``knob_prefix`` (so a
    ser-side dataset excludes ``des_*`` knobs and vice-versa), joins in the
    per-bench analytical features, and returns float32 arrays.
    """
    non_knob = set(ID_COLUMNS) | {LABEL_COLUMN}
    knob_columns = [
        c for c in sweep_df.columns
        if c.startswith(knob_prefix) and c not in non_knob
    ]

    feat_df = pd.DataFrame({
        b: feat_rows.get(b, {c: float("nan") for c in feat_columns})
        for b in sweep_df["bench"].unique()
    }).T.reset_index().rename(columns={"index": "bench"})

    merged = sweep_df.merge(feat_df, on="bench", how="left")
    merged = merged.dropna(subset=[LABEL_COLUMN])

    x_columns = knob_columns + feat_columns
    X = merged[x_columns].to_numpy(dtype=np.float32)
    y = merged[LABEL_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X, y, x_columns


def split_dataset(X: np.ndarray, y: np.ndarray, train_split: float,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle jointly, then split into (train_X, train_y, test_X, test_y)."""
    n = len(X)
    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]
    split_idx = int(n * train_split)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def save_dataset(output_dir: Path,
                 train_X: np.ndarray, train_y: np.ndarray,
                 test_X: np.ndarray, test_y: np.ndarray,
                 x_columns: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_features.npy", train_X)
    np.save(output_dir / "train_labels.npy", train_y)
    np.save(output_dir / "test_features.npy", test_X)
    np.save(output_dir / "test_labels.npy", test_y)
    # Sidecar so column order can be recovered for predict.py / debugging.
    with (output_dir / "schema.json").open("w") as f:
        json.dump({
            "x_columns": x_columns,
            "label_column": LABEL_COLUMN,
            "train_shape": list(train_X.shape),
            "test_shape": list(test_X.shape),
        }, f, indent=2)
    print(f"Wrote {output_dir}:", file=sys.stderr)
    print(f"  train_features.npy shape={train_X.shape}", file=sys.stderr)
    print(f"  train_labels.npy   shape={train_y.shape}", file=sys.stderr)
    print(f"  test_features.npy  shape={test_X.shape}", file=sys.stderr)
    print(f"  test_labels.npy    shape={test_y.shape}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-csv", type=Path, required=True,
                    help="Sweep results CSV from run_sweep.sh (may contain "
                         "both ser and des rows; split by op column).")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                    help=f"Analytical features JSON (default: {DEFAULT_FEATURES})")
    ap.add_argument("--output-base-dir", type=Path, default=DEFAULT_OUTPUT_BASE,
                    help=f"Base dir; serializer_dataset/ and deserializer_dataset/ "
                         f"are written underneath (default: {DEFAULT_OUTPUT_BASE})")
    ap.add_argument("--train-split", type=float, default=0.75,
                    help="Fraction of samples assigned to the train set.")
    ap.add_argument("--seed", type=int, default=7,
                    help="RNG seed for the train/test shuffle.")
    args = ap.parse_args()

    if not args.sweep_csv.exists():
        sys.exit(f"Sweep CSV not found: {args.sweep_csv}")
    if not args.features.exists():
        sys.exit(f"Features JSON not found: {args.features}")

    sweep_df = pd.read_csv(args.sweep_csv)
    with args.features.open() as f:
        features = json.load(f)

    print(f"Sweep rows: {len(sweep_df)}", file=sys.stderr)
    print(f"Benches in sweep: {sorted(sweep_df['bench'].unique())}", file=sys.stderr)
    print(f"Benches in features: {sorted(features.keys())}", file=sys.stderr)

    if "op" not in sweep_df.columns:
        sys.exit("Sweep CSV missing 'op' column — needed to split ser vs des rows.")
    if LABEL_COLUMN not in sweep_df.columns:
        sys.exit(f"Sweep CSV missing label column '{LABEL_COLUMN}'")

    missing = set(sweep_df["bench"].unique()) - set(features.keys())
    if missing:
        print(f"WARNING: sweep CSV references benches not in features: {missing}",
              file=sys.stderr)

    feat_rows, feat_columns = flatten_features(features)
    rng = np.random.default_rng(args.seed)

    for op_value, (dataset_dir_name, knob_prefix) in OP_TO_DATASET.items():
        side_df = sweep_df[sweep_df["op"] == op_value]
        if side_df.empty:
            print(f"No rows with op={op_value!r} in sweep CSV; skipping "
                  f"{dataset_dir_name}.", file=sys.stderr)
            continue

        X, y, x_columns = build_side_dataset(side_df, feat_rows, feat_columns, knob_prefix)
        print(f"\n[{op_value}] {X.shape[0]} rows, {X.shape[1]} features "
              f"({len(x_columns) - len(feat_columns)} knobs + "
              f"{len(feat_columns)} analytical)", file=sys.stderr)

        train_X, train_y, test_X, test_y = split_dataset(X, y, args.train_split, rng)
        save_dataset(args.output_base_dir / dataset_dir_name,
                      train_X, train_y, test_X, test_y, x_columns)


if __name__ == "__main__":
    main()
