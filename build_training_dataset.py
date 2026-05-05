#!/usr/bin/env python3
"""
build_training_dataset.py — join Verilator sweep results with protobuf
analytical features into one enriched CSV for Lynx model training.

Inputs:
  --sweep-csv CSV      From Chipyard's
                       generators/protoacc/software/verilator-bench/run_sweep.sh
                       — one row per (hw_config x bench x op) with cycle/byte/
                       throughput and the hardware parameter knobs. Rows are
                       tagged with op=ser|des; this script keeps both.
  --features JSON      Default: <lynx-repo>/analytical_model/extracted_features.json
                       (produced by analytical_model/extract_features.py).
                       Keyed on bench name (``bench0``..``bench5``).

Output:
  <output-csv>
      One row per sweep row, with flattened analytical features appended.
      Includes both sides (`op=ser|des`) and a converted
      `throughput_gbits_per_sec` label.

Typical usage:
  python3 build_training_dataset.py \\
      --sweep-csv /tmp/combined_sweep.csv \\
      --output-csv ./ml_model/data/training_data.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

_LYNX_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = _LYNX_REPO_ROOT / "analytical_model" / "extracted_features.json"
DEFAULT_OUTPUT_CSV = _LYNX_REPO_ROOT / "ml_model" / "data" / "training_data.csv"

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

INPUT_LABEL_COLUMN = "throughput_bytes_per_sec"
OUTPUT_LABEL_COLUMN = "throughput_gbits_per_sec"


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


def build_training_dataframe(
    sweep_df: pd.DataFrame,
    feat_rows: Dict[str, Dict[str, float]],
    feat_columns: List[str],
) -> pd.DataFrame:
    """Join flattened analytical features onto each sweep row."""
    feat_df = pd.DataFrame({
        b: feat_rows.get(b, {c: float("nan") for c in feat_columns})
        for b in sweep_df["bench"].unique()
    }).T.reset_index().rename(columns={"index": "bench"})
    merged = sweep_df.merge(feat_df, on="bench", how="left")
    merged[OUTPUT_LABEL_COLUMN] = merged[INPUT_LABEL_COLUMN] * (8.0 / 1e9)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-csv", type=Path, required=True,
                    help="Sweep results CSV from run_sweep.sh (may contain "
                         "both ser and des rows; split by op column).")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                    help=f"Analytical features JSON (default: {DEFAULT_FEATURES})")
    ap.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
                    help=f"Path to enriched output CSV (default: {DEFAULT_OUTPUT_CSV})")
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

    if INPUT_LABEL_COLUMN not in sweep_df.columns:
        sys.exit(f"Sweep CSV missing label column '{INPUT_LABEL_COLUMN}'")

    missing = set(sweep_df["bench"].unique()) - set(features.keys())
    if missing:
        print(f"WARNING: sweep CSV references benches not in features: {missing}",
              file=sys.stderr)

    feat_rows, feat_columns = flatten_features(features)
    merged = build_training_dataframe(sweep_df, feat_rows, feat_columns)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"Wrote enriched training CSV: {args.output_csv}", file=sys.stderr)
    print(f"Rows: {len(merged)}", file=sys.stderr)
    print(f"Columns: {len(merged.columns)}", file=sys.stderr)


if __name__ == "__main__":
    main()
