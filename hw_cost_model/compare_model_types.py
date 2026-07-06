#!/usr/bin/env python3
"""
Compare different regression methods for hw_cost_model.

Tests 4 approaches:
1. Random Forest (per submodule)
2. Gradient Boosting (per submodule)
3. Ridge Regression (per submodule)
4. Two-Stage Random Forest (submodules sum to total, no global bias)

Evaluates on:
- Holdout set from training data
- Validation configs (newly collected data)
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Add parent to path for imports
_LYNX_DIR = Path(__file__).resolve().parent.parent
if str(_LYNX_DIR) not in sys.path:
    sys.path.insert(0, str(_LYNX_DIR))

from hw_cost_model.defaults import DES_PARAM_COLUMNS, SER_PARAM_COLUMNS
from hw_cost_model.structural_features import (
    DES_SUBMODULES,
    SER_SUBMODULES,
    features_for_side,
)


# Model configurations
MODEL_CONFIGS = {
    "RandomForest": {
        "class": RandomForestRegressor,
        "params": {"n_estimators": 100, "max_depth": 20, "random_state": 42, "n_jobs": -1},
    },
    "GradientBoosting": {
        "class": GradientBoostingRegressor,
        "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "learning_rate": 0.1},
    },
    "Ridge": {
        "class": Ridge,
        "params": {"alpha": 1.0, "random_state": 42},
    },
}

LABELS = ("logic_cells", "ram_bits")


def load_data(csv_path: Path, side: str) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    """Load and split data into train/holdout."""
    df = pd.read_csv(csv_path)

    param_cols = DES_PARAM_COLUMNS if side == "des" else SER_PARAM_COLUMNS

    # Split based on sample_group if available
    if "sample_group" in df.columns and (df["sample_group"] == "holdout").any():
        train_df = df[df["sample_group"] != "holdout"].copy()
        holdout_df = df[df["sample_group"] == "holdout"].copy()
    else:
        # Random 90/10 split
        train_df = df.sample(frac=0.9, random_state=42)
        holdout_df = df.drop(train_df.index)

    # Extract configs and features
    train_cfgs = train_df[param_cols].to_dict(orient="records")
    holdout_cfgs = holdout_df[param_cols].to_dict(orient="records")

    train_feats = [features_for_side(c, side) for c in train_cfgs]
    holdout_feats = [features_for_side(c, side) for c in holdout_cfgs]

    return train_df, holdout_df, train_feats, holdout_feats, train_cfgs, holdout_cfgs


def get_bucket_features(side: str, bucket: str) -> List[str]:
    """Get feature names for a bucket (from fit_from_yosys.py logic)."""
    from hw_cost_model.defaults import DEFAULT_CONFIG_BY_SIDE
    from hw_cost_model.structural_features import DES_QUEUES, SER_QUEUES

    all_feats = features_for_side(DEFAULT_CONFIG_BY_SIDE[side], side)

    # Features prefixed with bucket name
    prefixed = sorted(k for k in all_feats if k.startswith(bucket + "__"))

    # Queue-derived features
    queues = DES_QUEUES if side == "des" else SER_QUEUES
    knob_features = []
    for q in queues:
        if q.submodule != bucket:
            continue
        knob_features.extend(sorted(k for k in all_feats if k.startswith(q.knob + "__")))

    combined = sorted(set(prefixed) | set(knob_features))
    return combined if combined else []


def train_model_type(
    model_name: str,
    train_df: pd.DataFrame,
    train_feats: List[dict],
    side: str,
) -> Dict[str, Dict[str, Any]]:
    """Train a model type for all submodules."""
    buckets = DES_SUBMODULES if side == "des" else SER_SUBMODULES

    model_config = MODEL_CONFIGS[model_name]
    per_submod = {}

    for bucket in buckets:
        feat_names = get_bucket_features(side, bucket)

        # Build feature matrix
        X = np.array(
            [[fd.get(k, 0.0) for k in feat_names] for fd in train_feats],
            dtype=np.float64,
        )

        per_submod[bucket] = {}

        for label in LABELS:
            col = f"submod_{bucket}_{label}"
            if col not in train_df.columns:
                continue

            y = train_df[col].to_numpy(dtype=np.float64)

            # Train model
            if X.shape[1] == 0:
                # No features - use mean
                mean_val = float(np.mean(y))
                per_submod[bucket][label] = {
                    "type": "constant",
                    "value": mean_val,
                    "feature_names": [],
                }
            else:
                model = model_config["class"](**model_config["params"])
                model.fit(X, y)
                per_submod[bucket][label] = {
                    "type": "sklearn",
                    "model": model,
                    "feature_names": feat_names,
                }

    return per_submod


def predict_with_model(
    per_submod: Dict[str, Dict[str, Any]],
    feat_dicts: List[dict],
    side: str,
    kappa: float = 0.0001,
    use_global_bias: bool = True,
    global_bias: Dict[str, float] = None,
) -> np.ndarray:
    """Make predictions with a trained model."""
    n_samples = len(feat_dicts)
    buckets = DES_SUBMODULES if side == "des" else SER_SUBMODULES

    # Predict each submodule
    logic_total = np.zeros(n_samples)
    ram_total = np.zeros(n_samples)

    for bucket in buckets:
        if bucket not in per_submod:
            continue

        for label, label_total in [("logic_cells", logic_total), ("ram_bits", ram_total)]:
            if label not in per_submod[bucket]:
                continue

            pred_info = per_submod[bucket][label]

            if pred_info["type"] == "constant":
                label_total += pred_info["value"]
            else:
                feat_names = pred_info["feature_names"]
                X = np.array(
                    [[fd.get(k, 0.0) for k in feat_names] for fd in feat_dicts],
                    dtype=np.float64,
                )
                preds = pred_info["model"].predict(X)
                # Ensure non-negative
                preds = np.maximum(preds, 0)
                label_total += preds

    # Add global bias if specified
    if use_global_bias and global_bias:
        logic_total += global_bias.get("logic_cells", 0)
        ram_total += global_bias.get("ram_bits", 0)

    # Compute scalar cost
    scalar_cost = logic_total + kappa * ram_total

    return scalar_cost


def compute_global_bias(
    per_submod: Dict[str, Dict[str, Any]],
    train_df: pd.DataFrame,
    train_feats: List[dict],
    side: str,
) -> Dict[str, float]:
    """Compute global bias to match total predictions."""
    # Predict submodule totals
    logic_pred = np.zeros(len(train_feats))
    ram_pred = np.zeros(len(train_feats))

    buckets = DES_SUBMODULES if side == "des" else SER_SUBMODULES

    for bucket in buckets:
        if bucket not in per_submod:
            continue

        for label, pred_total in [("logic_cells", logic_pred), ("ram_bits", ram_pred)]:
            if label not in per_submod[bucket]:
                continue

            pred_info = per_submod[bucket][label]

            if pred_info["type"] == "constant":
                pred_total += pred_info["value"]
            else:
                feat_names = pred_info["feature_names"]
                X = np.array(
                    [[fd.get(k, 0.0) for k in feat_names] for fd in train_feats],
                    dtype=np.float64,
                )
                preds = pred_info["model"].predict(X)
                preds = np.maximum(preds, 0)
                pred_total += preds

    # Compute residuals
    global_bias = {}
    for label, pred_total in [("logic_cells", logic_pred), ("ram_bits", ram_pred)]:
        col = f"total_{label}"
        if col in train_df.columns:
            actual = train_df[col].to_numpy()
            residual = float(np.mean(actual - pred_total))
            global_bias[label] = residual
        else:
            global_bias[label] = 0.0

    return global_bias


def evaluate_model(
    model_name: str,
    per_submod: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    cfgs: List[dict],
    feats: List[dict],
    side: str,
    kappa: float,
    use_global_bias: bool,
    global_bias: Dict[str, float] = None,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    # Predict
    preds = predict_with_model(per_submod, feats, side, kappa, use_global_bias, global_bias)

    # Actual
    actual = df["total_logic_cells"].values + kappa * df["total_ram_bits"].values

    # Metrics
    mae = mean_absolute_error(actual, preds)
    mape = mean_absolute_percentage_error(actual, preds) * 100
    errors_pct = ((preds - actual) / actual) * 100

    # Error by cost range
    bins = [0, 175000, 225000, 275000, 1000000]
    labels = ["<175k", "175-225k", "225-275k", ">275k"]

    range_errors = {}
    for i in range(len(bins) - 1):
        mask = (actual >= bins[i]) & (actual < bins[i+1])
        if mask.sum() > 0:
            range_errors[labels[i]] = float(np.mean(errors_pct[mask]))

    return {
        "mae": mae,
        "mape": mape,
        "mean_error_pct": float(np.mean(errors_pct)),
        "abs_mean_error_pct": float(np.mean(np.abs(errors_pct))),
        "min_error_pct": float(np.min(errors_pct)),
        "max_error_pct": float(np.max(errors_pct)),
        "range_errors": range_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare hw_cost_model regression methods")
    parser.add_argument("--side", choices=["ser", "des"], default="ser")
    parser.add_argument("--kappa", type=float, default=0.0001)
    parser.add_argument("--train-data", type=Path, default=Path("hw_cost_model/yosys_sweep_results_ser.csv"))
    parser.add_argument("--val-data", type=Path, default=Path("ml_model/results/hpb_verilator/serializer_ml_models_v1/pareto_comparison_ser/yosys_validation_ser.csv"))
    args = parser.parse_args()

    print(f"Loading data for {args.side}...")
    train_df, holdout_df, train_feats, holdout_feats, train_cfgs, holdout_cfgs = load_data(
        args.train_data, args.side
    )

    print(f"  Training samples: {len(train_df)}")
    print(f"  Holdout samples: {len(holdout_df)}")

    # Load validation data
    val_df = pd.read_csv(args.val_data)
    param_cols = DES_PARAM_COLUMNS if args.side == "des" else SER_PARAM_COLUMNS
    val_cfgs = val_df[param_cols].to_dict(orient="records")
    val_feats = [features_for_side(c, args.side) for c in val_cfgs]
    print(f"  Validation samples: {len(val_df)}")

    results = {}

    # Train and evaluate each model type
    for model_name in ["RandomForest", "GradientBoosting", "Ridge"]:
        print(f"\n{'='*70}")
        print(f"Training {model_name}...")
        print(f"{'='*70}")

        per_submod = train_model_type(model_name, train_df, train_feats, args.side)

        # Compute global bias (with bias)
        global_bias = compute_global_bias(per_submod, train_df, train_feats, args.side)

        # Evaluate with global bias
        print(f"\n{model_name} WITH global_bias:")
        print(f"  global_bias[logic]: {global_bias.get('logic_cells', 0):,.0f}")
        print(f"  global_bias[ram]: {global_bias.get('ram_bits', 0):,.0f}")

        holdout_metrics_with = evaluate_model(
            model_name, per_submod, holdout_df, holdout_cfgs, holdout_feats,
            args.side, args.kappa, True, global_bias
        )

        val_metrics_with = evaluate_model(
            model_name, per_submod, val_df, val_cfgs, val_feats,
            args.side, args.kappa, True, global_bias
        )

        print(f"\n  Holdout: MAE={holdout_metrics_with['abs_mean_error_pct']:.1f}%, range {holdout_metrics_with['min_error_pct']:+.0f}% to {holdout_metrics_with['max_error_pct']:+.0f}%")
        print(f"  Validation: MAE={val_metrics_with['abs_mean_error_pct']:.1f}%, range {val_metrics_with['min_error_pct']:+.0f}% to {val_metrics_with['max_error_pct']:+.0f}%")

        # Evaluate without global bias (Two-Stage approach)
        print(f"\n{model_name} WITHOUT global_bias (Two-Stage):")

        holdout_metrics_without = evaluate_model(
            model_name, per_submod, holdout_df, holdout_cfgs, holdout_feats,
            args.side, args.kappa, False, None
        )

        val_metrics_without = evaluate_model(
            model_name, per_submod, val_df, val_cfgs, val_feats,
            args.side, args.kappa, False, None
        )

        print(f"  Holdout: MAE={holdout_metrics_without['abs_mean_error_pct']:.1f}%, range {holdout_metrics_without['min_error_pct']:+.0f}% to {holdout_metrics_without['max_error_pct']:+.0f}%")
        print(f"  Validation: MAE={val_metrics_without['abs_mean_error_pct']:.1f}%, range {val_metrics_without['min_error_pct']:+.0f}% to {val_metrics_without['max_error_pct']:+.0f}%")

        results[f"{model_name}_with_bias"] = {
            "holdout": holdout_metrics_with,
            "validation": val_metrics_with,
        }
        results[f"{model_name}_no_bias"] = {
            "holdout": holdout_metrics_without,
            "validation": val_metrics_without,
        }

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Model':<30s} {'Holdout MAE%':<15s} {'Val MAE%':<15s} {'Val Range'}")
    print("-" * 70)

    for model_key in sorted(results.keys()):
        holdout_mae = results[model_key]["holdout"]["abs_mean_error_pct"]
        val_mae = results[model_key]["validation"]["abs_mean_error_pct"]
        val_min = results[model_key]["validation"]["min_error_pct"]
        val_max = results[model_key]["validation"]["max_error_pct"]

        print(f"{model_key:<30s} {holdout_mae:>13.1f}% {val_mae:>13.1f}% {val_min:>+6.0f}% to {val_max:>+6.0f}%")

    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]["validation"]["abs_mean_error_pct"])
    best_val_mae = results[best_model]["validation"]["abs_mean_error_pct"]

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model}")
    print(f"  Validation MAE: {best_val_mae:.1f}%")
    print(f"{'='*70}")

    # Detailed breakdown for best model
    print(f"\nDetailed breakdown for {best_model}:")
    print(f"\nHoldout set by cost range:")
    for range_label, error in results[best_model]["holdout"]["range_errors"].items():
        print(f"  {range_label:<12s}: {error:>+7.1f}%")

    print(f"\nValidation set by cost range:")
    for range_label, error in results[best_model]["validation"]["range_errors"].items():
        print(f"  {range_label:<12s}: {error:>+7.1f}%")


if __name__ == "__main__":
    main()
