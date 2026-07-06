#!/usr/bin/env python3
"""
Compare regression methods training DIRECTLY on total cost (not per-submodule).

This avoids the per-submodule summing issue and lets models learn
the full relationship between parameters and total cost.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from scipy.optimize import nnls

# Add parent to path
_LYNX_DIR = Path(__file__).resolve().parent.parent
if str(_LYNX_DIR) not in sys.path:
    sys.path.insert(0, str(_LYNX_DIR))

from hw_cost_model.defaults import DES_PARAM_COLUMNS, SER_PARAM_COLUMNS
from hw_cost_model.structural_features import features_for_side


def train_and_evaluate(
    model_name: str,
    model_class: any,
    model_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train a model and evaluate on holdout and validation sets."""

    # Train
    if model_name == "NNLS":
        # Special handling for NNLS
        X_aug = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        beta, _ = nnls(X_aug, y_train)
        coef = beta[:-1]
        intercept = beta[-1]

        # Predict
        holdout_preds = X_holdout @ coef + intercept
        val_preds = X_val @ coef + intercept
    else:
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Predict
        holdout_preds = model.predict(X_holdout)
        val_preds = model.predict(X_val)

    # Ensure non-negative
    holdout_preds = np.maximum(holdout_preds, 0)
    val_preds = np.maximum(val_preds, 0)

    # Compute errors
    holdout_errors = ((holdout_preds - y_holdout) / y_holdout) * 100
    val_errors = ((val_preds - y_val) / y_val) * 100

    return {
        "holdout_mae": float(np.mean(np.abs(holdout_errors))),
        "holdout_min": float(np.min(holdout_errors)),
        "holdout_max": float(np.max(holdout_errors)),
        "val_mae": float(np.mean(np.abs(val_errors))),
        "val_min": float(np.min(val_errors)),
        "val_max": float(np.max(val_errors)),
        "holdout_preds": holdout_preds,
        "val_preds": val_preds,
    }


def compute_range_errors(preds: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute errors by cost range."""
    errors = ((preds - actuals) / actuals) * 100

    bins = [0, 175000, 225000, 275000, 1000000]
    labels = ["<175k", "175-225k", "225-275k", ">275k"]

    range_errors = {}
    for i in range(len(bins) - 1):
        mask = (actuals >= bins[i]) & (actuals < bins[i + 1])
        if mask.sum() > 0:
            range_errors[labels[i]] = float(np.mean(errors[mask]))

    return range_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["ser", "des"], default="ser")
    parser.add_argument("--kappa", type=float, default=0.0001)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("hw_cost_model/yosys_sweep_results_ser.csv"),
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path(
            "ml_model/results/hpb_verilator/serializer_ml_models_v1/pareto_comparison_ser/yosys_validation_ser.csv"
        ),
    )
    args = parser.parse_args()

    print(f"Loading data for {args.side}...")

    # Load training data
    train_df = pd.read_csv(args.train_data)

    # Split train/holdout
    if "sample_group" in train_df.columns and (train_df["sample_group"] == "holdout").any():
        train_mask = train_df["sample_group"] != "holdout"
    else:
        train_mask = train_df.sample(frac=0.9, random_state=42).index
        train_mask = train_df.index.isin(train_mask)

    train_data = train_df[train_mask]
    holdout_data = train_df[~train_mask]

    # Load validation data
    val_df = pd.read_csv(args.val_data)

    print(f"  Training: {len(train_data)}")
    print(f"  Holdout: {len(holdout_data)}")
    print(f"  Validation: {len(val_df)}")

    # Extract features
    param_cols = DES_PARAM_COLUMNS if args.side == "des" else SER_PARAM_COLUMNS

    train_cfgs = train_data[param_cols].to_dict("records")
    holdout_cfgs = holdout_data[param_cols].to_dict("records")
    val_cfgs = val_df[param_cols].to_dict("records")

    train_feats = [features_for_side(c, args.side) for c in train_cfgs]
    holdout_feats = [features_for_side(c, args.side) for c in holdout_cfgs]
    val_feats = [features_for_side(c, args.side) for c in val_cfgs]

    # Get feature names (use first sample to get all keys)
    feat_names = sorted(train_feats[0].keys())
    print(f"  Features: {len(feat_names)}")

    # Build feature matrices
    X_train = np.array([[f.get(k, 0.0) for k in feat_names] for f in train_feats])
    X_holdout = np.array([[f.get(k, 0.0) for k in feat_names] for f in holdout_feats])
    X_val = np.array([[f.get(k, 0.0) for k in feat_names] for f in val_feats])

    # Target: total cost (logic + kappa * ram)
    y_train = (
        train_data["total_logic_cells"].values
        + args.kappa * train_data["total_ram_bits"].values
    )
    y_holdout = (
        holdout_data["total_logic_cells"].values
        + args.kappa * holdout_data["total_ram_bits"].values
    )
    y_val = (
        val_df["total_logic_cells"].values + args.kappa * val_df["total_ram_bits"].values
    )

    print(f"\nTarget range:")
    print(f"  Training: {y_train.min():,.0f} - {y_train.max():,.0f}")
    print(f"  Validation: {y_val.min():,.0f} - {y_val.max():,.0f}")

    # Models to test
    models = {
        "NNLS": (None, {}),
        "RandomForest": (
            RandomForestRegressor,
            {"n_estimators": 100, "max_depth": 20, "random_state": 42, "n_jobs": -1},
        ),
        "GradientBoosting": (
            GradientBoostingRegressor,
            {"n_estimators": 100, "max_depth": 10, "random_state": 42, "learning_rate": 0.1},
        ),
        "Ridge": (Ridge, {"alpha": 1.0, "random_state": 42}),
    }

    results = {}

    for model_name, (model_class, model_params) in models.items():
        print(f"\n{'='*70}")
        print(f"Training {model_name}...")
        print(f"{'='*70}")

        result = train_and_evaluate(
            model_name,
            model_class,
            model_params,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            X_val,
            y_val,
        )

        results[model_name] = result

        print(f"\nHoldout:    MAE={result['holdout_mae']:5.1f}%, range {result['holdout_min']:+6.0f}% to {result['holdout_max']:+6.0f}%")
        print(f"Validation: MAE={result['val_mae']:5.1f}%, range {result['val_min']:+6.0f}% to {result['val_max']:+6.0f}%")

        # Range breakdown
        holdout_ranges = compute_range_errors(result["holdout_preds"], y_holdout)
        val_ranges = compute_range_errors(result["val_preds"], y_val)

        print(f"\nValidation by cost range:")
        for label in ["<175k", "175-225k", "225-275k", ">275k"]:
            if label in val_ranges:
                print(f"  {label:<12s}: {val_ranges[label]:>+7.1f}%")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20s} {'Holdout MAE%':<15s} {'Val MAE%':<15s} {'Val Range'}")
    print("-" * 70)

    for model_name in ["NNLS", "RandomForest", "GradientBoosting", "Ridge"]:
        r = results[model_name]
        print(
            f"{model_name:<20s} {r['holdout_mae']:>13.1f}% {r['val_mae']:>13.1f}% {r['val_min']:>+6.0f}% to {r['val_max']:>+6.0f}%"
        )

    # Best model
    best = min(results.keys(), key=lambda k: results[k]["val_mae"])
    print(f"\n{'='*70}")
    print(f"BEST: {best} (Validation MAE: {results[best]['val_mae']:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
