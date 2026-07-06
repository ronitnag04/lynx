#!/usr/bin/env python3
"""
Fit a hw_cost_model that predicts total cost DIRECTLY (single whole-design
model) instead of decomposing into per-submodule predictions.

The per-submodule additive approach in ``fit_from_yosys.py`` introduces a
large systematic bias (up to +48% on small serializer configs) because each
submodule's intercept is fit independently and averaged, then a global bias
tries to reconcile the sum against the totals. Fitting the totals directly
sidesteps that entirely: the structural features encode the synthesis result
near-deterministically, so every regressor lands at <0.03% MAE.

To stay a drop-in for the existing pipeline (``use_trained_model`` ->
``TrainedCostModel.predict`` -> ``(N, 2)`` array), this emits a
``TrainedCostModel`` with a SINGLE synthetic bucket named ``"total"`` holding
two ``SubmodulePredictor``s: one fit against ``total_logic_cells``, one against
``total_ram_bits``. ``global_bias`` stays zero.

Model types:
  * ``NNLS``   -- linear, non-negative coefficients (monotone; recommended
                  for Pareto search since cost never drops as a knob grows).
  * ``Ridge``  -- linear, L2-regularized (coefficients may be negative).
  * ``RandomForest`` / ``GradientBoosting`` -- tree ensembles (kind="gbm").

Usage:
    python3 -m hw_cost_model.fit_direct_model \\
        --input hw_cost_model/yosys_sweep_results_ser.csv \\
        --side ser \\
        --model-type NNLS \\
        --output hw_cost_model/checkpoints/ser_cost_model.joblib \\
        --metrics-json hw_cost_model/checkpoints/ser_cost_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

# Add parent so ``hw_cost_model`` resolves when run as a module or script.
_LYNX_DIR = Path(__file__).resolve().parent.parent
if str(_LYNX_DIR) not in sys.path:
    sys.path.insert(0, str(_LYNX_DIR))

from hw_cost_model.defaults import DES_PARAM_COLUMNS, SER_PARAM_COLUMNS
from hw_cost_model.hw_cost_model import (
    DEFAULT_KAPPA,
    SubmodulePredictor,
    TrainedCostModel,
)
from hw_cost_model.structural_features import features_for_side

LABELS = ("logic_cells", "ram_bits")
BUCKET = "total"  # single synthetic bucket covering the whole design


def _fit_one_label(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
) -> SubmodulePredictor:
    """Fit a single (whole-design) predictor for one label."""
    if model_type == "NNLS":
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        beta, _ = nnls(X_aug, y)
        return SubmodulePredictor(
            submodule=BUCKET,
            label="",  # filled by caller
            feature_names=tuple(feat_names),
            kind="linear",
            coef=beta[:-1],
            intercept=float(beta[-1]),
        )
    if model_type == "Ridge":
        est = Ridge(alpha=1.0, random_state=42)
        est.fit(X, y)
        return SubmodulePredictor(
            submodule=BUCKET,
            label="",
            feature_names=tuple(feat_names),
            kind="linear",
            coef=np.asarray(est.coef_, dtype=np.float64),
            intercept=float(est.intercept_),
        )
    if model_type == "RandomForest":
        est = RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        )
    elif model_type == "GradientBoosting":
        est = GradientBoostingRegressor(
            n_estimators=100, max_depth=10, random_state=42, learning_rate=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")
    est.fit(X, y)
    return SubmodulePredictor(
        submodule=BUCKET,
        label="",
        feature_names=tuple(feat_names),
        kind="gbm",
        estimator=est,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--side", choices=["ser", "des"], required=True)
    parser.add_argument(
        "--model-type",
        choices=["NNLS", "Ridge", "RandomForest", "GradientBoosting"],
        default="NNLS",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument(
        "--kappa",
        type=float,
        default=DEFAULT_KAPPA,
        help="Only used for the scalar diagnostic printed below; NOT baked in.",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    if "side" in df.columns:
        df = df[df["side"] == args.side].copy()

    # Train / holdout split: honor the sample_group column if present.
    if "sample_group" in df.columns and (df["sample_group"] == "holdout").any():
        train_df = df[df["sample_group"] != "holdout"].copy()
        holdout_df = df[df["sample_group"] == "holdout"].copy()
    else:
        train_df = df.sample(frac=0.9, random_state=42)
        holdout_df = df.drop(train_df.index)

    print(f"  Training: {len(train_df)}")
    print(f"  Holdout:  {len(holdout_df)}")

    param_cols = DES_PARAM_COLUMNS if args.side == "des" else SER_PARAM_COLUMNS
    train_cfgs = train_df[param_cols].to_dict("records")
    holdout_cfgs = holdout_df[param_cols].to_dict("records")

    train_feats = [features_for_side(c, args.side) for c in train_cfgs]
    holdout_feats = [features_for_side(c, args.side) for c in holdout_cfgs]

    feat_names = sorted(train_feats[0].keys())
    print(f"  Features: {len(feat_names)}")

    X_train = np.array([[f.get(k, 0.0) for k in feat_names] for f in train_feats])

    # Fit one predictor per label against the DESIGN TOTAL for that label.
    print(f"\nTraining {args.model_type} (direct, whole-design)...")
    per_submod: dict[str, dict[str, SubmodulePredictor]] = {BUCKET: {}}
    for label in LABELS:
        col = f"total_{label}"
        if col not in train_df.columns:
            raise SystemExit(f"[{args.side}] missing required column {col!r}")
        y = train_df[col].to_numpy(dtype=np.float64)
        pred = _fit_one_label(args.model_type, X_train, y, feat_names)
        pred.label = label
        per_submod[BUCKET][label] = pred

    model = TrainedCostModel(side=args.side, per_submodule=per_submod)
    # global_bias intentionally left at its {logic_cells: 0, ram_bits: 0} default.

    # -- diagnostics: per-label + scalar MAPE on train & holdout -------------
    def report(split_name: str, sdf: pd.DataFrame, cfgs: list[dict]) -> dict:
        two_vec = model.predict(cfgs)
        out = {}
        for j, label in enumerate(LABELS):
            actual = sdf[f"total_{label}"].to_numpy(dtype=np.float64)
            pred = two_vec[:, j]
            denom = np.clip(np.abs(actual), 1e-9, None)
            mape = float(np.mean(np.abs((actual - pred) / denom))) * 100
            rho = (
                float(spearmanr(actual, pred).statistic)
                if len(actual) >= 3
                else float("nan")
            )
            out[f"{split_name}_{label}_mape"] = mape
            out[f"{split_name}_{label}_spearman"] = rho
        # scalar cost diagnostic
        actual_scalar = (
            sdf["total_logic_cells"].to_numpy(dtype=np.float64)
            + args.kappa * sdf["total_ram_bits"].to_numpy(dtype=np.float64)
        )
        pred_scalar = two_vec[:, 0] + args.kappa * two_vec[:, 1]
        denom = np.clip(np.abs(actual_scalar), 1e-9, None)
        out[f"{split_name}_scalar_mape"] = float(
            np.mean(np.abs((actual_scalar - pred_scalar) / denom))
        ) * 100
        return out

    metrics = {
        "model_type": args.model_type,
        "side": args.side,
        "kappa": args.kappa,
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
        "n_features": len(feat_names),
    }
    metrics.update(report("train", train_df, train_cfgs))
    metrics.update(report("holdout", holdout_df, holdout_cfgs))

    print("\nResults (MAPE %):")
    print(f"  train   logic={metrics['train_logic_cells_mape']:.3f}  "
          f"ram={metrics['train_ram_bits_mape']:.3f}  "
          f"scalar(k={args.kappa})={metrics['train_scalar_mape']:.3f}")
    print(f"  holdout logic={metrics['holdout_logic_cells_mape']:.3f}  "
          f"ram={metrics['holdout_ram_bits_mape']:.3f}  "
          f"scalar(k={args.kappa})={metrics['holdout_scalar_mape']:.3f}")
    print(f"  holdout spearman: logic={metrics['holdout_logic_cells_spearman']:.4f}  "
          f"ram={metrics['holdout_ram_bits_spearman']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"\nWrote {args.output}")

    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {args.metrics_json}")


if __name__ == "__main__":
    main()
