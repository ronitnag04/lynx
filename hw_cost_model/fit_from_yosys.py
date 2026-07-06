#!/usr/bin/env python3
"""
Fit a ``TrainedCostModel`` from a Yosys cell-count sweep.

Expected input CSV (produced by ``parse_yosys_stat.py``, one row per
config):

    config_name, side,
    <all des_* and ser_* param columns>,
    submod_<bucket>_logic_cells, submod_<bucket>_ram_bits,
    submod_<bucket>_ram_flop_cells,  (auditing only)
    total_logic_cells, total_ram_bits, total_ram_flop_cells

For each bucket, the fitter trains two independent per-submodule
predictors (linear + NNLS so all coefficients are >= 0, guaranteeing
monotonicity):

  * ``submod_<bucket>_logic_cells`` -- fit against the structural
    features assigned to that bucket in ``structural_features.py``.
  * ``submod_<bucket>_ram_bits`` -- same feature slice, separate model.

At prediction time the two vectors are combined by either the
`hardware_cost_2vec` API (multi-objective) or a weighted sum with
``kappa`` (scalar). ``kappa`` defaults to 0.15 and can be overridden per
call; it is NOT baked into the fitted model.

Usage:

    python -m hw_cost_model.fit_from_yosys \\
        --input yosys_sweep_results.csv \\
        --side des \\
        --output des_cost_model.joblib \\
        --metrics-json des_metrics.json \\
        --kappa 0.15   # only used for held-out scalar diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

from .defaults import DES_PARAM_COLUMNS, SER_PARAM_COLUMNS
from .hw_cost_model import DEFAULT_KAPPA, SubmodulePredictor, TrainedCostModel
from .structural_features import (
    DES_QUEUES,
    DES_SUBMODULES,
    SER_QUEUES,
    SER_SUBMODULES,
    bucket_knobs,
    features_for_side,
)


# Labels the fitter learns; must match the CSV column suffix pattern.
LABELS: tuple[str, ...] = ("logic_cells", "ram_bits")


def _feature_names_for_bucket(side: str, bucket: str) -> list[str]:
    """Return the feature-name subset relevant to ``bucket`` for ``side``.

    Filtered from the full feature dict of a defaults config so no name
    is hard-coded here. Rules:

      * L1MemHelper buckets (``descr_l1``, ``ml_l1``, ``hbw_l1``,
        ``fw_l1``) -- features prefixed with the bucket name plus
        ``__``.
      * Queue buckets (``cr``, ``dth``, ``mw``, ``ml``, ``fw``,
        ``hbw``) -- features from the queues declared under that
        bucket in ``DES_QUEUES`` / ``SER_QUEUES``, plus any bucket-
        specific constant features (e.g. ``ml__bytelane_bits``).
      * SER-specific buckets (``fh``, ``mfh``, ``varint``) -- features
        prefixed with the bucket name plus ``__``.
      * ``top``, ``tlb``, ``tl_glue`` -- no knob-driven features; the
        submodel will be an intercept-only fit against a single
        constant feature.
    """
    from .defaults import DEFAULT_CONFIG_BY_SIDE
    all_feats = features_for_side(DEFAULT_CONFIG_BY_SIDE[side], side)

    # Every bucket collects features whose name begins with "<bucket>__".
    prefixed = sorted(k for k in all_feats if k.startswith(bucket + "__"))

    # Also include queue-derived features whose knob's submodule bucket
    # is this bucket (from DES_QUEUES / SER_QUEUES tables).
    queues = DES_QUEUES if side == "des" else SER_QUEUES
    knob_features: list[str] = []
    for q in queues:
        if q.submodule != bucket:
            continue
        knob_features.extend(sorted(
            k for k in all_feats if k.startswith(q.knob + "__")
        ))

    combined = sorted(set(prefixed) | set(knob_features))
    if combined:
        return combined
    # Buckets with no structural features (top/tlb/tl_glue) get an
    # intercept-only submodel by returning empty; NNLS handles it.
    return []


def _fit_linear_nnls(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Non-negative least squares with an appended intercept column.

    NNLS forces every coefficient >= 0. Since the structural features
    are all non-negative and monotone in the underlying config knob,
    this guarantees the fitted prediction is monotone non-decreasing in
    every knob -- essential for correct Pareto search.
    """
    if X.shape[1] == 0:
        # Intercept-only: predict the mean.
        return np.zeros(0), float(np.mean(y)) if len(y) else 0.0
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    beta, _ = nnls(X_aug, y)
    return beta[:-1], float(beta[-1])


def _fit_side(
    df: pd.DataFrame, side: str, kappa: float
) -> tuple[TrainedCostModel, dict[str, dict[str, float]]]:
    param_cols = DES_PARAM_COLUMNS if side == "des" else SER_PARAM_COLUMNS
    buckets = DES_SUBMODULES if side == "des" else SER_SUBMODULES

    missing_params = [c for c in param_cols if c not in df.columns]
    if missing_params:
        raise SystemExit(f"[{side}] missing param columns: {missing_params}")

    df = df[df["side"] == side].copy() if "side" in df.columns else df.copy()
    if df.empty:
        raise SystemExit(f"[{side}] no rows found in input CSV")

    cfgs = df[param_cols].to_dict(orient="records")
    feat_dicts = [features_for_side(c, side) for c in cfgs]

    # Prefer the ``sample_group`` column emitted by ``-t synth-training``:
    # rows tagged ``holdout`` become the held-out set and every other row is
    # training. If the column is absent, fall back to a seeded 10 % random
    # split so this script keeps working on older CSVs.
    if "sample_group" in df.columns and (df["sample_group"] == "holdout").any():
        is_holdout = (df["sample_group"] == "holdout").to_numpy()
        idx = np.arange(len(df))
        test_idx = idx[is_holdout]
        train_idx = idx[~is_holdout]
    else:
        rng = np.random.default_rng(20260630)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = max(1, len(idx) // 10)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    per_submod: dict[str, dict[str, SubmodulePredictor]] = {}
    metrics: dict[str, dict[str, float]] = {}

    for bucket in buckets:
        feat_names = _feature_names_for_bucket(side, bucket)
        X = np.array(
            [[fd.get(k, 0.0) for k in feat_names] for fd in feat_dicts],
            dtype=np.float64,
        )
        for label in LABELS:
            col = f"submod_{bucket}_{label}"
            if col not in df.columns:
                print(
                    f"[{side}] warning: missing column {col!r}; skipping this "
                    "(bucket, label) -- residual will be absorbed by global_bias",
                    file=sys.stderr,
                )
                continue
            y = df[col].to_numpy(dtype=np.float64)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test,  y_test  = X[test_idx],  y[test_idx]

            coef, intercept = _fit_linear_nnls(X_train, y_train)
            pred = SubmodulePredictor(
                submodule=bucket,
                label=label,
                feature_names=tuple(feat_names),
                kind="linear",
                coef=coef,
                intercept=intercept,
            )
            per_submod.setdefault(bucket, {})[label] = pred

            if len(feat_names) > 0:
                y_hat = X_test @ coef + intercept
            else:
                y_hat = np.full_like(y_test, intercept)
            denom = np.clip(np.abs(y_test), 1e-9, None)
            mape = float(np.mean(np.abs((y_test - y_hat) / denom)))
            rho = float(spearmanr(y_test, y_hat).statistic) if len(y_test) >= 3 else float("nan")
            metrics[f"{bucket}/{label}"] = {
                "n_train": float(len(train_idx)),
                "n_test":  float(len(test_idx)),
                "mape":    mape,
                "spearman_rho": rho,
                "n_features": float(len(feat_names)),
            }

    # Top-level metrics: predicted totals vs. observed totals on the
    # held-out set, both per-label and combined via kappa.
    model = TrainedCostModel(side=side, per_submodule=per_submod)

    y_totals: dict[str, np.ndarray] = {}
    for label in LABELS:
        col = f"total_{label}"
        if col in df.columns:
            y_totals[label] = df[col].to_numpy(dtype=np.float64)

    # Compute residual bias so the summed submodule predictions match
    # the totals on average (small correction absorbing any bucket
    # coverage gaps).
    for label in LABELS:
        if label not in y_totals:
            continue
        y_hat_per_bucket = np.zeros(len(df))
        for bucket_preds in per_submod.values():
            pred = bucket_preds.get(label)
            if pred is None:
                continue
            y_hat_per_bucket += pred.predict(feat_dicts)
        residual = float(np.mean(y_totals[label] - y_hat_per_bucket))
        model.global_bias[label] = residual

    two_vec_pred = model.predict([cfgs[i] for i in test_idx])
    for j, label in enumerate(LABELS):
        if label not in y_totals:
            continue
        y_true = y_totals[label][test_idx]
        y_hat = two_vec_pred[:, j]
        denom = np.clip(np.abs(y_true), 1e-9, None)
        mape = float(np.mean(np.abs((y_true - y_hat) / denom)))
        rho = float(spearmanr(y_true, y_hat).statistic) if len(y_true) >= 3 else float("nan")
        metrics[f"_total_/{label}"] = {
            "n_train": float(len(train_idx)),
            "n_test":  float(len(test_idx)),
            "mape":    mape,
            "spearman_rho": rho,
            "global_bias": model.global_bias[label],
        }

    # Scalar diagnostics with kappa.
    if all(l in y_totals for l in LABELS):
        y_scalar = y_totals["logic_cells"][test_idx] + kappa * y_totals["ram_bits"][test_idx]
        y_hat_scalar = two_vec_pred[:, 0] + kappa * two_vec_pred[:, 1]
        denom = np.clip(np.abs(y_scalar), 1e-9, None)
        mape = float(np.mean(np.abs((y_scalar - y_hat_scalar) / denom)))
        rho = float(spearmanr(y_scalar, y_hat_scalar).statistic) if len(y_scalar) >= 3 else float("nan")
        metrics[f"_total_/scalar_kappa={kappa}"] = {
            "n_test": float(len(test_idx)),
            "mape": mape,
            "spearman_rho": rho,
        }

    # Per-sample-group MAPE breakdown (on TRAINING set -- checks fit
    # quality per stratum, not generalization). Uses whatever
    # sample_group values appear in the CSV.
    if "sample_group" in df.columns:
        groups = df["sample_group"].to_numpy()
        y_hat_train_two = model.predict([cfgs[i] for i in train_idx])
        for label in LABELS:
            if label not in y_totals:
                continue
            j = LABELS.index(label)
            y_true = y_totals[label][train_idx]
            y_hat = y_hat_train_two[:, j]
            train_groups = groups[train_idx]
            for g in sorted(set(train_groups.tolist())):
                if not g:
                    continue
                mask = train_groups == g
                if not mask.any():
                    continue
                yt = y_true[mask]
                yh = y_hat[mask]
                denom_g = np.clip(np.abs(yt), 1e-9, None)
                metrics[f"_train_group={g}/{label}"] = {
                    "n": float(int(mask.sum())),
                    "mape": float(np.mean(np.abs((yt - yh) / denom_g))),
                }

    return model, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with per-config yosys cell/ram_bits columns.")
    parser.add_argument("--side", choices=("des", "ser"), required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write the joblib-pickled TrainedCostModel.")
    parser.add_argument("--metrics-json", type=Path, default=None,
                        help="Optional path to write held-out metrics as JSON.")
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA,
                        help=("Gate-equivalents-per-SRAM-bit used only for the "
                              "held-out scalar-cost diagnostic. Not baked into "
                              f"the fitted model. Default {DEFAULT_KAPPA}."))
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    model, metrics = _fit_side(df, args.side, args.kappa)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"wrote {args.output}")

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.metrics_json}")

    print()
    print(f"{'name':>30s}  n_test  mape       rho")
    for name, m in metrics.items():
        n = int(m.get("n_test", 0))
        mape = m.get("mape", float("nan"))
        rho = m.get("spearman_rho", float("nan"))
        print(f"  {name:>28s}  {n:5d}   {mape:8.4f}   {rho:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
