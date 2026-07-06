#!/usr/bin/env python3
"""
Exhaustive hardware configuration search for Lynx ProtoAccel.

This mirrors the output + reporting style of `optimize_hw_config.py`, but instead
of sampling/refining (LHS + NSGA-II), it evaluates the entire discrete hardware
parameter grid and returns the strict Pareto front.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any
import joblib
import torch
from model import LynxMLModel

import numpy as np

from util import (
    DEFAULT_CONFIG_BY_SIDE,
    DEFAULT_KAPPA,
    DES_PARAM_COLUMNS,
    PARAM_VALUES_BY_SIDE,
    SER_PARAM_COLUMNS,
    SIDE_TO_NAME,
    hardware_cost,
    hardware_cost_2vec,
    use_trained_model,
)


def default_config(side: str) -> dict[str, int]:
    return dict(DEFAULT_CONFIG_BY_SIDE[side])


def get_param_values(side: str) -> dict[str, list[int]]:
    return PARAM_VALUES_BY_SIDE[side]


def flatten_extracted_features(bench_features: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in bench_features.items():
        if isinstance(v, list):
            for i, item in enumerate(v):
                out[f"{k}_{i}"] = float(item)
        else:
            out[k] = float(v)
    return out


def load_benchmark_feature_rows(features_path: Path) -> list[dict[str, float]]:
    raw = json.loads(features_path.read_text())
    rows: list[dict[str, float]] = []
    for bench_name, feats in raw.items():
        r = flatten_extracted_features(feats)
        r["bench_name"] = bench_name  # debug/traceability; not used in model features
        rows.append(r)
    return rows


def load_model(checkpoint_path: Path, input_size: int, device: str, model_type: str = "neural") -> Any:
    """Load a trained model - either neural network or sklearn model."""
    if model_type == "neural":
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Keep architecture identical to training/optimizer.
        model = LynxMLModel(input_size=input_size, hidden_dims=[256, 128], output_size=1)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        model.eval()
        return model
    else:
        # Load sklearn model
        model = joblib.load(checkpoint_path)
        return model


def cartesian_product_indices(sizes: np.ndarray) -> np.ndarray:
    """Return all index combinations for per-dimension sizes.

    Produces an array of shape (prod(sizes), len(sizes)) with dtype int64.
    This implementation avoids `np.meshgrid`'s large intermediate tensors.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.size == 0:
        return np.empty((0, 0), dtype=np.int64)
    if np.any(sizes <= 0):
        raise ValueError(f"All sizes must be >0; got {sizes.tolist()}")

    n = int(np.prod(sizes, dtype=np.int64))
    d = int(sizes.size)
    out = np.empty((n, d), dtype=np.int64)
    stride = 1
    for j in range(d - 1, -1, -1):
        v = np.arange(int(sizes[j]), dtype=np.int64)
        reps = n // (int(sizes[j]) * stride)
        col = np.tile(np.repeat(v, stride), reps)
        out[:, j] = col
        stride *= int(sizes[j])
    return out


@dataclass
class SearchSpace:
    param_values: dict[str, list[int]]
    search_param_order: list[str]
    param_n_values: np.ndarray
    param_col_idx: dict[str, int]

    @classmethod
    def from_param_values(cls, param_values: dict[str, list[int]]) -> "SearchSpace":
        order = list(param_values.keys())
        return cls(
            param_values=param_values,
            search_param_order=order,
            param_n_values=np.array([len(param_values[p]) for p in order], dtype=np.int64),
            param_col_idx={p: i for i, p in enumerate(order)},
        )

    def encode_cfg(self, cfg: dict[str, int]) -> np.ndarray:
        idx = np.empty(len(self.search_param_order), dtype=np.int64)
        for i, p in enumerate(self.search_param_order):
            idx[i] = self.param_values[p].index(int(cfg[p]))
        return idx

    def decode_indices(self, indices: np.ndarray) -> list[dict[str, int]]:
        cfgs: list[dict[str, int]] = []
        for row in indices:
            cfg = {p: int(self.param_values[p][int(row[i])]) for i, p in enumerate(self.search_param_order)}
            cfgs.append(cfg)
        return cfgs


class FeatureBuilder:
    def __init__(
        self,
        side: str,
        scaler: Any,
        space: SearchSpace,
        benchmark_rows: list[dict[str, float]],
    ) -> None:
        self.side = side
        self.space = space
        self.feature_names = [str(x) for x in scaler.feature_names_in_]
        self.n_features = len(self.feature_names)
        self.n_benchmarks = len(benchmark_rows)
        self._mean = np.asarray(scaler.mean_, dtype=np.float32)
        self._scale = np.asarray(scaler.scale_, dtype=np.float32)

        self._col_idx = {c: i for i, c in enumerate(self.feature_names)}
        self._param_cols = SER_PARAM_COLUMNS if side == "ser" else DES_PARAM_COLUMNS
        missing_params = [p for p in self._param_cols if p not in self._col_idx]
        if missing_params:
            raise ValueError(f"Scaler missing expected param columns: {missing_params}")

        # Precompute benchmark-only contribution tensor (n_benchmarks, n_features).
        bench = np.zeros((self.n_benchmarks, self.n_features), dtype=np.float32)
        for bi, row in enumerate(benchmark_rows):
            for k, v in row.items():
                feat_key = f"feat_{k}"
                if feat_key in self._col_idx:
                    bench[bi, self._col_idx[feat_key]] = float(v)
        self._bench_contrib = bench

        # Precompute per-knob LUTs into feature columns.
        self._param_luts: list[np.ndarray] = []
        for p in self.space.search_param_order:
            lut = np.zeros((len(self.space.param_values[p]), self.n_features), dtype=np.float32)
            if p in self._col_idx:
                for vi, val in enumerate(self.space.param_values[p]):
                    lut[vi, self._col_idx[p]] = float(val)
            self._param_luts.append(lut)

    def transform(self, indices: np.ndarray) -> np.ndarray:
        n_cfg = indices.shape[0]
        if n_cfg == 0:
            return np.empty((0, self.n_features), dtype=np.float32)
        cfg_rows = np.zeros((n_cfg, self.n_features), dtype=np.float32)
        for j, lut in enumerate(self._param_luts):
            cfg_rows += lut[indices[:, j]]
        out = self._bench_contrib[None, :, :] + cfg_rows[:, None, :]
        out = out.reshape(n_cfg * self.n_benchmarks, self.n_features)
        out = (out - self._mean) / self._scale
        return out


def evaluate_indices(
    indices: np.ndarray,
    side: str,
    model: Any,
    builder: FeatureBuilder,
    device: str,
    batch_size: int,
    *,
    num_objectives: int = 3,
    kappa: float = DEFAULT_KAPPA,
    model_type: str = "neural",
) -> np.ndarray:
    """Objective columns:
       * ``num_objectives=3`` -> ``[-throughput, logic_cells, ram_bits]``.
       * ``num_objectives=2`` -> ``[-throughput, logic_cells + kappa*ram_bits]``.
    """
    if indices.shape[0] == 0:
        return np.empty((0, num_objectives), dtype=np.float64)
    feats = builder.transform(indices)
    preds = np.empty(indices.shape[0] * builder.n_benchmarks, dtype=np.float32)

    if model_type == "neural":
        with torch.no_grad():
            for i in range(0, feats.shape[0], batch_size):
                j = min(i + batch_size, feats.shape[0])
                x = torch.from_numpy(feats[i:j]).to(device)
                y = model(x).squeeze(-1).detach().cpu().numpy()
                preds[i:j] = y
    else:
        # Sklearn model
        for i in range(0, feats.shape[0], batch_size):
            j = min(i + batch_size, feats.shape[0])
            y = model.predict(feats[i:j])
            preds[i:j] = y

    mean_tp = preds.reshape(indices.shape[0], builder.n_benchmarks).mean(axis=1).astype(np.float64)
    cfgs = builder.space.decode_indices(indices)
    if num_objectives == 3:
        two_vec = hardware_cost_2vec(cfgs, side=side)
        return np.stack([-mean_tp, two_vec[:, 0], two_vec[:, 1]], axis=1)
    if num_objectives == 2:
        costs = hardware_cost(cfgs, side=side, kappa=kappa)
        return np.stack([-mean_tp, costs], axis=1)
    raise ValueError(f"num_objectives must be 2 or 3, got {num_objectives}")


def evaluate_all_indices(
    all_idx: np.ndarray,
    side: str,
    model: Any,
    builder: FeatureBuilder,
    device: str,
    inference_batch_size: int,
    cfg_chunk_size: int,
    *,
    num_objectives: int = 3,
    kappa: float = DEFAULT_KAPPA,
    model_type: str = "neural",
) -> np.ndarray:
    if all_idx.shape[0] == 0:
        return np.empty((0, num_objectives), dtype=np.float64)

    f = np.empty((all_idx.shape[0], num_objectives), dtype=np.float64)
    for i in range(0, all_idx.shape[0], cfg_chunk_size):
        j = min(i + cfg_chunk_size, all_idx.shape[0])
        f[i:j] = evaluate_indices(
            all_idx[i:j], side, model, builder, device, inference_batch_size,
            num_objectives=num_objectives, kappa=kappa, model_type=model_type,
        )
    return f


def flag_validation_candidates(pareto_f: np.ndarray, k: int) -> np.ndarray:
    """Flag extreme + evenly-spaced Pareto points; handles ``n_obj`` in {2, 3}."""
    n = len(pareto_f)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags
    n_obj = pareto_f.shape[1]
    flags[int(np.argmin(pareto_f[:, 0]))] = True
    for c in range(1, n_obj):
        flags[int(np.argmin(pareto_f[:, c]))] = True

    n_flagged = int(flags.sum())
    n_extra = max(0, k - n_flagged)
    if n_extra > 0 and n >= 2:
        cost_score = np.sqrt(np.sum(pareto_f[:, 1:] ** 2, axis=1))
        order = np.argsort(cost_score)
        pos = np.linspace(0, n - 1, n_extra + 2)[1:-1].astype(int)
        for p in pos:
            flags[int(order[p])] = True
    return flags


def strict_pareto_front(f: np.ndarray) -> np.ndarray:
    """Strict non-dominated set for k-D minimization. Returns indices."""
    if f.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if f.shape[1] == 2:
        return _strict_pareto_front_2d_min(f)
    # General nd fallback using pymoo's NDS (already imported by the optimizer).
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    front = NonDominatedSorting().do(f, only_non_dominated_front=True)
    return np.asarray(front, dtype=np.int64)


def _strict_pareto_front_2d_min(f: np.ndarray) -> np.ndarray:
    idx = np.lexsort((f[:, 1], f[:, 0]))
    f_sorted = f[idx]

    keep_sorted_pos = np.zeros(f_sorted.shape[0], dtype=bool)
    best_f1 = np.inf

    start = 0
    n = f_sorted.shape[0]
    while start < n:
        f0_val = f_sorted[start, 0]
        end = start + 1
        while end < n and f_sorted[end, 0] == f0_val:
            end += 1

        group_min_f1 = float(f_sorted[start, 1])
        if group_min_f1 < best_f1:
            m = start + 1
            while m < end and float(f_sorted[m, 1]) == group_min_f1:
                m += 1
            keep_sorted_pos[start:m] = True
            best_f1 = group_min_f1

        start = end

    return idx[keep_sorted_pos]


def final_pareto(archive_x: np.ndarray, archive_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if archive_x.shape[0] == 0:
        return archive_x, archive_f
    _, keep = np.unique(archive_x, axis=0, return_index=True)
    keep.sort()
    x = archive_x[keep]
    f = archive_f[keep]
    pareto_idx = strict_pareto_front(f)
    return x[pareto_idx], f[pareto_idx]


def compute_sensitivity(
    pareto_X: np.ndarray,
    pareto_F: np.ndarray,
    space: SearchSpace,
    eval_fn: Any,
) -> list[dict[str, dict[str, float]]]:
    """Per Pareto point, for each parameter, try index ±1 and report worst |Δ|
    across every objective column."""
    if pareto_X.size == 0:
        return []

    n_params = len(space.search_param_order)
    n_obj = pareto_F.shape[1]
    perturb_rows: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []
    for i in range(len(pareto_X)):
        for j in range(n_params):
            for delta in (-1, 1):
                new_val = int(pareto_X[i, j]) + delta
                if not (0 <= new_val < space.param_n_values[j]):
                    continue
                pert = pareto_X[i].copy()
                pert[j] = new_val
                perturb_rows.append(pert)
                meta.append((i, j))

    if not perturb_rows:
        return [{} for _ in range(len(pareto_X))]

    perturb_X = np.stack(perturb_rows, axis=0)
    F_pert = eval_fn(perturb_X)

    buckets: list[dict[str, dict[str, list[float]]]] = [
        {p: {f"dobj{k}": [] for k in range(n_obj)} for p in space.search_param_order}
        for _ in range(len(pareto_X))
    ]
    for k, (i, j) in enumerate(meta):
        param = space.search_param_order[j]
        for obj_idx in range(n_obj):
            d = float(F_pert[k, obj_idx] - pareto_F[i, obj_idx])
            buckets[i][param][f"dobj{obj_idx}"].append(d)

    obj_labels_by_ncol = {
        2: ("max_abs_dthroughput", "max_abs_dcost"),
        3: ("max_abs_dthroughput", "max_abs_dlogic_cells", "max_abs_dram_bits"),
    }
    labels = obj_labels_by_ncol.get(n_obj) or tuple(
        f"max_abs_dobj{k}" for k in range(n_obj)
    )
    reduced: list[dict[str, dict[str, float]]] = []
    for b in buckets:
        entry: dict[str, dict[str, float]] = {}
        for p, d in b.items():
            if d["dobj0"]:
                entry[p] = {
                    labels[k]: float(np.max(np.abs(d[f"dobj{k}"])))
                    for k in range(n_obj)
                }
        reduced.append(entry)
    return reduced


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exhaustively evaluate all Lynx hardware configurations and extract the Pareto front."
    )
    p.add_argument("--side", choices=["ser", "des"], required=True, help="Search serializer or deserializer.")
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing <side>_checkpoint.pt and <side>_scaler.joblib.",
    )
    p.add_argument(
        "--features-file",
        type=Path,
        default=Path("../analytical_model/extracted_features.json"),
        help="Path to extracted_features.json.",
    )
    p.add_argument("--output", type=Path, default=Path("pareto_front_exhaustive.json"), help="Output JSON path.")
    p.add_argument("--batch-size", type=int, default=1024, help="Inference batch size (rows of feature tensor).")
    p.add_argument(
        "--cfg-chunk-size",
        type=int,
        default=4096,
        help="How many configs to evaluate per chunk (memory bound).",
    )
    p.add_argument("--validation-k", type=int, default=8, help="Validation candidate count.")
    p.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip per-parameter ±1 index sensitivity on the final Pareto front.",
    )
    p.add_argument(
        "--limit-configs",
        type=int,
        default=0,
        help="If >0, only evaluate the first N configs (debug). Default evaluates all.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed used only for reproducible debug limiting.")
    p.add_argument("--device", type=str, default="cpu", help="Torch device.")
    p.add_argument(
        "--hw-cost-model",
        type=Path,
        default=None,
        help=(
            "Optional path to a trained TrainedCostModel joblib file "
            "(produced by hw_cost_model/fit_from_yosys.py). When omitted, "
            "the structural RTL-derived cost estimator is used."
        ),
    )
    p.add_argument(
        "--num-objectives",
        type=int,
        choices=(2, 3),
        default=3,
        help=(
            "3 -> Pareto over (throughput, logic_cells, ram_bits) [default]. "
            "2 -> Pareto over (throughput, logic_cells + kappa * ram_bits)."
        ),
    )
    p.add_argument(
        "--kappa",
        type=float,
        default=DEFAULT_KAPPA,
        help=f"Cost-combining weight used only when --num-objectives=2. Default {DEFAULT_KAPPA}.",
    )
    p.add_argument(
        "--model-type",
        type=str,
        choices=[
            "neural",
            "LinearRegression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "DecisionTree",
            "RandomForest",
            "GradientBoosting",
            "KNN",
            "SVR",
        ],
        default="neural",
        help=(
            "Type of model to use for throughput prediction. "
            "'neural' uses the neural network checkpoint. "
            "Other options use sklearn models saved by train_all_models.py."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t_total = time.perf_counter()

    if args.hw_cost_model is not None:
        use_trained_model(args.side, args.hw_cost_model)
        print(f"[cost] using trained hw-cost model: {args.hw_cost_model}")

    side_name = SIDE_TO_NAME[args.side]

    # Determine checkpoint path based on model type
    if args.model_type == "neural":
        ckpt = args.checkpoint_dir / f"{side_name}_checkpoint.pt"
    else:
        ckpt = args.checkpoint_dir / f"{side_name}_{args.model_type}_model.joblib"

    scl = args.checkpoint_dir / f"{side_name}_scaler.joblib"

    if not ckpt.is_file() or not scl.is_file():
        raise FileNotFoundError(f"Missing checkpoint/scaler in {args.checkpoint_dir}")

    param_values = get_param_values(args.side)
    space = SearchSpace.from_param_values(param_values)

    scaler = joblib.load(scl)
    benches = load_benchmark_feature_rows(args.features_file)
    builder = FeatureBuilder(args.side, scaler, space, benches)
    model = load_model(ckpt, input_size=builder.n_features, device=args.device, model_type=args.model_type)

    def eval_fn(xi: np.ndarray) -> np.ndarray:
        return evaluate_indices(
            xi, args.side, model, builder, args.device, args.batch_size,
            num_objectives=args.num_objectives, kappa=args.kappa, model_type=args.model_type,
        )

    def cost_summary(F_row: np.ndarray) -> dict[str, float]:
        if args.num_objectives == 3:
            return {
                "predicted_logic_cells": float(F_row[1]),
                "predicted_ram_bits":    float(F_row[2]),
                "predicted_cost_scalar_kappa": float(F_row[1] + args.kappa * F_row[2]),
            }
        return {"predicted_cost": float(F_row[1])}

    baseline_cfg = default_config(args.side)
    t0 = time.perf_counter()
    baseline_idx = space.encode_cfg(baseline_cfg).reshape(1, -1)
    baseline_F = eval_fn(baseline_idx)[0]
    baseline_tp = float(-baseline_F[0])
    baseline_costs = cost_summary(baseline_F)
    print(
        f"Baseline (DEFAULT_CONFIG): predicted_throughput={baseline_tp:.4f} Gbit/s, "
        f"cost={baseline_costs} ({time.perf_counter() - t0:.2f}s)"
    )

    t_enum = time.perf_counter()
    all_idx = cartesian_product_indices(space.param_n_values)
    total = int(all_idx.shape[0])
    print(
        f"Enumerated {total} configs over {len(space.search_param_order)} params "
        f"({time.perf_counter() - t_enum:.2f}s)"
    )

    if args.limit_configs and args.limit_configs > 0:
        n = min(int(args.limit_configs), total)
        choose = rng.choice(total, size=n, replace=False)
        choose.sort()
        all_idx = all_idx[choose]
        print(f"Debug limit enabled: evaluating {len(all_idx)} / {total} configs")

    t_eval = time.perf_counter()
    all_f = evaluate_all_indices(
        all_idx,
        args.side,
        model,
        builder,
        args.device,
        args.batch_size,
        args.cfg_chunk_size,
        num_objectives=args.num_objectives,
        kappa=args.kappa,
        model_type=args.model_type,
    )
    print(f"Evaluated {len(all_idx)} configs ({time.perf_counter() - t_eval:.2f}s)")

    stage4_t0 = time.perf_counter()
    pareto_x, pareto_f = final_pareto(all_idx, all_f)
    order = np.argsort(pareto_f[:, 1])
    pareto_x = pareto_x[order]
    pareto_f = pareto_f[order]

    if args.skip_sensitivity:
        sensitivities: list[dict[str, dict[str, float]]] = [{} for _ in range(len(pareto_x))]
    else:
        t_s = time.perf_counter()
        print("Computing per-parameter sensitivity")
        sensitivities = compute_sensitivity(pareto_x, pareto_f, space, eval_fn)
        print(f"Sensitivity computed in {time.perf_counter() - t_s:.2f}s")

    flags = flag_validation_candidates(pareto_f, args.validation_k)
    cfgs = space.decode_indices(pareto_x)
    print(f"Pareto extraction + output prep: {time.perf_counter() - stage4_t0:.2f}s")

    front: list[dict[str, object]] = []
    for i, cfg in enumerate(cfgs):
        entry: dict[str, object] = {
            "predicted_throughput_gbits_per_sec": float(-pareto_f[i, 0]),
            "validation_candidate": bool(flags[i]),
            "sensitivity": sensitivities[i] if i < len(sensitivities) else {},
            "config": cfg,
        }
        entry.update(cost_summary(pareto_f[i]))
        front.append(entry)

    baseline_out: dict[str, object] = {
        "predicted_throughput_gbits_per_sec": baseline_tp,
        "config": baseline_cfg,
    }
    baseline_out.update(baseline_costs)

    out = {
        "side": args.side,
        "model_type": args.model_type,
        "num_objectives": args.num_objectives,
        "kappa": args.kappa if args.num_objectives == 2 else None,
        "benchmark_count": len(benches),
        "baseline": baseline_out,
        "pareto_front": front,
        "stats": {
            "search": "exhaustive",
            "grid_total_configs": int(total),
            "evaluated_configs": int(len(all_idx)),
            "n_pareto_final": int(len(front)),
            "n_validation_candidates": int(int(flags.sum())),
            "cfg_chunk_size": int(args.cfg_chunk_size),
            "inference_batch_size": int(args.batch_size),
            "total_duration_s": round(time.perf_counter() - t_total, 2),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(front)} Pareto points -> {args.output}")


if __name__ == "__main__":
    main()
