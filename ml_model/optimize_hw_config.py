#!/usr/bin/env python3
"""
Pareto-optimal hardware configuration search for Lynx ProtoAccel.

This optimizer mirrors the staged search style used in Peregrine's
`optimize_hw_config.py`, but uses benchmark-level extracted features from
`analytical_model/extracted_features.json` (not ronamol traces).

Objectives:
  1) Maximize predicted throughput (modeled as minimizing -throughput).
  2) Minimize a ProtoAccel hardware cost proxy derived from Scala sources.

Staged flow vs Peregrine (four conceptual stages):

  Stage 1 (Peregrine): constraint pruning over encoded configs. We skip this
  for Lynx: the ProtoAccel sweep grid has no separate structural validity
  predicate (every knob combination is instantiated in Chisel).

  Stage 2 (Peregrine): Latin Hypercube global sampling. We do this as our
  first numerical stage (labeled "Stage 2" in logs to match Peregrine).

  Stage 3 (Peregrine): NSGA-II refinement. Same here.

  Stage 4 (Peregrine): archive merge, strict Pareto extraction, optional
  per-parameter sensitivity, validation-candidate flagging, JSON output.
  Sensitivity can be skipped with ``--skip-sensitivity``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from model import LynxMLModel

# ---------------------------------------------------------------------------
# Parameterization Grid
# ---------------------------------------------------------------------------

SIDE_TO_NAME = {"ser": "serializer", "des": "deserializer"}

DES_PARAM_COLUMNS = [
    "des_top_descriptor_reqs",
    "des_top_memloader_reqs",
    "des_cr_rocc_commands",
    "des_dth_l1_reqs",
    "des_dth_fd_reqs",
    "des_dth_fd_resps",
    "des_fw_l1_reqs",
    "des_ml_buf_info_q",
    "des_ml_load_info_q",
]

SER_PARAM_COLUMNS = [
    "ser_field_handlers",
    "ser_cr_rocc_commands",
    "ser_dth_hasbits_reqs",
    "ser_dth_descriptor_reqs",
    "ser_dth_reg_resps",
    "ser_dth_reqs_meta",
    "ser_dth_fh_outputs",
    "ser_mw_write_input",
    "ser_mw_write_inject",
    "ser_mw_write_ptrs",
]

# Directly inlined from lynx/sample_protoacc_model/parameter_sweep.json.
PARAM_VALUES_BY_SIDE: dict[str, dict[str, list[int]]] = {
    "des": {
        "des_top_descriptor_reqs": [2, 4, 8, 16],
        "des_top_memloader_reqs": [16, 32, 64, 128],
        "des_cr_rocc_commands": [2, 4, 8],
        "des_dth_l1_reqs": [2, 4, 8, 16],
        "des_dth_fd_reqs": [2, 4, 8, 16],
        "des_dth_fd_resps": [2, 4, 8, 16],
        "des_fw_l1_reqs": [2, 4, 8, 16],
        "des_ml_buf_info_q": [8, 16, 32, 64],
        "des_ml_load_info_q": [128, 256, 512, 1024],
    },
    "ser": {
        "ser_field_handlers": [2, 4, 6, 8],
        "ser_cr_rocc_commands": [2, 4, 8],
        "ser_dth_hasbits_reqs": [2, 4, 8, 16],
        "ser_dth_descriptor_reqs": [2, 4, 8, 16],
        "ser_dth_reg_resps": [5, 10, 20],
        "ser_dth_reqs_meta": [2, 4, 8, 16],
        "ser_dth_fh_outputs": [2, 4, 8, 16],
        "ser_mw_write_input": [2, 4, 8, 16],
        "ser_mw_write_inject": [2, 4, 8, 16],
        "ser_mw_write_ptrs": [5, 10, 20],
    },
}

# Default ProtoAccel parameterization (Scala `util.scala` Field defaults), using
# the same column names as `train.py` / the scaler (des_* / ser_*).
DEFAULT_CONFIG_BY_SIDE: dict[str, dict[str, int]] = {
    "des": {
        "des_top_descriptor_reqs": 4,
        "des_top_memloader_reqs": 64,
        "des_cr_rocc_commands": 2,
        "des_dth_l1_reqs": 4,
        "des_dth_fd_reqs": 4,
        "des_dth_fd_resps": 4,
        "des_fw_l1_reqs": 4,
        "des_ml_buf_info_q": 16,
        "des_ml_load_info_q": 256,
    },
    "ser": {
        "ser_field_handlers": 6,
        "ser_cr_rocc_commands": 2,
        "ser_dth_hasbits_reqs": 4,
        "ser_dth_descriptor_reqs": 4,
        "ser_dth_reg_resps": 10,
        "ser_dth_reqs_meta": 4,
        "ser_dth_fh_outputs": 4,
        "ser_mw_write_input": 4,
        "ser_mw_write_inject": 4,
        "ser_mw_write_ptrs": 10,
    },
}


def default_config(side: str) -> dict[str, int]:
    return dict(DEFAULT_CONFIG_BY_SIDE[side])


def get_param_values(side: str) -> dict[str, list[int]]:
    return PARAM_VALUES_BY_SIDE[side]


def load_model(checkpoint_path: Path, input_size: int, device: str) -> LynxMLModel:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LynxMLModel(input_size=input_size, hidden_dims=[256, 128], output_size=1)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model


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
    rows = []
    for bench_name, feats in raw.items():
        r = flatten_extracted_features(feats)
        r["bench_name"] = bench_name
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def hardware_cost(cfgs: list[dict[str, int]], side: str) -> np.ndarray:
    if not cfgs:
        return np.empty(0, dtype=np.float64)
    c = cfgs
    out = np.zeros(len(c), dtype=np.float64)
    if side == "ser":
        # Derived from ProtoAccel serializer Scala:
        # - field_handlers duplicates SerFieldHandler + L1MemHelper/PTW wiring.
        # - other knobs mostly queue depths with per-entry width-based weighting.
        for i, x in enumerate(c):
            out[i] = (
                40.0 * x["ser_field_handlers"]
                + 2.0 * x["ser_cr_rocc_commands"]
                + 3.0 * x["ser_dth_hasbits_reqs"]
                + 3.0 * x["ser_dth_descriptor_reqs"]
                + 0.5 * x["ser_dth_reg_resps"]
                + 3.0 * x["ser_dth_reqs_meta"]
                + 5.0 * x["ser_dth_fh_outputs"]
                + 8.0 * x["ser_mw_write_input"]
                + 8.0 * x["ser_mw_write_inject"]
                + 2.0 * x["ser_mw_write_ptrs"]
            )
    else:
        # Derived from ProtoAccel deserializer Scala:
        # top outstanding-req knobs and L1 request queues model larger state.
        for i, x in enumerate(c):
            out[i] = (
                6.0 * x["des_top_descriptor_reqs"]
                + 5.0 * x["des_top_memloader_reqs"]
                + 2.0 * x["des_cr_rocc_commands"]
                + 6.0 * x["des_dth_l1_reqs"]
                + 3.0 * x["des_dth_fd_reqs"]
                + 5.0 * x["des_dth_fd_resps"]
                + 6.0 * x["des_fw_l1_reqs"]
                + 4.0 * x["des_ml_buf_info_q"]
                + 0.5 * x["des_ml_load_info_q"]
            )
    return out


# ---------------------------------------------------------------------------
# Config encoding / search space
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Feature assembly (extracted features + hardware knobs, scaled like training)
# ---------------------------------------------------------------------------


class FeatureBuilder:
    def __init__(
        self,
        side: str,
        scaler: StandardScaler,
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


# ---------------------------------------------------------------------------
# Model + batched objective evaluation
# ---------------------------------------------------------------------------


def evaluate_indices(
    indices: np.ndarray,
    side: str,
    model: LynxMLModel,
    builder: FeatureBuilder,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if indices.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    feats = builder.transform(indices)
    preds = np.empty(indices.shape[0] * builder.n_benchmarks, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, feats.shape[0], batch_size):
            j = min(i + batch_size, feats.shape[0])
            x = torch.from_numpy(feats[i:j]).to(device)
            y = model(x).squeeze(-1).detach().cpu().numpy()
            preds[i:j] = y
    mean_tp = preds.reshape(indices.shape[0], builder.n_benchmarks).mean(axis=1).astype(np.float64)
    costs = hardware_cost(builder.space.decode_indices(indices), side=side)
    return np.stack([-mean_tp, costs], axis=1)


# ---------------------------------------------------------------------------
# Stage 2: Latin Hypercube global sampling
# ---------------------------------------------------------------------------


def dedup_rows(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return x
    return np.unique(x, axis=0)


def lhs_sample(space: SearchSpace, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(space.search_param_order), seed=rng)
    u = sampler.random(n_samples)
    idx = np.floor(u * space.param_n_values).astype(np.int64)
    return np.minimum(idx, space.param_n_values - 1)


# ---------------------------------------------------------------------------
# Stage 3: NSGA-II
# ---------------------------------------------------------------------------


class HWConfigProblem(Problem):
    def __init__(self, space: SearchSpace, eval_fn: Any) -> None:
        super().__init__(
            n_var=len(space.search_param_order),
            n_obj=2,
            xl=np.zeros(len(space.search_param_order), dtype=float),
            xu=(space.param_n_values - 1).astype(float),
            vtype=int,
        )
        self._eval_fn = eval_fn
        self.archive_X: list[np.ndarray] = []
        self.archive_F: list[np.ndarray] = []

    def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        xi = np.round(X).astype(np.int64)
        f = self._eval_fn(xi)
        out["F"] = f
        self.archive_X.append(xi.copy())
        self.archive_F.append(f.copy())


# ---------------------------------------------------------------------------
# Stage 4: Pareto extraction, sensitivity, validation flagging, output
# ---------------------------------------------------------------------------


def final_pareto(archive_x: np.ndarray, archive_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, keep = np.unique(archive_x, axis=0, return_index=True)
    keep.sort()
    x = archive_x[keep]
    f = archive_f[keep]
    rows = NonDominatedSorting().do(f, only_non_dominated_front=True)
    return x[rows], f[rows]


def compute_sensitivity(
    pareto_X: np.ndarray,
    pareto_F: np.ndarray,
    space: SearchSpace,
    side: str,
    model: LynxMLModel,
    builder: FeatureBuilder,
    device: str,
    batch_size: int,
) -> list[dict[str, dict[str, float]]]:
    """Per Pareto point, for each parameter, try index ±1 and report worst |Δ| in
    throughput and cost (Lynx has no extra feasibility predicate beyond grid bounds)."""
    if pareto_X.size == 0:
        return []

    n_params = len(space.search_param_order)
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
    F_pert = evaluate_indices(perturb_X, side, model, builder, device, batch_size)

    buckets: list[dict[str, dict[str, list[float]]]] = [
        {p: {"dtp": [], "dcost": []} for p in space.search_param_order}
        for _ in range(len(pareto_X))
    ]
    for k, (i, j) in enumerate(meta):
        param = space.search_param_order[j]
        d_obj0 = float(F_pert[k, 0] - pareto_F[i, 0])
        d_obj1 = float(F_pert[k, 1] - pareto_F[i, 1])
        buckets[i][param]["dtp"].append(-d_obj0)
        buckets[i][param]["dcost"].append(d_obj1)

    reduced: list[dict[str, dict[str, float]]] = []
    for b in buckets:
        entry: dict[str, dict[str, float]] = {}
        for p, d in b.items():
            if d["dtp"]:
                entry[p] = {
                    "max_abs_dthroughput": float(np.max(np.abs(d["dtp"]))),
                    "max_abs_dcost": float(np.max(np.abs(d["dcost"]))),
                }
        reduced.append(entry)
    return reduced


def flag_validation_candidates(pareto_f: np.ndarray, k: int) -> np.ndarray:
    n = len(pareto_f)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags
    flags[int(np.argmin(pareto_f[:, 0]))] = True  # throughput optimum (max tp)
    flags[int(np.argmin(pareto_f[:, 1]))] = True  # cost optimum
    n_extra = max(0, k - 2)
    if n_extra > 0 and n >= 2:
        order = np.argsort(pareto_f[:, 1])
        pos = np.linspace(0, n - 1, n_extra + 2)[1:-1].astype(int)
        for p in pos:
            flags[int(order[p])] = True
    return flags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pareto-optimal Lynx hardware configuration search.")
    p.add_argument("--side", choices=["ser", "des"], required=True, help="Optimize serializer or deserializer.")
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
    p.add_argument("--output", type=Path, default=Path("pareto_front.json"), help="Output JSON path.")
    p.add_argument("--lhs-samples", type=int, default=64*1024, help="Number of LHS samples for Stage 1.")
    p.add_argument("--near-pareto-frac", type=float, default=0.05, help="Fraction retained for NSGA seed.")
    p.add_argument("--nsga-pop-size", type=int, default=256, help="NSGA-II population size.")
    p.add_argument("--nsga-generations", type=int, default=80, help="NSGA-II generations.")
    p.add_argument("--batch-size", type=int, default=1024, help="Inference batch size.")
    p.add_argument("--validation-k", type=int, default=8, help="Validation candidate count.")
    p.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip per-parameter ±1 index sensitivity on the final Pareto front (Stage 4).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", help="Torch device.")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-generation NSGA-II progress from pymoo (same as Peregrine).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t_total = time.perf_counter()

    side_name = SIDE_TO_NAME[args.side]
    ckpt = args.checkpoint_dir / f"{side_name}_checkpoint.pt"
    scl = args.checkpoint_dir / f"{side_name}_scaler.joblib"
    if not ckpt.is_file() or not scl.is_file():
        raise FileNotFoundError(f"Missing checkpoint/scaler in {args.checkpoint_dir}")

    param_values = get_param_values(args.side)
    space = SearchSpace.from_param_values(param_values)

    scaler: StandardScaler = joblib.load(scl)
    benches = load_benchmark_feature_rows(args.features_file)
    builder = FeatureBuilder(args.side, scaler, space, benches)
    model = load_model(ckpt, input_size=builder.n_features, device=args.device)

    # -------------------------------------------------------------------------
    # Stage 1: (skipped) — Peregrine prunes invalid configs; Lynx grid is all valid.
    # -------------------------------------------------------------------------

    baseline_cfg = default_config(args.side)
    t0 = time.perf_counter()
    baseline_idx = space.encode_cfg(baseline_cfg).reshape(1, -1)
    baseline_F = evaluate_indices(
        baseline_idx, args.side, model, builder, args.device, args.batch_size
    )[0]
    baseline_tp = float(-baseline_F[0])
    baseline_cost = float(baseline_F[1])
    print(
        f"Baseline (DEFAULT_CONFIG): predicted_throughput={baseline_tp:.4f} Gbit/s, "
        f"cost={baseline_cost:.2f} ({time.perf_counter() - t0:.2f}s)"
    )

    # -------------------------------------------------------------------------
    # Stage 2: Latin Hypercube global sampling + seed pool for NSGA-II
    # -------------------------------------------------------------------------
    print(f"[Stage 2] LHS {args.lhs_samples} samples over {len(space.search_param_order)} params")
    x1 = lhs_sample(space, args.lhs_samples, rng)
    x1 = dedup_rows(x1)
    f1 = evaluate_indices(x1, args.side, model, builder, args.device, args.batch_size)
    nds = NonDominatedSorting()
    front_rows = nds.do(f1, only_non_dominated_front=True)
    n_seed = max(len(front_rows), int(len(x1) * args.near_pareto_frac))
    seed = x1[np.argsort(f1[:, 0])[:n_seed]].astype(float)
    print(f"[Stage 2] unique={len(x1)} pareto={len(front_rows)} seed={len(seed)}")

    # -------------------------------------------------------------------------
    # Stage 3: NSGA-II refinement
    # -------------------------------------------------------------------------
    print(f"[Stage 3] NSGA-II pop={args.nsga_pop_size}, gen={args.nsga_generations}")
    if len(seed) < args.nsga_pop_size:
        extra = lhs_sample(space, args.nsga_pop_size - len(seed), rng).astype(float)
        seed = np.concatenate([seed, extra], axis=0)
    elif len(seed) > args.nsga_pop_size:
        choose = rng.choice(len(seed), size=args.nsga_pop_size, replace=False)
        seed = seed[choose]

    problem = HWConfigProblem(
        space=space,
        eval_fn=lambda xi: evaluate_indices(xi, args.side, model, builder, args.device, args.batch_size),
    )
    algo = NSGA2(
        pop_size=args.nsga_pop_size,
        sampling=seed,
        crossover=SBX(prob=0.9, eta=15, vtype=float),
        mutation=PM(prob=1.0 / len(space.search_param_order), eta=20, vtype=float),
        eliminate_duplicates=True,
    )
    # pymoo prints a table each generation (n_gen, n_eval, n_nds, …) when verbose=True;
    # n_nds tracks the nondominated set in the current population — i.e. Pareto-front size.
    minimize(
        problem,
        algo,
        ("n_gen", args.nsga_generations),
        seed=int(rng.integers(0, 2**31 - 1)),
        verbose=not args.quiet,
    )

    x2 = np.concatenate(problem.archive_X, axis=0) if problem.archive_X else np.empty((0, len(space.search_param_order)), dtype=np.int64)
    full_x = dedup_rows(np.concatenate([x1, x2], axis=0))
    full_f = evaluate_indices(full_x, args.side, model, builder, args.device, args.batch_size)

    # -------------------------------------------------------------------------
    # Stage 4: final Pareto extraction, sensitivity, validation flagging, output
    # -------------------------------------------------------------------------
    stage4_t0 = time.perf_counter()
    pareto_x, pareto_f = final_pareto(full_x, full_f)
    order = np.argsort(pareto_f[:, 1])
    pareto_x = pareto_x[order]
    pareto_f = pareto_f[order]

    if args.skip_sensitivity:
        sensitivities: list[dict[str, dict[str, float]]] = [{} for _ in range(len(pareto_x))]
    else:
        t_s = time.perf_counter()
        print("[Stage 4] Computing per-parameter sensitivity")
        sensitivities = compute_sensitivity(
            pareto_x, pareto_f, space, args.side, model, builder, args.device, args.batch_size
        )
        print(f"[Stage 4] Sensitivity computed in {time.perf_counter() - t_s:.2f}s")

    flags = flag_validation_candidates(pareto_f, args.validation_k)
    cfgs = space.decode_indices(pareto_x)
    print(f"[Stage 4] Total: {time.perf_counter() - stage4_t0:.2f}s")

    front: list[dict[str, Any]] = []
    for i, cfg in enumerate(cfgs):
        front.append(
            {
                "predicted_throughput_gbits_per_sec": float(-pareto_f[i, 0]),
                "predicted_cost": float(pareto_f[i, 1]),
                "validation_candidate": bool(flags[i]),
                "sensitivity": sensitivities[i] if i < len(sensitivities) else {},
                "config": cfg,
            }
        )

    out = {
        "side": args.side,
        "benchmark_count": len(benches),
        "baseline": {
            "predicted_throughput_gbits_per_sec": baseline_tp,
            "predicted_cost": baseline_cost,
            "config": baseline_cfg,
        },
        "pareto_front": front,
        "stats": {
            "lhs_samples": int(args.lhs_samples),
            "lhs_unique": int(len(x1)),
            "archive_total": int(len(full_x)),
            "n_pareto_final": int(len(front)),
            "n_validation_candidates": int(int(flags.sum())),
            "total_duration_s": round(time.perf_counter() - t_total, 2),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(front)} Pareto points -> {args.output}")


if __name__ == "__main__":
    main()
