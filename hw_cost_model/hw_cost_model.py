"""
Hardware area cost model for ProtoAcc (Serializer + Deserializer).

Two modes:

  1. ``StructuralCostModel`` -- an untrained, RTL-derived predictor that
     scores a config by summing the structural bit counts from
     ``structural_features.py``. Used as a placeholder before any Sky130
     synthesis area data is collected. Values are in "structural bit"
     units, NOT in um^2; only relative rankings matter.

  2. ``TrainedCostModel`` -- a per-submodule additive linear (or GBM)
     model fit against Sky130 hierarchical area reports. Coefficients
     live in a joblib file loaded at construction time.

Both models expose the same ``predict(cfgs, side) -> np.ndarray`` shape
so ``optimize_hw_config.py`` and ``exhaustive_search.py`` can swap
between them via a single flag.

The ``hardware_cost`` top-level function preserves the interface of the
old ``ml_model/util.hardware_cost`` for backward-compat drop-in; it
defaults to the structural model when no trained checkpoint is passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .defaults import DEFAULT_CONFIG_BY_SIDE
from .structural_features import (
    DES_QUEUES,
    DES_SUBMODULES,
    SER_QUEUES,
    SER_SUBMODULES,
    des_features,
    features_matrix,
    features_for_side,
    ser_features,
)


# ---------------------------------------------------------------------------
# Untrained structural predictor.
# ---------------------------------------------------------------------------

# Per-submodule weight on the total-bits feature. All 1.0 by default (pure
# bit-count sum). Tuned once against a first batch of synth results by
# fitting a single scalar per submodule against the hierarchical area
# report; kept overridable so downstream users can plug in fitted values.
_DES_STRUCTURAL_WEIGHTS: dict[str, float] = {
    "descr_l1__total_bits":           1.0,
    "ml_l1__total_bits":              1.0,
    "hbw_l1__total_bits":             1.0,
    "fw_l1__total_bits":              1.0,
    # Queue bit counts (multiplicity is already folded in).
    "des_cr_rocc_commands__bits":     1.0,
    "des_dth_l1_reqs__bits":          1.0,
    "des_dth_fd_reqs__bits":          1.0,
    "des_dth_fd_resps__bits":         1.0,
    "des_fw_l1_reqs__bits":           1.0,
    "des_ml_buf_info_q__bits":        1.0,
    "des_ml_load_info_q__bits":       1.0,
    # Fixed submodule bits, attributed to their proper buckets.
    "ml__bytelane_bits":              1.0,
    "hbw__buffer_bits":               1.0,
    "fh__stack_bits":                 1.0,
}

_SER_STRUCTURAL_WEIGHTS: dict[str, float] = {
    "fh__per_handler_bits":           1.0,
    "fh__handlers_x_log2":            32.0,   # NlogN arbitration overhead
    "mfh__linear":                    64.0,   # per-handler arbitration
    "mfh__handlers_x_log2":           32.0,
    "fh_l1__handlers":                256.0,  # per-handler L1MemHelper base cost
    "varint__count":                  80.0,   # 80-bit encoder per instance
    "ser_cr_rocc_commands__bits":     1.0,
    "ser_dth_hasbits_reqs__bits":     1.0,
    "ser_dth_descriptor_reqs__bits":  1.0,
    "ser_dth_reg_resps__bits":        1.0,
    "ser_dth_reqs_meta__bits":        1.0,
    "ser_dth_fh_outputs__bits":       1.0,
    "ser_mw_write_input__bits":       1.0,
    "ser_mw_write_inject__bits":      1.0,
    "ser_mw_write_ptrs__bits":        1.0,
    "dth__stack_bits":                1.0,
    "mw__stack_bits":                 1.0,
    "mw__bytelane_bits":              1.0,
}


class StructuralCostModel:
    """Pure-RTL cost model. No fitting; sums weighted bit counts.

    Purpose: give ``optimize_hw_config.py`` a monotone, physically
    plausible cost signal *before* the first synthesis run completes.
    Once a Sky130 dataset exists, prefer ``TrainedCostModel``.
    """

    def __init__(
        self,
        des_weights: Mapping[str, float] | None = None,
        ser_weights: Mapping[str, float] | None = None,
    ) -> None:
        self._des_weights = dict(des_weights or _DES_STRUCTURAL_WEIGHTS)
        self._ser_weights = dict(ser_weights or _SER_STRUCTURAL_WEIGHTS)

    def _predict_one(self, cfg: Mapping[str, int], side: str) -> float:
        feats = features_for_side(cfg, side)
        weights = self._des_weights if side == "des" else self._ser_weights
        return float(sum(w * feats.get(k, 0.0) for k, w in weights.items()))

    def predict(
        self, cfgs: Sequence[Mapping[str, int]], side: str
    ) -> np.ndarray:
        if not cfgs:
            return np.empty(0, dtype=np.float64)
        out = np.empty(len(cfgs), dtype=np.float64)
        for i, cfg in enumerate(cfgs):
            out[i] = self._predict_one(cfg, side)
        return out


# ---------------------------------------------------------------------------
# Trained per-submodule additive model.
# ---------------------------------------------------------------------------

@dataclass
class SubmodulePredictor:
    """A fitted per-submodule, per-label predictor.

    One instance holds the coefficients for ONE (submodule, label) pair,
    where label is either ``"logic_cells"`` or ``"ram_bits"``. Two of
    these live inside ``TrainedCostModel.per_submodule[bucket]``.

    Backends:
      * ``kind == 'linear'``: dot product with ``coef`` + ``intercept``.
      * ``kind == 'gbm'``: sklearn/xgboost regressor stored under ``estimator``.
    Both are duck-typed to a ``.predict(X)`` interface.
    """
    submodule: str
    label: str
    feature_names: tuple[str, ...]
    kind: str = "linear"
    coef: np.ndarray | None = None
    intercept: float = 0.0
    estimator: object | None = None

    def predict(self, feat_dicts: Sequence[Mapping[str, float]]) -> np.ndarray:
        X = np.array(
            [[fd.get(k, 0.0) for k in self.feature_names] for fd in feat_dicts],
            dtype=np.float64,
        )
        if self.kind == "linear":
            if self.coef is None:
                raise RuntimeError(
                    f"linear submodule {self.submodule!r}/{self.label!r} has no coef"
                )
            return X @ self.coef + self.intercept
        if self.kind == "gbm":
            if self.estimator is None:
                raise RuntimeError(
                    f"gbm submodule {self.submodule!r}/{self.label!r} has no estimator"
                )
            return np.asarray(self.estimator.predict(X), dtype=np.float64)
        raise ValueError(f"Unknown SubmodulePredictor kind: {self.kind!r}")


DEFAULT_KAPPA = 0.15
"""Default gate-equivalents-per-SRAM-bit. Used to collapse the 2-vector
``(logic_cells, ram_bits)`` prediction to a single scalar cost when
downstream tools (e.g. legacy scalar Pareto search) need one number.
Chosen so a ~200-cell SRAM macro is worth roughly (200 / (D*W)) per bit;
tune per technology."""


@dataclass
class TrainedCostModel:
    """Per-submodule additive predictor for Yosys ``synth`` outputs.

    Trained from a Yosys sweep (see ``fit_from_yosys.py``). Each bucket
    has TWO ``SubmodulePredictor`` instances -- one for logic cell count,
    one for SRAM bits -- fit independently against per-bucket labels
    from ``parse_yosys_stat.py``.

    ``predict()`` returns a (N, 2) matrix of ``(logic_cells, ram_bits)``
    predictions. ``predict_scalar()`` collapses to a single scalar via
    ``logic_cells + kappa * ram_bits``.
    """
    side: str                              # "des" or "ser"
    per_submodule: dict[str, dict[str, SubmodulePredictor]] = field(default_factory=dict)
    """bucket -> {"logic_cells": SubmodulePredictor, "ram_bits": SubmodulePredictor}."""
    global_bias: dict[str, float] = field(default_factory=lambda: {"logic_cells": 0.0, "ram_bits": 0.0})
    cost_unit: str = "yosys_cell_count"

    LABELS: tuple[str, ...] = ("logic_cells", "ram_bits")

    def _labeled_predict(
        self, cfgs: Sequence[Mapping[str, int]], label: str
    ) -> np.ndarray:
        feat_dicts = [features_for_side(c, self.side) for c in cfgs]
        total = np.full(len(cfgs), self.global_bias.get(label, 0.0), dtype=np.float64)
        for bucket_preds in self.per_submodule.values():
            pred = bucket_preds.get(label)
            if pred is None:
                continue
            total += pred.predict(feat_dicts)
        return total

    def predict(
        self, cfgs: Sequence[Mapping[str, int]], side: str | None = None
    ) -> np.ndarray:
        """Return a ``(N, 2)`` array ``[[logic_cells, ram_bits], ...]``."""
        s = side if side is not None else self.side
        if s != self.side:
            raise ValueError(
                f"TrainedCostModel is for side={self.side!r}, called with side={s!r}"
            )
        if not cfgs:
            return np.empty((0, 2), dtype=np.float64)
        logic = self._labeled_predict(cfgs, "logic_cells")
        rambit = self._labeled_predict(cfgs, "ram_bits")
        return np.stack([logic, rambit], axis=1)

    def predict_scalar(
        self,
        cfgs: Sequence[Mapping[str, int]],
        *,
        kappa: float = DEFAULT_KAPPA,
        side: str | None = None,
    ) -> np.ndarray:
        """Return a length-N scalar cost = ``logic_cells + kappa * ram_bits``."""
        two_vec = self.predict(cfgs, side=side)
        if two_vec.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        return two_vec[:, 0] + kappa * two_vec[:, 1]

    def predict_by_submodule(
        self, cfgs: Sequence[Mapping[str, int]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return a per-submodule, per-label breakdown of the prediction.
        Handy for auditing which knob dominates a Pareto-optimal config."""
        feat_dicts = [features_for_side(c, self.side) for c in cfgs]
        out: dict[str, dict[str, np.ndarray]] = {}
        for bucket, preds in self.per_submodule.items():
            out[bucket] = {label: p.predict(feat_dicts) for label, p in preds.items()}
        return out

    # --- persistence -------------------------------------------------------

    def save(self, path: Path | str) -> None:
        import joblib
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: Path | str) -> "TrainedCostModel":
        import joblib
        m = joblib.load(Path(path))
        if not isinstance(m, TrainedCostModel):
            raise TypeError(f"{path}: expected TrainedCostModel, got {type(m).__name__}")
        return m


# ---------------------------------------------------------------------------
# Public interface: matches the old ml_model/util.hardware_cost signature.
# ---------------------------------------------------------------------------

_STRUCTURAL_SINGLETON: StructuralCostModel | None = None
_TRAINED_MODELS: dict[str, TrainedCostModel] = {}


def _structural() -> StructuralCostModel:
    global _STRUCTURAL_SINGLETON
    if _STRUCTURAL_SINGLETON is None:
        _STRUCTURAL_SINGLETON = StructuralCostModel()
    return _STRUCTURAL_SINGLETON


def use_trained_model(side: str, path: Path | str) -> None:
    """Register a trained ``TrainedCostModel`` for the given side.

    Call once at CLI startup; subsequent ``hardware_cost(..., side)`` calls
    (without an explicit ``trained_model_path``) will use it instead of
    the structural fallback. Enables threading a trained model through
    downstream scripts without adding a param to every callsite.
    """
    model = TrainedCostModel.load(path)
    if model.side != side:
        raise ValueError(
            f"{path}: model.side={model.side!r} but registered for side={side!r}"
        )
    _TRAINED_MODELS[side] = model


def hardware_cost_2vec(
    cfgs: Sequence[Mapping[str, int]],
    side: str,
    *,
    trained_model_path: Path | str | None = None,
) -> np.ndarray:
    """Predict ``(logic_cells, ram_bits)`` for a batch of configs.

    Returns a ``(N, 2)`` array. The first column is combinational + flop
    logic cells (excludes SRAM storage); the second column is the total
    bits of inferred SRAM (from the ``ram_<D>x<W>`` firtool memory
    modules). Structural fallback synthesises both columns from the
    RTL-derived bit counts so the shape is stable when no trained model
    exists yet.

    Resolution order:
      1. ``trained_model_path`` arg, if given -- loads a fresh model.
      2. Any model previously registered via ``use_trained_model(side, ...)``.
      3. The structural, bit-count-based fallback.
    """
    if trained_model_path is not None:
        model = TrainedCostModel.load(trained_model_path)
        return model.predict(list(cfgs), side)
    if side in _TRAINED_MODELS:
        return _TRAINED_MODELS[side].predict(list(cfgs), side)
    # Structural fallback: split RTL bits into "logic-like" and "ram-like"
    # halves. All queue bit counts are treated as ram-like (they map to
    # firtool ram_<D>x<W> modules), everything else is logic-like.
    struct = _structural()
    total = struct.predict(list(cfgs), side)
    # Approximate: attribute queue+L1MemHelper response-vec bits to ram
    # side; everything else is logic. Deliberately coarse -- structural
    # fallback only exists for smoke testing before yosys data lands.
    ram = np.zeros_like(total)
    for i, cfg in enumerate(cfgs):
        feats = features_for_side(cfg, side)
        ram_bits = 0.0
        for k, v in feats.items():
            # Any feature suffixed __bits from a queue spec is memory-like.
            if k.endswith("__bits") and "__" in k and not k.startswith(("mfh__", "fh__stack", "hbw__buffer", "ml__bytelane", "dth__stack", "mw__stack", "mw__bytelane")):
                ram_bits += v
        ram[i] = ram_bits
    logic = np.maximum(total - ram, 0.0)
    return np.stack([logic, ram], axis=1)


def hardware_cost(
    cfgs: Sequence[Mapping[str, int]],
    side: str,
    *,
    trained_model_path: Path | str | None = None,
    kappa: float = DEFAULT_KAPPA,
) -> np.ndarray:
    """Scalar hardware cost: ``logic_cells + kappa * ram_bits``.

    Drop-in replacement for the legacy ``ml_model/util.hardware_cost``
    scalar interface. Use ``hardware_cost_2vec`` when you want the raw
    two-vector for multi-objective Pareto search.
    """
    two_vec = hardware_cost_2vec(cfgs, side, trained_model_path=trained_model_path)
    if two_vec.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return two_vec[:, 0] + kappa * two_vec[:, 1]


__all__ = [
    "DEFAULT_KAPPA",
    "StructuralCostModel",
    "SubmodulePredictor",
    "TrainedCostModel",
    "hardware_cost",
    "hardware_cost_2vec",
    "use_trained_model",
]
