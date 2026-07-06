"""ProtoAcc hardware cost model.

Public entry points:
    hardware_cost(cfgs, side, *, trained_model_path=None) -> np.ndarray
    StructuralCostModel  -- untrained, RTL-derived predictor
    TrainedCostModel     -- Sky130-fitted per-submodule additive predictor
    features_for_side(cfg, side) -> dict[str, float]
    features_matrix(cfgs, side) -> (X, feature_names)
"""

from .defaults import (
    DEFAULT_CONFIG_BY_SIDE,
    DES_PARAM_COLUMNS,
    PARAM_VALUES_BY_SIDE,
    SER_PARAM_COLUMNS,
    SIDE_TO_NAME,
)
from .hw_cost_model import (
    DEFAULT_KAPPA,
    StructuralCostModel,
    SubmodulePredictor,
    TrainedCostModel,
    hardware_cost,
    hardware_cost_2vec,
    use_trained_model,
)
from .structural_features import (
    DES_BUCKET_KNOBS,
    DES_BUCKET_TO_YOSYS_TOPS,
    DES_QUEUES,
    DES_SUBMODULES,
    SER_BUCKET_KNOBS,
    SER_BUCKET_TO_YOSYS_TOPS,
    SER_QUEUES,
    SER_SUBMODULES,
    bucket_knobs,
    bucket_to_yosys_tops,
    feature_names_for_side,
    features_for_side,
    features_matrix,
)

__all__ = [
    "DEFAULT_CONFIG_BY_SIDE",
    "DES_PARAM_COLUMNS",
    "DES_QUEUES",
    "DES_SUBMODULES",
    "PARAM_VALUES_BY_SIDE",
    "SER_PARAM_COLUMNS",
    "SER_QUEUES",
    "SER_SUBMODULES",
    "SIDE_TO_NAME",
    "StructuralCostModel",
    "SubmodulePredictor",
    "TrainedCostModel",
    "feature_names_for_side",
    "features_for_side",
    "features_matrix",
    "DEFAULT_KAPPA",
    "hardware_cost",
    "hardware_cost_2vec",
    "use_trained_model",
]
