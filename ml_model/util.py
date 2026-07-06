"""
Compatibility shim: re-exports parameter tables and ``hardware_cost`` from
the ``hw_cost_model`` package so existing imports (``from util import ...``)
keep working.

New code should import directly from ``hw_cost_model``:

    from hw_cost_model import (
        DEFAULT_CONFIG_BY_SIDE, DES_PARAM_COLUMNS, PARAM_VALUES_BY_SIDE,
        SER_PARAM_COLUMNS, SIDE_TO_NAME, hardware_cost,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the parent (``lynx/``) so the ``hw_cost_model`` package resolves when
# the ml_model scripts are invoked as top-level scripts (they use ``from
# util import ...``, not the packaged form).
_LYNX_DIR = Path(__file__).resolve().parent.parent
if str(_LYNX_DIR) not in sys.path:
    sys.path.insert(0, str(_LYNX_DIR))

from hw_cost_model import (  # noqa: E402  -- import-after-sys.path is intentional
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

__all__ = [
    "DEFAULT_CONFIG_BY_SIDE",
    "DEFAULT_KAPPA",
    "DES_PARAM_COLUMNS",
    "PARAM_VALUES_BY_SIDE",
    "SER_PARAM_COLUMNS",
    "SIDE_TO_NAME",
    "hardware_cost",
    "hardware_cost_2vec",
    "use_trained_model",
]
