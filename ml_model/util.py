"""
Utility functions for the ProtoAccel hardware configuration optimization.
"""

import numpy as np

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


def hardware_cost(cfgs: list[dict[str, int]], side: str) -> np.ndarray:
    """
    Compute hardware cost of a list of configurations.

    Args:
        cfgs: List of configurations.
        side: Side of the hardware.

    Returns:
        np.array of hardware costs.
    """
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

