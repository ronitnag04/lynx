"""
Parameter names, sweep grids, and defaults for ProtoAcc hardware configs.

Single source of truth used by both the ML model side (``ml_model/``) and
the structural-feature/hardware-cost side (``hw_cost_model/``). Keep in
sync with:
  - ``generators/protoacc/src/main/scala/configs.scala`` (Field defaults)
  - ``generators/protoacc/software/verilator-bench/gen_protoacc_sweep_configs.py``
"""

from __future__ import annotations


SIDE_TO_NAME: dict[str, str] = {"ser": "serializer", "des": "deserializer"}

DES_PARAM_COLUMNS: list[str] = [
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

SER_PARAM_COLUMNS: list[str] = [
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
        "des_top_memloader_reqs":  [16, 32, 64, 128],
        "des_cr_rocc_commands":    [2, 4, 8],
        "des_dth_l1_reqs":         [2, 4, 8, 16],
        "des_dth_fd_reqs":         [2, 4, 8, 16],
        "des_dth_fd_resps":        [2, 4, 8, 16],
        "des_fw_l1_reqs":          [2, 4, 8, 16],
        "des_ml_buf_info_q":       [8, 16, 32, 64],
        "des_ml_load_info_q":      [128, 256, 512, 1024],
    },
    "ser": {
        "ser_field_handlers":        [2, 4, 6, 8],
        "ser_cr_rocc_commands":      [2, 4, 6, 8],
        "ser_dth_hasbits_reqs":      [2, 4, 8, 16],
        "ser_dth_descriptor_reqs":   [2, 4, 8, 16],
        "ser_dth_reg_resps":         [5, 10, 20],
        "ser_dth_reqs_meta":         [2, 4, 8, 16],
        "ser_dth_fh_outputs":        [2, 4, 8, 16],
        "ser_mw_write_input":        [2, 4, 8, 16],
        "ser_mw_write_inject":       [2, 4, 8, 16],
        "ser_mw_write_ptrs":         [5, 10, 20],
    },
}

DEFAULT_CONFIG_BY_SIDE: dict[str, dict[str, int]] = {
    "des": {
        "des_top_descriptor_reqs": 4,
        "des_top_memloader_reqs":  64,
        "des_cr_rocc_commands":    2,
        "des_dth_l1_reqs":         4,
        "des_dth_fd_reqs":         4,
        "des_dth_fd_resps":        4,
        "des_fw_l1_reqs":          4,
        "des_ml_buf_info_q":       16,
        "des_ml_load_info_q":      256,
    },
    "ser": {
        "ser_field_handlers":       6,
        "ser_cr_rocc_commands":     2,
        "ser_dth_hasbits_reqs":     4,
        "ser_dth_descriptor_reqs":  4,
        "ser_dth_reg_resps":        10,
        "ser_dth_reqs_meta":        4,
        "ser_dth_fh_outputs":       4,
        "ser_mw_write_input":       4,
        "ser_mw_write_inject":      4,
        "ser_mw_write_ptrs":        10,
    },
}
