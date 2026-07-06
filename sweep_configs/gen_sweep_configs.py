#!/usr/bin/env python3
"""
Generate chipyard.ProtoAccelSweepConfigs.scala (and side-specific companions)
from parameter domains split between deserializer (``des_*``) and serializer
(``ser_*``).

Parameter tables come from ``hw_cost_model.defaults`` so this script and
the cost-model fitter share a single source of truth.

Sweep types (``-t`` / ``--sweep-type``):
- random: sample N distinct combinations over the **active** parameter axes only
  (mixed-radix index decode). The inactive side stays at defaults.
- ofat: one-factor-at-a-time over active axes only; inactive side at defaults.
- default: a single full row matching merged defaults.
- tweak: one config per **active** parameter; first non-default value in that
  axis's value list.
- synth-training: emits the full cost-model training set in one shot --
  defaults + all-min + all-max + OFAT + Latin-hypercube (main seed) +
  2-3-axis stress corners + a held-out LHS (different seed). Class-name
  prefix is ``ProtoAccel{Des,Ser}Synth`` (distinct from the verilator
  sweep's ``ProtoAccel{Des,Ser}Sweep``) so both coexist in Chipyard.
  Adds a ``sample_group`` column to the CSV so the fitter can use the
  ``holdout`` slice for validation.

Emit modes (``--emit``):
- both (default): ``ProtoAccelDesSweepConfigs`` + ``ProtoAccelSerSweepConfigs``
  with descriptive classes ``ProtoAccelDesSweep{ACRONYM+VALUE}*Config`` /
  ``ProtoAccelSerSweep{ACRONYM+VALUE}*Config``. Each class name encodes only
  the **active** side's parameter values (inactive side stays at defaults) so
  CSV rows from different sweeps can be deduped/merged on ``config_name``.
  Serializer parameters vary only in the ser block; deserializer only in the
  des block. ``-n`` applies **per side** for random sweeps.
- des / ser: emit only that side's object + classes.
- joint: legacy single object ``ProtoAccelSweepConfigs`` + ``ProtoAccelSweep*``
  classes encoding **all** params in the name, varying all axes together.

Class-name encoding uses the acronym tables (``DES_ACRONYM_LABEL_BY_KEY`` /
``SER_ACRONYM_LABEL_BY_KEY``). Example: a des-side sample with
``des_cr_rocc_commands=4``, ``des_dth_fd_reqs=4``, ... ends up as
``ProtoAccelDesSweepDC4DDFQ4DDFP2DDL4DFL4DMB16DML256DTD4DTM64Config``.
Debug (``--debug``) variants get a ``Debug`` infix after the side tag, e.g.
``ProtoAccelDesSweepDebugDC4...Config``. Colliding combos (common under
``-t ofat`` where every axis re-emits the defaults row) are deduplicated.

Mixed-radix axis order within a side is sorted keys for that side
(``des_*`` then ``ser_*`` when combined for fragment emission order).

A separate mode ``--write-default`` bypasses sweep generation entirely and
emits ``ProtoAccelDefaultConfigs.scala`` with a single ``ProtoAccelDefaultConfig``
class holding every ``des_*`` and ``ser_*`` parameter at its merged default
(plus ``ProtoAccelDefaultDebugConfig`` under ``--debug``). No acronym suffix
is appended to the class name, so this config serves as a stable baseline.

Examples:
  python3 gen_protoacc_sweep_configs.py -t random -n 32 -s 42
  python3 gen_protoacc_sweep_configs.py --emit des -t ofat -o /tmp/out.scala
  python3 gen_protoacc_sweep_configs.py --emit joint -t random -n 16 -s 0
  python3 gen_protoacc_sweep_configs.py --write-default
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
# Add ``lynx/`` to sys.path so ``import hw_cost_model.defaults`` works when
# invoked as a top-level script. When run as ``python -m
# lynx.sweep_configs.gen_sweep_configs`` this is a no-op.
_LYNX_DIR = SCRIPT_DIR.parent
if str(_LYNX_DIR) not in sys.path:
    sys.path.insert(0, str(_LYNX_DIR))

from hw_cost_model.defaults import (  # noqa: E402 -- import after sys.path setup
    DEFAULT_CONFIG_BY_SIDE as _CANON_DEFAULTS,
    DES_PARAM_COLUMNS as _CANON_DES_COLS,
    PARAM_VALUES_BY_SIDE as _CANON_PARAM_VALUES,
    SER_PARAM_COLUMNS as _CANON_SER_COLS,
)


DEFAULT_OUT = (
    SCRIPT_DIR.parent.parent.parent.parent
    / "chipyard"
    / "src"
    / "main"
    / "scala"
    / "config"
    / "ProtoAccelSweepConfigs.scala"
)
DEFAULT_CSV_OUT = SCRIPT_DIR / "sweep_configs.csv"
DEFAULT_SCALA_CONFIG_DIR = DEFAULT_OUT.parent

# --- Parameter tables: pulled from hw_cost_model.defaults (single source of truth).
DES_PARAM_VALUES: Dict[str, List[int]] = dict(_CANON_PARAM_VALUES["des"])
DES_DEFAULT_PARAM_VALUES: Dict[str, int] = dict(_CANON_DEFAULTS["des"])
SER_PARAM_VALUES: Dict[str, List[int]] = dict(_CANON_PARAM_VALUES["ser"])
SER_DEFAULT_PARAM_VALUES: Dict[str, int] = dict(_CANON_DEFAULTS["ser"])

# --- Scala-emission-specific tables (labels, acronyms, WithClass names).

DES_WITH_CLASS_BY_KEY: Dict[str, str] = {
    "des_top_descriptor_reqs": "WithProtoAccelDesDescrOutstanding",
    "des_top_memloader_reqs": "WithProtoAccelDesMemloaderOutstanding",
    "des_cr_rocc_commands": "WithProtoAccelDesCrRoccCommands",
    "des_dth_l1_reqs": "WithProtoAccelDesDthL1Reqs",
    "des_dth_fd_reqs": "WithProtoAccelDesDthFdReqs",
    "des_dth_fd_resps": "WithProtoAccelDesDthFdResps",
    "des_fw_l1_reqs": "WithProtoAccelDesFwL1Reqs",
    "des_ml_buf_info_q": "WithProtoAccelDesMlBufInfoQ",
    "des_ml_load_info_q": "WithProtoAccelDesMlLoadInfoQ",
}

DES_SHORT_LABEL_BY_KEY: Dict[str, str] = {
    "des_top_descriptor_reqs": "DesTopDescr",
    "des_top_memloader_reqs": "DesTopMemloader",
    "des_cr_rocc_commands": "DesCrRocc",
    "des_dth_l1_reqs": "DesDthL1",
    "des_dth_fd_reqs": "DesDthFdReqs",
    "des_dth_fd_resps": "DesDthFdResps",
    "des_fw_l1_reqs": "DesFwL1",
    "des_ml_buf_info_q": "DesMlBufInfo",
    "des_ml_load_info_q": "DesMlLoadInfo",
}

DES_ACRONYM_LABEL_BY_KEY: Dict[str, str] = {
    "des_top_descriptor_reqs": "DTD",
    "des_top_memloader_reqs": "DTM",
    "des_cr_rocc_commands": "DC",
    "des_dth_l1_reqs": "DDL",
    "des_dth_fd_reqs": "DDFQ",
    "des_dth_fd_resps": "DDFP",
    "des_fw_l1_reqs": "DFL",
    "des_ml_buf_info_q": "DMB",
    "des_ml_load_info_q": "DML",
}

SER_WITH_CLASS_BY_KEY: Dict[str, str] = {
    "ser_field_handlers": "WithProtoAccelSerFieldHandlers",
    "ser_cr_rocc_commands": "WithProtoAccelSerCrRoccCommands",
    "ser_dth_hasbits_reqs": "WithProtoAccelSerDthHasbitsReqs",
    "ser_dth_descriptor_reqs": "WithProtoAccelSerDthDescriptorReqs",
    "ser_dth_reg_resps": "WithProtoAccelSerDthRegResps",
    "ser_dth_reqs_meta": "WithProtoAccelSerDthReqsMeta",
    "ser_dth_fh_outputs": "WithProtoAccelSerDthFhOutputs",
    "ser_mw_write_input": "WithProtoAccelSerMwWriteInput",
    "ser_mw_write_inject": "WithProtoAccelSerMwWriteInject",
    "ser_mw_write_ptrs": "WithProtoAccelSerMwWritePtrs",
}

SER_SHORT_LABEL_BY_KEY: Dict[str, str] = {
    "ser_field_handlers": "SerFieldHandlers",
    "ser_cr_rocc_commands": "SerCrRocc",
    "ser_dth_hasbits_reqs": "SerDthHasbits",
    "ser_dth_descriptor_reqs": "SerDthDescr",
    "ser_dth_reg_resps": "SerDthRegResps",
    "ser_dth_reqs_meta": "SerDthReqsMeta",
    "ser_dth_fh_outputs": "SerDthFhOutputs",
    "ser_mw_write_input": "SerMwWriteInput",
    "ser_mw_write_inject": "SerMwWriteInject",
    "ser_mw_write_ptrs": "SerMwWritePtrs",
}

SER_ACRONYM_LABEL_BY_KEY: Dict[str, str] = {
    "ser_field_handlers": "SF",
    "ser_cr_rocc_commands": "SC",
    "ser_dth_hasbits_reqs": "SDH",
    "ser_dth_descriptor_reqs": "SDD",
    "ser_dth_reg_resps": "SDQ",
    "ser_dth_reqs_meta": "SDM",
    "ser_dth_fh_outputs": "SDF",
    "ser_mw_write_input": "SMI",
    "ser_mw_write_inject": "SMJ",
    "ser_mw_write_ptrs": "SMP",
}

# Merged tables (must match generators/protoacc/src/main/scala/util.scala defaults).
PARAM_VALUES: Dict[str, List[int]] = {**DES_PARAM_VALUES, **SER_PARAM_VALUES}
DEFAULT_PARAM_VALUES: Dict[str, int] = {**DES_DEFAULT_PARAM_VALUES, **SER_DEFAULT_PARAM_VALUES}
WITH_CLASS_BY_KEY: Dict[str, str] = {**DES_WITH_CLASS_BY_KEY, **SER_WITH_CLASS_BY_KEY}
SHORT_LABEL_BY_KEY: Dict[str, str] = {**DES_SHORT_LABEL_BY_KEY, **SER_SHORT_LABEL_BY_KEY}
ACRONYM_LABEL_BY_KEY: Dict[str, str] = {**DES_ACRONYM_LABEL_BY_KEY, **SER_ACRONYM_LABEL_BY_KEY}

DES_KEYS: Tuple[str, ...] = tuple(sorted(DES_PARAM_VALUES.keys()))
SER_KEYS: Tuple[str, ...] = tuple(sorted(SER_PARAM_VALUES.keys()))
FULL_PARAM_KEYS: Tuple[str, ...] = tuple(sorted(PARAM_VALUES.keys()))

# Canonical column order used by run_sweep.sh's sweep.csv: des_* (dict order
# as listed in DES_PARAM_VALUES) then ser_*. Kept separate from DES_KEYS /
# SER_KEYS (which are sorted) so CSV joins on config_name remain compatible.
CSV_PARAM_KEYS: Tuple[str, ...] = (
    tuple(DES_PARAM_VALUES.keys()) + tuple(SER_PARAM_VALUES.keys())
)
# ``sample_group`` is populated only by ``-t synth-training``; other sweep
# types emit an empty string so downstream schemas stay stable.
CSV_COLUMNS: Tuple[str, ...] = ("config_name", "side", "sample_group") + CSV_PARAM_KEYS

_MAX_INDICES_MATERIALIZE = 10**9

# Same stack as HyperscaleConfigs.scala `ProtoAccelRocketBaseConfig` (single reference).
BASE_FRAGMENTS = """  new ProtoAccelRocketBaseConfig)"""

DEBUG_PREFIX = "  new protoacc.WithProtoAccelPrintf ++\n"

# Second side random sweep seed offset (avoid duplicating the same index draw).
_SER_SEED_OFFSET = 1_000_003


def _validate_tables() -> None:
    des_k = set(DES_PARAM_VALUES)
    ser_k = set(SER_PARAM_VALUES)
    if des_k & ser_k:
        raise SystemExit(f"DES and SER parameter keys overlap: {des_k & ser_k!r}")
    if set(PARAM_VALUES) != des_k | ser_k:
        raise SystemExit("PARAM_VALUES must be the disjoint union of DES and SER keys")

    for side, keys, defaults, pvals in (
        ("DES", des_k, DES_DEFAULT_PARAM_VALUES, DES_PARAM_VALUES),
        ("SER", ser_k, SER_DEFAULT_PARAM_VALUES, SER_PARAM_VALUES),
    ):
        if set(defaults) != keys:
            raise SystemExit(f"{side} defaults keys {set(defaults)!r} != param keys {keys!r}")
        for k in keys:
            if defaults[k] not in pvals[k]:
                raise SystemExit(
                    f"{side} default for {k!r} is {defaults[k]!r} but must appear in sweep values"
                )

    merged_defaults = {**DES_DEFAULT_PARAM_VALUES, **SER_DEFAULT_PARAM_VALUES}
    if merged_defaults != DEFAULT_PARAM_VALUES:
        raise SystemExit("DEFAULT_PARAM_VALUES must merge DES and SER defaults")

    for label, d in (
        ("WITH_CLASS_BY_KEY", WITH_CLASS_BY_KEY),
        ("SHORT_LABEL_BY_KEY", SHORT_LABEL_BY_KEY),
        ("ACRONYM_LABEL_BY_KEY", ACRONYM_LABEL_BY_KEY),
    ):
        if set(d) != set(PARAM_VALUES):
            raise SystemExit(f"{label} keys must match PARAM_VALUES")

    acronyms = list(ACRONYM_LABEL_BY_KEY.values())
    if len(set(acronyms)) != len(acronyms):
        dupes = [a for a in acronyms if acronyms.count(a) > 1]
        raise SystemExit(f"ACRONYM_LABEL_BY_KEY has duplicate acronyms: {sorted(set(dupes))!r}")


def first_non_default_value(
    key: str, *, defaults: Mapping[str, int], param_values: Mapping[str, Sequence[int]]
) -> int | None:
    d = defaults[key]
    for v in param_values[key]:
        if v != d:
            return v
    return None


def total_combinations(
    param_values: Mapping[str, Sequence[int]], keys: Sequence[str]
) -> int:
    n = 1
    for k in keys:
        n *= len(param_values[k])
    return n


def _compute_strides(
    param_values: Mapping[str, Sequence[int]], keys: Sequence[str]
) -> Tuple[List[int], List[int]]:
    sizes = [len(param_values[k]) for k in keys]
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return sizes, strides


def index_to_combination(
    index: int,
    param_values: Mapping[str, Sequence[int]],
    keys: Sequence[str],
    *,
    sizes: List[int] | None = None,
    strides: List[int] | None = None,
) -> Dict[str, int]:
    """Decode linear index into a partial parameter dict (mixed-radix over ``keys``)."""
    value_lists = [param_values[k] for k in keys]
    if sizes is None or strides is None:
        sizes, strides = _compute_strides(param_values, keys)
    digits = [(index // strides[i]) % sizes[i] for i in range(len(sizes))]
    return dict(zip(keys, [value_lists[i][d] for i, d in enumerate(digits)]))


def sample_random_indices(total: int, n: int, offset: int = 0) -> List[int]:
    """
    Return ``n`` distinct random indices in ``[0, total)`` without materializing
    ``range(total)``. When ``offset > 0``, draws ``offset + n`` distinct indices
    using the current RNG state and returns only the tail (the last ``n``). That
    way, two machines with the same seed but different offsets ``(0, n)`` and
    ``(n, n)`` produce disjoint samples that together equal the ``2n``-sample
    from offset ``0``.
    """
    draw = offset + n
    if draw > total:
        raise SystemExit(
            f"offset({offset}) + num_configs({n}) = {draw} exceeds total "
            f"combinations ({total})."
        )
    chosen: List[int] = []
    seen: set[int] = set()
    while len(chosen) < draw:
        idx = random.randrange(total)
        if idx not in seen:
            seen.add(idx)
            chosen.append(idx)
    return chosen[offset:]


def merge_full_combo(partial: Mapping[str, int]) -> Dict[str, int]:
    """Fill a partial sweep (one side's axes) with defaults for the full config."""
    row = dict(DEFAULT_PARAM_VALUES)
    row.update(partial)
    return row


def _combo_comment(combo: Mapping[str, int]) -> str:
    parts = [f"{SHORT_LABEL_BY_KEY[k]}={combo[k]}" for k in FULL_PARAM_KEYS]
    return ", ".join(parts)


def _scala_fragments(combo: Mapping[str, int]) -> str:
    lines = [
        f"  new protoacc.{WITH_CLASS_BY_KEY[k]}({combo[k]}) ++" for k in FULL_PARAM_KEYS
    ]
    return "\n".join(lines)


def build_combinations(
    sweep_type: str,
    num_configs: int,
    seed: int,
    active_keys: Sequence[str],
    offset: int = 0,
) -> Tuple[List[Dict[str, int]], str]:
    """Return (list of **full** combo dicts, human-readable summary line).

    For random sweeps, ``offset`` skips the first ``offset`` samples from the
    seeded draw, letting one machine emit indices [0, N) and another [N, 2N)
    with guaranteed non-overlap under the same seed.
    """
    slice_pv = {k: PARAM_VALUES[k] for k in active_keys}
    side = "joint" if len(active_keys) == len(FULL_PARAM_KEYS) else (
        "deserializer" if set(active_keys) <= set(DES_KEYS) else (
            "serializer" if set(active_keys) <= set(SER_KEYS) else "custom"
        )
    )

    if sweep_type == "random":
        random.seed(seed)
        total = total_combinations(slice_pv, active_keys)
        if offset < 0:
            raise SystemExit(f"--offset must be >= 0 (got {offset}).")
        if offset + num_configs > total:
            raise SystemExit(
                f"[{side}] offset({offset}) + num_configs({num_configs}) = "
                f"{offset + num_configs} exceeds total combinations ({total})."
            )
        n = num_configs
        summary = (
            f"[{side}] Total combinations on active axes: {total:.2e}. "
            f"Sampling {n} random indices (seed={seed}, offset={offset})."
        )
        if total > _MAX_INDICES_MATERIALIZE and offset + n >= total:
            raise SystemExit(
                f"Active space has {total} combinations (> {_MAX_INDICES_MATERIALIZE}); "
                "cannot enumerate. Use offset + n < total to sample."
            )
        random_indices = sample_random_indices(total, n, offset=offset)
        sizes, strides = _compute_strides(slice_pv, active_keys)
        combinations = [
            merge_full_combo(
                index_to_combination(
                    k, slice_pv, active_keys, sizes=sizes, strides=strides
                )
            )
            for k in random_indices
        ]
        return combinations, summary

    if sweep_type == "ofat":
        combinations = []
        for key in active_keys:
            for value in PARAM_VALUES[key]:
                combo = dict(DEFAULT_PARAM_VALUES)
                combo[key] = value
                combinations.append(combo)
        summary = (
            f"[{side}] OFAT sweep over {len(active_keys)} parameters, "
            f"{len(combinations)} total rows (--num-configs and --seed ignored)."
        )
        return combinations, summary

    if sweep_type == "default":
        combinations = [dict(DEFAULT_PARAM_VALUES)]
        summary = (
            f"[{side}] Single default baseline (--num-configs and --seed ignored)."
        )
        return combinations, summary

    if sweep_type == "tweak":
        combinations = []
        skipped: List[str] = []
        defaults = {k: DEFAULT_PARAM_VALUES[k] for k in active_keys}
        for key in active_keys:
            alt = first_non_default_value(
                key, defaults=defaults, param_values=slice_pv
            )
            if alt is None:
                skipped.append(key)
                continue
            row = dict(DEFAULT_PARAM_VALUES)
            row[key] = alt
            combinations.append(row)
        summary = (
            f"[{side}] Tweak sweep: {len(combinations)} rows. "
            "--num-configs and --seed ignored."
        )
        if skipped:
            summary += (
                f" Skipped {len(skipped)} parameter(s) with no value != default: "
                + ", ".join(skipped)
            )
        return combinations, summary

    raise SystemExit(f"Unknown sweep type: {sweep_type}")


# ---------------------------------------------------------------------------
# synth-training strategy: full cost-model training set in one call.
#
# Groups (in priority order for dedup -- earlier groups win collisions):
#   defaults      : 1 row  (all-default)
#   corner_min    : 1 row  (every active knob at its minimum)
#   corner_max    : 1 row  (every active knob at its maximum)
#   ofat          : |active_keys| * <values-per-key> rows
#                   (each active knob swept over its grid, others at defaults)
#   lhs           : args.n_lhs rows (Latin-hypercube sample, --seed)
#   stress        : args.n_stress rows (2-3 axes randomly forced to their max,
#                   others at defaults; seed = --seed + STRESS_SEED_OFFSET)
#   holdout       : args.n_holdout rows (LHS with --holdout-seed)
# ---------------------------------------------------------------------------

STRESS_SEED_OFFSET = 7_654_321
HOLDOUT_SEED_OFFSET = 2_718_281

# Priority order for cross-group dedup on the (config_name) key.
SYNTH_TRAINING_GROUP_PRIORITY: Tuple[str, ...] = (
    "defaults",
    "corner_min",
    "corner_max",
    "ofat",
    "lhs",
    "stress",
    "holdout",
)


def _latin_hypercube_indices(
    n_samples: int,
    active_keys: Sequence[str],
    slice_pv: Mapping[str, Sequence[int]],
    seed: int,
) -> List[Dict[str, int]]:
    """Latin-hypercube sample over the active grid.

    Each column of the LHS matrix is a random permutation of
    ``[0, n_samples)`` scaled to that axis's number of grid values. Uses
    ``random.Random`` (seeded) so we avoid a hard scipy dependency; the
    quality difference vs. ``scipy.stats.qmc.LatinHypercube`` is
    negligible for a few hundred points across ~10 axes.
    """
    rng = random.Random(seed)
    n_axes = len(active_keys)
    axis_sizes = [len(slice_pv[k]) for k in active_keys]
    columns: List[List[int]] = []
    for size in axis_sizes:
        # Draw n_samples index values, evenly stratified across `size` bins.
        base = [i * size // n_samples for i in range(n_samples)]
        rng.shuffle(base)
        columns.append(base)
    out: List[Dict[str, int]] = []
    for i in range(n_samples):
        row: Dict[str, int] = {}
        for j, k in enumerate(active_keys):
            row[k] = slice_pv[k][columns[j][i]]
        out.append(row)
    return out


def _stress_rows(
    n_samples: int,
    active_keys: Sequence[str],
    slice_pv: Mapping[str, Sequence[int]],
    seed: int,
    *,
    axes_min: int = 2,
    axes_max: int = 3,
) -> List[Dict[str, int]]:
    """Force ``axes_min``..``axes_max`` random axes to their maximum
    simultaneously (others at defaults). Deterministic under ``seed``.
    """
    rng = random.Random(seed)
    out: List[Dict[str, int]] = []
    keys = list(active_keys)
    for _ in range(n_samples):
        row = dict(DEFAULT_PARAM_VALUES)
        n_extreme = rng.randint(axes_min, min(axes_max, len(keys)))
        picked = rng.sample(keys, n_extreme)
        for k in picked:
            row[k] = max(slice_pv[k])
        out.append(row)
    return out


def build_synth_training_combinations(
    active_keys: Sequence[str],
    *,
    n_lhs: int,
    n_stress: int,
    n_holdout: int,
    seed: int,
) -> Tuple[List[Tuple[Dict[str, int], str]], str]:
    """Return ``[(full_combo_dict, sample_group), ...]`` for one side.

    The returned combos each have every knob filled in (inactive-side
    knobs come from ``DEFAULT_PARAM_VALUES``). The caller deduplicates by
    class name and, on collision, keeps the row with the earlier group
    in ``SYNTH_TRAINING_GROUP_PRIORITY``.
    """
    side = ("deserializer" if set(active_keys) <= set(DES_KEYS)
            else "serializer" if set(active_keys) <= set(SER_KEYS)
            else "joint")
    slice_pv = {k: PARAM_VALUES[k] for k in active_keys}

    tagged: List[Tuple[Dict[str, int], str]] = []

    # defaults
    tagged.append((dict(DEFAULT_PARAM_VALUES), "defaults"))

    # corners
    row_min = dict(DEFAULT_PARAM_VALUES)
    for k in active_keys:
        row_min[k] = min(slice_pv[k])
    tagged.append((row_min, "corner_min"))
    row_max = dict(DEFAULT_PARAM_VALUES)
    for k in active_keys:
        row_max[k] = max(slice_pv[k])
    tagged.append((row_max, "corner_max"))

    # ofat
    for key in active_keys:
        for value in slice_pv[key]:
            row = dict(DEFAULT_PARAM_VALUES)
            row[key] = value
            tagged.append((row, "ofat"))

    # LHS
    for row in _latin_hypercube_indices(n_lhs, active_keys, slice_pv, seed):
        full = dict(DEFAULT_PARAM_VALUES)
        full.update(row)
        tagged.append((full, "lhs"))

    # Stress corners
    for row in _stress_rows(
        n_stress, active_keys, slice_pv, seed + STRESS_SEED_OFFSET
    ):
        tagged.append((row, "stress"))

    # Holdout LHS with a different seed.
    for row in _latin_hypercube_indices(
        n_holdout, active_keys, slice_pv, seed + HOLDOUT_SEED_OFFSET
    ):
        full = dict(DEFAULT_PARAM_VALUES)
        full.update(row)
        tagged.append((full, "holdout"))

    n_by_group: Dict[str, int] = {}
    for _, g in tagged:
        n_by_group[g] = n_by_group.get(g, 0) + 1
    summary = (
        f"[{side}] synth-training: "
        + ", ".join(f"{g}={n_by_group[g]}" for g in SYNTH_TRAINING_GROUP_PRIORITY if g in n_by_group)
        + f" (pre-dedup total={len(tagged)}, seed={seed})"
    )
    return tagged, summary


def _encode_combo_name(
    combo: Mapping[str, int], name_keys: Sequence[str]
) -> str:
    """Return the acronym+value encoding for `combo` over `name_keys`."""
    return "".join(f"{ACRONYM_LABEL_BY_KEY[k]}{combo[k]}" for k in name_keys)


def _class_basename(
    *,
    class_prefix: str,
    class_debug_prefix: str,
    name_keys: Sequence[str],
    combo: Mapping[str, int],
    debug: bool,
) -> str:
    prefix = class_debug_prefix if debug else class_prefix
    return f"{prefix}{_encode_combo_name(combo, name_keys)}Config"


def _dedupe_by_name(
    combinations: Sequence[Mapping[str, int]], name_keys: Sequence[str]
) -> List[Dict[str, int]]:
    """Drop combos whose name-key projection was already seen earlier."""
    seen: set[Tuple[int, ...]] = set()
    out: List[Dict[str, int]] = []
    for combo in combinations:
        sig = tuple(combo[k] for k in name_keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(dict(combo))
    return out


def _dedupe_tagged_by_name(
    tagged: Sequence[Tuple[Mapping[str, int], str]],
    name_keys: Sequence[str],
    group_priority: Sequence[str],
) -> List[Tuple[Dict[str, int], str]]:
    """Dedup by class-name signature; on collision keep the row whose
    ``sample_group`` comes earlier in ``group_priority``.

    Preserves the *first* appearance's ordering of unique signatures --
    if a later row has a higher-priority (=earlier) group, it *replaces*
    the earlier one but keeps the earlier position in the output.
    """
    pri = {g: i for i, g in enumerate(group_priority)}
    sig_to_pos: Dict[Tuple[int, ...], int] = {}
    out: List[Tuple[Dict[str, int], str]] = []
    for combo, group in tagged:
        sig = tuple(combo[k] for k in name_keys)
        if sig not in sig_to_pos:
            sig_to_pos[sig] = len(out)
            out.append((dict(combo), group))
            continue
        # Collision: keep whichever group has higher priority (lower idx).
        pos = sig_to_pos[sig]
        existing_group = out[pos][1]
        if pri.get(group, 10**9) < pri.get(existing_group, 10**9):
            out[pos] = (dict(combo), group)
    return out


def _emit_side_block(
    *,
    lines: List[str],
    object_name: str,
    class_prefix: str,
    class_debug_prefix: str,
    name_keys: Sequence[str],
    combinations: Sequence[Mapping[str, int]],
    sweep_type: str,
    seed: int,
    num_configs_requested: int,
    generate_debug: bool,
) -> int:
    """Emit one ``object`` + Scala classes. Returns the emitted row count (post-dedupe)."""
    combos = _dedupe_by_name(combinations, name_keys)
    n = len(combos)

    lines.append(f"object {object_name} {{")
    lines.append("")
    lines.append(f'  val generationSweepType: String = "{sweep_type}"')
    lines.append(f"  val generationSeed: Long = {seed}L")
    lines.append(f"  val generationNumConfigsRequested: Int = {num_configs_requested}")
    lines.append(f"  val generationNumConfigsEmitted: Int = {n}")
    lines.append("")
    lines.append("  /** CONFIG= names for non-debug generated classes (basename only). */")
    lines.append("  val normalSweepConfigNames: Seq[String] = Seq(")
    for combo in combos:
        name = _class_basename(
            class_prefix=class_prefix,
            class_debug_prefix=class_debug_prefix,
            name_keys=name_keys,
            combo=combo,
            debug=False,
        )
        lines.append(f'    "{name}",')
    lines.append("  )")

    if generate_debug:
        lines += [
            "",
            "  /** CONFIG= names for debug (printf) variants. */",
            "  val debugSweepConfigNames: Seq[String] = Seq(",
        ]
        for combo in combos:
            dname = _class_basename(
                class_prefix=class_prefix,
                class_debug_prefix=class_debug_prefix,
                name_keys=name_keys,
                combo=combo,
                debug=True,
            )
            lines.append(f'    "{dname}",')
        lines.append("  )")
    else:
        lines += [
            "",
            "  /** Populate with ``--debug``. */",
            "  val debugSweepConfigNames: Seq[String] = Seq.empty[String]",
        ]

    lines.append("}")
    lines.append("")

    for i, combo in enumerate(combos):
        cmt = _combo_comment(combo)
        name = _class_basename(
            class_prefix=class_prefix,
            class_debug_prefix=class_debug_prefix,
            name_keys=name_keys,
            combo=combo,
            debug=False,
        )
        lines.append(f"/** Sweep row {i + 1}/{n} ({sweep_type}): {cmt} */")
        lines.append(f"class {name} extends Config(")
        lines.append(_scala_fragments(combo))
        lines.append(BASE_FRAGMENTS)
        lines.append("")

        if generate_debug:
            dname = _class_basename(
                class_prefix=class_prefix,
                class_debug_prefix=class_debug_prefix,
                name_keys=name_keys,
                combo=combo,
                debug=True,
            )
            lines.append(f"/** Debug printf variant of `{name}`. */")
            lines.append(f"class {dname} extends Config(")
            lines.append(DEBUG_PREFIX.rstrip("\n"))
            lines.append(_scala_fragments(combo))
            lines.append(BASE_FRAGMENTS)
            lines.append("")

    return n


def _csv_side_tag(name_keys: Sequence[str]) -> str:
    if len(name_keys) == len(FULL_PARAM_KEYS):
        return "joint"
    if set(name_keys) <= set(DES_KEYS):
        return "des"
    if set(name_keys) <= set(SER_KEYS):
        return "ser"
    return "custom"


def _combo_csv_row(
    *,
    class_prefix: str,
    class_debug_prefix: str,
    name_keys: Sequence[str],
    combo: Mapping[str, int],
    side_tag: str,
    sample_group: str = "",
) -> Dict[str, object]:
    name = _class_basename(
        class_prefix=class_prefix,
        class_debug_prefix=class_debug_prefix,
        name_keys=name_keys,
        combo=combo,
        debug=False,
    )
    row: Dict[str, object] = {
        "config_name": name,
        "side": side_tag,
        "sample_group": sample_group,
    }
    for k in CSV_PARAM_KEYS:
        row[k] = combo[k]
    return row


def write_configs_csv(
    out_path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write a plain CSV of generated configs (no simulation data).

    Missing keys (notably ``sample_group`` for non-synth strategies) are
    written as empty strings so the schema stays stable across sweep
    types.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def _validation_artifacts_from_pareto_json(
    pareto_json_path: Path,
) -> Tuple[
    str,
    Tuple[str, ...],
    str,
    str,
    str,
    List[Dict[str, int]],
    Path,
]:
    """Return side metadata + deduped validation combos + output CSV path."""
    try:
        payload = json.loads(pareto_json_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SystemExit(f"Failed to read pareto JSON {pareto_json_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {pareto_json_path}: {e}") from e

    side = payload.get("side")
    if side == "des":
        name_keys = DES_KEYS
        class_prefix = "ProtoAccelDesSweep"
        class_debug_prefix = "ProtoAccelDesSweepDebug"
    elif side == "ser":
        name_keys = SER_KEYS
        class_prefix = "ProtoAccelSerSweep"
        class_debug_prefix = "ProtoAccelSerSweepDebug"
    else:
        raise SystemExit(
            f"{pareto_json_path}: expected top-level 'side' to be 'des' or 'ser', got {side!r}"
        )

    pareto_front = payload.get("pareto_front")
    if not isinstance(pareto_front, list):
        raise SystemExit(f"{pareto_json_path}: missing or invalid 'pareto_front' list")

    combos: List[Dict[str, int]] = []
    for i, point in enumerate(pareto_front):
        if not isinstance(point, Mapping):
            raise SystemExit(f"{pareto_json_path}: pareto_front[{i}] must be an object")
        if not bool(point.get("validation_candidate", False)):
            continue
        cfg = point.get("config")
        if not isinstance(cfg, Mapping):
            raise SystemExit(f"{pareto_json_path}: pareto_front[{i}].config must be an object")

        combo = dict(DEFAULT_PARAM_VALUES)
        for k in name_keys:
            if k not in cfg:
                raise SystemExit(
                    f"{pareto_json_path}: pareto_front[{i}].config missing key {k!r}"
                )
            combo[k] = int(cfg[k])
        combos.append(combo)

    out_path = pareto_json_path.parent / f"{side}_pareto_validation_sweep_configs.csv"
    deduped = _dedupe_by_name(combos, name_keys)
    return (
        side,
        name_keys,
        class_prefix,
        class_debug_prefix,
        _csv_side_tag(name_keys),
        deduped,
        out_path,
    )


def write_validation_csv_from_pareto_json(pareto_json_path: Path) -> Path:
    """Extract validation candidates from a pareto-front JSON and emit sweep CSV."""
    (
        _side,
        name_keys,
        class_prefix,
        class_debug_prefix,
        side_tag,
        deduped,
        out_path,
    ) = _validation_artifacts_from_pareto_json(pareto_json_path)
    rows = [
        _combo_csv_row(
            class_prefix=class_prefix,
            class_debug_prefix=class_debug_prefix,
            name_keys=name_keys,
            combo=c,
            side_tag=side_tag,
        )
        for c in deduped
    ]
    write_configs_csv(out_path, rows)
    return out_path


def write_validation_scala_from_pareto_json(
    pareto_json_path: Path, *, generate_debug: bool
) -> Tuple[Path, int, int]:
    """Emit side-specific Scala configs for validation candidates."""
    (
        side,
        name_keys,
        _class_prefix,
        _class_debug_prefix,
        _side_tag,
        deduped,
        _csv_out_path,
    ) = _validation_artifacts_from_pareto_json(pareto_json_path)

    if side == "des":
        object_name = "ProtoAccelDeserializerValidationConfigs"
        class_prefix = "ProtoAccelDesSweep"
        class_debug_prefix = "ProtoAccelDesSweepDebug"
        scala_name = "ProtoAccelDeserializerValidationConfigs.scala"
    else:
        object_name = "ProtoAccelSerializerValidationConfigs"
        class_prefix = "ProtoAccelSerSweep"
        class_debug_prefix = "ProtoAccelSerSweepDebug"
        scala_name = "ProtoAccelSerializerValidationConfigs.scala"

    lines: List[str] = [
        "// GENERATED FILE — do not edit by hand.",
        "// Regenerate:",
        "//   python3 generators/protoacc/software/verilator-bench/gen_protoacc_sweep_configs.py \\",
        "//       --pareto-front-json <path/to/*_pareto_front.json> [--debug]",
        "//",
        f"// source={pareto_json_path}",
        "// Composes on `ProtoAccelRocketBaseConfig` (HyperscaleConfigs.scala).",
        "",
        "package chipyard",
        "",
        "import org.chipsalliance.cde.config.Config",
        "",
    ]

    emitted = _emit_side_block(
        lines=lines,
        object_name=object_name,
        class_prefix=class_prefix,
        class_debug_prefix=class_debug_prefix,
        name_keys=name_keys,
        combinations=deduped,
        sweep_type="pareto_validation",
        seed=0,
        num_configs_requested=len(deduped),
        generate_debug=generate_debug,
    )

    out_path = DEFAULT_SCALA_CONFIG_DIR / scala_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total_classes = emitted * (2 if generate_debug else 1)
    return out_path, emitted, total_classes


def write_default_scala(
    out_path: Path, *, generate_debug: bool
) -> Tuple[Path, int]:
    """Emit ``ProtoAccelDefaultConfigs.scala`` with one default baseline class.

    The class name is ``ProtoAccelDefaultConfig`` (no acronym suffix), since
    every parameter sits at its merged default value. Returns (path, total
    Scala class count emitted).
    """
    combo = dict(DEFAULT_PARAM_VALUES)
    lines: List[str] = [
        "// GENERATED FILE — do not edit by hand.",
        "// Regenerate:",
        "//   python3 generators/protoacc/software/verilator-bench/gen_protoacc_sweep_configs.py \\",
        "//       --write-default [--debug]",
        "//",
        "// Single baseline with every des_* and ser_* parameter at its default.",
        "// Composes on `ProtoAccelRocketBaseConfig` (HyperscaleConfigs.scala).",
        "",
        "package chipyard",
        "",
        "import org.chipsalliance.cde.config.Config",
        "",
    ]

    emitted = _emit_side_block(
        lines=lines,
        object_name="ProtoAccelDefaultConfigs",
        class_prefix="ProtoAccelSweepDefault",
        class_debug_prefix="ProtoAccelSweepDefaultDebug",
        name_keys=(),
        combinations=[combo],
        sweep_type="default",
        seed=0,
        num_configs_requested=1,
        generate_debug=generate_debug,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total_classes = emitted * (2 if generate_debug else 1)
    return out_path, total_classes


def render_file_joint(
    *,
    out_path: Path,
    combinations: Sequence[Mapping[str, int]],
    sweep_type: str,
    seed: int,
    num_configs_requested: int,
    generate_debug: bool,
    summaries: Sequence[str],
) -> int:
    """Emit legacy joint ``ProtoAccelSweepConfigs``. Returns total Scala class count."""
    lines: List[str] = [
        "// GENERATED FILE — do not edit by hand.",
        "// Regenerate:",
        "//   python3 generators/protoacc/software/verilator-bench/gen_protoacc_sweep_configs.py \\",
        "//       --emit joint -t random -n <N> -s <seed> [--debug]",
        "//",
        f"// emit=joint sweep-type={sweep_type} requested_n={num_configs_requested} seed={seed}",
        "// Composes on `ProtoAccelRocketBaseConfig` (HyperscaleConfigs.scala).",
        "",
        "package chipyard",
        "",
        "import org.chipsalliance.cde.config.Config",
        "",
    ]
    for s in summaries:
        lines.append(f"// {s}")
    lines.append("")

    emitted = _emit_side_block(
        lines=lines,
        object_name="ProtoAccelSweepConfigs",
        class_prefix="ProtoAccelSweep",
        class_debug_prefix="ProtoAccelSweepDebug",
        name_keys=FULL_PARAM_KEYS,
        combinations=combinations,
        sweep_type=sweep_type,
        seed=seed,
        num_configs_requested=num_configs_requested,
        generate_debug=generate_debug,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total_classes = emitted * (2 if generate_debug else 1)
    return total_classes


def render_file_split(
    *,
    out_path: Path,
    blocks: Sequence[
        Tuple[str, str, str, Tuple[str, ...], str, List[Dict[str, int]], int]
    ],
    sweep_type: str,
    seed_des: int,
    seed_ser: int,
    num_configs_requested: int,
    generate_debug: bool,
    summaries: Sequence[str],
) -> int:
    """
    ``blocks``: (object_name, class_prefix, class_debug_prefix, name_keys,
    side_label, combos, seed_for_block)
    """
    lines: List[str] = [
        "// GENERATED FILE — do not edit by hand.",
        "// Regenerate (deserializer and serializer sweeps are independent):",
        "//   python3 generators/protoacc/software/verilator-bench/gen_protoacc_sweep_configs.py \\",
        "//       --emit both -t random -n <N> -s <seed> [--debug]",
        "//",
        f"// sweep-type={sweep_type} requested_n_per_side={num_configs_requested} "
        f"seed_des={seed_des} seed_ser={seed_ser}",
        "// Composes on `ProtoAccelRocketBaseConfig` (HyperscaleConfigs.scala).",
        "",
        "package chipyard",
        "",
        "import org.chipsalliance.cde.config.Config",
        "",
    ]
    for s in summaries:
        lines.append(f"// {s}")
    lines.append("")

    total_classes = 0
    for object_name, class_pfx, class_debug_pfx, name_keys, _label, combos, block_seed in blocks:
        emitted = _emit_side_block(
            lines=lines,
            object_name=object_name,
            class_prefix=class_pfx,
            class_debug_prefix=class_debug_pfx,
            name_keys=name_keys,
            combinations=combos,
            sweep_type=sweep_type,
            seed=block_seed,
            num_configs_requested=num_configs_requested,
            generate_debug=generate_debug,
        )
        total_classes += emitted * (2 if generate_debug else 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return total_classes


# ---------------------------------------------------------------------------
# synth-training top-level driver: emits ProtoAccel{Des,Ser}SynthConfigs.scala
# with ProtoAccel{Des,Ser}Synth<acronyms>Config classes and a CSV with the
# ``sample_group`` column populated.
# ---------------------------------------------------------------------------

SYNTH_SCALA_FILENAME = "ProtoAccelSynthSweepConfigs.scala"
SYNTH_CSV_FILENAME   = "yosys_synth_sweep_configs.csv"


def _emit_synth_side_block(
    *,
    lines: List[str],
    object_name: str,
    class_prefix: str,
    class_debug_prefix: str,
    name_keys: Sequence[str],
    tagged_combos: Sequence[Tuple[Mapping[str, int], str]],
    seed: int,
    generate_debug: bool,
) -> int:
    """Like ``_emit_side_block`` but takes pre-deduped tagged combos and
    tags each class with its ``sample_group`` in the Scaladoc comment.
    Returns the row count emitted."""
    n = len(tagged_combos)
    lines.append(f"object {object_name} {{")
    lines.append("")
    lines.append(f'  val generationSweepType: String = "synth-training"')
    lines.append(f"  val generationSeed: Long = {seed}L")
    lines.append(f"  val generationNumConfigsEmitted: Int = {n}")
    lines.append("")
    lines.append("  /** CONFIG= names for non-debug generated classes (basename only). */")
    lines.append("  val normalSweepConfigNames: Seq[String] = Seq(")
    for combo, _group in tagged_combos:
        name = _class_basename(
            class_prefix=class_prefix,
            class_debug_prefix=class_debug_prefix,
            name_keys=name_keys,
            combo=combo,
            debug=False,
        )
        lines.append(f'    "{name}",')
    lines.append("  )")

    if generate_debug:
        lines += [
            "",
            "  /** CONFIG= names for debug (printf) variants. */",
            "  val debugSweepConfigNames: Seq[String] = Seq(",
        ]
        for combo, _group in tagged_combos:
            dname = _class_basename(
                class_prefix=class_prefix,
                class_debug_prefix=class_debug_prefix,
                name_keys=name_keys,
                combo=combo,
                debug=True,
            )
            lines.append(f'    "{dname}",')
        lines.append("  )")
    else:
        lines += [
            "",
            "  /** Populate with ``--debug``. */",
            "  val debugSweepConfigNames: Seq[String] = Seq.empty[String]",
        ]

    lines.append("}")
    lines.append("")

    for i, (combo, group) in enumerate(tagged_combos):
        cmt = _combo_comment(combo)
        name = _class_basename(
            class_prefix=class_prefix,
            class_debug_prefix=class_debug_prefix,
            name_keys=name_keys,
            combo=combo,
            debug=False,
        )
        lines.append(f"/** Synth row {i + 1}/{n} (group={group}): {cmt} */")
        lines.append(f"class {name} extends Config(")
        lines.append(_scala_fragments(combo))
        lines.append(BASE_FRAGMENTS)
        lines.append("")

        if generate_debug:
            dname = _class_basename(
                class_prefix=class_prefix,
                class_debug_prefix=class_debug_prefix,
                name_keys=name_keys,
                combo=combo,
                debug=True,
            )
            lines.append(f"/** Debug printf variant of `{name}`. */")
            lines.append(f"class {dname} extends Config(")
            lines.append(DEBUG_PREFIX.rstrip("\n"))
            lines.append(_scala_fragments(combo))
            lines.append(BASE_FRAGMENTS)
            lines.append("")

    return n


def _run_synth_training(args: argparse.Namespace) -> None:
    emit = args.emit  # "both" | "des" | "ser"
    sides: List[str] = []
    if emit in ("both", "des"):
        sides.append("des")
    if emit in ("both", "ser"):
        sides.append("ser")

    # File names default to the shared Chipyard config dir (same as sweep).
    scala_out = args.output
    if scala_out == DEFAULT_OUT:
        scala_out = DEFAULT_SCALA_CONFIG_DIR / SYNTH_SCALA_FILENAME
    csv_out = args.csv_output
    if csv_out == DEFAULT_CSV_OUT:
        csv_out = SCRIPT_DIR / SYNTH_CSV_FILENAME

    per_side_blocks: List[Tuple[str, str, str, Tuple[str, ...], str, List[Tuple[Dict[str, int], str]], int]] = []
    summaries: List[str] = []

    for side in sides:
        keys = DES_KEYS if side == "des" else SER_KEYS
        side_seed = args.seed if side == "des" else args.seed + _SER_SEED_OFFSET
        tagged, summary = build_synth_training_combinations(
            keys,
            n_lhs=args.n_lhs,
            n_stress=args.n_stress,
            n_holdout=args.n_holdout,
            seed=side_seed,
        )
        deduped = _dedupe_tagged_by_name(tagged, keys, SYNTH_TRAINING_GROUP_PRIORITY)
        summaries.append(summary + f" (post-dedup={len(deduped)})")

        if side == "des":
            per_side_blocks.append((
                "ProtoAccelDesSynthConfigs",
                "ProtoAccelDesSynth",
                "ProtoAccelDesSynthDebug",
                keys,
                "deserializer",
                deduped,
                side_seed,
            ))
        else:
            per_side_blocks.append((
                "ProtoAccelSerSynthConfigs",
                "ProtoAccelSerSynth",
                "ProtoAccelSerSynthDebug",
                keys,
                "serializer",
                deduped,
                side_seed,
            ))

    # Emit Scala file (one file, both objects).
    lines: List[str] = [
        "// GENERATED FILE — do not edit by hand.",
        "// Regenerate:",
        "//   python3 -m lynx.sweep_configs.gen_sweep_configs \\",
        "//       --emit both -t synth-training [--n-lhs N] [--seed S] [--debug]",
        "//",
        f"// sweep-type=synth-training seed_des={args.seed} "
        f"seed_ser={args.seed + _SER_SEED_OFFSET} "
        f"n_lhs={args.n_lhs} n_stress={args.n_stress} n_holdout={args.n_holdout}",
        "// Composes on `ProtoAccelRocketBaseConfig` (HyperscaleConfigs.scala).",
        "",
        "package chipyard",
        "",
        "import org.chipsalliance.cde.config.Config",
        "",
    ]
    for s in summaries:
        lines.append(f"// {s}")
    lines.append("")

    total_classes = 0
    for object_name, class_pfx, class_debug_pfx, name_keys, _side_label, tagged_combos, side_seed in per_side_blocks:
        emitted = _emit_synth_side_block(
            lines=lines,
            object_name=object_name,
            class_prefix=class_pfx,
            class_debug_prefix=class_debug_pfx,
            name_keys=name_keys,
            tagged_combos=tagged_combos,
            seed=side_seed,
            generate_debug=args.debug,
        )
        total_classes += emitted * (2 if args.debug else 1)

    scala_out.parent.mkdir(parents=True, exist_ok=True)
    scala_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Emit CSV (with sample_group).
    csv_rows: List[Dict[str, object]] = []
    for _object_name, class_pfx, class_debug_pfx, name_keys, _side_label, tagged_combos, _seed in per_side_blocks:
        side_tag = _csv_side_tag(name_keys)
        for combo, group in tagged_combos:
            csv_rows.append(
                _combo_csv_row(
                    class_prefix=class_pfx,
                    class_debug_prefix=class_debug_pfx,
                    name_keys=name_keys,
                    combo=combo,
                    side_tag=side_tag,
                    sample_group=group,
                )
            )
    write_configs_csv(csv_out, csv_rows)

    for s in summaries:
        print(s)
    n_non_debug = total_classes // (2 if args.debug else 1)
    print(
        f"Wrote {scala_out} ({n_non_debug} distinct configs across "
        f"{len(per_side_blocks)} side block(s), {total_classes} Scala classes)"
    )
    print(f"Wrote {csv_out} ({len(csv_rows)} config rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output Scala path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--emit",
        type=str,
        default="both",
        choices=["both", "des", "ser", "joint"],
        help=(
            "both: ProtoAccelDesSweepConfigs + ProtoAccelSerSweepConfigs; "
            "des/ser: one side only; joint: legacy single-object joint sweep."
        ),
    )
    parser.add_argument(
        "-t",
        "--sweep-type",
        type=str,
        default="random",
        choices=["random", "ofat", "default", "tweak", "synth-training"],
        help=(
            "random: sample N combos on active axes; "
            "ofat: every value of each active param; default: one baseline; "
            "tweak: one row per active param off-default; "
            "synth-training: defaults+corners+ofat+LHS+stress+holdout in one "
            "call (emits ProtoAccel{Des,Ser}Synth<acronyms>Config classes)."
        ),
    )
    parser.add_argument(
        "--n-lhs",
        type=int,
        default=440,
        help=(
            "Latin-hypercube sample size for -t synth-training (per side). "
            "Default 440 -> total ~512 per side after adding "
            "defaults/corners/ofat/stress/holdout."
        ),
    )
    parser.add_argument(
        "--n-stress",
        type=int,
        default=25,
        help="Stress-corner sample count for -t synth-training (per side).",
    )
    parser.add_argument(
        "--n-holdout",
        type=int,
        default=40,
        help="Held-out LHS sample count for -t synth-training (per side).",
    )
    parser.add_argument(
        "-n",
        "--num-configs",
        type=int,
        default=32,
        help=(
            "For random sweeps: sample count. With --emit both, this is **per side** "
            "(des and ser each get N samples). Default 32 keeps the generated Scala "
            "roughly the same size as the old 32-row joint sweep."
        ),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=262,
        help="Random seed for random sweeps (ser side uses seed+1000003 when --emit both).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "For random sweeps: skip the first OFFSET samples from the seeded "
            "draw. Lets two machines with the same seed produce disjoint "
            "samples — e.g. one runs with -n N --offset 0, another with "
            "-n N --offset N. Applied per side with --emit both. Ignored for "
            "non-random sweep types."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also emit WithProtoAccelPrintf twin for each config",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUT,
        help=(
            f"CSV file listing the emitted configs (no simulation data). "
            f"Default: {DEFAULT_CSV_OUT}"
        ),
    )
    parser.add_argument(
        "--write-default",
        action="store_true",
        help=(
            "Emit ProtoAccelDefaultConfigs.scala with a single "
            "ProtoAccelDefaultConfig class holding every des_*/ser_* parameter "
            "at its merged default value. Skips sweep generation; other sweep "
            "flags are ignored."
        ),
    )
    parser.add_argument(
        "--pareto-front-json",
        type=Path,
        default=None,
        help=(
            "If set, read pareto_front JSON, select points with "
            "validation_candidate=true, and emit <side>_pareto_validation_sweep_configs.csv "
            "next to the JSON. Skips Scala/random sweep generation."
        ),
    )
    args = parser.parse_args()

    _validate_tables()

    if args.write_default:
        out_scala = DEFAULT_SCALA_CONFIG_DIR / "ProtoAccelDefaultConfigs.scala"
        out_path, total = write_default_scala(out_scala, generate_debug=args.debug)
        print(f"Wrote {out_path} (1 default config, {total} Scala class(es))")
        return

    if args.pareto_front_json is not None:
        out_csv = write_validation_csv_from_pareto_json(args.pareto_front_json)
        out_scala, emitted, total = write_validation_scala_from_pareto_json(
            args.pareto_front_json, generate_debug=args.debug
        )
        print(f"Wrote {out_csv} (validation candidates from {args.pareto_front_json})")
        print(
            f"Wrote {out_scala} ({emitted} validation configs, {total} Scala classes)"
        )
        return

    emit = args.emit
    sweep_type = args.sweep_type
    n = args.num_configs
    seed = args.seed
    offset = args.offset

    if offset and sweep_type != "random":
        print(
            f"warning: --offset={offset} is ignored for sweep-type={sweep_type!r} "
            "(only affects random sweeps)."
        )

    if sweep_type == "synth-training":
        if emit == "joint":
            raise SystemExit("--emit joint is not supported for -t synth-training.")
        _run_synth_training(args)
        return

    if emit == "joint":
        combos, summary = build_combinations(
            sweep_type, n, seed, FULL_PARAM_KEYS, offset=offset
        )
        summaries = [summary]
        total = render_file_joint(
            out_path=args.output,
            combinations=combos,
            sweep_type=sweep_type,
            seed=seed,
            num_configs_requested=n,
            generate_debug=args.debug,
            summaries=summaries,
        )
        deduped = _dedupe_by_name(combos, FULL_PARAM_KEYS)
        csv_rows = [
            _combo_csv_row(
                class_prefix="ProtoAccelSweep",
                class_debug_prefix="ProtoAccelSweepDebug",
                name_keys=FULL_PARAM_KEYS,
                combo=c,
                side_tag=_csv_side_tag(FULL_PARAM_KEYS),
            )
            for c in deduped
        ]
        write_configs_csv(args.csv_output, csv_rows)
        print(summary)
        n_non_debug = total // (2 if args.debug else 1)
        print(f"Wrote {args.output} ({n_non_debug} distinct configs, {total} Scala classes)")
        print(f"Wrote {args.csv_output} ({len(csv_rows)} config rows)")
        return

    seed_des = seed
    seed_ser = seed + _SER_SEED_OFFSET
    summaries: List[str] = []
    blocks: List[
        Tuple[str, str, str, Tuple[str, ...], str, List[Dict[str, int]], int]
    ] = []

    if emit in ("both", "des"):
        c_des, s_des = build_combinations(
            sweep_type, n, seed_des, DES_KEYS, offset=offset
        )
        summaries.append(s_des)
        blocks.append(
            (
                "ProtoAccelDesSweepConfigs",
                "ProtoAccelDesSweep",
                "ProtoAccelDesSweepDebug",
                DES_KEYS,
                "deserializer",
                c_des,
                seed_des,
            )
        )
    if emit in ("both", "ser"):
        c_ser, s_ser = build_combinations(
            sweep_type, n, seed_ser, SER_KEYS, offset=offset
        )
        summaries.append(s_ser)
        blocks.append(
            (
                "ProtoAccelSerSweepConfigs",
                "ProtoAccelSerSweep",
                "ProtoAccelSerSweepDebug",
                SER_KEYS,
                "serializer",
                c_ser,
                seed_ser,
            )
        )

    for s in summaries:
        print(s)
    total = render_file_split(
        out_path=args.output,
        blocks=blocks,
        sweep_type=sweep_type,
        seed_des=seed_des,
        seed_ser=seed_ser,
        num_configs_requested=n,
        generate_debug=args.debug,
        summaries=summaries,
    )

    csv_rows: List[Dict[str, object]] = []
    for object_name, class_pfx, class_debug_pfx, name_keys, _label, combos, _seed in blocks:
        deduped = _dedupe_by_name(combos, name_keys)
        side_tag = _csv_side_tag(name_keys)
        for c in deduped:
            csv_rows.append(
                _combo_csv_row(
                    class_prefix=class_pfx,
                    class_debug_prefix=class_debug_pfx,
                    name_keys=name_keys,
                    combo=c,
                    side_tag=side_tag,
                )
            )
    write_configs_csv(args.csv_output, csv_rows)

    n_non_debug = total // (2 if args.debug else 1)
    print(
        f"Wrote {args.output} ({n_non_debug} distinct configs across {len(blocks)} "
        f"side block(s), {total} Scala classes)"
    )
    print(f"Wrote {args.csv_output} ({len(csv_rows)} config rows)")


if __name__ == "__main__":
    main()
