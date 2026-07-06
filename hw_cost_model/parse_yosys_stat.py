#!/usr/bin/env python3
"""
Parse a Yosys ``synth.log`` into per-bucket ``(logic_cells, ram_bits)``
counts for one ProtoAcc synth run.

Input: a log with one ``=== <module> ===`` block per module (produced by
``synth`` followed by ``stat``). Each block reports total cells and the
list of child modules with instance counts. ``ram_<D>x<W>`` modules are
firtool-emitted memory register files -- we separate their cell counts
from the "logic" total and expose the storage as ``D * W`` bits.

Output: a single CSV row for one config, joining the sweep-config table
with the per-bucket cost labels expected by ``fit_from_yosys.py``.

Usage as a library::

    from hw_cost_model.parse_yosys_stat import parse_synth_log
    row = parse_synth_log(Path("synth.log"), side="des",
                          config_name="ProtoAccelDesSweep...Config",
                          config_params=cfg_dict)

Usage as a CLI (per-config worker)::

    python -m hw_cost_model.parse_yosys_stat \\
        --synth-log path/to/synth.log \\
        --side des \\
        --config-name ProtoAccelDesSweep...Config \\
        --config-params-json '{"des_top_descriptor_reqs":8,...}' \\
        --csv-append yosys_sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .defaults import DES_PARAM_COLUMNS, SER_PARAM_COLUMNS
from .structural_features import (
    DES_BUCKET_TO_YOSYS_TOPS,
    DES_SUBMODULES,
    SER_BUCKET_TO_YOSYS_TOPS,
    SER_SUBMODULES,
    bucket_to_yosys_tops,
)


_BLOCK_HEADER = re.compile(r"^===\s+(.+?)\s+===\s*$")
_NUM_CELLS    = re.compile(r"^\s*Number of cells:\s*(\d+)\s*$")
_NUM_WIRES    = re.compile(r"^\s*Number of wires:\s*(\d+)\s*$")
_NUM_MEMBITS  = re.compile(r"^\s*Number of memory bits:\s*(\d+)\s*$")
_CHILD_LINE   = re.compile(r"^\s{5,}(\S+)\s+(\d+)\s*$")
_RAM_MODULE   = re.compile(r"^ram(_data)?_(\d+)x(\d+)$")


@dataclass
class ModuleStat:
    """Per-module stat block parsed from a Yosys ``synth.log``.

    ``cells`` is the total from the ``Number of cells:`` line -- includes
    both leaf primitives ($_DFF, $_MUX, ...) and instantiated child
    modules (each counted once per line, times the instance count on
    that line). To get the "cells that live directly in this module",
    subtract the transitive rollup for all children. To get the
    "cells reachable from this module", roll children up recursively.
    """
    name: str
    cells: int = 0
    wires: int = 0
    mem_bits: int = 0
    # child module name -> instance count on this parent
    children: dict[str, int] = field(default_factory=dict)
    # Primitive-cell counts (e.g. $_DFF_PP_, $_MUX_) recorded verbatim.
    primitive_cells: dict[str, int] = field(default_factory=dict)


def _is_module_name(tok: str) -> bool:
    """Yosys child-line tokens are either primitive cells ($_...) or
    module names (start with an identifier char, no ``$`` prefix)."""
    return not tok.startswith("$")


def parse_stat_blocks(log_text: str) -> dict[str, ModuleStat]:
    """Return every ``=== <mod> ===`` block, keyed by module name.

    Yosys logs sometimes contain multiple ``stat`` dumps (e.g. when the
    driving script calls ``stat`` more than once). Duplicate blocks per
    module name are collapsed: the last one wins, which is the state
    after the final synthesis pass and therefore what we want.
    """
    out: dict[str, ModuleStat] = {}
    current: ModuleStat | None = None
    for line in log_text.splitlines():
        m = _BLOCK_HEADER.match(line)
        if m:
            name = m.group(1)
            # ``=== design hierarchy ===`` is a separate section whose
            # body is an indented tree that looks like child lines. Skip
            # it entirely; anything with whitespace in the "name" is
            # such a section header, not a real module.
            if " " in name:
                current = None
                continue
            # Replace (rather than accumulate) if we've seen this module
            # before -- Yosys re-prints the same numbers on repeated stat.
            current = ModuleStat(name=name)
            out[name] = current
            continue
        if current is None:
            continue
        m = _NUM_CELLS.match(line)
        if m:
            current.cells = int(m.group(1))
            continue
        m = _NUM_WIRES.match(line)
        if m:
            current.wires = int(m.group(1))
            continue
        m = _NUM_MEMBITS.match(line)
        if m:
            current.mem_bits = int(m.group(1))
            continue
        m = _CHILD_LINE.match(line)
        if m:
            tok, count = m.group(1), int(m.group(2))
            if _is_module_name(tok):
                current.children[tok] = current.children.get(tok, 0) + count
            else:
                current.primitive_cells[tok] = current.primitive_cells.get(tok, 0) + count
    return out


def ram_bits(module_name: str) -> int:
    """If ``module_name`` matches ``ram_<D>x<W>`` or ``ram_data_<D>x<W>``,
    return D*W bits; else 0. These are firtool's register-file emissions
    for Chisel ``Mem`` / ``SyncReadMem``; Yosys leaves them as flops so
    their cell count already includes storage.
    """
    m = _RAM_MODULE.match(module_name)
    if not m:
        return 0
    depth, width = int(m.group(2)), int(m.group(3))
    return depth * width


def _transitive_instance_counts(
    top: str,
    blocks: Mapping[str, ModuleStat],
    cache: dict[str, dict[str, int]],
    stop_at: frozenset[str],
) -> dict[str, int]:
    """Return ``{module_name: transitive_instance_count}`` reachable from
    ``top``, treating multiple hierarchical paths as additive.

    ``stop_at``: set of module names that terminate descent. When the
    walk encounters a child in ``stop_at``, the child is included in
    the returned dict (so an outer bucket can still count IT), but its
    subtree is NOT expanded. This lets buckets partition the hierarchy
    -- each module belongs to exactly one bucket. Empty set = descend
    everywhere.

    Memoized on ``(top, stop_at)`` via the caller's ``cache``. Cyclic
    instantiation is broken by inserting an empty placeholder before
    recursing.
    """
    if top in cache:
        return cache[top]
    if top not in blocks:
        cache[top] = {}
        return cache[top]
    cache[top] = {}  # placeholder to break cycles
    out: dict[str, int] = {top: 1}
    stat = blocks[top]
    for child, count in stat.children.items():
        if child in stop_at:
            # Include the immediate child but do not descend.
            out[child] = out.get(child, 0) + count
            continue
        sub = _transitive_instance_counts(child, blocks, cache, stop_at)
        for m, c in sub.items():
            out[m] = out.get(m, 0) + count * c
    cache[top] = out
    return out


def bucket_reachable(
    bucket_tops: tuple[str, ...],
    blocks: Mapping[str, ModuleStat],
    *,
    stop_at: frozenset[str] = frozenset(),
) -> dict[str, int]:
    """For a bucket whose Yosys tops are ``bucket_tops``, return a
    ``{module_name: instance_count}`` map of every module reachable
    from those tops, stopping descent at any name in ``stop_at`` (so
    buckets can partition the hierarchy).

    The bucket's own tops are always included even if they appear in
    ``stop_at`` (a top can't be pruned by itself).
    """
    cache: dict[str, dict[str, int]] = {}
    stop_but_not_our_tops = frozenset(stop_at - set(bucket_tops))
    reachable: dict[str, int] = {}
    for top in bucket_tops:
        for m, c in _transitive_instance_counts(top, blocks, cache, stop_but_not_our_tops).items():
            reachable[m] = reachable.get(m, 0) + c
    # Any ``stop_at`` entries that showed up as included-but-not-expanded
    # children get filtered out here -- they'll be counted by their own
    # bucket instead. Only keep our own tops.
    filtered = {}
    for m, c in reachable.items():
        if m in stop_but_not_our_tops:
            continue
        filtered[m] = c
    return filtered


def _module_direct_cells(stat: ModuleStat, blocks: Mapping[str, ModuleStat]) -> int:
    """Cells that live *directly* in a module (its primitive cells only),
    excluding transitive children. Since ``stat.cells`` already sums all
    primitive-cell lines PLUS all module-instance lines, and each
    module-instance is a leaf from *this* module's stat block's
    perspective, ``direct_cells`` == sum of primitive-cell lines.
    """
    return sum(stat.primitive_cells.values())


def bucket_costs(
    bucket_tops: tuple[str, ...],
    blocks: Mapping[str, ModuleStat],
    *,
    stop_at: frozenset[str] = frozenset(),
) -> tuple[int, int, int]:
    """Return ``(logic_cells, ram_bits, ram_flop_cells)`` for a bucket.

    * ``logic_cells``  -- transitive primitive cell count, excluding cells
                          inside any ``ram_<D>x<W>`` submodule.
    * ``ram_bits``     -- sum of D*W across every ``ram_*`` module reached,
                          weighted by transitive instance count.
    * ``ram_flop_cells`` -- transitive primitive-cell count inside ram_
                            modules (returned for auditing; the training
                            label uses ``ram_bits``, not this).

    ``stop_at``: other buckets' top module names; descent halts there
    so the buckets partition the hierarchy cleanly. Any bucket with
    ``top`` in its own top list (``ProtoAccel`` / ``ProtoAccelSerializer``)
    should be handled with ``stop_at`` covering all other bucket tops so
    the top-level bucket represents only the RoCC glue.
    """
    reachable = bucket_reachable(bucket_tops, blocks, stop_at=stop_at)
    logic_cells = 0
    ram_b = 0
    ram_c = 0
    for mod_name, count in reachable.items():
        stat = blocks.get(mod_name)
        if stat is None:
            continue
        direct = _module_direct_cells(stat, blocks)
        if ram_bits(mod_name) > 0:
            ram_c += count * direct
            ram_b += count * ram_bits(mod_name)
        else:
            logic_cells += count * direct
    return logic_cells, ram_b, ram_c


def _expand_bucket_tops(
    bucket_map: Mapping[str, tuple[str, ...]],
    blocks: Mapping[str, ModuleStat],
) -> dict[str, tuple[str, ...]]:
    """Expand any bucket top that is a *prefix pattern* to the actual set
    of matching module names in ``blocks``.

    Rule: a bucket top is treated as a prefix if it doesn't itself
    appear as a module in ``blocks`` and no ``blocks`` module equals it
    exactly. In that case we include every module whose name starts
    with the pattern. Exact matches take precedence when both exist.

    Example: ``"TLBuffer"`` in ``blocks`` where the actual modules are
    ``TLBuffer_a32d128s3k3z4u``, ``TLBuffer_a32d128s5k3z4u``, etc.
    """
    expanded: dict[str, tuple[str, ...]] = {}
    for bucket, tops in bucket_map.items():
        out: list[str] = []
        for top in tops:
            if top in blocks:
                out.append(top)
            else:
                out.extend(sorted(m for m in blocks if m.startswith(top)))
        expanded[bucket] = tuple(out)
    return expanded


def _row_for(
    *,
    config_name: str,
    side: str,
    config_params: Mapping[str, int],
    blocks: Mapping[str, ModuleStat],
    sample_group: str = "",
) -> dict[str, object]:
    # Guard against the failure mode where yosys ran but its ``stat``
    # output was suppressed (e.g. ``yosys -q``). Without at least the
    # top-module block, every ``submod_*`` column would silently be
    # zero; refuse to write that row so the CSV can be trusted.
    top_mod = ("ProtoAccel" if side == "des" else "ProtoAccelSerializer")
    if top_mod not in blocks:
        raise ValueError(
            f"synth.log has no '=== {top_mod} ===' block "
            f"(saw {len(blocks)} block(s)); "
            "yosys did not emit its ``stat`` output. "
            "Check that the yosys command runs 'stat' and is not '-q'."
        )

    param_cols = DES_PARAM_COLUMNS if side == "des" else SER_PARAM_COLUMNS
    buckets = DES_SUBMODULES if side == "des" else SER_SUBMODULES
    bucket_map = _expand_bucket_tops(bucket_to_yosys_tops(side), blocks)

    row: dict[str, object] = {
        "config_name": config_name,
        "side": side,
        "sample_group": sample_group,
    }
    for col in param_cols:
        row[col] = config_params.get(col, 0)

    all_tops: set[str] = set()
    for tops in bucket_map.values():
        all_tops.update(tops)

    total_logic = 0
    total_ram_bits = 0
    total_ram_cells = 0
    for bucket in buckets:
        tops = bucket_map[bucket]
        if not tops:
            row[f"submod_{bucket}_logic_cells"] = 0
            row[f"submod_{bucket}_ram_bits"] = 0
            row[f"submod_{bucket}_ram_flop_cells"] = 0
            continue
        stop_at = frozenset(all_tops - set(tops))
        lc, rb, rc = bucket_costs(tops, blocks, stop_at=stop_at)
        row[f"submod_{bucket}_logic_cells"] = lc
        row[f"submod_{bucket}_ram_bits"] = rb
        row[f"submod_{bucket}_ram_flop_cells"] = rc
        total_logic += lc
        total_ram_bits += rb
        total_ram_cells += rc

    row["total_logic_cells"] = total_logic
    row["total_ram_bits"] = total_ram_bits
    row["total_ram_flop_cells"] = total_ram_cells
    return row


def parse_synth_log(
    log_path: Path | str,
    *,
    side: str,
    config_name: str,
    config_params: Mapping[str, int],
    sample_group: str = "",
) -> dict[str, object]:
    log_text = Path(log_path).read_text(encoding="utf-8")
    blocks = parse_stat_blocks(log_text)
    return _row_for(
        config_name=config_name,
        side=side,
        config_params=config_params,
        blocks=blocks,
        sample_group=sample_group,
    )


def csv_columns_for_side(side: str) -> list[str]:
    """Column names for the CSV that ``fit_from_yosys.py`` consumes."""
    param_cols = DES_PARAM_COLUMNS if side == "des" else SER_PARAM_COLUMNS
    buckets = DES_SUBMODULES if side == "des" else SER_SUBMODULES
    cols = ["config_name", "side", "sample_group"] + list(param_cols)
    for b in buckets:
        cols += [
            f"submod_{b}_logic_cells",
            f"submod_{b}_ram_bits",
            f"submod_{b}_ram_flop_cells",
        ]
    cols += ["total_logic_cells", "total_ram_bits", "total_ram_flop_cells"]
    return cols


def append_row_locked(csv_path: Path, row: Mapping[str, object], side: str) -> None:
    """Append one row to ``csv_path`` under ``flock`` so parallel workers
    don't corrupt the CSV. Writes the header if the file doesn't exist.

    Mirrors the locking pattern used by
    ``verilator-bench/run_sweep.sh``. Columns are determined by
    ``csv_columns_for_side(side)`` -- callers must not mix rows from
    different sides in the same file.
    """
    cols = csv_columns_for_side(side)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=cols)
            if new_file:
                writer.writeheader()
            writer.writerow({k: row.get(k, 0) for k in cols})
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth-log", type=Path, required=True)
    parser.add_argument("--side", choices=("des", "ser"), required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--config-params-json",
        required=True,
        help="JSON dict mapping every param key to its value for this config.",
    )
    parser.add_argument(
        "--sample-group",
        default="",
        help=(
            "Optional sample_group tag from ``gen_sweep_configs.py -t "
            "synth-training`` (defaults / corner_min / corner_max / ofat / "
            "lhs / stress / holdout). Passed straight through to the CSV so "
            "the fitter can hold out the 'holdout' rows."
        ),
    )
    parser.add_argument(
        "--csv-append",
        type=Path,
        default=None,
        help="If set, append one row to this CSV under a file lock. "
             "Otherwise print the row as JSON to stdout.",
    )
    args = parser.parse_args(argv)

    params = json.loads(args.config_params_json)
    row = parse_synth_log(
        args.synth_log,
        side=args.side,
        config_name=args.config_name,
        config_params=params,
        sample_group=args.sample_group,
    )
    if args.csv_append is not None:
        append_row_locked(args.csv_append, row, args.side)
    else:
        json.dump(row, sys.stdout, indent=2, default=str)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
