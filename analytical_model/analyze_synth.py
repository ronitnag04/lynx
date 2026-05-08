#!/usr/bin/env python3
"""
analyze_synth.py — analyze synthetic fleetbench-derived benches and append
their entries to ``protobuf_analysis_verilator_bench.json`` so the existing
feature pipeline (``extract_features.py`` → ``extracted_features.json``) picks
them up alongside the HPB benches.

Why a separate script (instead of extending protobuf_analyzer.py's
``analyze_hyperprotobench``): that path is tightly coupled to HPB's
``benchmark.inc`` — it regex-parses ``<Msg>_Set_F1`` bodies and uses
``r'set_([a-z0-9]+)'`` which breaks on fleetbench's ``set_f_0`` identifiers.
Reusing ``ProtobufAnalyzer`` (schema parsing) and layering a small
runtime-lengths-driven sizer on top keeps the moving parts minimal and avoids
touching the HPB analysis path.

Inputs per synthetic bench (produced by ``gen_synth_proto.py``):
    <synth-root>/bench<N>/benchmark.proto
    <synth-root>/bench<N>/runtime_lengths.json   {msg_simple: {field: [lens...]}}

Output (shape required by ``extract_features.extract_features_for_benchmark``):
    {
      "bench<N>": {
        "total_size_bytes": <int>,
        "messages": [
          {
            "name": <str>,
            "depth": <int>,
            "total_fields": <int>,
            "nested_message_count": <int>,
            "serialized_size_bytes": <int>,
          },
          ...
        ],
      },
      ...
    }

Typical usage (from the repo root):

    # 1. Generate synthetic protos + runtime lengths (in verilator-bench tree).
    make -C ../verilator-bench synth SYNTH_COUNT=20 SYNTH_SEED=42

    # 2. Append their analysis entries to the verilator-bench JSON.
    python3 analyze_synth.py \\
        --synth-root ../verilator-bench/gen/synth \\
        --in-json protobuf_analysis_verilator_bench.json \\
        --out-json protobuf_analysis_verilator_bench.json

    # 3. Re-run the feature extractor on the updated JSON.
    python3 extract_features.py \\
        --input protobuf_analysis_verilator_bench.json \\
        --output extracted_features.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ProtobufAnalyzer + Field/Message live next to this script.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from protobuf_analyzer import (
    ProtobufAnalyzer, WorkloadProfile, Message, Field,
)

# Tag size + varint helpers, duplicated here rather than imported so this
# script has no dependency on ProtobufAnalyzer's private statics being
# exported. These are tiny and the canonical copies live in protobuf_analyzer.
def _tag_size(field_number: int) -> int:
    tag_val = (field_number << 3) | 5
    if tag_val < 128: return 1
    if tag_val < 16384: return 2
    if tag_val < 2097152: return 3
    if tag_val < 268435456: return 4
    return 5


def _varint_size(value: int) -> int:
    # Unsigned 64-bit varint size.
    v = value & ((1 << 64) - 1)
    if v < (1 << 7):  return 1
    if v < (1 << 14): return 2
    if v < (1 << 21): return 3
    if v < (1 << 28): return 4
    if v < (1 << 35): return 5
    if v < (1 << 42): return 6
    if v < (1 << 49): return 7
    if v < (1 << 56): return 8
    if v < (1 << 63): return 9
    return 10


# Nominal values used when we have no runtime observation for a VARINT field.
# Matches what proto_to_accel.py writes (0..2^31-1 for 32-bit, 0..2^63-1 for
# 64-bit), so features and measurements line up.
_NOMINAL_VARINT_VALUE = 0x7FFFFFFF


def _pick_string_length(
    rm_name: str,
    field_name: str,
    runtime_lengths: Dict[str, Dict[str, List[int]]],
    max_string_len: Optional[int],
    fallback: int = 8,
) -> int:
    """Pick a representative string/bytes length for this field.

    Uses the median observation if available (matches the "typical" value the
    serializer emits better than a mean would when the runtime distribution is
    skewed); falls back to a small constant if the message isn't known. This
    matches the rough behavior of ``proto_to_accel.py``'s ``_pick_string_length``
    but returns a single scalar (we're computing a size, not sampling).
    """
    per_msg = runtime_lengths.get(rm_name, {})
    lengths = per_msg.get(field_name, [])
    if lengths:
        length = int(statistics.median(lengths))
    else:
        length = fallback
    if max_string_len is not None:
        length = min(length, max_string_len)
    return max(0, length)


def _collect_all_messages(top: List[Message]) -> Dict[str, Message]:
    """Flatten a list of top-level Messages into a {simple_name: Message} map.

    Fleetbench uses ``package fleetbench.proto;`` with `Message<K>` containing
    nested `M1`/`M2`/... — simple names may collide across different top-level
    messages in the same combined bench (e.g. two different ``M1``s). In that
    case the last one encountered wins; the sizer below prefers resolving
    through the parent's ``nested_messages`` list when it can, and only falls
    back to this map for top-level references.
    """
    flat: Dict[str, Message] = {}
    def walk(m: Message) -> None:
        flat[m.name] = m
        for nm in m.nested_messages:
            walk(nm)
    for m in top:
        walk(m)
    return flat


def _compute_message_size(
    msg: Message,
    all_top: Dict[str, Message],
    runtime_lengths: Dict[str, Dict[str, List[int]]],
    profile: WorkloadProfile,
    depth_override: Optional[int] = None,
    memo: Optional[Dict[str, int]] = None,
) -> int:
    """Serialized-size estimate for one message under the workload profile.

    Mirrors how proto_to_accel.py fills an instance: one value per present
    field, strings sampled at ``runtime_lengths`` median (capped at
    ``max_string_len``), nested messages recursed up to ``max_nested_depth``,
    ``repeated`` fields skipped when ``profile.skip_repeated``.

    The ``memo`` dict caches sizes by message name to make deeply recursive
    schemas cheap and to avoid pathological blow-up on cyclic references —
    the memoization is keyed on the simple name, which is correct as long as
    the depth cap is enforced (the profile's ``max_nested_depth`` guarantees
    termination).
    """
    if memo is None:
        memo = {}
    effective_depth = depth_override if depth_override is not None else msg.depth
    if profile.max_nested_depth is not None and effective_depth >= profile.max_nested_depth:
        return 0
    cached = memo.get(msg.name)
    if cached is not None:
        return cached

    total = 0
    for f in msg.fields:
        if profile.skip_repeated and f.cardinality == "repeated":
            continue
        tag = _tag_size(f.field_number)
        wire = f.wire_type
        if f.is_nested_message:
            nested: Optional[Message] = None
            # Prefer an actual nested child of this message.
            for nm in msg.nested_messages:
                if nm.name == f.nested_message_name:
                    nested = nm
                    break
            if nested is None:
                nested = all_top.get(f.nested_message_name or "")
            if nested is None:
                continue
            body = _compute_message_size(
                nested, all_top, runtime_lengths, profile,
                depth_override=effective_depth + 1,
                memo=memo,
            )
            if body == 0:
                continue
            total += tag + _varint_size(body) + body
        elif wire == "I32":
            total += tag + 4
        elif wire == "I64":
            total += tag + 8
        elif wire == "VARINT":
            # bool/enum are tiny; other varints we size at the nominal generator value.
            if f.field_type == "bool" or f.is_enum:
                total += tag + 1
            else:
                total += tag + _varint_size(_NOMINAL_VARINT_VALUE)
        elif wire == "LEN":
            # string / bytes
            length = _pick_string_length(
                msg.name, f.name, runtime_lengths, profile.max_string_len,
            )
            total += tag + _varint_size(length) + length
        else:
            # Unknown / group — skip, matches proto_to_accel behavior.
            continue

    memo[msg.name] = total
    return total


def analyze_one_bench(
    bench_dir: Path,
    profile: WorkloadProfile,
) -> Dict[str, object]:
    """Analyze a single synthetic bench dir → feature-extractor-compatible dict."""
    proto_path = bench_dir / "benchmark.proto"
    runtime_path = bench_dir / "runtime_lengths.json"
    if not proto_path.is_file():
        raise FileNotFoundError(f"Missing {proto_path}")
    runtime_lengths: Dict[str, Dict[str, List[int]]] = {}
    if runtime_path.is_file():
        runtime_lengths = json.loads(runtime_path.read_text())

    analyzer = ProtobufAnalyzer(str(proto_path), workload_profile=profile)
    analysis = analyzer.analyze()

    all_top = _collect_all_messages(analysis.messages)

    # Walk every message (top + nested) and emit a record. This is the shape
    # extract_features_for_benchmark expects.
    memo: Dict[str, int] = {}
    messages_out: List[Dict[str, object]] = []

    def emit(msg: Message) -> None:
        size = _compute_message_size(msg, all_top, runtime_lengths, profile, memo=memo)
        messages_out.append({
            "name": msg.name,
            "depth": msg.depth,
            "total_fields": msg.total_fields,
            "nested_message_count": len(msg.nested_messages),
            "serialized_size_bytes": size,
        })
        for nm in msg.nested_messages:
            emit(nm)

    for top in analysis.messages:
        emit(top)

    total = sum(int(m["serialized_size_bytes"]) for m in messages_out
                if m["serialized_size_bytes"])

    return {
        "benchmark_name": bench_dir.name,
        "proto_file_path": str(proto_path),
        "syntax_version": analysis.syntax_version,
        "total_size_bytes": total,
        "top_level_messages": sorted(m.name for m in analysis.messages),
        "messages": messages_out,
        "statistics": {
            "total_messages": analysis.total_messages,
            "total_fields": analysis.total_fields,
            "max_nesting_depth": analysis.max_nesting_depth,
            "repeated_field_count": analysis.repeated_field_count,
            "enum_count": analysis.enum_count,
            "nested_message_count": analysis.nested_message_count,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-root", type=Path, required=True,
                    help="Root dir holding bench<N>/{benchmark.proto,runtime_lengths.json}.")
    ap.add_argument("--in-json", type=Path,
                    default=_THIS_DIR / "protobuf_analysis_verilator_bench.json",
                    help="Existing analysis JSON to extend (default: the one "
                         "next to this script). Pass a nonexistent path to "
                         "start fresh.")
    ap.add_argument("--out-json", type=Path,
                    default=_THIS_DIR / "protobuf_analysis_verilator_bench.json",
                    help="Where to write the merged analysis JSON.")
    ap.add_argument("--max-string-len", type=int, default=1024,
                    help="Cap per-field string/bytes payload length (default: "
                         "1024, matches proto_to_accel.py and the HPB feature "
                         "extraction profile).")
    ap.add_argument("--max-nested-depth", type=int, default=5,
                    help="Max submessage depth (default: 5, matches "
                         "proto_to_accel.py's DEFAULT_MAX_NESTED_DEPTH).")
    ap.add_argument("--skip-repeated", action="store_true", default=True,
                    help="Treat repeated fields as absent (default: on, "
                         "matches the Verilator bench profile).")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) // 2),
                    help="Parallel worker processes (default: half of nproc). "
                         "Each bench's analysis is independent — the loop is "
                         "trivially parallel and `ProtobufAnalyzer.analyze()` "
                         "dominates runtime on the 300+-message combined "
                         "protos synth benches produce. Pass 1 to analyze "
                         "serially (debugging / deterministic stdout order).")
    args = ap.parse_args()

    profile = WorkloadProfile(
        max_string_len=args.max_string_len,
        max_nested_depth=args.max_nested_depth,
        skip_repeated=args.skip_repeated,
    )

    merged: Dict[str, object] = {}
    if args.in_json.is_file():
        merged = json.loads(args.in_json.read_text())

    bench_dirs = sorted(
        (p for p in args.synth_root.glob("bench*") if p.is_dir()),
        key=lambda p: int(p.name.replace("bench", "")),
    )
    if not bench_dirs:
        sys.exit(f"No bench<N> dirs under {args.synth_root}")

    total = len(bench_dirs)
    if args.workers == 1:
        for i, bd in enumerate(bench_dirs, start=1):
            print(f"[{i}/{total}] analyzing {bd.name}", file=sys.stderr)
            merged[bd.name] = analyze_one_bench(bd, profile)
    else:
        # ProcessPool: each worker re-imports this module and runs
        # analyze_one_bench in isolation. No shared state across workers,
        # each bench reads its own benchmark.proto + runtime_lengths.json.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(analyze_one_bench, bd, profile): bd
                       for bd in bench_dirs}
            for i, fut in enumerate(as_completed(futures), start=1):
                bd = futures[fut]
                merged[bd.name] = fut.result()
                print(f"[{i}/{total}] analyzed {bd.name}", file=sys.stderr)

    args.out_json.write_text(json.dumps(merged, indent=2))
    print(f"Wrote {args.out_json} ({len(merged)} benches total)", file=sys.stderr)


if __name__ == "__main__":
    main()
