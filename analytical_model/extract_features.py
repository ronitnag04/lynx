#!/usr/bin/env python3
"""
Feature Extraction Script for Protobuf Analysis

This script reads protobuf_analysis.json and extracts features for ML model training:
- total_size_bytes for each benchmark
- For each message (including nested): size_bytes, depth, total_fields, nested_message_count
- Converts message features into 10-point frequency distributions

A WorkloadProfile can be applied at this stage to clamp the full HyperProtoBench
payloads down to the subset the Verilator bench generator actually exercises
(see verilator-bench/proto_to_accel.py). We do the clamping here (rather than
inside protobuf_analyzer.py) so the analyzer output stays a faithful, profile-
independent description of the protos + their runtime Set_F1 data, and each
feature-set variant is just a different post-processing pass.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional


# Defaults mirror verilator-bench/proto_to_accel.py:
#   DEFAULT_MAX_NESTED_DEPTH = 5
#   DEFAULT_MAX_STRING_LEN   = 1024
# and the generator's baked-in decision to skip all repeated fields.
VERILATOR_MAX_STRING_LEN = 1024
VERILATOR_MAX_NESTED_DEPTH = 5
VERILATOR_SKIP_REPEATED = True


@dataclass
class WorkloadProfile:
    """Caps that shrink the HPB payload to what the Verilator bench runs.

    - ``max_string_len``: clamp per-value string/bytes payload length (bytes).
      Repeated string/bytes fields are left unclamped (their per-element sizes
      aren't preserved in protobuf_analysis.json); use ``skip_repeated`` to
      drop them instead.
    - ``max_nested_depth``: messages at schema depth > this are removed from
      the messages list, and any nested-message field pointing to a removed
      child contributes 0 to the parent's size. Matches the generator's
      "depth >= max_nested_depth => do not instantiate child" cutoff.
    - ``skip_repeated``: drop all repeated fields (size and field count).
    """
    max_string_len: Optional[int] = None
    max_nested_depth: Optional[int] = None
    skip_repeated: bool = False

    @staticmethod
    def verilator_bench_default() -> "WorkloadProfile":
        return WorkloadProfile(
            max_string_len=VERILATOR_MAX_STRING_LEN,
            max_nested_depth=VERILATOR_MAX_NESTED_DEPTH,
            skip_repeated=VERILATOR_SKIP_REPEATED,
        )

    def is_noop(self) -> bool:
        return (self.max_string_len is None
                and self.max_nested_depth is None
                and not self.skip_repeated)


def _varint_size(n: int) -> int:
    """Number of bytes a non-negative integer takes as a protobuf varint."""
    n = max(0, n)
    size = 1
    while n >= 128:
        n >>= 7
        size += 1
    return size


def _tag_size(field_number: int) -> int:
    # Tag = (field_number << 3) | wire_type; use max wire_type (5) for worst case.
    return _varint_size((field_number << 3) | 5)


def _clamped_string_bound(field_number: int, max_len: int) -> int:
    """Upper bound on a single string/bytes field's serialized size after clamp."""
    return _tag_size(field_number) + _varint_size(max_len) + max_len


def apply_workload_profile(benchmark_data: Dict[str, Any],
                           profile: WorkloadProfile) -> Dict[str, Any]:
    """Return a new benchmark_data dict with the profile caps applied.

    Recomputes each surviving message's ``serialized_size_bytes`` from its
    (clamped) field sizes, because clamping strings or dropping a deeply-nested
    child changes the wire-format length bubbled up through every ancestor.
    Also rewrites ``total_fields``, ``nested_message_count``, ``has_repeated_fields``,
    ``has_nested_messages``, and ``nested_messages`` on each message to match
    what survived, plus the benchmark-wide ``total_size_bytes`` (kept as the
    sum of all surviving messages' serialized sizes, matching the analyzer's
    original semantic so the ratio against ``size_bytes_distribution`` is
    preserved).

    Message names are assumed unique within a single benchmark's ``messages``
    list (verified true for all HPB benches in protobuf_analysis.json).
    """
    if profile.is_noop():
        return benchmark_data

    original_messages = benchmark_data.get("messages", [])
    if not original_messages:
        return benchmark_data

    by_name = {m["name"]: m for m in original_messages}
    max_depth = profile.max_nested_depth

    def child_excluded(child_name: Optional[str]) -> bool:
        if child_name is None:
            return True
        child = by_name.get(child_name)
        if child is None:
            return True
        return max_depth is not None and child["depth"] > max_depth

    def field_survives(f: Dict[str, Any]) -> bool:
        if profile.skip_repeated and f["cardinality"] == "repeated":
            return False
        if f["is_nested_message"] and child_excluded(f["nested_message_name"]):
            return False
        return True

    # Memoized per-message recompute. Returns raw payload size (no outer
    # tag/length); None if the message is excluded at its schema depth.
    cache: Dict[str, Optional[int]] = {}

    def recompute(msg: Dict[str, Any]) -> Optional[int]:
        name = msg["name"]
        if name in cache:
            return cache[name]
        if max_depth is not None and msg["depth"] > max_depth:
            cache[name] = None
            return None
        total = 0
        for f in msg["fields"]:
            if profile.skip_repeated and f["cardinality"] == "repeated":
                continue
            if f["is_nested_message"]:
                child_name = f["nested_message_name"]
                if child_excluded(child_name):
                    continue
                child_size = recompute(by_name[child_name])
                if child_size is None:
                    continue
                total += (_tag_size(f["field_number"])
                          + _varint_size(child_size)
                          + child_size)
            else:
                sz = f.get("size_bytes")
                if sz is None:
                    continue
                if (profile.max_string_len is not None
                        and f["field_type"] in ("string", "bytes")
                        and f["cardinality"] != "repeated"):
                    sz = min(sz, _clamped_string_bound(f["field_number"],
                                                      profile.max_string_len))
                total += sz
        cache[name] = total
        return total

    new_messages: List[Dict[str, Any]] = []
    for m in original_messages:
        new_size = recompute(m)
        if new_size is None:
            continue

        surviving_fields = [f for f in m["fields"] if field_survives(f)]
        surviving_children = [
            c for c in m.get("nested_messages", [])
            if not child_excluded(c)
        ]

        new_m = dict(m)
        new_m["serialized_size_bytes"] = new_size
        new_m["total_fields"] = len(surviving_fields)
        new_m["nested_message_count"] = len(surviving_children)
        new_m["nested_messages"] = surviving_children
        new_m["has_nested_messages"] = bool(surviving_children)
        new_m["has_repeated_fields"] = any(f["cardinality"] == "repeated"
                                           for f in surviving_fields)
        new_messages.append(new_m)

    new_bench = dict(benchmark_data)
    new_bench["messages"] = new_messages
    new_bench["total_size_bytes"] = sum(m["serialized_size_bytes"] for m in new_messages)
    return new_bench


def create_frequency_distribution(values: List[float], num_bins: int = 10) -> List[float]:
    """
    Create a normalized frequency distribution (histogram) with num_bins bins.
    Returns a list of frequencies normalized to sum to 1.0.
    """
    # Create histogram
    hist, bin_edges = np.histogram(values, bins=num_bins)

    frequencies = hist / len(values)

    return frequencies.tolist()

def create_counter_list(values: List[float], num_bins: int = 10) -> List[int]:
    """
    Create a counter list from a list of values.
    Returns a list of counts for each value.
    """
    values = np.array(values)
    counts = np.bincount(values)

    if len(counts) < num_bins:
        counts = np.pad(counts, (0, num_bins - len(counts)), 'constant', constant_values=0)
    elif len(counts) > num_bins:
        counts[num_bins-1] += sum(counts[num_bins:])
        counts = counts[:num_bins]

    return counts.tolist()

def extract_features_for_benchmark(benchmark_data: Dict[str, Any],
                                   depth_counter_bins: int = 15) -> Dict[str, Any]:
    """
    Extract features for a single benchmark.
    Returns a dictionary with total_size_bytes and frequency distributions.
    """
    # Get total_size_bytes
    total_size_bytes = benchmark_data.get("total_size_bytes", 0)

    # Collect all messages (including nested)
    all_messages = benchmark_data.get("messages", [])

    # Extract feature lists
    size_bytes_list = [msg["serialized_size_bytes"] for msg in all_messages]
    depth_list = [msg["depth"] for msg in all_messages]
    total_fields_list = [msg["total_fields"] for msg in all_messages]
    nested_message_count_list = [msg["nested_message_count"] for msg in all_messages]

    # Create 10-point frequency distributions
    size_bytes_dist = create_frequency_distribution(size_bytes_list, num_bins=10)
    total_fields_dist = create_frequency_distribution(total_fields_list, num_bins=10)
    nested_message_count_dist = create_frequency_distribution(nested_message_count_list, num_bins=10)

    depth_counter_list = create_counter_list(depth_list, num_bins=depth_counter_bins)

    # Add summary statistics
    min_size_bytes = min(size_bytes_list)
    max_size_bytes = max(size_bytes_list)
    avg_size_bytes = sum(size_bytes_list) / len(size_bytes_list)
    min_total_fields = min(total_fields_list)
    max_total_fields = max(total_fields_list)
    avg_total_fields = sum(total_fields_list) / len(total_fields_list)
    min_nested_message_count = min(nested_message_count_list)
    max_nested_message_count = max(nested_message_count_list)
    avg_nested_message_count = sum(nested_message_count_list) / len(nested_message_count_list)
    min_depth = min(depth_list)
    max_depth = max(depth_list)
    avg_depth = sum(depth_list) / len(depth_list)


    return {
        "total_size_bytes": total_size_bytes,
        "num_messages": len(all_messages),

        "min_size_bytes": min_size_bytes,
        "max_size_bytes": max_size_bytes,
        "avg_size_bytes": avg_size_bytes,
        "size_bytes_distribution": size_bytes_dist,

        "min_total_fields": min_total_fields,
        "max_total_fields": max_total_fields,
        "avg_total_fields": avg_total_fields,
        "total_fields_distribution": total_fields_dist,

        "min_nested_message_count": min_nested_message_count,
        "max_nested_message_count": max_nested_message_count,
        "avg_nested_message_count": avg_nested_message_count,
        "nested_message_count_distribution": nested_message_count_dist,

        "min_depth": min_depth,
        "max_depth": max_depth,
        "avg_depth": avg_depth,
        "depth_counter_list": depth_counter_list,
    }


def main():
    """Extract features from protobuf_analysis.json, optionally applying a
    workload profile that clamps the HPB payload to the Verilator bench scope.

    Typical usage:

        # Raw analyzer output for full-HPB features:
        python3 protobuf_analyzer.py
        python3 extract_features.py --output extracted_features_full.json

        # Verilator-bench-scoped features (matches sims/verilator measurements):
        python3 extract_features.py --verilator-bench-profile \
                                    --output extracted_features.json
    """
    import argparse

    script_dir = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=script_dir / "protobuf_analysis.json",
                    help="Analyzer output JSON (default: protobuf_analysis.json).")
    ap.add_argument("--output", type=Path,
                    default=script_dir / "extracted_features.json",
                    help="Feature JSON to write (default: extracted_features.json).")
    ap.add_argument("--verilator-bench-profile", action="store_true",
                    help=(f"Apply the Verilator bench caps "
                          f"(skip_repeated=True, "
                          f"max_string_len={VERILATOR_MAX_STRING_LEN}, "
                          f"max_nested_depth={VERILATOR_MAX_NESTED_DEPTH})."))
    ap.add_argument("--max-string-len", type=int, default=None,
                    help=("Cap single-value string/bytes payload length. "
                          "Overrides --verilator-bench-profile."))
    ap.add_argument("--max-nested-depth", type=int, default=None,
                    help=("Drop messages at schema depth > this. "
                          "Overrides --verilator-bench-profile."))
    ap.add_argument("--skip-repeated", action="store_true",
                    help=("Drop repeated fields entirely. "
                          "Overrides --verilator-bench-profile."))
    args = ap.parse_args()

    profile = (WorkloadProfile.verilator_bench_default()
               if args.verilator_bench_profile else WorkloadProfile())
    if args.max_string_len is not None:
        profile.max_string_len = args.max_string_len
    if args.max_nested_depth is not None:
        profile.max_nested_depth = args.max_nested_depth
    if args.skip_repeated:
        profile.skip_repeated = True

    if not args.input.exists():
        raise FileNotFoundError(f"Could not find {args.input}")

    print(f"Reading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)

    print(f"Workload profile: max_string_len={profile.max_string_len} "
          f"max_nested_depth={profile.max_nested_depth} "
          f"skip_repeated={profile.skip_repeated}")

    # Shrink depth_counter_list to 6 bins (depths 0..5) when the Verilator
    # profile's nesting cap is active; otherwise keep the original 15 bins.
    depth_counter_bins = (profile.max_nested_depth + 1
                          if profile.max_nested_depth is not None
                          else 15)

    features = {}
    for benchmark_name, benchmark_data in data.items():
        print(f"Processing {benchmark_name}...")
        clamped = apply_workload_profile(benchmark_data, profile)
        features[benchmark_name] = extract_features_for_benchmark(
            clamped, depth_counter_bins=depth_counter_bins)

    print(f"Writing features to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(features, f, indent=2)

    print(f"Successfully extracted features for {len(features)} benchmarks")
    print(f"Features saved to {args.output}")


if __name__ == "__main__":
    main()
