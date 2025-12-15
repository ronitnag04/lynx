#!/usr/bin/env python3
"""
Feature Extraction Script for Protobuf Analysis

This script reads protobuf_analysis.json and extracts features for ML model training:
- total_size_bytes for each benchmark
- For each message (including nested): size_bytes, depth, total_fields, nested_message_count
- Converts message features into 100-point frequency distributions
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

def create_frequency_distribution(values: List[float], num_bins: int = 100) -> List[float]:
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
        counts[-1] += sum(counts[num_bins:])

    return counts.tolist()

def extract_features_for_benchmark(benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
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
    min_size_bytes = min(size_bytes_list)
    max_size_bytes = max(size_bytes_list)
    size_bytes_dist = create_frequency_distribution(size_bytes_list, num_bins=10)
    min_total_fields = min(total_fields_list)
    max_total_fields = max(total_fields_list)
    total_fields_dist = create_frequency_distribution(total_fields_list, num_bins=10)
    min_nested_message_count = min(nested_message_count_list)
    max_nested_message_count = max(nested_message_count_list)
    nested_message_count_dist = create_frequency_distribution(nested_message_count_list, num_bins=10)

    # Create 10-point counter lists
    depth_counter_list = create_counter_list(depth_list, num_bins=10)

    return {
        "total_size_bytes": total_size_bytes,
        "num_messages": len(all_messages),
        "min_size_bytes": min_size_bytes,
        "max_size_bytes": max_size_bytes,
        "size_bytes_distribution": size_bytes_dist,
        "min_total_fields": min_total_fields,
        "max_total_fields": max_total_fields,
        "total_fields_distribution": total_fields_dist,
        "min_nested_message_count": min_nested_message_count,
        "max_nested_message_count": max_nested_message_count,
        "nested_message_count_distribution": nested_message_count_dist,
        "depth_counter_list": depth_counter_list,
    }


def main():
    """Main function to extract features from protobuf_analysis.json"""
    # Path to the analysis JSON file
    script_dir = Path(__file__).parent
    json_path = script_dir / "protobuf_analysis.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Could not find {json_path}")
    
    # Read the JSON file
    print(f"Reading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract features for each benchmark
    features = {}
    for benchmark_name, benchmark_data in data.items():
        print(f"Processing {benchmark_name}...")
        features[benchmark_name] = extract_features_for_benchmark(benchmark_data)
    
    # Output the features
    output_path = script_dir / "extracted_features.json"
    print(f"Writing features to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
    
    print(f"Successfully extracted features for {len(features)} benchmarks")
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()
