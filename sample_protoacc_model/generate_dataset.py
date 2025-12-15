"""Generate dataset for Lynx ML model from protobuf model throughput estimates.

This script:
1. Generates parameter vectors from parameter sweep
2. Estimates throughput for each parameter vector + benchmark combination
3. Combines parameter vectors with benchmark features as ML features
4. Splits into train/test sets
5. Saves as .npy files for serializer and deserializer
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from protobuf_model import (
    estimated_throughput,
    serializer_parameter_sweep,
    deserializer_parameter_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset from protobuf model throughput estimates."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="Number of random parameter combinations to generate per operation type.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.75,
        help="Fraction of samples assigned to the train set (rest for test).",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path(__file__).parent.parent / "analytical_model" / "extracted_features.json",
        help="Path to extracted_features.json file.",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=Path(__file__).parent.parent / "ml_model" / "data",
        help="Base directory where serializer_dataset and deserializer_dataset folders will be created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="RNG seed for repeatability.",
    )
    return parser.parse_args()


def generate_parameter_vectors(parameter_sweep: Dict[str, List], max_combinations: int = 1000) -> List[Dict]:
    """Generate random parameter vectors from parameter sweep.
    
    Similar to generate_random_parameter_vectors - randomly selects parameter values
    and ensures uniqueness using a set.
    
    Args:
        parameter_sweep: Dictionary mapping parameter names to lists of possible values
        max_combinations: Number of random parameter combinations to generate
    
    Returns:
        List of dictionaries, each containing a random parameter combination
    """
    parameter_vectors_set = set()
    while len(parameter_vectors_set) < max_combinations:
        param_vector = {}
        for param_name, param_values in parameter_sweep.items():
            param_vector[param_name] = random.choice(param_values)
        # Use tuple of items to ensure uniqueness in set
        parameter_vectors_set.add(tuple(param_vector.items()))
    
    # Convert set of tuples back to list of dictionaries
    return [dict(param_vector) for param_vector in parameter_vectors_set]


def flatten_benchmark_features(benchmark_features: Dict) -> np.ndarray:
    """Flatten benchmark features into a feature vector.
    
    Args:
        benchmark_features: Dictionary containing benchmark features
    
    Returns:
        Flattened feature vector as numpy array
    """
    features = []
    
    # Scalar features
    features.append(benchmark_features["total_size_bytes"])
    features.append(benchmark_features["num_messages"])
    features.append(benchmark_features["min_size_bytes"])
    features.append(benchmark_features["max_size_bytes"])
    features.append(benchmark_features["avg_size_bytes"])
    features.append(benchmark_features["min_total_fields"])
    features.append(benchmark_features["max_total_fields"])
    features.append(benchmark_features["avg_total_fields"])
    features.append(benchmark_features["min_nested_message_count"])
    features.append(benchmark_features["max_nested_message_count"])
    features.append(benchmark_features["avg_nested_message_count"])
    features.append(benchmark_features["min_depth"])
    features.append(benchmark_features["max_depth"])
    features.append(benchmark_features["avg_depth"])
    
    # Distribution arrays (flatten)
    features.extend(benchmark_features["size_bytes_distribution"])
    features.extend(benchmark_features["total_fields_distribution"])
    features.extend(benchmark_features["nested_message_count_distribution"])
    features.extend(benchmark_features["depth_counter_list"])
    
    return np.array(features, dtype=np.float32)


def flatten_parameter_vector(parameter_vector: Dict, param_order: List[str]) -> np.ndarray:
    """Flatten parameter vector into a feature vector.
    
    Args:
        parameter_vector: Dictionary containing parameter values
        param_order: Ordered list of parameter names
    
    Returns:
        Flattened parameter vector as numpy array
    """
    return np.array([parameter_vector[name] for name in param_order], dtype=np.float32)


def generate_dataset_for_op_type(
    op_type: str,
    parameter_sweep: Dict[str, List],
    benchmark_features: Dict,
    benchmarks: List[str],
    max_combinations: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dataset for a specific operation type (serializer or deserializer).
    
    Args:
        op_type: "serializer" or "deserializer"
        parameter_sweep: Parameter sweep dictionary for this operation type
        benchmark_features: Dictionary mapping benchmark names to their features
        benchmarks: List of benchmark names
        max_combinations: Maximum number of parameter combinations
    
    Returns:
        Tuple of (features, labels) where features is (N, feature_dim) and labels is (N, 1)
    """
    # Generate parameter vectors
    parameter_vectors = generate_parameter_vectors(parameter_sweep, max_combinations)
    param_order = list(parameter_sweep.keys())
    
    # Get parameter vector dimension
    param_dim = len(param_order)
    
    # Get benchmark feature dimension (from first benchmark)
    first_benchmark = benchmarks[0]
    benchmark_feat_dim = len(flatten_benchmark_features(benchmark_features[first_benchmark]))
    
    # Total feature dimension
    feature_dim = param_dim + benchmark_feat_dim
    
    # Collect all features and labels
    all_features = []
    all_labels = []
    
    for benchmark_name in benchmarks:
        # Flatten benchmark features
        bench_features = flatten_benchmark_features(benchmark_features[benchmark_name])
        for param_vector in parameter_vectors:
            # Estimate throughput
            parameters = {op_type: param_vector}
            throughput = estimated_throughput(parameters, benchmark_name)
            
            # Flatten features
            param_features = flatten_parameter_vector(param_vector, param_order)
            combined_features = np.concatenate([param_features, bench_features])
            
            all_features.append(combined_features)
            all_labels.append(throughput)
    
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)[:, np.newaxis]
    
    return features, labels


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    train_split: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train and test sets.
    
    Args:
        features: Feature array of shape (N, feature_dim)
        labels: Label array of shape (N, 1)
        train_split: Fraction of samples for training
    
    Returns:
        Tuple of (train_features, train_labels, test_features, test_labels)
    """
    split_idx = int(len(features) * train_split)
    train_feats = features[:split_idx]
    train_labels = labels[:split_idx]
    test_feats = features[split_idx:]
    test_labels = labels[split_idx:]
    return train_feats, train_labels, test_feats, test_labels


def save_numpy_arrays(
    output_dir: Path,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    """Save numpy arrays to .npy files.
    
    Args:
        output_dir: Directory to save files in
        train_features: Training features array
        train_labels: Training labels array
        test_features: Test features array
        test_labels: Test labels array
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_features_path = output_dir / "train_features.npy"
    train_labels_path = output_dir / "train_labels.npy"
    test_features_path = output_dir / "test_features.npy"
    test_labels_path = output_dir / "test_labels.npy"
    
    np.save(train_features_path, train_features)
    np.save(train_labels_path, train_labels)
    np.save(test_features_path, test_features)
    np.save(test_labels_path, test_labels)
    
    print(f"Saved to {output_dir}:")
    print(f"  - {train_features_path} (shape: {train_features.shape})")
    print(f"  - {train_labels_path} (shape: {train_labels.shape})")
    print(f"  - {test_features_path} (shape: {test_features.shape})")
    print(f"  - {test_labels_path} (shape: {test_labels.shape})")


def main() -> None:
    args = parse_args()
    
    # Set random seed for both numpy and Python's random module
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load benchmark features
    with open(args.features_file, "r") as f:
        benchmark_features = json.load(f)
    
    benchmarks = list(benchmark_features.keys())
    print(f"Found benchmarks: {benchmarks}")
    
    # Generate dataset for deserializer
    print("\nGenerating deserializer dataset...")
    deserializer_features, deserializer_labels = generate_dataset_for_op_type(
        "deserializer",
        deserializer_parameter_sweep,
        benchmark_features,
        benchmarks,
        args.samples,
    )
    print(f"Deserializer dataset: {deserializer_features.shape[0]} samples, {deserializer_features.shape[1]} features")
    
    # Split deserializer dataset
    deser_train_feats, deser_train_labels, deser_test_feats, deser_test_labels = split_dataset(
        deserializer_features,
        deserializer_labels,
        args.train_split,
    )
    
    # Save deserializer dataset
    deserializer_output_dir = args.output_base_dir / "deserializer_dataset"
    save_numpy_arrays(
        deserializer_output_dir,
        deser_train_feats,
        deser_train_labels,
        deser_test_feats,
        deser_test_labels,
    )
    
    # Generate dataset for serializer
    print("\nGenerating serializer dataset...")
    serializer_features, serializer_labels = generate_dataset_for_op_type(
        "serializer",
        serializer_parameter_sweep,
        benchmark_features,
        benchmarks,
        args.samples,
    )
    print(f"Serializer dataset: {serializer_features.shape[0]} samples, {serializer_features.shape[1]} features")
    
    # Split serializer dataset
    ser_train_feats, ser_train_labels, ser_test_feats, ser_test_labels = split_dataset(
        serializer_features,
        serializer_labels,
        args.train_split,
    )
    
    # Save serializer dataset
    serializer_output_dir = args.output_base_dir / "serializer_dataset"
    save_numpy_arrays(
        serializer_output_dir,
        ser_train_feats,
        ser_train_labels,
        ser_test_feats,
        ser_test_labels,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

