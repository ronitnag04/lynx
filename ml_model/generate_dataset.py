"""Utility script to generate synthetic data for training Lynx ML model.

The goal is to sanity-check the Lynx ML model and its training pipeline by 
fitting it on a hand-crafted nonlinear function with light noise. The script 
emits the generated `.npy` files so they can be used with the train.py script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from model import INPUT_SIZE, OUTPUT_SIZE

assert OUTPUT_SIZE == 1, "Output size must be 1 for current data generation setup."

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for Lynx ML model on a complex target."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="Total synthetic samples to generate.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.75,
        help="Fraction of samples assigned to the train set (rest for test). \
         The user must account for batch_size when setting this value. \
         Neuron/XLA compilation requires the batch dimension to be a multiple \
         of the batch_size to avoid errors with the trailing iteration having a different shape.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Std. dev. for Gaussian label noise to make the problem realistic.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_suffix("").parent / "data",
        help="Directory where .npy files will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="RNG seed for repeatability.",
    )
    return parser.parse_args()


def complex_function(features: np.ndarray) -> np.ndarray:
    """Nonlinear target leveraging multiple interactions."""

    first = np.sin(features[:, :32]).sum(axis=1)
    second = np.cos(features[:, 32:64] * 1.5).sum(axis=1)
    third = np.tanh(features[:, 64:96] * np.sqrt(2)).sum(axis=1)
    poly = np.square(features[:, 96:]).sum(axis=1)
    cross = (features[:, :32] * features[:, 32:64]).sum(axis=1)

    return 0.35 * first + 0.25 * second + 0.2 * third + 0.15 * poly + 0.05 * cross


def generate_dataset(samples: int, noise_std: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(
        loc=0.0,
        scale=0.8,
        size=(samples, INPUT_SIZE),
    ).astype(np.float32)
    targets = complex_function(features)
    noise = rng.normal(loc=0.0, scale=noise_std, size=targets.shape)
    labels = (targets + noise).astype(np.float32)
    return features, labels[:, np.newaxis]


def split_dataset(features: np.ndarray, labels: np.ndarray, train_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_idx = int(len(features) * train_split)
    train_feats, test_feats = features[:split_idx], features[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    return train_feats, train_labels, test_feats, test_labels


def save_numpy_arrays(output_dir: Path, train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray, test_labels: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_features_path = output_dir / "train_features.npy"
    train_labels_path = output_dir / "train_labels.npy"
    test_features_path = output_dir / "test_features.npy"
    test_labels_path = output_dir / "test_labels.npy"

    np.save(train_features_path, train_features)
    np.save(train_labels_path, train_labels)
    np.save(test_features_path, test_features)
    np.save(test_labels_path, test_labels)

    print(
        f"Wrote {train_features_path}, {train_labels_path}, "
        f"{test_features_path}, {test_labels_path}"
    )


def main() -> None:
    args = parse_args()
    features, labels = generate_dataset(args.samples, args.noise_std, args.seed)
    train_feats, train_labels, test_feats, test_labels = split_dataset(features, labels, args.train_split)
    save_numpy_arrays(args.output_dir, train_feats, train_labels, test_feats, test_labels)


if __name__ == "__main__":
    main()


