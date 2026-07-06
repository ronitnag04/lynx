#!/usr/bin/env python3
"""
Train and save multiple ML models for Lynx throughput prediction.

This script trains all the ML models from the comparison experiments and saves
them along with their scalers for later use in optimization and search tasks.
Supports both deserializer and serializer sides.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Constants from train.py
LABEL_COLUMN = "throughput_gbits_per_sec"
FALLBACK_LABEL_COLUMN = "throughput_bytes_per_sec"
FEATURE_PREFIX = "feat_"
SIDE_TO_NAME = {"des": "deserializer", "ser": "serializer"}

DES_PARAM_VALUES = [
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

SER_PARAM_VALUES = [
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


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(dataset_path)


def pre_process_dataset(
    dataset: pd.DataFrame,
    side: str,
    one_hot_bmark: bool,
    no_feat_distributions: bool = False,
    benchmark_categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Preprocess dataset using same logic as train.py."""
    side_name = SIDE_TO_NAME[side]

    if dataset.empty:
        raise ValueError(f"Dataset is empty for side '{side_name}'.")

    if LABEL_COLUMN in dataset.columns:
        label_column = LABEL_COLUMN
    elif FALLBACK_LABEL_COLUMN in dataset.columns:
        dataset = dataset.copy()
        dataset[LABEL_COLUMN] = dataset[FALLBACK_LABEL_COLUMN] * (8.0 / 1e9)
        label_column = LABEL_COLUMN
    else:
        raise ValueError(
            f"Input CSV is missing '{LABEL_COLUMN}' (or fallback '{FALLBACK_LABEL_COLUMN}')."
        )

    side_param_values = DES_PARAM_VALUES if side == "des" else SER_PARAM_VALUES
    missing_param_columns = [col for col in side_param_values if col not in dataset.columns]
    if missing_param_columns:
        raise ValueError(
            f"Input CSV is missing expected {side_name} config columns: {missing_param_columns}"
        )

    knob_columns = list(side_param_values)

    if one_hot_bmark:
        bmark_column = "bench"
        if benchmark_categories is not None:
            dataset = dataset.copy()
            dataset[bmark_column] = pd.Categorical(
                dataset[bmark_column], categories=list(benchmark_categories)
            )
        bmark_one_hot = pd.get_dummies(dataset[bmark_column])
        dataset = pd.concat([dataset, bmark_one_hot], axis=1)
        dataset = dataset.drop(columns=[bmark_column])
        feature_columns = list(bmark_one_hot.columns) + knob_columns
    else:
        analytical_columns = [col for col in dataset.columns if col.startswith(FEATURE_PREFIX)]
        if no_feat_distributions:
            analytical_columns = [
                col
                for col in analytical_columns
                if "_distribution_" not in col and "depth_counter_list_" not in col
            ]
        feature_columns = analytical_columns + knob_columns

    if not feature_columns:
        raise ValueError(
            f"No feature columns detected for side '{side_name}'. "
            f"Expected config columns from {side_name} params and/or '{FEATURE_PREFIX}'."
        )

    pruned = dataset.dropna(subset=feature_columns + [label_column]).copy()
    model_df = pruned[feature_columns + [label_column]].copy()
    return model_df.astype(float)


def split_features_and_labels(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and labels."""
    features = model_df.drop(columns=[LABEL_COLUMN])
    labels = model_df[LABEL_COLUMN]
    return features, labels


def get_ml_models() -> dict[str, Any]:
    """Return dictionary of ML models to train and save."""
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    }


def train_and_save_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    side_name: str,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """Train all ML models, save them, and return metrics."""
    print(f"\n{'='*80}")
    print(f"Training ML models for {side_name}")
    print(f"{'='*80}")
    print(f"Train set: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"Test set: {len(X_test)} samples")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler (shared for all models)
    scaler_path = output_dir / f"{side_name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler: {scaler_path}")

    models = get_ml_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = float(root_mean_squared_error(y_train, y_pred_train))
        test_rmse = float(root_mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Percent error
        eps = 1e-8
        train_percent_error = (
            (pd.Series(y_pred_train) - y_train).abs() / (y_train.abs() + eps)
        ).mean() * 100
        test_percent_error = (
            (pd.Series(y_pred_test, index=y_test.index) - y_test).abs() / (y_test.abs() + eps)
        ).mean() * 100

        results[name] = {
            'train_time': float(train_time),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_percent_error': float(train_percent_error),
            'test_percent_error': float(test_percent_error),
        }

        # Save the model
        model_path = output_dir / f"{side_name}_{name}_model.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved model: {model_path}")
        print(f"  Train Time: {train_time:.3f}s")
        print(f"  Test MAE: {test_mae:.4e} | Test R²: {test_r2:.6f} | Test % Error: {test_percent_error:.4f}%")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and save multiple ML models for Lynx throughput prediction."
    )
    parser.add_argument(
        "-d",
        "--dataset-path",
        default="data/training_data.csv",
        help="Path to enriched CSV produced by build_training_dataset.py",
    )
    parser.add_argument(
        "--side",
        choices=["des", "ser"],
        required=True,
        help="Which side to train on: des or ser.",
    )
    parser.add_argument(
        "--synth-only",
        action="store_true",
        help="Only use synthetic benchmarks for training and testing (drop hpb benchmarks)",
    )
    parser.add_argument(
        "--test-size",
        default=0.25,
        type=float,
        help="Fraction of the dataset to use for testing",
    )
    parser.add_argument(
        "--ood-benchmark",
        type=str,
        help="OOD benchmark to use for testing",
    )
    parser.add_argument(
        "--ood-train-size",
        type=int,
        help="Number of OOD data points to use in training set",
        default=0,
    )
    parser.add_argument(
        "--test-hpb",
        action="store_true",
        help="Use HPB data for testing",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for models and metrics files",
        default="results",
    )
    parser.add_argument(
        "--one-hot-bmark",
        action="store_true",
        help="Don't use features, only one-hot encode the benchmark",
    )
    parser.add_argument(
        "--no-feat-distributions",
        action="store_true",
        help="Don't use distributions in the feature columns, only the min/max/avg and other feature columns",
    )
    return parser.parse_args()


def train_neural_network(args: argparse.Namespace, side_name: str) -> None:
    """Train neural network using existing train.py script."""
    import subprocess

    print(f"\n{'='*80}")
    print(f"Training Neural Network for {side_name}")
    print(f"{'='*80}")

    train_cmd = [
        "python3", "train.py",
        "--side", args.side,
        "--dataset-path", args.dataset_path,
        "--output-dir", args.output_dir,
        "--test-size", str(args.test_size),
    ]

    if args.synth_only:
        train_cmd.append("--synth-only")
    if args.ood_benchmark:
        train_cmd.extend(["--ood-benchmark", args.ood_benchmark])
    if args.ood_train_size:
        train_cmd.extend(["--ood-train-size", str(args.ood_train_size)])
    if args.test_hpb:
        train_cmd.append("--test-hpb")
    if args.one_hot_bmark:
        train_cmd.append("--one-hot-bmark")
    if args.no_feat_distributions:
        train_cmd.append("--no-feat-distributions")

    print(f"Running: {' '.join(train_cmd)}")
    print()

    try:
        subprocess.run(train_cmd, check=True)
        print(f"\n✓ Neural network training complete")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Neural network training failed: {e}")
        raise


def main() -> None:
    """Main execution function."""
    args = parse_args()
    side_name = SIDE_TO_NAME[args.side]

    print(f"\n{'='*80}")
    print(f"TRAINING ALL MODELS FOR {side_name.upper()}")
    print(f"{'='*80}")

    print(f"\nLoading dataset from {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    print(f"Dataset size: {dataset.shape[0]}")

    if "op" not in dataset.columns:
        raise ValueError("Input CSV is missing required column 'op'.")
    side_dataset = dataset[dataset["side"] == args.side].copy()
    if side_dataset.empty:
        raise ValueError(f"No rows found for side '{side_name}' (side={args.side!r}).")

    if args.synth_only:
        side_dataset = side_dataset[~side_dataset["bench"].isin([f"bench{i}" for i in range(6)])]
        if side_dataset.empty:
            raise ValueError(f"No rows found for side '{side_name}' after filtering out HPB benchmarks for synth-only mode.")

    benchmark_categories = None
    if args.one_hot_bmark:
        benchmark_categories = sorted(side_dataset["bench"].dropna().unique().tolist())

    # Split data
    if args.ood_benchmark:
        test_df = side_dataset[side_dataset["bench"] == args.ood_benchmark]
        train_df = side_dataset[side_dataset["bench"] != args.ood_benchmark]
        if args.ood_train_size > 0:
            ood_train_df = test_df.sample(args.ood_train_size, random_state=42)
            train_df = pd.concat([train_df, ood_train_df])
            test_df = test_df.drop(ood_train_df.index)
    elif args.test_hpb:
        hpb_benches = [f"bench{i}" for i in range(0, 6)]
        mask = side_dataset["bench"].isin(hpb_benches)
        test_df = side_dataset[mask]
        train_df = side_dataset[~mask]
    else:
        train_df, test_df = train_test_split(
            side_dataset,
            test_size=args.test_size,
            random_state=42,
        )

    # Preprocess
    train_df = pre_process_dataset(
        train_df,
        args.side,
        args.one_hot_bmark,
        no_feat_distributions=args.no_feat_distributions,
        benchmark_categories=benchmark_categories,
    )
    test_df = pre_process_dataset(
        test_df,
        args.side,
        args.one_hot_bmark,
        no_feat_distributions=args.no_feat_distributions,
        benchmark_categories=benchmark_categories,
    )

    train_features, train_labels = split_features_and_labels(train_df)
    test_features, test_labels = split_features_and_labels(test_df)

    print(
        f"Final split: {len(train_features)} train, {len(test_features)} test "
        f"({len(test_features)/(len(train_features)+len(test_features)):.2%} test)"
    )

    if args.one_hot_bmark:
        print("WARNING: NO analytical model features are being used, using one-hot encoding benchmarks instead.")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train and save sklearn models
    results = train_and_save_models(
        train_features,
        test_features,
        train_labels,
        test_labels,
        side_name,
        output_dir,
    )

    # Save metrics
    metrics_path = output_dir / f"{side_name}_all_models_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_path": args.dataset_path,
                "side": args.side,
                "side_name": side_name,
                "test_size": args.test_size,
                "num_features": int(train_features.shape[1]),
                "num_train_samples": int(len(train_features)),
                "num_test_samples": int(len(test_features)),
                "sklearn_models": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved sklearn metrics: {metrics_path}")

    # Save args
    args_path = output_dir / f"{side_name}_all_models_input_args.txt"
    with open(args_path, "w", encoding="utf-8") as f:
        for key in sorted(vars(args)):
            f.write(f"{key}={getattr(args, key)}\n")

    # Train neural network
    train_neural_network(args, side_name)

    print(f"\n{'='*80}")
    print(f"ALL TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"All models saved to: {output_dir}/")
    print(f"  - Sklearn models: {side_name}_<ModelName>_model.joblib")
    print(f"  - Neural network: {side_name}_checkpoint.pt")
    print(f"  - Scaler: {side_name}_scaler.joblib")
    print(f"  - Sklearn metrics: {side_name}_all_models_metrics.json")
    print(f"  - Neural metrics: {side_name}_metrics.json")


if __name__ == "__main__":
    main()
