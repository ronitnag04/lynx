from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterable

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import LynxMLModel

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
    return pd.read_csv(dataset_path)


def pre_process_dataset(dataset: pd.DataFrame, side: str) -> pd.DataFrame:
    side_name = SIDE_TO_NAME[side]

    if dataset.empty:
        raise ValueError(f"Dataset is empty for side '{side_name}'.")

    if LABEL_COLUMN in dataset.columns:
        label_column = LABEL_COLUMN
    elif FALLBACK_LABEL_COLUMN in dataset.columns:
        # Backward-compatible fallback if the dataset still stores bytes/s.
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
    analytical_columns = [col for col in dataset.columns if col.startswith(FEATURE_PREFIX)]
    feature_columns = knob_columns + analytical_columns
    if not feature_columns:
        raise ValueError(
            f"No feature columns detected for side '{side_name}'. "
            f"Expected config columns from {side_name} params and/or '{FEATURE_PREFIX}'."
        )

    pruned = dataset.dropna(subset=feature_columns + [label_column]).copy()
    model_df = pruned[feature_columns + [label_column]].copy()
    return model_df.astype(float)


def split_features_and_labels(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = model_df.drop(columns=[LABEL_COLUMN])
    labels = model_df[LABEL_COLUMN]
    return features, labels


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        torch_xla.sync()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_percent_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            torch_xla.sync()
            total_loss += loss.item()
            eps = 1e-8
            batch_percent_error = (
                (torch.abs(outputs - targets) / (torch.abs(targets) + eps)).mean() * 100.0
            )
            total_percent_error += batch_percent_error.item()
            num_batches += 1

    denom = max(num_batches, 1)
    avg_loss = total_loss / denom
    avg_percent_error = total_percent_error / denom
    return float(avg_loss), float(avg_percent_error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Lynx ML model using PyTorch and XLA with the AWS Neuron SDK.")
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
        "-o",
        "--output-dir",
        help="Output directory for checkpoint and metrics files",
        default="training_results",
    )
    parser.add_argument(
        "--add-final-predictions",
        action="store_true",
        help="Write a CSV with actual/predicted throughput plus config_name and bench for the final test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    side_name = SIDE_TO_NAME[args.side]

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    print(f"Dataset size: {dataset.shape[0]}")

    if "op" not in dataset.columns:
        raise ValueError("Input CSV is missing required column 'op'.")
    side_dataset = dataset[dataset["op"] == args.side].copy()
    if side_dataset.empty:
        raise ValueError(f"No rows found for side '{side_name}' (op={args.side!r}).")

    if args.ood_benchmark:
        test_df = side_dataset[side_dataset["bench"] == args.ood_benchmark]
        train_df = side_dataset[side_dataset["bench"] != args.ood_benchmark]
        if args.ood_train_size > 0:
            ood_train_df = test_df.sample(args.ood_train_size, random_state=42)
            train_df = pd.concat([train_df, ood_train_df])
            test_df = test_df.drop(ood_train_df.index)
    else:
        train_df, test_df = train_test_split(
            side_dataset,
            test_size=args.test_size,
            random_state=42,
        )

    train_df = pre_process_dataset(train_df, args.side)
    test_df = pre_process_dataset(test_df, args.side)
    train_features, train_labels = split_features_and_labels(train_df)
    test_features, test_labels = split_features_and_labels(test_df)
    print(
        f"Final split: {len(train_features)} train, {len(test_features)} test "
        f"({len(test_features)/(len(train_features)+len(test_features)):.2%} test)"
    )
    print(f"Train dataset shape: {train_features.shape}")
    print(f"Test dataset shape: {test_features.shape}")

    scaler = StandardScaler()
    train_features = train_features.copy().astype(float)
    test_features = test_features.copy().astype(float)
    train_features[train_features.columns] = scaler.fit_transform(train_features[train_features.columns])
    test_features[train_features.columns] = scaler.transform(test_features[train_features.columns])

    train_features_t = torch.from_numpy(train_features.to_numpy(copy=True)).float()
    train_labels_t = torch.from_numpy(train_labels.to_numpy(copy=True)).float().unsqueeze(1)
    test_features_t = torch.from_numpy(test_features.to_numpy(copy=True)).float()
    test_labels_t = torch.from_numpy(test_labels.to_numpy(copy=True)).float().unsqueeze(1)

    train_ds = TensorDataset(train_features_t, train_labels_t)
    test_ds = TensorDataset(test_features_t, test_labels_t)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = "xla"
    epochs = 200
    model = LynxMLModel(input_size=train_features.shape[1], hidden_dims=[256, 128], output_size=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.L1Loss()

    early_stop_patience = 10
    best_eval_loss = float("inf")
    epochs_without_improvement = 0

    print("----------- Start Training --------------")
    epochs_data: list[dict[str, float]] = []
    total_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        eval_loss, percent_error = evaluate(model, test_loader, loss_fn, device)
        epoch_duration = time.perf_counter() - epoch_start
        epochs_data.append(
            {
                "duration": epoch_duration,
                "train_loss": float(train_loss),
                "eval_loss": float(eval_loss),
                "percent_error": float(percent_error),
            }
        )
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Percent Error: {percent_error:.2f}% | "
            f"Time: {epoch_duration:.2f}s"
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping: no improvement for {early_stop_patience} epochs.")
                break

    print("------------ End Training ---------------")
    total_duration = time.perf_counter() - total_start

    os.makedirs(args.output_dir, exist_ok=True)

    args_path = os.path.join(args.output_dir, f"{side_name}_input_args.txt")
    with open(args_path, "w", encoding="utf-8") as f:
        for key in sorted(vars(args)):
            f.write(f"{key}={getattr(args, key)}\n")

    checkpoint_path = os.path.join(args.output_dir, f"{side_name}_checkpoint.pt")
    checkpoint = {"state_dict": model.state_dict()}
    xm.save(checkpoint, checkpoint_path)

    scaler_path = os.path.join(args.output_dir, f"{side_name}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    final_predictions_duration = None

    if args.add_final_predictions:
        final_predictions_start = time.perf_counter()
        full_df = pre_process_dataset(side_dataset, args.side)
        full_features, full_labels = split_features_and_labels(full_df)
        full_features = full_features.copy().astype(float)
        full_features[train_features.columns] = scaler.transform(full_features[train_features.columns])

        config_name_values = side_dataset.loc[full_df.index, "config_name"].to_numpy(copy=True)
        bench_values = side_dataset.loc[full_df.index, "bench"].to_numpy(copy=True)

        model.eval()
        with torch.no_grad():
            full_inputs = torch.from_numpy(full_features.to_numpy(copy=True)).float().to(device)
            predictions = model(full_inputs).detach().cpu().squeeze(1).numpy()
            torch_xla.sync()

        final_predictions_df = pd.DataFrame(
            {
                "config_name": config_name_values,
                "bench": bench_values,
                "actual_throughput": full_labels.to_numpy(copy=True),
                "predicted_throughput": predictions,
            }
        )
        predictions_path = os.path.join(args.output_dir, f"{side_name}_final_predictions.csv")
        final_predictions_df.to_csv(predictions_path, index=False)
        final_predictions_duration = time.perf_counter() - final_predictions_start
        print(f"Final predictions time: {final_predictions_duration:.2f}s")

    metrics_path = os.path.join(args.output_dir, f"{side_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_path": args.dataset_path,
                "side": args.side,
                "side_name": side_name,
                "test_size": args.test_size,
                "num_features": int(train_features.shape[1]),
                "total_duration": total_duration,
                "final_predictions_duration": final_predictions_duration,
                "epochs": epochs_data,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

