from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor

from model import LynxMLModel

# XLA imports
import torch_xla.core.xla_model as xm
import torch_xla
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass(frozen=True)
class HyperParams:
    learning_rate: float
    batch_size: int
    epochs: int


DEFAULT_HPARAMS = HyperParams(
    learning_rate=1e-3,
    batch_size=64,
    epochs=1000,
)


def load_numpy_dataset(features_path: str, labels_path: str) -> TensorDataset:
    features = torch.from_numpy(np.load(features_path)).float()
    labels = torch.from_numpy(np.load(labels_path)).float()
    return TensorDataset(features, labels)


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for inputs, targets in loader:
        inputs = inputs.view(inputs.size(0), -1)
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
    return float(total_loss / max(num_batches, 1))


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_percent_error = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.view(inputs.size(0), -1)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            torch_xla.sync()
            total_loss += loss.item()
            # Compute mean absolute percent error for this batch; avoid div-by-zero with eps.
            eps = 1e-8
            batch_percent_error = ((torch.abs(outputs - targets) / (torch.abs(targets) + eps)).mean() * 100.0)
            total_percent_error += batch_percent_error.item()
            num_batches += 1

    denom = max(num_batches, 1)
    avg_loss = total_loss / denom
    avg_percent_error = total_percent_error / denom
    return float(avg_loss), float(avg_percent_error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Lynx ML model using PyTorch and XLA with the AWS Neuron SDK.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to folder containing 'deserializer_dataset' and 'serializer_dataset' folders.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="training_results",
        help="Output directory for checkpoint and metrics files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hyperparams = DEFAULT_HPARAMS

    for dataset_type in ["deserializer", "serializer"]:
        dataset_dir = os.path.join(args.data_dir, dataset_type + "_dataset")

        train_features_path = os.path.join(dataset_dir, "train_features.npy")
        train_labels_path = os.path.join(dataset_dir, "train_labels.npy")
        test_features_path = os.path.join(dataset_dir, "test_features.npy")
        test_labels_path = os.path.join(dataset_dir, "test_labels.npy")

        # Load raw numpy arrays
        train_features_np = np.load(train_features_path)
        train_labels_np = np.load(train_labels_path)
        test_features_np = np.load(test_features_path)
        test_labels_np = np.load(test_labels_path)

        # Standardize features with a shared scaler (fit on train)
        scaler = StandardScaler()
        train_features_np = scaler.fit_transform(train_features_np)
        test_features_np = scaler.transform(test_features_np)

        # Convert to tensors and datasets
        train_features_t = torch.from_numpy(train_features_np).float()
        train_labels_t = torch.from_numpy(train_labels_np).float()
        test_features_t = torch.from_numpy(test_features_np).float()
        test_labels_t = torch.from_numpy(test_labels_np).float()

        train_ds = TensorDataset(train_features_t, train_labels_t)
        test_ds = TensorDataset(test_features_t, test_labels_t)

        train_size = len(train_ds)
        test_size = len(test_ds)
        batch_size = hyperparams.batch_size
        input_size = train_features_t.shape[1]

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        device = "xla"

        model = LynxMLModel(input_size=input_size, hidden_dims=(64, 32, 16), output_size=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
        loss_fn = nn.L1Loss()

        print('----------- Start Training --------------')
        print(f"HyperParams: {asdict(hyperparams)}")
        epochs_data: list[dict[str, float]] = []
        total_start = time.perf_counter()
        for epoch in range(1, hyperparams.epochs + 1):
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
        print('------------ End Training ---------------')
        total_duration = time.perf_counter() - total_start

        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.output_dir, f"{dataset_type}_checkpoint.pt")
        checkpoint = {'state_dict': model.state_dict()}
        xm.save(checkpoint, checkpoint_path)

        # Persist preprocessing artifacts so inference can reproduce training transforms.
        scaler_path = os.path.join(args.output_dir, f"{dataset_type}_scaler.joblib")
        joblib.dump(scaler, scaler_path)

        metrics_path = os.path.join(args.output_dir, f"{dataset_type}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_duration": total_duration,
                    "epochs": epochs_data,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()

