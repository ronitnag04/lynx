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


@dataclass(frozen=True)
class HyperParams:
    learning_rate: float
    batch_size: int
    epochs: int


DEFAULT_HPARAMS = HyperParams(
    learning_rate=1e-3,
    batch_size=64,
    epochs=20,
)


def load_numpy_dataset(features_path: str, labels_path: str) -> TensorDataset:
    features = torch.from_numpy(np.load(features_path)).float()
    labels = torch.from_numpy(np.load(labels_path)).float()
    return TensorDataset(features, labels)


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
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
        total_loss += loss.detach().to("cpu")
    return total_loss / max(len(loader), 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.view(inputs.size(0), -1)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            torch_xla.sync() 
            total_loss += loss.detach().to("cpu")
    return total_loss / max(len(loader), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Lynx ML model using PyTorch and XLA with the AWS Neuron SDK.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to folder containing 'deserializer_dataset' and 'serializer_dataset' folders.",
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

        train_ds = load_numpy_dataset(train_features_path, train_labels_path)
        test_ds = load_numpy_dataset(test_features_path, test_labels_path)

        train_size = len(train_ds)
        test_size = len(test_ds)
        batch_size = hyperparams.batch_size
        input_size = train_ds[0][0].shape[0]

        # Batch dimension must be a multiple of the batch size for Neuron/XLA compilation.
        if train_size % batch_size != 0:
            raise ValueError(f"Train dataset batch dimension {train_size} must be a multiple of the batch size {batch_size}.")
        if test_size % batch_size != 0:
            raise ValueError(f"Test dataset batch dimension {test_size} must be a multiple of the batch size {batch_size}.")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        device = "xla"

        model = LynxMLModel(input_size=input_size, hidden_dims=(64, 32, 16), output_size=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
        loss_fn = nn.L1Loss()

        print('----------- Start Training --------------')
        print(f"HyperParams: {asdict(hyperparams)}")
        metrics: list[dict[str, float]] = []
        epoch_timings: list[dict[str, float]] = []
        total_start = time.perf_counter()
        for epoch in range(1, hyperparams.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
            eval_loss = evaluate(model, test_loader, loss_fn, device)
            epoch_duration = time.perf_counter() - epoch_start
            metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "eval_loss": float(eval_loss),
                }
            )
            epoch_timings.append(
                {
                    "epoch": epoch,
                    "duration_seconds": epoch_duration,
                }
            )
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")
        print('------------ End Training ---------------')
        total_duration = time.perf_counter() - total_start

        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = os.path.join("checkpoints", f"{dataset_type}_checkpoint.pt")
        checkpoint = {'state_dict': model.state_dict()}
        xm.save(checkpoint, checkpoint_path)

        os.makedirs("training_metrics", exist_ok=True)
        metrics_path = os.path.join("training_metrics", f"{dataset_type}_losses.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        timings_path = os.path.join("training_metrics", f"{dataset_type}_timings.json")
        with open(timings_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_duration_seconds": total_duration,
                    "epochs": epoch_timings,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()

