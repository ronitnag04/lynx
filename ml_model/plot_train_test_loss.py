"""Plot train/eval loss curves for serializer and deserializer models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


def load_losses(path: Path) -> Tuple[list[int], list[float], list[float]]:
    """Read epochs, train_loss, eval_loss lists from a JSON metrics file."""
    with path.open() as f:
        data = json.load(f)

    entries = data["epochs"]

    epochs: list[int] = []
    train_losses: list[float] = []
    eval_losses: list[float] = []

    for i, entry in enumerate(entries):
        epochs.append(i)
        train_losses.append(float(entry["train_loss"]))
        eval_losses.append(float(entry["eval_loss"]))

    return epochs, train_losses, eval_losses


def plot_losses(
    epochs: Iterable[int],
    train_losses: Iterable[float],
    eval_losses: Iterable[float],
    title: str,
    output_path: Path,
) -> None:
    """Generate and save a loss plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss", marker="o")
    plt.plot(epochs, eval_losses, label="Eval loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:

    serializer_dir = Path("results/hpb_verilator/serializer_test_0.25").absolute()
    deserializer_dir = Path("results/hpb_verilator/deserializer_test_0.25").absolute()

    serializer_path = Path(serializer_dir) / "serializer_metrics.json"
    deserializer_path = Path(deserializer_dir) / "deserializer_metrics.json"

    plt.switch_backend("Agg")  # ensure headless execution is fine

    serializer_epochs, serializer_train, serializer_eval = load_losses(serializer_path)

    plot_losses(
        serializer_epochs,
        serializer_train,
        serializer_eval,
        title="Serializer Train vs Eval Loss",
        output_path=serializer_dir / "serializer_losses.png",
    )

    print(f"Saved serializer loss plot to {serializer_dir / 'serializer_losses.png'}")

    deserializer_epochs, deserializer_train, deserializer_eval = load_losses(
        deserializer_path
    )
    plot_losses(
        deserializer_epochs,
        deserializer_train,
        deserializer_eval,
        title="Deserializer Train vs Eval Loss",
        output_path=deserializer_dir / "deserializer_losses.png",
    )

    print(f"Saved deserializer loss plot to {deserializer_dir / 'deserializer_losses.png'}")


if __name__ == "__main__":
    main()
