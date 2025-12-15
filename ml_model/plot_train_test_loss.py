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

    epochs: list[int] = []
    train_losses: list[float] = []
    eval_losses: list[float] = []

    for entry in data:
        # Defensive: skip malformed rows rather than failing mid-loop.
        if not all(k in entry for k in ("epoch", "train_loss", "eval_loss")):
            continue
        epochs.append(int(entry["epoch"]))
        train_losses.append(float(entry["train_loss"]))
        eval_losses.append(float(entry["eval_loss"]))

    # # Remove the first epoch (outlier)
    # epochs = epochs[1:]
    # train_losses = train_losses[1:]
    # eval_losses = eval_losses[1:]

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
    base_dir = Path(__file__).resolve().parent
    metrics_dir = base_dir / "training_metrics"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    serializer_path = metrics_dir / "serializer_losses.json"
    deserializer_path = metrics_dir / "deserializer_losses.json"

    plt.switch_backend("Agg")  # ensure headless execution is fine

    serializer_epochs, serializer_train, serializer_eval = load_losses(serializer_path)

    plot_losses(
        serializer_epochs,
        serializer_train,
        serializer_eval,
        title="Serializer Train vs Eval Loss",
        output_path=plots_dir / "serializer_losses.png",
    )

    deserializer_epochs, deserializer_train, deserializer_eval = load_losses(
        deserializer_path
    )
    plot_losses(
        deserializer_epochs,
        deserializer_train,
        deserializer_eval,
        title="Deserializer Train vs Eval Loss",
        output_path=plots_dir / "deserializer_losses.png",
    )

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
