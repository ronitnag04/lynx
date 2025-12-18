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

    epochs_data = data["epochs"]

    epochs = range(len(epochs_data))
    train_losses = [epoch["train_loss"] for epoch in epochs_data]
    eval_losses = [epoch["eval_loss"] for epoch in epochs_data]

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
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    metrics_dir = base_dir / "training_metrics"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sha3_path = metrics_dir / "sha3_metrics.json"

    plt.switch_backend("Agg")  # ensure headless execution is fine

    sha3_epochs, sha3_train, sha3_eval = load_losses(sha3_path)

    plot_losses(
        sha3_epochs,
        sha3_train,
        sha3_eval,
        title="SHA3 Train vs Eval Loss",
        output_path=plots_dir / "sha3_losses.png",
    )

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
