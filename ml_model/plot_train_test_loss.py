"""Plot train/eval loss curves for serializer and deserializer models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


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
    max_points: Optional[int] = None,
) -> None:
    """Generate and save a loss plot.
    
    Args:
        epochs: Epoch numbers
        train_losses: Training loss values
        eval_losses: Evaluation loss values
        title: Plot title
        output_path: Path to save the plot
        max_points: Maximum number of points to plot per curve. If None, plots all points.
    """
    # Convert to lists for indexing
    epochs_list = list(epochs)
    train_losses_list = list(train_losses)
    eval_losses_list = list(eval_losses)
    
    # Downsample if max_points is specified
    if max_points is not None and len(epochs_list) > max_points:
        indices = np.linspace(0, len(epochs_list) - 1, max_points, dtype=int)
        epochs_list = [epochs_list[i] for i in indices]
        train_losses_list = [train_losses_list[i] for i in indices]
        eval_losses_list = [eval_losses_list[i] for i in indices]
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_list, train_losses_list, label="Train loss", marker="o")
    plt.plot(epochs_list, eval_losses_list, label="Eval loss", marker="o")
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
    # Maximum number of points to plot per curve (None = plot all points)
    MAX_POINTS = 50
    
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
        max_points=MAX_POINTS,
    )

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
