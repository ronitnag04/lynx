#!/usr/bin/env python3
"""
Compare Pareto frontiers across different ML models.

This script runs exhaustive search with different ML models and compares
their Pareto frontiers to validate model predictions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_exhaustive_search(
    side: str,
    model_type: str,
    checkpoint_dir: Path,
    features_file: Path,
    output_dir: Path,
    hw_cost_model_path: Path | None = None,
    num_objectives: int = 3,
    kappa: float = 0.0001,
    limit_configs: int = 0,
) -> Path:
    """Run exhaustive search for a single model type."""
    output_path = output_dir / f"pareto_{side}_{model_type}.json"

    cmd = [
        "python3",
        "exhaustive_search.py",
        "--side", side,
        "--model-type", model_type,
        "--checkpoint-dir", str(checkpoint_dir),
        "--features-file", str(features_file),
        "--output", str(output_path),
        "--num-objectives", str(num_objectives),
        "--kappa", str(kappa),
    ]

    if hw_cost_model_path is not None:
        cmd.extend(["--hw-cost-model", str(hw_cost_model_path)])

    if limit_configs > 0:
        cmd.extend(["--limit-configs", str(limit_configs)])

    print(f"\n{'='*80}")
    print(f"Running exhaustive search for {side} with {model_type}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, check=True, capture_output=False)
    duration = time.time() - start_time

    print(f"Completed in {duration:.2f}s")
    return output_path


def load_pareto_data(path: Path) -> dict[str, Any]:
    """Load Pareto front data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def extract_pareto_metrics(data: dict[str, Any]) -> pd.DataFrame:
    """Extract Pareto front points into a DataFrame."""
    rows = []
    for point in data["pareto_front"]:
        row = {
            "throughput": point["predicted_throughput_gbits_per_sec"],
            "validation_candidate": point["validation_candidate"],
        }
        if "predicted_logic_cells" in point:
            row["logic_cells"] = point["predicted_logic_cells"]
            row["ram_bits"] = point["predicted_ram_bits"]
            row["cost"] = point["predicted_cost_scalar_kappa"]
        else:
            row["cost"] = point["predicted_cost"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pareto_comparison(
    results: dict[str, pd.DataFrame],
    side: str,
    output_dir: Path,
    num_objectives: int,
) -> None:
    """Plot Pareto frontiers for all models."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    if num_objectives == 3:
        # 3D objectives: plot throughput vs each cost component
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Throughput vs Logic Cells
        ax1 = axes[0]
        for model_type, df in results.items():
            ax1.scatter(
                df["logic_cells"],
                df["throughput"],
                label=model_type,
                alpha=0.7,
                s=50,
            )
        ax1.set_xlabel("Logic Cells", fontsize=12)
        ax1.set_ylabel("Throughput (Gbit/s)", fontsize=12)
        ax1.set_title(f"{side} - Pareto Front: Throughput vs Logic Cells", fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Throughput vs RAM Bits
        ax2 = axes[1]
        for model_type, df in results.items():
            ax2.scatter(
                df["ram_bits"],
                df["throughput"],
                label=model_type,
                alpha=0.7,
                s=50,
            )
        ax2.set_xlabel("RAM Bits", fontsize=12)
        ax2.set_ylabel("Throughput (Gbit/s)", fontsize=12)
        ax2.set_title(f"{side} - Pareto Front: Throughput vs RAM Bits", fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / f"{side}_pareto_comparison_3d.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")

    else:
        # 2D objectives: plot throughput vs cost
        fig, ax = plt.subplots(figsize=(10, 8))
        for model_type, df in results.items():
            ax.scatter(
                df["cost"],
                df["throughput"],
                label=model_type,
                alpha=0.7,
                s=50,
            )
        ax.set_xlabel("Hardware Cost", fontsize=12)
        ax.set_ylabel("Throughput (Gbit/s)", fontsize=12)
        ax.set_title(f"{side} - Pareto Front Comparison", fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / f"{side}_pareto_comparison_2d.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")


def compute_pareto_statistics(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute statistics comparing Pareto frontiers."""
    stats = []
    for model_type, df in results.items():
        stats.append({
            "Model": model_type,
            "Pareto Points": len(df),
            "Max Throughput": df["throughput"].max(),
            "Min Throughput": df["throughput"].min(),
            "Mean Throughput": df["throughput"].mean(),
            "Min Cost": df["cost"].min(),
            "Max Cost": df["cost"].max(),
            "Mean Cost": df["cost"].mean(),
        })
    return pd.DataFrame(stats).sort_values("Max Throughput", ascending=False)


def compute_pareto_overlap(results: dict[str, pd.DataFrame], tolerance: float = 0.01) -> pd.DataFrame:
    """Compute pairwise overlap between Pareto frontiers."""
    model_types = list(results.keys())
    n_models = len(model_types)
    overlap_matrix = np.zeros((n_models, n_models))

    for i, model_i in enumerate(model_types):
        df_i = results[model_i]
        for j, model_j in enumerate(model_types):
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue

            df_j = results[model_j]

            # Count how many points in i are "close" to points in j
            matches = 0
            for _, row_i in df_i.iterrows():
                for _, row_j in df_j.iterrows():
                    tp_diff = abs(row_i["throughput"] - row_j["throughput"]) / (row_i["throughput"] + 1e-8)
                    cost_diff = abs(row_i["cost"] - row_j["cost"]) / (row_i["cost"] + 1e-8)
                    if tp_diff < tolerance and cost_diff < tolerance:
                        matches += 1
                        break

            overlap_matrix[i, j] = matches / len(df_i) if len(df_i) > 0 else 0.0

    return pd.DataFrame(overlap_matrix, index=model_types, columns=model_types)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Pareto frontiers across different ML models."
    )
    parser.add_argument(
        "--side",
        choices=["des", "ser"],
        required=True,
        help="Which side to evaluate: des or ser.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing trained models and scalers.",
    )
    parser.add_argument(
        "--hw-cost-model-path",
        type=Path,
        default=Path("../hw_cost_model/checkpoints"),
        help="Path to trained hardware cost model directory.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path("../analytical_model/extracted_features.json"),
        help="Path to extracted_features.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pareto_comparison"),
        help="Output directory for comparison results.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "neural",
            "LinearRegression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "DecisionTree",
            "RandomForest",
            "GradientBoosting",
            "KNN",
            "SVR",
        ],
        help="List of model types to compare.",
    )
    parser.add_argument(
        "--num-objectives",
        type=int,
        choices=(2, 3),
        default=3,
        help="Number of objectives (2 or 3).",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.0001,
        help="Cost-combining weight for 2-objective case.",
    )
    parser.add_argument(
        "--limit-configs",
        type=int,
        default=0,
        help="If >0, only evaluate the first N configs (debug).",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip running exhaustive search, only analyze existing results.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()

    print(f"\n{'='*80}")
    print("PARETO FRONTIER COMPARISON")
    print(f"{'='*80}")
    print(f"Side: {args.side}")
    print(f"Models: {', '.join(args.models)}")
    print(f"HW Cost Model Path: {args.hw_cost_model_path}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine hw cost model path based on side
    hw_cost_model = None
    if args.hw_cost_model_path and args.hw_cost_model_path.exists():
        hw_cost_model = args.hw_cost_model_path / f"{args.side}_cost_model.joblib"
        if hw_cost_model.exists():
            print(f"Using HW cost model: {hw_cost_model}")
        else:
            print(f"WARNING: HW cost model not found at {hw_cost_model}, using structural cost estimator")
            hw_cost_model = None
    else:
        print("WARNING: HW cost model path does not exist, using structural cost estimator")

    # Run exhaustive search for each model
    pareto_files = {}
    if not args.skip_search:
        for model_type in args.models:
            try:
                output_path = run_exhaustive_search(
                    args.side,
                    model_type,
                    args.checkpoint_dir,
                    args.features_file,
                    args.output_dir,
                    hw_cost_model,
                    args.num_objectives,
                    args.kappa,
                    args.limit_configs,
                )
                pareto_files[model_type] = output_path
            except Exception as e:
                print(f"\nERROR: Failed to run exhaustive search for {model_type}: {e}")
    else:
        # Load existing results
        for model_type in args.models:
            path = args.output_dir / f"pareto_{args.side}_{model_type}.json"
            if path.exists():
                pareto_files[model_type] = path
            else:
                print(f"WARNING: Missing results for {model_type}: {path}")

    if not pareto_files:
        print("\nERROR: No Pareto frontiers available for comparison.")
        return

    # Load and extract Pareto metrics
    print(f"\n{'='*80}")
    print("Loading and analyzing Pareto frontiers")
    print(f"{'='*80}")

    results = {}
    for model_type, path in pareto_files.items():
        data = load_pareto_data(path)
        df = extract_pareto_metrics(data)
        results[model_type] = df
        print(f"{model_type}: {len(df)} Pareto points")

    # Compute statistics
    stats_df = compute_pareto_statistics(results)
    print(f"\n{'='*80}")
    print("Pareto Front Statistics")
    print(f"{'='*80}")
    print(stats_df.to_string(index=False))

    # Save statistics
    stats_path = args.output_dir / f"{args.side}_pareto_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved statistics: {stats_path}")

    # Compute overlap
    overlap_df = compute_pareto_overlap(results)
    print(f"\n{'='*80}")
    print("Pareto Front Overlap Matrix (fraction of points matched)")
    print(f"{'='*80}")
    print(overlap_df.to_string())

    # Save overlap
    overlap_path = args.output_dir / f"{args.side}_pareto_overlap.csv"
    overlap_df.to_csv(overlap_path)
    print(f"\nSaved overlap matrix: {overlap_path}")

    # Plot comparison
    plot_pareto_comparison(results, args.side, args.output_dir, args.num_objectives)

    # Generate summary report
    summary = {
        "side": args.side,
        "num_objectives": args.num_objectives,
        "models_compared": list(results.keys()),
        "statistics": stats_df.to_dict(orient="records"),
        "overlap_matrix": overlap_df.to_dict(),
    }

    summary_path = args.output_dir / f"{args.side}_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Export validation configs to CSV using standalone script
    print(f"\n{'='*80}")
    print("Exporting validation configs to CSV")
    print(f"{'='*80}")
    import subprocess
    export_cmd = [
        "python3",
        "export_pareto_configs.py",
        "--input-dir", str(args.output_dir),
        "--side", args.side,
    ]
    try:
        subprocess.run(export_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"WARNING: CSV export failed: {e}")
    except FileNotFoundError:
        print("WARNING: export_pareto_configs.py not found, skipping CSV export")

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
