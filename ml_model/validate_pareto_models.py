#!/usr/bin/env python3
"""
Script to validate ML model predictions against actual hardware results.
For each ML model, plots pareto fronts comparing predicted vs actual values
and calculates percent errors.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import os

# Model names to process
ML_MODELS = [
    "neural",
    "SVR",
    "LinearRegression",
    "Ridge",
    "GradientBoosting",
    "DecisionTree",
    "KNN",
    "RandomForest",
    "ElasticNet",
    "Lasso"
]

def load_pareto_data(pareto_file: Path) -> Dict:
    """Load pareto front data from JSON file."""
    with open(pareto_file, 'r') as f:
        return json.load(f)

def load_validation_data(yosys_file: Path, sweep_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load validation results from CSV files."""
    yosys_df = pd.read_csv(yosys_file)
    sweep_df = pd.read_csv(sweep_file)
    return yosys_df, sweep_df

def get_actual_cost(config_name: str, yosys_df: pd.DataFrame) -> float:
    """Get actual hardware cost for a config from yosys results."""
    row = yosys_df[yosys_df['config_name'] == config_name]
    if row.empty:
        return None
    # Cost = total_logic_cells + total_ram_flop_cells
    return float(row['total_logic_cells'].iloc[0] + row['total_ram_flop_cells'].iloc[0])

def get_actual_throughput(config_name: str, sweep_df: pd.DataFrame) -> float:
    """Get actual average throughput for a config from benchmark sweep results."""
    rows = sweep_df[sweep_df['config_name'] == config_name]
    if rows.empty:
        return None
    # Average throughput across all benchmarks in Gbit/s
    avg_throughput = rows['throughput_bytes_per_sec'].mean() * 8 / 1e9
    return avg_throughput

def config_to_name(config: Dict, side: str) -> str:
    """Convert config dict to config name string.

    Uses the correct acronym ordering from gen_pareto_validation_configs.py:
    - Serializer: SC, SDD, SDF, SDH, SDQ, SDM, SF, SMJ, SMI, SMP
    - Deserializer: DC, DDFQ, DDFP, DDL, DFL, DMB, DML, DTD, DTM
    """
    if side == "ser":
        prefix = "ProtoAccelSerParetoValidation"
        # Correct order from SER_KEYS in gen_sweep_configs.py
        parts = [
            f"SC{config['ser_cr_rocc_commands']}",      # SC
            f"SDD{config['ser_dth_descriptor_reqs']}",  # SDD
            f"SDF{config['ser_dth_fh_outputs']}",       # SDF
            f"SDH{config['ser_dth_hasbits_reqs']}",     # SDH
            f"SDQ{config['ser_dth_reg_resps']}",        # SDQ
            f"SDM{config['ser_dth_reqs_meta']}",        # SDM
            f"SF{config['ser_field_handlers']}",        # SF
            f"SMJ{config['ser_mw_write_inject']}",      # SMJ
            f"SMI{config['ser_mw_write_input']}",       # SMI
            f"SMP{config['ser_mw_write_ptrs']}"         # SMP
        ]
    else:  # des
        prefix = "ProtoAccelDesParetoValidation"
        # Correct order from DES_KEYS in gen_sweep_configs.py
        parts = [
            f"DC{config['des_cr_rocc_commands']}",      # DC
            f"DDFQ{config['des_dth_fd_reqs']}",         # DDFQ
            f"DDFP{config['des_dth_fd_resps']}",        # DDFP
            f"DDL{config['des_dth_l1_reqs']}",          # DDL
            f"DFL{config['des_fw_l1_reqs']}",           # DFL
            f"DMB{config['des_ml_buf_info_q']}",        # DMB
            f"DML{config['des_ml_load_info_q']}",       # DML
            f"DTD{config['des_top_descriptor_reqs']}",  # DTD
            f"DTM{config['des_top_memloader_reqs']}"    # DTM
        ]

    return prefix + "".join(parts) + "Config"

def calculate_percent_error(predicted: float, actual: float) -> float:
    """Calculate percent error: (predicted - actual) / actual * 100."""
    if actual == 0:
        return 0.0
    return ((predicted - actual) / actual) * 100.0

def plot_pareto_front(pareto_data: Dict, yosys_df: pd.DataFrame, sweep_df: pd.DataFrame,
                      output_dir: Path, model_name: str):
    """Plot pareto front with predicted vs actual points."""
    side = pareto_data['side']
    pareto_points = pareto_data['pareto_front']

    # Extract predicted values
    predicted_costs = []
    predicted_throughputs = []
    is_validation = []
    actual_costs = []
    actual_throughputs = []
    config_names = []

    for point in pareto_points:
        predicted_costs.append(point['predicted_cost'])
        predicted_throughputs.append(point['predicted_throughput_gbits_per_sec'])
        is_validation.append(point.get('validation_candidate', False))

        # Get actual values for validation candidates
        if point.get('validation_candidate', False):
            config_name = config_to_name(point['config'], side)
            config_names.append(config_name)
            actual_cost = get_actual_cost(config_name, yosys_df)
            actual_tput = get_actual_throughput(config_name, sweep_df)
            actual_costs.append(actual_cost)
            actual_throughputs.append(actual_tput)

    predicted_costs = np.array(predicted_costs)
    predicted_throughputs = np.array(predicted_throughputs)
    is_validation = np.array(is_validation)

    # Plot 1: Cost vs Throughput
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot all pareto points
    ax.scatter(predicted_costs[~is_validation], predicted_throughputs[~is_validation],
               c="blue", label="Predicted (non-validation)", alpha=0.6, s=50)

    # Plot validation candidate predictions
    ax.scatter(predicted_costs[is_validation], predicted_throughputs[is_validation],
               c="red", marker='o', label="Predicted (validation)", alpha=0.7, s=80)

    # Plot actual validation results
    if actual_costs and actual_throughputs:
        valid_actuals = [(c, t) for c, t in zip(actual_costs, actual_throughputs) if c is not None and t is not None]
        if valid_actuals:
            actual_costs_valid, actual_throughputs_valid = zip(*valid_actuals)
            ax.scatter(actual_costs_valid, actual_throughputs_valid,
                       c="green", marker='x', s=150, linewidths=3,
                       label="Actual (validation)", alpha=0.9)

    # Plot baseline if present
    if 'baseline' in pareto_data:
        baseline = pareto_data['baseline']
        ax.scatter(baseline['predicted_cost'], baseline['predicted_throughput_gbits_per_sec'],
                   c="purple", marker='D', s=100, label="Baseline", alpha=0.9)

    ax.set_xlabel("Hardware Cost (logic cells + RAM flop cells)", fontsize=14)
    ax.set_ylabel("Throughput (Gbit/s)", fontsize=14)
    ax.set_title(f"{side.upper()} {model_name}: Pareto Front (Predicted vs Actual)", fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"{model_name}_cost_vs_throughput.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Cost Error comparison (only validation points)
    if actual_costs and actual_throughputs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        valid_indices = [i for i, (c, t) in enumerate(zip(actual_costs, actual_throughputs))
                        if c is not None and t is not None]

        if valid_indices:
            pred_costs_val = predicted_costs[is_validation][valid_indices]
            pred_tput_val = predicted_throughputs[is_validation][valid_indices]
            actual_costs_val = [actual_costs[i] for i in valid_indices]
            actual_tput_val = [actual_throughputs[i] for i in valid_indices]

            # Cost comparison
            x_pos = np.arange(len(valid_indices))
            width = 0.35

            ax1.bar(x_pos - width/2, pred_costs_val, width, label='Predicted', alpha=0.8, color='red')
            ax1.bar(x_pos + width/2, actual_costs_val, width, label='Actual', alpha=0.8, color='green')
            ax1.set_xlabel("Validation Config Index", fontsize=12)
            ax1.set_ylabel("Hardware Cost", fontsize=12)
            ax1.set_title(f"{model_name}: Cost Comparison", fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

            # Throughput comparison
            ax2.bar(x_pos - width/2, pred_tput_val, width, label='Predicted', alpha=0.8, color='red')
            ax2.bar(x_pos + width/2, actual_tput_val, width, label='Actual', alpha=0.8, color='green')
            ax2.set_xlabel("Validation Config Index", fontsize=12)
            ax2.set_ylabel("Throughput (Gbit/s)", fontsize=12)
            ax2.set_title(f"{model_name}: Throughput Comparison", fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            fig.savefig(output_dir / f"{model_name}_validation_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()

def calculate_errors(pareto_data: Dict, yosys_df: pd.DataFrame, sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate percent errors for validation configs."""
    side = pareto_data['side']
    pareto_points = pareto_data['pareto_front']

    results = []

    for point in pareto_points:
        if not point.get('validation_candidate', False):
            continue

        config_name = config_to_name(point['config'], side)
        predicted_cost = point['predicted_cost']
        predicted_tput = point['predicted_throughput_gbits_per_sec']

        actual_cost = get_actual_cost(config_name, yosys_df)
        actual_tput = get_actual_throughput(config_name, sweep_df)

        if actual_cost is not None and actual_tput is not None:
            cost_error = calculate_percent_error(predicted_cost, actual_cost)
            tput_error = calculate_percent_error(predicted_tput, actual_tput)

            results.append({
                'Config': config_name,
                'Predicted Cost': f"{predicted_cost:.2f}",
                'Actual Cost': f"{actual_cost:.2f}",
                'Cost Error (%)': f"{cost_error:.2f}",
                'Predicted Throughput (Gbit/s)': f"{predicted_tput:.3f}",
                'Actual Throughput (Gbit/s)': f"{actual_tput:.3f}",
                'Throughput Error (%)': f"{tput_error:.2f}",
                'cost_error_float': cost_error,
                'tput_error_float': tput_error
            })

    return pd.DataFrame(results)

def process_side(side: str, base_dir: Path, output_base: Path):
    """Process all models for one side (ser or des)."""
    comparison_dir = base_dir / f"pareto_comparison_{side}"
    yosys_file = comparison_dir / f"yosys_validation_{side}.csv"
    sweep_file = comparison_dir / f"{side}_pareto_validation_sweep_results.csv"

    print(f"\n{'='*80}")
    print(f"Processing {side.upper()} models")
    print(f"{'='*80}\n")

    # Load validation data
    yosys_df, sweep_df = load_validation_data(yosys_file, sweep_file)
    print(f"Loaded validation data:")
    print(f"  - Yosys results: {len(yosys_df)} configs")
    print(f"  - Sweep results: {len(sweep_df)} benchmark runs")

    # Create output directory
    plot_dir = comparison_dir / "pareto_front_plots"
    plot_dir.mkdir(exist_ok=True)

    all_errors = []

    for model_name in ML_MODELS:
        pareto_file = comparison_dir / f"pareto_{side}_{model_name}.json"

        if not pareto_file.exists():
            print(f"  ⚠ Skipping {model_name}: {pareto_file} not found")
            continue

        print(f"\n  Processing {model_name}...")

        # Load pareto data
        pareto_data = load_pareto_data(pareto_file)

        # Plot pareto fronts
        plot_pareto_front(pareto_data, yosys_df, sweep_df, plot_dir, model_name)
        print(f"    ✓ Generated plots")

        # Calculate errors
        error_df = calculate_errors(pareto_data, yosys_df, sweep_df)

        if not error_df.empty:
            # Calculate averages
            avg_cost_error = error_df['cost_error_float'].abs().mean()
            avg_tput_error = error_df['tput_error_float'].abs().mean()

            all_errors.append({
                'Model': model_name,
                'Side': side.upper(),
                'Num Validation Configs': len(error_df),
                'Avg Cost Error (%)': avg_cost_error,
                'Avg Throughput Error (%)': avg_tput_error,
                'error_df': error_df
            })

            print(f"    ✓ Calculated errors for {len(error_df)} validation configs")
            print(f"      - Avg Cost Error: {avg_cost_error:.2f}%")
            print(f"      - Avg Throughput Error: {avg_tput_error:.2f}%")
        else:
            print(f"    ⚠ No validation data found")

    return all_errors, plot_dir

def generate_markdown_report(ser_errors: List[Dict], des_errors: List[Dict], output_file: Path):
    """Generate markdown report with all error tables."""

    with open(output_file, 'w') as f:
        f.write("# ML Model Validation Results\n\n")
        f.write("This report compares predicted vs actual hardware cost and throughput for validation configurations.\n\n")

        # Serializer results
        f.write("## Serializer Models\n\n")
        for model_info in ser_errors:
            model_name = model_info['Model']
            error_df = model_info['error_df']

            f.write(f"### {model_name}\n\n")
            f.write(f"**Number of validation configs:** {len(error_df)}\n\n")

            # Write table
            display_df = error_df.drop(columns=['cost_error_float', 'tput_error_float'])
            f.write(display_df.to_markdown(index=False))
            f.write("\n\n")

            # Write averages
            avg_cost_err = model_info['Avg Cost Error (%)']
            avg_tput_err = model_info['Avg Throughput Error (%)']
            f.write(f"**Average Absolute Cost Error:** {avg_cost_err:.2f}%  \n")
            f.write(f"**Average Absolute Throughput Error:** {avg_tput_err:.2f}%\n\n")
            f.write("---\n\n")

        # Deserializer results
        f.write("\n## Deserializer Models\n\n")
        for model_info in des_errors:
            model_name = model_info['Model']
            error_df = model_info['error_df']

            f.write(f"### {model_name}\n\n")
            f.write(f"**Number of validation configs:** {len(error_df)}\n\n")

            # Write table
            display_df = error_df.drop(columns=['cost_error_float', 'tput_error_float'])
            f.write(display_df.to_markdown(index=False))
            f.write("\n\n")

            # Write averages
            avg_cost_err = model_info['Avg Cost Error (%)']
            avg_tput_err = model_info['Avg Throughput Error (%)']
            f.write(f"**Average Absolute Cost Error:** {avg_cost_err:.2f}%  \n")
            f.write(f"**Average Absolute Throughput Error:** {avg_tput_err:.2f}%\n\n")
            f.write("---\n\n")

        # Summary table
        f.write("\n## Summary: Average Errors Across All Models\n\n")

        summary_data = []
        for model_info in ser_errors + des_errors:
            summary_data.append({
                'Model': model_info['Model'],
                'Side': model_info['Side'],
                'Validation Configs': model_info['Num Validation Configs'],
                'Avg Cost Error (%)': f"{model_info['Avg Cost Error (%)']:.2f}",
                'Avg Throughput Error (%)': f"{model_info['Avg Throughput Error (%)']:.2f}"
            })

        summary_df = pd.DataFrame(summary_data)
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")

        # Overall averages
        all_cost_errors = [m['Avg Cost Error (%)'] for m in ser_errors + des_errors]
        all_tput_errors = [m['Avg Throughput Error (%)'] for m in ser_errors + des_errors]

        f.write("### Overall Averages\n\n")
        f.write(f"**Mean Cost Error across all models:** {np.mean(all_cost_errors):.2f}%  \n")
        f.write(f"**Mean Throughput Error across all models:** {np.mean(all_tput_errors):.2f}%  \n")

def main():
    # Setup paths
    base_dir = Path("/home/ubuntu/lynx/ml_model/results/hpb_verilator")
    ser_dir = base_dir / "serializer_ml_models"
    des_dir = base_dir / "deserializer_ml_models"

    # Process both sides
    ser_errors, ser_plot_dir = process_side("ser", ser_dir, ser_dir)
    des_errors, des_plot_dir = process_side("des", des_dir, des_dir)

    # Generate markdown report
    report_file = base_dir / "ml_model_validation_report.md"
    generate_markdown_report(ser_errors, des_errors, report_file)

    print(f"\n{'='*80}")
    print(f"✓ Validation complete!")
    print(f"{'='*80}\n")
    print(f"Outputs:")
    print(f"  - Serializer plots: {ser_plot_dir}")
    print(f"  - Deserializer plots: {des_plot_dir}")
    print(f"  - Full report: {report_file}")
    print()

if __name__ == "__main__":
    main()
