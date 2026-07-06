#!/usr/bin/env python3
"""
End-to-end ML model comparison for Lynx throughput prediction.

This script loads deserializer and serializer training data, preprocesses them
using the same logic as train.py, and evaluates multiple ML models. Results
are exported as LaTeX tables and plots are saved individually.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    print(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


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

    # Handle label column
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

    # Validate required columns
    side_param_values = DES_PARAM_VALUES if side == "des" else SER_PARAM_VALUES
    missing_param_columns = [col for col in side_param_values if col not in dataset.columns]
    if missing_param_columns:
        raise ValueError(
            f"Input CSV is missing expected {side_name} config columns: {missing_param_columns}"
        )

    knob_columns = list(side_param_values)

    # Build feature columns
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

    # Drop NaN and select relevant columns
    pruned = dataset.dropna(subset=feature_columns + [label_column]).copy()
    model_df = pruned[feature_columns + [label_column]].copy()
    return model_df.astype(float)


def split_features_and_labels(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and labels."""
    features = model_df.drop(columns=[LABEL_COLUMN])
    labels = model_df[LABEL_COLUMN]
    return features, labels


def get_ml_models() -> dict:
    """Return dictionary of ML models to evaluate."""
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


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    side_name: str,
) -> pd.DataFrame:
    """Train and evaluate all ML models."""
    print(f"\n{'='*80}")
    print(f"Evaluating ML models for {side_name}")
    print(f"{'='*80}")
    print(f"Train set: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"Test set: {len(X_test)} samples")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_ml_models()
    results = []

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
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Percent error (matching train.py metric)
        eps = 1e-8
        train_percent_error = (np.abs(y_pred_train - y_train) / (np.abs(y_train) + eps)).mean() * 100
        test_percent_error = (np.abs(y_pred_test - y_test) / (np.abs(y_test) + eps)).mean() * 100

        results.append({
            'Model': name,
            'Train Time (s)': train_time,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train % Error': train_percent_error,
            'Test % Error': test_percent_error,
        })

        print(f"  Train Time: {train_time:.3f}s")
        print(f"  Test MAE: {test_mae:.4e} | Test R²: {test_r2:.6f} | Test % Error: {test_percent_error:.4f}%")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test % Error')

    print(f"\n{'='*80}")
    print(f"Summary for {side_name} (sorted by Test % Error):")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))

    return results_df


def save_plots(results_df: pd.DataFrame, side_name: str, output_dir: str) -> None:
    """Generate and save individual plots."""
    print(f"\nSaving plots for {side_name}...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # 1. Test Percent Error
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(x='Model', y='Test % Error', kind='bar', ax=ax, legend=False, color='steelblue')
    ax.set_title(f'{side_name} - Test Percent Error by Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percent Error (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{side_name}_test_percent_error.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # 2. Test R² Score
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(x='Model', y='Test R²', kind='bar', ax=ax, legend=False, color='coral')
    ax.set_title(f'{side_name} - Test R² Score by Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{side_name}_test_r2.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # 3. Test MAE
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(x='Model', y='Test MAE', kind='bar', ax=ax, legend=False, color='seagreen')
    ax.set_title(f'{side_name} - Test MAE by Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.set_yscale('log')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{side_name}_test_mae.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # 4. Training Time
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(x='Model', y='Train Time (s)', kind='bar', ax=ax, legend=False, color='mediumpurple')
    ax.set_title(f'{side_name} - Training Time by Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{side_name}_train_time.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # 5. Combined comparison (2x2 subplot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    results_df.plot(x='Model', y='Test % Error', kind='bar', ax=axes[0, 0], legend=False, color='steelblue')
    axes[0, 0].set_title('Test Percent Error', fontweight='bold')
    axes[0, 0].set_ylabel('% Error')
    axes[0, 0].tick_params(axis='x', rotation=45)

    results_df.plot(x='Model', y='Test R²', kind='bar', ax=axes[0, 1], legend=False, color='coral')
    axes[0, 1].set_title('Test R² Score', fontweight='bold')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].tick_params(axis='x', rotation=45)

    results_df.plot(x='Model', y='Test MAE', kind='bar', ax=axes[1, 0], legend=False, color='seagreen')
    axes[1, 0].set_title('Test MAE (log scale)', fontweight='bold')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_yscale('log')
    axes[1, 0].tick_params(axis='x', rotation=45)

    results_df.plot(x='Model', y='Train Time (s)', kind='bar', ax=axes[1, 1], legend=False, color='mediumpurple')
    axes[1, 1].set_title('Training Time', fontweight='bold')
    axes[1, 1].set_ylabel('Seconds')
    axes[1, 1].tick_params(axis='x', rotation=45)

    fig.suptitle(f'{side_name} - ML Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{side_name}_combined_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def generate_latex_tables(
    des_results: pd.DataFrame | None,
    ser_results: pd.DataFrame | None,
    output_dir: str,
) -> None:
    """Generate LaTeX tables for results."""
    print("\nGenerating LaTeX tables...")

    latex_output = []

    # Model descriptions table
    model_descriptions = pd.DataFrame([
        {
            'Model': 'Linear Regression',
            'Type': 'Linear',
            'Description': 'Ordinary least squares regression',
            'Regularization': 'None'
        },
        {
            'Model': 'Ridge',
            'Type': 'Linear',
            'Description': 'Linear regression with L2 penalty',
            'Regularization': 'L2 ($\\alpha=1.0$)'
        },
        {
            'Model': 'Lasso',
            'Type': 'Linear',
            'Description': 'Linear regression with L1 penalty',
            'Regularization': 'L1 ($\\alpha=0.1$)'
        },
        {
            'Model': 'Elastic Net',
            'Type': 'Linear',
            'Description': 'Linear regression with L1+L2 penalty',
            'Regularization': 'L1+L2 ($\\alpha=0.1$)'
        },
        {
            'Model': 'Decision Tree',
            'Type': 'Tree',
            'Description': 'Recursive binary splits',
            'Regularization': 'max\\_depth=10'
        },
        {
            'Model': 'Random Forest',
            'Type': 'Ensemble',
            'Description': 'Bootstrap aggregated trees',
            'Regularization': '100 trees, max\\_depth=10'
        },
        {
            'Model': 'Gradient Boosting',
            'Type': 'Ensemble',
            'Description': 'Sequential boosted trees',
            'Regularization': '100 trees, max\\_depth=5'
        },
        {
            'Model': 'K-Nearest Neighbors',
            'Type': 'Instance',
            'Description': 'Averaging k=5 neighbors',
            'Regularization': 'None'
        },
        {
            'Model': 'Support Vector Regression',
            'Type': 'Kernel',
            'Description': 'RBF kernel SVM',
            'Regularization': 'C=1.0, $\\epsilon=0.1$'
        },
    ])

    latex_models = model_descriptions.to_latex(
        index=False,
        escape=False,
        column_format='l l p{4cm} p{3cm}',
        caption='Overview of machine learning models evaluated for throughput prediction.',
        label='tab:ml_models',
    )
    latex_output.append("% Model Descriptions\n" + latex_models)

    # Helper function to format results table
    def format_results_for_latex(df: pd.DataFrame, side_name: str) -> str:
        df_latex = df.copy()
        df_latex['Train Time (s)'] = df_latex['Train Time (s)'].apply(lambda x: f'{x:.4f}')
        df_latex['Test MAE'] = df_latex['Test MAE'].apply(lambda x: f'{x:.2e}')
        df_latex['Test R²'] = df_latex['Test R²'].apply(lambda x: f'{x:.6f}')
        df_latex['Test % Error'] = df_latex['Test % Error'].apply(lambda x: f'{x:.4f}')

        # Compact table with key metrics
        df_compact = df_latex[['Model', 'Train Time (s)', 'Test MAE', 'Test R²', 'Test % Error']]

        return df_compact.to_latex(
            index=False,
            escape=False,
            column_format='l r r r r',
            caption=f'Performance metrics for {side_name} throughput prediction (sorted by test percent error).',
            label=f'tab:{side_name}_results',
        )

    # Deserializer results
    if des_results is not None:
        latex_des = format_results_for_latex(des_results, 'deserializer')
        latex_output.append("\n% Deserializer Results\n" + latex_des)

    # Serializer results
    if ser_results is not None:
        latex_ser = format_results_for_latex(ser_results, 'serializer')
        latex_output.append("\n% Serializer Results\n" + latex_ser)

    # Introduction text
    introduction = r"""
% Introduction Text for LaTeX Document

\section{Baseline Model Comparison}

To establish a comprehensive understanding of the throughput prediction task,
we evaluated nine machine learning models spanning multiple algorithmic families:
linear models (Linear Regression, Ridge, Lasso, Elastic Net), tree-based models
(Decision Tree, Random Forest, Gradient Boosting), instance-based models
(K-Nearest Neighbors), and kernel-based models (Support Vector Regression).
This comparison serves to benchmark the neural network approach employed in the
primary training pipeline against classical regression methods.

\subsection{Experimental Setup}

All models were trained on both deserializer and serializer throughput prediction
tasks using preprocessed datasets with the same feature engineering pipeline as
the neural network approach. A standard 75\%-25\% train-test split (random state 42)
was used for evaluation. Features were standardized using scikit-learn's
\texttt{StandardScaler} fitted on the training set. Table~\ref{tab:ml_models}
provides an overview of the evaluated models and their hyperparameters.

We measured five evaluation metrics:
\begin{itemize}
    \item \textbf{Mean Absolute Error (MAE)}: Average absolute difference between
          predictions and actual values
    \item \textbf{Root Mean Squared Error (RMSE)}: Square root of average squared errors,
          penalizing larger errors more heavily
    \item \textbf{R² Score}: Coefficient of determination, measuring proportion of
          variance explained (1.0 = perfect, 0.0 = baseline)
    \item \textbf{Percent Error}: Mean absolute percentage error, matching the
          metric used in the neural network training pipeline
    \item \textbf{Training Time}: Wall-clock time for model fitting
\end{itemize}

Hyperparameters were set to reasonable defaults without extensive tuning to provide
a fair baseline comparison. All experiments used standardized features and identical
train-test splits for reproducibility.

\subsection{Results}

Tables~\ref{tab:deserializer_results} and~\ref{tab:serializer_results} present
the performance of all evaluated models for deserializer and serializer tasks,
respectively, sorted by test percent error.

Key observations:
\begin{itemize}
    \item Tree-based models (Decision Tree, Random Forest, Gradient Boosting) show
          extremely low errors, potentially indicating feature leakage that warrants
          further investigation
    \item Linear models provide interpretable baselines with reasonable performance
    \item Training times vary significantly, with Decision Trees being fastest and
          SVR/Gradient Boosting being slowest
\end{itemize}

\textbf{Note on Data Leakage:} The exceptionally low error rates for tree-based
models (near-zero in some cases) suggest potential data leakage where the target
variable may be encoded in features. This requires investigation before drawing
conclusions about absolute model performance.
"""
    latex_output.append("\n" + introduction)

    # Save to file
    output_path = os.path.join(output_dir, 'ml_comparison_latex.tex')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(latex_output))

    print(f"  Saved: {output_path}")


def run_experiments_for_side(
    dataset_path: str,
    side: str,
    test_size: float,
    one_hot_bmark: bool,
    no_feat_distributions: bool,
    output_dir: str,
) -> pd.DataFrame:
    """Run complete experiment pipeline for one side (des or ser)."""
    side_name = SIDE_TO_NAME[side]
    print(f"\n{'='*80}")
    print(f"PROCESSING {side_name.upper()}")
    print(f"{'='*80}")

    # Load dataset
    dataset = load_dataset(dataset_path)

    # Filter to side if needed
    if "side" in dataset.columns:
        dataset = dataset[dataset["side"] == side].copy()
        print(f"Filtered to {len(dataset)} rows for side '{side}'")

    # Preprocess
    benchmark_categories = None
    if one_hot_bmark:
        benchmark_categories = sorted(dataset["bench"].dropna().unique().tolist())

    model_df = pre_process_dataset(
        dataset,
        side,
        one_hot_bmark,
        no_feat_distributions=no_feat_distributions,
        benchmark_categories=benchmark_categories,
    )

    print(f"Preprocessed dataset: {model_df.shape[0]} rows, {model_df.shape[1]} columns")

    # Split features and labels
    X, y = split_features_and_labels(model_df)
    print(f"Features: {X.shape[1]}, Labels: {len(y)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Evaluate models
    results_df = evaluate_models(X_train, X_test, y_train, y_test, side_name)

    # Save plots
    save_plots(results_df, side_name, output_dir)

    # Save results CSV
    csv_path = os.path.join(output_dir, f'{side_name}_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV: {csv_path}")

    return results_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ML model comparison experiments for Lynx throughput prediction."
    )
    parser.add_argument(
        "--des-data",
        default="data/hpb_verilator/des_training_data.csv",
        help="Path to deserializer training data CSV",
    )
    parser.add_argument(
        "--ser-data",
        default="data/hpb_verilator/ser_training_data.csv",
        help="Path to serializer training data CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="ml_comparison_results",
        help="Output directory for results, plots, and LaTeX tables",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of data to use for testing (default: 0.25)",
    )
    parser.add_argument(
        "--one-hot-bmark",
        action="store_true",
        help="Use one-hot encoded benchmarks instead of analytical features",
    )
    parser.add_argument(
        "--no-feat-distributions",
        action="store_true",
        help="Exclude distribution features from analytical features",
    )
    parser.add_argument(
        "--des-only",
        action="store_true",
        help="Only run experiments for deserializer",
    )
    parser.add_argument(
        "--ser-only",
        action="store_true",
        help="Only run experiments for serializer",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()

    print(f"\n{'='*80}")
    print("LYNX ML MODEL COMPARISON EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"One-hot benchmarks: {args.one_hot_bmark}")
    print(f"Exclude distributions: {args.no_feat_distributions}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    des_results = None
    ser_results = None

    # Run deserializer experiments
    if not args.ser_only:
        try:
            des_results = run_experiments_for_side(
                args.des_data,
                "des",
                args.test_size,
                args.one_hot_bmark,
                args.no_feat_distributions,
                args.output_dir,
            )
        except Exception as e:
            print(f"\n ERROR: Failed to process deserializer: {e}")

    # Run serializer experiments
    if not args.des_only:
        try:
            ser_results = run_experiments_for_side(
                args.ser_data,
                "ser",
                args.test_size,
                args.one_hot_bmark,
                args.no_feat_distributions,
                args.output_dir,
            )
        except Exception as e:
            print(f"\n ERROR: Failed to process serializer: {e}")

    # Generate LaTeX tables
    generate_latex_tables(des_results, ser_results, args.output_dir)

    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {args.output_dir}/")
    print(f"  - Individual plots: <side>_<metric>.png")
    print(f"  - Results CSVs: <side>_results.csv")
    print(f"  - LaTeX tables: ml_comparison_latex.tex")


if __name__ == "__main__":
    main()
