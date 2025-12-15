#!/usr/bin/env python3
"""
Plot Features Script for Protobuf Analysis

This script reads extracted_features.json and plots distribution features
for all benchmarks on a single plot.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Any

# Available distribution features to plot
DISTRIBUTION_FEATURES = [
    "size_bytes_distribution",
    "total_fields_distribution",
    "nested_message_count_distribution",
    "depth_counter_list"
]

# Map feature names to their min/max keys and x-axis labels
FEATURE_CONFIG = {
    "size_bytes_distribution": {
        "min_key": "min_size_bytes",
        "max_key": "max_size_bytes",
        "xlabel": "Size (bytes)"
    },
    "total_fields_distribution": {
        "min_key": "min_total_fields",
        "max_key": "max_total_fields",
        "xlabel": "Total Fields"
    },
    "nested_message_count_distribution": {
        "min_key": "min_nested_message_count",
        "max_key": "max_nested_message_count",
        "xlabel": "Nested Message Count"
    },
    "depth_counter_list": {
        "min_key": None,
        "max_key": None,
        "xlabel": "Depth"
    }
}

def load_features(json_path: Path) -> Dict[str, Any]:
    """Load features from JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Could not find {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_distribution_feature(
    features: Dict[str, Any],
    feature_name: str,
    output_path: Path = None
):
    """
    Plot a distribution feature for all benchmarks on a single plot.
    
    Args:
        features: Dictionary of benchmark features
        feature_name: Name of the distribution feature to plot
        output_path: Optional path to save the plot
    """
    if feature_name not in DISTRIBUTION_FEATURES:
        raise ValueError(
            f"Unknown feature: {feature_name}. "
            f"Available features: {', '.join(DISTRIBUTION_FEATURES)}"
        )
    
    # Get feature configuration
    config = FEATURE_CONFIG[feature_name]
    xlabel = config["xlabel"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each benchmark
    for benchmark_name in sorted(features.keys()):
        benchmark_data = features[benchmark_name]
        
        if feature_name not in benchmark_data:
            print(f"Warning: {feature_name} not found in {benchmark_name}, skipping...")
            continue
        
        distribution = benchmark_data[feature_name]
        
        # Create x-axis values based on feature type
        if feature_name == "depth_counter_list":
            # For depth_counter_list, x-axis is just the depth indices (0, 1, 2, ...)
            x = np.arange(len(distribution))
        else:
            # For distribution features, use min/max to create bin centers
            min_key = config["min_key"]
            max_key = config["max_key"]
            min_val = benchmark_data.get(min_key, 0)
            max_val = benchmark_data.get(max_key, len(distribution) - 1)
            
            # Create bin centers from min to max
            num_bins = len(distribution)
            if num_bins > 1:
                # Create bin edges and use bin centers
                bin_edges = np.linspace(min_val, max_val, num_bins + 1)
                x = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers
            else:
                x = np.array([min_val])
        
        # Plot the distribution
        ax.plot(x, distribution, marker='o', label=benchmark_name, linewidth=2, markersize=6)
    
    # Customize plot
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel('Frequency / Count', fontsize=18)
    ax.set_title(f'{feature_name.replace("_", " ").title()}', fontsize=20, fontweight='bold')
    ax.legend(loc='upper right', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Increase tick mark font sizes
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.close(fig)  # Close figure to free memory
    else:
        plt.show()

def main():
    """Main function to plot features."""
    parser = argparse.ArgumentParser(
        description='Plot distribution features from extracted_features.json'
    )
    parser.add_argument(
        '--feature',
        type=str,
        choices=DISTRIBUTION_FEATURES,
        default=None,
        help='Distribution feature to plot (default: plot all features)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to extracted_features.json (default: same directory as script)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='plots',
        help='Path to save the plots directory (default: plots)'
    )
    
    args = parser.parse_args()
    
    # Determine input path
    script_dir = Path(__file__).parent
    if args.input:
        json_path = Path(args.input)
    else:
        json_path = script_dir / "extracted_features.json"
    
    # Load features
    print(f"Loading features from {json_path}...")
    features = load_features(json_path)
    
    # Determine output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Determine which features to plot
    features_to_plot = [args.feature] if args.feature else DISTRIBUTION_FEATURES
    
    # Plot each feature
    for feature_name in features_to_plot:
        output_path = output_dir / f"{feature_name}.png"
        print(f"Plotting {feature_name}...")
        plot_distribution_feature(features, feature_name, output_path)
    
    print(f"\nDone! Plotted {len(features_to_plot)} feature(s) to {output_dir}")

if __name__ == "__main__":
    main()
