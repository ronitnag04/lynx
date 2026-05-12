#!/bin/bash

# Run sweep_train_test.sh with different ablation configurations.
# Tests both serializer and deserializer sides with:
#   - Normal (baseline)
#   - One-hot benchmark encoding
#   - No feature distributions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SWEEP_SCRIPT="./sweep_train_test.sh"

if [ ! -f "$SWEEP_SCRIPT" ]; then
    echo "Error: $SWEEP_SCRIPT not found"
    exit 1
fi

echo "Starting Lynx ablation study sweep"
echo "Running train/test split sweeps for all ablation configurations"
echo "====================================="
echo ""

# Array of configurations: "SIDE ONE_HOT_BMARK NO_FEAT_DIST DATASET_PATH OUTPUT_DIR_SUFFIX"
CONFIGS=(
    "des 0 0 data/hpb_verilator/des_training_data.csv sweep_results/hpb_verilator/train_test_split_des"
    "ser 0 0 data/hpb_verilator/ser_training_data.csv sweep_results/hpb_verilator/train_test_split_ser"
    "des 0 1 data/hpb_verilator/des_training_data.csv sweep_results/hpb_verilator/train_test_split_des_no_feat_dist"
    "ser 0 1 data/hpb_verilator/ser_training_data.csv sweep_results/hpb_verilator/train_test_split_ser_no_feat_dist"
    "des 1 0 data/hpb_verilator/des_training_data.csv sweep_results/hpb_verilator/train_test_split_des_one_hot_bmark"
    "ser 1 0 data/hpb_verilator/ser_training_data.csv sweep_results/hpb_verilator/train_test_split_ser_one_hot_bmark"
)

TOTAL_CONFIGS=${#CONFIGS[@]}
CURRENT=0

for config in "${CONFIGS[@]}"; do
    read -r SIDE ONE_HOT_BMARK NO_FEAT_DIST DATASET_PATH SWEEP_DIR <<< "$config"
    CURRENT=$((CURRENT + 1))

    echo "====================================="
    echo "Configuration $CURRENT/$TOTAL_CONFIGS"
    echo "  Side: $SIDE"
    echo "  Dataset: $DATASET_PATH"
    echo "  One-hot benchmark: $ONE_HOT_BMARK"
    echo "  No feature distributions: $NO_FEAT_DIST"
    echo "  Output directory: $SWEEP_DIR"
    echo "  Started at: $(date)"
    echo "====================================="
    echo ""

    export SIDE="$SIDE"
    export DATASET_PATH="$DATASET_PATH"
    export ONE_HOT_BMARK="$ONE_HOT_BMARK"
    export NO_FEAT_DIST="$NO_FEAT_DIST"
    export SWEEP_DIR="$SWEEP_DIR"

    if bash "$SWEEP_SCRIPT"; then
        echo ""
        echo "✓ Configuration $CURRENT/$TOTAL_CONFIGS completed successfully"
        echo ""
    else
        echo ""
        echo "✗ Configuration $CURRENT/$TOTAL_CONFIGS failed"
        echo ""
    fi
done

echo "====================================="
echo "Ablation study sweep completed at $(date)"
echo "====================================="
echo ""
echo "Results summary:"
for config in "${CONFIGS[@]}"; do
    read -r SIDE ONE_HOT_BMARK NO_FEAT_DIST DATASET_PATH SWEEP_DIR <<< "$config"
    if [ -f "$SWEEP_DIR/summary.txt" ]; then
        echo "  ✓ $SWEEP_DIR"
    else
        echo "  ✗ $SWEEP_DIR (failed or incomplete)"
    fi
done
echo ""
