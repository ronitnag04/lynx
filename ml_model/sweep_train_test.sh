#!/bin/bash

# Sweep train/test split ratios for Lynx train.py.
# For each test size, train on the selected side and store per-run outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_PATH="${DATASET_PATH:-data/training_data.csv}"
PYTHON_SCRIPT="train.py"
# Default: deserializer. Override: SIDE=ser ./sweep_train_test.sh
SIDE="${SIDE:-des}"
ONE_HOT_BMARK="${ONE_HOT_BMARK:-0}"

# Fractions passed directly to train.py --test-size.
TEST_SIZES=(0.25 0.5 0.75 0.8 0.85 0.9 0.95 0.955 0.96 0.965 0.97 0.975 0.98 0.985 0.99)


echo "Starting Lynx train/test split sweep"
echo "Dataset: $DATASET_PATH"
echo "Side: $SIDE"
echo "Test sizes: ${TEST_SIZES[*]}"
echo "====================================="

mkdir -p sweep_results
SWEEP_DIR="${SWEEP_DIR:-sweep_results/train_test_split_${SIDE}}"
mkdir -p "$SWEEP_DIR"

SWEEP_LOG="$SWEEP_DIR/sweep_log.txt"
echo "Sweep started at $(date)" > "$SWEEP_LOG"
echo "Dataset: $DATASET_PATH" >> "$SWEEP_LOG"
echo "Side: $SIDE" >> "$SWEEP_LOG"
echo "Test sizes: ${TEST_SIZES[*]}" >> "$SWEEP_LOG"
echo "=====================================" >> "$SWEEP_LOG"

for test_size in "${TEST_SIZES[@]}"; do
    echo ""
    echo "Running: test_size=$test_size | side=$SIDE"
    echo "Time: $(date)"

    echo "" >> "$SWEEP_LOG"
    echo "Test size: $test_size | side: $SIDE - Started at $(date)" >> "$SWEEP_LOG"

    RUN_DIR="$SWEEP_DIR/test_size_${test_size}"
    mkdir -p "$RUN_DIR"
    OUTPUT_DIR="$RUN_DIR"

    if [ "$ONE_HOT_BMARK" -eq 1 ]; then
        ONE_HOT_BMARK_FLAG="--one-hot-bmark"
    else
        ONE_HOT_BMARK_FLAG=""
    fi

    if python "$PYTHON_SCRIPT" -d "$DATASET_PATH" --side "$SIDE" \
        --test-size "$test_size" --output-dir "$OUTPUT_DIR" $ONE_HOT_BMARK_FLAG > "$RUN_DIR/training_output.txt" 2>&1; then
        echo "✓ Completed: test_size=$test_size"
        echo "Test size: $test_size - COMPLETED at $(date)" >> "$SWEEP_LOG"
    else
        echo "✗ Failed: test_size=$test_size"
        echo "Test size: $test_size - FAILED at $(date)" >> "$SWEEP_LOG"
    fi
done

echo ""
echo "====================================="
echo "Sweep completed at $(date)"
echo "Results: $SWEEP_DIR"
echo "Sweep completed at $(date)" >> "$SWEEP_LOG"

SUMMARY_FILE="$SWEEP_DIR/summary.txt"
{
    echo "Lynx train/test split sweep summary"
    echo "Generated at: $(date)"
    echo "Dataset: $DATASET_PATH"
    echo "Side: $SIDE"
    echo ""
} > "$SUMMARY_FILE"

for test_size in "${TEST_SIZES[@]}"; do
    RUN_DIR="$SWEEP_DIR/test_size_${test_size}"
    if [ -f "$RUN_DIR/training_output.txt" ]; then
        echo "Test size: $test_size | side: $SIDE" >> "$SUMMARY_FILE"
        if grep -q "Percent Error" "$RUN_DIR/training_output.txt"; then
            FINAL_METRICS=$(tail -20 "$RUN_DIR/training_output.txt" | grep "Percent Error" | tail -1)
            PERCENT_ERROR=$(echo "$FINAL_METRICS" | grep -oE "Percent Error: [0-9]+\.[0-9]+%" | cut -d' ' -f3)
            echo "  Final Percent Error: $PERCENT_ERROR" >> "$SUMMARY_FILE"
        fi
        echo "" >> "$SUMMARY_FILE"
    fi
done

echo "Summary: $SUMMARY_FILE"
