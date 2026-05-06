#!/bin/bash

# Sweep OOD benchmark onboarding cost for Lynx train.py (HyperProtoBench bench0..bench5).
# For each held-out benchmark, trains with 0..N OOD rows added to the training set and
# evaluates on the remaining OOD rows (see --ood-benchmark / --ood-train-size in train.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_PATH="${DATASET_PATH:-data/training_data.csv}"
PYTHON_SCRIPT="train.py"
# Default: deserializer. Override: SIDE=ser ./sweep_ood_benchmark_onboarding.sh
SIDE="${SIDE:-des}"

BENCHMARKS=(bench0 bench1 bench2 bench3 bench4 bench5)
# Each size must be strictly less than the row count for that benchmark (same op/side).
ONBOARDING_SIZES=(0 4 8 16 32 64 128)

echo "Starting Lynx OOD benchmark onboarding sweep"
echo "Dataset: $DATASET_PATH"
echo "Side: $SIDE"
echo "OOD benchmarks: ${BENCHMARKS[*]}"
echo "Onboarding sizes: ${ONBOARDING_SIZES[*]}"
echo "=========================================="

mkdir -p sweep_results
SWEEP_DIR="${SWEEP_DIR:-sweep_results/ood_bench_onboarding_${SIDE}}"
mkdir -p "$SWEEP_DIR"

SWEEP_LOG="$SWEEP_DIR/sweep_log.txt"
echo "Sweep started at $(date)" > "$SWEEP_LOG"
echo "Dataset: $DATASET_PATH" >> "$SWEEP_LOG"
echo "Side: $SIDE" >> "$SWEEP_LOG"
echo "OOD benchmarks: ${BENCHMARKS[*]}" >> "$SWEEP_LOG"
echo "========================================" >> "$SWEEP_LOG"

for benchmark in "${BENCHMARKS[@]}"; do
    for onboarding_size in "${ONBOARDING_SIZES[@]}"; do
        echo ""
        echo "Running: OOD benchmark=$benchmark | ood_train_size=$onboarding_size | side=$SIDE"
        echo "Time: $(date)"

        echo "" >> "$SWEEP_LOG"
        echo "OOD benchmark: $benchmark | OOD train size: $onboarding_size | side: $SIDE - Started at $(date)" >> "$SWEEP_LOG"

        RUN_DIR="$SWEEP_DIR/ood_benchmark_${benchmark}/ood_train_size_${onboarding_size}"
        mkdir -p "$RUN_DIR"
        OUTPUT_DIR="$RUN_DIR"

        if python "$PYTHON_SCRIPT" -d "$DATASET_PATH" --side "$SIDE" \
            --ood-benchmark "$benchmark" --ood-train-size "$onboarding_size" \
            --output-dir "$OUTPUT_DIR" > "$RUN_DIR/training_output.txt" 2>&1; then
            echo "✓ Completed: $benchmark | ood_train_size=$onboarding_size"
            echo "OOD benchmark: $benchmark | OOD train size: $onboarding_size - COMPLETED at $(date)" >> "$SWEEP_LOG"
        else
            echo "✗ Failed: $benchmark | ood_train_size=$onboarding_size"
            echo "OOD benchmark: $benchmark | OOD train size: $onboarding_size - FAILED at $(date)" >> "$SWEEP_LOG"
        fi
    done
done

echo ""
echo "=========================================="
echo "Sweep completed at $(date)"
echo "Results: $SWEEP_DIR"
echo "Sweep completed at $(date)" >> "$SWEEP_LOG"

SUMMARY_FILE="$SWEEP_DIR/summary.txt"
{
    echo "Lynx OOD benchmark onboarding sweep summary"
    echo "Generated at: $(date)"
    echo "Dataset: $DATASET_PATH"
    echo "Side: $SIDE"
    echo ""
} > "$SUMMARY_FILE"

for benchmark in "${BENCHMARKS[@]}"; do
    for onboarding_size in "${ONBOARDING_SIZES[@]}"; do
        RUN_DIR="$SWEEP_DIR/ood_benchmark_${benchmark}/ood_train_size_${onboarding_size}"
        if [ -f "$RUN_DIR/training_output.txt" ]; then
            echo "OOD benchmark: $benchmark | OOD train size: $onboarding_size | side: $SIDE" >> "$SUMMARY_FILE"
            if grep -q "Percent Error" "$RUN_DIR/training_output.txt"; then
                FINAL_METRICS=$(tail -20 "$RUN_DIR/training_output.txt" | grep "Percent Error" | tail -1)
                PERCENT_ERROR=$(echo "$FINAL_METRICS" | grep -oE "Percent Error: [0-9]+\.[0-9]+%" | cut -d' ' -f3)
                echo "  Final Percent Error: $PERCENT_ERROR" >> "$SUMMARY_FILE"
            fi
            echo "" >> "$SUMMARY_FILE"
        fi
    done
done

echo "Summary: $SUMMARY_FILE"
