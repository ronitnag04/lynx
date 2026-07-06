#!/bin/bash
#
# Full ML Model Training and Pareto Frontier Validation Workflow
#
# This script trains all ML models and compares their Pareto frontiers
# to validate throughput predictions across different model types.
#

set -e  # Exit on error

# Default configuration
SIDE="${SIDE:-des}"
DATASET_PATH="${DATASET_PATH:-data/hpb_verilator/${SIDE}_training_data.csv}"
FEATURES_FILE="${FEATURES_FILE:-../analytical_model/extracted_features.json}"
if [ "$SIDE" == "ser" ]; then
    RESULTS_DIR="${RESULTS_DIR:-results/hpb_verilator/serializer_ml_models}"
else
    RESULTS_DIR="${RESULTS_DIR:-results/hpb_verilator/deserializer_ml_models}"
fi
if [ "$SIDE" == "ser" ]; then
    COMPARISON_DIR="${COMPARISON_DIR:-results/hpb_verilator/serializer_ml_models/pareto_comparison_${SIDE}}"
else
    COMPARISON_DIR="${COMPARISON_DIR:-results/hpb_verilator/deserializer_ml_models/pareto_comparison_${SIDE}}"
fi
NUM_OBJECTIVES="${NUM_OBJECTIVES:-2}"
TEST_SIZE="${TEST_SIZE:-0.25}"

# Model selection (can be overridden by environment variable)
if [ -z "$MODELS" ]; then
    MODELS="neural LinearRegression Ridge Lasso ElasticNet DecisionTree RandomForest GradientBoosting KNN SVR"
fi

# Debug mode
DEBUG_MODE="${DEBUG_MODE:-0}"
LIMIT_CONFIGS="${LIMIT_CONFIGS:-0}"

# Print configuration
echo "=============================================================================="
echo "ML Model Training and Pareto Frontier Validation"
echo "=============================================================================="
echo "Side:              $SIDE"
echo "Dataset:           $DATASET_PATH"
echo "Features:          $FEATURES_FILE"
echo "Results Dir:       $RESULTS_DIR"
echo "Comparison Dir:    $COMPARISON_DIR"
echo "Models:            $MODELS"
echo "Num Objectives:    $NUM_OBJECTIVES"
echo "Test Size:         $TEST_SIZE"
if [ "$DEBUG_MODE" = "1" ] || [ "$LIMIT_CONFIGS" -gt 0 ]; then
    echo "Debug Mode:        Enabled (limit configs: $LIMIT_CONFIGS)"
fi
echo "=============================================================================="
echo ""

# Step 1: Train all models
echo "STEP 1: Training all ML models for $SIDE"
echo "=============================================================================="

TRAIN_CMD="python3 train_all_models.py \
    --side $SIDE \
    --dataset-path $DATASET_PATH \
    --output-dir $RESULTS_DIR \
    --test-size $TEST_SIZE"

if [ "$DEBUG_MODE" = "1" ]; then
    echo "Running: $TRAIN_CMD"
fi

eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "Training complete. Models saved to $RESULTS_DIR/"
echo ""

# Step 2: Compare Pareto frontiers
echo "STEP 2: Comparing Pareto frontiers across models"
echo "=============================================================================="

COMPARE_CMD="python3 compare_pareto_frontiers.py \
    --side $SIDE \
    --checkpoint-dir $RESULTS_DIR \
    --features-file $FEATURES_FILE \
    --output-dir $COMPARISON_DIR \
    --models $MODELS \
    --num-objectives $NUM_OBJECTIVES"

if [ "$LIMIT_CONFIGS" -gt 0 ]; then
    COMPARE_CMD="$COMPARE_CMD --limit-configs $LIMIT_CONFIGS"
fi

if [ "$DEBUG_MODE" = "1" ]; then
    echo "Running: $COMPARE_CMD"
fi

eval $COMPARE_CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Pareto frontier comparison failed"
    exit 1
fi

echo ""
echo "Comparison complete. Results saved to $COMPARISON_DIR/"
echo ""

# Step 3: Display summary
echo "STEP 3: Validation Summary"
echo "=============================================================================="

if [ -f "$COMPARISON_DIR/${SIDE}_pareto_statistics.csv" ]; then
    echo "Pareto Front Statistics:"
    cat "$COMPARISON_DIR/${SIDE}_pareto_statistics.csv"
    echo ""
fi

if [ -f "$COMPARISON_DIR/${SIDE}_comparison_summary.json" ]; then
    echo "Full summary available at: $COMPARISON_DIR/${SIDE}_comparison_summary.json"
fi

echo ""
echo "=============================================================================="
echo "VALIDATION COMPLETE"
echo "=============================================================================="
echo "Results:"
echo "  - Trained models:          $RESULTS_DIR/"
echo "  - Pareto frontiers:        $COMPARISON_DIR/pareto_${SIDE}_*.json"
echo "  - Statistics:              $COMPARISON_DIR/${SIDE}_pareto_statistics.csv"
echo "  - Overlap matrix:          $COMPARISON_DIR/${SIDE}_pareto_overlap.csv"
echo "  - Visualizations:          $COMPARISON_DIR/${SIDE}_pareto_comparison_*.png"
echo "  - Summary:                 $COMPARISON_DIR/${SIDE}_comparison_summary.json"
echo ""
echo "Validation Configs for Hardware Testing:"
echo "  - Combined JSON:           $COMPARISON_DIR/${SIDE}_pareto_validation_configs.json"
echo ""
echo "Next steps:"
echo "  1. Use sweep_configs/gen_sweep_configs.py to generate Scala configs"
echo "  2. Run synthesis on validation candidates"
echo "  3. Compare predictions with actual hardware measurements"
echo "=============================================================================="
