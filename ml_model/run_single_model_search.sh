#!/bin/bash
#
# Run exhaustive search with a single ML model type
#
# Usage: ./run_single_model_search.sh <model_type> [side] [output_name]
#

set -e

# Parse arguments
MODEL_TYPE="${1:-neural}"
SIDE="${2:-des}"
OUTPUT_NAME="${3:-pareto_${SIDE}_${MODEL_TYPE}.json}"

# Configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results}"
FEATURES_FILE="${FEATURES_FILE:-../analytical_model/extracted_features.json}"
NUM_OBJECTIVES="${NUM_OBJECTIVES:-3}"
KAPPA="${KAPPA:-0.0001}"
LIMIT_CONFIGS="${LIMIT_CONFIGS:-0}"

# Validate model type
VALID_MODELS="neural LinearRegression Ridge Lasso ElasticNet DecisionTree RandomForest GradientBoosting KNN SVR"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_TYPE"; then
    echo "ERROR: Invalid model type '$MODEL_TYPE'"
    echo "Valid types: $VALID_MODELS"
    exit 1
fi

echo "=============================================================================="
echo "Running Exhaustive Search with $MODEL_TYPE"
echo "=============================================================================="
echo "Model Type:        $MODEL_TYPE"
echo "Side:              $SIDE"
echo "Checkpoint Dir:    $CHECKPOINT_DIR"
echo "Features File:     $FEATURES_FILE"
echo "Output:            $OUTPUT_NAME"
echo "Num Objectives:    $NUM_OBJECTIVES"
echo "Kappa:             $KAPPA"
if [ "$LIMIT_CONFIGS" -gt 0 ]; then
    echo "Limit Configs:     $LIMIT_CONFIGS (debug mode)"
fi
echo "=============================================================================="
echo ""

# Build command
CMD="python3 exhaustive_search.py \
    --side $SIDE \
    --model-type $MODEL_TYPE \
    --checkpoint-dir $CHECKPOINT_DIR \
    --features-file $FEATURES_FILE \
    --output $OUTPUT_NAME \
    --num-objectives $NUM_OBJECTIVES \
    --kappa $KAPPA"

if [ "$LIMIT_CONFIGS" -gt 0 ]; then
    CMD="$CMD --limit-configs $LIMIT_CONFIGS"
fi

echo "Command: $CMD"
echo ""

# Run search
eval $CMD

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Exhaustive search failed"
    exit 1
fi

echo ""
echo "=============================================================================="
echo "SEARCH COMPLETE"
echo "=============================================================================="
echo "Results saved to: $OUTPUT_NAME"

# Display summary if jq is available
if command -v jq &> /dev/null; then
    echo ""
    echo "Pareto Front Summary:"
    echo "  Total configs evaluated:  $(jq -r '.stats.evaluated_configs' $OUTPUT_NAME)"
    echo "  Pareto points found:      $(jq -r '.stats.n_pareto_final' $OUTPUT_NAME)"
    echo "  Validation candidates:    $(jq -r '.stats.n_validation_candidates' $OUTPUT_NAME)"
    echo "  Duration:                 $(jq -r '.stats.total_duration_s' $OUTPUT_NAME)s"
    echo ""
    echo "Baseline throughput:        $(jq -r '.baseline.predicted_throughput_gbits_per_sec' $OUTPUT_NAME) Gbit/s"
fi

echo "=============================================================================="
