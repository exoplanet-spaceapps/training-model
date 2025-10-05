#!/bin/bash

# NASA Exoplanet ML Models - Local Batch Execution Script
# Runs all models sequentially with unified pipeline

set -e  # Exit on error (but we'll handle errors manually)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "=================================================="
echo "NASA Exoplanet ML - Batch Training"
echo "Started: $TIMESTAMP"
echo "=================================================="
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p artifacts
mkdir -p results
mkdir -p artifacts/xgboost
mkdir -p artifacts/random_forest
mkdir -p artifacts/mlp
mkdir -p artifacts/logistic_regression
mkdir -p artifacts/svm

# Models to train
MODELS=("xgboost" "random_forest" "mlp" "logistic_regression" "svm")
SUCCESSFUL=0
FAILED=0
FAILED_MODELS=()

# Function to run a model
run_model() {
    local model=$1
    echo ""
    echo "=================================================="
    echo "Training: $model"
    echo "=================================================="

    START_TIME=$(date +%s)

    if python -m src.models.$model.train --config configs/local.yaml; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "${GREEN}✓ $model completed successfully (${DURATION}s)${NC}"
        ((SUCCESSFUL++))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "${RED}✗ $model failed (${DURATION}s)${NC}"
        ((FAILED++))
        FAILED_MODELS+=("$model")
    fi
}

# Train all models
for model in "${MODELS[@]}"; do
    run_model "$model"
done

# Generate benchmark summary
echo ""
echo "=================================================="
echo "Generating benchmark summary..."
echo "=================================================="

python scripts/generate_summary.py

# Final summary
END_TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "=================================================="
echo "Batch Training Complete"
echo "=================================================="
echo "Finished: $END_TIMESTAMP"
echo ""
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
    echo "Failed models: ${FAILED_MODELS[@]}"
fi
echo ""
echo "Results saved to: results/benchmark_summary.md"
echo "=================================================="

# Exit with error if any model failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
