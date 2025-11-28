#!/bin/bash
#
# YOLO11n Experiment Pipeline
#
# Usage:
#   ./scripts/run_experiment.sh                # Run full pipeline (train -> export -> compare)
#   ./scripts/run_experiment.sh train          # Only training
#   ./scripts/run_experiment.sh export         # Only ONNX export
#   ./scripts/run_experiment.sh compare        # Only comparison
#

set -Eeuo pipefail
cd "$(dirname "$0")/.."

# Extract dataset from config file
CONFIG="${CONFIG:-config/experiment.yaml}"
DATASET=$(grep "^dataset:" "$CONFIG" | awk '{print $2}')

case "${1:-all}" in
    train)
        python -m src.run_train "${@:2}"
        ;;
    export)
        python -m src.run_export --dataset "$DATASET" "${@:2}"
        ;;
    compare)
        python -m src.run_comparison --dataset "$DATASET" "${@:2}"
        ;;
    all)
        python -m src.run_train "${@:2}"
        python -m src.run_export --dataset "$DATASET"
        python -m src.run_comparison --dataset "$DATASET"
        ;;
    *)
        echo "Usage: $0 [train|export|compare|all]"
        exit 1
        ;;
esac
