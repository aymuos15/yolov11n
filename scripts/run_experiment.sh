#!/bin/bash
#
# YOLO11n Experiment Pipeline
#
# Usage:
#   ./scripts/run_experiment.sh <DATASET>              # Run full pipeline
#   ./scripts/run_experiment.sh <DATASET> train        # Only training
#   ./scripts/run_experiment.sh <DATASET> export       # Only ONNX export
#   ./scripts/run_experiment.sh <DATASET> compare      # Only comparison
#
# Examples:
#   ./scripts/run_experiment.sh TXL
#   ./scripts/run_experiment.sh Cellpose
#   ./scripts/run_experiment.sh Cellpose train
#

set -Eeuo pipefail
cd "$(dirname "$0")/.."

# Check for dataset argument
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <DATASET> [train|export|compare|all]"
    echo ""
    echo "Available datasets:"
    ls -1 config/datasets/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//'
    exit 1
fi

DATASET="$1"
CONFIG="config/datasets/${DATASET}.yaml"

# Validate config exists
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config not found: $CONFIG"
    echo ""
    echo "Available datasets:"
    ls -1 config/datasets/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//'
    exit 1
fi

echo "Using config: $CONFIG"

case "${2:-all}" in
    train)
        python -m src.run_train --config "$CONFIG" "${@:3}"
        ;;
    export)
        python -m src.run_export --dataset "$DATASET" "${@:3}"
        ;;
    compare)
        python -m src.run_comparison --dataset "$DATASET" "${@:3}"
        ;;
    all)
        python -m src.run_train --config "$CONFIG" "${@:3}"
        python -m src.run_export --dataset "$DATASET"
        python -m src.run_comparison --dataset "$DATASET"
        ;;
    *)
        echo "Usage: $0 <DATASET> [train|export|compare|all]"
        exit 1
        ;;
esac
