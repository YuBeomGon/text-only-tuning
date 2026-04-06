#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [baseline_metrics.json]"
  exit 1
fi

CONFIG_PATH="$1"
BASELINE_METRICS="${2:-outputs/baseline/metrics.json}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH"
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
CONFIG_NAME="$(basename "$CONFIG_PATH" .yaml)"
RUN_ID="${TIMESTAMP}_${CONFIG_NAME}_${GIT_SHA}"

RUN_DIR="outputs/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

export RUN_ID
export RUN_DIR
export CONFIG_PATH
export BASELINE_METRICS

echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] RUN_DIR=${RUN_DIR}"
echo "[INFO] CONFIG=${CONFIG_PATH}"

cp "$CONFIG_PATH" "${RUN_DIR}/config.snapshot.yaml"

bash harness/run_once.sh "$CONFIG_PATH" "$RUN_DIR"

if [[ ! -f "${RUN_DIR}/metrics.json" ]]; then
  echo "[ERROR] metrics.json not produced"
  exit 1
fi

bash harness/score.sh "$BASELINE_METRICS" "${RUN_DIR}/metrics.json" | tee "${RUN_DIR}/score.txt"

echo "[INFO] Finished run: ${RUN_ID}"
echo "[INFO] Artifacts:"
echo "  - ${RUN_DIR}/config.snapshot.yaml"
echo "  - ${RUN_DIR}/metrics.json"
echo "  - ${RUN_DIR}/score.txt"
