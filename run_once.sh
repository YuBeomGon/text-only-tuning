#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config.yaml> <run_dir>"
  exit 1
fi

CONFIG_PATH="$1"
RUN_DIR="$2"

mkdir -p "$RUN_DIR"

# Replace these with your real Python entrypoints.
EVAL_JSON="${RUN_DIR}/metrics.json"
RAW_JSON="${RUN_DIR}/raw_predictions.json"
LOG_FILE="${RUN_DIR}/run.log"

echo "[INFO] Starting deterministic run" | tee "$LOG_FILE"
echo "[INFO] Config: ${CONFIG_PATH}" | tee -a "$LOG_FILE"
echo "[INFO] Output dir: ${RUN_DIR}" | tee -a "$LOG_FILE"

# Example placeholders:
# 1) optional training stage
# python -m src.train.train_text_only --config "$CONFIG_PATH" --output_dir "$RUN_DIR" 2>&1 | tee -a "$LOG_FILE"
#
# 2) eval stage
# python -m src.eval.run_eval --config "$CONFIG_PATH" --output_json "$EVAL_JSON" --raw_json "$RAW_JSON" 2>&1 | tee -a "$LOG_FILE"

# Temporary stub so the harness is runnable before integration.
cat > "$EVAL_JSON" <<'JSON'
{
  "run_id": "stub",
  "cer": 0.065,
  "domain_term_recall": 0.710,
  "hallucination_rate": 0.012,
  "timestamp_error": 0.085,
  "latency_ms": 640
}
JSON

cat > "$RAW_JSON" <<'JSON'
{
  "note": "replace this stub with real eval outputs"
}
JSON

echo "[INFO] Wrote stub metrics to ${EVAL_JSON}" | tee -a "$LOG_FILE"
echo "[INFO] Completed run_once" | tee -a "$LOG_FILE"
