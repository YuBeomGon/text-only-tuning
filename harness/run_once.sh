#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config.yaml> <run_dir>"
  exit 1
fi

CONFIG_PATH="$1"
RUN_DIR="$2"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/run.log"
echo "[INFO] Starting run" | tee "$LOG_FILE"
echo "[INFO] Config: ${CONFIG_PATH}" | tee -a "$LOG_FILE"

ALPHA=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['prior']['alpha'])")
PRIOR_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['prior'].get('prior_path', 'none'))")
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['serving']['model_path'])")
MANIFEST=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c['datasets']['eval']['in_domain_clean'])")
LEXICON=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c['domain'].get('lexicon_path', ''))")
LANGUAGE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['domain'].get('language', 'ko'))")

PRIOR_ARG=""
if [[ "$PRIOR_PATH" != "none" ]] && [[ -f "$PRIOR_PATH" ]]; then
  PRIOR_ARG="--prior_path $PRIOR_PATH"
fi

LEXICON_ARG=""
if [[ -n "$LEXICON" ]] && [[ -f "$LEXICON" ]]; then
  LEXICON_ARG="--lexicon $LEXICON"
fi

python -m src.eval.run_eval \
  --model_name "$MODEL_NAME" \
  --manifest "$MANIFEST" \
  $PRIOR_ARG \
  --alpha "$ALPHA" \
  $LEXICON_ARG \
  --output_json "${RUN_DIR}/metrics.json" \
  --language "$LANGUAGE" \
  2>&1 | tee -a "$LOG_FILE"

echo "[INFO] Completed run_once" | tee -a "$LOG_FILE"
