#!/usr/bin/env bash
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 BASELINE_JSON CURRENT_BEST_JSON CANDIDATE_JSON [RUN_ID]" >&2
  exit 2
fi

BASELINE_JSON="$1"
CURRENT_BEST_JSON="$2"
CANDIDATE_JSON="$3"
RUN_ID="${4:-unknown}"

for f in "$BASELINE_JSON" "$CURRENT_BEST_JSON" "$CANDIDATE_JSON"; do
  if [ ! -f "$f" ]; then
    echo "Missing json: $f" >&2
    exit 2
  fi
done

export CER_REL_PASS="${CER_REL_PASS:-0.03}"
export TERM_RECALL_REL_PASS="${TERM_RECALL_REL_PASS:-0.05}"
export HALLUCINATION_ABS_MAX="${HALLUCINATION_ABS_MAX:-0.01}"
export TIMESTAMP_REL_MAX="${TIMESTAMP_REL_MAX:-0.05}"
export LATENCY_REL_MAX="${LATENCY_REL_MAX:-0.10}"
export CER_HOLD_FLOOR="${CER_HOLD_FLOOR:--0.02}"
export TERM_HOLD_FLOOR="${TERM_HOLD_FLOOR:--0.03}"

python3 - "$BASELINE_JSON" "$CURRENT_BEST_JSON" "$CANDIDATE_JSON" "$RUN_ID" <<'PY'
import json, os, sys
from pathlib import Path

baseline = json.loads(Path(sys.argv[1]).read_text("utf-8"))
current_best = json.loads(Path(sys.argv[2]).read_text("utf-8"))
candidate = json.loads(Path(sys.argv[3]).read_text("utf-8"))
run_id = sys.argv[4]

def f(x): return float(x)
def rel_gain(old, new):
    """Relative gain where lower is better (CER, error rates). Positive = improvement."""
    if old == 0: return 0.0
    return (old - new) / old
def rel_gain_higher(old, new):
    """Relative gain where higher is better (recall). Positive = improvement."""
    if old == 0: return 1.0 if new > 0 else 0.0
    return (new - old) / old

# --- Stage 1: Viability check (candidate vs baseline) ---
bl_cer_gain = rel_gain(f(baseline["cer"]), f(candidate["cer"]))
bl_term_gain = rel_gain_higher(f(baseline["domain_term_recall"]), f(candidate["domain_term_recall"]))

# Safety: absolute and relative checks
c_hall = f(candidate["hallucination_rate"])
hall_abs_max = float(os.environ["HALLUCINATION_ABS_MAX"])
ts_increase = (f(candidate["timestamp_error"]) - f(baseline["timestamp_error"])) / f(baseline["timestamp_error"]) if f(baseline["timestamp_error"]) else 0.0
lat_increase = (f(candidate["latency_ms"]) - f(baseline["latency_ms"])) / f(baseline["latency_ms"]) if f(baseline["latency_ms"]) else 0.0

viable = bl_cer_gain > 0 or bl_term_gain > 0
safety_ok = (
    c_hall <= hall_abs_max
    and ts_increase <= float(os.environ["TIMESTAMP_REL_MAX"])
    and lat_increase <= float(os.environ["LATENCY_REL_MAX"])
)

# HUMAN_REVIEW check BEFORE PASS (higher priority than promotion)
safety_borderline = (
    c_hall > hall_abs_max * 0.8
    or ts_increase > float(os.environ["TIMESTAMP_REL_MAX"]) * 0.8
    or lat_increase > float(os.environ["LATENCY_REL_MAX"]) * 0.8
)

cb_cer_gain = 0.0
cb_term_gain = 0.0

if not viable or not safety_ok:
    decision = "FAIL"
elif safety_borderline:
    decision = "HUMAN_REVIEW"
else:
    # --- Stage 2: Promotion check (candidate vs current_best) ---
    cb_cer_gain = rel_gain(f(current_best["cer"]), f(candidate["cer"]))
    cb_term_gain = rel_gain_higher(f(current_best["domain_term_recall"]), f(candidate["domain_term_recall"]))

    cer_passes = cb_cer_gain >= float(os.environ["CER_REL_PASS"])
    term_passes = cb_term_gain >= float(os.environ["TERM_RECALL_REL_PASS"])
    cer_acceptable = cb_cer_gain >= float(os.environ["CER_HOLD_FLOOR"])
    term_acceptable = cb_term_gain >= float(os.environ["TERM_HOLD_FLOOR"])

    if cer_passes and term_passes:
        decision = "PASS"
    elif (cer_passes and term_acceptable) or (term_passes and cer_acceptable):
        decision = "REPRO_REQUIRED"
    else:
        decision = "VIABLE"

summary = {
    "run_id": run_id,
    "decision": decision,
    "viability": {"cer_gain_vs_baseline": round(bl_cer_gain, 6), "term_gain_vs_baseline": round(bl_term_gain, 6)},
    "promotion": {"cer_gain_vs_best": round(cb_cer_gain, 6), "term_gain_vs_best": round(cb_term_gain, 6)},
    "safety": {"hall_abs": round(c_hall, 6), "ts_increase": round(ts_increase, 6), "lat_increase": round(lat_increase, 6)},
}
print(f"DECISION={decision}")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
