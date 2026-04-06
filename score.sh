#!/usr/bin/env bash
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 BASELINE_JSON CANDIDATE_JSON [RUN_ID]" >&2
  exit 2
fi

BASELINE_JSON="$1"
CANDIDATE_JSON="$2"
RUN_ID="${3:-unknown}"

if [ ! -f "$BASELINE_JSON" ]; then
  echo "Missing baseline json: $BASELINE_JSON" >&2
  exit 2
fi

if [ ! -f "$CANDIDATE_JSON" ]; then
  echo "Missing candidate json: $CANDIDATE_JSON" >&2
  exit 2
fi

export CER_REL_PASS="${CER_REL_PASS:-0.03}"
export TERM_RECALL_REL_PASS="${TERM_RECALL_REL_PASS:-0.05}"
export HALLUCINATION_ABS_MAX="${HALLUCINATION_ABS_MAX:-0.01}"
export TIMESTAMP_REL_MAX="${TIMESTAMP_REL_MAX:-0.05}"
export LATENCY_REL_MAX="${LATENCY_REL_MAX:-0.10}"

python3 - "$BASELINE_JSON" "$CANDIDATE_JSON" "$RUN_ID" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
run_id = sys.argv[3]

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

required = [
    "cer",
    "domain_term_recall",
    "hallucination_rate",
    "timestamp_error",
    "latency_ms",
]

for name in required:
    if name not in baseline:
        raise SystemExit(f"Missing metric '{name}' in baseline JSON")
    if name not in candidate:
        raise SystemExit(f"Missing metric '{name}' in candidate JSON")


def f(x):
    return float(x)

b_cer = f(baseline["cer"])
c_cer = f(candidate["cer"])
b_term = f(baseline["domain_term_recall"])
c_term = f(candidate["domain_term_recall"])
b_hall = f(baseline["hallucination_rate"])
c_hall = f(candidate["hallucination_rate"])
b_ts = f(baseline["timestamp_error"])
c_ts = f(candidate["timestamp_error"])
b_lat = f(baseline["latency_ms"])
c_lat = f(candidate["latency_ms"])

cer_rel_gain = (b_cer - c_cer) / b_cer if b_cer else 0.0
term_rel_gain = (c_term - b_term) / b_term if b_term else (1.0 if c_term > b_term else 0.0)
hall_abs_increase = c_hall - b_hall
ts_rel_increase = (c_ts - b_ts) / b_ts if b_ts else (0.0 if c_ts <= b_ts else 1.0)
lat_rel_increase = (c_lat - b_lat) / b_lat if b_lat else (0.0 if c_lat <= b_lat else 1.0)

pass_rules = {
    "cer": cer_rel_gain >= float(os.environ["CER_REL_PASS"]),
    "term_recall": term_rel_gain >= float(os.environ["TERM_RECALL_REL_PASS"]),
    "hallucination": hall_abs_increase <= float(os.environ["HALLUCINATION_ABS_MAX"]),
    "timestamp": ts_rel_increase <= float(os.environ["TIMESTAMP_REL_MAX"]),
    "latency": lat_rel_increase <= float(os.environ["LATENCY_REL_MAX"]),
}

score = (
    4.0 * cer_rel_gain
    + 3.0 * term_rel_gain
    - 3.0 * max(0.0, hall_abs_increase)
    - 1.5 * max(0.0, ts_rel_increase)
    - 1.0 * max(0.0, lat_rel_increase)
)

safety_ok = pass_rules["hallucination"] and pass_rules["timestamp"] and pass_rules["latency"]

cer_passes = pass_rules["cer"]
term_passes = pass_rules["term_recall"]

# HOLD tolerances: the non-improving axis must not regress beyond this floor.
CER_HOLD_FLOOR = -0.02       # allow up to 2% relative CER regression
TERM_HOLD_FLOOR = -0.03      # allow up to 3% relative term recall regression

cer_acceptable = cer_rel_gain >= CER_HOLD_FLOOR
term_acceptable = term_rel_gain >= TERM_HOLD_FLOOR

# PASS:  both axes meet their thresholds + safety
# HOLD:  one axis meets its threshold, the other stays within hold floor + safety
# FAIL:  everything else
if cer_passes and term_passes and safety_ok:
    decision = "PASS"
elif (cer_passes and term_acceptable and safety_ok) or \
     (term_passes and cer_acceptable and safety_ok):
    decision = "HOLD"
else:
    decision = "FAIL"

summary = {
    "run_id": run_id,
    "decision": decision,
    "score": round(score, 6),
    "deltas": {
        "cer_rel_gain": round(cer_rel_gain, 6),
        "term_recall_rel_gain": round(term_rel_gain, 6),
        "hallucination_abs_increase": round(hall_abs_increase, 6),
        "timestamp_rel_increase": round(ts_rel_increase, 6),
        "latency_rel_increase": round(lat_rel_increase, 6),
    },
    "rules": pass_rules,
}

print(f"DECISION={decision}")
print(f"SCORE={score:.6f}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
