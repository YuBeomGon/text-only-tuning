import json
import subprocess
import tempfile
from pathlib import Path

SCORE_SH = str(Path(__file__).parent.parent / "harness" / "score.sh")


def _write_json(path, data):
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def _run_score(baseline, current_best, candidate, run_id="test"):
    with tempfile.TemporaryDirectory() as td:
        b = f"{td}/baseline.json"
        cb = f"{td}/current_best.json"
        c = f"{td}/candidate.json"
        _write_json(b, baseline)
        _write_json(cb, current_best)
        _write_json(c, candidate)
        result = subprocess.run(
            ["bash", SCORE_SH, b, cb, c, run_id],
            capture_output=True, text=True,
        )
        for line in result.stdout.strip().split("\n"):
            if line.startswith("DECISION="):
                return line.split("=", 1)[1]
    return "ERROR"


BASELINE = {
    "cer": 0.10, "domain_term_recall": 0.60,
    "hallucination_rate": 0.002, "timestamp_error": 0.08,
    "latency_ms": 500,
}


def test_pass():
    current_best = BASELINE.copy()
    candidate = {
        "cer": 0.09, "domain_term_recall": 0.65,
        "hallucination_rate": 0.002, "timestamp_error": 0.08,
        "latency_ms": 510,
    }
    assert _run_score(BASELINE, current_best, candidate) == "PASS"


def test_fail_worse_than_baseline():
    candidate = {
        "cer": 0.12, "domain_term_recall": 0.55,
        "hallucination_rate": 0.03, "timestamp_error": 0.10,
        "latency_ms": 600,
    }
    assert _run_score(BASELINE, BASELINE, candidate) == "FAIL"


def test_viable_beats_baseline_not_best():
    current_best = {
        "cer": 0.08, "domain_term_recall": 0.70,
        "hallucination_rate": 0.002, "timestamp_error": 0.08,
        "latency_ms": 500,
    }
    candidate = {
        "cer": 0.09, "domain_term_recall": 0.65,
        "hallucination_rate": 0.002, "timestamp_error": 0.08,
        "latency_ms": 510,
    }
    assert _run_score(BASELINE, current_best, candidate) == "VIABLE"
