import pytest
from src.eval.metrics import compute_cer, compute_wer, compute_term_recall, detect_hallucination, compute_hallucination_rate


def test_cer_identical():
    assert compute_cer("hello world", "hello world") == 0.0


def test_cer_one_char_diff():
    result = compute_cer("hello", "hallo")
    assert abs(result - 0.2) < 1e-6


def test_wer_identical():
    assert compute_wer("hello world", "hello world") == 0.0


def test_wer_one_word_diff():
    result = compute_wer("hello world", "hello earth")
    assert abs(result - 0.5) < 1e-6


def test_term_recall_all_found():
    ref = "paracetamol is used for pain"
    hyp = "paracetamol is used for pain"
    terms = ["paracetamol"]
    assert compute_term_recall(ref, hyp, terms) == 1.0


def test_term_recall_none_found():
    ref = "paracetamol is used for pain"
    hyp = "acetaminophen is used for pain"
    terms = ["paracetamol"]
    assert compute_term_recall(ref, hyp, terms) == 0.0


def test_term_recall_partial():
    ref = "broncol and paracetamol are drugs"
    hyp = "broncol and acetaminophen are drugs"
    terms = ["broncol", "paracetamol"]
    assert compute_term_recall(ref, hyp, terms) == 0.5


def test_term_recall_empty_terms():
    assert compute_term_recall("hello", "hello", []) == 1.0


# --- Hallucination tests ---

def test_hallucination_pattern_match():
    result = detect_hallucination("네 시청해 주셔서 감사합니다")
    assert result["pattern_match"] is True
    assert result["is_hallucination"] is True


def test_hallucination_no_pattern():
    result = detect_hallucination("네 여보세요 보험료 관련해서요")
    assert result["pattern_match"] is False


def test_hallucination_length_anomaly():
    # 2 seconds of audio, but 30 chars = 15 chars/sec > threshold 10
    result = detect_hallucination("가" * 30, duration_sec=2.0)
    assert result["length_anomaly"] is True
    assert result["is_hallucination"] is True


def test_hallucination_length_normal():
    # 10 seconds of audio, 30 chars = 3 chars/sec, normal
    result = detect_hallucination("가" * 30, duration_sec=10.0)
    assert result["length_anomaly"] is False


def test_hallucination_repetition():
    # Same trigram repeated 3+ times
    result = detect_hallucination("보험료 납입 기간 보험료 납입 기간 보험료 납입 기간 보험료 납입 기간")
    assert result["repetition"] is True
    assert result["is_hallucination"] is True


def test_hallucination_no_repetition():
    result = detect_hallucination("보험료 납입 기간은 20년입니다")
    assert result["repetition"] is False


def test_hallucination_rate_batch():
    hyps = [
        "정상 발화입니다",
        "시청해 주셔서 감사합니다",
        "또 정상 발화입니다",
    ]
    durs = [5.0, 5.0, 5.0]
    rate = compute_hallucination_rate(hyps, durs)
    assert abs(rate - 1 / 3) < 1e-6
