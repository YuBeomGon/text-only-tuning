import pytest
from src.eval.metrics import compute_cer, compute_wer, compute_term_recall


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
