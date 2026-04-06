from jiwer import cer as jiwer_cer, wer as jiwer_wer


def compute_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer_cer(reference, hypothesis)


def compute_wer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer_wer(reference, hypothesis)


def compute_term_recall(
    reference: str, hypothesis: str, domain_terms: list[str]
) -> float:
    if not domain_terms:
        return 1.0
    ref_lower = reference.lower()
    hyp_lower = hypothesis.lower()
    hits = 0
    expected = 0
    for term in domain_terms:
        t = term.lower()
        if t in ref_lower:
            expected += 1
            if t in hyp_lower:
                hits += 1
    if expected == 0:
        return 1.0
    return hits / expected


def compute_hallucination_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not hyp_words:
        return 0.0
    ref_set = set(ref_words)
    insertions = sum(1 for w in hyp_words if w not in ref_set)
    return insertions / len(hyp_words)
