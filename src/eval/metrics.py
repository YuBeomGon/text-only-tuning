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


## Hallucination detection (broadcast pattern only)

HALLUCINATION_PATTERNS = [
    # 방송 멘트
    "시청해 주셔서 감사합니다",
    "시청해주셔서 감사합니다",
    "구독과 좋아요",
    "구독과 좋아요 부탁드립니다",
    "좋아요와 구독",
    "채널에 가입",
    "구독 부탁",
    "알림 설정",
    # 뉴스/방송 식별
    "MBC 뉴스",
    "KBS 뉴스",
    "SBS 뉴스",
    "YTN 뉴스",
    "JTBC 뉴스",
    "연합뉴스",
    "뉴스데스크",
    # 자막/번역
    "자막 제공",
    "자막 협찬",
    "한국어 자막",
    "영어 자막",
    "번역 자막",
    # 기타 hallucination
    "다음 영상에서 만나요",
    "다음 시간에 만나요",
    "오늘 영상은 여기까지",
]


def detect_hallucination(hypothesis: str) -> dict:
    """Detect hallucination by checking for known broadcast patterns.

    Returns dict with:
      - is_hallucination: bool
      - matched_patterns: list[str]
    """
    hyp = hypothesis.strip()
    matched = [p for p in HALLUCINATION_PATTERNS if p in hyp]
    return {
        "is_hallucination": len(matched) > 0,
        "matched_patterns": matched,
    }


def compute_hallucination_rate(hypotheses: list[str]) -> float:
    """Compute hallucination rate: fraction of utterances containing broadcast patterns."""
    if not hypotheses:
        return 0.0
    flags = sum(
        1 for hyp in hypotheses
        if detect_hallucination(hyp)["is_hallucination"]
    )
    return flags / len(hypotheses)
