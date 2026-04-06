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


## Hallucination detection

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

# Korean speech: typical 3-5 chars/sec. Above this threshold = suspicious.
CHARS_PER_SEC_THRESHOLD = 10.0

# Repeated trigram threshold
REPEAT_NGRAM_N = 3
REPEAT_NGRAM_MAX = 3


def detect_hallucination(
    hypothesis: str,
    duration_sec: float | None = None,
) -> dict:
    """Detect hallucination signals in a hypothesis.

    Returns dict with:
      - pattern_match: bool (known hallucination phrase found)
      - length_anomaly: bool (too many chars per second of audio)
      - repetition: bool (repeated n-grams detected)
      - is_hallucination: bool (any signal triggered)
      - matched_patterns: list[str]
    """
    hyp = hypothesis.strip()
    result = {
        "pattern_match": False,
        "length_anomaly": False,
        "repetition": False,
        "is_hallucination": False,
        "matched_patterns": [],
    }

    # 1. Pattern matching
    for pattern in HALLUCINATION_PATTERNS:
        if pattern in hyp:
            result["pattern_match"] = True
            result["matched_patterns"].append(pattern)

    # 2. Length anomaly (chars per second)
    if duration_sec and duration_sec > 0:
        cps = len(hyp) / duration_sec
        if cps > CHARS_PER_SEC_THRESHOLD:
            result["length_anomaly"] = True

    # 3. Repetition detection (n-gram)
    words = hyp.split()
    if len(words) >= REPEAT_NGRAM_N:
        ngram_counts: dict[tuple, int] = {}
        for i in range(len(words) - REPEAT_NGRAM_N + 1):
            ng = tuple(words[i : i + REPEAT_NGRAM_N])
            ngram_counts[ng] = ngram_counts.get(ng, 0) + 1
        for ng, count in ngram_counts.items():
            if count >= REPEAT_NGRAM_MAX:
                result["repetition"] = True
                break

    result["is_hallucination"] = (
        result["pattern_match"]
        or result["length_anomaly"]
        or result["repetition"]
    )
    return result


def compute_hallucination_rate(
    hypotheses: list[str],
    durations: list[float | None],
) -> float:
    """Compute hallucination rate over a list of hypotheses.

    Returns: fraction of utterances flagged as hallucination.
    """
    if not hypotheses:
        return 0.0
    flags = sum(
        1
        for hyp, dur in zip(hypotheses, durations)
        if detect_hallucination(hyp, dur)["is_hallucination"]
    )
    return flags / len(hypotheses)
