"""Regex-индикаторы типов claims."""
import re

PATTERNS = {
    "guard_absent": re.compile(
        r"\b(no|without|missing|absent|unchecked|unvalidated)\s+"
        r"(check|validation|bound|guard|verification)\b",
        re.IGNORECASE,
    ),
    "bound": re.compile(
        r"\b(bounded|checked|clamped|limited|validated|constrained)\s+by\b",
        re.IGNORECASE,
    ),
    "arith": re.compile(
        r"\b(overflow|underflow|wrap|wraparound|truncat|saturat)\w*\b",
        re.IGNORECASE,
    ),
    "dataflow": re.compile(
        r"\b(flows?\s+to|passed\s+to|reaches?|leads?\s+to|propagat)\w*\b",
        re.IGNORECASE,
    ),
    "exists": re.compile(
        r"\b(variable|buffer|field|array|pointer|param)\s+"
        r"[A-Za-z_][A-Za-z0-9_]*",
        re.IGNORECASE,
    ),
    "type": re.compile(
        r"\b(uint\d+|int\d+|size_t|unsigned|long|char|short|byte)\b",
        re.IGNORECASE,
    ),
}


def extract_regex_features(text: str) -> dict:
    """Возвращает счётчики по каждому типу claim."""
    if not text:
        return {k: 0 for k in PATTERNS}
    return {k: len(p.findall(text)) for k, p in PATTERNS.items()}


FEATURE_ORDER = list(PATTERNS.keys())