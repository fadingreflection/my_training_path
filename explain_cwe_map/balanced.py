"""14 largest BigVul CWE labels and equal-count sampling."""

from __future__ import annotations

import random
from collections import defaultdict

# Named CWE with >= 100 vul=1 rows in raw BigVul (empty CWE excluded).
LARGE_CWE: tuple[str, ...] = (
    "CWE-119",
    "CWE-20",
    "CWE-399",
    "CWE-125",
    "CWE-200",
    "CWE-264",
    "CWE-189",
    "CWE-416",
    "CWE-190",
    "CWE-362",
    "CWE-476",
    "CWE-787",
    "CWE-284",
    "CWE-254",
)


def keep_large(rows: list[dict]) -> list[dict]:
    allowed = set(LARGE_CWE)
    return [r for r in rows if r.get("cwe") in allowed]


def balance_equal(rows: list[dict], seed: int) -> tuple[list[dict], int]:
    """Undersample every CWE to the size of the smallest class."""
    by: dict[str, list[dict]] = defaultdict(list)
    for rec in rows:
        by[str(rec["cwe"])].append(rec)
    if not by:
        return [], 0
    k = min(len(items) for items in by.values())
    rng = random.Random(seed)
    out: list[dict] = []
    for cwe in LARGE_CWE:
        items = list(by.get(cwe, []))
        rng.shuffle(items)
        out.extend(items[:k])
    rng.shuffle(out)
    return out, k


def stratified_sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Take (almost) the same number from each CWE. limit<=0 keeps the whole list."""
    rng = random.Random(seed)
    if limit <= 0 or limit >= len(rows):
        out = list(rows)
        rng.shuffle(out)
        return out
    by: dict[str, list[dict]] = defaultdict(list)
    for rec in rows:
        by[str(rec["cwe"])].append(rec)
    classes = [c for c in LARGE_CWE if c in by]
    n_cls = len(classes)
    if n_cls == 0:
        return []
    base, extra = divmod(limit, n_cls)
    out: list[dict] = []
    for i, cwe in enumerate(classes):
        take = min(len(by[cwe]), base + (1 if i < extra else 0))
        items = list(by[cwe])
        rng.shuffle(items)
        out.extend(items[:take])
    rng.shuffle(out)
    return out
