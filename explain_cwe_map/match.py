"""Score predicted CWE hierarchy against BigVul ground-truth labels.

Success: gold matches the prediction at any level — raw id, canonical class
(CWE-125 vs CWE-119), or coarse family (MEM/RES/...).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sft_datasets_ready"))

from cwe_taxonomy import map_cwe, map_family  # noqa: E402

CWE_RE = re.compile(r"CWE-?(\d+)", re.I)
LEVEL_RE = re.compile(
    r"(?i)\b(pillar|class|base|variant)\b[^\n]{0,80}?CWE-?(\d+)"
)


def norm_cwe(raw: str | None) -> str | None:
    if not raw:
        return None
    m = CWE_RE.search(str(raw).upper().replace("CWE--", "CWE-"))
    if not m:
        digits = re.sub(r"\D", "", str(raw))
        if not digits:
            return None
        return f"CWE-{int(digits)}"
    return f"CWE-{int(m.group(1))}"


def extract_cwes(text: str) -> list[str]:
    seen: list[str] = []
    for m in CWE_RE.finditer(text or ""):
        cid = f"CWE-{int(m.group(1))}"
        if cid not in seen:
            seen.append(cid)
    return seen


def extract_levels(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in LEVEL_RE.finditer(text or ""):
        level = m.group(1).lower()
        cid = f"CWE-{int(m.group(2))}"
        out.setdefault(level, cid)
    return out


def score(
    gold_raw: str | None,
    mapping_text: str,
    *,
    said_safe: bool = False,
    said_insufficient: bool = False,
) -> dict:
    """Score one mapping. Abstaining (SAFE or INSUFFICIENT_INFO) predicts nothing.

    Both abstain flags are reported separately so callers can tell a refusal
    apart from a wrong CWE; either one suppresses predictions, because the
    two-step chain skips the mapping turn on an abstain verdict.
    """
    abstained = said_safe or said_insufficient
    gold = norm_cwe(gold_raw)
    gold_canon = map_cwe(gold) if gold else None
    gold_fam = map_family(gold) if gold else None
    pred = extract_cwes(mapping_text) if not abstained else []
    pred_canon = [map_cwe(c) or c for c in pred]
    pred_fam = [map_family(c) for c in pred if map_family(c)]
    levels = extract_levels(mapping_text) if not abstained else {}

    hit_raw = bool(gold) and gold in pred
    hit_canon = bool(gold_canon) and gold_canon in pred_canon
    hit_family = bool(gold_fam) and gold_fam in pred_fam
    any_level = hit_raw or hit_canon or hit_family
    return {
        "gold_raw": gold,
        "gold_canonical": gold_canon,
        "gold_family": gold_fam,
        "pred_cwes": pred,
        "pred_canonical": [c for c in pred_canon if c],
        "pred_families": pred_fam,
        "pred_levels": levels,
        "said_safe": said_safe,
        "said_insufficient": said_insufficient,
        "hit_raw": hit_raw,
        "hit_canonical": hit_canon,
        "hit_family": hit_family,
        "any_level_hit": any_level,
    }
