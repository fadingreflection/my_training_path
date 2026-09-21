#!/usr/bin/env python3
"""No-train ablations on existing SFT/holdout traces.

  .venv/bin/python explain_cwe_map/ablations/offline_audit.py

Writes explain_cwe_map/ablations/offline_audit.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "explain_cwe_map"))

from cwe_taxonomy import DROPPED_CATEGORIES, map_cwe, map_family  # noqa: E402
from match import extract_cwes, score  # noqa: E402

TRAIN = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits" / "train.jsonl"
V1 = ROOT / "explain_cwe_map" / "runs" / "qwen17b_lora_holdout.jsonl"
V2 = ROOT / "explain_cwe_map" / "runs" / "qwen17b_lora_glm_ds_v2_holdout.jsonl"
TEACHER = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits" / "holdout_teacher.jsonl"
OUT = ROOT / "explain_cwe_map" / "ablations" / "offline_audit.json"

CWE_RE = re.compile(r"CWE-?(\d+)", re.I)
PRIMARY_RE = re.compile(
    r"(?:most precise single classification is|"
    r"primary(?:[, ]+most accurate)? classification is|"
    r"primary mapping is|"
    r"most accurate classification is|"
    r"primary, most accurate classification is)"
    r"[^\n]{0,80}?CWE-?(\d+)",
    re.I,
)

# Direct weakness names — the critic's "CWE-family labels" vs dataflow.
LABEL_CUES: dict[str, re.Pattern[str]] = {
    "buffer_overflow": re.compile(
        r"\b(?:buffer overflow|heap overflow|stack overflow|buffer overrun|"
        r"heap[- ]based buffer overflow|stack[- ]based buffer overflow)\b",
        re.I,
    ),
    "oob": re.compile(
        r"\b(?:out[- ]of[- ]bounds(?: read| write)?|over[- ]?read|over[- ]?write|"
        r"\boob\b)\b",
        re.I,
    ),
    "uaf": re.compile(r"\b(?:use[- ]after[- ]free|\buaf\b|double[- ]free)\b", re.I),
    "null_deref": re.compile(
        r"\b(?:null[- ]pointer(?: dereference)?|NULL pointer|null deref(?:erence)?)\b",
        re.I,
    ),
    "int_overflow": re.compile(
        r"\b(?:integer overflow|integer underflow|wraparound|integer wrap)\b", re.I
    ),
    "info_disclosure": re.compile(
        r"\b(?:information disclosure|information leak|sensitive information)\b", re.I
    ),
    "access_control": re.compile(
        r"\b(?:access control|authorization|privilege escalation)\b", re.I
    ),
    "race": re.compile(r"\b(?:race condition|TOCTOU|time[- ]of[- ]check)\b", re.I),
    "input_validation": re.compile(
        r"\b(?:improper input validation|missing (?:input )?validation)\b", re.I
    ),
}

FAMILY_CUES: dict[str, tuple[str, ...]] = {
    "MEM": ("buffer_overflow", "oob", "uaf", "null_deref", "int_overflow"),
    "AUTH": ("info_disclosure", "access_control"),
    "RACE": ("race",),
    "INJ": (),
    "RES": (),
    "LOGIC": ("input_validation",),
}

DATAFLOW_CUES = re.compile(
    r"\b(?:attacker[- ]controlled|missing check|sink|dataflow|"
    r"unvalidated length|without (?:a )?bounds check)\b",
    re.I,
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cue_hits(text: str) -> list[str]:
    return [name for name, pat in LABEL_CUES.items() if pat.search(text or "")]


def expl_families(cues: list[str]) -> list[str]:
    fams: list[str] = []
    for fam, names in FAMILY_CUES.items():
        if any(n in cues for n in names):
            fams.append(fam)
    return fams


def primary_cwe(mapping: str, sc: dict) -> str | None:
    m = PRIMARY_RE.search(mapping or "")
    if m:
        return f"CWE-{int(m.group(1))}"
    levels = sc.get("pred_levels") or {}
    for key in ("variant", "base", "class", "pillar"):
        if levels.get(key):
            return levels[key]
    preds = sc.get("pred_cwes") or []
    return preds[0] if preds else None


def rescore_one(gold: str, mapping: str, pred: str | None) -> dict:
    if not pred:
        return {"hit_raw": False, "hit_canonical": False, "hit_family": False, "any": False}
    fake = f"{pred}\n"
    sc = score(gold, fake)
    return {
        "hit_raw": sc["hit_raw"],
        "hit_canonical": sc["hit_canonical"],
        "hit_family": sc["hit_family"],
        "any": sc["any_level_hit"],
        "pred": pred,
        "pred_canon": map_cwe(pred),
        "pred_fam": map_family(pred),
    }


def rate(n: int, d: int) -> float | None:
    if d == 0:
        return None
    return round(n / d, 4)


def holdout_slice(rows: list[dict], tag: str) -> dict:
    by_gold: dict[str, Counter] = defaultdict(Counter)
    n_pred: list[int] = []
    miss_buckets = Counter()
    n_cwe_in_expl = 0
    n_label = 0
    n_dataflow = 0
    n_label_only = 0
    n_expl_fam_match_gold = 0
    n_expl_fam_match_pred = 0
    n_expl_has_fam = 0
    n_mapped_gold = 0
    n_dropped_gold = 0
    mapping_break = 0  # family/canon hit, not raw
    dump_hit = 0  # any_level via listing gold, primary would miss
    expl_right_map_wrong = 0
    mem_on_non_mem = 0
    primary_any = 0
    first_any = 0
    hits_any = 0
    hits_raw = 0
    hits_canon = 0
    hits_fam = 0
    examples = {k: [] for k in (
        "dump_hit",
        "expl_right_map_wrong",
        "mem_on_non_mem",
        "dropped_gold_miss",
        "mapping_break",
    )}

    for rec in rows:
        gold = rec.get("gold_cwe") or rec.get("cwe") or ""
        expl = rec.get("explanation") or ""
        mapping = rec.get("mapping") or ""
        sc = rec.get("score") or score(gold, mapping)
        cues = cue_hits(expl)
        fams = expl_families(cues)
        gold_fam = map_family(gold)
        gold_canon = map_cwe(gold)
        dropped = gold in DROPPED_CATEGORIES or gold_canon is None
        preds = sc.get("pred_cwes") or extract_cwes(mapping)
        n_pred.append(len(preds))
        prim = primary_cwe(mapping, sc)
        prim_sc = rescore_one(gold, mapping, prim)
        first_sc = rescore_one(gold, mapping, preds[0] if preds else None)

        if sc.get("any_level_hit"):
            hits_any += 1
        if sc.get("hit_raw"):
            hits_raw += 1
        if sc.get("hit_canonical"):
            hits_canon += 1
        if sc.get("hit_family"):
            hits_fam += 1
        if prim_sc["any"]:
            primary_any += 1
        if first_sc["any"]:
            first_any += 1

        by_gold[gold]["n"] += 1
        by_gold[gold]["any"] += int(bool(sc.get("any_level_hit")))
        by_gold[gold]["raw"] += int(bool(sc.get("hit_raw")))
        by_gold[gold]["canon"] += int(bool(sc.get("hit_canonical")))
        by_gold[gold]["family"] += int(bool(sc.get("hit_family")))
        by_gold[gold]["primary_any"] += int(prim_sc["any"])
        by_gold[gold]["dropped"] += int(dropped)
        by_gold[gold]["label_cue"] += int(bool(cues))
        by_gold[gold]["n_pred_cwes"] += len(preds)

        if CWE_RE.search(expl):
            n_cwe_in_expl += 1
        if cues:
            n_label += 1
        if DATAFLOW_CUES.search(expl):
            n_dataflow += 1
        if cues and not DATAFLOW_CUES.search(expl):
            n_label_only += 1

        if gold_fam:
            n_mapped_gold += 1
        else:
            n_dropped_gold += 1

        if fams:
            n_expl_has_fam += 1
            pred_fams = set(sc.get("pred_families") or [])
            if gold_fam and gold_fam in fams:
                n_expl_fam_match_gold += 1
            if pred_fams & set(fams):
                n_expl_fam_match_pred += 1

        if sc.get("hit_family") and not sc.get("hit_raw"):
            mapping_break += 1
            if len(examples["mapping_break"]) < 5:
                examples["mapping_break"].append(sample(rec, cues, fams, prim, gold_fam))

        if sc.get("any_level_hit") and not prim_sc["any"]:
            dump_hit += 1
            if len(examples["dump_hit"]) < 5:
                examples["dump_hit"].append(sample(rec, cues, fams, prim, gold_fam))

        # explanation names gold family, mapping misses even family
        if gold_fam and gold_fam in fams and not sc.get("any_level_hit"):
            expl_right_map_wrong += 1
            if len(examples["expl_right_map_wrong"]) < 8:
                examples["expl_right_map_wrong"].append(
                    sample(rec, cues, fams, prim, gold_fam)
                )

        # MEM template on non-MEM / dropped gold
        if "MEM" in fams and gold_fam not in (None, "MEM") and not sc.get("any_level_hit"):
            mem_on_non_mem += 1
            if len(examples["mem_on_non_mem"]) < 8:
                examples["mem_on_non_mem"].append(sample(rec, cues, fams, prim, gold_fam))
        if "MEM" in fams and dropped and not sc.get("any_level_hit"):
            # dropped gold + MEM expl: likely child-of-category, not always hacking
            miss_buckets["dropped_gold_mem_expl"] += 1
            if len(examples["dropped_gold_miss"]) < 8:
                examples["dropped_gold_miss"].append(sample(rec, cues, fams, prim, gold_fam))

        if not sc.get("any_level_hit"):
            if dropped:
                miss_buckets["dropped_gold"] += 1
            elif gold_fam and gold_fam in fams:
                miss_buckets["expl_family_ok_map_miss"] += 1
            elif gold_fam and fams and gold_fam not in fams:
                miss_buckets["expl_family_mismatch"] += 1
            elif not fams:
                miss_buckets["expl_no_label_cue"] += 1
            else:
                miss_buckets["other"] += 1

    n = len(rows)
    gold_table = []
    for g, c in sorted(by_gold.items(), key=lambda kv: (-kv[1]["any"], kv[0])):
        nn = c["n"]
        gold_table.append({
            "gold": g,
            "n": nn,
            "dropped": bool(c["dropped"]),
            "family": map_family(g),
            "any": c["any"],
            "raw": c["raw"],
            "canonical": c["canon"],
            "family_hit": c["family"],
            "primary_any": c["primary_any"],
            "label_cue": c["label_cue"],
            "mean_pred_cwes": round(c["n_pred_cwes"] / nn, 2) if nn else None,
        })

    return {
        "tag": tag,
        "n": n,
        "metrics": {
            "hit_raw": rate(hits_raw, n),
            "hit_canonical": rate(hits_canon, n),
            "hit_family": rate(hits_fam, n),
            "any_level_hit": rate(hits_any, n),
            "any_if_primary_only": rate(primary_any, n),
            "any_if_first_pred_only": rate(first_any, n),
            "delta_any_minus_primary": rate(hits_any - primary_any, n),
        },
        "counts": {
            "hits_any": hits_any,
            "primary_any": primary_any,
            "dump_hits": dump_hit,
            "mapping_break_family_not_raw": mapping_break,
            "expl_right_map_wrong": expl_right_map_wrong,
            "mem_template_on_mapped_non_mem": mem_on_non_mem,
            "mapped_gold": n_mapped_gold,
            "dropped_gold": n_dropped_gold,
            "cwe_id_in_explanation": n_cwe_in_expl,
            "label_cue": n_label,
            "dataflow_cue": n_dataflow,
            "label_without_dataflow": n_label_only,
            "expl_has_family_cue": n_expl_has_fam,
            "expl_family_matches_gold": n_expl_fam_match_gold,
            "expl_family_matches_pred": n_expl_fam_match_pred,
        },
        "rates": {
            "label_cue": rate(n_label, n),
            "dataflow_cue": rate(n_dataflow, n),
            "cwe_id_in_explanation": rate(n_cwe_in_expl, n),
            "expl_family_match_gold_given_mapped": rate(
                n_expl_fam_match_gold, n_mapped_gold
            ),
            "expl_family_match_pred_given_cue": rate(
                n_expl_fam_match_pred, n_expl_has_fam
            ),
            "mean_pred_cwes": round(sum(n_pred) / n, 2) if n else None,
            "median_pred_cwes": sorted(n_pred)[n // 2] if n else None,
        },
        "slices": {
            "mapped_gold": {
                "n": n_mapped_gold,
                "any": rate(
                    sum(c["any"] for g, c in by_gold.items() if map_cwe(g)),
                    n_mapped_gold,
                ),
            },
            "dropped_gold": {
                "n": n_dropped_gold,
                "any": rate(
                    sum(c["any"] for g, c in by_gold.items() if not map_cwe(g)),
                    n_dropped_gold,
                ),
            },
            "mem_gold": {
                "n": sum(c["n"] for g, c in by_gold.items() if map_family(g) == "MEM"),
                "any": rate(
                    sum(c["any"] for g, c in by_gold.items() if map_family(g) == "MEM"),
                    sum(c["n"] for g, c in by_gold.items() if map_family(g) == "MEM"),
                ),
            },
            "non_mem_mapped": {
                "n": sum(
                    c["n"]
                    for g, c in by_gold.items()
                    if map_family(g) not in (None, "MEM")
                ),
                "any": rate(
                    sum(
                        c["any"]
                        for g, c in by_gold.items()
                        if map_family(g) not in (None, "MEM")
                    ),
                    sum(
                        c["n"]
                        for g, c in by_gold.items()
                        if map_family(g) not in (None, "MEM")
                    ),
                ),
            },
        },
        "miss_buckets": dict(miss_buckets),
        "by_gold": gold_table,
        "examples": examples,
    }


def sample(rec: dict, cues: list[str], fams: list[str], prim: str | None, gold_fam: str | None) -> dict:
    sc = rec.get("score") or {}
    return {
        "id": rec.get("id"),
        "gold": rec.get("gold_cwe"),
        "gold_family": gold_fam,
        "cues": cues,
        "expl_families": fams,
        "primary": prim,
        "pred_cwes": (sc.get("pred_cwes") or [])[:8],
        "pred_families": sc.get("pred_families") or [],
        "any": bool(sc.get("any_level_hit")),
        "expl_head": " ".join((rec.get("explanation") or "").split())[:280],
    }


def train_audit(rows: list[dict]) -> dict:
    reviews = [r for r in rows if r.get("turn") == "review"]
    maps = [r for r in rows if r.get("turn") == "map"]
    n = len(reviews)
    n_label = n_dataflow = n_cwe = n_label_only = 0
    cue_counts = Counter()
    by_gold = defaultdict(lambda: Counter())
    both = 0
    for rec in reviews:
        text = rec.get("completion") or ""
        gold = rec.get("gold_cwe") or ""
        cues = cue_hits(text)
        for c in cues:
            cue_counts[c] += 1
        has_label = bool(cues)
        has_df = bool(DATAFLOW_CUES.search(text))
        has_cwe = bool(CWE_RE.search(text))
        n_label += int(has_label)
        n_dataflow += int(has_df)
        n_cwe += int(has_cwe)
        n_label_only += int(has_label and not has_df)
        both += int(has_label and has_df)
        by_gold[gold]["n"] += 1
        by_gold[gold]["label"] += int(has_label)
        by_gold[gold]["cwe_id"] += int(has_cwe)
        by_gold[gold]["dropped"] += int(gold in DROPPED_CATEGORIES)

    gold_table = []
    for g, c in sorted(by_gold.items(), key=lambda kv: kv[0]):
        nn = c["n"]
        gold_table.append({
            "gold": g,
            "n": nn,
            "dropped": bool(c["dropped"]),
            "family": map_family(g),
            "label_cue": c["label"],
            "label_rate": rate(c["label"], nn),
            "cwe_id_in_expl": c["cwe_id"],
        })

    # If we strip label-cue reviews, remaining n (both turns: review+map of those ids)
    keep_ids = set()
    for rec in reviews:
        if not cue_hits(rec.get("completion") or ""):
            keep_ids.add(rec["id"])
    n_turns_keep = sum(1 for r in rows if r.get("id") in keep_ids)

    return {
        "n_review_turns": n,
        "n_map_turns": len(maps),
        "n_ids": len({r["id"] for r in reviews}),
        "label_cue": n_label,
        "label_cue_rate": rate(n_label, n),
        "dataflow_cue": n_dataflow,
        "dataflow_cue_rate": rate(n_dataflow, n),
        "both_label_and_dataflow": both,
        "label_without_dataflow": n_label_only,
        "cwe_id_in_explanation": n_cwe,
        "cwe_id_rate": rate(n_cwe, n),
        "cue_counts": dict(cue_counts.most_common()),
        "by_gold": gold_table,
        "keyword_strip_remaining": {
            "review_ids_without_label_cue": len(keep_ids),
            "sft_turns_if_drop_labeled_expl": n_turns_keep,
            "sft_turns_now": len(rows),
        },
    }


def main() -> None:
    train = load_jsonl(TRAIN)
    v1 = load_jsonl(V1)
    v2 = load_jsonl(V2)
    teacher = load_jsonl(TEACHER)
    out = {
        "headline_metrics": {
            "note": (
                "any_level_hit = hit_raw OR hit_canonical OR hit_family. "
                "Gold CWE-20/189/254/264/284/399 are DROPPED_CATEGORIES: "
                "canonical/family can never hit; only listing the raw id counts."
            ),
            "v1": None,
            "v2": None,
            "teacher": None,
        },
        "train_keyword_audit": train_audit(train),
        "holdout": {
            "v1": holdout_slice(v1, "v1_lora"),
            "v2": holdout_slice(v2, "glm_ds_v2"),
            "teacher": holdout_slice(teacher, "teacher_glm"),
        },
    }
    for key in ("v1", "v2", "teacher"):
        out["headline_metrics"][key] = out["holdout"][key]["metrics"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "headline": out["headline_metrics"],
        "train_label_rate": out["train_keyword_audit"]["label_cue_rate"],
        "keyword_strip": out["train_keyword_audit"]["keyword_strip_remaining"],
        "v1_counts": out["holdout"]["v1"]["counts"],
        "v1_slices": out["holdout"]["v1"]["slices"],
        "teacher_slices": out["holdout"]["teacher"]["slices"],
        "v1_miss": out["holdout"]["v1"]["miss_buckets"],
        "v1_by_gold": out["holdout"]["v1"]["by_gold"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
