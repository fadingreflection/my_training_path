#!/usr/bin/env python3
"""Mix GLM replay + quality-filtered DeepSeek hits for continue-SFT.

DeepSeek dumps (raw reasoning, truncated catalogs, 50k-char traces) are dropped.
GLM train.jsonl is replayed in full to limit catastrophic forgetting.
Mapping collisions: GLM wins — DeepSeek map turns are kept only when GLM
did not already hit that id. DeepSeek reviews are still added as paraphrases.

  .venv/bin/python explain_cwe_map/make_sft_glm_ds_clean.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain_cwe_map"))

from make_sft import cwe_counts, latest_by_id, load_jsonl, load_pool, write_jsonl  # noqa: E402
from make_sft_ds_continue import TokenBudget, load_id_set, strip_think  # noqa: E402
from match import extract_cwes  # noqa: E402
from prompts import MAP_CWE, _head_token, review_user_message, skip_cwe_mapping  # noqa: E402

PAIRS = ROOT / "raw_data" / "bigvul_func_before_balanced14.jsonl"
DS_RUN = ROOT / "explain_cwe_map" / "runs" / "v41flash_full.SNAPSHOT_qc.jsonl"
GLM_HITS = ROOT / "explain_cwe_map" / "runs" / "glm53flash_full_hits.jsonl"
GLM_TRAIN = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits" / "train.jsonl"
HOLDOUT = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits" / "holdout_ids.jsonl"
GLM_VAL = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits" / "val.jsonl"
OUT_DIR = ROOT / "explain_cwe_map" / "sft" / "glm_ds_clean"
TOKENIZER = ROOT / "models" / "base"

VERDICTS = {"VULNERABLE", "INSUFFICIENT_INFO", "SAFE"}
DUMP_HEAD = re.compile(
    r"^(we need|need |okay|ok,|looking at|i need|first,|analysis:|but task:|the user|let me )",
    re.I,
)
CWE_RE = re.compile(r"CWE-?\d+", re.I)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
MAX_EXPL_CHARS = 3500
MAX_EXPL_WORDS = 1000
MAX_MAP_CHARS = 4500
MAX_MAP_WORDS = 900
MAX_MAP_CWES = 20


def words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def is_dump_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if DUMP_HEAD.search(t):
        return True
    low = t.lower()
    if "we need answer" in low or "need analyze" in low or "need comply" in low:
        return True
    if len(CWE_RE.findall(t)) > 25:
        return True
    return False


def explanation_issues(expl: str) -> list[str]:
    issues: list[str] = []
    if _head_token(expl) not in VERDICTS:
        issues.append("no_verdict")
    if skip_cwe_mapping(expl):
        issues.append("abstain")
    if is_dump_text(expl):
        issues.append("dump")
    if words(expl) > MAX_EXPL_WORDS or len(expl) > MAX_EXPL_CHARS:
        issues.append("too_long")
    if CJK_RE.search(expl):
        issues.append("cjk")
    if CWE_RE.search(expl):
        issues.append("cwe_in_explanation")
    if expl.count("\n") > 40:
        issues.append("too_many_newlines")
    return issues


def mapping_issues(mapping: str) -> list[str]:
    issues: list[str] = []
    if not (mapping or "").strip():
        return ["empty"]
    if is_dump_text(mapping):
        issues.append("dump")
    if len(mapping) > MAX_MAP_CHARS or words(mapping) > MAX_MAP_WORDS:
        issues.append("too_long")
    if not CWE_RE.search(mapping):
        issues.append("no_cwe")
    n = len(extract_cwes(mapping))
    if n > MAX_MAP_CWES:
        issues.append("too_many_cwes")
    if CJK_RE.search(mapping):
        issues.append("cjk")
    return issues


def is_hit(rec: dict | None) -> bool:
    return bool(rec) and bool((rec.get("score") or {}).get("any_level_hit"))


def content_key(rec: dict) -> str:
    blob = f"{rec.get('prompt') or ''}||{rec.get('completion') or ''}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def sft_row(teacher: dict, *, turn: str, prompt: str, completion: str, extra: dict) -> dict:
    score = teacher.get("score") or {}
    return {
        "id": teacher["id"],
        "gold_cwe": teacher["gold_cwe"],
        "cve": teacher.get("cve") or "",
        "project": teacher.get("project") or "",
        "commit_id": teacher.get("commit_id") or "",
        "any_level_hit": bool(score.get("any_level_hit")),
        "hit_raw": bool(score.get("hit_raw")),
        "hit_canonical": bool(score.get("hit_canonical")),
        "hit_family": bool(score.get("hit_family")),
        "turn": turn,
        "prompt": prompt,
        "completion": completion,
        **extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek", type=Path, default=DS_RUN)
    ap.add_argument("--glm-hits", type=Path, default=GLM_HITS)
    ap.add_argument("--glm-train", type=Path, default=GLM_TRAIN)
    ap.add_argument("--data", type=Path, default=PAIRS)
    ap.add_argument("--holdout", type=Path, default=HOLDOUT)
    ap.add_argument("--glm-val", type=Path, default=GLM_VAL)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds_all = load_jsonl(args.deepseek)
    ds_by_id = latest_by_id(ds_all)
    glm_by_id = latest_by_id(load_jsonl(args.glm_hits)) if args.glm_hits.exists() else {}
    glm_train_rows = load_jsonl(args.glm_train)
    holdout = load_id_set(args.holdout)
    glm_val_ids = load_id_set(args.glm_val)
    glm_train_ids = {r["id"] for r in glm_train_rows}
    pool = load_pool(args.data)
    budget = TokenBudget(args.tokenizer, args.max_seq_length)

    hits = {i: r for i, r in ds_by_id.items() if is_hit(r)}
    qc_rows = []
    ds_rows: list[dict] = []
    stats = Counter()

    for rid, rec in sorted(hits.items()):
        expl = strip_think(rec.get("explanation") or "")
        mapping = strip_think(rec.get("mapping") or "")
        e_iss = explanation_issues(expl)
        m_iss = mapping_issues(mapping)
        row = {
            "id": rid,
            "gold_cwe": rec.get("gold_cwe"),
            "expl_chars": len(expl),
            "map_chars": len(mapping),
            "expl_issues": e_iss,
            "map_issues": m_iss,
            "holdout": rid in holdout,
            "glm_val": rid in glm_val_ids,
            "glm_train": rid in glm_train_ids,
        }
        qc_rows.append(row)
        if rid in holdout or rid in glm_val_ids:
            stats["skipped_split"] += 1
            continue
        code = (pool.get(rid) or {}).get("func_before") or ""
        if not code:
            stats["skipped_no_code"] += 1
            continue
        if e_iss:
            stats["skipped_bad_expl"] += 1
            continue
        review_prompt = review_user_message(code)
        if not budget.fits(review_prompt, expl):
            stats["skipped_review_budget"] += 1
            continue
        ds_rows.append(
            sft_row(
                rec,
                turn="review",
                prompt=review_prompt,
                completion=expl,
                extra={"source": "deepseek", "map_source": None},
            )
        )
        stats["ds_review"] += 1

        glm_hit = is_hit(glm_by_id.get(rid))
        if glm_hit:
            stats["ds_map_skipped_glm_priority"] += 1
            continue
        if m_iss:
            stats["skipped_bad_map"] += 1
            continue
        map_prompt = f"{review_user_message(code).rstrip()}\n\n{expl}\n\n{MAP_CWE}"
        if not budget.fits(map_prompt, mapping):
            stats["skipped_map_budget"] += 1
            continue
        ds_rows.append(
            sft_row(
                rec,
                turn="map",
                prompt=map_prompt,
                completion=mapping,
                extra={"source": "deepseek", "map_source": "deepseek"},
            )
        )
        stats["ds_map"] += 1

    glm_replay = []
    for rec in glm_train_rows:
        row = dict(rec)
        row.setdefault("source", "glm")
        row.setdefault("map_source", "glm" if row.get("turn") == "map" else None)
        glm_replay.append(row)
        stats["glm_replay"] += 1

    banned = {content_key(r) for r in glm_replay}
    ds_deduped = []
    for rec in ds_rows:
        if content_key(rec) in banned:
            stats["ds_dedup_identical_glm"] += 1
            continue
        ds_deduped.append(rec)
        banned.add(content_key(rec))

    train_rows = glm_replay + ds_deduped
    rng = random.Random(args.seed)
    rng.shuffle(train_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "deepseek_clean_turns.jsonl", ds_deduped)

    expl_issue_counts = Counter()
    map_issue_counts = Counter()
    for row in qc_rows:
        expl_issue_counts.update(row["expl_issues"] or ["ok_expl"])
        map_issue_counts.update(row["map_issues"] or ["ok_map"])

    qc = {
        "deepseek_file": str(args.deepseek),
        "n_ds_rows": len(ds_all),
        "n_ds_unique": len(ds_by_id),
        "n_hits": len(hits),
        "n_hits_audited": len(qc_rows),
        "expl_issue_counts": dict(expl_issue_counts.most_common()),
        "map_issue_counts": dict(map_issue_counts.most_common()),
        "filters": {
            "max_expl_chars": MAX_EXPL_CHARS,
            "max_expl_words": MAX_EXPL_WORDS,
            "max_map_chars": MAX_MAP_CHARS,
            "max_map_words": MAX_MAP_WORDS,
            "max_map_cwes": MAX_MAP_CWES,
            "dump_head": DUMP_HEAD.pattern,
        },
        "dirty": [
            {
                "id": r["id"],
                "gold_cwe": r["gold_cwe"],
                "expl_chars": r["expl_chars"],
                "map_chars": r["map_chars"],
                "expl_issues": r["expl_issues"],
                "map_issues": r["map_issues"],
            }
            for r in qc_rows
            if r["expl_issues"] or (r["map_issues"] and r["map_issues"] != ["empty"])
        ],
    }
    (args.out_dir / "qc_report.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "priority": ["glm", "deepseek"],
        "rule": (
            "replay full GLM train; add clean DeepSeek reviews; "
            "DeepSeek map only if GLM did not hit that id"
        ),
        "stats": dict(stats),
        "train": {
            "n_turns": len(train_rows),
            "n_glm": stats["glm_replay"],
            "n_ds": len(ds_deduped),
            "n_review": sum(1 for r in train_rows if r["turn"] == "review"),
            "n_map": sum(1 for r in train_rows if r["turn"] == "map"),
            "source": dict(Counter(r.get("source") for r in train_rows)),
            "gold_cwe_turns": cwe_counts(train_rows),
            "out": str(args.out_dir / "train.jsonl"),
            "shuffled_seed": args.seed,
        },
        "val": {"reuse": str(args.glm_val)},
        "qc": str(args.out_dir / "qc_report.json"),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
