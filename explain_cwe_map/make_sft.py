#!/usr/bin/env python3
"""Build prompt/completion SFT from GLM teacher traces + a CWE-stratified holdout.

Train target is the same two-step chain as run.py:
  1) review turn: REVIEW + code → explanation
  2) map turn:    that explanation + MAP_CWE → CWE hierarchy

Eval stays on run.py: generate both turns, score any_level_hit.

  python3 explain_cwe_map/make_sft.py
  python3 explain_cwe_map/run.py --limit 0 --tag glm53flash_holdout \\
      --only-from explain_cwe_map/sft/glm53flash_hits/holdout_ids.jsonl --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain_cwe_map"))

from prompts import MAP_CWE, review_user_message, skip_cwe_mapping  # noqa: E402
PAIRS = ROOT / "raw_data" / "bigvul_func_before_balanced14.jsonl"
TEACHER = ROOT / "explain_cwe_map" / "runs" / "glm53flash_full.jsonl"
HITS = ROOT / "explain_cwe_map" / "runs" / "glm53flash_full_hits.jsonl"
OUT_DIR = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits"


def row_id(rec: dict) -> str:
    blob = f"{rec.get('cve')}|{rec.get('commit_id')}|{rec.get('func_before')[:200]}"
    return hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()[:16]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_pool(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            code = (rec.get("func_before") or "").strip()
            if not code:
                continue
            rid = row_id(rec)
            by_id[rid] = {
                "id": rid,
                "cve": rec.get("cve") or "",
                "cwe": rec.get("cwe") or "",
                "project": rec.get("project") or "",
                "commit_id": rec.get("commit_id") or "",
                "func_before": code,
            }
    return by_id


def latest_by_id(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rec in rows:
        out[rec["id"]] = rec
    return out


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    n = max(len(ok), 1)
    keys = (
        "said_safe",
        "said_insufficient",
        "hit_raw",
        "hit_canonical",
        "hit_family",
        "any_level_hit",
    )
    out = {"n": len(ok), "n_error": sum(1 for r in rows if r.get("error"))}
    for k in keys:
        out[k] = sum(1 for r in ok if r["score"][k]) / n if ok else 0.0
    out["success"] = out["any_level_hit"]
    return out


def stratified_holdout_ids(rows: list[dict], n_per_class: int, seed: int) -> list[str]:
    by: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for rec in rows:
        rid = rec["id"]
        if rid in seen:
            continue
        seen.add(rid)
        by[rec["gold_cwe"]].append(rid)
    rng = random.Random(seed)
    out: list[str] = []
    for cwe in sorted(by):
        ids = list(by[cwe])
        rng.shuffle(ids)
        out.extend(ids[:n_per_class])
    rng.shuffle(out)
    return out


def sft_turns(teacher: dict, code: str) -> list[dict]:
    explanation = (teacher.get("explanation") or "").strip()
    mapping = (teacher.get("mapping") or "").strip()
    gold = teacher["gold_cwe"]
    rid = teacher["id"]
    score = teacher.get("score") or {}
    meta = {
        "id": rid,
        "gold_cwe": gold,
        "cve": teacher.get("cve") or "",
        "project": teacher.get("project") or "",
        "commit_id": teacher.get("commit_id") or "",
        "any_level_hit": bool(score.get("any_level_hit")),
        "hit_raw": bool(score.get("hit_raw")),
        "hit_canonical": bool(score.get("hit_canonical")),
        "hit_family": bool(score.get("hit_family")),
    }
    turns = [
        {
            **meta,
            "turn": "review",
            "prompt": review_user_message(code),
            "completion": explanation,
        }
    ]
    if mapping and not skip_cwe_mapping(explanation):
        turns.append(
            {
                **meta,
                "turn": "map",
                "prompt": (
                    f"{review_user_message(code).rstrip()}\n\n"
                    f"{explanation}\n\n{MAP_CWE}"
                ),
                "completion": mapping,
            }
        )
    return turns


def cwe_counts(rows: list[dict], field: str = "gold_cwe") -> dict[str, int]:
    return dict(Counter(r[field] for r in rows).most_common())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, default=TEACHER)
    ap.add_argument("--hits", type=Path, default=HITS)
    ap.add_argument("--data", type=Path, default=PAIRS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--holdout-per-class", type=int, default=10)
    ap.add_argument("--val-per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = load_pool(args.data)
    teacher_all = load_jsonl(args.teacher)
    hits_all = load_jsonl(args.hits)
    teacher_by_id = latest_by_id(teacher_all)
    hits_by_id = latest_by_id(hits_all)

    holdout_ids = stratified_holdout_ids(teacher_all, args.holdout_per_class, args.seed)
    holdout_set = set(holdout_ids)

    train_hit_ids = [i for i in hits_by_id if i not in holdout_set and i in pool]
    rng = random.Random(args.seed)
    by_cwe: dict[str, list[str]] = defaultdict(list)
    for rid in train_hit_ids:
        by_cwe[hits_by_id[rid]["gold_cwe"]].append(rid)
    val_ids: list[str] = []
    for cwe in sorted(by_cwe):
        ids = list(by_cwe[cwe])
        rng.shuffle(ids)
        val_ids.extend(ids[: args.val_per_class])
    val_set = set(val_ids)
    train_ids = [i for i in train_hit_ids if i not in val_set]

    def pack(ids: list[str], src: dict[str, dict]) -> list[dict]:
        rows: list[dict] = []
        missing = 0
        for rid in ids:
            rec = src[rid]
            code = (pool.get(rid) or {}).get("func_before") or ""
            if not code:
                missing += 1
                continue
            rows.extend(sft_turns(rec, code))
        if missing:
            print(f"skip no-code={missing}")
        return rows

    train_rows = pack(train_ids, hits_by_id)
    val_rows = pack(val_ids, hits_by_id)
    holdout_teacher = [teacher_by_id[i] for i in holdout_ids if i in teacher_by_id]
    holdout_id_rows = [{"id": i, "gold_cwe": teacher_by_id[i]["gold_cwe"]} for i in holdout_ids]

    out = args.out_dir
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    write_jsonl(out / "holdout_ids.jsonl", holdout_id_rows)
    write_jsonl(out / "holdout_teacher.jsonl", holdout_teacher)

    def split_stats(ids: list[str], src: dict[str, dict]) -> dict:
        recs = [src[i] for i in ids if i in src]
        return {
            "n_ids": len(ids),
            "n_hits": sum(1 for r in recs if (r.get("score") or {}).get("any_level_hit")),
            "gold_cwe": cwe_counts(recs),
        }

    manifest = {
        "teacher": str(args.teacher),
        "hits": str(args.hits),
        "holdout_per_class": args.holdout_per_class,
        "val_per_class": args.val_per_class,
        "seed": args.seed,
        "metric": "any_level_hit on generated explanation→CWE chain (explain_cwe_map/run.py)",
        "train": {
            **split_stats(train_ids, hits_by_id),
            "n_turns": len(train_rows),
            "n_review": sum(1 for r in train_rows if r["turn"] == "review"),
            "n_map": sum(1 for r in train_rows if r["turn"] == "map"),
            "out": str(out / "train.jsonl"),
        },
        "val": {
            **split_stats(val_ids, hits_by_id),
            "n_turns": len(val_rows),
            "n_review": sum(1 for r in val_rows if r["turn"] == "review"),
            "n_map": sum(1 for r in val_rows if r["turn"] == "map"),
            "out": str(out / "val.jsonl"),
        },
        "holdout": {
            **split_stats(holdout_ids, teacher_by_id),
            "teacher_metrics": summarize(holdout_teacher),
            "ids": str(out / "holdout_ids.jsonl"),
            "teacher_preds": str(out / "holdout_teacher.jsonl"),
            "eval": (
                "python3 explain_cwe_map/run.py --limit 0 "
                f"--only-from {out / 'holdout_ids.jsonl'} "
                "--tag <student_tag> --workers 16"
            ),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
