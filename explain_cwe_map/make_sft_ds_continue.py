#!/usr/bin/env python3
"""Build continue-SFT from DeepSeek-v4.1-flash hits only.

Review turns keep DeepSeek explanations (paraphrases vs GLM).
Map turns: if GLM also hit the same id, GLM mapping wins (priority glm > deepseek).
The map prompt still uses the DeepSeek explanation so the student sees a different
writeup of the same bug mapping onto the GLM CWE chain.

Holdout ids stay frozen. GLM val ids are excluded from train.
Ramble / non-verdict DeepSeek dumps are dropped. Completions that do not fit
max_seq_length after the chat template are dropped.

  .venv/bin/python explain_cwe_map/make_sft_ds_continue.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain_cwe_map"))

from make_sft import (  # noqa: E402
    load_jsonl,
    load_pool,
    latest_by_id,
    write_jsonl,
    cwe_counts,
)
from prompts import MAP_CWE, _head_token, review_user_message, skip_cwe_mapping  # noqa: E402

PAIRS = ROOT / "raw_data" / "bigvul_func_before_balanced14.jsonl"
DS_RUN = ROOT / "explain_cwe_map" / "runs" / "v41flash_full.jsonl"
GLM_HITS = ROOT / "explain_cwe_map" / "runs" / "glm53flash_full_hits.jsonl"
HOLDOUT = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits" / "holdout_ids.jsonl"
GLM_VAL = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits" / "val.jsonl"
OUT_DIR = ROOT / "explain_cwe_map" / "sft" / "ds41flash_continue"
TOKENIZER = ROOT / "models" / "base"
VERDICTS = {"VULNERABLE", "INSUFFICIENT_INFO", "SAFE"}
LEN_MARGIN = 32


def strip_think(text: str) -> str:
    t = (text or "").strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    while t.startswith("<think>"):
        t = t[len("<think>") :].strip()
    return t.strip()


def is_hit(rec: dict | None) -> bool:
    if not rec:
        return False
    return bool((rec.get("score") or {}).get("any_level_hit"))


def wellformed_explanation(text: str) -> bool:
    return _head_token(text) in VERDICTS


def load_id_set(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for rec in load_jsonl(path):
        rid = rec.get("id")
        if rid:
            ids.add(rid)
    return ids


class TokenBudget:
    def __init__(self, tokenizer_path: Path, max_seq_length: int, margin: int = LEN_MARGIN):
        self.tok = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
        self.max_seq_length = max_seq_length
        self.margin = margin

    def fits(self, prompt: str, completion: str) -> bool:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            wrapped = self.tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            wrapped = self.tok.apply_chat_template(messages, **kwargs)
        n = len(self.tok(wrapped + completion, add_special_tokens=False)["input_ids"])
        return n <= self.max_seq_length - self.margin


def pred_cwe_set(rec: dict | None) -> frozenset[str]:
    if not rec:
        return frozenset()
    return frozenset((rec.get("score") or {}).get("pred_cwes") or [])


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


def build_turns(
    ids: list[str],
    ds_by_id: dict[str, dict],
    glm_by_id: dict[str, dict],
    pool: dict[str, dict],
    budget: TokenBudget,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats: dict = {
        "n_ids": 0,
        "n_review": 0,
        "n_map": 0,
        "map_source": Counter(),
        "gold_cwe": Counter(),
        "skipped_no_code": 0,
        "skipped_not_hit": 0,
        "skipped_malformed": 0,
        "skipped_review_over_budget": 0,
        "skipped_map_over_budget": 0,
        "skipped_map_no_text": 0,
        "glm_map_priority": 0,
        "ds_map_no_glm": 0,
        "mapping_cwe_set_equal": 0,
        "mapping_cwe_set_collision": 0,
        "identical_explanation_to_glm": 0,
    }

    for rid in ids:
        ds = ds_by_id.get(rid)
        if not is_hit(ds):
            stats["skipped_not_hit"] += 1
            continue
        code = (pool.get(rid) or {}).get("func_before") or ""
        if not code:
            stats["skipped_no_code"] += 1
            continue
        expl = strip_think(ds.get("explanation") or "")
        if not wellformed_explanation(expl) or skip_cwe_mapping(expl):
            stats["skipped_malformed"] += 1
            continue

        glm = glm_by_id.get(rid) if is_hit(glm_by_id.get(rid)) else None
        if glm:
            glm_expl = strip_think(glm.get("explanation") or "")
            if glm_expl.strip() == expl.strip():
                stats["identical_explanation_to_glm"] += 1

        review_prompt = review_user_message(code)
        if not budget.fits(review_prompt, expl):
            stats["skipped_review_over_budget"] += 1
            continue

        stats["n_ids"] += 1
        stats["gold_cwe"][ds["gold_cwe"]] += 1
        rows.append(
            sft_row(
                ds,
                turn="review",
                prompt=review_prompt,
                completion=expl,
                extra={"source": "deepseek", "map_source": None},
            )
        )
        stats["n_review"] += 1

        glm_map = strip_think((glm or {}).get("mapping") or "") if glm else ""
        ds_map = strip_think(ds.get("mapping") or "")
        if glm and glm_map and not skip_cwe_mapping(strip_think(glm.get("explanation") or "")):
            map_src = "glm"
            mapping = glm_map
            stats["glm_map_priority"] += 1
            if pred_cwe_set(glm) == pred_cwe_set(ds):
                stats["mapping_cwe_set_equal"] += 1
            else:
                stats["mapping_cwe_set_collision"] += 1
        elif ds_map:
            map_src = "deepseek"
            mapping = ds_map
            stats["ds_map_no_glm"] += 1
        else:
            stats["skipped_map_no_text"] += 1
            continue

        map_prompt = f"{review_user_message(code).rstrip()}\n\n{expl}\n\n{MAP_CWE}"
        if not budget.fits(map_prompt, mapping):
            stats["skipped_map_over_budget"] += 1
            continue
        rows.append(
            sft_row(
                ds,
                turn="map",
                prompt=map_prompt,
                completion=mapping,
                extra={"source": "deepseek", "map_source": map_src},
            )
        )
        stats["n_map"] += 1
        stats["map_source"][map_src] += 1

    stats["gold_cwe"] = dict(stats["gold_cwe"].most_common())
    stats["map_source"] = dict(stats["map_source"])
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek", type=Path, default=DS_RUN)
    ap.add_argument("--glm-hits", type=Path, default=GLM_HITS)
    ap.add_argument("--data", type=Path, default=PAIRS)
    ap.add_argument("--holdout", type=Path, default=HOLDOUT)
    ap.add_argument("--glm-val", type=Path, default=GLM_VAL)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    args = ap.parse_args()

    ds_all = load_jsonl(args.deepseek)
    ds_by_id = latest_by_id(ds_all)
    glm_by_id = latest_by_id(load_jsonl(args.glm_hits)) if args.glm_hits.exists() else {}
    holdout = load_id_set(args.holdout)
    glm_val_ids = load_id_set(args.glm_val)
    pool = load_pool(args.data)
    budget = TokenBudget(args.tokenizer, args.max_seq_length)

    hits_by_id = {i: r for i, r in ds_by_id.items() if is_hit(r)}
    train_ids = sorted(
        i for i in hits_by_id if i not in holdout and i not in glm_val_ids and i in pool
    )

    hits_rows = [hits_by_id[i] for i in sorted(hits_by_id)]
    write_jsonl(args.out_dir / "deepseek_hits.jsonl", hits_rows)

    train_rows, train_stats = build_turns(train_ids, ds_by_id, glm_by_id, pool, budget)
    write_jsonl(args.out_dir / "train.jsonl", train_rows)

    n_hits = len(hits_by_id)
    n_hits_wellformed = sum(
        1
        for r in hits_by_id.values()
        if wellformed_explanation(strip_think(r.get("explanation") or ""))
        and not skip_cwe_mapping(strip_think(r.get("explanation") or ""))
    )
    manifest = {
        "priority": ["glm", "deepseek"],
        "rule": (
            "train = DeepSeek hits only; review completion = DeepSeek explanation; "
            "map completion = GLM mapping if GLM hit else DeepSeek mapping"
        ),
        "filters": {
            "verdict_header": "explanation must open with VULNERABLE./INSUFFICIENT_INFO./SAFE",
            "skip_abstain": True,
            "max_seq_length": args.max_seq_length,
            "len_margin": LEN_MARGIN,
            "tokenizer": str(args.tokenizer),
            "exclude_holdout": True,
            "exclude_glm_val_ids": True,
        },
        "sources": {
            "deepseek": {
                "path": str(args.deepseek),
                "n_rows": len(ds_all),
                "n_unique": len(ds_by_id),
                "n_hits": n_hits,
                "n_hits_wellformed_vulnerable": n_hits_wellformed,
                "n_hits_holdout": sum(1 for i in hits_by_id if i in holdout),
            },
            "glm_hits": {
                "path": str(args.glm_hits),
                "n_unique": len(glm_by_id),
            },
        },
        "holdout": str(args.holdout),
        "glm_val": str(args.glm_val),
        "n_glm_val_ids": len(glm_val_ids),
        "train": {
            **train_stats,
            "n_turns": len(train_rows),
            "out": str(args.out_dir / "train.jsonl"),
            "gold_cwe_turns": cwe_counts(train_rows),
        },
        "val": {
            "reuse": str(args.glm_val),
            "note": "original GLM val (44 turns / 22 ids) for early-stop / forgetting guard",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
