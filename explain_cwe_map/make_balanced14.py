#!/usr/bin/env python3
"""Build an equal-count sample of the 14 largest BigVul CWE labels.

Reads raw_data/bigvul_func_before.jsonl, keeps CWE with >=100 raw rows,
drops bodies longer than --max-chars, then takes min(class size) rows
from each label.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from balanced import LARGE_CWE, balance_equal, keep_large

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "raw_data" / "bigvul_func_before.jsonl"
OUT = ROOT / "raw_data" / "bigvul_func_before_balanced14.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=6000)
    args = ap.parse_args()

    rows = []
    with args.src.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            code = (rec.get("func_before") or "").strip()
            if not code:
                continue
            if args.max_chars and len(code) > args.max_chars:
                continue
            rows.append(rec)

    kept = keep_large(rows)
    balanced, k = balance_equal(kept, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in balanced:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    counts = Counter(r["cwe"] for r in balanced)
    print(
        f"large14_after_len_filter={len(kept)} per_class={k} "
        f"balanced={len(balanced)} -> {args.out}"
    )
    for cwe in LARGE_CWE:
        print(f"  {cwe:10s} {counts[cwe]}")


if __name__ == "__main__":
    main()
