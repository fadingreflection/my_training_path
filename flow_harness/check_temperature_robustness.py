#!/usr/bin/env python3
"""Temperature robustness check (Section 3.1.3).

Reads flow_harness/v0_outputs.jsonl (200 functions x 3 temperatures),
computes within-function token overlap, cross-function baseline,
and the gap.

  python3 temperature_robustness.py
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE / "v0_outputs.jsonl"
OUT_PATH = HERE / "temperature_robustness.json"

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
GAP_THRESHOLD = 0.05
CROSS_SAMPLE_SIZE = 5000
SEED = 42


def tokenize(text: str) -> set[str]:
    return set(t.lower() for t in TOKEN_RE.findall(text or ""))


def overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def load_outputs(path: Path) -> dict[str, list[dict]]:
    by_input: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_input[rec["input_id"]].append(rec)
    return by_input


def within_similarity(records: list[dict]) -> float | None:
    if len(records) < 2:
        return None
    tokens = [tokenize(r.get("flow_text") or "") for r in records]
    pairs = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            pairs.append(overlap(tokens[i], tokens[j]))
    return sum(pairs) / len(pairs) if pairs else None


def cross_similarity(
    by_input: dict[str, list[dict]], n_pairs: int, rng: random.Random
) -> float:
    input_ids = list(by_input.keys())
    values = []
    attempts = 0
    max_attempts = n_pairs * 20
    while len(values) < n_pairs and attempts < max_attempts:
        attempts += 1
        a_id, b_id = rng.sample(input_ids, 2)
        a_rec = rng.choice(by_input[a_id])
        b_rec = rng.choice(by_input[b_id])
        ta = tokenize(a_rec.get("flow_text") or "")
        tb = tokenize(b_rec.get("flow_text") or "")
        if not ta or not tb:
            continue
        values.append(overlap(ta, tb))
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"missing input: {INPUT_PATH}")

    by_input = load_outputs(INPUT_PATH)
    rng = random.Random(SEED)

    within_values = []
    per_input = {}
    for iid, records in by_input.items():
        sim = within_similarity(records)
        per_input[iid] = {
            "n_samples": len(records),
            "within": sim,
        }
        if sim is not None:
            within_values.append(sim)

    if not within_values:
        raise SystemExit("no within-function pairs computed")

    mean_within = statistics.mean(within_values)
    mean_cross = cross_similarity(by_input, CROSS_SAMPLE_SIZE, rng)
    gap = mean_within - mean_cross

    summary = {
        "n_functions": len(by_input),
        "n_within_values": len(within_values),
        "mean_within": round(mean_within, 4),
        "median_within": round(statistics.median(within_values), 4),
        "stdev_within": round(statistics.pstdev(within_values), 4),
        "mean_cross": round(mean_cross, 4),
        "gap": round(gap, 4),
        "threshold": GAP_THRESHOLD,
        "verdict": "pass" if gap >= GAP_THRESHOLD else "fail",
        "cross_sample_size": CROSS_SAMPLE_SIZE,
        "seed": SEED,
        "per_input": per_input,
    }

    OUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"functions:        {summary['n_functions']}")
    print(f"mean within:      {summary['mean_within']}")
    print(f"median within:    {summary['median_within']}")
    print(f"stdev within:     {summary['stdev_within']}")
    print(f"mean cross:       {summary['mean_cross']}")
    print(f"gap:              {summary['gap']}")
    print(f"threshold:        {summary['threshold']}")
    print(f"verdict:          {summary['verdict']}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()