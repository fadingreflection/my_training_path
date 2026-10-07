#!/usr/bin/env python3
"""Report frame fill rates: overall, by role, and by source category.

Reads frames/_index.json, produced by extract_frames.py.

Reports:
  - how many frames are fully filled (5/5), partial (<5), empty (0)
  - distribution by number of filled roles
  - which roles are empty most often
  - fill rates broken down by source category
    (original vs transformed, per class)

Usage:
    python check_frames.py
    python check_frames.py --strict   # exit code 1 if < 50% fully filled
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMES = HERE / "frames" / "_index.json"
ROLES = ["source", "operation", "missing_check", "sink", "consequence"]


def classify(labels: list[str]) -> str:
    """Classify a frame by its primary source category.

    Labels look like:
      par:orig:<flow_id>
      par:tr:<flow_id>:<model_name>
      sty:orig:<flow_id>
      sty:tr:<flow_id>:<model_name>
    """
    for lbl in labels:
        for prefix, name in (
            ("par:tr:", "paraphrase (transformed)"),
            ("sty:tr:", "style (transformed)"),
            ("par:orig:", "paraphrase (original)"),
            ("sty:orig:", "style (original)"),
        ):
            if lbl.startswith(prefix):
                return name
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit with code 1 if fewer than 50%% of frames are fully filled",
    )
    args = ap.parse_args()

    if not FRAMES.exists():
        raise SystemExit(f"missing {FRAMES} — run extract_frames.py first")

    idx = json.loads(FRAMES.read_text(encoding="utf-8"))
    if not idx:
        raise SystemExit(f"{FRAMES} is empty — extract_frames.py produced no frames")

    n_total = len(idx)
    fill_count = Counter()              # number of filled roles -> count of frames
    role_empty = Counter()              # role -> how many frames have it empty
    by_source = defaultdict(Counter)    # source category -> fill_count

    fully = partial = 0

    for h, v in idx.items():
        frame = v["frame"]
        labels = v.get("labels", [])

        empty_roles = [r for r in ROLES if not (frame.get(r) or "").strip()]
        n_filled = len(ROLES) - len(empty_roles)

        fill_count[n_filled] += 1
        for r in empty_roles:
            role_empty[r] += 1

        src = classify(labels)
        by_source[src][n_filled] += 1

        if n_filled == len(ROLES):
            fully += 1
        else:
            partial += 1

    print(f"total frames:   {n_total}")
    print(f"fully filled:   {fully}  ({fully / n_total:.1%})")
    print(f"partial:        {partial}  ({partial / n_total:.1%})")

    print()
    print("distribution by number of filled roles:")
    for n in range(len(ROLES) + 1):
        print(f"  {n}/5: {fill_count[n]:5d}  ({fill_count[n] / n_total:6.1%})")

    print()
    print("empty roles (across all frames):")
    for r in ROLES:
        print(f"  {r:15s} {role_empty[r]:5d}  ({role_empty[r] / n_total:6.1%})")

    print()
    print("by source category:")
    for src in sorted(by_source.keys()):
        counts = by_source[src]
        n_src = sum(counts.values())
        if n_src == 0:
            continue
        full_src = counts.get(len(ROLES), 0)
        print(
            f"  {src:30s} "
            f"n={n_src:5d}  "
            f"fully filled={full_src:5d}  "
            f"({full_src / n_src:.1%})"
        )

    print()
    if fully / n_total < 0.5:
        msg = f"WARNING: only {fully / n_total:.1%} of frames are fully filled"
        print(msg, file=sys.stderr)
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()