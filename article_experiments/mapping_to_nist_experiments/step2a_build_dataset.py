#!/usr/bin/env python3
"""Step 2A: Build the classification dataset.

For each function in step1_output/functions_with_bf.jsonl:
  - locate its original frame in frames/_index.json
    (label = par:orig:<input_id>_t0.0)
  - locate its original explanation in v0_outputs.jsonl
    (input_id, temperature = 0.0)
  - serialize the frame to text

Outputs:
  step2_output/dataset.jsonl      one row per function
  step2_output/summary.txt        counts and losses
  step2_output/skipped.jsonl      functions dropped, with reasons

Usage:
  python3 step2a_build_dataset.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent            # my_training_path/
FRAMES_INDEX = PROJECT_ROOT / "article_experiments" / "frames" / "_index.json"
V0_OUTPUTS = PROJECT_ROOT / "flow_harness" / "v0_outputs.jsonl"
STEP1_OUT = HERE / "step1_output" / "functions_with_bf.jsonl"
OUT_DIR = HERE / "step2_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLES = ["source", "operation", "missing_check", "sink", "consequence"]


# ---- Loaders -----------------------------------------------------------------

def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def index_frames_by_input_id() -> dict[str, dict]:
    """Map input_id -> frame, using only 'par:orig:<id>_t0.0' labels."""
    if not FRAMES_INDEX.exists():
        raise SystemExit(f"missing {FRAMES_INDEX}")
    idx = json.loads(FRAMES_INDEX.read_text(encoding="utf-8"))

    by_input_id: dict[str, dict] = {}
    for h, v in idx.items():
        for label in v.get("labels", []):
            # exact match: par:orig:<input_id>_t0.0
            if not label.startswith("par:orig:"):
                continue
            tail = label[len("par:orig:"):]          # "<input_id>_t0.0"
            iid = tail.split("_t")[0]
            if not iid:
                continue
            # sanity: the frame must have all five roles filled
            frame = v.get("frame") or {}
            if not all((frame.get(r) or "").strip() for r in ROLES):
                continue
            if iid not in by_input_id:
                by_input_id[iid] = frame
    return by_input_id


def index_explanations_at_temp0() -> dict[str, str]:
    """Map input_id -> original explanation text (temperature 0.0)."""
    out: dict[str, str] = {}
    for r in load_jsonl(V0_OUTPUTS):
        iid = r.get("input_id") or r.get("id")
        t = r.get("temperature")
        if t is None:
            t = r.get("temp", 0.0)
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        if abs(t - 0.0) > 1e-6:
            continue
        text = (r.get("flow_text") or r.get("explanation") or "").strip()
        if iid and text:
            out[iid] = text
    return out


# ---- Frame serialization -----------------------------------------------------

def serialize_frame(frame: dict) -> str:
    """Convert a frame dict into a deterministic text representation."""
    lines = []
    for role in ROLES:
        value = (frame.get(role) or "").strip()
        lines.append(f"{role}: {value}")
    return "\n".join(lines)


# ---- Main --------------------------------------------------------------------

def main() -> None:
    step1 = load_jsonl(STEP1_OUT)
    print(f"step1 functions: {len(step1)}")

    frames = index_frames_by_input_id()
    explanations = index_explanations_at_temp0()
    print(f"indexed frames:  {len(frames)}")
    print(f"indexed expl.:   {len(explanations)}")
    print()

    dataset = []
    skipped = []

    for r in step1:
        iid = r["id"]
        frame = frames.get(iid)
        expl = explanations.get(iid)

        if frame is None:
            skipped.append({"id": iid, "reason": "no original frame"})
            continue
        if expl is None:
            skipped.append({"id": iid, "reason": "no explanation at temp=0.0"})
            continue

        dataset.append({
            "id": iid,
            "bf_class": r["bf_class"],
            "cwe": r["cwe"],
            "frame": {role: frame[role] for role in ROLES},
            "frame_text": serialize_frame(frame),
            "explanation": expl,
        })

    # ---- Write outputs ----

    with (OUT_DIR / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (OUT_DIR / "skipped.jsonl").open("w", encoding="utf-8") as f:
        for row in skipped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    bf_counts = Counter(r["bf_class"] for r in dataset)
    cwe_counts = Counter(r["cwe"] for r in dataset)
    skip_reasons = Counter(s["reason"] for s in skipped)

    lines = []
    lines.append("=== Step 2A: dataset construction ===")
    lines.append("")
    lines.append(f"Input functions (from step 1):    {len(step1)}")
    lines.append(f"Kept (frame + explanation found): {len(dataset)}")
    lines.append(f"Skipped:                          {len(skipped)}")
    for reason, n in skip_reasons.most_common():
        lines.append(f"  {reason:35s} {n}")
    lines.append("")
    lines.append("BF distribution:")
    for bf, n in sorted(bf_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {bf:6s} {n:4d}  ({n / max(len(dataset), 1):.1%})")
    lines.append("")
    lines.append("CWE distribution:")
    for cwe, n in sorted(cwe_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cwe:12s} {n}")
    lines.append("")
    lines.append("Sample serialized frame (first row):")
    if dataset:
        lines.append(dataset[0]["frame_text"])
    lines.append("")
    lines.append("Sample explanation prefix (first row):")
    if dataset:
        lines.append(dataset[0]["explanation"][:200])

    (OUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print()
    print(f"wrote {OUT_DIR}/dataset.jsonl")
    print(f"wrote {OUT_DIR}/skipped.jsonl")
    print(f"wrote {OUT_DIR}/summary.txt")


if __name__ == "__main__":
    main()