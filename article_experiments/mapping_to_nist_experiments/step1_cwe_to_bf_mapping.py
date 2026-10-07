#!/usr/bin/env python3
"""Step 1: Map CWE -> BF class and determine the classification task.

Inputs:
  flow_harness/fixed_200_inputs.jsonl   (200 functions with CWE labels)

Outputs:
  step1_output/cwe_to_bf.json           (mapping table actually used)
  step1_output/functions_with_bf.jsonl  (per-function BF label)
  step1_output/distribution.json        (counts, coverage, verdict)
  step1_output/summary.txt              (human-readable report)

Usage:
  python3 step1_cwe_to_bf_mapping.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
INPUTS_PATH = PROJECT_ROOT / "flow_harness" / "fixed_200_inputs.jsonl"
OUT_DIR = HERE / "step1_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- CWE -> BF mapping -------------------------------------------------------
# Conservative mapping. Each CWE is assigned either:
#   - a single BF class (unambiguous),
#   - a list of BF classes (ambiguous — will be excluded),
#   - None (no BF mapping — will be excluded).
#
# This is a working subset of the NIST CWE2BF mappings, restricted
# to the CWE classes present in our sample.

CWE_TO_BF: dict[str, str | list[str] | None] = {
    # Memory Use (MUS)
    "CWE-125": "MUS",     # out-of-bounds read
    "CWE-787": "MUS",     # out-of-bounds write
    "CWE-119": "MUS",     # improper restriction within memory buffer
    "CWE-476": "MUS",     # null pointer dereference

    # Type Computation (TCM)
    "CWE-189": "TCM",     # numeric errors (older CWE ID, now 190/191/...)
    "CWE-190": "TCM",     # integer overflow or wraparound

    # Data Validation (DVL)
    "CWE-20": "DVL",      # improper input validation

    # Memory Management (MMN)
    "CWE-416": "MMN",     # use after free
    "CWE-399": "MMN",     # resource management errors

    # Data Verification (DVR)
    "CWE-284": "DVR",     # improper access control
    "CWE-264": "DVR",     # permissions, privileges, access control

    # Outside BF or ambiguous — excluded
    "CWE-362": None,      # race condition (BF has no direct class)
    "CWE-254": None,      # security features (too general)
    "CWE-200": None,      # information exposure (spans several BF classes)
}


# ---- Main --------------------------------------------------------------------

def load_inputs() -> list[dict]:
    rows = []
    with INPUTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    inputs = load_inputs()
    print(f"loaded functions: {len(inputs)}")

    # CWE distribution in the sample
    cwe_counts = Counter(r.get("cwe", "UNKNOWN") for r in inputs)
    print(f"distinct CWE labels: {len(cwe_counts)}")

    # Apply mapping
    per_function = []
    excluded_no_mapping = []
    excluded_ambiguous = []

    for r in inputs:
        cwe = r.get("cwe", "UNKNOWN")
        bf = CWE_TO_BF.get(cwe, "__MISSING__")

        if bf is None or bf == "__MISSING__":
            excluded_no_mapping.append({"id": r["id"], "cwe": cwe})
            continue
        if isinstance(bf, list):
            excluded_ambiguous.append({"id": r["id"], "cwe": cwe, "candidates": bf})
            continue

        per_function.append({
            "id": r["id"],
            "cwe": cwe,
            "bf_class": bf,
        })

    # Distribution
    bf_counts = Counter(p["bf_class"] for p in per_function)
    n_kept = len(per_function)
    n_total = len(inputs)

    # Verdict: what classification task is feasible?
    n_classes = len(bf_counts)
    max_class_share = max(bf_counts.values()) / n_kept if n_kept else 0.0

    if n_classes >= 3 and all(n >= 20 for n in bf_counts.values()):
        verdict = "3-or-more-class classification is feasible"
    elif n_classes == 2 and all(n >= 40 for n in bf_counts.values()):
        verdict = "binary classification is feasible"
    elif n_classes >= 1 and max_class_share > 0.80:
        verdict = "single dominant class; classification not meaningful, consider vulnerable/not-vulnerable"
    else:
        verdict = "insufficient class balance; expand sample or merge classes"

    # ---- Write outputs ----

    (OUT_DIR / "cwe_to_bf.json").write_text(
        json.dumps(
            {k: v for k, v in CWE_TO_BF.items()},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with (OUT_DIR / "functions_with_bf.jsonl").open("w", encoding="utf-8") as f:
        for p in per_function:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    distribution = {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_excluded_no_mapping": len(excluded_no_mapping),
        "n_excluded_ambiguous": len(excluded_ambiguous),
        "cwe_distribution": dict(cwe_counts),
        "bf_distribution": dict(bf_counts),
        "n_bf_classes": n_classes,
        "max_class_share": round(max_class_share, 4),
        "verdict": verdict,
    }
    (OUT_DIR / "distribution.json").write_text(
        json.dumps(distribution, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Human-readable summary
    lines = []
    lines.append("=== Step 1: CWE -> BF mapping ===")
    lines.append("")
    lines.append(f"Total functions:            {n_total}")
    lines.append(f"Kept (unambiguous BF):      {n_kept}")
    lines.append(f"Excluded (no BF mapping):   {len(excluded_no_mapping)}")
    lines.append(f"Excluded (ambiguous BF):    {len(excluded_ambiguous)}")
    lines.append("")
    lines.append("CWE distribution (raw sample):")
    for cwe, n in sorted(cwe_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cwe:12s} {n}")
    lines.append("")
    lines.append("BF distribution (kept functions):")
    for bf, n in sorted(bf_counts.items(), key=lambda x: -x[1]):
        share = n / n_kept
        lines.append(f"  {bf:6s} {n:4d}  ({share:.1%})")
    lines.append("")
    lines.append(f"Verdict: {verdict}")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print(f"wrote {OUT_DIR}/summary.txt")
    print(f"wrote {OUT_DIR}/distribution.json")
    print(f"wrote {OUT_DIR}/functions_with_bf.jsonl")


if __name__ == "__main__":
    main()