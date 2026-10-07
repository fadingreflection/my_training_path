#!/usr/bin/env python3
"""Step 2D: LLM zero-shot BF-classification from three representations.

For each row in step2_output/dataset.jsonl, call an LLM three times:
  - on frame_text
  - on explanation
  - on code
Parse the returned BF class, compare against ground truth, and against
the logreg baseline from Step 2B/2C.

Outputs:
  step2_output/llm_classification.jsonl     per-row predictions
  step2_output/llm_summary.txt
  step2_output/llm_confusion_frame.txt
  step2_output/llm_confusion_explanation.txt
  step2_output/llm_confusion_code.txt

Usage:
  export OPENROUTER_API_KEY=...
  python3 step2d_llm_classify.py
  python3 step2d_llm_classify.py --model google/gemini-2.5-flash-lite
  python3 step2d_llm_classify.py --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DATA_PATH = HERE / "step2_output" / "dataset.jsonl"
INPUTS_PATH = PROJECT_ROOT / "flow_harness" / "fixed_200_inputs.jsonl"
OUT_DIR = HERE / "step2_output"
CACHE_DIR = OUT_DIR / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

DEFAULT_MODEL = "openai/gpt-4o-mini"
TEMP = 0.0
MAX_TOKENS = 20
TIMEOUT = 60.0

CLASSES = ["MUS", "TCM", "MMN", "DVR", "DVL"]


PROMPT_TEMPLATE = """You are a security analyst classifying vulnerabilities in C code.

Given an analysis input, determine which NIST Bugs Framework (BF) class best describes the vulnerability.

BF classes:
- MUS (Memory Use): buffer overflow, out-of-bounds read/write, null pointer dereference. 
- TCM (Type Computation): integer overflow, numeric errors, wraparound. 
- MMN (Memory Management): use-after-free, double-free, resource management errors. 
- DVR (Data Verification): improper access control, permission issues. 
- DVL (Data Validation): improper input validation. 

Return ONLY one class name from this list: MUS, TCM, MMN, DVR, DVL.
Do not include any explanation. Do not include punctuation.

INPUT:
{representation_text}

CLASS:
"""


# ---- Load --------------------------------------------------------------------

def load_dataset() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(f"missing {DATA_PATH}")
    return [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_code_by_input_id() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in INPUTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["id"]] = r.get("code") or ""
    return out


# ---- LLM ---------------------------------------------------------------------

_client = httpx.Client(timeout=TIMEOUT)


def _cache_path(text: str, model: str) -> Path:
    h = hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()[:32]
    return CACHE_DIR / f"{h}.txt"


def call_llm(prompt: str, model: str) -> str:
    cached = _cache_path(prompt, model)
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(3):
        try:
            r = _client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload, headers=headers,
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            if not text:
                reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                text = reasoning.strip()
            cached.write_text(text, encoding="utf-8")
            return text
        except Exception as e:
            last_err = e
    raise RuntimeError(f"LLM call failed: {last_err}")


_WORD_RE = re.compile(r"\b(MUS|TCM|MMN|DVR|DVL)\b")


def parse_class(raw: str) -> str | None:
    m = _WORD_RE.search(raw or "")
    return m.group(1) if m else None


# ---- Metrics -----------------------------------------------------------------

def format_confusion(cm: list[list[int]], classes: list[str]) -> str:
    lines = ["rows = true, cols = predicted"]
    lines.append("        " + "".join(f"{c:>8s}" for c in classes))
    for i, c in enumerate(classes):
        row = "".join(f"{v:>8d}" for v in cm[i])
        lines.append(f"{c:>8s}{row}")
    return "\n".join(lines)


def compute_metrics(y_true: list[str], y_pred: list[str | None]) -> dict:
    # keep only rows where prediction parsed
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    n_parsed = len(pairs)
    n_total = len(y_true)

    if n_parsed == 0:
        return {
            "n_total": n_total,
            "n_parsed": 0,
            "parse_rate": 0.0,
            "accuracy": None,
            "macro_f1": None,
            "confusion_matrix": None,
            "classes": CLASSES,
        }

    yt = [t for t, _ in pairs]
    yp = [p for _, p in pairs]

    cm = confusion_matrix(yt, yp, labels=CLASSES)
    return {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / n_total,
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "report": classification_report(yt, yp, labels=CLASSES, output_dict=True, zero_division=0),
    }


# ---- Main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0,
                    help="if > 0, only the first N rows (for smoke tests)")
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not set")

    rows = load_dataset()
    code_map = load_code_by_input_id()
    for r in rows:
        r["code"] = code_map.get(r["id"], "")

    if args.limit:
        rows = rows[: args.limit]

    print(f"rows:  {len(rows)}")
    print(f"model: {args.model}")
    print()

    preds_frame: list[str | None] = []
    preds_expl: list[str | None] = []
    preds_code: list[str | None] = []

    out_rows = []

    for i, r in enumerate(rows, 1):
        reps = {
            "frame": r["frame_text"],
            "explanation": r["explanation"],
            "code": r["code"],
        }

        row_out = {
            "id": r["id"],
            "bf_true": r["bf_class"],
            "cwe": r["cwe"],
        }

        for name, text in reps.items():
            if not text.strip():
                row_out[f"bf_pred_{name}"] = None
                continue
            prompt = PROMPT_TEMPLATE.format(representation_text=text)
            try:
                raw = call_llm(prompt, args.model)
                row_out[f"bf_pred_{name}_raw"] = raw[:120]
                row_out[f"bf_pred_{name}"] = parse_class(raw)
            except Exception as e:
                row_out[f"bf_pred_{name}"] = None
                row_out[f"bf_pred_{name}_raw"] = f"ERROR: {e}"

        preds_frame.append(row_out.get("bf_pred_frame"))
        preds_expl.append(row_out.get("bf_pred_explanation"))
        preds_code.append(row_out.get("bf_pred_code"))
        out_rows.append(row_out)

        if i % 10 == 0:
            print(f"  {i}/{len(rows)} done")

    # ---- Metrics ----

    y_true = [r["bf_class"] for r in rows]

    m_frame = compute_metrics(y_true, preds_frame)
    m_expl = compute_metrics(y_true, preds_expl)
    m_code = compute_metrics(y_true, preds_code)

    # ---- Write ----

    with (OUT_DIR / "llm_classification.jsonl").open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for name, m in [("frame", m_frame), ("explanation", m_expl), ("code", m_code)]:
        if m["confusion_matrix"] is not None:
            (OUT_DIR / f"llm_confusion_{name}.txt").write_text(
                format_confusion(m["confusion_matrix"], CLASSES) + "\n",
                encoding="utf-8",
            )

    lines = []
    lines.append("=== Step 2D: LLM zero-shot classification ===")
    lines.append("")
    lines.append(f"model:           {args.model}")
    lines.append(f"rows:            {len(rows)}")
    lines.append("")
    lines.append(f"{'representation':<16s}{'parse%':>10s}{'accuracy':>12s}{'macro-F1':>12s}")
    lines.append("-" * 50)
    for name, m in [("frame", m_frame), ("explanation", m_expl), ("code", m_code)]:
        parse = f"{m['parse_rate']:.3f}"
        acc = f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "n/a"
        f1 = f"{m['macro_f1']:.4f}" if m["macro_f1"] is not None else "n/a"
        lines.append(f"{name:<16s}{parse:>10s}{acc:>12s}{f1:>12s}")
    lines.append("")
    lines.append("confusion matrix (frame):")
    if m_frame["confusion_matrix"] is not None:
        lines.append(format_confusion(m_frame["confusion_matrix"], CLASSES))
    lines.append("")
    lines.append("confusion matrix (explanation):")
    if m_expl["confusion_matrix"] is not None:
        lines.append(format_confusion(m_expl["confusion_matrix"], CLASSES))
    lines.append("")
    lines.append("confusion matrix (code):")
    if m_code["confusion_matrix"] is not None:
        lines.append(format_confusion(m_code["confusion_matrix"], CLASSES))

    (OUT_DIR / "llm_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print(f"wrote {OUT_DIR}/llm_classification.jsonl")
    print(f"wrote {OUT_DIR}/llm_summary.txt")


if __name__ == "__main__":
    main()