#!/usr/bin/env python3
"""Step 2F: LLM zero-shot CWE classification from three representations.

Reads step2_output/dataset.jsonl, calls an LLM on frame_text,
explanation, and code, and compares the returned CWE against ground truth.

CWE class set is read from dataset.jsonl itself (10 classes after
filtering in step 2E). Classes with < MIN_CLASS_SIZE examples are dropped.

Outputs:
  step2_output/llm_cwe_classification.jsonl
  step2_output/llm_cwe_summary.txt
  step2_output/llm_cwe_confusion_frame.txt
  step2_output/llm_cwe_confusion_explanation.txt
  step2_output/llm_cwe_confusion_code.txt

Usage:
  python3 step2f_llm_cwe_classify.py
  python3 step2f_llm_cwe_classify.py --limit 10
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
CACHE_DIR = OUT_DIR / "llm_cwe_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

DEFAULT_MODEL = "openai/gpt-4o-mini"
TEMP = 0.0
MAX_TOKENS = 20
TIMEOUT = 60.0
MIN_CLASS_SIZE = 3


PROMPT_TEMPLATE = """You are a security analyst. Given an analysis input, determine which CWE (Common Weakness Enumeration) category best describes the vulnerability.

Return ONLY the CWE identifier in the format CWE-NNN, nothing else.
Do not include any explanation, punctuation, or additional text.

INPUT:
{representation_text}

CWE:
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


_CWE_RE = re.compile(r"CWE[- ]?(\d{1,4})", re.IGNORECASE)


def parse_cwe(raw: str) -> str | None:
    m = _CWE_RE.search(raw or "")
    if not m:
        return None
    return f"CWE-{m.group(1)}"


# ---- Metrics -----------------------------------------------------------------

def format_confusion(cm: list[list[int]], classes: list[str]) -> str:
    lines = ["rows = true, cols = predicted"]
    lines.append("        " + "".join(f"{c[-4:]:>8s}" for c in classes))
    for i, c in enumerate(classes):
        row = "".join(f"{v:>8d}" for v in cm[i])
        lines.append(f"{c:>8s}{row}")
    return "\n".join(lines)


def compute_metrics(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    n_total = len(y_true)
    n_parsed = len(pairs)

    if n_parsed == 0:
        return {
            "n_total": n_total, "n_parsed": 0, "parse_rate": 0.0,
            "accuracy": None, "macro_f1": None,
            "confusion_matrix": None, "classes": labels,
        }

    yt = [t for t, _ in pairs]
    yp = [p for _, p in pairs]

    cm = confusion_matrix(yt, yp, labels=labels)
    return {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / n_total,
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "confusion_matrix": cm.tolist(),
        "classes": labels,
        "report": classification_report(
            yt, yp, labels=labels, output_dict=True, zero_division=0
        ),
    }


# ---- Main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not set")

    rows = load_dataset()
    code_map = load_code_by_input_id()
    for r in rows:
        r["code"] = code_map.get(r["id"], "")

    # Filter rows without code
    rows = [r for r in rows if r.get("code")]

    # Keep only CWEs with enough examples
    cwe_counts = Counter(r["cwe"] for r in rows)
    keep = {c for c, n in cwe_counts.items() if n >= MIN_CLASS_SIZE}
    rows = [r for r in rows if r["cwe"] in keep]

    if args.limit:
        rows = rows[: args.limit]

    labels = sorted(keep)
    print(f"rows:    {len(rows)}")
    print(f"classes: {len(labels)}")
    print(f"model:   {args.model}")
    print()

    preds = {"frame": [], "explanation": [], "code": []}
    out_rows = []

    for i, r in enumerate(rows, 1):
        row_out = {"id": r["id"], "cwe_true": r["cwe"]}

        for name, text in [
            ("frame", r["frame_text"]),
            ("explanation", r["explanation"]),
            ("code", r["code"]),
        ]:
            if not text.strip():
                row_out[f"cwe_pred_{name}"] = None
                preds[name].append(None)
                continue
            prompt = PROMPT_TEMPLATE.format(representation_text=text)
            try:
                raw = call_llm(prompt, args.model)
                row_out[f"cwe_pred_{name}_raw"] = raw[:120]
                row_out[f"cwe_pred_{name}"] = parse_cwe(raw)
            except Exception as e:
                row_out[f"cwe_pred_{name}"] = None
                row_out[f"cwe_pred_{name}_raw"] = f"ERROR: {e}"
            preds[name].append(row_out.get(f"cwe_pred_{name}"))

        out_rows.append(row_out)
        if i % 10 == 0:
            print(f"  {i}/{len(rows)} done")

    y_true = [r["cwe"] for r in rows]

    metrics = {name: compute_metrics(y_true, preds[name], labels) for name in preds}

    # ---- Write ----

    with (OUT_DIR / "llm_cwe_classification.jsonl").open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for name, m in metrics.items():
        if m["confusion_matrix"] is not None:
            (OUT_DIR / f"llm_cwe_confusion_{name}.txt").write_text(
                format_confusion(m["confusion_matrix"], labels) + "\n",
                encoding="utf-8",
            )

    lines = []
    lines.append("=== Step 2F: LLM zero-shot CWE classification ===")
    lines.append("")
    lines.append(f"model:           {args.model}")
    lines.append(f"rows:            {len(rows)}")
    lines.append(f"CWE classes:     {len(labels)}")
    lines.append(f"random baseline: {1.0 / len(labels):.4f}")
    lines.append("")
    lines.append(f"{'representation':<16s}{'parse%':>10s}{'accuracy':>12s}{'macro-F1':>12s}")
    lines.append("-" * 50)
    for name in ["frame", "explanation", "code"]:
        m = metrics[name]
        parse = f"{m['parse_rate']:.3f}"
        acc = f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "n/a"
        f1 = f"{m['macro_f1']:.4f}" if m["macro_f1"] is not None else "n/a"
        lines.append(f"{name:<16s}{parse:>10s}{acc:>12s}{f1:>12s}")
    lines.append("")
    lines.append("class distribution:")
    for c, n in sorted(Counter(y_true).items(), key=lambda x: -x[1]):
        lines.append(f"  {c:12s} {n}")
    lines.append("")
    for name in ["frame", "explanation", "code"]:
        m = metrics[name]
        lines.append(f"confusion matrix ({name}):")
        if m["confusion_matrix"] is not None:
            lines.append(format_confusion(m["confusion_matrix"], labels))
        lines.append("")

    (OUT_DIR / "llm_cwe_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print(f"wrote {OUT_DIR}/llm_cwe_summary.txt")


if __name__ == "__main__":
    main()