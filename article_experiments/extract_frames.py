#!/usr/bin/env python3
"""Extract frames from sample_outputs and save them for inspection.

Model: deepseek/deepseek-v4.1-flash with reasoning limited to 512 tokens,
so content has room for the final JSON.

No FSS. No metrics. Just frames.

Outputs:
  frames/{hash}_{prompt_tag}.json  — extracted frame
  frames/_index.json               — hash -> {labels, text_len, frame}
  frames/_raw/{hash}.txt           — raw LLM response
  frames/_errors.jsonl             — failures with raw response
  frames/_texts/{hash}.txt         — input text per hash
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "sample_outputs"
FLOW_DIR = PROJECT_ROOT / "flow_harness"
FRAMES_DIR = HERE / "frames"
RAW_DIR = FRAMES_DIR / "_raw"
TEXTS_DIR = FRAMES_DIR / "_texts"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
TEXTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "deepseek/deepseek-v4.1-flash"
TEMP = 0.0
MAX_TOKENS = 8192
REASONING_MAX = 512

ROLES = ["source", "operation", "missing_check", "sink", "consequence"]

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


FRAME_PROMPT = """You extract a five-role frame from a vulnerability explanation.

ROLES:
  1. source        — where attacker-controlled data comes from
  2. operation     — what is done with the data
  3. missing_check — which guard or validation is absent
  4. sink          — where the dangerous action occurs
  5. consequence   — what results from the failure

YOUR TASK:
For each role, produce a SHORT ABSTRACT CONCEPT that captures the invariant
meaning of that role — independent of the specific wording used in the
explanation and independent of the specific variable names.

The concept should be the same whenever the underlying mechanism element is
the same, even if the explanation uses different words.

Do NOT use a fixed list of labels. Choose the most natural short concept
for this role in this explanation. Aim for 2–5 words.

LEVEL OF ABSTRACTION — examples of what is expected:

  Surface phrasing                              -> Abstract concept
  ─────────────────────────────────────────────────────────────────────
  "strcpy copies input"                         -> "unbounded string copy"
  "input copied via string copy without bound"  -> "unbounded string copy"
  "memcpy with attacker size"                   -> "memory copy with attacker length"
  "parse the markup string"                     -> "parsing attacker string"
  "markup is parsed and inserted"               -> "parsing attacker string"
  "no length check"                             -> "missing bounds check"
  "length not validated"                        -> "missing bounds check"
  "no sanitization of input"                    -> "missing input sanitization"
  "input is not escaped"                        -> "missing input sanitization"
  "stack buffer overflow"                       -> "buffer overflow"
  "OOB write past 256-byte buffer"              -> "buffer overflow"
  "heap corruption via overflow"                -> "buffer overflow"
  "script executes in wrong scope"              -> "code execution"
  "arbitrary code runs"                         -> "code execution"

Note how the same underlying mechanism maps to the same abstract concept
across different surface phrasings.

RULES:
- source and sink: 2–6 words, keep the identifier when it is named in the text
  (e.g., "markup parameter", "replaceChildrenWithFragment call", "buf stack array").
  Do NOT normalize identifiers to categories — they are names, not concepts.
- operation, missing_check, consequence: 2–5 words, abstract concept, NOT the
  verbatim phrase from the text.
- Do NOT use a fixed vocabulary. There is no list to choose from.
- Extract ONLY what is stated or directly implied.
- If a role is not mentioned, use "" (empty string).
- Output ONLY valid JSON, no code fences.
- Keys exactly: source, operation, missing_check, sink, consequence

EXPLANATION:
{text}

JSON:
"""


def call_llm(prompt: str, label: str = "") -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract structured data. Respond only with a single "
                    "valid JSON object. No preamble, no explanations, no fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"max_tokens": REASONING_MAX},
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    last_err = None
    for attempt in range(3):
        try:
            r = httpx.post(BASE_URL, json=payload, headers=headers, timeout=180.0)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason")

            text = (msg.get("content") or "").strip()
            if text:
                return text

            # fallback: try to pull JSON out of reasoning if content is empty
            reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "")
            m = JSON_RE.search(reasoning)
            if m:
                return m.group(0)

            last_err = RuntimeError(f"empty content (finish={finish})")
            print(f"  [empty] {label} attempt={attempt+1}/3 finish={finish}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            last_err = e
            print(f"  [error] {label} attempt={attempt+1}/3: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"llm failed: {last_err}")


def strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def normalize_keys(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        k2 = k.strip().strip('"').strip("'").replace("\n", "").strip()
        out[k2] = v
    return out


def parse_frame(raw: str) -> dict:
    text = strip_fences(raw)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = JSON_RE.search(text)
        if not m:
            raise RuntimeError(f"no JSON found in: {text[:200]!r}")
        obj = json.loads(m.group(0))

    if isinstance(obj, str):
        obj = json.loads(obj)

    if not isinstance(obj, dict):
        raise RuntimeError(f"parsed object is not a dict: {type(obj).__name__}")

    obj = normalize_keys(obj)

    frame = {}
    for r in ROLES:
        frame[r] = str(obj.get(r, "")).strip()
    return frame


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def hash_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def main() -> None:
    paraphrases = load_jsonl(OUT_DIR / "paraphrase.jsonl")
    styles = load_jsonl(OUT_DIR / "style.jsonl")

    # Map hashes to unique texts. Use model_name for labels (no index anymore).
    labelled: dict[str, list[str]] = {}

    def add(text: str, label: str) -> None:
        h = hash_text(text)
        labelled.setdefault(h, []).append(label)
        (TEXTS_DIR / f"{h}.txt").write_text(text, encoding="utf-8")

    for r in paraphrases:
        add(r["source_text"], f"par:orig:{r['flow_id']}")
        add(r["transformed_text"], f"par:tr:{r['flow_id']}:{r['model_name']}")

    for r in styles:
        add(r["source_text"], f"sty:orig:{r['flow_id']}")
        add(r["transformed_text"], f"sty:tr:{r['flow_id']}:{r['model_name']}")

    texts = {h: (TEXTS_DIR / f"{h}.txt").read_text(encoding="utf-8") for h in labelled}

    print(f"unique texts: {len(texts)}")
    print(f"model:        {MODEL}")
    print(f"reasoning:    max_tokens={REASONING_MAX}, max_tokens={MAX_TOKENS}")
    print()

    index: dict[str, dict] = {}
    errors: list[dict] = []
    role_fill = Counter()

    step = max(1, len(texts) // 10)

    for i, (h, text) in enumerate(sorted(texts.items()), 1):
        fp = FRAMES_DIR / f"{h}.json"
        if fp.exists():
            frame = json.loads(fp.read_text(encoding="utf-8"))
        else:
            try:
                raw = call_llm(FRAME_PROMPT.format(text=text), label=h)
                (RAW_DIR / f"{h}.txt").write_text(raw, encoding="utf-8")
                frame = parse_frame(raw)

                # reject all-empty frames
                if all(not frame.get(r, "").strip() for r in ROLES):
                    raise RuntimeError("all roles empty")

                fp.write_text(json.dumps(frame, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                errors.append({
                    "hash": h,
                    "labels": labelled[h],
                    "error": str(e),
                    "text_excerpt": text[:150],
                })
                print(f"  [{i}/{len(texts)}] FAIL {h}: {e}", file=sys.stderr)
                continue

        for r in ROLES:
            if frame.get(r, "").strip():
                role_fill[r] += 1

        index[h] = {
            "labels": labelled[h],
            "text_len": len(text),
            "frame": frame,
        }

        if i % step == 0 or i == len(texts):
            print(f"  {i}/{len(texts)}")
            (FRAMES_DIR / "_index.json").write_text(
                json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    # final index and error dump
    (FRAMES_DIR / "_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    with (FRAMES_DIR / "_errors.jsonl").open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print()
    print(f"extracted frames: {len(index)}")
    print(f"errors:           {len(errors)}")
    print()
    print("Role fill rate (non-empty):")
    for r in ROLES:
        n = role_fill[r]
        print(f"  {r:15s} {n}/{len(index)}  {n / max(len(index), 1):.3f}")


if __name__ == "__main__":
    main()