#!/usr/bin/env python3
"""Quick FSS on sample_outputs. Text-only parser (no code)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "sample_outputs"
FLOW_DIR = PROJECT_ROOT / "flow_harness"
CACHE_DIR = OUT_DIR / ".frame_cache"

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "deepseek/deepseek-v4.1-flash"
TEMP = 0.0
MAX_TOKENS = 2048
ROLES = ["source", "operation", "missing_check", "sink", "consequence"]
OVERLAP_THRESHOLD = 0.5

CACHE_DIR.mkdir(parents=True, exist_ok=True)


FRAME_PROMPT = """You extract a five-role frame from a vulnerability explanation.

Roles:
1. source       — where attacker-controlled data comes from
2. operation    — what is done with the data
3. missing_check — which guard or validation is absent
4. sink         — where the dangerous action occurs
5. consequence  — what results from the failure

Rules:
- Extract ONLY what is stated or directly implied in the text.
- Do NOT use outside knowledge of C or vulnerability classes.
- If a role is not mentioned, use "" (empty string).
- Keep fillers short (2-15 words).
- Output ONLY valid JSON, no fences.

Format:
{"source": "...", "operation": "...", "missing_check": "...", "sink": "...", "consequence": "..."}

EXPLANATION:
{text}

JSON:
"""


def llm(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last = None
    for attempt in range(3):
        try:
            r = httpx.post(BASE_URL, json=payload, headers=headers, timeout=120.0)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content")
            if text:
                return text
            last = RuntimeError("empty")
        except Exception as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"llm failed: {last}")


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_frame(text: str) -> dict:
    key = hashlib.sha256(text.encode()).hexdigest()[:32]
    cp = CACHE_DIR / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))

    raw = llm(FRAME_PROMPT.format(text=text))
    m = _JSON_RE.search(raw)
    if not m:
        raise RuntimeError(f"no JSON: {raw[:200]!r}")
    frame = json.loads(m.group(0))
    for r in ROLES:
        frame[r] = str(frame.get(r, "")).strip()
    cp.write_text(json.dumps(frame, ensure_ascii=False), encoding="utf-8")
    return frame


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokens(s: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(s or "")}


def overlap(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def role_ok(a: str, b: str) -> int:
    ae, be = not a.strip(), not b.strip()
    if ae and be:
        return 1
    if ae != be:
        return 0
    return 1 if overlap(a, b) >= OVERLAP_THRESHOLD else 0


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    paraphrases = load_jsonl(OUT_DIR / "paraphrase.jsonl")
    styles = load_jsonl(OUT_DIR / "style.jsonl")
    code_vars = load_jsonl(OUT_DIR / "code_variation.jsonl")
    originals = {f"{r.get('input_id') or r.get('id')}_t{r.get('temperature') or r.get('temp') or 0.0}":
                 (r.get("flow_text") or "").strip()
                 for r in load_jsonl(FLOW_DIR / "v0_outputs.jsonl")}

    print(f"loaded: {len(paraphrases)} par, {len(styles)} sty, {len(code_vars)} cv")

    pairs = []
    for r in paraphrases:
        pairs.append(("paraphrase", r["flow_id"], r["source_text"], r["transformed_text"]))
    for r in styles:
        pairs.append(("style", r["flow_id"], r["source_text"], r["transformed_text"]))
    for r in code_vars:
        iid = r["input_id"]
        orig = next((v for k, v in originals.items()
                     if k.startswith(f"{iid}_t") and (k.endswith("_t0.0") or k.endswith("_t0"))), None)
        if orig is None:
            orig = next((v for k, v in originals.items() if k.startswith(f"{iid}_t")), None)
        if orig is None:
            continue
        pairs.append(("code_variation", f"{iid}_t0.0", orig, r["generated_explanation"]))

    unique = set()
    for _, _, s, t in pairs:
        unique.add(s)
        unique.add(t)
    print(f"extracting frames for {len(unique)} unique texts")

    frames = {}
    for i, t in enumerate(unique, 1):
        try:
            frames[t] = extract_frame(t)
        except Exception as e:
            print(f"  fail: {e}")
        if i % 20 == 0:
            print(f"  {i}/{len(unique)}")

    per_pair = []
    for cls, fid, s, t in pairs:
        if s not in frames or t not in frames:
            continue
        deltas = {r: role_ok(frames[s][r], frames[t][r]) for r in ROLES}
        per_pair.append({
            "flow_id": fid, "class": cls,
            "fss": sum(deltas.values()) / len(ROLES),
            "deltas": deltas,
        })

    by_class = defaultdict(list)
    for r in per_pair:
        by_class[r["class"]].append(r["fss"])

    with (OUT_DIR / "fss_per_pair.jsonl").open("w", encoding="utf-8") as f:
        for r in per_pair:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {f"fss_{c}_mean": sum(v) / len(v) if v else None for c, v in by_class.items()}
    summary["n_pairs"] = len(per_pair)
    summary["parser_model"] = MODEL
    summary["overlap_threshold"] = OVERLAP_THRESHOLD

    (OUT_DIR / "fss_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== FSS ===")
    for k, v in summary.items():
        print(f"  {k:25s} {v if not isinstance(v, float) else f'{v:.4f}'}")


if __name__ == "__main__":
    main()