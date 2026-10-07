#!/usr/bin/env python3
"""Verify transformations: correctness vs original + agreement among models.

For each flow_id and each class (paraphrase, style):

  correctness:
    for each of the 5 transformations (one per model), compare against
    the original explanation.

  agreement:
    for each pair (i, j) of the 5 transformations, compare them to each other.

Outputs:
  sample_outputs/verified/paraphrase_correctness.jsonl
  sample_outputs/verified/style_correctness.jsonl
  sample_outputs/verified/paraphrase_agreement.jsonl
  sample_outputs/verified/style_agreement.jsonl
  sample_outputs/verified/per_input.json
  sample_outputs/verified/summary.json
  sample_outputs/verified/.cache/    (judge response cache)

Usage:
    python verify_transformations.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "sample_outputs"
VER_DIR = OUT_DIR / "verified"
VER_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = VER_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

JUDGE_MODEL = "meta-llama/llama-3.1-8b-instruct"#"deepseek/deepseek-v4.1-flash" #"anthropic/claude-3-haiku" 
JUDGE_TEMP = 0.0
MAX_TOKENS = 1024
CONCURRENCY = 6
MAX_RETRIES = 5


JUDGE_PROMPT = """You compare two vulnerability explanations.

TEXT A:
{a}

TEXT B:
{b}

Task: do these two texts describe the SAME vulnerability mechanism?

The mechanism has five roles: source, operation, missing_check, sink, consequence.
Both texts must refer to the same entities for all five roles.

Rules:
- Minor wording differences are OK.
- The two texts may differ in length or structure.
- Losing headers, lists, or code blocks is OK if the mechanism is preserved.
- If any role changed, disappeared, or was added -> FAIL.
- If the vulnerability type changed -> FAIL.
- If either text is empty or unintelligible -> FAIL.

Return ONLY valid JSON:
{{"verdict": "PASS" or "FAIL", "reason": "short reason"}}
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_client: httpx.AsyncClient | None = None
_sem = asyncio.Semaphore(CONCURRENCY)


def _cache_key(a: str, b: str) -> str:
    h = hashlib.sha256(f"{JUDGE_MODEL}\n{a}\n---\n{b}".encode()).hexdigest()[:32]
    return h


def _cache_get(a: str, b: str) -> dict | None:
    p = CACHE_DIR / f"{_cache_key(a, b)}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _cache_put(a: str, b: str, result: dict) -> None:
    p = CACHE_DIR / f"{_cache_key(a, b)}.json"
    p.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def _parse_judge_response(raw: str) -> dict:
    m = _JSON_RE.search(raw)
    if not m:
        return {"verdict": "ERROR", "reason": f"no JSON: {raw[:80]!r}"}
    try:
        obj = json.loads(m.group(0))
        v = str(obj.get("verdict", "ERROR")).upper()
        if v not in ("PASS", "FAIL", "ERROR"):
            v = "ERROR"
        return {"verdict": v, "reason": obj.get("reason", "")}
    except json.JSONDecodeError as e:
        return {"verdict": "ERROR", "reason": f"json error: {e}"}


async def judge(a: str, b: str) -> dict:
    """Compare two texts. Uses cache; returns {verdict, reason}."""
    cached = _cache_get(a, b)
    if cached is not None:
        return cached

    prompt = JUDGE_PROMPT.format(a=a, b=b)
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": JUDGE_TEMP,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with _sem:
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = await _client.post(BASE_URL, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"] or ""
                result = _parse_judge_response(raw)
                if result["verdict"] == "ERROR":
                    last_err = result["reason"]
                    print(f"  [judge retry {attempt+1}/5] parse error", file=sys.stderr)
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                _cache_put(a, b, result)
                return result
            except Exception as e:
                last_err = str(e)
                print(f"  [judge error {attempt+1}/5] {last_err}", file=sys.stderr)
                await asyncio.sleep(min(2 ** attempt, 8))

    result = {"verdict": "ERROR", "reason": f"failed after {MAX_RETRIES}: {last_err}"}
    return result


# --- I/O ---------------------------------------------------------------------

def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --- Workers -----------------------------------------------------------------

async def verify_class(cls: str) -> dict:
    """For one class (paraphrase or style), verify all flow_ids."""
    rows = load_jsonl(OUT_DIR / f"{cls}.jsonl")
    if not rows:
        print(f"  [{cls}] no rows, skipping")
        return {"n": 0}

    # group by flow_id
    by_flow: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_flow[r["flow_id"]].append(r)

    correctness_rows = []
    agreement_rows = []

    flow_ids = sorted(by_flow.keys())
    print(f"  [{cls}] {len(flow_ids)} flow_ids, {sum(len(v) for v in by_flow.values())} total records")

    for i, fid in enumerate(flow_ids, 1):
        recs = by_flow[fid]
        source = recs[0]["source_text"]

        # correctness: original vs each transformation
        corr_tasks = []
        for r in recs:
            corr_tasks.append(judge(source, r["transformed_text"]))
        corr_results = await asyncio.gather(*corr_tasks)

        for r, jr in zip(recs, corr_results):
            correctness_rows.append({
                "flow_id": fid,
                "class": cls,
                "model_name": r["model_name"],
                "style": r.get("style"),
                "verdict": jr["verdict"],
                "reason": jr["reason"],
            })

        # agreement: pairwise between transformations
        agree_tasks = []
        pairs = []
        for a in range(len(recs)):
            for b in range(a + 1, len(recs)):
                pairs.append((a, b))
                agree_tasks.append(judge(recs[a]["transformed_text"], recs[b]["transformed_text"]))
        agree_results = await asyncio.gather(*agree_tasks)

        for (a, b), jr in zip(pairs, agree_results):
            agreement_rows.append({
                "flow_id": fid,
                "class": cls,
                "model_a": recs[a]["model_name"],
                "model_b": recs[b]["model_name"],
                "verdict": jr["verdict"],
                "reason": jr["reason"],
            })

        if i % 5 == 0:
            print(f"    [{cls}] {i}/{len(flow_ids)} flow_ids processed")

    write_jsonl(VER_DIR / f"{cls}_correctness.jsonl", correctness_rows)
    write_jsonl(VER_DIR / f"{cls}_agreement.jsonl", agreement_rows)

    return {
        "n_flows": len(flow_ids),
        "correctness": correctness_rows,
        "agreement": agreement_rows,
    }


# --- Main --------------------------------------------------------------------

async def main() -> None:
    global _client
    _client = httpx.AsyncClient()

    try:
        print("=== paraphrase ===")
        p = await verify_class("paraphrase")

        print("=== style ===")
        s = await verify_class("style")

    finally:
        await _client.aclose()

    # ---- per-input and per-model summaries ----

    per_input: dict[str, dict] = {}

    def add_class(flow_id: str, cls: str, correctness: list, agreement: list):
        corr_sub = [r for r in correctness if r["flow_id"] == flow_id]
        agr_sub = [r for r in agreement if r["flow_id"] == flow_id]

        n_corr = len(corr_sub)
        n_corr_pass = sum(1 for r in corr_sub if r["verdict"] == "PASS")
        n_corr_err = sum(1 for r in corr_sub if r["verdict"] == "ERROR")

        n_agr = len(agr_sub)
        n_agr_pass = sum(1 for r in agr_sub if r["verdict"] == "PASS")
        n_agr_err = sum(1 for r in agr_sub if r["verdict"] == "ERROR")

        per_model = {}
        for r in corr_sub:
            m = r["model_name"]
            per_model.setdefault(m, {"corr_pass": 0, "corr_n": 0,
                                     "agr_pass": 0, "agr_n": 0})
            per_model[m]["corr_n"] += 1
            if r["verdict"] == "PASS":
                per_model[m]["corr_pass"] += 1
        for r in agr_sub:
            for m in (r["model_a"], r["model_b"]):
                per_model.setdefault(m, {"corr_pass": 0, "corr_n": 0,
                                         "agr_pass": 0, "agr_n": 0})
                per_model[m]["agr_n"] += 1
                if r["verdict"] == "PASS":
                    per_model[m]["agr_pass"] += 1

        per_input.setdefault(flow_id, {})[cls] = {
            "correctness_rate": n_corr_pass / n_corr if n_corr else None,
            "agreement_rate": n_agr_pass / n_agr if n_agr else None,
            "n_corr": n_corr,
            "n_agr": n_agr,
            "n_errors_corr": n_corr_err,
            "n_errors_agr": n_agr_err,
            "per_model": {
                m: {
                    "corr_pass_rate": v["corr_pass"] / v["corr_n"] if v["corr_n"] else None,
                    "agr_pass_rate": v["agr_pass"] / v["agr_n"] if v["agr_n"] else None,
                    "corr_n": v["corr_n"],
                    "agr_n": v["agr_n"],
                }
                for m, v in per_model.items()
            },
        }

    for fid in set(r["flow_id"] for r in p.get("correctness", [])) | \
               set(r["flow_id"] for r in p.get("agreement", [])):
        add_class(fid, "paraphrase", p.get("correctness", []), p.get("agreement", []))

    for fid in set(r["flow_id"] for r in s.get("correctness", [])) | \
               set(r["flow_id"] for r in s.get("agreement", [])):
        add_class(fid, "style", s.get("correctness", []), s.get("agreement", []))

    (VER_DIR / "per_input.json").write_text(
        json.dumps(per_input, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- aggregate summary ----

    def agg(rows: list[dict], key: str = "verdict") -> dict:
        c = defaultdict(int)
        for r in rows:
            c[r[key]] += 1
        n = len(rows)
        return {
            "n": n,
            "pass": c["PASS"],
            "fail": c["FAIL"],
            "error": c["ERROR"],
            "pass_rate": c["PASS"] / n if n else None,
        }

    summary = {
        "paraphrase": {
            "correctness": agg(p.get("correctness", [])),
            "agreement": agg(p.get("agreement", [])),
        },
        "style": {
            "correctness": agg(s.get("correctness", [])),
            "agreement": agg(s.get("agreement", [])),
        },
    }

    (VER_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- print ----

    print()
    print("=== VERIFICATION SUMMARY ===")
    for cls in ("paraphrase", "style"):
        print(f"\n{cls}:")
        for kind in ("correctness", "agreement"):
            b = summary[cls][kind]
            if b["n"] == 0:
                print(f"  {kind:12s} (no data)")
                continue
            print(f"  {kind:12s} n={b['n']:3d}  "
                  f"pass={b['pass']:3d}  fail={b['fail']:3d}  err={b['error']:3d}  "
                  f"pass_rate={b['pass_rate']:.3f}")

    print()
    print(f"wrote {VER_DIR}/paraphrase_correctness.jsonl")
    print(f"wrote {VER_DIR}/style_correctness.jsonl")
    print(f"wrote {VER_DIR}/paraphrase_agreement.jsonl")
    print(f"wrote {VER_DIR}/style_agreement.jsonl")
    print(f"wrote {VER_DIR}/per_input.json")
    print(f"wrote {VER_DIR}/summary.json")


if __name__ == "__main__":
    asyncio.run(main())