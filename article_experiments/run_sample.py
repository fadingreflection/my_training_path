#!/usr/bin/env python3
"""Generate paraphrase and style-shift transformations using 5 different models.

For each function, selects one canonical explanation (temperature --temp)
and generates:
  - 5 paraphrases       (one per model)
  - 5 style shifts      (one per model, alternating formal/informal)

Records are written to disk immediately after each generation.
On re-run, records whose (class, flow_id, model_name) already exist are skipped.
To force a full re-run, delete the corresponding .jsonl file.

Usage:
    export OPENROUTER_API_KEY=...
    python run_sample.py --n 10 --temp 0.0 --seed 42
    python run_sample.py --n none
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

from transforms_prompts import (
    PARAPHRASE_PROMPT,
    STYLE_FORMAL_PROMPT,
    STYLE_INFORMAL_PROMPT,
)


# --- Paths -------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
FLOW_DIR = PROJECT_ROOT / "flow_harness"
INPUTS_PATH = FLOW_DIR / "fixed_200_inputs.jsonl"
OUTPUTS_PATH = FLOW_DIR / "v0_outputs.jsonl"
OUT_DIR = HERE / "sample_outputs"

ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)


# --- Config ------------------------------------------------------------------

# Five models from different families, to decorrelate model-specific bias.
# Five cheap models from different families, to decorrelate model-specific bias.
DEFAULT_MODELS = [
    "qwen/qwen3.5-9b",                     # $0.05 / Mtok — самый дешёвый
    "google/gemini-2.5-flash-lite",        # $0.10 / Mtok
    "meta-llama/llama-3.3-70b-instruct",   # $0.10 / Mtok
    "openai/gpt-4o-mini",                  # $0.15 / Mtok
    "mistralai/mistral-small-2603",        # $0.15 / Mtok
]

MODELS = [
    m.strip()
    for m in os.environ.get("TRANSFORM_MODELS", ",".join(DEFAULT_MODELS)).split(",")
    if m.strip()
]
if len(MODELS) != 5:
    raise SystemExit(
        f"TRANSFORM_MODELS must contain exactly 5 models, got {len(MODELS)}: {MODELS}"
    )

CONCURRENCY = int(os.environ.get("TRANSFORM_CONCURRENCY", "6"))
MAX_RETRIES = 3
GEN_TEMP = 0.7
MAX_TOKENS = 4096
REASONING_MAX = 1024


# --- I/O ---------------------------------------------------------------------

def load_inputs() -> dict[str, dict]:
    inputs: dict[str, dict] = {}
    with INPUTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                inputs[r["id"]] = r
    return inputs


def load_outputs() -> list[dict]:
    rows: list[dict] = []
    with OUTPUTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_flow_text(rec: dict) -> str:
    return (rec.get("flow_text") or rec.get("explanation") or "").strip()


# --- Incremental writer ------------------------------------------------------

class IncrementalWriter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def load_existing_keys() -> set[tuple[str, str, str]]:
    """Return set of (class, flow_id, model_name) already on disk."""
    keys: set[tuple[str, str, str]] = set()
    for cls in ("paraphrase", "style"):
        path = OUT_DIR / f"{cls}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fid = r.get("flow_id")
                model = r.get("model_name")
                if fid is None or model is None:
                    continue
                keys.add((cls, fid, model))
    return keys


# --- LLM ---------------------------------------------------------------------

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)
sem = asyncio.Semaphore(CONCURRENCY)


async def llm(model: str, prompt: str, temperature: float, label: str = "") -> str:
    async with sem:
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": MAX_TOKENS,
                }
                # Only some models accept the reasoning hint; ignore errors.
                try:
                    resp = await client.chat.completions.create(
                        extra_body={"reasoning": {"max_tokens": REASONING_MAX}},
                        **kwargs,
                    )
                except Exception:
                    resp = await client.chat.completions.create(**kwargs)

                msg = resp.choices[0].message
                text = (msg.content or "").strip()
                if not text:
                    finish = resp.choices[0].finish_reason
                    print(
                        f"[empty] {label} model={model} finish={finish}",
                        file=sys.stderr,
                    )
                    last_err = RuntimeError("empty response")
                    await asyncio.sleep(2 ** attempt)
                    continue
                return text
            except Exception as e:
                last_err = e
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"llm failed ({model}): {last_err}")


# --- Workers -----------------------------------------------------------------

async def gen_paraphrase(
    flow_id: str,
    text: str,
    model_name: str,
    writer: IncrementalWriter,
    existing: set,
) -> dict | None:
    key = ("paraphrase", flow_id, model_name)
    if key in existing:
        return None

    prompt = PARAPHRASE_PROMPT.format(explanation=text)
    out = await llm(model_name, prompt, GEN_TEMP, label=f"par:{flow_id}:{model_name}")
    rec = {
        "flow_id": flow_id,
        "class": "paraphrase",
        "model_name": model_name,
        "source_text": text,
        "transformed_text": out,
    }
    writer.write(rec)
    existing.add(key)
    return rec


async def gen_style(
    flow_id: str,
    text: str,
    model_name: str,
    style: str,   # "formal" or "informal"
    writer: IncrementalWriter,
    existing: set,
) -> dict | None:
    key = ("style", flow_id, model_name)
    if key in existing:
        return None

    template = STYLE_FORMAL_PROMPT if style == "formal" else STYLE_INFORMAL_PROMPT
    prompt = template.format(explanation=text)
    out = await llm(model_name, prompt, GEN_TEMP, label=f"sty:{flow_id}:{model_name}")
    rec = {
        "flow_id": flow_id,
        "class": "style",
        "model_name": model_name,
        "style": style,
        "source_text": text,
        "transformed_text": out,
    }
    writer.write(rec)
    existing.add(key)
    return rec


# --- Main --------------------------------------------------------------------

def parse_n(value: str, total: int) -> int:
    v = value.strip().lower()
    if v in ("none", "all", "0", "-1"):
        return total
    try:
        n = int(v)
        if n <= 0 or n > total:
            return total
        return n
    except ValueError:
        raise SystemExit(f"invalid --n value: {value!r} (use integer, 'none' or 'all')")


async def main(n_arg: str, temp: float, seed: int) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set")

    outputs = load_outputs()
    by_input: dict[str, list[dict]] = {}
    for rec in outputs:
        iid = rec.get("input_id") or rec.get("id")
        by_input.setdefault(iid, []).append(rec)

    def pick_canonical(recs: list[dict]) -> dict:
        for r in recs:
            t = r.get("temperature", r.get("temp"))
            if t is not None and abs(float(t) - temp) < 1e-6:
                return r
        fallback_t = recs[0].get("temperature", recs[0].get("temp"))
        print(
            f"  [warn] no record at temperature={temp}, using {fallback_t}",
            file=sys.stderr,
        )
        return recs[0]

    def get_rec_temp(rec: dict) -> float:
        t = rec.get("temperature")
        if t is None:
            t = rec.get("temp", temp)
        return float(t)

    total = len(by_input)
    n = parse_n(n_arg, total)

    rng = random.Random(seed)
    input_ids = list(by_input.keys())
    rng.shuffle(input_ids)
    chosen = input_ids[:n]

    existing = load_existing_keys()
    print(f"already on disk: {len(existing)} records")

    writers = {
        "paraphrase": IncrementalWriter(OUT_DIR / "paraphrase.jsonl"),
        "style": IncrementalWriter(OUT_DIR / "style.jsonl"),
    }

    print(f"selected {len(chosen)} / {total} functions (seed={seed}, temp={temp})")
    print(f"models: {MODELS}")
    print()

    n_new = 0
    n_skipped = 0

    try:
        for i, iid in enumerate(chosen):
            rec = pick_canonical(by_input[iid])
            text = get_flow_text(rec)
            rec_temp = get_rec_temp(rec)
            flow_id = f"{iid}_t{rec_temp}"

            print(f"[{i+1}/{len(chosen)}] {flow_id}  (text {len(text)} chars)")

            tasks = []
            for j, model_name in enumerate(MODELS):
                # Paraphrase: one per model
                tasks.append(
                    gen_paraphrase(flow_id, text, model_name,
                                   writers["paraphrase"], existing)
                )
                # Style: one per model, alternating formal/informal
                style = "formal" if j % 2 == 0 else "informal"
                tasks.append(
                    gen_style(flow_id, text, model_name, style,
                              writers["style"], existing)
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            ok = fail = skip = 0
            for r in results:
                if isinstance(r, Exception):
                    fail += 1
                    print(f"  FAIL: {r}", file=sys.stderr)
                elif r is None:
                    skip += 1
                else:
                    ok += 1

            n_new += ok
            n_skipped += skip
            print(f"  ok={ok} skip={skip} fail={fail}")
    finally:
        for w in writers.values():
            w.close()

    meta = {
        "n_functions_requested": n_arg,
        "n_functions_run": len(chosen),
        "n_functions_total": total,
        "seed": seed,
        "temperature": temp,
        "gen_temperature": GEN_TEMP,
        "models": MODELS,
        "input_ids": chosen,
        "n_new_records": n_new,
        "n_skipped_records": n_skipped,
    }
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"new records:       {n_new}")
    print(f"skipped (on disk): {n_skipped}")
    print(f"wrote {OUT_DIR}/paraphrase.jsonl")
    print(f"wrote {OUT_DIR}/style.jsonl")
    print(f"wrote {OUT_DIR}/meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="10",
                    help="number of functions: integer, 'none', or 'all' (default: 10)")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asyncio.run(main(args.n, args.temp, args.seed))