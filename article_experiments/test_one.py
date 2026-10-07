#!/usr/bin/env python3
"""Smoke test: run transformation generation on a single function.

Takes the first explanation from v0_outputs.jsonl and generates:
  - 1 paraphrase
  - 1 style shift
  - 1 code variation

Prints everything to stdout. No files written.

Usage:
    export OPENROUTER_API_KEY=...
    python test_one.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

from transforms_prompts import (
    PARAPHRASE_PROMPT,
    STYLE_FORMAL_PROMPT,
    CODE_VARIATION_PROMPT,
    EXPLAIN_PROMPT,
)


from dotenv import load_dotenv


# --- Paths -------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
FLOW_DIR = PROJECT_ROOT / "flow_harness"
INPUTS_PATH = FLOW_DIR / "fixed_200_inputs.jsonl"
OUTPUTS_PATH = FLOW_DIR / "v0_outputs.jsonl"


# --- Env ---------------------------------------------------------------------

ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(f"[warn] no .env at {ENV_PATH}, using shell environment only")


# --- Config ------------------------------------------------------------------

MODEL = os.environ.get("TRANSFORM_MODEL", "z-ai/glm-4.6")
EXPLAIN_MODEL = os.environ.get("EXPLAIN_MODEL", MODEL)


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("ERROR: OPENROUTER_API_KEY is not set or empty.", file=sys.stderr)
        print("Check with:  echo $OPENROUTER_API_KEY", file=sys.stderr)
        print("Set with:    export OPENROUTER_API_KEY=sk-or-...", file=sys.stderr)
        sys.exit(1)
    print(f"OPENROUTER_API_KEY loaded: {len(key)} chars, prefix {key[:8]}...")
    return key


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=get_api_key(),
    )


def load_first_input() -> dict:
    with INPUTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise SystemExit("no inputs found")


def load_first_output() -> dict:
    with OUTPUTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise SystemExit("no outputs found")


def get_flow_text(rec: dict) -> str:
    return (rec.get("flow_text") or rec.get("explanation") or "").strip()


_FENCE_RE = re.compile(r"```(?:c|C)?\s*\n(.*?)\n```", re.DOTALL)


def extract_code_block(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def show(title: str, text: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(text)
    print()


async def llm(client: AsyncOpenAI, model: str, prompt: str, temperature: float = 0.7) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()


async def main() -> None:
    client = get_client()

    inp = load_first_input()
    out = load_first_output()

    code = inp["code"]
    explanation = get_flow_text(out)

    print()
    print(f"input_id:    {inp['id']}")
    print(f"flow_id:     {out.get('input_id')}_t{out.get('temperature')}")
    print(f"code length: {len(code)} chars")
    print(f"flow length: {len(explanation)} chars")
    print()

    show("ORIGINAL CODE", code)
    show("ORIGINAL EXPLANATION", explanation)

    print("Generating paraphrase...")
    para = await llm(client, MODEL, PARAPHRASE_PROMPT.format(explanation=explanation), temperature=0.7)
    show("PARAPHRASE", para)

    print("Generating formal style shift...")
    style = await llm(client, MODEL, STYLE_FORMAL_PROMPT.format(explanation=explanation), temperature=0.7)
    show("STYLE SHIFT (formal)", style)

    print("Generating code variation...")
    modified_raw = await llm(client, MODEL, CODE_VARIATION_PROMPT.format(code=code), temperature=0.7)
    modified = extract_code_block(modified_raw)
    show("MODIFIED CODE", modified)

    print("Generating new explanation from modified code...")
    new_explanation = await llm(client, EXPLAIN_MODEL, EXPLAIN_PROMPT.format(code=modified), temperature=0.0)
    show("NEW EXPLANATION (from modified code)", new_explanation)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())