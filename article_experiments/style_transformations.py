# #!/usr/bin/env python3
# """Generate meaning-preserving transformations for FSS computation.

# For each explanation in ../flow_harness/v0_outputs.jsonl:
#   - 5 paraphrases (text -> text)
#   - 5 style shifts (text -> text; alternating formal/informal)
#   - 5 code variations (code -> modified code -> new explanation)

# Outputs JSONL files in transforms/.

# Usage:
#     export OPENROUTER_API_KEY=...
#     python generate_transformations.py paraphrase
#     python generate_transformations.py style
#     python generate_transformations.py code
#     python generate_transformations.py all

# Prompts are defined in transforms_prompts.py.
# """

# from __future__ import annotations

# import asyncio
# import hashlib
# import json
# import os
# import re
# import sys
# from pathlib import Path

# from openai import AsyncOpenAI

# from transforms_prompts import (
#     PARAPHRASE_PROMPT,
#     STYLE_FORMAL_PROMPT,
#     STYLE_INFORMAL_PROMPT,
#     CODE_VARIATION_PROMPT,
#     EXPLAIN_PROMPT,
# )


# # --- Config -----------------------------------------------------------------

# HERE = Path(__file__).resolve().parent           # article_experiments/
# PROJECT_ROOT = HERE.parent                        # parent of article_experiments/
# FLOW_DIR = PROJECT_ROOT / "flow_harness"          # sibling of article_experiments/
# INPUTS_PATH = FLOW_DIR / "fixed_200_inputs.jsonl"
# OUTPUTS_PATH = FLOW_DIR / "v0_outputs.jsonl"
# TRANSFORMS_DIR = HERE / "transforms"
# CACHE_DIR = TRANSFORMS_DIR / ".cache"

# MODEL = os.environ.get("TRANSFORM_MODEL", "z-ai/glm-4.6")
# EXPLAIN_MODEL = os.environ.get("EXPLAIN_MODEL", MODEL)
# CONCURRENCY = int(os.environ.get("TRANSFORM_CONCURRENCY", "8"))
# MAX_RETRIES = 4
# N_PER_CLASS = 5
# BASE_TEMP = 0.7
# TEMP_STEP = 0.05


# # --- I/O ---------------------------------------------------------------------

# def load_inputs() -> dict[str, dict]:
#     inputs: dict[str, dict] = {}
#     with INPUTS_PATH.open(encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 r = json.loads(line)
#                 inputs[r["id"]] = r
#     return inputs


# def load_outputs() -> list[dict]:
#     rows: list[dict] = []
#     with OUTPUTS_PATH.open(encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 rows.append(json.loads(line))
#     return rows


# def get_flow_id(rec: dict) -> str:
#     iid = rec.get("input_id") or rec.get("id")
#     t = rec.get("temperature") or rec.get("temp") or "0"
#     return f"{iid}_t{t}"


# def get_flow_text(rec: dict) -> str:
#     return (rec.get("flow_text") or rec.get("explanation") or "").strip()


# def write_jsonl(path: Path, rows: list[dict]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")


# # --- Cache -------------------------------------------------------------------

# CACHE_DIR.mkdir(parents=True, exist_ok=True)


# def _cache_path(model: str, prompt: str) -> Path:
#     h = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:32]
#     return CACHE_DIR / f"{h}.txt"


# def cache_get(model: str, prompt: str) -> str | None:
#     p = _cache_path(model, prompt)
#     return p.read_text(encoding="utf-8") if p.exists() else None


# def cache_put(model: str, prompt: str, value: str) -> None:
#     _cache_path(model, prompt).write_text(value, encoding="utf-8")


# # --- LLM client --------------------------------------------------------------

# client = AsyncOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.environ.get("OPENROUTER_API_KEY", ""),
# )
# sem = asyncio.Semaphore(CONCURRENCY)


# async def llm(model: str, prompt: str, temperature: float = 0.7) -> str:
#     cached = cache_get(model, prompt)
#     if cached is not None:
#         return cached
#     async with sem:
#         last_err: Exception | None = None
#         for attempt in range(MAX_RETRIES):
#             try:
#                 resp = await client.chat.completions.create(
#                     model=model,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=temperature,
#                     max_tokens=2048,
#                 )
#                 text = (resp.choices[0].message.content or "").strip()
#                 if text:
#                     cache_put(model, prompt, text)
#                     return text
#                 last_err = RuntimeError("empty response")
#             except Exception as e:  # noqa: BLE001
#                 last_err = e
#             await asyncio.sleep(2 ** attempt)
#         raise RuntimeError(f"llm failed after {MAX_RETRIES} retries: {last_err}")


# # --- Workers -----------------------------------------------------------------

# async def gen_paraphrase(rec: dict, idx: int) -> dict | None:
#     text = get_flow_text(rec)
#     if not text:
#         return None
#     prompt = PARAPHRASE_PROMPT.format(explanation=text)
#     out = await llm(MODEL, prompt, temperature=BASE_TEMP + TEMP_STEP * idx)
#     if not out:
#         return None
#     return {
#         "flow_id": get_flow_id(rec),
#         "class": "paraphrase",
#         "index": idx,
#         "source_text": text,
#         "transformed_text": out,
#     }


# async def gen_style(rec: dict, idx: int) -> dict | None:
#     text = get_flow_text(rec)
#     if not text:
#         return None
#     formal = idx % 2 == 0
#     template = STYLE_FORMAL_PROMPT if formal else STYLE_INFORMAL_PROMPT
#     prompt = template.format(explanation=text)
#     out = await llm(MODEL, prompt, temperature=BASE_TEMP + TEMP_STEP * idx)
#     if not out:
#         return None
#     return {
#         "flow_id": get_flow_id(rec),
#         "class": "style",
#         "index": idx,
#         "style": "formal" if formal else "informal",
#         "source_text": text,
#         "transformed_text": out,
#     }


# _FENCE_RE = re.compile(r"```(?:c|C)?\s*\n(.*?)\n```", re.DOTALL)


# def extract_code_block(text: str) -> str:
#     m = _FENCE_RE.search(text)
#     if m:
#         return m.group(1).strip()
#     return text.strip()


# async def gen_code_variation(inp: dict, idx: int) -> dict | None:
#     code = (inp.get("code") or "").strip()
#     if not code:
#         return None
#     prompt = CODE_VARIATION_PROMPT.format(code=code)
#     modified_raw = await llm(MODEL, prompt, temperature=BASE_TEMP + TEMP_STEP * idx)
#     modified = extract_code_block(modified_raw)
#     if not modified or modified == code:
#         return None
#     explain_prompt = EXPLAIN_PROMPT.format(code=modified)
#     explanation = await llm(EXPLAIN_MODEL, explain_prompt, temperature=0.0)
#     if not explanation:
#         return None
#     return {
#         "input_id": inp["id"],
#         "class": "code_variation",
#         "index": idx,
#         "source_code": code,
#         "modified_code": modified,
#         "generated_explanation": explanation,
#     }


# # --- Drivers -----------------------------------------------------------------

# async def run_paraphrase() -> None:
#     outputs = load_outputs()
#     tasks = [gen_paraphrase(rec, i) for rec in outputs for i in range(N_PER_CLASS)]
#     results = await asyncio.gather(*tasks, return_exceptions=True)
#     rows = [r for r in results if isinstance(r, dict)]
#     write_jsonl(TRANSFORMS_DIR / "paraphrase.jsonl", rows)
#     print(f"paraphrase: {len(rows)} rows -> transforms/paraphrase.jsonl")


# async def run_style() -> None:
#     outputs = load_outputs()
#     tasks = [gen_style(rec, i) for rec in outputs for i in range(N_PER_CLASS)]
#     results = await asyncio.gather(*tasks, return_exceptions=True)
#     rows = [r for r in results if isinstance(r, dict)]
#     write_jsonl(TRANSFORMS_DIR / "style.jsonl", rows)
#     print(f"style: {len(rows)} rows -> transforms/style.jsonl")


# async def run_code() -> None:
#     inputs = load_inputs()
#     tasks = [gen_code_variation(inp, i) for inp in inputs.values() for i in range(N_PER_CLASS)]
#     results = await asyncio.gather(*tasks, return_exceptions=True)
#     rows = [r for r in results if isinstance(r, dict)]
#     write_jsonl(TRANSFORMS_DIR / "code_variation.jsonl", rows)
#     print(f"code_variation: {len(rows)} rows -> transforms/code_variation.jsonl")


# async def main(cmd: str) -> None:
#     if not os.environ.get("OPENROUTER_API_KEY"):
#         raise SystemExit("OPENROUTER_API_KEY not set")
#     TRANSFORMS_DIR.mkdir(parents=True, exist_ok=True)
#     if cmd in ("paraphrase", "all"):
#         await run_paraphrase()
#     if cmd in ("style", "all"):
#         await run_style()
#     if cmd in ("code", "all"):
#         await run_code()


# if __name__ == "__main__":
#     cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
#     if cmd not in {"paraphrase", "style", "code", "all"}:
#         raise SystemExit("usage: generate_transformations.py [paraphrase|style|code|all]")
#     asyncio.run(main(cmd))