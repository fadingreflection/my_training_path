"""
scripts/check_backend.py — smoke HF-бэкенда на N примерах из val_eval.jsonl.

Запуск ИЗ КОРНЯ inference_backend/:
    python3 -m scripts.check_backend --model qwen_base --n 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging

from transformers import AutoTokenizer

from backends import get_backend
from core.loader import load_model, load_prompt, load_task
from core.paths import CONFIGS_DIR, PROJECT_ROOT
from data.normalizer import Normalizer
from prompts.builder import PromptBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def make_chat_fn(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    def chat_fn(messages, add_generation_prompt):
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        try:
            return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tok.apply_chat_template(messages, **kwargs)
    return chat_fn, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen_base")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    task = load_task(CONFIGS_DIR / "tasks/cwe_extraction.yaml")
    prompt_spec = load_prompt(CONFIGS_DIR / "prompts/cwe_v1.yaml")
    model_spec = load_model(CONFIGS_DIR / "models/registry.yaml", name=args.model)

    print(f"Model: {model_spec.name} (backend={model_spec.backend}, path={model_spec.path})")

    norm = Normalizer(task)
    records = norm.load_jsonl(PROJECT_ROOT.parent.parent / "eda/val_eval.jsonl")
    examples, rejected = norm.normalize(records)
    print(f"Normalized: kept={len(examples)}, rejected={len(rejected)}")
    examples = examples[:args.n]

    chat_fn, tok = make_chat_fn(model_spec.path)
    builder = PromptBuilder(prompt_spec, chat_template_fn=chat_fn)
    requests = builder.build_batch(examples)
    print(f"Built {len(requests)} requests\n")

    backend = get_backend(model_spec, tokenizer=tok)
    responses = backend.generate_batch(requests)
    backend.close()

    for i, (req, resp) in enumerate(zip(requests, responses)):
        print(f"=== [{i}] {resp.request_id} ===")
        print(f"  sample_id:     {resp.sample_id}")
        print(f"  backend/model: {resp.backend}/{resp.model}")
        print(f"  latency_ms:    {resp.latency_ms:.1f}")
        print(f"  tokens_in/out: {resp.tokens_in}/{resp.tokens_out}")
        print(f"  finish_reason: {resp.finish_reason}")
        print(f"  error:         {resp.error}")
        print(f"  raw_text[:400]: {resp.raw_text[:400]!r}")
        print()


if __name__ == "__main__":
    main()