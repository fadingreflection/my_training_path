#!/usr/bin/env python3
"""Local two-step eval: explanation → CWE chain, score any_level_hit.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python explain_cwe_map/eval_local.py
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python explain_cwe_map/eval_local.py \\
      --adapter sft_lora_pipeline/output_explain_cwe_qwen17b \\
      --tag qwen17b_lora_holdout
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain_cwe_map"))

from make_sft import load_pool, summarize  # noqa: E402
from match import score  # noqa: E402
from prompts import (  # noqa: E402
    MAP_CWE,
    is_insufficient_verdict,
    is_safe_verdict,
    review_user_message,
    skip_cwe_mapping,
)


def load_id_order(path: Path) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rid = json.loads(line)["id"]
            if rid not in seen:
                seen.add(rid)
                order.append(rid)
    return order


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error"):
                done.discard(rec["id"])
            else:
                done.add(rec["id"])
    return done

log = logging.getLogger("eval_local")
OUT_DIR = ROOT / "explain_cwe_map" / "runs"
DEFAULT_MODEL = ROOT / "models" / "base"
DEFAULT_HOLDOUT = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits" / "holdout_ids.jsonl"
DEFAULT_DATA = ROOT / "raw_data" / "bigvul_func_before_balanced14.jsonl"


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def strip_think(text: str) -> str:
    t = (text or "").strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    while t.startswith("<think>"):
        t = t[len("<think>"):].strip()
    return t.strip()


def apply_chat(tokenizer, messages: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    return tokenizer.apply_chat_template(messages, **kwargs)


def _pad_id(tokenizer) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    return pad_id


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    messages: list[dict],
    *,
    max_new_tokens: int,
    max_input: int,
) -> str:
    device = next(model.parameters()).device
    prompt = apply_chat(tokenizer, messages)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = _pad_id(tokenizer)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][prompt_len:]
    return strip_think(tokenizer.decode(gen, skip_special_tokens=True))


def pack_row(
    rec: dict,
    *,
    explanation: str,
    mapping: str,
    error: str,
    elapsed: float,
    model_name: str,
) -> dict:
    scored = score(
        rec["cwe"],
        mapping,
        said_safe=is_safe_verdict(explanation),
        said_insufficient=is_insufficient_verdict(explanation),
    )
    if error:
        scored["error"] = True
    return {
        "id": rec["id"],
        "cve": rec.get("cve") or "",
        "project": rec.get("project") or "",
        "commit_id": rec.get("commit_id") or "",
        "gold_cwe": rec["cwe"],
        "model": model_name,
        "explanation": explanation,
        "mapping": mapping,
        "error": error,
        "score": scored,
        "elapsed": elapsed,
    }


def process_sample(
    model,
    tokenizer,
    rec: dict,
    *,
    max_new_tokens: int,
    max_input: int,
    index: int,
    n: int,
    model_name: str,
) -> dict:
    t0 = time.monotonic()
    log.info(
        "[%d/%d] start id=%s gold=%s project=%s code_chars=%d",
        index,
        n,
        rec["id"],
        rec["cwe"],
        rec.get("project") or "",
        len(rec["func_before"]),
    )
    messages = [{"role": "user", "content": review_user_message(rec["func_before"])}]
    explanation = mapping = error = ""
    try:
        explanation = generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            max_input=max_input,
        )
        safe = is_safe_verdict(explanation)
        insufficient = is_insufficient_verdict(explanation)
        skip = skip_cwe_mapping(explanation)
        log.info(
            "[%d/%d] review chars=%d safe=%s insufficient=%s skip_map=%s",
            index,
            n,
            len(explanation),
            safe,
            insufficient,
            skip,
        )
        if not skip:
            messages.append({"role": "assistant", "content": explanation})
            messages.append({"role": "user", "content": MAP_CWE})
            mapping = generate(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                max_input=max_input,
            )
            log.info("[%d/%d] map chars=%d", index, n, len(mapping))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.error("[%d/%d] failed: %s", index, n, error)

    row = pack_row(
        rec,
        explanation=explanation,
        mapping=mapping,
        error=error,
        elapsed=time.monotonic() - t0,
        model_name=model_name,
    )
    log.info(
        "[%d/%d] wrote id=%s err=%s hit=%s elapsed=%.1fs",
        index,
        n,
        rec["id"],
        bool(error),
        bool(row["score"].get("any_level_hit")),
        row["elapsed"],
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA adapter dir (adapter_config.json + weights). Omit for base model.",
    )
    ap.add_argument("--only-from", type=Path, default=DEFAULT_HOLDOUT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--tag", default="qwen17b_base_holdout")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--max-input", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_path = OUT_DIR / f"{args.tag}.jsonl"
    summary_path = OUT_DIR / f"{args.tag}_summary.json"
    log_path = OUT_DIR / f"{args.tag}.log"
    setup_logging(log_path)

    pool = load_pool(args.data)
    order = load_id_order(args.only_from)
    missing = [i for i in order if i not in pool]
    if missing:
        raise SystemExit(f"holdout ids not in pool: {missing[:8]}")
    rows = [pool[i] for i in order]
    if args.limit > 0:
        rows = rows[: args.limit]

    done = load_done(out_path)
    pending = [r for r in rows if r["id"] not in done]
    log.info(
        "model=%s cuda=%s gpu=%s holdout=%d already_ok=%d todo=%d out=%s",
        args.model,
        torch.cuda.is_available(),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        len(rows),
        len(done),
        len(pending),
        out_path,
    )
    print(
        f"model={args.model} holdout={len(rows)} already_ok={len(done)} "
        f"todo={len(pending)} out={out_path}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_name = "qwen3-1.7b-base"
    if args.adapter:
        from peft import PeftModel

        adapter = args.adapter.resolve()
        if not adapter.exists():
            raise SystemExit(f"adapter not found: {adapter}")
        log.info("loading adapter %s", adapter)
        model = PeftModel.from_pretrained(model, str(adapter))
        model_name = "qwen3-1.7b-lora"
    model.eval()
    model.config.pad_token_id = _pad_id(tokenizer)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = _pad_id(tokenizer)
    log.info(
        "loaded device=%s adapter=%s",
        next(model.parameters()).device,
        args.adapter,
    )

    results = []
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    results.append(json.loads(line))

    n = len(rows)
    want = {r["id"] for r in rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for rec in tqdm(pending, desc=args.tag):
            index = order.index(rec["id"]) + 1 if rec["id"] in order else len(results) + 1
            row = process_sample(
                model,
                tokenizer,
                rec,
                max_new_tokens=args.max_new_tokens,
                max_input=args.max_input,
                index=index,
                n=n,
                model_name=model_name,
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(row)
            scored_n = sum(1 for r in results if r.get("id") in want)
            if scored_n % 10 == 0:
                live = summarize([r for r in results if r.get("id") in want])
                log.info("progress n=%d metrics=%s", scored_n, json.dumps(live))

    sample = [r for r in results if r.get("id") in want]
    by_id = {}
    for rec in sample:
        by_id[rec["id"]] = rec
    sample = list(by_id.values())
    summary = {
        "model": model_name,
        "model_path": str(args.model),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "tag": args.tag,
        "n_requested": len(rows),
        "n_scored": len(sample),
        "metrics": summarize(sample),
        "out": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("summary %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
