#!/usr/bin/env python3
"""Merge a LoRA adapter into base weights and save to a NEW directory.

Does NOT modify base model or adapter paths.

  .venv/bin/python sft_lora_pipeline/merge_adapter.py \\
      --base models/base \\
      --adapter sft_lora_pipeline/output_explain_cwe_qwen17b/checkpoint-204 \\
      --out models/qwen17b_explain_cwe_lora_merged
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def restore_legacy_config_keys(base: Path, out: Path) -> list[str]:
    """Re-add top-level config keys that newer transformers drops on save.

    transformers >= 5 writes `rope_parameters`/`dtype` and omits the classic
    `rope_theta`/`torch_dtype`. Older transformers 4.x readers may ignore
    `rope_parameters` and fall back to the Qwen3 default rope_theta=10000
    instead of 1e6, which corrupts positions. Merging never changes
    architecture, so any key the base declares and the merged copy lacks is
    safe to copy back.
    """
    base_cfg_path, out_cfg_path = base / "config.json", out / "config.json"
    if not (base_cfg_path.exists() and out_cfg_path.exists()):
        return []
    base_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    out_cfg = json.loads(out_cfg_path.read_text(encoding="utf-8"))
    added = [k for k in base_cfg if k not in out_cfg and k != "transformers_version"]
    if not added:
        return []
    for key in added:
        out_cfg[key] = base_cfg[key]
    out_cfg_path.write_text(json.dumps(out_cfg, indent=2) + "\n", encoding="utf-8")
    print(f"config: restored legacy keys {sorted(added)}", flush=True)
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda for merge")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base = resolve(args.base, root)
    adapter = resolve(args.adapter, root)
    out = resolve(args.out, root)

    if not base.exists():
        raise SystemExit(f"base missing: {base}")
    if not adapter.exists():
        raise SystemExit(f"adapter missing: {adapter}")
    if out == base or base in out.parents or out in base.parents:
        raise SystemExit("refusing to write into base tree")
    if out == adapter or adapter in out.parents or out in adapter.parents:
        raise SystemExit("refusing to write into adapter tree")

    restore_legacy_config_keys(base, out)

    meta_path = out / "merge_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (out / "config.json").exists() and meta.get("status") == "ok":
            print(json.dumps(meta, indent=2))
            return

    out.mkdir(parents=True, exist_ok=True)
    print(f"merge base={base} adapter={adapter} -> out={out} device={args.device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    dtype = torch.bfloat16 if args.device != "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        dtype=dtype,
        device_map=args.device if args.device != "cpu" else None,
        trust_remote_code=True,
    )
    if args.device == "cpu":
        model = model.to("cpu")

    peft_model = PeftModel.from_pretrained(model, str(adapter))
    merged = peft_model.merge_and_unload()

    merged.save_pretrained(str(out), safe_serialization=True)
    restore_legacy_config_keys(base, out)
    # Copy full tokenizer from base (save_pretrained drops added_tokens_decoder).
    for name in (
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ):
        src = base / name
        if src.exists():
            shutil.copy2(src, out / name)
    for name in ("chat_template.jinja",):
        src = adapter / name
        if src.exists():
            shutil.copy2(src, out / name)
        elif (base / name).exists():
            shutil.copy2(base / name, out / name)

    meta = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": str(base),
        "adapter": str(adapter),
        "out": str(out),
        "note": "Merged copy only; base and adapter dirs unchanged.",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
