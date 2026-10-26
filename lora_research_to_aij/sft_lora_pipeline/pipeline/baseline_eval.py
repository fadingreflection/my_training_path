"""
baseline_eval.py — оценка базовой модели на val/holdout.

Запуск из корня проекта:
    python3 baseline_eval.py --config config.yaml \
        --data ../eda/val_sft.jsonl --out baseline_metrics.json
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pipeline.config import Config
from pipeline.evaluator import Evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="baseline_metrics.json")
    ap.add_argument("--sample_size", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)

    tok = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    use_4bit = getattr(cfg, "load_in_4bit", False)
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    records = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    print(f"Evaluating on {len(records)} records from {args.data}")

    ev = Evaluator(cfg, model, tok)
    metrics = ev.evaluate_and_log(
        records=records,
        sample_size=args.sample_size,
        log_file="baseline_predictions.jsonl",
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()