#!/usr/bin/env python3
"""End-to-end environment smoke test for the my_training_path SFT/LoRA project.

This is an *environment validation* harness (not part of the training product).
It proves the Cloud Agent environment can run the project's core stack without a
GPU by:

  1. Importing every third-party dependency the codebase relies on.
  2. Running the repository's own data pipeline code (Config + DataLoader +
     Metrics) against the real prompt/completion dataset shipped in the repo.
  3. Executing a genuine LoRA SFT fine-tune on CPU with a tiny model using the
     same trl/peft/transformers/datasets stack the pipeline uses, on real rows
     from the project's dataset.

Run: .venv/bin/python scripts/env_smoke_test.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPO = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO / "sft_lora_pipeline"
DATASET = REPO / "smoke_test_lora_sft" / "sft_data_prompt_completion.jsonl"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def step1_imports() -> None:
    section("STEP 1 — Import third-party stack")
    import datasets
    import numpy
    import peft
    import torch
    import transformers
    import trl
    import yaml  # noqa: F401
    import tqdm  # noqa: F401
    import openai  # noqa: F401
    import requests  # noqa: F401
    import json_repair  # noqa: F401

    print(f"torch        {torch.__version__} (cuda={torch.cuda.is_available()})")
    print(f"transformers {transformers.__version__}")
    print(f"trl          {trl.__version__}")
    print(f"peft         {peft.__version__}")
    print(f"datasets     {datasets.__version__}")
    print(f"numpy        {numpy.__version__}")
    print("STEP 1 PASS")


def step2_repo_data_pipeline():
    section("STEP 2 — Run repository data pipeline on real dataset")
    # Import the repo's own modules exactly like `python main.py` does.
    sys.path.insert(0, str(PIPELINE_DIR))
    from pipeline.config import Config
    from pipeline.data_loader import DataLoader
    from pipeline.metrics import Metrics

    cfg = Config.from_yaml(str(PIPELINE_DIR / "config.yaml"))
    # Point the loader at the in-repo dataset and cap size for a fast check.
    cfg.data_path = str(DATASET)
    cfg.data_limit = 64

    loader = DataLoader(cfg, tokenizer=None)
    records = loader.load_raw(cfg.data_path)
    assert records, "no records loaded"
    print(f"Loaded {len(records)} records from {DATASET.name}")

    # Exercise dedup + split (unmodified repo logic).
    deduped = loader.drop_content_overlap(records, records[:5])
    print(f"drop_content_overlap: {len(records)} -> {len(deduped)} (removed 5 known dups)")
    train, val, test = loader.split_data(records)
    print(f"split_data: train={len(train)} val={len(val)} test={len(test)}")

    train_ds, val_ds, _ = loader.prepare_datasets(train, val, [])
    print(f"prepare_datasets: train_ds={train_ds} val_ds={val_ds}")

    # Exercise the JSON extraction metric on a real completion.
    sample_completion = records[0]["completion"]
    extracted = Metrics.extract_json(sample_completion)
    valid = Metrics.compute_json_validity(sample_completion)
    assert valid, "expected valid findings JSON in dataset completion"
    print(f"Metrics.extract_json -> valid findings JSON: {extracted[:80]}...")
    print("STEP 2 PASS")
    return train, val


def step3_cpu_lora_sft(train_records, val_records) -> None:
    section("STEP 3 — Real LoRA SFT fine-tune on CPU (tiny model)")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    print(f"Loading tiny model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

    # Use the same prompt/completion schema and LoRA target modules the project uses.
    n = 16
    train_ds = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in train_records[:n]]
    )
    eval_ds = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in val_records[:4]]
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    out_dir = str(REPO / "smoke_out")
    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        max_steps=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        bf16=False,
        fp16=False,
        max_length=256,
        packing=False,
        completion_only_loss=True,
        remove_unused_columns=False,
        seed=42,
        use_cpu=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Confirm completion-only masking works, exactly like the repo's trainer does.
    batch = next(iter(trainer.get_train_dataloader()))
    mask_ratio = (batch["labels"] == -100).float().mean().item()
    print(f"Masked (prompt) token ratio: {mask_ratio:.2%}")
    assert mask_ratio > 0.0, "completion_only_loss masking not working"

    result = trainer.train()
    loss = result.metrics.get("train_loss")
    print(f"Training finished: steps={result.global_step}, train_loss={loss:.4f}")

    adapter = Path(out_dir) / "adapter"
    trainer.model.save_pretrained(str(adapter))
    saved = list(adapter.glob("adapter_model*"))
    assert saved, "LoRA adapter was not saved"
    print(f"Saved LoRA adapter -> {[p.name for p in saved]}")
    print("STEP 3 PASS")


def main() -> int:
    step1_imports()
    train, val = step2_repo_data_pipeline()
    step3_cpu_lora_sft(train, val)
    section("ALL STEPS PASSED — environment is functional end-to-end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
