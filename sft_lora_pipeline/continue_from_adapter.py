#!/usr/bin/env python3
"""Continue LoRA SFT from an existing adapter into a NEW output_dir.

Does not write into init_adapter. Loads base + PeftModel, trains on new data,
saves only under config.output_dir.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from pipeline.config import Config
from pipeline.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/continue_from_adapter.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("continue_from_adapter")


def resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


class JsonProgressCallback(TrainerCallback):
    """Write a small JSON snapshot on every trainer log/eval."""

    def __init__(self, path: Path):
        self.path = path
        self.history: list[dict] = []

    def _dump(self, state, logs=None) -> None:
        rec = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "best_metric": state.best_metric,
            "best_global_step": state.best_global_step,
            "logs": dict(logs or {}),
        }
        self.history.append(rec)
        payload = {"latest": rec, "history": self.history[-50:]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
        self._dump(state, logs)
        log.info("[PROGRESS] step=%s logs=%s", state.global_step, logs)


def main() -> None:
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "config_explain_cwe_ds_rescue.yaml")
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg_dir = cfg_path.resolve().parent
    os.makedirs("logs", exist_ok=True)

    init_adapter = resolve(raw["init_adapter"], cfg_dir)
    output_dir = resolve(raw["output_dir"], cfg_dir)
    data_path = resolve(raw["data_path"], cfg_dir)
    val_path = resolve(raw["val_data_path"], cfg_dir)
    base_model = resolve(raw["model_name_or_path"], cfg_dir)

    def _is_same_or_inside(child: Path, parent: Path) -> bool:
        child = child.resolve()
        parent = parent.resolve()
        if child == parent:
            return True
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    if _is_same_or_inside(output_dir, init_adapter):
        raise SystemExit("refusing to write into init_adapter; set a separate output_dir")

    if not init_adapter.exists():
        raise SystemExit(f"init_adapter missing: {init_adapter}")
    if not data_path.exists():
        raise SystemExit(f"data_path missing: {data_path}")

    # Reuse Config dataclass for DataLoader / SFT hyperparams.
    cfg = Config.from_yaml(str(cfg_path))
    cfg.output_dir = str(output_dir)
    cfg.data_path = str(data_path)
    cfg.val_data_path = str(val_path)
    cfg.model_name_or_path = str(base_model)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")

    log.info("base=%s init_adapter=%s out=%s", base_model, init_adapter, output_dir)
    log.info("train=%s val=%s", data_path, val_path)

    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    log.info("loading existing adapter (read-only source) %s", init_adapter)
    model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)

    loader = DataLoader(cfg, tokenizer)
    train_records = loader.load_raw(str(data_path))
    val_records = loader.load_raw(str(val_path)) if val_path.exists() else []
    if getattr(cfg, "apply_chat_template", False):
        train_records = loader.apply_chat_prompts(train_records)
        if val_records:
            val_records = loader.apply_chat_prompts(val_records)

    train_ds, val_ds, _ = loader.prepare_datasets(train_records, val_records or None, None)
    log.info("train_rows=%d val_rows=%d", len(train_records), len(val_records))

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=float(cfg.learning_rate),
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=[cfg.report_to] if cfg.use_tensorboard else [],
        bf16=True,
        fp16=False,
        eval_strategy="steps" if val_ds is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        remove_unused_columns=False,
        seed=cfg.seed,
        completion_only_loss=True,
        dataset_text_field=None,
        max_length=cfg.max_seq_length,
        packing=False,
        loss_type="nll",
        gradient_checkpointing=True,
    )

    callbacks = [JsonProgressCallback(output_dir / "train_progress.json")]
    if val_ds is not None and getattr(cfg, "early_stopping_patience", 0) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=getattr(cfg, "early_stopping_threshold", 0.0),
            )
        )

    # peft_config=None: continue the loaded adapter; do not re-init LoRA.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    t0 = time.time()
    log.info("[TRAIN] continue-from-adapter start")
    trainer.train()
    elapsed = time.time() - t0
    log.info("[TIME] done in %.1fs", elapsed)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    meta = {
        "init_adapter": str(init_adapter),
        "output_dir": str(output_dir),
        "train": str(data_path),
        "val": str(val_path),
        "elapsed_s": elapsed,
        "baseline_untouched": True,
    }
    (output_dir / "continue_meta.json").write_text(
        __import__("json").dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    log.info("[SAVE] wrote %s", output_dir)


if __name__ == "__main__":
    main()
