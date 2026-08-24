import logging
import os
import time

from peft import LoraConfig
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


class SFTTrainerWrapper:
    def __init__(self, config, model, tokenizer, train_dataset, val_dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self):
        logger.info("[TRAIN] Training started")
        start_time = time.time()

        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Lora configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Training arguments
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=float(self.config.learning_rate),
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to=[self.config.report_to] if self.config.use_tensorboard else [],
            bf16=True,
            fp16=False,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            remove_unused_columns=False,
            seed=self.config.seed,
            completion_only_loss=True,
            dataset_text_field=None,
            max_length=self.config.max_seq_length,
            packing=False,
            loss_type="nll",
        )

        # Early stopping callback
        callbacks = []
        if getattr(self.config, 'early_stopping_patience', 0) > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=getattr(self.config, 'early_stopping_threshold', 0.0)
                )
            )
            logger.info(
                f"[EARLYSTOP] Early stopping enabled: patience={self.config.early_stopping_patience}, "
                f"threshold={self.config.early_stopping_threshold}"
            )

        # Trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            peft_config=lora_config,
            callbacks=callbacks,
        )

        # Masking debug check
        logger.info(f"processing_class is tokenizer: {trainer.processing_class is self.tokenizer}")
        logger.info(f"completion_only_loss: {training_args.completion_only_loss}")

        try:
            batch = next(iter(trainer.get_train_dataloader()))
            mask_ratio = (batch['labels'] == -100).float().mean().item()
            logger.info(f"Masked token ratio: {mask_ratio:.2%}")
            if mask_ratio == 0.0:
                logger.warning(
                    "[WARN] Masking is not working. Check completion_only_loss and processing_class."
                )
            else:
                logger.info("[OK] Masking works.")
        except Exception as e:
            logger.warning(f"Could not check masking: {e}")

        # Resume from checkpoint
        resume_from_checkpoint = None
        resume_setting = getattr(self.config, 'resume_from_checkpoint', True)

        if resume_setting:
            if isinstance(resume_setting, str) and os.path.exists(resume_setting):
                resume_from_checkpoint = resume_setting
                logger.info(f"[RESUME] Resuming from specified checkpoint: {resume_from_checkpoint}")
            elif os.path.exists(output_dir):
                checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
                    resume_from_checkpoint = os.path.join(output_dir, latest)
                    logger.info(f"[RESUME] Resuming from latest checkpoint: {resume_from_checkpoint}")
                else:
                    logger.info("[OK] No checkpoints found, starting from scratch.")
            else:
                logger.info("[OK] Output directory empty, starting from scratch.")
        else:
            logger.info("[OK] Training from scratch (resume_from_checkpoint=False).")

        # Train
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        elapsed = time.time() - start_time
        logger.info(
            f"[TIME] Training finished in {elapsed//3600:.0f}h {elapsed%3600//60:.0f}m {elapsed%60:.0f}s"
        )
        logger.info("[SAVE] Saving final model...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return trainer