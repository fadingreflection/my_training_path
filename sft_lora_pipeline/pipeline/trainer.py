import os
import time
import logging
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

logger = logging.getLogger(__name__)

class SFTTrainerWrapper:
    def __init__(self, config, model, tokenizer, train_dataset, val_dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer          # сохраняем для сохранения
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self):
        logger.info("🔥 Начало обучения...")
        start_time = time.time()

        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1. Конфигурация LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # # 2. Аргументы обучения (стандартные TrainingArguments)
        # training_args = TrainingArguments(
        #     output_dir=output_dir,
        #     num_train_epochs=self.config.num_train_epochs,
        #     per_device_train_batch_size=self.config.per_device_train_batch_size,
        #     per_device_eval_batch_size=self.config.per_device_eval_batch_size,
        #     gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        #     learning_rate=self.config.learning_rate,
        #     warmup_steps=self.config.warmup_steps,
        #     weight_decay=self.config.weight_decay,
        #     logging_steps=self.config.logging_steps,
        #     save_steps=self.config.save_steps,
        #     eval_steps=self.config.eval_steps,
        #     save_total_limit=self.config.save_total_limit,
        #     load_best_model_at_end=self.config.load_best_model_at_end,
        #     metric_for_best_model=self.config.metric_for_best_model,
        #     greater_is_better=self.config.greater_is_better,
        #     report_to=[self.config.report_to] if self.config.use_tensorboard else [],
        #     bf16=True,
        #     fp16=False,
        #     eval_strategy="steps",
        #     save_strategy="steps",
        #     logging_strategy="steps",
        #     remove_unused_columns=False,
        #     seed=self.config.seed,

        # )
        from trl import SFTConfig

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

            # ----- SFT-специфичные параметры -----
            loss_type="nll",                     # <-- ключевое: отключаем chunked_nll
            max_length=self.config.max_seq_length,  # если у вас есть такой параметр
            # packing=self.config.packing,         # если используете упаковку
            # можно добавить другие параметры, например:
            # dataset_text_field=self.config.dataset_text_field,
            # formatting_func=...,
        )

        # 3. Создаём SFTTrainer
        # ВНИМАНИЕ: В версии 1.10.0 токенизатор передаётся через processing_class!
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            # tokenizer=self.tokenizer,      # <-- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
            peft_config=lora_config,
            # Эти параметры НЕ ПЕРЕДАЮТСЯ в SFTTrainer напрямую,
            # они должны быть в SFTConfig. Но для простоты мы
            # будем использовать дефолтные значения.
        )
        print("СОЗДАЛИ")
        # 4. Возобновление из чекпоинта (отказоустойчивость)
        resume_from_checkpoint = None
        if os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
                resume_from_checkpoint = os.path.join(output_dir, latest)
                logger.info(f"🔄 Возобновление из чекпоинта: {resume_from_checkpoint}")

        # 5. Запуск обучения
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # 6. Сохранение
        elapsed = time.time() - start_time
        logger.info(f"⏱️ Обучение завершено за {elapsed//3600:.0f}ч {elapsed%3600//60:.0f}м {elapsed%60:.0f}с")
        logger.info("💾 Сохранение финальной модели...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return trainer