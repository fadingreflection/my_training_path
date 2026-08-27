"""
Запуск только оценки на сохранённой модели.
Использование: python eval_only.py [config.yaml]
"""

import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from pipeline.config import Config
from pipeline.data_loader import DataLoader
from pipeline.evaluator import Evaluator
from pipeline.model_builder import ModelBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = Config.from_yaml(config_path)

    # 1. Загрузка модели и токенизатора
    builder = ModelBuilder(config)
    tokenizer = builder.load_tokenizer()
    model = builder.load_model()  # базовая модель

    # 2. Загрузка адаптера (LoRA) из output_dir
    adapter_path = config.output_dir
    if Path(adapter_path).exists():
        logger.info(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.warning("Adapter not found, using base model.")
    model.eval()

    # 3. Загрузка данных
    loader = DataLoader(config, tokenizer)
    records = loader.load_raw(config.data_path)
    train_rec, val_rec, test_rec = loader.split_data(records)

    # 4. Оценка на тесте
    evaluator = Evaluator(config, model, tokenizer)
    sample_size = getattr(config, 'eval_sample_size', None)
    # Преобразуем строку в число или None
    if isinstance(sample_size, str):
        if sample_size.lower() in ('null', 'none', ''):
            sample_size = None
        else:
            try:
                sample_size = int(sample_size)
            except ValueError:
                sample_size = None
                logger.warning(f"Invalid eval_sample_size value, using None (all records).")

    # Очистка кеша перед оценкой
    torch.cuda.empty_cache()

    metrics = evaluator.evaluate_and_log(
        raw_test_records=test_rec,
        sample_size=sample_size,
        log_file="test_predictions_eval_only.jsonl"
    )
    logger.info(f"Test results: {metrics}")

    # 5. (Опционально) Оценка на трейне
    if getattr(config, 'evaluate_on_train', True):
        metrics_train = evaluator.evaluate_and_log(
            raw_test_records=train_rec,
            sample_size=sample_size,
            log_file="train_predictions_eval_only.jsonl"
        )
        logger.info(f"Train results: {metrics_train}")

if __name__ == "__main__":
    main()