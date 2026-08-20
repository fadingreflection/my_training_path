import logging
import os
import subprocess
import torch
from .config import Config
from .data_loader import DataLoader
from .model_builder import ModelBuilder
from .trainer import SFTTrainerWrapper   # вместо SFTTrainer
from .evaluator import Evaluator

logger = logging.getLogger(__name__)

class SFTLPipeline:
    def __init__(self, config_path: str):
        self.config = Config.from_yaml(config_path)
        os.makedirs("logs", exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/training.log"),
                logging.StreamHandler()
            ]
        )

    def run(self):
        logger.info("🚀 Запуск SFT пайплайна")

        # 1. GPU проверка
        if not torch.cuda.is_available():
            raise RuntimeError("❌ GPU не доступен")
        logger.info(f"✅ Доступно GPU: {torch.cuda.device_count()}")

        # 1.5 Выбор свободного GPU
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                text=True
            )
            lines = output.strip().split('\n')
            best_gpu = None
            max_free = -1
            for line in lines:
                if not line.strip():
                    continue
                idx, free = line.split(',')
                idx = int(idx.strip())
                free = int(free.strip())
                if free > max_free:
                    max_free = free
                    best_gpu = idx
            if best_gpu is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                logger.info(f"✅ Выбран GPU {best_gpu} с {max_free} МБ свободной памяти")
            else:
                logger.warning("⚠️ Не удалось определить свободный GPU, используем стандартное поведение")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при выборе GPU: {e}. Используем стандартное поведение")

        # 2. Загрузка модели и токенизатора
        logger.info("⏳ Загрузка модели и токенизатора...")
        builder = ModelBuilder(self.config)
        tokenizer = builder.load_tokenizer()
        logger.info("✅ Токенизатор загружен")

        model = builder.load_model()
        logger.info("✅ Модель загружена")
            # После загрузки модели

        # Добавляем недостающий атрибут (костыль)
        if not hasattr(model.config, 'text_config'):
            model.config.text_config = model.config

        # LoRA будет применена внутри SFTTrainer через peft_config

        # 3. Данные
        logger.info("⏳ Загрузка данных...")
        loader = DataLoader(self.config, tokenizer)
        records = loader.load_raw(self.config.data_path)
        logger.info(f"✅ Загружено записей: {len(records)}")

        train_rec, val_rec, test_rec = loader.split_data(records)
        logger.info(f"✅ Разбивка: train={len(train_rec)}, val={len(val_rec)}, test={len(test_rec)}")

        train_ds, val_ds, test_ds = loader.prepare_datasets(train_rec, val_rec, test_rec)
        logger.info("✅ Датасеты подготовлены")

        # 4. Обучение
        logger.info("🔥 Запуск обучения...")
        trainer_obj = SFTTrainerWrapper(self.config, model, tokenizer, train_ds, val_ds)
        trainer = trainer_obj.train()

        # 5. Оценка
        evaluator = Evaluator(self.config, model, tokenizer)
        if self.config.evaluate_on_test and test_ds:
            json_acc = evaluator.evaluate_json_structure(test_ds, sample_size=20)
            logger.info(f"📈 JSON accuracy на тесте: {json_acc:.2%}")

        if test_ds:
            test_loss = trainer.evaluate(test_ds)["eval_loss"]
            logger.info(f"📉 Test loss: {test_loss:.4f}")

        logger.info("🎉 Пайплайн успешно завершён.")