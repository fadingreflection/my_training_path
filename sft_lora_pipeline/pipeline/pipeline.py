import logging
import os
import subprocess

import torch

from .config import Config
from .data_loader import DataLoader
from .evaluator import Evaluator
from .model_builder import ModelBuilder
from .trainer import SFTTrainerWrapper

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

    def _get_free_gpu(self):
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
            return best_gpu, max_free
        except Exception as e:
            logger.warning(f"Error selecting GPU: {e}. Using default behavior.")
            return None, 0

    def _evaluate(self, records, sample_size, log_file):
        if not records:
            logger.warning(f"Records list empty, skipping evaluation to {log_file}")
            return None
        evaluator = Evaluator(self.config, self.model, self.tokenizer)
        metrics = evaluator.evaluate_and_log(
            raw_test_records=records,
            sample_size=sample_size,
            log_file=log_file
        )
        logger.info(f"Evaluation results: json_accuracy={metrics.get('json_accuracy', 0):.2%}, "
                    f"cwe_accuracy={metrics.get('cwe_accuracy', 0):.2%}")
        return metrics

    def run(self):
        logger.info("Starting SFT pipeline")

        # 1. GPU check
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")

        # 2. Select free GPU
        best_gpu, max_free = self._get_free_gpu()
        if best_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            logger.info(f"Selected GPU {best_gpu} with {max_free} MB free memory")
        else:
            logger.warning("Could not determine free GPU, using default behavior")

        # 3. Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        builder = ModelBuilder(self.config)
        self.tokenizer = builder.load_tokenizer()
        logger.info("Tokenizer loaded")

        self.model = builder.load_model()
        logger.info("Model loaded")

        # 4. Load train pool; val = 10% от него (для eval_loss). Holdout — только финальный тест.
        logger.info("Loading data...")
        loader = DataLoader(self.config, self.tokenizer)
        records = loader.load_raw(self.config.data_path)
        if not records:
            raise ValueError("No data loaded from data_path")
        logger.info(f"Loaded {len(records)} train-pool records from {self.config.data_path}")

        holdout_path = getattr(self.config, "holdout_data_path", None)
        holdout_rec = []
        if holdout_path:
            holdout_rec = loader.load_raw(holdout_path)
            logger.info(f"Loaded {len(holdout_rec)} holdout records from {holdout_path}")
            # убрать точные дубликаты holdout из train-pool (content leak)
            before = len(records)
            records = loader.drop_content_overlap(records, holdout_rec)
            dropped = before - len(records)
            if dropped:
                logger.warning(f"Dropped {dropped} train-pool rows with exact prompt+completion overlap vs holdout")

        train_rec, val_rec, test_rec = loader.split_data(records)
        # финальный тест — только holdout; random test_split должен быть 0
        if holdout_rec:
            if test_rec:
                logger.warning(
                    f"test_split produced {len(test_rec)} rows, but holdout_data_path is set — "
                    f"using holdout only for final test"
                )
            test_rec = holdout_rec
        logger.info(f"Split: train={len(train_rec)}, val={len(val_rec)}, holdout_test={len(test_rec)}")

        train_ds, val_ds, _ = loader.prepare_datasets(train_rec, val_rec, [])
        logger.info("Datasets prepared")

        # 5. Training
        logger.info("Starting training...")
        trainer_obj = SFTTrainerWrapper(self.config, self.model, self.tokenizer, train_ds, val_ds)
        self.trainer = trainer_obj.train()

        # 6. Evaluation on holdout test
        if self.config.evaluate_on_test and test_rec:
            sample_size = getattr(self.config, 'eval_sample_size', None)
            self._evaluate(test_rec, sample_size, "holdout_predictions.jsonl")

        # 7. Evaluation on train (optional)
        if self.config.evaluate_on_train and train_rec:
            sample_size = getattr(self.config, 'train_eval_sample_size', None)
            self._evaluate(train_rec, sample_size, "train_predictions.jsonl")

        logger.info("Pipeline finished successfully")