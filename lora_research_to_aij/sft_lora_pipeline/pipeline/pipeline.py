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
            handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
        )

    def _get_free_gpu(self):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                text=True,
            )
            best_gpu, max_free = None, -1
            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                idx, free = line.split(",")
                idx, free = int(idx.strip()), int(free.strip())
                if free > max_free:
                    max_free, best_gpu = free, idx
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
            records=records, sample_size=sample_size, log_file=log_file
        )
        logger.info(
            f"Evaluation results: macro_f1={metrics.get('macro_f1', 0):.4f}, "
            f"json_validity={metrics.get('json_validity', 0):.2%}, "
            f"schema_ok={metrics.get('schema_ok', 0):.2%}"
        )
        return metrics

    def run(self):
        logger.info("Starting SFT pipeline")

        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")

        best_gpu, max_free = self._get_free_gpu()
        if best_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            logger.info(f"Selected GPU {best_gpu} with {max_free} MB free")

        logger.info("Loading model and tokenizer...")
        builder = ModelBuilder(self.config)
        self.tokenizer = builder.load_tokenizer()
        self.model = builder.load_model()
        logger.info("Model and tokenizer loaded")

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
            before = len(records)
            records = loader.drop_content_overlap(records, holdout_rec)
            dropped = before - len(records)
            if dropped:
                logger.warning(f"Dropped {dropped} train rows overlapping holdout")

        val_path = getattr(self.config, "val_data_path", None)
        if val_path:
            val_rec = loader.load_raw(val_path)
            train_rec, test_rec = records, []
            logger.info(f"Using val_data_path: train={len(train_rec)}, val={len(val_rec)}")
        else:
            train_rec, val_rec, test_rec = loader.split_data(records)

        if getattr(self.config, "apply_chat_template", False):
            train_rec = loader.apply_chat_prompts(train_rec)
            if val_rec:
                val_rec = loader.apply_chat_prompts(val_rec)
            if holdout_rec:
                holdout_rec = loader.apply_chat_prompts(holdout_rec)

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

        logger.info("Starting training...")
        trainer_obj = SFTTrainerWrapper(self.config, self.model, self.tokenizer, train_ds, val_ds)
        self.trainer = trainer_obj.train()

        if self.config.evaluate_on_test and test_rec:
            sample_size = getattr(self.config, "eval_sample_size", None)
            self._evaluate(test_rec, sample_size, "holdout_predictions.jsonl")

        if self.config.evaluate_on_train and train_rec:
            sample_size = getattr(self.config, "train_eval_sample_size", None)
            self._evaluate(train_rec, sample_size, "train_predictions.jsonl")

        logger.info("Pipeline finished successfully")