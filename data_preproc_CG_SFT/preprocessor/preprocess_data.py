"""
Единый препроцессор данных для SFT.
Извлекает сырые данные из двух датасетов (BigVul, CodeVuln), балансирует по CWE,
перемешивает и сохраняет в JSONL.
Поддерживает ограничение по количеству записей (для smoke test) и фиксированный seed.
"""

import json
import random
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from raw_extractors import RawBigVulExtractor, RawCodeVulnExtractor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(
        self,
        output_path: str,
        bigvul_path: str,
        codevuln_path: str,
        limit: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
        max_per_cwe: int = 50,
    ):
        """
        Args:
            output_path: Путь для сохранения итогового JSONL-файла.
            bigvul_path: Путь к датасету BigVul.
            codevuln_path: Путь к датасету CodeVuln.
            limit: Максимальное количество записей (общее).
            sample_size: Размер выборки после перемешивания (если нужен не весь датасет).
            seed: Seed для перемешивания.
            max_per_cwe: Максимальное количество примеров на один CWE.
        """
        self.output_path = Path(output_path)
        self.bigvul_path = bigvul_path
        self.codevuln_path = codevuln_path
        self.limit = limit
        self.sample_size = sample_size
        self.seed = seed
        self.max_per_cwe = max_per_cwe

        # Создаём экстракторы (Devign исключён, так как в нём нет CWE)
        self.extractors = [
            RawBigVulExtractor(bigvul_path),
            RawCodeVulnExtractor(codevuln_path)
        ]

    def extract_all(self) -> list:
        """Извлекает все записи из датасетов."""
        all_records = []
        total_records = sum(len(ext.dataset) for ext in self.extractors)
        logger.info(f"Total records to process: {total_records}")

        for ext in self.extractors:
            name = ext.__class__.__name__.replace("Raw", "").replace("Extractor", "")
            dataset_len = len(ext.dataset)
            logger.info(f"Processing {name}...")
            for raw in tqdm(ext.extract(), total=dataset_len, desc=name, unit="rec"):
                all_records.append(raw)
                if self.limit and len(all_records) >= self.limit:
                    logger.info(f"Reached limit {self.limit}, stopping extraction.")
                    return all_records
            logger.info(f"{name} processed: {dataset_len} records")

        return all_records

    def balance_by_cwe(self, records: list) -> list:
        """
        Балансирует записи по CWE, оставляя не более max_per_cwe на каждый CWE.
        Записи без raw_cwe отбрасываются.
        """
        if not records:
            return []

        logger.info(f"Balancing by CWE, max per CWE: {self.max_per_cwe}")
        grouped = defaultdict(list)
        no_cwe_count = 0

        for rec in records:
            cwe = rec.get("raw_cwe")
            if cwe:
                grouped[cwe].append(rec)
            else:
                no_cwe_count += 1

        if no_cwe_count:
            logger.warning(f"Skipped {no_cwe_count} records without CWE")

        balanced = []
        for cwe, recs in grouped.items():
            if len(recs) > self.max_per_cwe:
                sampled = random.sample(recs, self.max_per_cwe)
                balanced.extend(sampled)
                logger.info(f"CWE {cwe}: {len(recs)} -> {len(sampled)}")
            else:
                balanced.extend(recs)
                logger.info(f"CWE {cwe}: {len(recs)} (all)")

        logger.info(f"Balanced records: {len(balanced)} (from {len(records)})")
        return balanced

    def shuffle_and_save(self, records: list):
        """Перемешивает и сохраняет записи."""
        if not records:
            logger.warning("No records to save.")
            return

        random.seed(self.seed)
        random.shuffle(records)

        if self.sample_size and self.sample_size < len(records):
            records = records[:self.sample_size]
            logger.info(f"Sample taken: {len(records)} records (from {len(records)})")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for rec in tqdm(records, desc="Saving", unit="rec"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(records)} records to {self.output_path}")

    def run(self):
        """Запускает полный пайплайн."""
        logger.info("Starting data preprocessing...")
        records = self.extract_all()
        records = self.balance_by_cwe(records)
        self.shuffle_and_save(records)
        logger.info("Preprocessing finished.")


if __name__ == "__main__":
    # === НАСТРОЙКА ===
    BIGVUL_PATH = "../../raw_data/bigvul"
    CODEVULN_PATH = "../../raw_data/Code-Vulnerability-FineTune"
    OUTPUT_PATH = "./output/raw_data_parsed_shuffled_balanced.jsonl"

    # Максимальное количество примеров на CWE
    MAX_PER_CWE = 100
    LIMIT = None
    SAMPLE_SIZE = None

    preprocessor = DataPreprocessor(
        output_path=OUTPUT_PATH,
        bigvul_path=BIGVUL_PATH,
        codevuln_path=CODEVULN_PATH,
        limit=LIMIT,
        sample_size=SAMPLE_SIZE,
        seed=42,
        max_per_cwe=MAX_PER_CWE,
    )
    preprocessor.run()