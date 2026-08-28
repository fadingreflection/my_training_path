#!/usr/bin/env python3
import json
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset
from raw_extractors import RawDevignExtractor, RawBigVulExtractor, RawCodeVulnExtractor

def extract_all_and_save(output_path: str, limit: Optional[int] = None):
    """
    Запускает raw extraction для всех трёх датасетов с прогресс-баром.
    """
    # Создаём экстракторы (они загружают датасеты в память при инициализации)
    extractors = [
        RawDevignExtractor("/home/afedotova/my_training_path/raw_data/devign"),
        RawBigVulExtractor("/home/afedotova/my_training_path/raw_data/bigvul"),
        RawCodeVulnExtractor("/home/afedotova/my_training_path/raw_data/Code-Vulnerability-FineTune")
    ]

    # Общее количество записей для прогресс-бара
    total_records = sum(len(ext.dataset) for ext in extractors)
    print(f"📊 Всего записей для обработки: {total_records}")

    with open(output_path, "w", encoding="utf-8") as f:
        written = 0
        # Используем tqdm для общего прогресса, но проще по отдельности:
        for ext in extractors:
            # Название датасета для отображения
            name = ext.__class__.__name__.replace("Raw", "").replace("Extractor", "")
            dataset_len = len(ext.dataset)
            # Итерируем с tqdm
            for raw in tqdm(ext.extract(), total=dataset_len, desc=name, unit="rec"):
                f.write(json.dumps(raw, ensure_ascii=False) + "\n")
                written += 1
                if limit and written >= limit:
                    print(f"⏹️ Достигнут лимит {limit}, остановка.")
                    return
            print(f"✅ {name} обработан: {dataset_len} записей")

    print(f"🎉 Готово! Всего записано: {written} записей в {output_path}")

if __name__ == "__main__":
    # Для полного прогона оставляем limit=None
    extract_all_and_save("/home/afedotova/my_training_path/data_preproc_CG_SFT/raw_data_parsed.jsonl", limit=None)