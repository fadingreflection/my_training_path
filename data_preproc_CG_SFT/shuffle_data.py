#!/usr/bin/env python3
"""
Скрипт для перемешивания (shuffle) и/или создания выборки из JSONL-файла.
Поддерживает:
- Полное перемешивание всего файла.
- Извлечение случайной подвыборки заданного размера.
- Фиксированный seed для воспроизводимости.
- Прогресс-бар.
"""

import json
import random
import argparse
from tqdm import tqdm

def shuffle_dataset(
    input_file: str,
    output_file: str,
    sample_size: int = None,
    seed: int = 42,
):
    """
    Перемешивает записи из input_file и сохраняет в output_file.
    Если sample_size указан, берёт случайную подвыборку этого размера.
    """
    # Читаем все записи
    print(f"📖 Чтение данных из {input_file}...")
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Загрузка", unit="rec"):
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    print(f"✅ Загружено записей: {total}")

    # Перемешиваем
    random.seed(seed)
    random.shuffle(records)

    # Если нужна выборка
    if sample_size is not None and sample_size < total:
        records = records[:sample_size]
        print(f"📌 Взята выборка: {len(records)} записей (из {total})")

    # Сохраняем
    print(f"💾 Сохранение в {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in tqdm(records, desc="Сохранение", unit="rec"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"🎉 Готово! Сохранено записей: {len(records)}")

if __name__ == "__main__":
    input_path = "/home/afedotova/my_training_path/data_preproc_CG_SFT/raw_data_parsed.jsonl"
    output_path = input_path.replace(".jsonl", "_shuffled.jsonl")  # добавляем _shuffled

    # === НАСТРОЙКА ===
    sample_size = None   # для полного перемешивания
    # sample_size = 200      # для smoke test — раскомментируй нужную строку

    shuffle_dataset(input_path, output_path, sample_size=sample_size, seed=42)