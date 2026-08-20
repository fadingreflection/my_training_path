#!/usr/bin/env python3
"""
get_enriched_sample.py

Скрипт для smoke test: 
- Проверяет наличие локального файла raw_data_smoke.jsonl.
- Если нет — создаёт выборку (200 записей) из большого файла raw_data_parsed.jsonl с фиксированным seed.
- Затем запускает LLM-обогащение через process_file.
- Сохраняет результат в sft_data_smoke.jsonl.

Запуск: python get_enriched_sample.py
"""

import os
import sys
import json
import random
from pathlib import Path

# Добавляем путь к модулям llm_enricher (абсолютный путь)
llm_enricher_path = "/home/afedotova/my_training_path/data_preproc_CG_SFT/llm_enricher"
sys.path.append(llm_enricher_path)

# Импортируем process_file из main.py
try:
    from main import process_file
except ModuleNotFoundError:
    print(f"❌ Не удалось импортировать process_file. Проверьте путь: {llm_enricher_path}")
    sys.exit(1)

# --- Конфигурация ---
# АБСОЛЮТНЫЙ путь к большому файлу (где лежит raw_data_parsed.jsonl)
BIG_INPUT_FILE = "/home/afedotova/my_training_path/data_preproc_CG_SFT/raw_data_parsed.jsonl"

SAMPLE_SIZE = 200
SEED = 42
SAMPLE_RAW_FILE = "raw_data_smoke.jsonl"             # локальный файл с сырой выборкой
OUTPUT_FILE = "sft_data_smoke.jsonl"                 # конечный обогащённый файл

def ensure_sample_file():
    """Проверяет наличие локального файла с выборкой. Если нет — создаёт его."""
    if Path(SAMPLE_RAW_FILE).exists():
        print(f"✅ Файл {SAMPLE_RAW_FILE} уже существует. Пропускаем создание выборки.")
        return

    print(f"📖 Файл {SAMPLE_RAW_FILE} не найден. Создаю выборку из {BIG_INPUT_FILE}...")
    # Читаем все записи из большого файла
    records = []
    with open(BIG_INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    total = len(records)
    print(f"✅ Загружено записей: {total}")

    # Перемешиваем с фиксированным seed
    random.seed(SEED)
    random.shuffle(records)

    if SAMPLE_SIZE < total:
        records = records[:SAMPLE_SIZE]
        print(f"📌 Взята выборка: {len(records)} записей")
    else:
        print(f"⚠️ Размер выборки ({SAMPLE_SIZE}) >= общего числа записей ({total}). Использую все записи.")

    # Сохраняем выборку
    with open(SAMPLE_RAW_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"💾 Выборка сохранена в {SAMPLE_RAW_FILE}")

def run_smoke_test():
    # Создаём папку для логов, если её нет
    os.makedirs("logs", exist_ok=True)

    # 1. Убеждаемся, что сырая выборка есть
    ensure_sample_file()

    # 2. Запускаем обогащение
    print(f"🚀 Запуск LLM-обогащения на {SAMPLE_RAW_FILE}...")
    process_file(SAMPLE_RAW_FILE, OUTPUT_FILE, limit=None)
    print(f"🎉 Готово! Обогащённые данные сохранены в {OUTPUT_FILE}")

if __name__ == "__main__":
    # Проверяем наличие переменной окружения с ключом
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️ Предупреждение: переменная окружения OPENROUTER_API_KEY не установлена.")
        print("Установите её перед запуском: export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)

    run_smoke_test()