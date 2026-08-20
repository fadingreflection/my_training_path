#!/usr/bin/env python3
"""
Скрипт для подсчёта количества записей по источнику (source) в JSONL-файле.
Запуск: python count_sources.py
"""

import json
from collections import Counter

INPUT_FILE = "/home/afedotova/my_training_path/data_preproc_CG_SFT/raw_data_parsed.jsonl"

def count_sources(file_path):
    counter = Counter()
    total = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                source = data.get("source", "unknown")
                counter[source] += 1
                total += 1
            except json.JSONDecodeError:
                print(f"⚠️ Ошибка парсинга строки: {line[:100]}...")

    print("📊 Статистика по источникам:")
    for source, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} записей")
    print(f"\n✅ Всего записей: {total}")

if __name__ == "__main__":
    count_sources(INPUT_FILE)