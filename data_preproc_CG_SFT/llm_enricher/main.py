#!/usr/bin/env python3
"""
Обогащение сырых данных через LLM (OpenRouter) с поддержкой возобновления.
Использует батчевую параллельную обработку для ускорения.
Не перезаписывает уже обработанные записи.
"""

import os
import json
import re
import time
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from config import (
    TEACHER_MODEL,
    API_BASE_URL,
    API_KEY_ENV_VAR,
    MAX_TOKENS,
    TEMPERATURE,
    RATE_LIMIT_SLEEP,
    LOG_FILE,
    RAW_DATA_PATH,
    OUTPUT_PATH,
)
from prompt_templates import build_enrichment_prompt

# ========== Настройка логирования ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== Проверка API-ключа ==========
def validate_api_key():
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        logger.error(f"Переменная окружения {API_KEY_ENV_VAR} не задана. Выход.")
        sys.exit(1)
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(f"{API_BASE_URL}/models", headers=headers, timeout=10)
        resp.raise_for_status()
        logger.info("API ключ валиден, доступ к OpenRouter установлен.")
    except Exception as e:
        logger.error(f"Не удалось проверить API ключ: {e}. Выход.")
        sys.exit(1)
    return api_key

API_KEY = validate_api_key()

# ========== Создание клиента OpenAI ==========
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "LLM Enricher",
    }
)

# ========== Функции работы с моделью ==========
def call_teacher_model(prompt: str, retries: int = 3) -> Optional[str]:
    """Вызов модели с повторными попытками при ошибках."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            content = response.choices[0].message.content
            with open("raw_responses.log", "a", encoding="utf-8") as f:
                f.write(f"=== RESPONSE ===\n{content}\n\n")
            return content
        except Exception as e:
            logger.warning(f"Попытка {attempt+1}/{retries} не удалась: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    logger.error(f"Не удалось получить ответ от модели после {retries} попыток")
    return None

def parse_llm_output(output_text: str) -> Optional[List[Dict]]:
    """Извлекает JSON из ответа LLM."""
    json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and "findings" in data:
                return data["findings"]
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
    return None

def fallback_findings(raw: dict) -> List[Dict]:
    """Запасной вариант, если LLM не ответил."""
    return [{
        "concept_type": "pattern",
        "title": "Обнаружена уязвимость",
        "explanation": raw.get("raw_description") or "Требуется анализ.",
        "location": raw.get("raw_location") or "неизвестно",
        "severity": raw.get("raw_severity") or "medium",
        "recommendation": raw.get("raw_recommendation") or "Провести анализ."
    }]

def enrich_example(raw: dict) -> List[Dict]:
    """Обогащает один пример."""
    prompt = build_enrichment_prompt(raw)
    response_text = call_teacher_model(prompt)
    if response_text is None:
        return fallback_findings(raw)
    findings = parse_llm_output(response_text)
    if findings is None:
        return fallback_findings(raw)
    return findings

def process_single_record(raw: dict, idx: int) -> Tuple[int, Optional[Dict]]:
    """
    Обрабатывает одну запись и возвращает (индекс, SFT-запись или None).
    Используется для параллельной обработки.
    """
    code = raw.get("code", "")
    if not code:
        logger.warning(f"Запись {idx} пропущена: нет поля 'code'")
        return idx, None

    findings = enrich_example(raw)
    user_prompt = f"Проанализируй код и верни ТОЛЬКО JSON с полем findings:\n{code}\n\nКонтекст: {raw.get('context', '')}"
    completion = json.dumps({"findings": findings}, ensure_ascii=False)
    sft_record = {
        "prompt": user_prompt,
        "completion": completion
    }
    return idx, sft_record

# ========== Основная функция с батчевой обработкой ==========
def process_file(input_path: str, output_path: str,
                 limit: Optional[int] = None,
                 batch_size: int = 10,
                 max_workers: int = 10,
                 save_every: int = 1):
    """
    Обрабатывает файл с батчевой параллельной обработкой и поддержкой возобновления.
    - input_path: путь к входному JSONL
    - output_path: путь к выходному JSONL
    - limit: сколько записей обработать (None = все)
    - batch_size: размер батча (сколько записей обрабатывать параллельно)
    - max_workers: количество потоков в пуле (должно быть >= batch_size)
    - save_every: сохранять прогресс каждые N записей
    """
    # Проверяем входной файл
    if not Path(input_path).exists():
        logger.error(f"Входной файл не найден: {input_path}")
        return

    # Загружаем все записи
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    total = len(records)
    logger.info(f"Загружено {total} записей")

    # Определяем файл прогресса
    progress_file = output_path + ".progress"
    start_from = 0

    # Если есть файл прогресса, читаем его
    if Path(progress_file).exists():
        with open(progress_file, "r") as pf:
            try:
                start_from = int(pf.read().strip())
            except ValueError:
                start_from = 0
        if start_from > 0:
            logger.info(f"Resume: уже обработано {start_from} записей")
    else:
        # Если прогресс-файла нет, но выходной файл существует, считаем строки в нём
        if Path(output_path).exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_lines = sum(1 for _ in f)
            if existing_lines > 0:
                logger.info(f"Файл прогресса отсутствует, но обнаружено {existing_lines} записей в выходном файле. Продолжаем с этого индекса.")
                start_from = existing_lines
                with open(progress_file, 'w') as pf:
                    pf.write(str(start_from))
            else:
                logger.info("Выходной файл пуст, начинаем с нуля.")
        else:
            logger.info("Выходной файл отсутствует, начинаем с нуля.")

    if start_from >= total:
        logger.info("Все записи уже обработаны, выход")
        return

    # Ограничиваем выборку, если задан limit
    if limit is not None and limit > 0:
        end_at = min(start_from + limit, total)
    else:
        end_at = total

    records_to_process = records[start_from:end_at]
    total_to_process = len(records_to_process)
    logger.info(f"Осталось обработать: {total_to_process} записей (с {start_from} по {end_at-1})")

    # Создаём папку для выходного файла
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Открываем выходной файл в режиме append, если уже есть записи
    mode = 'a' if start_from > 0 else 'w'
    logger.info(f"Режим записи: {'append' if mode == 'a' else 'overwrite'}")

    processed_in_batch = 0
    with open(output_path, mode, encoding='utf-8') as f_out:
        # Разбиваем записи на батчи
        for batch_start in range(0, total_to_process, batch_size):
            batch_end = min(batch_start + batch_size, total_to_process)
            batch_records = records_to_process[batch_start:batch_end]
            batch_indices = list(range(start_from + batch_start, start_from + batch_end))

            logger.info(f"Обработка батча {batch_start//batch_size + 1}: записи {batch_indices[0]} - {batch_indices[-1]}")

            # Параллельно обрабатываем батч
            results = []  # список (idx, record)
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_records))) as executor:
                future_to_idx = {
                    executor.submit(process_single_record, rec, idx): idx
                    for rec, idx in zip(batch_records, batch_indices)
                }
                for future in as_completed(future_to_idx):
                    idx, result = future.result()
                    results.append((idx, result))

            # Сортируем результаты по индексу, чтобы сохранить порядок
            results.sort(key=lambda x: x[0])

            # Записываем в файл в правильном порядке
            for idx, result in results:
                if result is not None:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                processed_in_batch += 1
                # Сохраняем прогресс
                if processed_in_batch % save_every == 0:
                    current_total = start_from + processed_in_batch
                    with open(progress_file, "w") as pf:
                        pf.write(str(current_total))
                    logger.info(f"Прогресс сохранён: {current_total} записей")

            # Небольшая задержка между батчами для снижения нагрузки на API
            time.sleep(RATE_LIMIT_SLEEP)

    # Удаляем прогресс-файл, если обработаны все записи
    if start_from + processed_in_batch >= total:
        if Path(progress_file).exists():
            Path(progress_file).unlink()
            logger.info("Файл прогресса удалён (обработка завершена)")

    logger.info(f"✅ Готово. Обработано: {processed_in_batch} записей.")

if __name__ == "__main__":
    # Настройки скорости (можно вынести в config.py)
    BATCH_SIZE = 10          # сколько записей обрабатывать параллельно
    MAX_WORKERS = 10         # число потоков (должно быть >= BATCH_SIZE)
    SAVE_EVERY = 10          # сохранять прогресс каждые N записей

    process_file(
        input_path=RAW_DATA_PATH,
        output_path=OUTPUT_PATH,
        limit=None,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        save_every=SAVE_EVERY
    )
    logger.info("Скрипт завершён.")