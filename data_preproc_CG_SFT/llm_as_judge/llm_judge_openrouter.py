#!/usr/bin/env python3
"""
LLM-as-Judge с голосованием трёх моделей.
Каждая запись оценивается тремя судьями (параллельно), решение принимается большинством.
Результат записывается строго в один из трёх файлов: clean, corrected или rejected.
Поддерживается возобновление (resume) после сбоев.
Автоматическое определение прогресса из выходных файлов, если файл прогресса отсутствует.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

# Добавляем путь к модулям (если скрипт лежит в llm_enricher)
sys.path.append('/home/afedotova/my_training_path/llm_enricher')

from config import (
    JUDGE_MODELS,                     # список из трёх моделей
    API_BASE_URL,
    API_KEY_ENV_VAR,
    MAX_TOKENS,
    TEMPERATURE,
    MAX_WORKERS,
    RATE_LIMIT_SLEEP,
    LOG_FILE,
    JUDGE_INPUT_PATH,
    JUDGE_OUTPUT_CLEAN,
    JUDGE_OUTPUT_REJECTED,
    JUDGE_OUTPUT_CORRECTED,
    JUDGE_SAMPLE_SIZE,
    RESUME_FILE,
    SAVE_EVERY,
)
from utils import extract_code_from_prompt, extract_cwe_from_completion, parse_judge_response

# ========== Настройка логирования ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ========== Проверка API ключа ==========
def validate_api_key():
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        logger.error(f"Переменная окружения {API_KEY_ENV_VAR} не задана. Выход.")
        sys.exit(1)
    # Быстрая проверка: делаем тестовый запрос к моделям (список моделей)
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(f"{API_BASE_URL}/models", headers=headers, timeout=10)
        resp.raise_for_status()
        logger.info("API ключ валиден, доступ к OpenRouter установлен.")
    except Exception as e:
        logger.error(f"Не удалось проверить API ключ: {e}. Выход.")
        sys.exit(1)
    return api_key

API_KEY = validate_api_key()

# ========== Загрузка промпта ==========
PROMPT_FILE = Path(__file__).parent / "judge_prompt.txt"

def load_judge_prompt():
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Файл с промптом не найден: {PROMPT_FILE}")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

JUDGE_TEMPLATE = load_judge_prompt()

# ========== Вызов одной модели с повторными попытками ==========
def call_model(model_name: str, prompt: str, retries: int = 3) -> str | None:
    """Вызов одной модели через OpenRouter с повторными попытками."""
    for attempt in range(retries):
        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            }
            resp = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return content
        except Exception as e:
            logger.warning(f"Попытка {attempt+1}/{retries} для модели {model_name} не удалась: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # экспоненциальная задержка
    logger.error(f"Не удалось получить ответ от модели {model_name} после {retries} попыток")
    return None

def build_judge_prompt(code, cwe):
    return JUDGE_TEMPLATE.format(code=code, cwe=cwe)

# ========== Обработка одной записи (с параллельным опросом моделей) ==========
def process_record(record):
    """Обрабатывает одну запись: опрашивает трёх судей параллельно и принимает решение."""
    prompt_text = record.get("prompt", "")
    completion = record.get("completion", "")
    code = extract_code_from_prompt(prompt_text)
    orig_cwe = extract_cwe_from_completion(completion)

    if not code or not orig_cwe:
        # Не удалось извлечь код или CWE — сразу в rejected
        return record, "rejected", "Не удалось извлечь код или CWE", None

    judge_prompt = build_judge_prompt(code, orig_cwe)

    # Параллельный опрос всех моделей-судей
    votes = []  # каждый элемент: (verdict, correct_cwe, explanation)
    with ThreadPoolExecutor(max_workers=len(JUDGE_MODELS)) as model_executor:
        future_to_model = {
            model_executor.submit(call_model, model, judge_prompt): model
            for model in JUDGE_MODELS
        }
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            response = future.result()
            if response is None:
                continue  # пропускаем, если ошибка
            try:
                verdict, correct_cwe, explanation = parse_judge_response(response)
                # Коррекция: если судья сказал False, но предложил тот же CWE, считаем его голосом ЗА
                if not verdict and correct_cwe and correct_cwe == orig_cwe:
                    verdict = True
                    logger.debug(f"Коррекция: модель {model} подтвердила CWE {orig_cwe}, голос засчитан как True")
                votes.append((verdict, correct_cwe, explanation))
            except Exception as e:
                logger.error(f"Ошибка парсинга ответа от модели {model}: {e}")

    # Проверяем, сколько успешных голосов
    if len(votes) < 2:
        # Недостаточно голосов для принятия решения
        return record, "rejected", f"Недостаточно голосов (получено {len(votes)} из {len(JUDGE_MODELS)})", None

    # Подсчёт голосов ЗА (verdict=True)
    true_votes = [v for v in votes if v[0] is True]
    false_votes = [v for v in votes if v[0] is False]

    if len(true_votes) >= 2:
        # Большинство за чистоту → clean
        explanations = [v[2] for v in votes if v[2]]
        return record, "clean", " | ".join(explanations), None

    # Иначе большинство против (len(false_votes) >= 2)
    # Смотрим, предлагают ли они один и тот же CWE
    suggested_cwes = [v[1] for v in false_votes if v[1] and v[1] != "null"]
    if len(suggested_cwes) >= 2 and len(set(suggested_cwes)) == 1:
        # Все отвергающие модели предложили одинаковый CWE → corrected
        new_cwe = suggested_cwes[0]
        explanations = [v[2] for v in votes if v[2]]
        return record, "corrected", " | ".join(explanations), new_cwe
    else:
        # Нет консенсуса по исправлению → rejected
        explanations = [v[2] for v in votes if v[2]]
        return record, "rejected", " | ".join(explanations), None

# ========== Основная функция ==========
def main():
    logger.info("Запуск LLM-as-Judge (голосование трёх моделей, параллельный опрос)")
    logger.info(f"Модели-судьи: {JUDGE_MODELS}")
    logger.info(f"Входной файл: {JUDGE_INPUT_PATH}")
    logger.info(f"Выходной clean: {JUDGE_OUTPUT_CLEAN}")
    logger.info(f"Выходной rejected: {JUDGE_OUTPUT_REJECTED}")
    logger.info(f"Выходной corrected: {JUDGE_OUTPUT_CORRECTED}")

    if not Path(JUDGE_INPUT_PATH).exists():
        logger.error(f"Входной файл не найден: {JUDGE_INPUT_PATH}")
        return

    # ---- Загружаем все записи ----
    records = []
    with open(JUDGE_INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    total_loaded = len(records)
    logger.info(f"Загружено {total_loaded} записей")

    # ---- Автоматическое определение прогресса из выходных файлов (если RESUME_FILE отсутствует) ----
    def count_existing_records():
        """Подсчитывает общее количество записей в трёх выходных файлах."""
        total = 0
        for path in [JUDGE_OUTPUT_CLEAN, JUDGE_OUTPUT_REJECTED, JUDGE_OUTPUT_CORRECTED]:
            if path and Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    total += sum(1 for _ in f)
        return total

    start_from = 0
    if Path(RESUME_FILE).exists():
        with open(RESUME_FILE, 'r') as f:
            try:
                start_from = int(f.read().strip())
            except:
                start_from = 0
        if start_from > 0:
            logger.info(f"Resume: уже обработано {start_from} записей (из прогресс-файла)")
    else:
        existing = count_existing_records()
        if existing > 0:
            logger.info(f"Файл прогресса отсутствует, но обнаружено {existing} записей в выходных файлах. Продолжаем с этого индекса.")
            with open(RESUME_FILE, 'w') as pf:
                pf.write(str(existing))
            start_from = existing
        else:
            logger.info("Выходные файлы пусты или отсутствуют, начинаем с нуля.")

    # ---- Проверка, не обработаны ли уже все записи ----
    if start_from >= total_loaded:
        logger.info("Все записи уже обработаны, выход")
        return

    # Обрезаем список записей, оставляя только необработанные
    records = records[start_from:]
    logger.info(f"Осталось обработать: {len(records)} записей")

    # ---- Ограничиваем выборку для smoke-теста ----
    if JUDGE_SAMPLE_SIZE and JUDGE_SAMPLE_SIZE > 0:
        records = records[:JUDGE_SAMPLE_SIZE]
        logger.info(f"Режим SAMPLE: обрабатывается только {len(records)} записей (из {total_loaded})")

    # ---- Открываем выходные файлы (перезапись или добавление) ----
    mode = 'a' if start_from > 0 else 'w'
    logger.info(f"Режим записи: {'append' if mode == 'a' else 'overwrite'}")

    # Создаём папки для файлов, если их нет
    for path in [JUDGE_OUTPUT_CLEAN, JUDGE_OUTPUT_REJECTED, JUDGE_OUTPUT_CORRECTED]:
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Открываем файлы (режим зависит от start_from)
    out_clean = open(JUDGE_OUTPUT_CLEAN, mode, encoding='utf-8')
    out_rejected = open(JUDGE_OUTPUT_REJECTED, mode, encoding='utf-8') if JUDGE_OUTPUT_REJECTED else None
    out_corrected = open(JUDGE_OUTPUT_CORRECTED, mode, encoding='utf-8') if JUDGE_OUTPUT_CORRECTED else None

    processed = 0
    total_to_process = len(records)

    # Используем ThreadPoolExecutor для параллельной обработки записей (MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_record, rec): rec for rec in records}
        for future in tqdm(as_completed(futures), total=total_to_process, desc="Оценка"):
            rec, category, explanation, new_cwe = future.result()

            # Записываем результат строго в один файл
            if category == "clean":
                rec["judge_explanation"] = explanation
                out_clean.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_clean.flush()
            elif category == "corrected":
                new_rec = rec.copy()
                try:
                    comp = json.loads(rec["completion"])
                    if comp.get("findings"):
                        comp["findings"][0]["cwe_id"] = new_cwe
                    new_rec["completion"] = json.dumps(comp, ensure_ascii=False)
                    new_rec["judge_explanation"] = explanation
                    new_rec["judge_original_cwe"] = rec["completion"]
                    out_corrected.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    out_corrected.flush()
                except Exception as e:
                    logger.warning(f"Не удалось исправить CWE: {e}")
                    # Если не удалось исправить, пишем в rejected
                    rec["judge_explanation"] = f"Ошибка исправления: {e}"
                    out_rejected.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_rejected.flush()
            else:  # rejected
                rec["judge_explanation"] = explanation
                out_rejected.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_rejected.flush()

            processed += 1
            # Сохраняем прогресс после каждой записи (можно изменить SAVE_EVERY в конфиге)
            if processed % SAVE_EVERY == 0:
                with open(RESUME_FILE, "w") as f:
                    f.write(str(start_from + processed))
                logger.debug(f"Прогресс сохранён: {start_from + processed} записей")

            # Задержка между записями (усреднённая)
            time.sleep(RATE_LIMIT_SLEEP / MAX_WORKERS)

    # ---- Закрываем файлы ----
    out_clean.close()
    if out_rejected:
        out_rejected.close()
    if out_corrected:
        out_corrected.close()

    # ---- Удаляем файл прогресса после успешного завершения ----
    if Path(RESUME_FILE).exists():
        Path(RESUME_FILE).unlink()
        logger.info("Файл прогресса удалён (обработка завершена)")

    logger.info("✅ Обработка завершена")

if __name__ == "__main__":
    main()