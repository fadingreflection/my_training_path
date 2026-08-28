# #!/usr/bin/env python3
# import os
# import json
# import re
# import time
# import logging
# from pathlib import Path
# from typing import List, Dict, Optional
# from openai import OpenAI

# from config import (
#     TEACHER_MODEL,
#     API_BASE_URL,
#     API_KEY_ENV_VAR,
#     MAX_TOKENS,
#     TEMPERATURE,
#     RATE_LIMIT_SLEEP,
#     LOG_FILE,
#     RAW_DATA_PATH,
#     OUTPUT_PATH,
# )
# from prompt_templates import build_enrichment_prompt

# # Настройка логирования
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )

# # Создаём клиент OpenAI (как в рабочем коде)
# def get_client():
#     api_key = os.environ.get(API_KEY_ENV_VAR)
#     if not api_key:
#         raise ValueError(f"Environment variable {API_KEY_ENV_VAR} not set")
#     return OpenAI(
#         api_key=api_key,
#         base_url=API_BASE_URL,
#         default_headers={
#             "HTTP-Referer": "http://localhost",
#             "X-Title": "LLM Enricher",
#         }
#     )

# client = get_client()

# def call_teacher_model(prompt: str) -> Optional[str]:
#     try:
#         response = client.chat.completions.create(
#             model=TEACHER_MODEL,
#             messages=[
#                 {"role": "system", "content": "You are a cybersecurity expert."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=MAX_TOKENS,
#             temperature=TEMPERATURE,
#         )
#         content = response.choices[0].message.content
#         with open("raw_responses.log", "a", encoding="utf-8") as f:
#             f.write(f"=== RESPONSE ===\n{content}\n\n")
#         return content
#     except Exception as e:
#         logging.error(f"Ошибка при вызове модели: {e}")
#         return None

# def parse_llm_output(output_text: str) -> Optional[List[Dict]]:
#     """Извлекает JSON из ответа LLM."""
#     json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
#     if json_match:
#         try:
#             data = json.loads(json_match.group())
#             if isinstance(data, dict) and "findings" in data:
#                 return data["findings"]
#             elif isinstance(data, list):
#                 return data
#         except json.JSONDecodeError as e:
#             logging.error(f"Ошибка парсинга JSON: {e}")
#     return None

# def fallback_findings(raw: dict) -> List[Dict]:
#     return [{
#         "concept_type": "pattern",
#         "title": "Обнаружена уязвимость",
#         "explanation": raw.get("raw_description") or "Требуется анализ.",
#         "location": raw.get("raw_location") or "неизвестно",
#         "severity": raw.get("raw_severity") or "medium",
#         "recommendation": raw.get("raw_recommendation") or "Провести анализ."
#     }]

# def enrich_example(raw: dict) -> List[Dict]:
#     prompt = build_enrichment_prompt(raw)
#     response_text = call_teacher_model(prompt)
#     logging.info(f"Raw response (first 300 chars): {response_text[:300] if response_text else 'None'}")
#     if response_text is None:
#         return fallback_findings(raw)
#     findings = parse_llm_output(response_text)
#     if findings is None:
#         return fallback_findings(raw)
#     return findings

# def process_file(input_path: str, output_path: str, limit: Optional[int] = None):
#     total_processed = 0
#     total_skipped = 0
#     with open(input_path, "r", encoding="utf-8") as f_in, \
#          open(output_path, "w", encoding="utf-8") as f_out:
#         for line in f_in:
#             raw = json.loads(line.strip())
#             code = raw.get("code", "")
#             if not code:
#                 total_skipped += 1
#                 continue
#             findings = enrich_example(raw)
#             sft_record = {
#                 "input": {"code": code, "context": raw.get("context", "")},
#                 "output": {"findings": findings}
#             }
#             f_out.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
#             total_processed += 1
#             if total_processed % 10 == 0:
#                 logging.info(f"Обработано {total_processed} примеров")
#             time.sleep(RATE_LIMIT_SLEEP)
#             if limit and total_processed >= limit:
#                 break
#     logging.info(f"Готово. Обработано: {total_processed}, пропущено: {total_skipped}")

# if __name__ == "__main__":
#     process_file(RAW_DATA_PATH, OUTPUT_PATH, limit=1)
#     logging.info("Скрипт завершён.")



#!/usr/bin/env python3
import os
import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Создаём клиент OpenAI
def get_client():
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError(f"Environment variable {API_KEY_ENV_VAR} not set")
    return OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "LLM Enricher",
        }
    )

client = get_client()

def call_teacher_model(prompt: str) -> Optional[str]:
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
        logging.error(f"Ошибка при вызове модели: {e}")
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
            logging.error(f"Ошибка парсинга JSON: {e}")
    return None

def fallback_findings(raw: dict) -> List[Dict]:
    return [{
        "concept_type": "pattern",
        "title": "Обнаружена уязвимость",
        "explanation": raw.get("raw_description") or "Требуется анализ.",
        "location": raw.get("raw_location") or "неизвестно",
        "severity": raw.get("raw_severity") or "medium",
        "recommendation": raw.get("raw_recommendation") or "Провести анализ."
    }]

def enrich_example(raw: dict) -> List[Dict]:
    prompt = build_enrichment_prompt(raw)
    response_text = call_teacher_model(prompt)
    logging.info(f"Raw response (first 300 chars): {response_text[:300] if response_text else 'None'}")
    if response_text is None:
        return fallback_findings(raw)
    findings = parse_llm_output(response_text)
    if findings is None:
        return fallback_findings(raw)
    return findings

def process_file(input_path: str, output_path: str, limit: Optional[int] = None):
    total_processed = 0
    total_skipped = 0
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            raw = json.loads(line.strip())
            code = raw.get("code", "")
            if not code:
                total_skipped += 1
                continue

            findings = enrich_example(raw)

            # Преобразуем в формат Prompt-Completion
            # Формируем промпт (вопрос пользователя)
            user_prompt = f"Проанализируй код и верни ТОЛЬКО JSON с полем findings:\n{code}\n\nКонтекст: {raw.get('context', '')}"
            # Формируем комплишн (ответ ассистента) - это должен быть чистый JSON
            completion = json.dumps({"findings": findings}, ensure_ascii=False)

            # Сохраняем в формате Prompt-Completion
            sft_record = {
                "prompt": user_prompt,
                "completion": completion
            }

            f_out.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            total_processed += 1
            if total_processed % 10 == 0:
                logging.info(f"Обработано {total_processed} примеров")
            time.sleep(RATE_LIMIT_SLEEP)
            if limit and total_processed >= limit:
                break
    logging.info(f"Готово. Обработано: {total_processed}, пропущено: {total_skipped}")

if __name__ == "__main__":
    process_file(RAW_DATA_PATH, OUTPUT_PATH, limit=1)
    logging.info("Скрипт завершён.")