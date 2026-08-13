#!/usr/bin/env python3
"""
Пайплайн сбора данных для CWE Hypothesis Generator.
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI

# =========== КОНФИГУРАЦИЯ ===========
CONFIG = {
    "teacher_model": "nvidia/nemotron-3-ultra-550b-a55b:free",  # пробуем DeepSeek R1
    "api_base_url": "https://openrouter.ai/api/v1",
    "api_key_env_var": "OPENROUTER_API_KEY",
    "max_tokens": 2048,
    "temperature": 0.3,
    "rate_limit_sleep": 2.0,
    "log_file": "cwe_teacher_response.log",
}


def load_prompt(filepath: str = "cwe_prompt.txt") -> str:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Prompt file {filepath} not found.")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT_CWE = load_prompt("hypothesis_generator_prompt.txt")

def get_teacher_client() -> OpenAI:
    api_key = "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"
    print(f"🔍 API key from env: '{api_key[:15] if api_key else 'EMPTY'}'... (length: {len(api_key) if api_key else 0})")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set or empty.")
    return OpenAI(
        api_key=api_key,
        base_url=CONFIG["api_base_url"],
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "CWE Hypothesis Generator Data Pipeline",
        }
    )

def parse_json_response(content: str) -> Dict[str, Any]:
    """Извлекает JSON из ответа."""
    # Убираем маркеры и лишние пробелы
    cleaned = content.replace('\ufeff', '')
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()

    # Ищем JSON объект
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        json_str = match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Пробуем repair
            try:
                from json_repair import repair_json
                repaired = repair_json(json_str)
                return json.loads(repaired)
            except ImportError:
                pass
    # Если не нашли, пробуем найти только cwe_hypotheses
    match = re.search(r'"cwe_hypotheses"\s*:\s*(\[.*?\])', cleaned, re.DOTALL)
    if match:
        list_str = match.group(1)
        wrapped = f'{{"cwe_hypotheses": {list_str}}}'
        try:
            return json.loads(wrapped)
        except:
            pass
    # Ничего не вышло — возвращаем пустой объект
    print("⚠️ Could not parse JSON. Returning empty hypotheses.")
    return {"cwe_hypotheses": []}

# =========== КЛАСС ГЕНЕРАТОРА ===========

class CWEHypothesisGenerator:
    def __init__(self, model: str = CONFIG["teacher_model"]):
        self.client = get_teacher_client()
        self.model = model
        self.log_file = Path(CONFIG["log_file"])

    def generate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        arch_json = json.dumps(architecture, indent=2)
        user_content = SYSTEM_PROMPT_CWE.format(architecture_json=arch_json)
        system_msg = "You are a security expert. You must output only valid JSON. Your response must contain exactly one JSON object with a field 'cwe_hypotheses' (an array of objects with cwe_id, name, reasoning, confidence). If no vulnerabilities are likely, return an empty array."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content}
            ],
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"],
        )
        print("HERE")
        print(response)
        content = response.choices[0].message.content

        # Всегда выводим и логируем сырой ответ
        print("\n" + "="*80)
        print("📩 RAW RESPONSE FROM TEACHER:")
        print(content)
        print("="*80 + "\n")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"=== MODEL: {self.model} ===\n")
            f.write(f"=== FULL RESPONSE ===\n{content}\n")
            f.write("=" * 80 + "\n\n")

        # Пытаемся распарсить
        try:
            parsed = parse_json_response(content)
        except Exception as e:
            print(f"⚠️ Parsing failed: {e}")
            # Сохраняем сырой ответ в отдельный файл для диагностики
            with open("cwe_raw_responses.log", "a", encoding="utf-8") as f:
                f.write(f"=== FAILED TO PARSE ===\n{content}\n\n")
            # Возвращаем пустой список, чтобы не прерывать
            return {"cwe_hypotheses": []}

        # Валидация
        if "cwe_hypotheses" not in parsed:
            parsed["cwe_hypotheses"] = []
        return parsed

# =========== КЛАСС СБОРЩИКА ===========

class CWEHypothesisDatasetBuilder:
    def __init__(self, generator: CWEHypothesisGenerator,
                 input_file: str,
                 output_file: str):
        self.generator = generator
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def _extract_architecture(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if "messages" in record:
            assistant_msg = None
            for msg in record["messages"]:
                if msg["role"] == "assistant":
                    assistant_msg = msg
                    break
            if not assistant_msg:
                raise ValueError("No assistant message")
            content = assistant_msg["content"]
            # Ищем JSON
            match = re.search(r'\{.*"app_type".*\}', content, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError("No architecture JSON")
            arch_str = match.group()
            try:
                return json.loads(arch_str)
            except:
                try:
                    from json_repair import repair_json
                    return json.loads(repair_json(arch_str))
                except:
                    raise ValueError("Cannot parse architecture")
        else:
            if "architecture" in record:
                return record["architecture"]
            return record

    def _create_sample(self, architecture: Dict[str, Any],
                       cwe_hypotheses: Dict[str, Any]) -> Dict[str, Any]:
        system_content = "You are a security expert. Your task is to predict which CWE classes are most likely to be present in an application, based solely on its high-level architecture description."

        arch_json = json.dumps(architecture, indent=2)
        user_content = f"Architecture description:\n{arch_json}\n\nTask:\nAnalyze the application type and the responsibilities of each component. List the CWE classes that are commonly associated with such architecture. For each CWE:\n- Provide CWE ID and name.\n- Explain why this CWE is plausible, referencing specific components.\n- Assign a confidence level (high/medium/low) based on how typical this vulnerability is for this kind of application.\n\nReturn your answer ONLY in the following JSON format:\n{{\n  \"cwe_hypotheses\": [\n    {{\n      \"cwe_id\": \"string\",\n      \"name\": \"string\",\n      \"reasoning\": \"string\",\n      \"confidence\": \"high|medium|low\"\n    }}\n  ]\n}}"

        assistant_content = json.dumps(cwe_hypotheses, indent=2)

        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }

    def build(self, limit: Optional[int] = None):
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file {self.input_file} not found.")

        with open(self.input_file, "r", encoding="utf-8") as fin, \
             open(self.output_file, "a", encoding="utf-8") as fout:

            processed = 0
            for line in fin:
                if limit and processed >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON line")
                    continue

                try:
                    architecture = self._extract_architecture(record)
                except Exception as e:
                    print(f"⚠️ Could not extract architecture: {e}")
                    continue

                try:
                    print(architecture)
                    cwe_hypotheses = self.generator.generate(architecture)
                except Exception as e:
                    print(f"⚠️ Generation failed: {e}")
                    continue

                try:
                    sample = self._create_sample(architecture, cwe_hypotheses)
                except Exception as e:
                    print(f"⚠️ Sample creation failed: {e}")
                    continue

                fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
                fout.flush()
                processed += 1
                print(f"✅ Processed {processed} samples.")
                time.sleep(CONFIG["rate_limit_sleep"])

        print(f"🎉 Done. Total: {processed}. Output: {self.output_file}")

# =========== ЗАПУСК ===========
if __name__ == "__main__":
    if not os.environ.get(CONFIG["api_key_env_var"]):
        print("❌ OPENROUTER_API_KEY not set.")
        exit(1)

    generator = CWEHypothesisGenerator()
    builder = CWEHypothesisDatasetBuilder(
        generator=generator,
        input_file="architecture_samples.jsonl",
        output_file="cwe_hypotheses_samples.jsonl"
    )
    builder.build(limit=None)