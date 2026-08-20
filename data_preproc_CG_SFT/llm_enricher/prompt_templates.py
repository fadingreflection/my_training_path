def build_enrichment_prompt(raw: dict) -> str:
    """
    Формирует промпт для LLM на основе сырых данных.
    """
    code = raw.get("code", "")[:2500]  # ограничиваем длину
    context = raw.get("context") or "не указан"
    cwe = raw.get("raw_cwe") or "не указан"
    description = raw.get("raw_description") or "нет"
    recommendation = raw.get("raw_recommendation") or "нет"
    severity = raw.get("raw_severity") or "не указана"
    location = raw.get("raw_location") or "не указана"
    lines = raw.get("raw_lines", [])
    lines_str = "\n".join(lines[:5]) if lines else "нет"
    source = raw.get("source", "unknown")

    prompt = f"""
Ты — эксперт по кибербезопасности. На основе предоставленных данных о коде и уязвимости, заполни недостающие поля и верни JSON с findings.

Входные данные:
- Код:
```c
{code}
Контекст: {context}

Известный CWE: {cwe}

Описание: {description}

Рекомендация: {recommendation}

Уровень опасности: {severity}

Локализация: {location}

Уязвимые строки (из датасета):
{lines_str}

Источник данных: {source}

Твоя задача — создать объект findings (список), где каждый finding содержит:

concept_type: "cwe" (если известен CWE), иначе "pattern" или "architectural"

cwe_id: строка (если есть)

title: краткое название

explanation: понятное объяснение проблемы (используй описание, если есть, иначе сгенерируй на основе кода и контекста)

location: где находится проблема (строка или функция)

severity: "high", "medium", "low" (если не указано, выбери на основе описания)

recommendation: конкретная рекомендация по исправлению (если не указана, предложи свою)

Важно:

Не придумывай CWE, если его нет в данных. В таком случае используй concept_type="pattern".

Будь максимально точен и используй только ту информацию, что дана.

Ответ должен быть ТОЛЬКО валидным JSON без дополнительного текста.

Пример формата:
{{
"findings": [
{{
"concept_type": "cwe",
"cwe_id": "CWE-119",
"title": "Improper Restriction of Operations within the Bounds of a Memory Buffer",
"explanation": "Буфер фиксированного размера недостаточен для некоторых входных данных.",
"location": "функция parse_header",
"severity": "high",
"recommendation": "Использовать динамическое выделение памяти."
}}
]
}}
"""
    return prompt