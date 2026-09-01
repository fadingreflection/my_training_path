# llm_enricher/utils.py
import json
import re

def extract_code_from_prompt(prompt):
    """
    Извлекает код из промпта, сгенерированного enricher'ом.
    Ищет текст после "Проанализируй код и верни ТОЛЬКО JSON с полем findings:" до "Контекст:".
    """
    patterns = [
        r"Проанализируй код и верни ТОЛЬКО JSON с полем findings:\n(.*?)\n\nКонтекст:",
        r"findings:\n(.*?)\n\nКонтекст:",
        r"```(?:c|cpp|java|python|go|rust|...)?\n(.*?)\n```",  # если код в маркдауне
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None

def extract_cwe_from_completion(completion):
    """
    Извлекает CWE ID из поля completion (JSON).
    Возвращает строку или None.
    """
    try:
        data = json.loads(completion)
        findings = data.get("findings", [])
        if findings and "cwe_id" in findings[0]:
            return findings[0]["cwe_id"]
    except:
        pass
    return None

def parse_judge_response(response):
    """
    Парсит JSON-ответ судьи (вердикт, correct_cwe, explanation).
    Возвращает кортеж (verdict, correct_cwe, explanation) или (False, None, "ошибка").
    """
    if not response:
        return False, None, "Пустой ответ"
    # Ищем JSON-объект в ответе
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            verdict = data.get("verdict", False)
            correct_cwe = data.get("correct_cwe")
            explanation = data.get("explanation", "")
            return verdict, correct_cwe, explanation
        except:
            pass
    # Если JSON не найден, ищем "Да"/"Нет"
    if "да" in response.lower():
        return True, None, response
    if "нет" in response.lower():
        return False, None, response
    return False, None, "Не удалось определить вердикт"