import json
from typing import Any, Dict, List, Optional  # noqa: UP035


class Metrics:
    """Статические методы для работы с JSON и оценки качества ответов."""

    @staticmethod
    def extract_json(text: str) -> Optional[str]:
        """
        Извлекает первый JSON-объект, начинающийся с '{"findings"'.
        Использует балансировку фигурных скобок с учётом строк и экранирования.

        Args:
            text: Строка, содержащая JSON-объект (возможно, с лишним текстом).

        Returns:
            Извлечённая JSON-строка или None, если JSON не найден.
        """
        if not text:
            return None

        start = text.find('{"findings"')
        if start == -1:
            return None

        brace_count = 0
        in_string = False
        escape = False

        for i, ch in enumerate(text[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        json_str = text[start:end]
                        try:
                            json.loads(json_str)
                            return json_str
                        except json.JSONDecodeError:
                            # Если JSON невалидный, продолжаем поиск.
                            pass
        return None

    @staticmethod
    def compute_json_validity(text: str) -> bool:
        """
        Проверяет, содержит ли текст валидный JSON с полем 'findings'.

        Args:
            text: Строка для проверки.

        Returns:
            True, если извлечённый JSON валиден и имеет поле findings.
        """
        json_str = Metrics.extract_json(text)
        if not json_str:
            return False
        try:
            data = json.loads(json_str)
            return "findings" in data and isinstance(data["findings"], list)
        except json.JSONDecodeError:
            return False

    @staticmethod
    def compute_structure_accuracy(responses: List[str]) -> float:
        """
        Вычисляет долю ответов, содержащих валидный JSON с полем findings.

        Args:
            responses: Список строк с ответами.

        Returns:
            Доля валидных ответов в диапазоне [0.0, 1.0].
        """
        if not responses:
            return 0.0
        valid = sum(1 for r in responses if Metrics.compute_json_validity(r))
        return valid / len(responses)

    @staticmethod
    def compute_cwe_coverage(responses: List[Dict[str, Any]]) -> float:
        """
        Вычисляет долю ответов, в которых есть хотя бы один CWE_ID.

        Args:
            responses: Список распарсенных ответов (словарей с ключом 'findings').

        Returns:
            Доля ответов с CWE в диапазоне [0.0, 1.0].
        """
        if not responses:
            return 0.0

        has_cwe_count = 0
        for resp in responses:
            if isinstance(resp, dict):
                findings = resp.get("findings", [])
                for f in findings:
                    if f.get("cwe_id"):
                        has_cwe_count += 1
                        break
        return has_cwe_count / len(responses)