import json
import re
import torch
from typing import Dict, List, Any

class Metrics:
    @staticmethod
    def compute_json_validity(response: str) -> bool:
        """Проверяет, содержит ли ответ валидный JSON с полем findings."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return False
            data = json.loads(json_match.group())
            return "findings" in data and isinstance(data["findings"], list)
        except:
            return False

    @staticmethod
    def compute_structure_accuracy(responses: List[str]) -> float:
        """Доля ответов, содержащих валидный JSON с findings."""
        valid = sum(1 for r in responses if Metrics.compute_json_validity(r))
        return valid / len(responses) if responses else 0.0

    @staticmethod
    def compute_cwe_coverage(responses: List[Dict]) -> float:
        """
        Для заданных ответов (уже распарсенных) считает долю,
        в которых есть CWE_ID (не None).
        """
        if not responses:
            return 0.0
        has_cwe = 0
        for resp in responses:
            findings = resp.get("findings", [])
            for f in findings:
                if f.get("cwe_id"):
                    has_cwe += 1
                    break
        return has_cwe / len(responses)