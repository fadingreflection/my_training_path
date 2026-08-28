import json
import re
from typing import Dict, Iterator, Optional, List
from datasets import load_dataset
from raw_data_schema import make_raw_data


class RawDevignExtractor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = load_dataset(dataset_path, split="train")  

    def extract(self) -> Iterator[Dict]:
        dataset = load_dataset(self.dataset_path, split="train")
        for example in dataset:
            if not example.get("target", False):
                continue
            code = example.get("func", "")
            if not code:
                continue

            project = example.get("project", "")
            context = f"Проект: {project}" if project else ""

            # Извлекаем уязвимые строки через lines и label
            lines = example.get("lines", [])
            label = example.get("label", [])
            vul_lines_list = []
            vul_indices = []
            if lines and label and len(lines) == len(label):
                vul_indices = [i for i, lbl in enumerate(label) if lbl == 1]
                vul_lines_list = [lines[i] for i in vul_indices if i < len(lines)]

            # Формируем raw_location
            location = None
            if vul_indices:
                if len(vul_indices) > 5:
                    location = f"строки {vul_indices[:5]} ... (всего {len(vul_indices)})"
                else:
                    location = f"строки {vul_indices}"

            raw = make_raw_data(
                code=code,
                source="devign",
                context=context,
                raw_location=location,
                raw_lines=vul_lines_list,
                extra={"project": project, "vul_indices": vul_indices}
            )
            yield raw


class RawBigVulExtractor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = load_dataset(dataset_path, split="train")  

    def extract(self) -> Iterator[Dict]:
        dataset = load_dataset(self.dataset_path, split="train")
        for example in dataset:
            code = example.get("func_before", "")
            if not code:
                continue

            cwe = example.get("CWE ID", "")
            if not cwe:
                continue

            commit_msg = example.get("commit_message", "")
            project = example.get("project", "")
            cve = example.get("CVE ID", "")

            context_parts = []
            if project:
                context_parts.append(f"Проект: {project}")
            if cve:
                context_parts.append(f"CVE: {cve}")
            context = "; ".join(context_parts) if context_parts else ""

            # Определяем severity если есть в commit_message
            severity = None
            if commit_msg:
                low = "low" in commit_msg.lower()
                medium = "medium" in commit_msg.lower() or "moderate" in commit_msg.lower()
                if low:
                    severity = "low"
                elif medium:
                    severity = "medium"
                else:
                    severity = "high"  # если не указано, ставим high, но это не берём в raw_severity? – оставим None, чтобы LLM решила

            raw = make_raw_data(
                code=code,
                source="bigvul",
                context=context,
                raw_cwe=cwe,
                raw_title=cwe,  # можно использовать CWE как заголовок
                raw_description=commit_msg,
                raw_severity=None,  # не угадываем
                raw_recommendation=None,
                extra={"project": project, "cve": cve, "commit_message": commit_msg}
            )
            yield raw


class RawCodeVulnExtractor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = load_dataset(dataset_path, split="train")

    def extract(self) -> Iterator[Dict]:
        for example in self.dataset:
            conversations = example.get("conversations", [])
            if not conversations:
                continue

            human_msg = None
            gpt_msg = None
            for msg in conversations:
                if msg.get("from") == "human":
                    human_msg = msg.get("value", "")
                elif msg.get("from") == "gpt":      # <-- ИСПРАВЛЕНО: было "assistant"
                    gpt_msg = msg.get("value", "")

            if not human_msg or not gpt_msg:
                continue

            # Извлекаем код из human_msg
            import re
            code_match = re.search(r"```(?:c|cpp)?\s*(.*?)```", human_msg, re.DOTALL)
            code = code_match.group(1).strip() if code_match else human_msg

            # Парсим ответ gpt
            cwe_match = re.search(r"CWE-(\d+)", gpt_msg)
            cwe_id = f"CWE-{cwe_match.group(1)}" if cwe_match else None

            severity = None
            if "High" in gpt_msg or "высокая" in gpt_msg:
                severity = "high"
            elif "Medium" in gpt_msg or "средняя" in gpt_msg:
                severity = "medium"
            elif "Low" in gpt_msg or "низкая" in gpt_msg:
                severity = "low"

            # Извлекаем рекомендацию
            rec_match = re.search(r"(?:Recommendation|Рекомендация)[:\s]+(.+?)(?:\n|$)", gpt_msg, re.IGNORECASE | re.DOTALL)
            recommendation = rec_match.group(1).strip() if rec_match else None

            # Описание — всё, что до "Recommendation"
            description = gpt_msg[:500]

            raw = make_raw_data(
                code=code,
                source="codevuln",
                context="Синтетический датасет на основе DiverseVul.",
                raw_cwe=cwe_id,
                raw_title=cwe_id if cwe_id else None,
                raw_description=description,
                raw_recommendation=recommendation,
                raw_severity=severity,
                extra={"human": human_msg, "gpt": gpt_msg}
            )
            yield raw