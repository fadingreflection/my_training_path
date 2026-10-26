"""
data/normalizer.py — применяет TaskSpec.extractor к сырым записям.

Извлекает:
  - input       (обязателен) — то, что пойдёт в PromptBuilder;
  - target      (опционален) — метка (CWE-XXX) для macro_f1;
  - completion  (опционален) — эталонный ответ (полный JSON-строкой)
                              для сравнения и анализа ошибок.

sample_id:
  1) из TaskSpec.sample_id_expr (если задан в YAML);
  2) иначе record['sample_id'] / record['id'];
  3) иначе auto-хеш от input + target (стабильный, уникальный).

Импорт абсолютный от корня пакета (from core.schema import ...),
потому что normalizer.py лежит в data/, а schema.py — в core/.
Запуск — из корня inference_backend/:
    python3 -m scripts.check_normalize
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.schema import Example, TaskSpec

logger = logging.getLogger(__name__)

CWE_RE = re.compile(r"CWE-?\s*(\d+)", re.IGNORECASE)

DEFAULT_SAMPLE_ID_EXPR = "record.get('sample_id') or record.get('id')"


def normalize_cwe(x: Any) -> str | None:
    if not isinstance(x, str):
        return None
    m = CWE_RE.search(x)
    return f"CWE-{m.group(1)}" if m else None


def _auto_sample_id(input_str: str, target: str | None) -> str:
    """Стабильный id: одинаковый input+target -> одинаковый id."""
    key = f"{input_str}||{target or ''}"
    return "auto:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class Rejected:
    raw: dict[str, Any]
    reason: str


def _safe_eval(expr: str, record: dict[str, Any]) -> tuple[bool, Any]:
    try:
        return True, eval(expr, {"__builtins__": {}}, {"record": record})
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


class Normalizer:
    def __init__(self, task: TaskSpec):
        self.task = task

    def _apply_extractor(self, record: dict[str, Any]) -> tuple[bool, Example | Rejected]:
        # --- input (обязателен) ---
        ok, inp = _safe_eval(self.task.extractor.input, record)
        if not ok:
            return False, Rejected(record, f"input extractor failed: {inp}")
        if not isinstance(inp, str) or not inp.strip():
            return False, Rejected(record, "input is empty or not str")

        # --- target (метка, опционален) ---
        target = None
        if self.task.extractor.target:
            ok, tgt = _safe_eval(self.task.extractor.target, record)
            if not ok:
                return False, Rejected(record, f"target extractor failed: {tgt}")
            if self.task.target.type == "cwe":
                target = normalize_cwe(tgt)
                if target is None:
                    return False, Rejected(record, f"target not a CWE: {tgt!r}")
            else:
                target = tgt

        # --- completion (эталонный ответ, опционален) ---
        completion = None
        if self.task.extractor.completion:
            ok, comp = _safe_eval(self.task.extractor.completion, record)
            if ok and comp is not None:
                if isinstance(comp, str):
                    completion = comp
                else:
                    completion = json.dumps(comp, ensure_ascii=False)
            # если не извлеклось — не режем запись, completion опционален

        # --- sample_id: expr -> fallback -> auto ---
        sid = None
        expr = self.task.sample_id_expr or DEFAULT_SAMPLE_ID_EXPR
        ok, sid_val = _safe_eval(expr, record)
        if ok and sid_val:
            sid = str(sid_val)
        else:
            sid = _auto_sample_id(inp, target)

        return True, Example(
            sample_id=sid,
            input=inp,
            input_type=self.task.input.type,
            target=target,
            target_type=self.task.target.type,
            completion=completion,
            metadata={"source_keys": sorted(record.keys())},
            prebuilt=self.task.extractor.prebuilt,
        )

    def normalize(self, records: list[dict[str, Any]]) -> tuple[list[Example], list[Rejected]]:
        examples, rejected = [], []
        for rec in records:
            ok, result = self._apply_extractor(rec)
            if ok:
                examples.append(result)          # type: ignore[arg-type]
            else:
                rejected.append(result)          # type: ignore[arg-type]
        logger.info(f"Normalized: kept={len(examples)}, rejected={len(rejected)}")
        return examples, rejected

    @staticmethod
    def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out

    @staticmethod
    def save_rejected(rejected: list[Rejected], path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rejected:
                f.write(json.dumps(
                    {"reason": r.reason, "raw_keys": sorted(r.raw.keys()), "raw": r.raw},
                    ensure_ascii=False,
                ) + "\n")