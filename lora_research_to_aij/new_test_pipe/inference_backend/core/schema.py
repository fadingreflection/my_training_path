"""
schema.py — контракты данных для инференс-пайплайна.

Иерархия:
    TaskSpec      — что за задача (вход/таргет/правило извлечения)
    PromptSpec    — как спросить модель (system/user_template/output_schema)
    ModelSpec     — какая модель и как её грузить
    Example       — нормализованная запись (результат Normalizer)
    Request       — готовый промпт для бэкенда
    RawResponse   — сырой ответ бэкенда (всегда сохраняется)
    Extraction    — результат парсинга ответа
    Prediction    — итоговая запись (raw + extraction + label)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================
# TaskSpec — что за задача
# ============================================================
@dataclass
class InputSpec:
    type: str                # "code", "prompt_text", "code_with_context"
    format: str = "c"        # "c", "cpp", "text"


@dataclass
class TargetSpec:
    type: str                # "cwe", "json", "patch"
    format: str = "normalized_cwe"   # "normalized_cwe", "json", "diff"


@dataclass
class ExtractorSpec:
    """
    Правила извлечения из сырой записи.
    Это Python-выражения, вычисляемые в контексте `record` (dict).
    Пример: "record['function']", "record['original_cwes'][0]".
    Если prebuilt=True, input/target — уже готовые строки (prompt/completion).
    """
    input: str
    target: Optional[str] = None
    completion: Optional[str] = None   # правило извлечения эталонного ответа
    prebuilt: bool = False


@dataclass
class TaskSpec:
    name: str
    description: str
    input: InputSpec
    target: TargetSpec
    extractor: ExtractorSpec
    sample_id_expr: str = "record.get('sample_id') or record.get('id')"


# ============================================================
# PromptSpec — как спросить модель
# ============================================================
@dataclass
class PromptSpec:
    name: str
    system: str
    user_template: str                   # "{input}" подставляется
    output_schema: dict[str, Any] = field(default_factory=dict)
    # output_schema: {"required": [...], "field_order": [...], "cwe_field": "CWE"}


# ============================================================
# ModelSpec — какая модель
# ============================================================
@dataclass
class ModelSpec:
    name: str
    backend: str                         # "vllm", "hf", "api"
    path: Optional[str] = None           # локальный путь или HF-id
    provider: Optional[str] = None       # для api: "openai", "anthropic"
    model: Optional[str] = None          # для api: "gpt-4o"
    gen_kwargs: dict[str, Any] = field(default_factory=dict)
    backend_kwargs: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Example — нормализованная запись
# ============================================================
@dataclass
class Example:
    sample_id: str
    input: str
    input_type: str
    target: Optional[str]              # метка (CWE-XXX) для macro_f1
    target_type: Optional[str]
    completion: Optional[str] = None   # эталонный ответ (полный JSON-строкой) из record['completion']
    metadata: dict[str, Any] = field(default_factory=dict)
    prebuilt: bool = False


# ============================================================
# Request — готовый промпт
# ============================================================
@dataclass
class Request:
    request_id: str
    sample_id: str
    prompt_text: str
    meta: dict[str, Any] = field(default_factory=dict)


# ============================================================
# RawResponse — сырой ответ (всегда сохраняется)
# ============================================================
@dataclass
class RawResponse:
    request_id: str
    sample_id: str
    raw_text: str
    backend: str
    model: str
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None


# ============================================================
# Extraction — результат парсинга
# ============================================================
@dataclass
class Extraction:
    parsed: Optional[dict[str, Any]]
    json_valid: bool
    schema_ok: bool
    target_value: Optional[str]          # нормализованный target (CWE и т.п.)
    think_leak: bool = False
    parse_error: Optional[str] = None


# ============================================================
# Prediction — итоговая запись
# ============================================================
@dataclass
class Prediction:
    sample_id: str
    request_id: str
    raw: RawResponse
    extraction: Extraction
    label: Optional[str]                 # GT из Example.target
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "request_id": self.request_id,
            "raw": self.raw.__dict__,
            "extraction": self.extraction.__dict__,
            "label": self.label,
            "metadata": self.metadata,
        }