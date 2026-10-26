"""
backends/base.py — интерфейс InferenceBackend.

Контракт:
    generate_batch(list[Request], gen_kwargs_override=None) -> list[RawResponse]

Гарантии:
  - длина ответа == длина запроса;
  - при ошибке на конкретном запросе RawResponse.error заполнен, raw_text="";
  - сырой текст всегда возвращается (даже если JSON невалидный);
  - бэкенд не парсит JSON, не извлекает CWE — это работа Postprocessor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.schema import ModelSpec, RawResponse, Request


class InferenceBackend(ABC):
    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec
        self.gen_kwargs = dict(model_spec.gen_kwargs or {})

    @abstractmethod
    def generate_batch(
        self,
        requests: list[Request],
        gen_kwargs_override: dict | None = None,
    ) -> list[RawResponse]:
        """Сгенерировать ответы для списка Request."""

    def close(self) -> None:
        """Освободить ресурсы (по умолчанию — no-op)."""
        return None