"""backends/__init__.py — registry бэкендов."""
from .base import InferenceBackend
from core.schema import ModelSpec


def get_backend(model_spec: ModelSpec, tokenizer=None) -> InferenceBackend:
    """
    Возвращает инстанс бэкенда по model_spec.backend.
    Доступные: hf, vllm, api.
    tokenizer — опционально (если уже загружен, HFBackend переиспользует).
    """
    name = (model_spec.backend or "").lower()
    if name == "hf":
        from .hf_backend import HFBackend
        return HFBackend(model_spec, tokenizer=tokenizer)
    if name == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(model_spec)
    if name == "api":
        from .api_backend import APIBackend
        return APIBackend(model_spec)
    raise ValueError(
        f"Unknown backend: {name!r}. Available: 'hf', 'vllm', 'api'"
    )


__all__ = ["InferenceBackend", "get_backend"]