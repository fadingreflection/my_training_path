"""
backends/hf_backend.py — HF Transformers backend (offline, без vLLM).

Особенности:
  - загрузка модели через from_pretrained, device_map="auto";
  - опционально 4-bit (QLoRA-style) через backend_kwargs.load_in_4bit;
  - батчинг с left-padding (padding_side="left"), чтобы генерация была корректной;
  - декодируются только новые токены (не промпт);
  - latency_ms — на батч, поделённая на размер батча.

Использование:
    from backends import get_backend
    backend = get_backend(model_spec)
    responses = backend.generate_batch(requests)
"""
from __future__ import annotations

import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.schema import ModelSpec, RawResponse, Request
from .base import InferenceBackend

logger = logging.getLogger(__name__)


class HFBackend(InferenceBackend):
    def __init__(self, model_spec: ModelSpec, tokenizer=None):
        super().__init__(model_spec)
        self.batch_size = int(model_spec.backend_kwargs.get("batch_size", 1))
        self.load_in_4bit = bool(model_spec.backend_kwargs.get("load_in_4bit", False))
        self.max_input_length = int(model_spec.backend_kwargs.get("max_input_length", 2048))

        self._provided_tokenizer = tokenizer
        self.tokenizer = None
        self.model = None
        self._load()

    # ------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------
    def _load(self) -> None:
        path = self.model_spec.path
        if not path:
            raise ValueError("HFBackend requires ModelSpec.path")

        # --- tokenizer ---
        if self._provided_tokenizer is not None:
            self.tokenizer = self._provided_tokenizer
        else:
            logger.info(f"Loading tokenizer from {path}")
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left-padding обязателен для батчевой генерации decoder-only
        self.tokenizer.padding_side = "left"

        # --- model ---
        logger.info(f"Loading model from {path} (4bit={self.load_in_4bit})")
        qconfig = None
        if self.load_in_4bit:
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=qconfig,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("HF model loaded")

    # ------------------------------------------------------------
    # Генерация
    # ------------------------------------------------------------
    def generate_batch(
        self,
        requests: list[Request],
        gen_kwargs_override: dict | None = None,
    ) -> list[RawResponse]:
        if not requests:
            return []

        merged = dict(self.gen_kwargs)
        if gen_kwargs_override:
            merged.update(gen_kwargs_override)
        merged.setdefault("do_sample", False)
        merged.setdefault("max_new_tokens", 768)
        merged["pad_token_id"] = self.tokenizer.pad_token_id

        responses: list[RawResponse] = []
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i:i + self.batch_size]
            responses.extend(self._generate_chunk(chunk, merged))
        return responses

    def _generate_chunk(
        self,
        chunk: list[Request],
        gen_kwargs: dict,
    ) -> list[RawResponse]:
        prompts = [r.prompt_text for r in chunk]
        t0 = time.time()

        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            logger.error(f"Generation failed: {e}")
            return [
                RawResponse(
                    request_id=r.request_id,
                    sample_id=r.sample_id,
                    raw_text="",
                    backend="hf",
                    model=self.model_spec.name,
                    latency_ms=elapsed,
                    error=f"{type(e).__name__}: {e}",
                )
                for r in chunk
            ]

        elapsed = (time.time() - t0) * 1000

        # attention_mask.sum() — реальная длина промпта (без left-pad)
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        responses: list[RawResponse] = []
        for j, req in enumerate(chunk):
            actual_in = int(input_lengths[j])
            new_tokens = outputs[j, actual_in:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            n_out = int(new_tokens.shape[0])

            last_token = new_tokens[-1].item() if n_out > 0 else None
            finish = "stop" if last_token == self.tokenizer.eos_token_id else "length"

            responses.append(RawResponse(
                request_id=req.request_id,
                sample_id=req.sample_id,
                raw_text=text,
                backend="hf",
                model=self.model_spec.name,
                latency_ms=elapsed / len(chunk),
                tokens_in=actual_in,
                tokens_out=n_out,
                finish_reason=finish,
            ))
        return responses

    def close(self) -> None:
        try:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass