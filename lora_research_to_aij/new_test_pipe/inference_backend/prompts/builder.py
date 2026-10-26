
### 2. `prompts/builder.py`

"""
prompts/builder.py — сборка Request из Example + PromptSpec.

Поддерживает два режима:
  - prebuilt=True  : Example.input уже готовая строка промпта (не трогаем);
  - prebuilt=False : подставляем Example.input в user_template, добавляем system,
                     применяем chat template (если передан) или склеиваем текстом.

Запуск — из корня inference_backend/:
    python3 -m scripts.check_prompt
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Optional

from core.schema import Example, PromptSpec, Request

logger = logging.getLogger(__name__)


def _make_request_id(sample_id: str, prompt_text: str) -> str:
    """Стабильный request_id: одинаковый sample_id + prompt -> одинаковый id."""
    h = hashlib.sha1(f"{sample_id}||{prompt_text}".encode("utf-8")).hexdigest()[:16]
    return f"req:{h}"


class PromptBuilder:
    def __init__(
        self,
        prompt_spec: PromptSpec,
        chat_template_fn: Optional[Callable[[list[dict], bool], str]] = None,
    ):
        """
        prompt_spec      — PromptSpec из configs/prompts/*.yaml
        chat_template_fn — функция (messages, add_generation_prompt) -> str.
                           Обычно это tokenizer.apply_chat_template с
                           tokenize=False. Если None — fallback на склейку текстом.
        """
        self.spec = prompt_spec
        self.chat_template_fn = chat_template_fn

    # ------------------------------------------------------------
    # Сборка одного Request
    # ------------------------------------------------------------
    def build(self, example: Example) -> Request:
        if example.prebuilt:
            prompt_text = example.input
        else:
            prompt_text = self._render(example)

        request_id = _make_request_id(example.sample_id, prompt_text)
        return Request(
            request_id=request_id,
            sample_id=example.sample_id,
            prompt_text=prompt_text,
            meta={
                "prompt_name": self.spec.name,
                "input_type": example.input_type,
                "target_type": example.target_type,
                "prebuilt": example.prebuilt,
            },
        )

    def build_batch(self, examples: list[Example]) -> list[Request]:
        return [self.build(ex) for ex in examples]

    # ------------------------------------------------------------
    # Внутренняя логика
    # ------------------------------------------------------------
    def _render(self, example: Example) -> str:
        # 1) подставить input в user_template
        user_text = self.spec.user_template.replace("{input}", example.input)

        # 2) собрать messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.spec.system},
            {"role": "user",   "content": user_text},
        ]

        # 3) применить chat template или fallback
        if self.chat_template_fn is not None:
            try:
                return self.chat_template_fn(messages, True)
            except Exception as e:
                logger.warning(
                    f"chat_template_fn failed ({type(e).__name__}: {e}); "
                    f"fallback to plain text"
                )

        # fallback: обычный текст без спецтокенов
        return f"{self.spec.system}\n\n{user_text}\n\n"