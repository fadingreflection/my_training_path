import hashlib
import json
import logging
from typing import Dict, List, Tuple  # noqa: UP035

import numpy as np
from datasets import Dataset  # type: ignore

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def load_raw(self, path: str) -> List[Dict]:
        records = []
        limit = getattr(self.config, 'data_limit', 0)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                    if limit and len(records) >= limit:
                        break
        logger.info(f"Loaded {len(records)} records from {path} (limit={limit if limit else 'all'})")
        return records

    @staticmethod
    def _content_key(rec: Dict) -> str:
        prompt = rec.get("prompt") or ""
        completion = rec.get("completion") or ""
        return hashlib.sha1(f"{prompt}||{completion}".encode("utf-8")).hexdigest()

    def drop_content_overlap(self, records: List[Dict], reference: List[Dict]) -> List[Dict]:
        """Удаляет из records строки с точным совпадением prompt+completion с reference."""
        banned = {self._content_key(r) for r in reference}
        return [r for r in records if self._content_key(r) not in banned]

    def split_data(self, records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        if not records:
            raise ValueError("Cannot split empty records list.")

        rng = np.random.RandomState(self.config.seed)
        indices = np.arange(len(records))
        rng.shuffle(indices)

        n = len(records)
        val_size = int(n * self.config.validation_split)
        test_size = int(n * self.config.test_split)

        train_indices = indices[:n - val_size - test_size]
        val_indices = indices[n - val_size - test_size : n - test_size]
        test_indices = indices[n - test_size:] if test_size > 0 else []

        logger.info(f"Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        return (
            [records[i] for i in train_indices],
            [records[i] for i in val_indices],
            [records[i] for i in test_indices]
        )

    def apply_chat_prompts(self, records: List[Dict]) -> List[Dict]:
        """Wrap each prompt as a user turn, matching explain_cwe_map/eval_local.py."""
        wrapped = []
        for rec in records:
            rec = dict(rec)
            messages = [{"role": "user", "content": rec["prompt"]}]
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            try:
                rec["prompt"] = self.tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **kwargs
                )
            except TypeError:
                rec["prompt"] = self.tokenizer.apply_chat_template(messages, **kwargs)
            wrapped.append(rec)
        logger.info("Applied chat template to %d records (enable_thinking=False)", len(wrapped))
        return wrapped

    @staticmethod
    def _sft_rows(records: List[Dict] | None) -> List[Dict] | None:
        if not records:
            return None
        return [{"prompt": r["prompt"], "completion": r["completion"]} for r in records]

    def prepare_datasets(self, train_records, val_records, test_records):
        # Для формата Prompt-Completion оставляем поля prompt и completion.
        # Токенизация будет выполнена внутри SFTTrainer.
        train_rows = self._sft_rows(train_records)
        val_rows = self._sft_rows(val_records)
        test_rows = self._sft_rows(test_records)
        train_dataset = Dataset.from_list(train_rows) if train_rows else None
        val_dataset = Dataset.from_list(val_rows) if val_rows else None
        test_dataset = Dataset.from_list(test_rows) if test_rows else None

        logger.info("Datasets prepared (no tokenization applied yet).")
        return train_dataset, val_dataset, test_dataset