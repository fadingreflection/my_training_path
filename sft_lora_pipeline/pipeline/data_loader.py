import json
import numpy as np
from datasets import Dataset
from typing import Tuple, List, Dict
from transformers import PreTrainedTokenizer

class DataLoader:
    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def load_raw(self, path: str) -> List[Dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def split_data(self, records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        rng = np.random.RandomState(self.config.seed)
        indices = np.arange(len(records))
        rng.shuffle(indices)
        n = len(records)
        val_size = int(n * self.config.validation_split)
        test_size = int(n * self.config.test_split)
        train_indices = indices[:n - val_size - test_size]
        val_indices = indices[n - val_size - test_size : n - test_size]
        test_indices = indices[n - test_size:]
        return (
            [records[i] for i in train_indices],
            [records[i] for i in val_indices],
            [records[i] for i in test_indices]
        )

    def format_conversation(self, example: Dict) -> List[Dict]:
        code = example["input"]["code"]
        context = example["input"]["context"]
        findings = example["output"]["findings"]
        messages = [
            {"role": "system", "content": "Ты — эксперт по кибербезопасности."},
            {"role": "user", "content": f"Проанализируй код:\n{code}\n\nКонтекст: {context}"},
            {"role": "assistant", "content": json.dumps({"findings": findings}, ensure_ascii=False)}
        ]
        return messages

    def prepare_datasets(self, train_records, val_records, test_records):
        def prepare_texts(records):
            texts = []
            for ex in records:
                messages = self.format_conversation(ex)
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append({"text": text})
            return texts

        train_texts = prepare_texts(train_records)
        val_texts = prepare_texts(val_records)
        test_texts = prepare_texts(test_records) if test_records else None

        train_dataset = Dataset.from_list(train_texts)
        val_dataset = Dataset.from_list(val_texts)
        test_dataset = Dataset.from_list(test_texts) if test_texts else None

        # Токенизация с batched=True (правильная работа с батчами)
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        if test_dataset:
            test_dataset = test_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=test_dataset.column_names
            )

        return train_dataset, val_dataset, test_dataset

    def tokenize_function(self, examples):
        """
        Принимает батч (словарь с ключом "text" и списком значений).
        Возвращает токенизированный батч с labels.
        """
        texts = examples["text"]
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized