from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class Config:
    model_name_or_path: str
    data_path: str
    output_dir: str
    validation_split: float
    test_split: float
    seed: int
    max_seq_length: int
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    use_tensorboard: bool
    report_to: str
    evaluate_on_test: bool
    response_template: str  # <-- НОВОЕ
    resume_from_checkpoint: bool | str = False   # или False, или путь
    evaluate_on_train: bool = True
    early_stopping_patience: int = 0
    early_stopping_threshold: float = 0.0

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)