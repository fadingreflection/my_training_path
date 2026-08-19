import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ModelBuilder:
    def __init__(self, config):
        self.config = config

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            dtype=torch.bfloat16,           # Исправлено: torch_dtype → dtype
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        return model