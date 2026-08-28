import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class ModelBuilder:
    def __init__(self, config):
        self.config = config
        self.model_path = self.config.model_name_or_path

    def load_tokenizer(self):
        logger.info(f"Loading tokenizer from {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        return tokenizer

    def load_model(self):
        logger.info(f"Loading model from {self.model_path}")
        use_4bit = getattr(self.config, 'load_in_4bit', False)

        if use_4bit:
            logger.info("Using 4-bit quantization (QLoRA)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None
            logger.info("Loading model without quantization (full precision)")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully")
        return model