import json
import torch
import logging
from tqdm import tqdm
from .metrics import Metrics

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_json_structure(self, dataset, sample_size=20):
        """Оценивает долю валидных JSON на выборке."""
        self.model.eval()
        sample = dataset.select(range(min(len(dataset), sample_size)))
        valid_count = 0
        for ex in tqdm(sample, desc="Оценка структуры JSON"):
            # Формируем промпт
            code = ex.get("input", {}).get("code", "")
            context = ex.get("input", {}).get("context", "")
            messages = [
                {"role": "system", "content": "Ты — эксперт по кибербезопасности."},
                {"role": "user", "content": f"Проанализируй код:\n{code}\n\nКонтекст: {context}"}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Извлекаем часть после assistant
            assistant_marker = "assistant"
            if assistant_marker in response:
                assistant_response = response.split(assistant_marker)[-1].strip()
            else:
                assistant_response = response
            if Metrics.compute_json_validity(assistant_response):
                valid_count += 1
        accuracy = valid_count / len(sample) if sample else 0.0
        logger.info(f"📊 Доля валидных JSON: {valid_count}/{len(sample)} = {accuracy:.2%}")
        return accuracy

    def evaluate_cwe_coverage(self, dataset, sample_size=50):
        """Оценивает, сколько ответов содержат CWE."""
        # Здесь мы используем сырые данные из dataset, но для SFT у нас есть ответы.
        # Для этой метрики мы можем взять ground truth из dataset и подсчитать долю с CWE.
        # Однако это оценка данных, а не модели. Пока оставим заглушку.
        # Можно считать по сгенерированным ответам, но для этого нужно их сначала сгенерировать.
        pass