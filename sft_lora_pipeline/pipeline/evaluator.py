import json
import logging
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import Metrics

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.log_dir = Path(config.output_dir) / "eval_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_log(self, raw_test_records, sample_size=50, log_file="predictions.jsonl"):
        if not raw_test_records:
            logger.warning("[WARN] No records provided for evaluation.")
            return {"json_accuracy": 0.0, "cwe_accuracy": 0.0, "log_file": ""}

        self.model.eval()
        random.seed(self.config.seed)
        sample = random.sample(raw_test_records, min(len(raw_test_records), sample_size))

        results = []
        valid_json_count = 0
        cwe_correct_count = 0
        total = len(sample)

        for ex in tqdm(sample, desc="Evaluating samples"):
            # Build prompt and target from record
            prompt, target = self._build_prompt_and_target(ex)

            # Generate response
            generated_full = self._generate_response(prompt)

            # Extract and validate JSON
            extracted_json = Metrics.extract_json(generated_full)
            if extracted_json:
                is_valid_json = Metrics.compute_json_validity(extracted_json)
                generated = extracted_json
            else:
                is_valid_json = False
                generated = generated_full

            # Compare CWE
            target_findings = self._extract_findings(target)
            if is_valid_json:
                valid_json_count += 1
                try:
                    gen_data = json.loads(generated)
                    gen_findings = gen_data.get("findings", [])
                    gen_cwe = {f.get("cwe_id") for f in gen_findings if f.get("cwe_id")}
                    target_cwe = {f.get("cwe_id") for f in target_findings if f.get("cwe_id")}
                    if gen_cwe == target_cwe:
                        cwe_correct_count += 1
                except Exception:
                    pass

            results.append({
                "prompt": prompt,
                "generated": generated_full,
                "extracted_json": extracted_json,
                "target": target,
                "is_valid_json": is_valid_json,
            })

        # Save results
        log_path = self.log_dir / log_file
        with open(log_path, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        json_accuracy = valid_json_count / total if total else 0.0
        cwe_accuracy = cwe_correct_count / total if total else 0.0

        logger.info(f"[STATS] JSON validity: {valid_json_count}/{total} = {json_accuracy:.2%}")
        logger.info(f"[STATS] CWE accuracy: {cwe_correct_count}/{total} = {cwe_accuracy:.2%}")
        logger.info(f"[LOG] Results saved to {log_path}")

        return {
            "json_accuracy": json_accuracy,
            "cwe_accuracy": cwe_accuracy,
            "log_file": str(log_path)
        }

    def _build_prompt_and_target(self, record):
        """
        Builds prompt and target from record.
        Supports both old format (input/output) and new format (prompt/completion).
        """
        if "prompt" in record and "completion" in record:
            prompt = record["prompt"]
            target = record["completion"]
        else:
            code = record["input"]["code"]
            context = record["input"]["context"]
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert."},
                {"role": "user", "content": (
                    f"Analyze the code and return ONLY JSON with findings field:\n{code}\n\nContext: {context}"
                )}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            target = json.dumps({"findings": record["output"]["findings"]}, ensure_ascii=False)
        return prompt, target

    def _generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=getattr(self.config, 'max_new_tokens', 512),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_full

    def _extract_findings(self, target):
        if isinstance(target, str):
            try:
                parsed = json.loads(target)
                return parsed.get("findings", [])
            except json.JSONDecodeError:
                return []
        elif isinstance(target, dict):
            return target.get("findings", [])
        return []