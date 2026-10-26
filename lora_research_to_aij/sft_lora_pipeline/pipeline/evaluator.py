import json
import logging
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import parse_response, schema_ok, normalize_cwe, macro_f1

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.log_dir = Path(config.output_dir) / "eval_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_log(self, records, sample_size=None, log_file="predictions.jsonl"):
        if not records:
            logger.warning("[WARN] No records for evaluation.")
            return {"macro_f1": 0.0, "json_validity": 0.0, "schema_ok": 0.0,
                    "think_leak_rate": 0.0}

        self.model.eval()
        random.seed(self.config.seed)

        sample = records if (sample_size is None or sample_size >= len(records)) \
                 else random.sample(records, sample_size)

        results, y_true, y_pred = [], [], []
        n_valid_json = n_schema_ok = n_think_leak = 0

        for ex in tqdm(sample, desc="Evaluating"):
            prompt, target = self._build_prompt_and_target(ex)
            gen = self._generate(prompt)
            leaked = "<think>" in gen.lower()
            if leaked:
                n_think_leak += 1
            obj, _ = parse_response(gen)

            ok_json = obj is not None
            ok_schema = ok_json and schema_ok(obj)
            if ok_json:
                n_valid_json += 1
            if ok_schema:
                n_schema_ok += 1

            t_obj, _ = parse_response(target)
            t_cwe = normalize_cwe(t_obj.get("CWE")) if t_obj else None
            p_cwe = normalize_cwe(obj.get("CWE")) if obj else None
            y_true.append(t_cwe)
            y_pred.append(p_cwe)

            results.append({
                "prompt": prompt,
                "generated": gen,
                "target": target,
                "target_cwe": t_cwe,
                "pred_cwe": p_cwe,
                "json_valid": ok_json,
                "schema_ok": ok_schema,
                "think_leak": leaked,
            })
            torch.cuda.empty_cache()

        m = macro_f1(y_true, y_pred)
        out = {
            "json_validity": n_valid_json / len(sample),
            "schema_ok": n_schema_ok / len(sample),
            "think_leak_rate": n_think_leak / len(sample),
            **m,
        }
        log_path = self.log_dir / log_file
        with open(log_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(
            f"[STATS] json={out['json_validity']:.2%} schema={out['schema_ok']:.2%} "
            f"macro_f1={out['macro_f1']:.4f} acc={out['accuracy']:.4f} "
            f"think_leak={out['think_leak_rate']:.2%}"
        )
        for lab, s in sorted(m["per_class"].items()):
            logger.info(
                f"  {lab:>10}  P={s['precision']:.2f} R={s['recall']:.2f} "
                f"F1={s['f1']:.2f} n={s['support']}"
            )
        return out

    def _build_prompt_and_target(self, record):
        if isinstance(record, dict) and "prompt" in record and "completion" in record:
            return record["prompt"], record["completion"]
        if isinstance(record, dict) and "input" in record and "output" in record:
            code = record["input"]
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user",   "content": f"```c\n{code}\n``` /no_think"},
            ]
            prompt = self._apply_chat(messages)
            return prompt, record["output"]
        if isinstance(record, list):
            prompt = self._apply_chat(record[:-1])
            return prompt, record[-1]["content"]
        raise ValueError(f"Unknown record format: {type(record)}")

    def _apply_chat(self, messages):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a vulnerability-analysis assistant. "
            "Given a C function, return a JSON object with fields: "
            "explanation (string), frame (object with 'pattern' and 'bindings'), "
            "triplet (list of objects with cause/operation/consequence), "
            "bf (list of strings), CWE (string like 'CWE-125'). "
            "Return valid JSON only, no markdown, no reasoning."
        )

    def _generate(self, prompt):
        torch.cuda.empty_cache()
        max_len = getattr(self.config, "max_prompt_length", 2048)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_len,
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=getattr(self.config, "max_new_tokens", 768),
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if "</think>" in text:
            text = text.split("</think>", 1)[1].lstrip()
        return text