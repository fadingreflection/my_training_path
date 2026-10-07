"""
stub_annotator.py

Генерирует заглушку GT для frame-аннотаций через LLM.
Прогоняет 240 flow дважды (эмуляция двух разметчиков).

Вход:
  - v0_outputs.jsonl            (600 flow)
  - fixed_200_inputs.jsonl      (200 функций с кодом)
  - annotator_prompt.md (промпт из ТЗ)

Выход:
  - annotations_annotator_1.jsonl
  - annotations_annotator_2.jsonl
  - gt_consistency.json
  - selected_240_flows.jsonl
"""

import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

# ==== Конфигурация ====
load_dotenv("/home/afedotova/my_training_path/.env")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "deepseek/deepseek-v4.1-flash"  # ← проверь точный slug на openrouter.ai/models
ANNOTATOR_1 = {"model": MODEL, "temperature": 0.0}
ANNOTATOR_2 = {"model": MODEL, "temperature": 0.6}

PROMPT_PATH = Path("annotator_prompt.md")

SEED = 42
MAX_TOKENS = 4096
N_RETRIES = 3
RETRY_DELAY = 5

SMOKE_TEST = False   # ← выключить для полного прогона

N_SINGLE = 50 if SMOKE_TEST else 150
N_TRIPLE = 0 if SMOKE_TEST else 30



def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


USER_TEMPLATE = """=== Function code ===
```c
{code}
=== Flow explanation ===
{flow_text}

Annotate this flow. Return JSON only."""


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def save_jsonl(path, rows):
    Path(path).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def call_llm(messages, model, temperature):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        # "response_format": {"type": "json_object"},  ← убираем, DeepSeek Flash его не держит
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(N_RETRIES):
        try:
            r = httpx.post(BASE_URL, json=payload, headers=headers, timeout=120.0)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content")
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            # если content пустой — берём reasoning
            text = content if content else reasoning
            if text is None:
                raise RuntimeError(f"No content and no reasoning. Full msg keys: {list(msg.keys())}")
            # сохраняем сырой ответ для диагностики
            with open("raw_llm_responses.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "model": model,
                    "temperature": temperature,
                    "content_len": len(content or ""),
                    "reasoning_len": len(reasoning or ""),
                    "text_preview": text[:200],
                }, ensure_ascii=False) + "\n")
            return text
        except Exception as e:
            last_err = e
            time.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError(f"LLM call failed after {N_RETRIES} retries: {last_err}")

def select_240_flows(outputs, n_single=N_SINGLE, n_triple=N_TRIPLE, seed=SEED):
    rng = random.Random(seed)

    by_func = defaultdict(list)
    for row in outputs:
        by_func[row["input_id"]].append(row)

    function_ids = list(by_func.keys())
    need = n_single + n_triple
    if len(function_ids) < need:
        raise ValueError(f"Need {need} functions, got {len(function_ids)}")

    rng.shuffle(function_ids)
    single_funcs = function_ids[:n_single]
    triple_funcs = function_ids[n_single:n_single + n_triple]

    selected = []

    for fid in single_funcs:
        flows = sorted(by_func[fid], key=lambda r: r.get("temperature", 0))
        selected.append(flows[0])

    for fid in triple_funcs:
        flows = sorted(by_func[fid], key=lambda r: r.get("temperature", 0))
        selected.extend(flows[:3])

    rng.shuffle(selected)
    return selected


def annotate_all(flows, code_by_id, annotator_cfg, system_prompt, user_template, name):
    results, errors = [], []

    for i, flow in enumerate(flows, 1):
        fid = flow["input_id"]
        code = code_by_id.get(fid, "")
        if not code:
            errors.append({"flow_id": flow["input_id"], "error": "no code"})
            continue

        user_content = user_template.format(
            code=code,
            flow_text=flow["flow_text"],
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = call_llm(
                messages,
                annotator_cfg["model"],
                annotator_cfg["temperature"],
            )
            parsed = json.loads(strip_json_fences(raw))
            parsed["flow_id"] = flow["input_id"]
            results.append(parsed)
        except Exception as e:
            errors.append({"flow_id": flow["input_id"], "error": str(e)})

        if i % 20 == 0:
            print(f"[{name}] {i}/{len(flows)} done, {len(errors)} errors")

    print(f"[{name}] finished: {len(results)} ok, {len(errors)} errors")
    return results, errors


def token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def per_field_agreement(ann1, ann2, threshold=0.5):
    by_id_1 = {r["flow_id"]: r for r in ann1}
    by_id_2 = {r["flow_id"]: r for r in ann2}
    common = set(by_id_1) & set(by_id_2)

    fields = ["source", "operation", "missing_check", "sink", "consequence", "mechanism_text"]
    counts = {f: 0 for f in fields}
    for fid in common:
        for f in fields:
            if token_overlap(by_id_1[fid].get(f, ""), by_id_2[fid].get(f, "")) >= threshold:
                counts[f] += 1

    n = len(common)
    return {f: counts[f] / n if n else 0.0 for f in fields}, n


def main():
    print(f"Mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'}")
    print(f"Model: {MODEL}")
    print(f"Temperatures: {ANNOTATOR_1['temperature']} / {ANNOTATOR_2['temperature']}\n")

    system_prompt = load_system_prompt(PROMPT_PATH)
    print(f"System prompt: {len(system_prompt)} chars")

    print("Loading data...")
    outputs = load_jsonl("../flow_harness/v0_outputs.jsonl")
    inputs = load_jsonl("../flow_harness/fixed_200_inputs.jsonl")
    code_by_id = {row["id"]: row["code"] for row in inputs}
    print(f"Loaded {len(outputs)} flow, {len(code_by_id)} functions")

    selected = select_240_flows(outputs)
    print(f"Selected {len(selected)} flow ({N_SINGLE} single + {N_TRIPLE}×3 triple)")

    prefix = "test_" if SMOKE_TEST else ""
    save_jsonl(f"{prefix}selected_240_flows.jsonl", selected)

    print("\n=== Annotator 1 ===")
    ann1, err1 = annotate_all(
        selected, code_by_id, ANNOTATOR_1,
        system_prompt, USER_TEMPLATE, "annotator_1",
    )
    save_jsonl(f"{prefix}annotations_annotator_1.jsonl", ann1)
    save_jsonl(f"{prefix}annotations_annotator_1.errors.jsonl", err1)

    print("\n=== Annotator 2 ===")
    ann2, err2 = annotate_all(
        selected, code_by_id, ANNOTATOR_2,
        system_prompt, USER_TEMPLATE, "annotator_2",
    )
    save_jsonl(f"{prefix}annotations_annotator_2.jsonl", ann2)
    save_jsonl(f"{prefix}annotations_annotator_2.errors.jsonl", err2)

    print("\n=== Agreement ===")
    agreement, n_common = per_field_agreement(ann1, ann2)

    report = {
        "smoke_test": SMOKE_TEST,
        "n_selected": len(selected),
        "n_annotator_1": len(ann1),
        "n_annotator_2": len(ann2),
        "n_common": n_common,
        "errors_1": len(err1),
        "errors_2": len(err2),
        "per_field_agreement": agreement,
        "threshold": 0.5,
        "models": {"annotator_1": ANNOTATOR_1, "annotator_2": ANNOTATOR_2},
        "prompts": {
            "system": str(PROMPT_PATH),
            "user_template": "<inline>",
        },
        "seed": SEED,
    }
    Path(f"{prefix}gt_consistency.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nAgreement per field:")
    for f, v in agreement.items():
        print(f"  {f:20s}  {v:.3f}")
    print(f"\nCommon flow: {n_common}")


if __name__ == "__main__":
    main()