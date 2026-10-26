"""
prepare_sft.py — полный скрипт подготовки SFT-датасета.

Вход:  JSON (list[dict]) с трейсами модели-учителя.
Выход: train_sft.jsonl, val_sft.jsonl  — формат {"prompt", "completion"}
         (prompt уже с применённым chat template + add_generation_prompt=True)
       sft_full.jsonl, val_eval.jsonl   — с метаданными для анализа

Правила валидации:
  - json_response: непустой dict;
  - explanation: непустая строка;
  - CWE: нормализуется ('125', 'cwe-125', 'CWE-125 (OOB)' -> 'CWE-125');
         с GT (original_cwes) НЕ сравниваем;
  - frame: непустой dict, содержит непустой 'pattern' со всеми 5 ключами
           (source, sink, operation, violated_invariant, consequence —
           непустые строки) и непустой 'bindings';
  - triplet: непустой list[dict];
  - bf: если есть — list;
  - function: непустая строка.

Порядок полей в итоговом JSON-ответе строго:
  explanation -> frame -> triplet -> bf -> CWE.

Формат выхода для SFT:
  {"prompt": <chat template system+user + generation prompt>,
   "completion": <assistant JSON>}
"""

import json
import re
import pandas as pd
from transformers import AutoTokenizer

# ------------------------- настройки -------------------------
INPUT_PATH     = "../golden_set_data/with_valid_abstruction.json"
OUT_DIR        = "."
MODEL_NAME     = "Qwen/Qwen2.5-7B-Instruct"   # <-- тот же, что в config.yaml!
MIN_PER_CLASS  = 30
SEED           = 42
ENABLE_THINKING = False   # для Qwen3 / thinking-моделей; для Qwen2.5 игнорируется

SYSTEM = (
    "You are a vulnerability-analysis assistant. "
    "Given a C function, return a JSON object with fields: "
    "explanation (string), frame (object with 'pattern' and 'bindings'), "
    "triplet (list of objects with cause/operation/consequence), "
    "bf (list of strings), CWE (string like 'CWE-125'). "
    "Return valid JSON only, no markdown, no reasoning."
)

FRAME_PATTERN_KEYS = {"source", "sink", "operation", "violated_invariant", "consequence"}
OUTPUT_FIELD_ORDER = ("explanation", "frame", "triplet", "bf", "CWE")
OUTPUT_REQUIRED    = ("explanation", "frame", "triplet", "CWE")

# =============================================================
# 0. TOKENIZER (для chat template)
# =============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def apply_chat(msgs):
    """Применяет chat template с add_generation_prompt=True (как в DataLoader.apply_chat_prompts)."""
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=ENABLE_THINKING, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, **kwargs)

# =============================================================
# 1. ЗАГРУЗКА
# =============================================================
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

def pick_json_response(rec):
    """Достаёт таргет из json_response -> response -> message.content."""
    jr = rec.get("json_response")
    if isinstance(jr, dict) and jr:
        return jr
    v = rec.get("response")
    if isinstance(v, str):
        try:
            d = json.loads(v)
            if isinstance(d, dict) and d:
                return d
        except json.JSONDecodeError:
            pass
    msg = rec.get("message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str):
        try:
            d = json.loads(content)
            if isinstance(d, dict) and d:
                return d
        except json.JSONDecodeError:
            pass
    return {}

rows = []
for rec in raw:
    meta = rec.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    rows.append({
        "sample_id":        rec.get("sample_id"),
        "function":         rec.get("function"),
        "original_cwes":    rec.get("original_cwes"),
        "metadata_split":   meta.get("split"),
        "metadata_project": meta.get("project"),
        "metadata_lang":    meta.get("lang"),
        "json_response":    pick_json_response(rec),
    })

df = pd.DataFrame(rows)
print(f"Загружено строк: {len(df)}")
print(f"Доля непустых json_response: {df['json_response'].apply(bool).mean():.3f}")

# =============================================================
# 2. ВАЛИДАЦИЯ + СБОРКА ПРИМЕРА
# =============================================================
def normalize_cwe(x):
    if not isinstance(x, str):
        return None
    m = re.search(r"CWE-?\s*(\d+)", x, re.IGNORECASE)
    if not m:
        m = re.fullmatch(r"\s*(\d+)\s*", x)
    return f"CWE-{m.group(1)}" if m else None

def validate_frame(frame):
    if not isinstance(frame, dict) or not frame:
        return False
    pattern = frame.get("pattern")
    if not isinstance(pattern, dict) or not pattern:
        return False
    if not FRAME_PATTERN_KEYS.issubset(pattern.keys()):
        return False
    for k in FRAME_PATTERN_KEYS:
        v = pattern[k]
        if not isinstance(v, str) or not v.strip():
            return False
    bindings = frame.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        return False
    return True

def ordered_output(jr):
    if any(k not in jr for k in OUTPUT_REQUIRED):
        return None
    out = {}
    for k in OUTPUT_FIELD_ORDER:
        if k in jr:
            out[k] = jr[k]
    for k, v in jr.items():
        if k not in out:
            out[k] = v
    return out

def build_example(row):
    jr = row["json_response"]
    if not isinstance(jr, dict) or not jr:
        return None

    expl = jr.get("explanation")
    if not isinstance(expl, str) or not expl.strip():
        return None

    cwe = normalize_cwe(jr.get("CWE"))
    if cwe is None:
        return None
    jr = {**jr, "CWE": cwe}

    if not validate_frame(jr.get("frame")):
        return None

    t = jr.get("triplet")
    if not isinstance(t, list) or not t:
        return None
    if not all(isinstance(x, dict) for x in t):
        return None

    if "bf" in jr and not isinstance(jr["bf"], list):
        return None

    code = row.get("function")
    if not isinstance(code, str) or not code.strip():
        return None

    ordered = ordered_output(jr)
    if ordered is None:
        return None
    completion = json.dumps(ordered, ensure_ascii=False)

    user_content = f"```c\n{code}\n```"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_content},
    ]
    prompt = apply_chat(messages)

    return {
        "sample_id":  row.get("sample_id"),
        "input":      code,
        "output":     completion,
        "prompt":     prompt,
        "completion": completion,
        "CWE":        cwe,
        "split":      row.get("metadata_split"),
        "project":    row.get("metadata_project"),
        "lang":       row.get("metadata_lang"),
    }

examples = [build_example(r) for _, r in df.iterrows()]
examples = [e for e in examples if e is not None]
print(f"Валидных примеров: {len(examples)} из {len(df)}")

if not examples:
    raise SystemExit("Нет валидных примеров — проверьте json_response / function.")

sft = pd.DataFrame(examples)
print("Распределение CWE до балансировки:")
print(sft["CWE"].value_counts())

# =============================================================
# 3. БАЛАНСИРОВКА ПО CWE
# =============================================================
def balance(df, label_col="CWE", min_per_class=MIN_PER_CLASS, seed=SEED):
    counts = df[label_col].value_counts()
    parts = []
    for cls, n in counts.items():
        sub = df[df[label_col] == cls]
        if n > min_per_class:
            sub = sub.sample(n=min_per_class, random_state=seed)
        elif n < min_per_class:
            extra = sub.sample(n=min_per_class - n, replace=True, random_state=seed)
            sub = pd.concat([sub, extra], ignore_index=True)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)

sft_bal = balance(sft)
print("\nПосле балансировки:")
print(sft_bal["CWE"].value_counts())

# =============================================================
# 4. TRAIN / VAL SPLIT
# =============================================================
if sft_bal["split"].notna().any():
    train = sft_bal[sft_bal["split"] == "train"]
    val   = sft_bal[sft_bal["split"].isin(["val", "valid", "validation"])]
    if len(val) == 0:
        val = train.sample(frac=0.1, random_state=SEED)
        train = train.drop(val.index)
else:
    val = sft_bal.sample(frac=0.1, random_state=SEED)
    train = sft_bal.drop(val.index)

# =============================================================
# 5. СОХРАНЕНИЕ (prompt/completion для SFTTrainer)
# =============================================================
def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sft_rows(df_):
    return [{"prompt": r["prompt"], "completion": r["completion"]} for _, r in df_.iterrows()]

save_jsonl(sft_rows(train), f"{OUT_DIR}/train_sft.jsonl")
save_jsonl(sft_rows(val),   f"{OUT_DIR}/val_sft.jsonl")

sft_bal.drop(columns=["prompt"]).to_json(
    f"{OUT_DIR}/sft_full.jsonl", orient="records", lines=True, force_ascii=False)
val.drop(columns=["prompt"]).to_json(
    f"{OUT_DIR}/val_eval.jsonl", orient="records", lines=True, force_ascii=False)

# =============================================================
# 6. ПРОВЕРКА ПОРЯДКА ПОЛЕЙ И КОНСИСТЕНТНОСТИ PROMPT
# =============================================================
def check_outputs(path):
    bad = 0
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            keys = list(json.loads(obj["completion"]).keys())
            expected = [k for k in OUTPUT_FIELD_ORDER if k in keys]
            if keys != expected:
                bad += 1
                if bad <= 3:
                    print(f"  [order] {path}: {keys} != {expected}")
    print(f"{path}: проверено {total}, нарушений порядка — {bad}")
    return bad

def check_prompt_consistency(path, expected_system=SYSTEM):
    """prompt должен содержать system-текст и user-код, оканчиваться на generation prompt."""
    bad = 0
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            prompt = obj["prompt"]
            if expected_system[:60] not in prompt or "```c" not in prompt:
                bad += 1
                if bad <= 3:
                    print(f"  [prompt] {path}: не найден system/код в prompt (len={len(prompt)})")
    print(f"{path}: проверено {total}, нарушений prompt — {bad}")
    return bad

bad_order  = check_outputs(f"{OUT_DIR}/train_sft.jsonl") + check_outputs(f"{OUT_DIR}/val_sft.jsonl")
bad_prompt = check_prompt_consistency(f"{OUT_DIR}/train_sft.jsonl") + \
             check_prompt_consistency(f"{OUT_DIR}/val_sft.jsonl")

print(f"\nСохранено:")
print(f"  train: {len(train)} -> {OUT_DIR}/train_sft.jsonl")
print(f"  val:   {len(val)}   -> {OUT_DIR}/val_sft.jsonl")
print(f"  full:  {len(sft_bal)} -> {OUT_DIR}/sft_full.jsonl")
print(f"  eval:  {len(val)}   -> {OUT_DIR}/val_eval.jsonl")
print(f"  нарушений порядка полей: {bad_order}")
print(f"  нарушений prompt:        {bad_prompt}")