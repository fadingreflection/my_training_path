"""
scripts/check_normalize.py — проверка Normalizer на val_eval.jsonl.

Запуск ИЗ КОРНЯ inference_backend/:
    python3 -m scripts.check_normalize

Не запускать как `python3 scripts/check_normalize.py` — тогда Python
добавит в sys.path папку scripts/, а не корень пакета, и `from core...` упадёт.
"""
from collections import Counter

from core.loader import load_task
from core.paths import CONFIGS_DIR, PROJECT_ROOT
from data.normalizer import Normalizer

# пути задаются от PROJECT_ROOT (= inference_backend/), не от cwd
TASK_PATH     = CONFIGS_DIR / "tasks/cwe_extraction.yaml"   # или cwe_prebuilt.yaml
DATA_PATH     = PROJECT_ROOT.parent.parent / "eda/val_eval.jsonl"
REJECTED_PATH = PROJECT_ROOT / "rejected.jsonl"


def main():
    task = load_task(TASK_PATH)
    print(f"Task: {task.name} ({task.description})")
    print(f"  input.type  = {task.input.type}")
    print(f"  target.type = {task.target.type}")
    print(f"  extractor.input  = {task.extractor.input}")
    print(f"  extractor.target = {task.extractor.target}")
    print(f"  prebuilt = {task.extractor.prebuilt}")
    print()

    norm = Normalizer(task)
    records = norm.load_jsonl(DATA_PATH)
    print(f"Loaded {len(records)} raw records from {DATA_PATH}")

    examples, rejected = norm.normalize(records)
    print(f"kept={len(examples)}, rejected={len(rejected)}")

    if rejected:
        print("\nRejection reasons (top-10):")
        print(Counter(r.reason.split(":")[0] for r in rejected).most_common(10))
        norm.save_rejected(rejected, REJECTED_PATH)
        print(f"Saved rejected -> {REJECTED_PATH}")

    if examples:
        print("\nFirst example:")
        print(examples[0])


if __name__ == "__main__":
    main()