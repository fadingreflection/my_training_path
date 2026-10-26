"""
scripts/check_prompt.py — проверка PromptBuilder на val_eval.jsonl.

Запуск ИЗ КОРНЯ inference_backend/:
    python3 -m scripts.check_prompt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.loader import load_prompt, load_task
from core.paths import CONFIGS_DIR, PROJECT_ROOT
from data.normalizer import Normalizer
from prompts.builder import PromptBuilder

TASK_PATH   = CONFIGS_DIR / "tasks/cwe_extraction.yaml"
PROMPT_PATH = CONFIGS_DIR / "prompts/cwe_v1.yaml"
DATA_PATH   = PROJECT_ROOT.parent.parent / "eda/val_eval.jsonl"


def main():
    task = load_task(TASK_PATH)
    prompt_spec = load_prompt(PROMPT_PATH)
    print(f"Task:   {task.name}")
    print(f"Prompt: {prompt_spec.name}")
    print(f"  system[:60] = {prompt_spec.system[:60]!r}")
    print(f"  user_template has {{input}}: {'{input}' in prompt_spec.user_template}")
    print(f"  output_schema.required = {prompt_spec.output_schema.get('required')}")
    print()

    norm = Normalizer(task)
    records = norm.load_jsonl(DATA_PATH)
    examples, rejected = norm.normalize(records)
    print(f"Normalized: kept={len(examples)}, rejected={len(rejected)}")

    builder = PromptBuilder(prompt_spec, chat_template_fn=None)   # без токенизатора
    requests = builder.build_batch(examples)
    print(f"Built {len(requests)} requests")

    r = requests[0]
    print("\nFirst request:")
    print(f"  request_id = {r.request_id}")
    print(f"  sample_id  = {r.sample_id}")
    print(f"  meta       = {r.meta}")
    print(f"  prompt_text[:300] = {r.prompt_text[:300]!r}")

    # проверки
    assert len(requests) == len(examples), "requests != examples"
    assert all(r.prompt_text for r in requests), "empty prompt_text"
    assert all("{input}" not in r.prompt_text for r in requests), "unsubstituted {input}"
    print("\nOK: all checks passed")


if __name__ == "__main__":
    main()