#!/usr/bin/env python3
import json
import argparse
import requests
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=10, help="Короткие ответы, т.к. нужны только буквы")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt_template", default="/workspace/benchmarks/scripts/prompt_template.txt",
                        help="Путь к файлу с шаблоном промпта")
    return parser.parse_args()

def main():
    args = parse_args()

    # Читаем шаблон
    with open(args.prompt_template, "r", encoding="utf-8") as f:
        template = f.read()

    # Загружаем вопросы
    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    url = f"{args.base_url}/completions"

    total = len(questions)
    for i in tqdm(range(0, total, args.batch_size), desc="Batches"): #change to total
        batch = questions[i:i+args.batch_size]
        # Формируем промпты для каждого вопроса
        prompts = []
        for q in batch:
            # Варианты объединяем в многострочный текст
            choices_text = "\n".join(q.get("choices", []))
            # Подставляем в шаблон
            prompt = template.format(question=q["question"], choices=choices_text)
            prompts.append(prompt)

        payload = {
            "prompt": prompts,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if len(choices) != len(batch):
                if len(choices) == 1:
                    choices = choices * len(batch)
            for j, choice in enumerate(choices):
                generated = choice.get("text", "").strip()
                results.append({
                    "id": batch[j].get("id"),
                    "question": batch[j]["question"],
                    "choices": batch[j].get("choices"),
                    "expected": batch[j].get("answer", ""),
                    "generated": generated
                })
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            for q in batch:
                results.append({
                    "id": q.get("id"),
                    "question": q["question"],
                    "choices": q.get("choices"),
                    "expected": q.get("answer", ""),
                    "generated": error_msg
                })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()