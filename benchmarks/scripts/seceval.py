#!/usr/bin/env python3
"""
Оценка SecEval с использованием vLLM для максимальной скорости.
Поддерживает тензорный параллелизм на двух A100.
"""

VLLM_USE_TRITON_FLASH_ATTN=0

import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Путь к папке с моделью")
    parser.add_argument("--questions_file", required=True, help="Путь к questions.json")
    parser.add_argument("--output", required=True, help="Куда сохранить результаты (JSON)")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча для инференса (vLLM обрабатывает пачками)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Максимум генерируемых токенов")
    parser.add_argument("--temperature", type=float, default=0.0, help="Температура (0 = жадный поиск)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Количество GPU для тензорного параллелизма")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Загружаем вопросы
    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)  # список dict с ключами "question" и "answer"
    
    print(f"Загружено {len(questions)} вопросов.")
    
    # 2. Инициализируем vLLM
    print("Загрузка модели через vLLM...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",          # или "float16", если нужно
        trust_remote_code=True,
        max_model_len=4096,        # можно увеличить до 32768, если нужно
        gpu_memory_utilization=0.9, # используем 90% памяти
        enforce_eager=True,
        # model_impl="Qwen3_5ForConditionalGeneration",
    )
    print("Модель загружена.")
    
    # 3. Параметры генерации
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=None,                 # можно добавить стоп-токены, если есть
    )
    
    # 4. Подготовка промптов
    prompts = [q["question"] for q in questions]
    
    # 5. Батчевая генерация с прогресс-баром
    results = []
    total = len(prompts)
    batch_size = args.batch_size
    
    # Используем tqdm для отслеживания прогресса
    pbar = tqdm(total=total, desc="Генерация ответов")
    
    # vLLM может принимать сразу весь список, но для прогресса разобьём на батчи
    for i in range(0, 16, batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Генерация
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Сохраняем результаты для батча
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            results.append({
                "question": batch_prompts[j],
                "expected": questions[i+j].get("answer", ""),
                "generated": generated_text
            })
        
        pbar.update(len(batch_prompts))
    
    pbar.close()
    
    # 6. Сохраняем результаты
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты сохранены в {output_path}")

if __name__ == "__main__":
    main()