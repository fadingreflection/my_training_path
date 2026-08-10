import re
import json
import sys
import argparse
from typing import List, Set, Dict, Any
from collections import defaultdict

# ================================================================
# ЗАХАРДКОЖЕННЫЕ ПУТИ (измените под свои)
# ================================================================
INFERENCE_PATH = "/home/afedotova/my_training_path/benchmarks/results/seceval_prompted.json"
GROUND_TRUTH_PATH = "/home/afedotova/my_training_path/benchmarks/data/seceval_questions.json"
# ================================================================

# Берём категории только из поля 'topics'
CATEGORY_FIELDS = ['topics']   # <-- только topics

def parse_generated_answer(generated: str) -> Set[str]:
    """Извлекает множество букв-ответов, игнорируя незакрытый <think>."""
    text = generated
    start_tag = '<think>'
    end_tag = '</think>'
    
    start_pos = text.lower().find(start_tag)
    if start_pos == -1:
        cleaned = text
    else:
        end_pos = text.lower().find(end_tag, start_pos + len(start_tag))
        if end_pos == -1:
            return set()
        cleaned = text[:start_pos] + text[end_pos + len(end_tag):]
    
    match = re.search(r'[A-Z](?:\s*,\s*[A-Z])*', cleaned)
    if match:
        return set(re.findall(r'[A-Z]', match.group()))
    
    letters = re.findall(r'\b([A-Z])\b', cleaned)
    return set(letters)

def parse_expected_answer(expected: str) -> Set[str]:
    parts = [p.strip().upper() for p in expected.split(',') if p.strip()]
    return set(parts)

def is_correct(predicted: Set[str], expected: Set[str]) -> bool:
    return predicted == expected

def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_category_values(item: Dict[str, Any], fields: List[str]) -> List[str]:
    """Извлекает все значения из указанных полей (поддерживает строки и списки)."""
    values = []
    for field in fields:
        val = item.get(field)
        if val is None:
            continue
        if isinstance(val, str):
            if val.strip():
                values.append(val.strip())
        elif isinstance(val, list):
            for v in val:
                if v and isinstance(v, str) and v.strip():
                    values.append(v.strip())
    return values

def main():
    parser = argparse.ArgumentParser(
        description='Оценка ответов модели с агрегацией только по topics.'
    )
    parser.add_argument(
        'inference_file', 
        nargs='?', 
        default=None,
        help='Путь к JSON-файлу с результатами инференса (если не указан, используется захардкоженный путь)'
    )
    parser.add_argument(
        'ground_truth_file', 
        nargs='?', 
        default=None,
        help='Путь к JSON-файлу с исходными вопросами (если не указан, используется захардкоженный путь)'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Выводить список неправильных ID')
    args = parser.parse_args()

    inf_path = args.inference_file if args.inference_file else INFERENCE_PATH
    gt_path = args.ground_truth_file if args.ground_truth_file else GROUND_TRUTH_PATH

    import os
    if not os.path.exists(inf_path):
        print(f"Ошибка: файл инференса не найден: {inf_path}")
        sys.exit(1)
    if not os.path.exists(gt_path):
        print(f"Ошибка: файл исходных данных не найден: {gt_path}")
        sys.exit(1)

    inference_data = load_json(inf_path)
    ground_data = load_json(gt_path)

    if isinstance(inference_data, dict):
        inference_data = [inference_data]
    if isinstance(ground_data, dict):
        ground_data = [ground_data]

    ground_by_id = {}
    for item in ground_data:
        gid = item.get('id')
        if gid:
            ground_by_id[gid] = item

    merged = []
    for inf_item in inference_data:
        inf_id = inf_item.get('id')
        if not inf_id:
            continue
        ground_item = ground_by_id.get(inf_id)
        if not ground_item:
            print(f"Предупреждение: id {inf_id} не найден в исходных данных. Пропускаем.")
            continue

        expected = ground_item.get('answer') or ground_item.get('expected')
        if not expected:
            print(f"Предупреждение: для id {inf_id} нет поля answer/expected. Пропускаем.")
            continue

        merged_item = {
            'id': inf_id,
            'generated': inf_item.get('generated', ''),
            'expected': expected,
            'topics': ground_item.get('topics', []),
        }
        merged.append(merged_item)

    if not merged:
        print("Нет данных для оценки после объединения. Проверьте id.")
        sys.exit(1)

    total = len(merged)
    correct_total = 0
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    wrong_ids = []

    for item in merged:
        predicted = parse_generated_answer(item['generated'])
        expected = parse_expected_answer(item['expected'])
        correct = is_correct(predicted, expected)
        if correct:
            correct_total += 1
        else:
            wrong_ids.append(item['id'])

        # Берём все значения из topics
        cat_values = get_category_values(item, CATEGORY_FIELDS)
        for cat in cat_values:
            category_stats[cat]['total'] += 1
            if correct:
                category_stats[cat]['correct'] += 1

    # Вывод
    print(f"{'='*60}")
    print(f"Всего примеров (после объединения): {total}")
    print(f"Правильных ответов:                  {correct_total}")
    print(f"Общая точность:                      {correct_total / total * 100:.2f}%")

    if category_stats:
        print(f"\n{'='*60}")
        print("Статистика по топикам (поле topics):")
        print(f"{'Топик':<40} | {'Всего':>6} | {'Правильно':>9} | {'Точность':>8}")
        print("-" * 70)
        for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            total_cat = stats['total']
            correct_cat = stats['correct']
            acc_cat = correct_cat / total_cat * 100 if total_cat > 0 else 0
            print(f"{cat:<40} | {total_cat:>6} | {correct_cat:>9} | {acc_cat:>7.2f}%")
    else:
        print("\n⚠️ Поле 'topics' отсутствует или пусто.")

    if args.verbose and wrong_ids:
        print(f"\n{'='*60}")
        print("ID неправильных ответов:")
        for wid in wrong_ids:
            print(f"  {wid}")

if __name__ == "__main__":
    main()