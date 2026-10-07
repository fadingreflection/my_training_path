"""
Анализ функций, где siblings не попали в топ-5 ближайших соседей.

Что делает:
  1. Загружает sibling_misses.json (список function_id).
  2. Для каждой miss-функции печатает flow_text всех трёх температур.
  3. Считает распределение verdict-паттернов внутри misses.
  4. Сравнивает с распределением по всем 200 функциям.
  5. Сохраняет отчёт в sibling_misses_report.json.

Использование:
    python3 analyze_sibling_misses.py              # первые 10 функций, превью 400 символов
    python3 analyze_sibling_misses.py --all        # все 89 функций
    python3 analyze_sibling_misses.py --preview 800
    python3 analyze_sibling_misses.py --verdict-only
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MISSES_PATH = "sibling_misses.json"
FLOWS_PATH = "../flow_harness/v0_outputs.jsonl"
REPORT_PATH = "sibling_misses_report.json"

TEMPERATURES = ["0.0", "0.4", "0.8"]


def load_flows(path):
    """Загружает v0_outputs и индексирует по function_id и uid."""
    by_func = defaultdict(dict)
    by_uid = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        fid = r.get("input_id")
        temp = str(r.get("temperature", "na"))
        uid = f"{fid}_t{temp}"
        by_func[fid][temp] = r
        by_uid[uid] = r
    return by_func, by_uid


def preview(text, n):
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


def print_function(fid, flows_for_func, preview_len):
    """Печатает flow_text трёх температур одной функции."""
    print(f"\n{'=' * 78}")
    print(f"FUNCTION: {fid}")
    print('=' * 78)
    for temp in TEMPERATURES:
        row = flows_for_func.get(temp)
        if not row:
            print(f"\n--- t={temp} --- (missing)")
            continue
        verdict = row.get("verdict", "?")
        text = row.get("flow_text", "")
        print(f"\n--- t={temp}  verdict={verdict}  len={len(text)} ---")
        print(preview(text, preview_len))


def verdict_patterns(misses, by_func):
    """Считает распределение verdict-паттернов внутри misses."""
    patterns = Counter()
    for fid in misses:
        verdicts = []
        for temp in TEMPERATURES:
            row = by_func.get(fid, {}).get(temp)
            verdicts.append(row.get("verdict", "?") if row else "?")
        patterns[tuple(verdicts)] += 1
    return patterns


def all_verdict_patterns(by_func):
    """Считает распределение verdict-паттернов по всем функциям (baseline)."""
    patterns = Counter()
    for fid, temps in by_func.items():
        verdicts = []
        for temp in TEMPERATURES:
            row = temps.get(temp)
            verdicts.append(row.get("verdict", "?") if row else "?")
        patterns[tuple(verdicts)] += 1
    return patterns


def per_temp_verdict(misses, by_func):
    """Распределение вердиктов по температуре внутри misses."""
    counts = {temp: Counter() for temp in TEMPERATURES}
    for fid in misses:
        for temp in TEMPERATURES:
            row = by_func.get(fid, {}).get(temp)
            counts[temp][row.get("verdict", "?") if row else "?"] += 1
    return counts


def length_stats(misses, by_func):
    """Статистика длины flow внутри misses и по всем функциям."""
    miss_lens = []
    for fid in misses:
        for temp in TEMPERATURES:
            row = by_func.get(fid, {}).get(temp)
            if row:
                miss_lens.append(len(row.get("flow_text", "")))

    all_lens = []
    for fid, temps in by_func.items():
        for temp in TEMPERATURES:
            row = temps.get(temp)
            if row:
                all_lens.append(len(row.get("flow_text", "")))

    def stats(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        return {
            "n": len(xs),
            "mean": sum(xs) / len(xs),
            "median": xs[len(xs) // 2],
            "min": xs[0],
            "max": xs[-1],
        }

    return {"misses": stats(miss_lens), "all": stats(all_lens)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="показать все miss-функции")
    parser.add_argument("--preview", type=int, default=400,
                        help="длина превью flow_text")
    parser.add_argument("--verdict-only", action="store_true",
                        help="только статистика вердиктов, без текстов")
    args = parser.parse_args()

    # Загрузка
    misses = json.loads(Path(MISSES_PATH).read_text())
    by_func, _ = load_flows(FLOWS_PATH)

    print(f"Loaded {len(misses)} miss-function ids")
    print(f"Total functions in dataset: {len(by_func)}")

    # --- Распределение verdict-паттернов ---
    print("\n" + "=" * 78)
    print("VERDICT PATTERNS (misses vs all)")
    print("=" * 78)

    miss_patterns = verdict_patterns(misses, by_func)
    all_patterns = all_verdict_patterns(by_func)

    total_miss = sum(miss_patterns.values())
    total_all = sum(all_patterns.values())

    print(f"\n{'pattern':<45} {'misses':>10} {'all':>10}")
    print("-" * 68)
    for pattern, count in miss_patterns.most_common():
        miss_share = count / total_miss if total_miss else 0
        all_share = all_patterns.get(pattern, 0) / total_all if total_all else 0
        pattern_str = " / ".join(pattern)
        print(f"{pattern_str:<45} {count:>4} ({miss_share:>5.1%}) "
              f"{all_patterns.get(pattern, 0):>4} ({all_share:>5.1%})")

    # --- Распределение вердиктов по температуре ---
    print("\n" + "=" * 78)
    print("VERDICT BY TEMPERATURE (within misses)")
    print("=" * 78)

    per_temp = per_temp_verdict(misses, by_func)
    for temp in TEMPERATURES:
        c = per_temp[temp]
        total = sum(c.values())
        print(f"\nt={temp}:")
        for verdict, count in c.most_common():
            print(f"  {verdict:<15} {count:>4} ({count/total:>5.1%})")

    # --- Длина flow ---
    print("\n" + "=" * 78)
    print("FLOW LENGTH STATS")
    print("=" * 78)
    lens = length_stats(misses, by_func)
    print(f"\nMisses: {json.dumps(lens['misses'], indent=2)}")
    print(f"\nAll:    {json.dumps(lens['all'], indent=2)}")

    # --- Тексты ---
    if not args.verdict_only:
        show_n = len(misses) if args.all else min(10, len(misses))
        print("\n" + "=" * 78)
        print(f"FLOW TEXTS (showing {show_n} of {len(misses)})")
        print("=" * 78)
        for fid in misses[:show_n]:
            print_function(fid, by_func.get(fid, {}), args.preview)

    # --- Сохранение отчёта ---
    report = {
        "n_misses": len(misses),
        "n_functions_total": len(by_func),
        "miss_share": len(misses) / len(by_func) if by_func else 0,
        "verdict_patterns_misses": {
            " / ".join(k): v for k, v in miss_patterns.most_common()
        },
        "verdict_patterns_all": {
            " / ".join(k): v for k, v in all_patterns.most_common()
        },
        "verdict_by_temperature": {
            temp: dict(per_temp[temp]) for temp in TEMPERATURES
        },
        "length_stats": lens,
    }
    Path(REPORT_PATH).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
