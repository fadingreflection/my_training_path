"""
diagnose_clusters.py

Join mechanism labels with flows/scores + диагностика кластеров.

Проверяет:
  1. Распределение кластеров.
  2. Crosstab: cluster × verdict.
  3. Consistency в тройках (все 3 flow одной функции → один кластер?).
  4. Парные переходы между кластерами внутри функции.
  5. Cluster × project.
  6. Cluster × temperature.
  7. Length по кластерам.
  8. Cluster × gold CWE family (если scores доступны).
"""
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    flows = load_jsonl("../flow_harness/v0_outputs.jsonl")
    labels = load_jsonl("mechanism_labels_vuln.jsonl")

    # --- Join ---
    flow_by_key = {(f["input_id"], f["temperature"]): f for f in flows}
    joined = []
    for r in labels:
        input_id, temp_s = r["flow_id"].rsplit("_t", 1)
        key = (input_id, float(temp_s))
        if key not in flow_by_key:
            continue
        f = flow_by_key[key]
        joined.append({
            "input_id": input_id,
            "temperature": f["temperature"],
            "cluster_id": r["cluster_id"],
            "verdict": f["verdict"],
            "flow_text": f["flow_text"],
            "len": len(f["flow_text"]),
        })

    print(f"Labels: {len(labels)}, matched: {len(joined)}\n")

    # --- 1. Распределение кластеров ---
    print("=" * 60)
    print("1. Распределение кластеров")
    print("=" * 60)
    dist = Counter(r["cluster_id"] for r in joined)
    total = len(joined)
    for cl, cnt in sorted(dist.items()):
        print(f"  cluster {cl:>3}: {cnt:>4} ({cnt/total:>5.1%})")

    # --- 2. Cluster × verdict ---
    print("\n" + "=" * 60)
    print("2. Cluster × verdict")
    print("=" * 60)
    ct = defaultdict(Counter)
    for r in joined:
        ct[r["cluster_id"]][r["verdict"]] += 1
    verdicts = sorted({r["verdict"] for r in joined})
    header = "  " + "cluster".rjust(8) + "".join(v.rjust(14) for v in verdicts)
    print(header)
    for cl in sorted(ct):
        row = f"  {cl:>8}"
        for v in verdicts:
            row += f"{ct[cl][v]:>14}"
        print(row)

    # --- 3. Consistency в тройках ---
    print("\n" + "=" * 60)
    print("3. Consistency в тройках (все flow функции в одном кластере?)")
    print("=" * 60)
    by_input = defaultdict(list)
    for r in joined:
        by_input[r["input_id"]].append(r["cluster_id"])

    stats = Counter()
    for iid, cls in by_input.items():
        distinct = set(cls)
        if distinct == {-1}:
            stats["all_outliers"] += 1
        elif len(distinct) == 1:
            stats["all_same_cluster"] += 1
        elif -1 in distinct:
            stats["mixed_with_outlier"] += 1
        else:
            stats["fully_split"] += 1

    n = len(by_input)
    for k in ["all_same_cluster", "mixed_with_outlier", "fully_split", "all_outliers"]:
        cnt = stats[k]
        print(f"  {k:>22}: {cnt:>4} ({cnt/n:>5.1%})")

    # --- 4. Парные переходы ---
    print("\n" + "=" * 60)
    print("4. Парные переходы между кластерами внутри функции (top-10)")
    print("=" * 60)
    pairs = Counter()
    for iid, cls in by_input.items():
        clean = sorted({c for c in cls if c != -1})
        for a, b in combinations(clean, 2):
            pairs[(a, b)] += 1
    for (a, b), cnt in pairs.most_common(10):
        print(f"  {a:>3} <-> {b:>3}: {cnt}")

    # --- 5. Cluster × project (если fixed_200 доступен) ---
    inputs_path = Path("fixed_200_inputs.jsonl")
    if inputs_path.exists():
        print("\n" + "=" * 60)
        print("5. Cluster × project (top-3 на кластер)")
        print("=" * 60)
        inputs = load_jsonl(str(inputs_path))
        proj_by_input = {r["id"]: r.get("project", "?") for r in inputs}
        by_cluster_proj = defaultdict(Counter)
        for r in joined:
            if r["cluster_id"] == -1:
                continue
            proj = proj_by_input.get(r["input_id"], "?")
            by_cluster_proj[r["cluster_id"]][proj] += 1
        for cl in sorted(by_cluster_proj):
            top = by_cluster_proj[cl].most_common(3)
            total_cl = sum(by_cluster_proj[cl].values())
            s = "  ".join(f"{p}={c/total_cl:.0%}" for p, c in top)
            print(f"  cluster {cl:>3} (n={total_cl:>4}): {s}")

    # --- 6. Cluster × temperature ---
    print("\n" + "=" * 60)
    print("6. Cluster × temperature")
    print("=" * 60)
    by_cluster_temp = defaultdict(Counter)
    for r in joined:
        if r["cluster_id"] == -1:
            continue
        by_cluster_temp[r["cluster_id"]][r["temperature"]] += 1
    temps = sorted({r["temperature"] for r in joined})
    header = "  " + "cluster".rjust(8) + "".join(f"t={t}".rjust(12) for t in temps)
    print(header)
    for cl in sorted(by_cluster_temp):
        row = f"  {cl:>8}"
        tot = sum(by_cluster_temp[cl].values())
        for t in temps:
            share = by_cluster_temp[cl][t] / tot
            row += f"{share:>12.1%}"
        print(row)

    # --- 7. Length по кластерам ---
    print("\n" + "=" * 60)
    print("7. Length flow по кластерам (mean / median)")
    print("=" * 60)
    lens = defaultdict(list)
    for r in joined:
        lens[r["cluster_id"]].append(r["len"])
    for cl in sorted(lens):
        xs = sorted(lens[cl])
        mean = sum(xs) / len(xs)
        med = xs[len(xs) // 2]
        print(f"  cluster {cl:>3}: n={len(xs):>4}  mean={mean:>7.0f}  median={med:>7}")

    # --- 8. Cluster × gold CWE family (если scores есть) ---
    scores_path = Path("scores.jsonl")
    if scores_path.exists():
        print("\n" + "=" * 60)
        print("8. Cluster × gold CWE family")
        print("=" * 60)
        scores = load_jsonl(str(scores_path))
        fam_by_id = {}
        for s in scores:
            fam = s.get("score", {}).get("gold_family")
            if fam:
                fam_by_id[s["id"]] = fam

        by_cluster_fam = defaultdict(Counter)
        for r in joined:
            fam = fam_by_id.get(r["input_id"])
            if fam and r["cluster_id"] != -1:
                by_cluster_fam[r["cluster_id"]][fam] += 1

        for cl in sorted(by_cluster_fam):
            top = by_cluster_fam[cl].most_common(3)
            tot = sum(by_cluster_fam[cl].values())
            s = "  ".join(f"{f}={c/tot:.0%}" for f, c in top)
            print(f"  cluster {cl:>3} (n={tot:>4}): {s}")

    # --- 9. Cluster × hit_family (если scores есть) ---
    if scores_path.exists():
        print("\n" + "=" * 60)
        print("9. Cluster × hit_family (правильность предсказания)")
        print("=" * 60)
        hit_by_id = {}
        for s in scores:
            hit = s.get("score", {}).get("hit_family")
            if hit is not None:
                hit_by_id[s["id"]] = hit

        by_cluster_hit = defaultdict(lambda: [0, 0])  # [hit, miss]
        for r in joined:
            h = hit_by_id.get(r["input_id"])
            if h is None or r["cluster_id"] == -1:
                continue
            by_cluster_hit[r["cluster_id"]][0 if h else 1] += 1

        print(f"  {'cluster':>8} {'hit':>6} {'miss':>6} {'total':>6} {'hit_rate':>9}")
        for cl in sorted(by_cluster_hit):
            hits, miss = by_cluster_hit[cl]
            tot = hits + miss
            rate = hits / tot if tot else 0
            print(f"  {cl:>8} {hits:>6} {miss:>6} {tot:>6} {rate:>9.2%}")


if __name__ == "__main__":
    main()