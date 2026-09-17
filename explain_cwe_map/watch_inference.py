#!/usr/bin/env python3
import json
import glob
import time
import os
import sys

def get_latest_run():
    files = glob.glob('explain_cwe_map/runs/*.jsonl')
    files = [f for f in files if not f.endswith('_hits.jsonl')]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def analyze_jsonl(path):
    if not os.path.exists(path):
        return None
    total, hits, raw, canon, fam = 0, 0, 0, 0, 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                total += 1
                sc = d.get('score', {})
                if sc.get('any_level_hit'): hits += 1
                if sc.get('hit_raw'): raw += 1
                if sc.get('hit_canonical'): canon += 1
                if sc.get('hit_family'): fam += 1
            except Exception:
                pass
    pct = (hits / total * 100) if total > 0 else 0.0
    return {
        'path': path,
        'total': total,
        'hits': hits,
        'pct': pct,
        'raw': raw,
        'canonical': canon,
        'family': fam
    }

def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print("=== Inference Logger / Watcher Started ===")
    last_path = None
    last_total = -1

    while True:
        path = get_latest_run()
        if not path:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for inference jsonl runs in explain_cwe_map/runs/...")
        else:
            res = analyze_jsonl(path)
            if res:
                if path != last_path or res['total'] != last_total:
                    print(f"[{time.strftime('%H:%M:%S')}] {os.path.basename(res['path'])} -> {res['total']}/140 samples | hits: {res['hits']} ({res['pct']:.1f}%) | raw: {res['raw']} | canonical: {res['canonical']} | family: {res['family']}")
                    last_path = path
                    last_total = res['total']
        time.sleep(interval)

if __name__ == '__main__':
    main()
