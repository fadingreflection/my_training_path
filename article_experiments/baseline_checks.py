#!/usr/bin/env python3
"""Three controls for testing the frame invariance hypothesis.

1. Random baseline
   FSS on pairs where the transformed_text is taken from a DIFFERENT flow_id.
   If this "random" FSS is close to the real FSS, the metric is not
   discriminating between same-mechanism and different-mechanism pairs.

2. Parser self-consistency
   Re-run the frame parser on a subset of texts at a non-zero temperature
   and compare the new frames to the cached frames. Bounds the contribution
   of parser noise to the observed FSS.

3. Embedding Stability (ES)
   Cosine similarity between raw explanation texts (source_text vs
   transformed_text) using the same encoder. Compared against FSS on the
   same set of pairs. If ES >= FSS and correlates strongly with it, frame
   extraction adds no information over raw embeddings.

All estimates are reported with 95% confidence intervals obtained by
cluster bootstrap at the explanation level (flow_id).

Reads:
  frames/_index.json
  sample_outputs/verified/paraphrase_correctness.jsonl
  sample_outputs/verified/style_correctness.jsonl
  sample_outputs/paraphrase.jsonl
  sample_outputs/style.jsonl

Writes:
  sample_outputs/baselines/random_baseline.json
  sample_outputs/baselines/parser_self_consistency.json
  sample_outputs/baselines/es_baseline.json
  sample_outputs/baselines/summary.json

Usage:
  python3 baseline_checks.py              # all three tests
  python3 baseline_checks.py --skip-parser
  python3 baseline_checks.py --parser-n 100
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "sample_outputs"
VER_DIR = OUT_DIR / "verified"
BASE_DIR = OUT_DIR / "baselines"
BASE_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_INDEX = HERE / "frames" / "_index.json"

load_dotenv(PROJECT_ROOT / ".env")

ROLES = ["source", "operation", "missing_check", "sink", "consequence"]
CLASSES = ["paraphrase", "style"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PARSER_MODEL = "deepseek/deepseek-v4.1-flash"
PARSER_TEMP = 0.3
PARSER_MAX_TOKENS = 8192
PARSER_REASONING_MAX = 512
PARSER_CONCURRENCY = 4
PARSER_MAX_RETRIES = 3

SEED = 42
N_BOOT = 1000
CI_LEVEL = 95.0


# ---- Encoder -----------------------------------------------------------------

print(f"loading encoder: {EMBED_MODEL}")
_encoder = SentenceTransformer(EMBED_MODEL)
_emb_cache: dict[str, np.ndarray] = {}


def embed(text: str) -> np.ndarray:
    key = hashlib.sha256(text.encode()).hexdigest()[:24]
    if key not in _emb_cache:
        v = _encoder.encode(text or " ", normalize_embeddings=True)
        _emb_cache[key] = v
    return _emb_cache[key]


def cos_sim(a: str, b: str) -> float:
    return float(np.dot(embed(a or " "), embed(b or " ")))


# ---- I/O ---------------------------------------------------------------------

def hash_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_verified() -> dict[tuple[str, str], set[str]]:
    passed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cls in CLASSES:
        path = VER_DIR / f"{cls}_correctness.jsonl"
        if not path.exists():
            continue
        for r in load_jsonl(path):
            if r.get("verdict") != "PASS":
                continue
            fid = r.get("flow_id")
            model = r.get("model_name")
            if fid is not None and model is not None:
                passed[(cls, fid)].add(model)
    return passed


def load_pairs() -> list[tuple[str, str, str, str, str]]:
    pairs: list[tuple[str, str, str, str, str]] = []
    for cls in CLASSES:
        for r in load_jsonl(OUT_DIR / f"{cls}.jsonl"):
            pairs.append((cls, r["flow_id"], r["model_name"],
                          r["source_text"], r["transformed_text"]))
    return pairs


def load_frames() -> dict[str, dict]:
    if not FRAMES_INDEX.exists():
        raise SystemExit(f"missing {FRAMES_INDEX}")
    idx = json.loads(FRAMES_INDEX.read_text(encoding="utf-8"))
    return {h: v["frame"] for h, v in idx.items()}


def frame_of(frames: dict[str, dict], text: str) -> dict | None:
    return frames.get(hash_text(text))


def is_fully_filled(frame: dict) -> bool:
    return all((frame.get(r) or "").strip() for r in ROLES)


def role_sim_cos(fa: dict, fb: dict) -> dict[str, float]:
    return {r: cos_sim(fa[r], fb[r]) for r in ROLES}


def fss_from_frames(fa: dict, fb: dict) -> float:
    sims = role_sim_cos(fa, fb)
    return sum(sims.values()) / len(ROLES)


# ---- Pair filtering (same as compute_fss) ------------------------------------

def filtered_pairs() -> list[tuple[str, str, str, str, str]]:
    frames = load_frames()
    verified = load_verified()
    pairs = load_pairs()
    out = []
    for cls, fid, model, src, tgt in pairs:
        if model not in verified.get((cls, fid), set()):
            continue
        fs = frame_of(frames, src)
        ft = frame_of(frames, tgt)
        if fs is None or ft is None:
            continue
        if not is_fully_filled(fs) or not is_fully_filled(ft):
            continue
        out.append((cls, fid, model, src, tgt))
    return out


# ---- Stats helpers -----------------------------------------------------------

def pearson(xs, ys):
    if len(xs) < 2:
        return None
    x, y = np.array(xs, float), np.array(ys, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    x, y = np.array(xs, float), np.array(ys, float)
    if x.std() == 0 or y.std() == 0:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def ci_from_array(stats: list[float]) -> tuple[float | None, float | None]:
    if not stats:
        return (None, None)
    alpha = (100.0 - CI_LEVEL) / 2.0
    return (float(np.percentile(stats, alpha)),
            float(np.percentile(stats, 100.0 - alpha)))


# ---- Test 1: random baseline -------------------------------------------------

def test_random_baseline(n_trials: int = 1000, seed: int = SEED) -> dict:
    frames = load_frames()
    pairs = filtered_pairs()
    if not pairs:
        print("  no filtered pairs — skipping random baseline")
        return {"n": 0}

    by_class: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for cls, fid, model, src, tgt in pairs:
        by_class[cls].append((fid, src, tgt))

    rng = random.Random(seed)

    # real FSS per pair, grouped by flow_id for cluster bootstrap
    real_by_flow: dict[str, list[float]] = defaultdict(list)
    for cls, fid, model, src, tgt in pairs:
        fs = frame_of(frames, src)
        ft = frame_of(frames, tgt)
        if fs is None or ft is None:
            continue
        real_by_flow[fid].append(fss_from_frames(fs, ft))

    # random FSS: source from one flow, transformed text from a different flow
    n_trials_eff = max(n_trials, len(pairs))
    rand_vals: list[float] = []
    for _ in range(n_trials_eff):
        a = rng.choice(pairs)
        cls_a, fid_a, _, src_a, _ = a
        cands = [p for p in by_class[cls_a] if p[0] != fid_a]
        if not cands:
            continue
        _, _, tgt_b = rng.choice(cands)
        fs = frame_of(frames, src_a)
        ft = frame_of(frames, tgt_b)
        if fs is None or ft is None:
            continue
        rand_vals.append(fss_from_frames(fs, ft))

    # cluster bootstrap of means and gap
    fids = list(real_by_flow.keys())
    rng_b = np.random.default_rng(seed)
    boot_real, boot_rand, boot_gap = [], [], []
    for _ in range(N_BOOT):
        sampled = rng_b.choice(fids, size=len(fids), replace=True)
        real_sub: list[float] = []
        for fid in sampled:
            real_sub.extend(real_by_flow[fid])

        if rand_vals:
            rand_sub_arr = rng_b.choice(rand_vals, size=len(rand_vals), replace=True)
            rand_sub = rand_sub_arr.tolist()
        else:
            rand_sub = []

        if len(real_sub) == 0 or len(rand_sub) == 0:
            continue

        m_real = float(np.mean(real_sub))
        m_rand = float(np.mean(rand_sub))
        boot_real.append(m_real)
        boot_rand.append(m_rand)
        boot_gap.append(m_real - m_rand)

    real_mean = float(np.mean([v for vs in real_by_flow.values() for v in vs]))
    rand_mean = float(np.mean(rand_vals)) if rand_vals else None

    result = {
        "n_real": sum(len(v) for v in real_by_flow.values()),
        "n_random": len(rand_vals),
        "fss_real_mean": real_mean,
        "fss_real_ci95": list(ci_from_array(boot_real)),
        "fss_random_mean": rand_mean,
        "fss_random_ci95": list(ci_from_array(boot_rand)),
        "gap": (real_mean - rand_mean) if rand_mean is not None else None,
        "gap_ci95": list(ci_from_array(boot_gap)),
        "n_boot": N_BOOT,
    }
    write_json(BASE_DIR / "random_baseline.json", result)
    return result


# ---- Test 2: parser self-consistency -----------------------------------------

FRAME_PROMPT_PATH = HERE / "extract_frames.py"
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_frame_prompt() -> str:
    src = FRAME_PROMPT_PATH.read_text(encoding="utf-8")
    m = re.search(r'FRAME_PROMPT\s*=\s*"""(.*?)"""', src, re.DOTALL)
    if not m:
        raise SystemExit("could not locate FRAME_PROMPT in extract_frames.py")
    return m.group(1)


def _parse_frame(raw: str) -> dict:
    m = JSON_RE.search(raw)
    if not m:
        raise RuntimeError(f"no JSON: {raw[:120]!r}")
    obj = json.loads(m.group(0))
    return {r: str(obj.get(r, "")).strip() for r in ROLES}


async def _parser_call(client: httpx.AsyncClient, prompt: str, sem) -> dict:
    payload = {
        "model": PARSER_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You extract structured data. Respond only with a single "
                "valid JSON object. No preamble, no explanations, no fences."
            )},
            {"role": "user", "content": prompt},
        ],
        "temperature": PARSER_TEMP,
        "max_tokens": PARSER_MAX_TOKENS,
        "reasoning": {"max_tokens": PARSER_REASONING_MAX},
    }
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }
    async with sem:
        for attempt in range(PARSER_MAX_RETRIES):
            try:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload, headers=headers, timeout=180.0,
                )
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                text = (msg.get("content") or "").strip()
                if not text:
                    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                    m = JSON_RE.search(reasoning)
                    if m:
                        text = m.group(0)
                if not text:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return _parse_frame(text)
            except Exception:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError("parser call failed")


async def _run_parser_batch(texts: list[str]) -> list[dict]:
    prompt_template = _load_frame_prompt()
    sem = asyncio.Semaphore(PARSER_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        tasks = [_parser_call(client, prompt_template.format(text=t), sem) for t in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)


def test_parser_self_consistency(n_samples: int = 100, seed: int = SEED) -> dict:
    if "OPENROUTER_API_KEY" not in os.environ:
        print("  no OPENROUTER_API_KEY — skipping parser self-consistency")
        return {"n": 0}

    frames = load_frames()
    pairs = filtered_pairs()
    if not pairs:
        print("  no filtered pairs — skipping parser self-consistency")
        return {"n": 0}

    rng = random.Random(seed)
    pool = []
    for cls, fid, model, src, tgt in pairs:
        pool.append(src)
        pool.append(tgt)
    pool = list(set(pool))
    rng.shuffle(pool)
    sample = pool[:n_samples]

    print(f"  re-parsing {len(sample)} texts at temp={PARSER_TEMP}")
    try:
        new_frames = asyncio.run(_run_parser_batch(sample))
    except Exception as e:
        print(f"  parser batch failed: {e}")
        return {"n": 0, "error": str(e)}

    per_role: dict[str, list[float]] = {r: [] for r in ROLES}
    overall: list[float] = []

    for text, new_frame in zip(sample, new_frames):
        if isinstance(new_frame, Exception):
            continue
        cached = frames.get(hash_text(text))
        if cached is None:
            continue
        sims = {r: cos_sim(cached[r], new_frame[r]) for r in ROLES}
        for r in ROLES:
            per_role[r].append(sims[r])
        overall.append(sum(sims.values()) / len(ROLES))

    def boot_mean_ci(vals: list[float]) -> tuple[float | None, float | None]:
        if len(vals) < 2:
            return (None, None)
        rng_b = np.random.default_rng(seed)
        arr = np.array(vals, dtype=float)
        boot = [float(np.mean(rng_b.choice(arr, size=len(arr), replace=True)))
                for _ in range(N_BOOT)]
        return ci_from_array(boot)

    result = {
        "n_samples_requested": n_samples,
        "n_samples_used": len(overall),
        "parser_model": PARSER_MODEL,
        "parser_temp": PARSER_TEMP,
        "overall_self_consistency": float(np.mean(overall)) if overall else None,
        "overall_self_consistency_ci95": list(boot_mean_ci(overall)),
        "per_role_self_consistency": {
            r: (float(np.mean(per_role[r])) if per_role[r] else None)
            for r in ROLES
        },
        "per_role_self_consistency_ci95": {
            r: list(boot_mean_ci(per_role[r])) for r in ROLES
        },
        "n_boot": N_BOOT,
    }
    write_json(BASE_DIR / "parser_self_consistency.json", result)
    return result


# ---- Test 3: embedding stability ---------------------------------------------

def test_es_baseline(seed: int = SEED) -> dict:
    pairs = filtered_pairs()
    if not pairs:
        print("  no filtered pairs — skipping ES baseline")
        return {"n": 0}

    frames = load_frames()

    per_pair: list[dict] = []
    for cls, fid, model, src, tgt in pairs:
        fs = frame_of(frames, src)
        ft = frame_of(frames, tgt)
        if fs is None or ft is None:
            continue
        sims = role_sim_cos(fs, ft)
        per_pair.append({
            "flow_id": fid,
            "class": cls,
            "es": cos_sim(src, tgt),
            "fss": sum(sims.values()) / len(ROLES),
        })

    if not per_pair:
        return {"n": 0}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    def boot_mean_ci(vals_by_flow: dict[str, list[float]]) -> tuple[float | None, float | None]:
        if not vals_by_flow:
            return (None, None)
        rng_b = np.random.default_rng(seed)
        fids = list(vals_by_flow.keys())
        boot = []
        for _ in range(N_BOOT):
            sampled = rng_b.choice(fids, size=len(fids), replace=True)
            vals = []
            for fid in sampled:
                vals.extend(vals_by_flow[fid])
            if vals:
                boot.append(float(np.mean(vals)))
        return ci_from_array(boot)

    def boot_corr_ci(xs, ys, method: str = "pearson") -> tuple[float | None, float | None]:
        rng_b = np.random.default_rng(seed)
        n = len(xs)
        boot = []
        for _ in range(N_BOOT):
            idx = rng_b.integers(0, n, n)
            x_sub = [xs[i] for i in idx]
            y_sub = [ys[i] for i in idx]
            c = pearson(x_sub, y_sub) if method == "pearson" else spearman(x_sub, y_sub)
            if c is not None:
                boot.append(c)
        return ci_from_array(boot)

    def aggregate(rows: list[dict]):
        by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"es": [], "fss": []})
        by_flow_es: dict[str, list[float]] = defaultdict(list)
        by_flow_fss: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_class[r["class"]]["es"].append(r["es"])
            by_class[r["class"]]["fss"].append(r["fss"])
            by_flow_es[r["flow_id"]].append(r["es"])
            by_flow_fss[r["flow_id"]].append(r["fss"])

        es_vals = [r["es"] for r in rows]
        fss_vals = [r["fss"] for r in rows]

        def two_level(by_flow: dict[str, list[float]]) -> float | None:
            per_expl = [float(np.mean(v)) for v in by_flow.values() if v]
            return float(np.mean(per_expl)) if per_expl else None

        es_two = two_level(by_flow_es)
        fss_two = two_level(by_flow_fss)

        result = {
            "n_pairs": len(rows),
            "es_mean_simple": mean(es_vals),
            "fss_mean_simple": mean(fss_vals),
            "es_mean_two_level": es_two,
            "fss_mean_two_level": fss_two,
            "es_minus_fss_simple": (mean(es_vals) - mean(fss_vals)) if es_vals and fss_vals else None,
            "es_minus_fss_two_level": (es_two - fss_two) if es_two is not None and fss_two is not None else None,
            "es_mean_ci95": list(boot_mean_ci(dict(by_flow_es))),
            "fss_mean_ci95": list(boot_mean_ci(dict(by_flow_fss))),
            "correlation_es_vs_fss_pearson": pearson(es_vals, fss_vals),
            "correlation_es_vs_fss_pearson_ci95": list(boot_corr_ci(es_vals, fss_vals, "pearson")),
            "correlation_es_vs_fss_spearman": spearman(es_vals, fss_vals),
            "correlation_es_vs_fss_spearman_ci95": list(boot_corr_ci(es_vals, fss_vals, "spearman")),
        }
        for cls in CLASSES:
            if by_class[cls]["es"] and by_class[cls]["fss"]:
                result[f"es_mean_{cls}"] = mean(by_class[cls]["es"])
                result[f"fss_mean_{cls}"] = mean(by_class[cls]["fss"])
                result[f"es_minus_fss_{cls}"] = (
                    mean(by_class[cls]["es"]) - mean(by_class[cls]["fss"])
                )
        return result

    overall = aggregate(per_pair)
    by_cls = {cls: aggregate([r for r in per_pair if r["class"] == cls]) for cls in CLASSES}

    result = {
        "overall": overall,
        "by_class": by_cls,
        "n_boot": N_BOOT,
    }
    write_json(BASE_DIR / "es_baseline.json", result)
    return result


# ---- Main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parser", action="store_true")
    ap.add_argument("--parser-n", type=int, default=100)
    ap.add_argument("--random-trials", type=int, default=1000)
    args = ap.parse_args()

    pairs = filtered_pairs()
    print(f"filtered pairs: {len(pairs)}")
    print()

    summary: dict = {}

    print("=== Test 1: random baseline ===")
    r = test_random_baseline(n_trials=args.random_trials)
    summary["random_baseline"] = r
    if r.get("fss_real_mean") is not None:
        print(f"  fss_real_mean:    {r['fss_real_mean']:.4f}  CI={r['fss_real_ci95']}")
        print(f"  fss_random_mean:  {r['fss_random_mean']:.4f}  CI={r['fss_random_ci95']}")
        print(f"  gap:              {r['gap']:.4f}  CI={r['gap_ci95']}")
    print()

    if not args.skip_parser:
        print("=== Test 2: parser self-consistency ===")
        r = test_parser_self_consistency(n_samples=args.parser_n)
        summary["parser_self_consistency"] = r
        if r.get("overall_self_consistency") is not None:
            print(f"  overall:  {r['overall_self_consistency']:.4f}  "
                  f"CI={r['overall_self_consistency_ci95']}")
            for role, v in r.get("per_role_self_consistency", {}).items():
                ci = r["per_role_self_consistency_ci95"].get(role, [None, None])
                if v is not None:
                    print(f"  {role:15s} {v:.4f}  CI={ci}")
                else:
                    print(f"  {role:15s} n/a")
        print()
    else:
        print("=== Test 2: parser self-consistency [skipped] ===\n")

    print("=== Test 3: ES baseline ===")
    r = test_es_baseline()
    summary["es_baseline"] = r
    o = r.get("overall", {})
    if o.get("es_mean_simple") is not None:
        print(f"  es_mean (simple):     {o['es_mean_simple']:.4f}  CI={o['es_mean_ci95']}")
        print(f"  fss_mean (simple):    {o['fss_mean_simple']:.4f}  CI={o['fss_mean_ci95']}")
        print(f"  es - fss (simple):    {o['es_minus_fss_simple']:.4f}")
        print(f"  es_mean (two-level):  {o['es_mean_two_level']:.4f}")
        print(f"  fss_mean (two-level): {o['fss_mean_two_level']:.4f}")
        print(f"  es - fss (two-level): {o['es_minus_fss_two_level']:.4f}")
        print(f"  corr pearson:         {o['correlation_es_vs_fss_pearson']:.4f}  "
              f"CI={o['correlation_es_vs_fss_pearson_ci95']}")
        print(f"  corr spearman:        {o['correlation_es_vs_fss_spearman']:.4f}  "
              f"CI={o['correlation_es_vs_fss_spearman_ci95']}")
    for cls in CLASSES:
        b = r.get("by_class", {}).get(cls, {})
        if b.get("es_mean_simple") is not None:
            print(f"  [{cls}]  es={b['es_mean_simple']:.4f}  "
                  f"fss={b['fss_mean_simple']:.4f}  "
                  f"gap={b['es_minus_fss_simple']:.4f}")
    print()

    write_json(BASE_DIR / "summary.json", summary)
    print(f"wrote {BASE_DIR}/summary.json")


if __name__ == "__main__":
    main()