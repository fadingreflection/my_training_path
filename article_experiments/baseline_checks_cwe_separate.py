#!/usr/bin/env python3
"""Random baseline with CWE-aware pair selection.

Problem with the plain random baseline:
  Cross-pairs are formed by taking source_text from flow_id A and
  transformed_text from flow_id B. But A and B may share the same CWE
  label (e.g. two unrelated buffer overflows). Such pairs do not test
  "different mechanism" — they test "different instance of the same
  class", which inflates the random FSS.

Fix:
  A cross-pair is admitted only if cwe(input_id_a) != cwe(input_id_b),
  where input_id is recovered from the frame hash via _index.json labels.

Also provides a helper to map any frame hash back to its source input_id.

Outputs go to a SEPARATE directory (baselines_cwe_separate/) so that
running this module never overwrites the outputs of baseline_checks.py.

Reads:
  frames/_index.json
  flow_harness/fixed_200_inputs.jsonl
  sample_outputs/verified/paraphrase_correctness.jsonl
  sample_outputs/verified/style_correctness.jsonl
  sample_outputs/paraphrase.jsonl
  sample_outputs/style.jsonl

Writes:
  sample_outputs/baselines_cwe_separate/random_baseline.json

Usage:
  python3 baseline_checks_cwe_separate.py
  python3 baseline_checks_cwe_separate.py --n-trials 2000 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "sample_outputs"
VER_DIR = OUT_DIR / "verified"
BASE_DIR = OUT_DIR / "baselines_cwe_separate"
BASE_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_INDEX = HERE / "frames" / "_index.json"
INPUTS_PATH = PROJECT_ROOT / "flow_harness" / "fixed_200_inputs.jsonl"

ROLES = ["source", "operation", "missing_check", "sink", "consequence"]
CLASSES = ["paraphrase", "style"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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


def load_cwe_map() -> dict[str, str]:
    """input_id -> cwe (from BigVul)."""
    out: dict[str, str] = {}
    for r in load_jsonl(INPUTS_PATH):
        out[r["id"]] = r.get("cwe") or "UNKNOWN"
    return out


# ---- Frame hash -> input_id --------------------------------------------------

def build_hash_to_input_ids() -> dict[str, set[str]]:
    """For every frame hash, collect the set of input_ids that reference it.

    Labels look like:
        par:orig:<input_id>_t0.0
        par:tr:<input_id>_t0.0:<model_name>
        sty:orig:<input_id>_t0.0
        sty:tr:<input_id>_t0.0:<model_name>
    """
    if not FRAMES_INDEX.exists():
        raise SystemExit(f"missing {FRAMES_INDEX}")
    idx = json.loads(FRAMES_INDEX.read_text(encoding="utf-8"))

    hash_to_ids: dict[str, set[str]] = {}
    for h, v in idx.items():
        ids: set[str] = set()
        for label in v.get("labels", []):
            parts = label.split(":")
            if len(parts) < 3:
                continue
            tail = parts[2]                      # "<input_id>_t0.0"
            iid = tail.split("_t")[0]
            if iid and all(c in "0123456789abcdef" for c in iid):
                ids.add(iid)
        hash_to_ids[h] = ids
    return hash_to_ids


def resolve_input_id(frame_hash: str, hash_to_ids: dict[str, set[str]]) -> str | None:
    """Recover the input_id from a frame hash.

    A hash may be referenced by multiple labels, but all of them share
    the same underlying input_id (an explanation belongs to exactly one
    source function). If the set has more than one element, we return
    None and log it — this should not happen in practice.
    """
    ids = hash_to_ids.get(frame_hash)
    if not ids:
        return None
    if len(ids) == 1:
        return next(iter(ids))
    # ambiguity — should not occur
    return None


# ---- Pair filtering (same as compute_fss) ------------------------------------

def filtered_pairs() -> list[tuple[str, str, str, str, str]]:
    frames = load_frames()
    verified = load_verified()
    pairs = load_pairs()
    out = []
    for cls, fid, model, src, tgt in pairs:
        if model not in verified.get((cls, fid), set()):
            continue
        fs = frames.get(hash_text(src))
        ft = frames.get(hash_text(tgt))
        if fs is None or ft is None:
            continue
        if not all((fs.get(r) or "").strip() for r in ROLES):
            continue
        if not all((ft.get(r) or "").strip() for r in ROLES):
            continue
        out.append((cls, fid, model, src, tgt))
    return out


# ---- FSS helpers -------------------------------------------------------------

def fss_from_frames(fa: dict, fb: dict) -> float:
    sims = [cos_sim(fa[r], fb[r]) for r in ROLES]
    return sum(sims) / len(sims)


# ---- Stats -------------------------------------------------------------------

def ci_from_array(stats: list[float]) -> tuple[float | None, float | None]:
    if not stats:
        return (None, None)
    alpha = (100.0 - CI_LEVEL) / 2.0
    return (float(np.percentile(stats, alpha)),
            float(np.percentile(stats, 100.0 - alpha)))


# ---- Main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=2000,
                    help="number of cross-pair trials (default: 2000)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    frames = load_frames()
    cwe_map = load_cwe_map()
    hash_to_ids = build_hash_to_input_ids()

    pairs = filtered_pairs()
    if not pairs:
        raise SystemExit("no filtered pairs")

    # Attach input_id and cwe to every pair
    enriched: list[dict] = []
    ambiguous = 0
    for cls, fid, model, src, tgt in pairs:
        iid_src = resolve_input_id(hash_text(src), hash_to_ids)
        iid_tgt = resolve_input_id(hash_text(tgt), hash_to_ids)
        if iid_src is None or iid_tgt is None:
            ambiguous += 1
            continue
        if iid_src != iid_tgt:
            # a transformed text must belong to the same input as its source
            ambiguous += 1
            continue
        enriched.append({
            "class": cls,
            "flow_id": fid,
            "model": model,
            "input_id": iid_src,
            "cwe": cwe_map.get(iid_src, "UNKNOWN"),
            "src": src,
            "tgt": tgt,
        })

    print(f"filtered pairs:         {len(pairs)}")
    print(f"resolved unambiguously: {len(enriched)}")
    print(f"skipped (unresolved):   {ambiguous}")

    # Real FSS, grouped by flow_id for cluster bootstrap
    real_by_flow: dict[str, list[float]] = defaultdict(list)
    for e in enriched:
        fs = frames[hash_text(e["src"])]
        ft = frames[hash_text(e["tgt"])]
        real_by_flow[e["flow_id"]].append(fss_from_frames(fs, ft))

    # Group by class for cross-pair sampling
    by_class: dict[str, list[dict]] = defaultdict(list)
    for e in enriched:
        by_class[e["class"]].append(e)

    # Random FSS: src from pair a, tgt from pair b, with cwe(a) != cwe(b)
    rand_vals: list[float] = []
    rejected_same_cwe = 0
    rejected_unknown_cwe = 0
    attempts = 0
    max_attempts = args.n_trials * 50

    while len(rand_vals) < args.n_trials and attempts < max_attempts:
        attempts += 1
        a = rng.choice(enriched)
        cands = [b for b in by_class[a["class"]]
                 if b["flow_id"] != a["flow_id"]]
        if not cands:
            continue
        b = rng.choice(cands)

        if a["cwe"] == "UNKNOWN" or b["cwe"] == "UNKNOWN":
            rejected_unknown_cwe += 1
            continue
        if a["cwe"] == b["cwe"]:
            rejected_same_cwe += 1
            continue

        fa = frames[hash_text(a["src"])]
        fb = frames[hash_text(b["tgt"])]
        rand_vals.append(fss_from_frames(fa, fb))

    print()
    print(f"cross-pair attempts:      {attempts}")
    print(f"accepted (CWE differ):    {len(rand_vals)}")
    print(f"rejected (same CWE):      {rejected_same_cwe}")
    print(f"rejected (UNKNOWN CWE):   {rejected_unknown_cwe}")

    # Cluster bootstrap of means and gap
    fids = list(real_by_flow.keys())
    rng_b = np.random.default_rng(args.seed)
    boot_real, boot_rand, boot_gap = [], [], []
    for _ in range(N_BOOT):
        sampled = rng_b.choice(fids, size=len(fids), replace=True)
        real_sub: list[float] = []
        for fid in sampled:
            real_sub.extend(real_by_flow[fid])
        if rand_vals:
            rand_sub = rng_b.choice(rand_vals, size=len(rand_vals),
                                    replace=True).tolist()
        else:
            rand_sub = []
        if not real_sub or not rand_sub:
            continue
        m_real = float(np.mean(real_sub))
        m_rand = float(np.mean(rand_sub))
        boot_real.append(m_real)
        boot_rand.append(m_rand)
        boot_gap.append(m_real - m_rand)

    real_mean = float(np.mean([v for vs in real_by_flow.values() for v in vs]))
    rand_mean = float(np.mean(rand_vals)) if rand_vals else None

    # CWE distribution in the enriched pool
    cwe_distribution = Counter(e["cwe"] for e in enriched)

    result = {
        "config": {
            "n_trials_requested": args.n_trials,
            "n_trials_accepted": len(rand_vals),
            "n_trials_rejected_same_cwe": rejected_same_cwe,
            "n_trials_rejected_unknown_cwe": rejected_unknown_cwe,
            "n_attempts": attempts,
            "seed": args.seed,
            "n_boot": N_BOOT,
            "ci_level": CI_LEVEL,
        },
        "n_pairs_total": len(pairs),
        "n_pairs_resolved": len(enriched),
        "n_pairs_unresolved": ambiguous,
        "n_flows_real": len(real_by_flow),
        "fss_real_mean": real_mean,
        "fss_real_ci95": list(ci_from_array(boot_real)),
        "fss_random_mean": rand_mean,
        "fss_random_ci95": list(ci_from_array(boot_rand)),
        "gap": (real_mean - rand_mean) if rand_mean is not None else None,
        "gap_ci95": list(ci_from_array(boot_gap)),
        "cwe_distribution": dict(cwe_distribution),
    }

    write_json(BASE_DIR / "random_baseline.json", result)

    # print summary
    print()
    print("=== RANDOM BASELINE (CWE-separated) ===")
    print(f"  real FSS:    {result['fss_real_mean']:.4f}  CI={result['fss_real_ci95']}")
    print(f"  random FSS:  {result['fss_random_mean']:.4f}  CI={result['fss_random_ci95']}")
    print(f"  gap:         {result['gap']:.4f}  CI={result['gap_ci95']}")
    print()
    print("  CWE distribution (real pairs):")
    for cwe, n in sorted(cwe_distribution.items(), key=lambda x: -x[1]):
        print(f"    {cwe:20s} {n}")
    print()
    print(f"wrote {BASE_DIR}/random_baseline.json")


if __name__ == "__main__":
    main()