#!/usr/bin/env python3
"""Compute Frame Stability Score (FSS) with bootstrap confidence intervals.

Primary FSS per pair:
  fss = mean over 5 roles of cosine similarity between role fillers

Token-overlap FSS computed as a robustness check.

Confidence intervals are estimated by cluster bootstrap at the explanation
level (1000 resamples), since pairs from the same explanation are not
independent observations.

Pairs with any empty role in either frame are excluded.
Pairs whose transformation failed correctness verification are excluded.

Reads:
  frames/_index.json
  sample_outputs/verified/paraphrase_correctness.jsonl
  sample_outputs/verified/style_correctness.jsonl
  sample_outputs/paraphrase.jsonl
  sample_outputs/style.jsonl

Writes:
  sample_outputs/fss_per_pair.jsonl
  sample_outputs/fss_per_explanation.json
  sample_outputs/fss_per_role.json
  sample_outputs/fss_summary.json
  sample_outputs/fss_correlations.json
  sample_outputs/fss_skipped.jsonl
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "sample_outputs"
VER_DIR = OUT_DIR / "verified"
FRAMES_INDEX = HERE / "frames" / "_index.json"

ROLES = ["source", "operation", "missing_check", "sink", "consequence"]
CLASSES = ["paraphrase", "style"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

N_BOOT = 1000
BOOT_SEED = 42
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


# ---- Token overlap ----------------------------------------------------------

def tokens(s: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(s or "")}


def token_overlap(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# ---- I/O ---------------------------------------------------------------------

def hash_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_verified() -> dict[tuple[str, str], set[str]]:
    passed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cls in CLASSES:
        path = VER_DIR / f"{cls}_correctness.jsonl"
        if not path.exists():
            print(f"  warning: no {path}")
            continue
        for r in load_jsonl(path):
            if r.get("verdict") != "PASS":
                continue
            fid = r.get("flow_id")
            model = r.get("model_name")
            if fid is None or model is None:
                continue
            passed[(cls, fid)].add(model)
    return passed


def build_pairs() -> list[tuple[str, str, str, str, str]]:
    pairs = []
    for cls in CLASSES:
        for r in load_jsonl(OUT_DIR / f"{cls}.jsonl"):
            pairs.append((
                cls,
                r["flow_id"],
                r["model_name"],
                r["source_text"],
                r["transformed_text"],
            ))
    return pairs


# ---- Stats ------------------------------------------------------------------

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


# ---- Cluster bootstrap -------------------------------------------------------

def cluster_bootstrap_ci(rows: list[dict], stat_per_expl, n_boot: int = N_BOOT,
                         seed: int = BOOT_SEED) -> tuple[float | None, float | None]:
    """Cluster bootstrap at flow_id level.

    stat_per_expl(rows_of_one_explanation) -> float or None.
    Returns (lo, hi) at CI_LEVEL% or (None, None).
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["flow_id"]].append(r)
    fids = list(groups.keys())
    n = len(fids)
    if n < 2:
        return (None, None)
    stats = []
    for _ in range(n_boot):
        sampled = rng.choice(fids, size=n, replace=True)
        vals = []
        for fid in sampled:
            v = stat_per_expl(groups[fid])
            if v is not None:
                vals.append(v)
        if vals:
            stats.append(float(np.mean(vals)))
    if not stats:
        return (None, None)
    alpha = (100.0 - CI_LEVEL) / 2.0
    lo = float(np.percentile(stats, alpha))
    hi = float(np.percentile(stats, 100.0 - alpha))
    return (lo, hi)


def cluster_bootstrap_corr_ci(rows: list[dict], x_fn, y_fn,
                              n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                              method: str = "pearson") -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["flow_id"]].append(r)
    fids = list(groups.keys())
    n = len(fids)
    if n < 2:
        return (None, None)
    stats = []
    for _ in range(n_boot):
        sampled = rng.choice(fids, size=n, replace=True)
        sub = []
        for fid in sampled:
            sub.extend(groups[fid])
        xs = [x_fn(r) for r in sub]
        ys = [y_fn(r) for r in sub]
        if len(xs) < 2:
            continue
        if method == "pearson":
            c = pearson(xs, ys)
        else:
            c = spearman(xs, ys)
        if c is not None:
            stats.append(c)
    if not stats:
        return (None, None)
    alpha = (100.0 - CI_LEVEL) / 2.0
    return (float(np.percentile(stats, alpha)),
            float(np.percentile(stats, 100.0 - alpha)))


# ---- Per-explanation stat functions ------------------------------------------

def per_expl_class_mean(rows_expl: list[dict], cls: str, attr: str):
    vals = [r[attr] for r in rows_expl if r["class"] == cls]
    return float(np.mean(vals)) if vals else None


def per_expl_overall_mean(rows_expl: list[dict], attr: str):
    by_cls: dict[str, list[float]] = defaultdict(list)
    for r in rows_expl:
        by_cls[r["class"]].append(r[attr])
    present = [sum(v) / len(v) for v in by_cls.values()]
    return float(np.mean(present)) if present else None


def per_expl_role_mean(rows_expl: list[dict], role: str, attr: str):
    vals = [r[attr][role] for r in rows_expl]
    return float(np.mean(vals)) if vals else None


def per_expl_class_role_mean(rows_expl: list[dict], cls: str, role: str, attr: str):
    vals = [r[attr][role] for r in rows_expl if r["class"] == cls]
    return float(np.mean(vals)) if vals else None


# ---- Main --------------------------------------------------------------------

def main() -> None:
    if not FRAMES_INDEX.exists():
        raise SystemExit(f"missing {FRAMES_INDEX}")

    frames = {h: v["frame"]
              for h, v in json.loads(FRAMES_INDEX.read_text(encoding="utf-8")).items()}
    verified = load_verified()
    pairs = build_pairs()

    print(f"pairs total:    {len(pairs)}")

    per_pair = []
    skipped = []

    for cls, fid, model, src, tgt in pairs:
        if model not in verified.get((cls, fid), set()):
            skipped.append({"class": cls, "flow_id": fid, "model_name": model,
                            "reason": "verify_fail"})
            continue

        fs = frames.get(hash_text(src))
        ft = frames.get(hash_text(tgt))
        if fs is None or ft is None:
            skipped.append({"class": cls, "flow_id": fid, "model_name": model,
                            "reason": "missing_frame"})
            continue

        empty_src = [r for r in ROLES if not (fs.get(r) or "").strip()]
        empty_tgt = [r for r in ROLES if not (ft.get(r) or "").strip()]
        if empty_src or empty_tgt:
            skipped.append({"class": cls, "flow_id": fid, "model_name": model,
                            "reason": "empty_role"})
            continue

        cos_d = {}
        ov_d = {}
        for role in ROLES:
            a = fs[role].strip()
            b = ft[role].strip()
            cos_d[role] = cos_sim(a, b)
            ov_d[role] = token_overlap(a, b)

        cos_vals = list(cos_d.values())
        ov_vals = list(ov_d.values())

        per_pair.append({
            "flow_id": fid,
            "class": cls,
            "model_name": model,
            "fss_cos": sum(cos_vals) / len(cos_vals),
            "fss_ov": sum(ov_vals) / len(ov_vals),
            "cos": cos_d,
            "ov": ov_d,
        })

    print(f"pairs scored:   {len(per_pair)}")
    print(f"pairs skipped:  {len(skipped)}")
    for reason in ("verify_fail", "missing_frame", "empty_role"):
        n = sum(1 for s in skipped if s["reason"] == reason)
        print(f"  {reason:15s} {n}")

    # ---- per-explanation aggregation ----
    buf_cos = defaultdict(list)
    buf_ov = defaultdict(list)
    for r in per_pair:
        buf_cos[(r["flow_id"], r["class"])].append(r["fss_cos"])
        buf_ov[(r["flow_id"], r["class"])].append(r["fss_ov"])

    per_expl_cos = defaultdict(dict)
    per_expl_ov = defaultdict(dict)
    for (fid, cls), vals in buf_cos.items():
        per_expl_cos[fid][cls] = sum(vals) / len(vals)
    for (fid, cls), vals in buf_ov.items():
        per_expl_ov[fid][cls] = sum(vals) / len(vals)

    per_expl_out = {}
    for fid in set(per_expl_cos) | set(per_expl_ov):
        by_cos = per_expl_cos.get(fid, {})
        by_ov = per_expl_ov.get(fid, {})
        present_cos = [by_cos[c] for c in CLASSES if c in by_cos]
        present_ov = [by_ov[c] for c in CLASSES if c in by_ov]
        per_expl_out[fid] = {
            "fss_cos_paraphrase": by_cos.get("paraphrase"),
            "fss_cos_style": by_cos.get("style"),
            "fss_cos_overall": sum(present_cos) / len(present_cos) if present_cos else None,
            "fss_ov_paraphrase": by_ov.get("paraphrase"),
            "fss_ov_style": by_ov.get("style"),
            "fss_ov_overall": sum(present_ov) / len(present_ov) if present_ov else None,
        }

    # ---- per-role ----
    per_role = {}
    for cls in CLASSES + ["overall"]:
        rows = per_pair if cls == "overall" else [r for r in per_pair if r["class"] == cls]
        if not rows:
            per_role[cls] = {"cos": {r: None for r in ROLES},
                             "ov": {r: None for r in ROLES}}
            continue
        per_role[cls] = {
            "cos": {role: sum(r["cos"][role] for r in rows) / len(rows) for role in ROLES},
            "ov": {role: sum(r["ov"][role] for r in rows) / len(rows) for role in ROLES},
        }

    # ---- bootstrap CIs ----
    print()
    print("computing bootstrap CIs (cluster level = flow_id, n_boot=%d)..." % N_BOOT)

    def ci(stat_fn):
        return cluster_bootstrap_ci(per_pair, stat_fn)

    # primary FSS CIs (cosine)
    ci_cos_par = ci(lambda rs: per_expl_class_mean(rs, "paraphrase", "fss_cos"))
    ci_cos_sty = ci(lambda rs: per_expl_class_mean(rs, "style", "fss_cos"))
    ci_cos_all = ci(lambda rs: per_expl_overall_mean(rs, "fss_cos"))

    # primary FSS CIs (overlap)
    ci_ov_par = ci(lambda rs: per_expl_class_mean(rs, "paraphrase", "fss_ov"))
    ci_ov_sty = ci(lambda rs: per_expl_class_mean(rs, "style", "fss_ov"))
    ci_ov_all = ci(lambda rs: per_expl_overall_mean(rs, "fss_ov"))

    # per-role CIs (overall)
    ci_role_cos = {}
    ci_role_ov = {}
    for role in ROLES:
        ci_role_cos[role] = ci(lambda rs, r=role: per_expl_role_mean(rs, r, "cos"))
        ci_role_ov[role] = ci(lambda rs, r=role: per_expl_role_mean(rs, r, "ov"))

    # per-role per-class CIs
    ci_role_cls_cos = {}
    ci_role_cls_ov = {}
    for cls in CLASSES:
        ci_role_cls_cos[cls] = {}
        ci_role_cls_ov[cls] = {}
        for role in ROLES:
            ci_role_cls_cos[cls][role] = ci(
                lambda rs, c=cls, r=role: per_expl_class_role_mean(rs, c, r, "cos")
            )
            ci_role_cls_ov[cls][role] = ci(
                lambda rs, c=cls, r=role: per_expl_class_role_mean(rs, c, r, "ov")
            )

    # correlation CIs
    def ci_corr(x_fn, y_fn, method="pearson"):
        return cluster_bootstrap_corr_ci(per_pair, x_fn, y_fn, method=method)

    ci_corr_cos_ov_pear = ci_corr(lambda r: r["fss_cos"], lambda r: r["fss_ov"], "pearson")
    ci_corr_cos_ov_spear = ci_corr(lambda r: r["fss_cos"], lambda r: r["fss_ov"], "spearman")

    ci_role_corr = {}
    for role in ROLES:
        ci_role_corr[role] = ci_corr(
            lambda r, rr=role: r["cos"][rr],
            lambda r, rr=role: r["ov"][rr],
            "pearson",
        )

    # ---- summary ----
    def mean_of(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def class_mean_primary(attr, cls):
        return mean_of([r[attr] for r in per_pair if r["class"] == cls])

    n_by_class = {c: sum(1 for r in per_pair if r["class"] == c) for c in CLASSES}

    def pack(mean, ci):
        return {"mean": mean, "ci95": [ci[0], ci[1]]}

    summary = {
        "n_pairs_scored": len(per_pair),
        "n_pairs_skipped": len(skipped),
        "n_explanations": len(per_expl_out),
        "n_pairs_by_class": n_by_class,
        "ci_level": CI_LEVEL,
        "n_boot": N_BOOT,
        "bootstrap_unit": "flow_id (cluster)",

        "fss_cos": {
            "paraphrase": pack(class_mean_primary("fss_cos", "paraphrase"), ci_cos_par),
            "style":      pack(class_mean_primary("fss_cos", "style"), ci_cos_sty),
            "overall":    pack(mean_of([v["fss_cos_overall"] for v in per_expl_out.values()]), ci_cos_all),
        },
        "fss_ov": {
            "paraphrase": pack(class_mean_primary("fss_ov", "paraphrase"), ci_ov_par),
            "style":      pack(class_mean_primary("fss_ov", "style"), ci_ov_sty),
            "overall":    pack(mean_of([v["fss_ov_overall"] for v in per_expl_out.values()]), ci_ov_all),
        },

        "per_role": {
            "cos": {
                role: pack(per_role["overall"]["cos"][role], ci_role_cos[role])
                for role in ROLES
            },
            "ov": {
                role: pack(per_role["overall"]["ov"][role], ci_role_ov[role])
                for role in ROLES
            },
            "per_class": {
                cls: {
                    "cos": {
                        role: pack(per_role[cls]["cos"][role], ci_role_cls_cos[cls][role])
                        for role in ROLES
                    },
                    "ov": {
                        role: pack(per_role[cls]["ov"][role], ci_role_cls_ov[cls][role])
                        for role in ROLES
                    },
                }
                for cls in CLASSES
            },
        },

        "embedding_model": EMBED_MODEL,
        "primary_metric": "fss = mean over 5 roles of cosine similarity on role fillers",
        "secondary_metric": "token overlap on role fillers, as robustness check",
        "exclusion_rule": "pairs with any empty role or failing correctness are excluded",
    }

    # ---- correlations ----
    fss_cos_vals = [r["fss_cos"] for r in per_pair]
    fss_ov_vals = [r["fss_ov"] for r in per_pair]

    correlations = {
        "primary_fss_cos_vs_fss_ov": {
            "pearson": pearson(fss_cos_vals, fss_ov_vals),
            "pearson_ci95": list(ci_corr_cos_ov_pear),
            "spearman": spearman(fss_cos_vals, fss_ov_vals),
            "spearman_ci95": list(ci_corr_cos_ov_spear),
            "n_pairs": len(per_pair),
        },
        "per_role_cos_vs_ov": {
            role: {
                "pearson": pearson(
                    [r["cos"][role] for r in per_pair],
                    [r["ov"][role] for r in per_pair],
                ),
                "pearson_ci95": list(ci_role_corr[role]),
            }
            for role in ROLES
        },
    }

    # ---- write ----
    with (OUT_DIR / "fss_per_pair.jsonl").open("w", encoding="utf-8") as f:
        for r in per_pair:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (OUT_DIR / "fss_skipped.jsonl").open("w", encoding="utf-8") as f:
        for s in skipped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    (OUT_DIR / "fss_per_explanation.json").write_text(
        json.dumps(per_expl_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "fss_per_role.json").write_text(
        json.dumps(per_role, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "fss_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "fss_correlations.json").write_text(
        json.dumps(correlations, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- print ----
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else "n/a"

    def fmt_ci(ci):
        if ci[0] is None or ci[1] is None:
            return "n/a"
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

    print()
    print("=== FSS SUMMARY (with 95% CI, cluster bootstrap) ===")
    print(f"  n_pairs:              {summary['n_pairs_scored']}")
    print(f"  n_explanations:       {summary['n_explanations']}")
    print()

    print("  PRIMARY — cosine similarity:")
    for cls in ("paraphrase", "style", "overall"):
        b = summary["fss_cos"][cls]
        print(f"    {cls:15s} mean={fmt(b['mean'])}  CI={fmt_ci(b['ci95'])}")

    print()
    print("  PRIMARY — token overlap:")
    for cls in ("paraphrase", "style", "overall"):
        b = summary["fss_ov"][cls]
        print(f"    {cls:15s} mean={fmt(b['mean'])}  CI={fmt_ci(b['ci95'])}")

    print()
    print("=== PER-ROLE MEAN (overall, 95% CI) ===")
    header = f"  {'role':15s} {'cos_mean':>10s} {'cos_CI':>18s} {'ov_mean':>10s} {'ov_CI':>18s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for role in ROLES:
        cb = summary["per_role"]["cos"][role]
        ob = summary["per_role"]["ov"][role]
        print(f"  {role:15s} {fmt(cb['mean']):>10s} {fmt_ci(cb['ci95']):>18s} "
              f"{fmt(ob['mean']):>10s} {fmt_ci(ob['ci95']):>18s}")

    print()
    print("=== CORRELATIONS ===")
    c = correlations["primary_fss_cos_vs_fss_ov"]
    print(f"  primary fss_cos vs fss_ov (n={c['n_pairs']}):")
    print(f"    pearson:   {fmt(c['pearson'])}  CI={fmt_ci(c['pearson_ci95'])}")
    print(f"    spearman:  {fmt(c['spearman'])}  CI={fmt_ci(c['spearman_ci95'])}")
    print()
    print("  per-role cosine vs overlap:")
    for role in ROLES:
        b = correlations["per_role_cos_vs_ov"][role]
        print(f"    {role:15s} {fmt(b['pearson'])}  CI={fmt_ci(b['pearson_ci95'])}")

    print()
    print(f"wrote {OUT_DIR}/fss_per_pair.jsonl")
    print(f"wrote {OUT_DIR}/fss_per_explanation.json")
    print(f"wrote {OUT_DIR}/fss_per_role.json")
    print(f"wrote {OUT_DIR}/fss_summary.json")
    print(f"wrote {OUT_DIR}/fss_correlations.json")


if __name__ == "__main__":
    main()