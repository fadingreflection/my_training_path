"""CPU metrics + 95% bootstrap CI over input_id."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

import numpy as np

from normalize import normalize
from verdict import parse_verdict

_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_BACKTICK = re.compile(r"`([^`]+)`")
KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline",
    "int", "long", "register", "restrict", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void",
    "volatile", "while", "bool", "true", "false", "class", "namespace",
    "template", "public", "private", "protected", "try", "catch", "throw",
    "new", "delete", "this", "operator", "friend", "using", "virtual",
    "override", "constexpr", "nullptr", "typename", "sizeof", "vulnerable",
    "safe", "insufficient", "insufficient_info", "the", "a", "an", "is", "are",
    "be", "to", "of", "and", "or", "not", "this", "that", "then", "than",
    "with", "from", "for", "on", "in", "into", "by", "as", "at", "it", "its",
    "if", "else", "when", "which", "who", "what", "there", "here", "no", "yes",
}


def repetition_ratio(text: str, n: int = 8) -> float:
    toks = (text or "").split()
    if len(toks) < n:
        return 1.0
    grams = [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def flow_identifiers(text: str) -> list[str]:
    out: list[str] = []
    for m in _IDENT.finditer(text or ""):
        tok = m.group(0)
        if tok.lower() in KEYWORDS:
            continue
        if tok.isupper() and len(tok) <= 2:
            continue
        out.append(tok)
    return out


def identifier_precision(flow: str, ast_idents: Iterable[str]) -> float:
    pred = flow_identifiers(flow)
    if not pred:
        return 1.0
    gold = set(ast_idents)
    return sum(1 for t in pred if t in gold) / len(pred)


def quote_precision(flow: str, code: str, rules: list[str]) -> float:
    quotes = [m.group(1) for m in _BACKTICK.finditer(flow or "")]
    quotes = [q for q in quotes if len(q.strip()) >= 8]
    if not quotes:
        return 1.0
    hay = normalize(code, rules)
    hits = 0
    for q in quotes:
        needle = normalize(q, rules)
        if needle and needle in hay:
            hits += 1
    return hits / len(quotes)


def _ci(samples: list[float], rng: np.random.Generator, n_boot: int) -> dict:
    arr = np.asarray(samples, dtype=float)
    point = float(arr.mean()) if len(arr) else 0.0
    if len(arr) == 0:
        return {"value": 0.0, "ci95": [0.0, 0.0]}
    stats = np.empty(n_boot, dtype=float)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = float(arr[idx].mean())
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"value": round(point, 6), "ci95": [round(float(lo), 6), round(float(hi), 6)]}


def evaluate_outputs(
    rows: list[dict],
    inputs_by_id: dict[str, dict],
    *,
    norm_rules: list[str],
    bootstrap_seed: int,
    n_boot: int = 1000,
) -> dict[str, Any]:
    by_id: dict[str, list[dict]] = {}
    for rec in rows:
        by_id.setdefault(rec["input_id"], []).append(rec)

    trunc: list[float] = []
    reps: list[float] = []
    idp: list[float] = []
    qp: list[float] = []
    verdicts = Counter()
    unparsed = 0
    n_rows = 0

    for iid, runs in by_id.items():
        src = inputs_by_id[iid]
        ast_idents = src.get("ast_identifiers") or []
        t_flags = []
        r_vals = []
        i_vals = []
        q_vals = []
        for rec in runs:
            n_rows += 1
            flow = rec.get("flow_text") or ""
            verd = parse_verdict(flow)
            rec["verdict"] = verd
            verdicts[verd] += 1
            if verd == "UNPARSED":
                unparsed += 1
            t_flags.append(1.0 if rec.get("truncated_flag") else 0.0)
            r_vals.append(repetition_ratio(flow))
            i_vals.append(identifier_precision(flow, ast_idents))
            q_vals.append(quote_precision(flow, src["code"], norm_rules))
        trunc.append(float(np.mean(t_flags)))
        reps.append(float(np.mean(r_vals)))
        idp.append(float(np.mean(i_vals)))
        qp.append(float(np.mean(q_vals)))

    rng = np.random.default_rng(bootstrap_seed)
    n_inputs = len(by_id)
    dist = {k: round(verdicts[k] / max(n_rows, 1), 6) for k in ("VULNERABLE", "SAFE", "INSUFFICIENT", "UNPARSED")}
    dist_ci = {}
    for label in dist:
        per_input = []
        for runs in by_id.values():
            per_input.append(sum(1 for r in runs if parse_verdict(r.get("flow_text") or "") == label) / max(len(runs), 1))
        dist_ci[label] = _ci(per_input, np.random.default_rng(bootstrap_seed), n_boot)

    return {
        "n_inputs": n_inputs,
        "n_rows": n_rows,
        "unparsed_count": unparsed,
        "truncation_rate": _ci(trunc, np.random.default_rng(bootstrap_seed), n_boot),
        "repetition_ratio": _ci(reps, np.random.default_rng(bootstrap_seed), n_boot),
        "identifier_precision": _ci(idp, np.random.default_rng(bootstrap_seed), n_boot),
        "quote_precision": _ci(qp, np.random.default_rng(bootstrap_seed), n_boot),
        "verdict_distribution": dist,
        "verdict_distribution_ci": dist_ci,
        "verdict_counts": dict(verdicts),
        "bootstrap": {"seed": bootstrap_seed, "resamples": n_boot, "unit": "input_id"},
    }
