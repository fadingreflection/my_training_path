"""CPU metrics + 95% bootstrap CI over input_id."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

import numpy as np

from ast_util import parse_function
from code_idents import (
    KEYWORDS,
    expanded_snippet_gold,
    flow_code_identifiers,
    flow_code_quotes,
    flow_quotes_raw,
    precision,
    quote_matches_code,
)
from normalize import normalize
from verdict import parse_verdict

_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_BACKTICK = re.compile(r"`([^`]+)`")


def repetition_ratio(text: str, n: int = 8) -> float:
    toks = (text or "").split()
    if len(toks) < n:
        return 1.0
    grams = [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def flow_identifiers_raw(text: str) -> list[str]:
    """Legacy extractor: every English-like token (diagnostic only)."""
    out: list[str] = []
    for m in _IDENT.finditer(text or ""):
        tok = m.group(0)
        if tok.lower() in KEYWORDS:
            continue
        if tok.isupper() and len(tok) <= 2:
            continue
        out.append(tok)
    return out


flow_identifiers = flow_identifiers_raw


def ast_identifiers_for_input(src: dict) -> list[str]:
    """Fresh parse so AST gold picks up primitive_type / preproc names."""
    parsed = parse_function(src.get("code") or "")
    if parsed.parse_ok and parsed.identifiers:
        return parsed.identifiers
    return list(src.get("ast_identifiers") or [])


def identifier_precision_raw(flow: str, ast_idents: Iterable[str]) -> float | None:
    return precision(flow_identifiers_raw(flow), set(ast_idents))


def identifier_precision(flow: str, ast_idents: Iterable[str]) -> float | None:
    """Gate 2 candidate: code-shaped tokens vs function-body AST."""
    return precision(flow_code_identifiers(flow), set(ast_idents))


def identifier_precision_expanded(flow: str, code: str, ast_idents: Iterable[str]) -> float | None:
    """Code-shaped tokens vs expanded snippet gold (diagnostic)."""
    return precision(flow_code_identifiers(flow), expanded_snippet_gold(code, ast_idents))


def quote_precision_raw(flow: str, code: str, rules: list[str]) -> float | None:
    quotes = flow_quotes_raw(flow)
    if not quotes:
        return None
    hay = normalize(code, rules)
    hits = sum(1 for q in quotes if quote_matches_code(q, hay, rules))
    return hits / len(quotes)


def quote_precision(flow: str, code: str, rules: list[str]) -> float | None:
    """Code-shaped backtick quotes vs snippet (expression token fallback)."""
    quotes = flow_code_quotes(flow)
    if not quotes:
        return None
    hay = normalize(code, rules)
    hits = sum(1 for q in quotes if quote_matches_code(q, hay, rules))
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


def _mean_or_none(values: list[float | None]) -> float | None:
    ok = [v for v in values if v is not None]
    return float(np.mean(ok)) if ok else None


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
    reps_nonempty: list[float] = []
    idp: list[float] = []
    idp_raw: list[float] = []
    idp_exp: list[float] = []
    qp: list[float] = []
    qp_raw: list[float] = []
    verdicts = Counter()
    unparsed = 0
    n_rows = 0
    n_no_code_idents = 0
    n_no_code_quotes = 0
    n_pred_code = 0
    n_raw_quotes = 0
    n_code_quotes = 0

    for iid, runs in by_id.items():
        src = inputs_by_id[iid]
        ast_idents = ast_identifiers_for_input(src)
        code = src["code"]
        t_flags = []
        r_vals = []
        i_vals: list[float | None] = []
        i_raw_vals: list[float | None] = []
        i_exp_vals: list[float | None] = []
        q_vals: list[float | None] = []
        q_raw_vals: list[float | None] = []
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
            if flow.strip():
                reps_nonempty.append(repetition_ratio(flow))
            pred_code = flow_code_identifiers(flow)
            n_pred_code += len(pred_code)
            if not pred_code:
                n_no_code_idents += 1
            raw_q = flow_quotes_raw(flow)
            code_q = flow_code_quotes(flow)
            n_raw_quotes += len(raw_q)
            n_code_quotes += len(code_q)
            if not code_q:
                n_no_code_quotes += 1
            i_vals.append(identifier_precision(flow, ast_idents))
            i_raw_vals.append(identifier_precision_raw(flow, ast_idents))
            i_exp_vals.append(identifier_precision_expanded(flow, code, ast_idents))
            q_vals.append(quote_precision(flow, code, norm_rules))
            q_raw_vals.append(quote_precision_raw(flow, code, norm_rules))
        trunc.append(float(np.mean(t_flags)))
        reps.append(float(np.mean(r_vals)))
        for bucket, vals in ((idp, i_vals), (idp_raw, i_raw_vals), (idp_exp, i_exp_vals)):
            m = _mean_or_none(vals)
            if m is not None:
                bucket.append(m)
        for bucket, vals in ((qp, q_vals), (qp_raw, q_raw_vals)):
            m = _mean_or_none(vals)
            if m is not None:
                bucket.append(m)

    dist = {k: round(verdicts[k] / max(n_rows, 1), 6) for k in ("VULNERABLE", "SAFE", "INSUFFICIENT", "UNPARSED")}
    dist_ci = {}
    for label in dist:
        per_input = []
        for runs in by_id.values():
            per_input.append(sum(1 for r in runs if parse_verdict(r.get("flow_text") or "") == label) / max(len(runs), 1))
        dist_ci[label] = _ci(per_input, np.random.default_rng(bootstrap_seed), n_boot)

    rep_nonempty_ci = _ci(reps_nonempty, np.random.default_rng(bootstrap_seed), n_boot) if reps_nonempty else None

    return {
        "n_inputs": len(by_id),
        "n_rows": n_rows,
        "unparsed_count": unparsed,
        "truncation_rate": _ci(trunc, np.random.default_rng(bootstrap_seed), n_boot),
        "repetition_ratio": _ci(reps, np.random.default_rng(bootstrap_seed), n_boot),
        "repetition_ratio_nonempty": rep_nonempty_ci,
        "identifier_precision": _ci(idp, np.random.default_rng(bootstrap_seed), n_boot),
        "identifier_precision_raw": _ci(idp_raw, np.random.default_rng(bootstrap_seed), n_boot),
        "identifier_precision_expanded": _ci(idp_exp, np.random.default_rng(bootstrap_seed), n_boot),
        "quote_precision": _ci(qp, np.random.default_rng(bootstrap_seed), n_boot),
        "quote_precision_raw": _ci(qp_raw, np.random.default_rng(bootstrap_seed), n_boot),
        "identifier_precision_note": (
            "identifier_precision = code-shaped tokens vs function-body AST (Gate 2 candidate). "
            "identifier_precision_raw = legacy all-words metric. "
            "quote_precision = code-shaped backticks only; quote_precision_raw = all backticks ≥8 chars."
        ),
        "n_no_code_idents_rows": n_no_code_idents,
        "n_no_code_quotes_rows": n_no_code_quotes,
        "n_pred_code_tokens": n_pred_code,
        "n_raw_quote_spans": n_raw_quotes,
        "n_code_quote_spans": n_code_quotes,
        "verdict_distribution": dist,
        "verdict_distribution_ci": dist_ci,
        "verdict_counts": dict(verdicts),
        "bootstrap": {"seed": bootstrap_seed, "resamples": n_boot, "unit": "input_id"},
    }
