#!/usr/bin/env python3
"""Audit identifier_precision: category split + 50 false-token review + expanded gold.

Does not overwrite v0_baseline.json. Raw 0.20 stays the frozen number.

  python3 flow_harness/ident_audit.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ast_util import parse_function  # noqa: E402
from metrics import KEYWORDS, flow_identifiers  # noqa: E402

_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_ARROW_FLOW = re.compile(r"->\s*([A-Za-z_][A-Za-z0-9_]*)")
_DOT_FLOW = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\.\s*([a-z_][A-Za-z0-9_]*)")
_CALL_FLOW = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_STRUCT_FLOW = re.compile(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)")
_BACKTICK = re.compile(r"`([^`]+)`")

PROSE = {
    "attacker", "path", "check", "out", "so", "bounds", "sink", "controlled",
    "input", "only", "function", "can", "buffer", "before", "loop", "hypotheses",
    "after", "missing", "overflow", "competing", "ruled", "hypothesis", "pointer",
    "chosen", "value", "used", "then", "when", "this", "that", "from", "into",
    "with", "without", "because", "however", "therefore", "reachable", "shown",
    "code", "call", "caller", "callee", "body", "snippet", "statement", "access",
    "length", "size", "index", "offset", "count", "data", "value", "error",
    "null", "valid", "invalid", "safe", "vulnerable", "issue", "bug", "flaw",
    "first", "second", "third", "single", "most", "more", "less", "same",
    "next", "previous", "other", "another", "each", "every", "any", "all",
    "does", "doesn", "did", "done", "doing", "make", "makes", "made",
    "return", "returns", "returned", "via", "per", "vs", "etc",
    "memory", "write", "written", "reading", "writes", "reads",
    "under", "over", "between", "within", "across", "through",
    "case", "cases", "condition", "conditions", "check", "checks",
    "never", "always", "still", "also", "just", "even", "already",
    "known", "real", "actual", "possible", "potential",
    "linux", "kernel", "user", "guest", "host",
    "passes", "opens", "loops", "succeeds", "fails", "handling",
    "initialization", "bytes", "array", "argument", "terminates",
    "checked", "negative", "itself", "blob", "connection", "lengths",
    "limit", "allocation", "object", "reconnect", "additionally",
    "nothing", "critical", "since", "once", "during", "covers",
    "accounts", "includes", "including",
}

LIBC_CALLEES = {
    "memcpy", "memmove", "memset", "memcmp", "strcpy", "strncpy", "strcat",
    "strncat", "sprintf", "snprintf", "printf", "fprintf", "scanf", "sscanf",
    "malloc", "calloc", "realloc", "free", "alloca", "strlen", "strcmp",
    "strncmp", "strchr", "strstr", "atoi", "atol", "read", "write", "recv",
    "send", "open", "close", "ioctl", "system", "exec", "popen", "qsort",
    "bcopy", "gets", "puts", "fgets", "fputs",
}


def source_tokens(code: str) -> set[str]:
    return {m.group(0) for m in _IDENT.finditer(code or "") if m.group(0).lower() not in KEYWORDS}


def source_fields(code: str) -> set[str]:
    return set(_ARROW_FLOW.findall(code or "")) | set(_DOT_FLOW.findall(code or ""))


def source_callees(code: str) -> set[str]:
    return set(_CALL_FLOW.findall(code or ""))


def source_macros(code: str) -> set[str]:
    out = set()
    for tok in source_tokens(code):
        if tok.isupper() and any(c.isalpha() for c in tok) and len(tok) >= 3:
            out.add(tok)
    return out


def is_prose(tok: str) -> bool:
    low = tok.lower()
    if low in KEYWORDS or low in PROSE:
        return True
    if tok[:1].isupper() and tok[1:].islower() and low in PROSE | KEYWORDS | {
        "competing", "hypothesis", "vulnerable", "insufficient", "chosen",
        "attacker", "missing", "ruled",
    }:
        return True
    return False


def extract_code_idents(flow: str) -> list[str]:
    """Identifiers that look like code, not explanation English."""
    out: list[str] = []

    def take(tok: str) -> None:
        if not tok or len(tok) < 2:
            return
        if is_prose(tok) or tok.lower() in KEYWORDS:
            return
        out.append(tok)

    for tok in _ARROW_FLOW.findall(flow or ""):
        take(tok)
    for tok in _DOT_FLOW.findall(flow or ""):
        take(tok)
    for tok in _CALL_FLOW.findall(flow or ""):
        take(tok)
    for tok in _IDENT.findall(flow or ""):
        if "_" in tok or (tok.isupper() and any(c.isalpha() for c in tok) and len(tok) >= 3):
            take(tok)
    for q in _BACKTICK.findall(flow or ""):
        q = q.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", q):
            take(q)
            continue
        for tok in _ARROW_FLOW.findall(q):
            take(tok)
        for tok in _DOT_FLOW.findall(q):
            take(tok)
        for tok in _CALL_FLOW.findall(q):
            take(tok)
        for tok in _IDENT.findall(q):
            if "_" in tok or (tok.isupper() and any(c.isalpha() for c in tok) and len(tok) >= 3):
                take(tok)
    return out


def flow_category(tok: str, flow: str) -> str:
    if re.search(rf"->\s*{re.escape(tok)}\b", flow) or re.search(
        rf"[A-Za-z_][A-Za-z0-9_]*\s*\.\s*{re.escape(tok)}\b", flow
    ):
        return "field"
    if re.search(rf"\b{re.escape(tok)}\s*\(", flow) or tok in LIBC_CALLEES:
        return "callee"
    if tok.isupper() and any(c.isalpha() for c in tok) and len(tok) >= 3:
        return "macro"
    if re.search(rf"\bstruct\s+{re.escape(tok)}\b", flow):
        return "type"
    if in_backticks_single(tok, flow) or "_" in tok:
        return "local_or_param"
    return "other"


def in_backticks_single(tok: str, flow: str) -> bool:
    for q in _BACKTICK.findall(flow or ""):
        q = q.strip()
        if q == tok:
            return True
        if re.search(rf"\b{re.escape(tok)}\b", q) and ("_" in tok or tok.isupper()):
            return True
    return False


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_outputs(path: Path) -> dict[str, dict]:
    out = {}
    for rec in load_jsonl(path):
        out.setdefault(rec["input_id"], rec)
    return out


def expanded_gold(code: str, ast_idents: set[str], parsed) -> dict[str, set[str]]:
    cats = {k: set(parsed.by_category.get(k, [])) for k in ("param", "local", "field", "callee", "type", "macro", "other")}
    cats["field"] |= source_fields(code)
    cats["callee"] |= source_callees(code)
    cats["macro"] |= source_macros(code)
    body_tokens = source_tokens(code) | ast_idents
    cats["body_tokens"] = body_tokens
    return cats


def label_false(tok: str, code: str, category: str) -> str:
    if is_prose(tok):
        return "prose_leak"
    if tok in source_tokens(code) or tok in source_fields(code) or tok in source_callees(code):
        return "ast_limitation"
    if category in {"field", "callee", "macro", "type"}:
        # mentioned as a real program symbol but not in this function body
        return "ast_limitation"
    return "invented"


def main() -> None:
    inputs = {r["id"]: r for r in load_jsonl(HERE / "fixed_200_inputs.jsonl")}
    outs = first_outputs(HERE / "v0_outputs.jsonl")

    raw_pred = raw_hit = 0
    code_pred = code_hit_ast = code_hit_exp = 0
    by_cat_ast = defaultdict(lambda: {"n": 0, "hit": 0})
    by_cat_exp = defaultdict(lambda: {"n": 0, "hit": 0})
    by_cat_role = defaultdict(lambda: {"n": 0, "hit": 0})
    false_code: list[dict] = []
    prose_fp = 0
    code_fp = 0

    role_gold_key = {
        "local_or_param": "local_or_param",
        "field": "field",
        "callee": "callee",
        "macro": "macro",
        "type": "type",
        "other": "body_tokens",
    }

    for iid, rec in outs.items():
        src = inputs[iid]
        code = src["code"]
        parsed = parse_function(code)
        ast_set = set(src.get("ast_identifiers") or parsed.identifier_set)
        ast_cat = {k: set(parsed.by_category.get(k, [])) for k in parsed.by_category}
        gold_x = expanded_gold(code, ast_set, parsed)
        gold_x["local_or_param"] = ast_cat.get("param", set()) | ast_cat.get("local", set())
        gold_x["type"] |= set(_STRUCT_FLOW.findall(code or ""))
        flow = rec.get("flow_text") or ""
        pred = flow_identifiers(flow)
        code_pred_list = extract_code_idents(flow)
        for tok in pred:
            raw_pred += 1
            in_ast = tok in ast_set
            if in_ast:
                raw_hit += 1
            if is_prose(tok) and not in_ast:
                prose_fp += 1
        for tok in code_pred_list:
            in_ast = tok in ast_set
            code_pred += 1
            cat = flow_category(tok, flow)
            by_cat_ast[cat]["n"] += 1
            by_cat_exp[cat]["n"] += 1
            by_cat_role[cat]["n"] += 1
            if in_ast:
                code_hit_ast += 1
                by_cat_ast[cat]["hit"] += 1
            if tok in gold_x["body_tokens"]:
                code_hit_exp += 1
                by_cat_exp[cat]["hit"] += 1
            role_gold = gold_x.get(role_gold_key.get(cat, "body_tokens"), set())
            if tok in role_gold:
                by_cat_role[cat]["hit"] += 1
            if not in_ast:
                code_fp += 1
                false_code.append(
                    {
                        "input_id": iid,
                        "token": tok,
                        "category": cat,
                        "in_source_text": tok in gold_x["body_tokens"],
                        "in_source_fields": tok in gold_x["field"],
                        "in_source_callees": tok in gold_x["callee"],
                        "in_role_gold": tok in role_gold,
                        "project": src.get("project"),
                        "cwe": src.get("cwe"),
                        "flow_excerpt": next(
                            (ln.strip() for ln in flow.splitlines() if re.search(rf"\b{re.escape(tok)}\b", ln)),
                            "",
                        )[:220],
                    }
                )

    rng = random.Random(20260918)
    uniq = []
    seen = set()
    rng.shuffle(false_code)
    for item in false_code:
        key = (item["token"], item["category"])
        if key in seen:
            continue
        seen.add(key)
        item["audit_label"] = label_false(
            item["token"],
            inputs[item["input_id"]]["code"],
            item["category"],
        )
        uniq.append(item)
        if len(uniq) >= 50:
            break

    label_counts = Counter(x["audit_label"] for x in uniq)

    def rate(d):
        n = d["n"]
        return None if n == 0 else round(d["hit"] / n, 4)

    report = {
        "gate2_verdict": None,
        "gate2_note": (
            "identifier_precision 0.20 is under metric audit. "
            "Do not start constrained-decoding / model-swap. "
            "Raw number is kept in v0_baseline.json."
        ),
        "raw_body_ast": {
            "precision": round(raw_hit / max(raw_pred, 1), 4),
            "n_pred_tokens": raw_pred,
            "n_hit": raw_hit,
            "note": "current harness metric: every English-like token in the flow vs function-body AST",
        },
        "code_like_vs_body_ast": {
            "precision": round(code_hit_ast / max(code_pred, 1), 4),
            "n_pred_tokens": code_pred,
            "n_hit": code_hit_ast,
        },
        "code_like_vs_expanded_gold": {
            "precision": round(code_hit_exp / max(code_pred, 1), 4),
            "n_pred_tokens": code_pred,
            "n_hit": code_hit_exp,
            "gold": "function-body AST identifiers ∪ identifier tokens ∪ ->fields ∪ name( callees ∪ ALL_CAPS macros in the snippet",
        },
        "by_category_vs_body_ast": {k: {"n": v["n"], "hit": v["hit"], "precision": rate(v)} for k, v in sorted(by_cat_ast.items())},
        "by_category_vs_expanded_gold": {k: {"n": v["n"], "hit": v["hit"], "precision": rate(v)} for k, v in sorted(by_cat_exp.items())},
        "by_category_vs_role_gold": {
            k: {"n": v["n"], "hit": v["hit"], "precision": rate(v)}
            for k, v in sorted(by_cat_role.items())
        },
        "role_gold_definition": {
            "local_or_param": "AST parameter_declaration ∪ local declaration identifiers",
            "field": "AST field_identifier ∪ ->field / obj.field in the snippet",
            "callee": "AST call_expression function ∪ name( in the snippet",
            "macro": "ALL_CAPS tokens in the snippet ∪ AST preproc/macro-like identifiers",
            "type": "AST type_identifier ∪ struct Name in the snippet",
        },
        "false_split": {
            "prose_tokens_counted_as_false": prose_fp,
            "code_like_false_vs_body_ast": code_fp,
        },
        "audit_50": {
            "n": len(uniq),
            "seed": 20260918,
            "label_counts": dict(label_counts),
            "rule": (
                "prose_leak = English explanation word; "
                "ast_limitation = token exists in snippet or is a field/callee/macro/type "
                "used as a real symbol; invented = neither"
            ),
            "rows": uniq,
        },
    }
    out = HERE / "ident_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Identifier-precision audit (not a gate-2 verdict)",
        "",
        "Raw `identifier_precision` in `v0_baseline.json` stays **0.2027**. Gate 2 is **not set**.",
        "",
        "## What 0.20 actually is",
        "",
        f"- Raw tokens vs function-body AST: **{report['raw_body_ast']['precision']}** "
        f"({raw_hit}/{raw_pred}).",
        f"- Of raw false tokens, **{prose_fp}** are English prose (`attacker`, `path`, `sink`…).",
        f"- Code-like tokens vs the same AST gold: **{report['code_like_vs_body_ast']['precision']}** "
        f"({code_hit_ast}/{code_pred}).",
        f"- Code-like vs expanded snippet gold: **{report['code_like_vs_expanded_gold']['precision']}** "
        f"({code_hit_exp}/{code_pred}).",
        "",
        "## Category precision (code-like tokens)",
        "",
        "| category | n | vs body AST | vs expanded snippet | vs role gold |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat in ("local_or_param", "field", "callee", "type", "macro", "other"):
        a = report["by_category_vs_body_ast"].get(cat, {"n": 0, "precision": None})
        b = report["by_category_vs_expanded_gold"].get(cat, {"precision": None})
        c = report["by_category_vs_role_gold"].get(cat, {"precision": None})
        lines.append(
            f"| {cat} | {a['n']} | {a['precision']} | {b['precision']} | {c['precision']} |"
        )
    lines += [
        "",
        "## Manual-rule audit of 50 code-like false tokens (vs body AST)",
        "",
        f"Labels: {dict(label_counts)}",
        "",
        "If `ast_limitation` dominates, the gold is the function body, not the model.",
        "Expanded gold still cannot see declarations in other files / headers; that is Phase 1 if needed.",
        "",
        "| token | category | label | in snippet | excerpt |",
        "|---|---|---|---|---|",
    ]
    for row in uniq:
        ex = row["flow_excerpt"].replace("|", "/")
        lines.append(
            f"| `{row['token']}` | {row['category']} | {row['audit_label']} | "
            f"{row['in_source_text']} | {ex[:80]} |"
        )
    (HERE / "ident_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ast_path = HERE / "ast_check.json"
    ast = json.loads(ast_path.read_text(encoding="utf-8")) if ast_path.exists() else {}
    ast["gate_decision"] = {
        "dataset_unparseable_pct": ast.get("fail_rate"),
        "threshold_pct": 0.05,
        "factor_over_threshold": round((ast.get("fail_rate") or 0) / 0.05, 2),
        "cause": "TBD",
        "cause_provisional": {
            "no_func_sig_head": "snippet does not start with a recognizable function signature",
            "has_preproc": "#define / #ifdef in the body",
            "cppish": "template / :: / class",
            "truncated_tail": "body does not end with } or ;",
            "kernel_macros": "e.g. kvm_for_each_memslot — parser sees a call-like identifier, often still parse_ok; remaining fails mix macros + chopped BigVul text",
        },
        "representativeness": (
            "Phase 0 battery is sampled from the parseable 82.6% only. "
            "Conclusions do not cover the unparseable 17.4%. "
            "If those functions are systematically harder (macro-heavy, truncated), "
            "the training/eval distribution is biased. Estimate in Phase 1."
        ),
        "decision": "logged as a gate record; not a silent footnote",
    }
    ast_path.write_text(json.dumps(ast, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "raw_body_ast", "code_like_vs_body_ast", "code_like_vs_expanded_gold",
        "by_category_vs_body_ast", "by_category_vs_expanded_gold",
        "by_category_vs_role_gold",
        "false_split",
    )}, indent=2))
    print("audit_50", dict(label_counts))
    print("wrote", out, HERE / "ident_audit.md")


if __name__ == "__main__":
    main()
