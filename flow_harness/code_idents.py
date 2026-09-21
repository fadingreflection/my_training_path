"""Extract code-shaped identifier tokens from flow text (Gate 2 metric)."""
from __future__ import annotations

import re
from typing import Iterable

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

_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_ARROW_FLOW = re.compile(r"->\s*([A-Za-z_][A-Za-z0-9_]*)")
_DOT_FLOW = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\.\s*([a-z_][A-Za-z0-9_]*)")
_CALL_FLOW = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_BACKTICK = re.compile(r"`([^`]+)`")


def _bare_ident_code_like(tok: str) -> bool:
    """Underscore, ALL_CAPS macro, or multi-capital CamelCase — not TitleCase prose."""
    if not tok or len(tok) < 2:
        return False
    if "_" in tok:
        return True
    if tok.isupper() and any(c.isalpha() for c in tok) and len(tok) >= 3:
        return True
    if tok[:1].isupper() and sum(1 for c in tok if c.isupper()) >= 2:
        return True
    return False


def _take(tok: str, out: list[str]) -> None:
    if not tok or len(tok) < 2:
        return
    if tok.lower() in KEYWORDS:
        return
    out.append(tok)


def flow_code_identifiers(text: str) -> list[str]:
    """Allowlist: backticks, ->/. fields, calls, _, CamelCase, ALL_CAPS — not plain English."""
    out: list[str] = []
    flow = text or ""

    for tok in _ARROW_FLOW.findall(flow):
        _take(tok, out)
    for tok in _DOT_FLOW.findall(flow):
        _take(tok, out)
    for tok in _CALL_FLOW.findall(flow):
        _take(tok, out)
    for tok in _IDENT.findall(flow):
        if _bare_ident_code_like(tok):
            _take(tok, out)
    for q in _BACKTICK.findall(flow):
        q = q.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", q):
            _take(q, out)
            continue
        for tok in _ARROW_FLOW.findall(q):
            _take(tok, out)
        for tok in _DOT_FLOW.findall(q):
            _take(tok, out)
        for tok in _CALL_FLOW.findall(q):
            _take(tok, out)
        for tok in _IDENT.findall(q):
            if _bare_ident_code_like(tok):
                _take(tok, out)
    return out


def source_tokens(code: str) -> set[str]:
    return {m.group(0) for m in _IDENT.finditer(code or "") if m.group(0).lower() not in KEYWORDS}


def source_fields(code: str) -> set[str]:
    return set(_ARROW_FLOW.findall(code or "")) | set(_DOT_FLOW.findall(code or ""))


def source_callees(code: str) -> set[str]:
    return set(_CALL_FLOW.findall(code or ""))


def source_macros(code: str) -> set[str]:
    return {
        tok
        for tok in source_tokens(code)
        if tok.isupper() and any(c.isalpha() for c in tok) and len(tok) >= 3
    }


def expanded_snippet_gold(code: str, ast_idents: Iterable[str]) -> set[str]:
    """AST identifiers ∪ snippet tokens ∪ fields ∪ callees ∪ ALL_CAPS macros."""
    ast = set(ast_idents)
    return ast | source_tokens(code) | source_fields(code) | source_callees(code) | source_macros(code)


def precision(pred: list[str], gold: set[str]) -> float | None:
    if not pred:
        return None
    return sum(1 for t in pred if t in gold) / len(pred)


_CODE_IN_QUOTE = re.compile(r"[;()\[\]=<>]|->|\+\+|--|==|!=|\|\||&&|\{|\}")


def flow_code_quotes(text: str) -> list[str]:
    """Backtick spans that look like code, not English prose in backticks."""
    out: list[str] = []
    for q in (m.group(1).strip() for m in _BACKTICK.finditer(text or "")):
        if len(q) < 8:
            continue
        if "\n" in q and ";" not in q and "{" not in q:
            continue
        if _CODE_IN_QUOTE.search(q):
            out.append(q)
            continue
        if "_" in q:
            out.append(q)
            continue
        if "->" in q or re.search(r"\w\s*\(", q):
            out.append(q)
            continue
        if sum(1 for c in q if c.isupper()) >= 2:
            out.append(q)
    return out


def flow_quotes_raw(text: str) -> list[str]:
    return [q.strip() for q in _BACKTICK.findall(text or "") if len(q.strip()) >= 8]


def quote_matches_code(q: str, hay: str, norm_rules: list[str]) -> bool:
    from normalize import normalize

    needle = normalize(q, norm_rules)
    if needle and needle in hay:
        return True
    toks = [t for t in _IDENT.findall(q) if t.lower() not in KEYWORDS and len(t) >= 2]
    if len(toks) >= 2 and all(t in hay for t in toks):
        return True
    return False
