"""Normalization rules frozen in the generator manifest."""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_DASH = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]")
_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2033": '"',
    "\u00ab": '"',
    "\u00bb": '"',
}

RULE_ORDER = (
    "strip",
    "unify_quotes",
    "unify_dashes",
    "collapse_whitespace",
)


def apply_rule(text: str, rule: str) -> str:
    if rule == "strip":
        return (text or "").strip()
    if rule == "collapse_whitespace":
        return _WS.sub(" ", text or "").strip()
    if rule == "unify_quotes":
        out = text or ""
        for src, dst in _QUOTES.items():
            out = out.replace(src, dst)
        return out
    if rule == "unify_dashes":
        return _DASH.sub("-", text or "")
    raise ValueError(f"unknown normalization rule: {rule!r}")


def normalize(text: str, rules: list[str]) -> str:
    out = text or ""
    for rule in rules:
        out = apply_rule(out, rule)
    return out
