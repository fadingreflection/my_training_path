"""First-line verdict parser. UNPARSED is a parser defect, never silenced."""
from __future__ import annotations

import re

ALLOWED = ("VULNERABLE", "SAFE", "INSUFFICIENT")
_HEAD = re.compile(r"^[\s#>*-]*")
# INSUFFICIENT_INFO before INSUFFICIENT. No trailing \b — handles SAFECompeting…
_VERDICT_HEAD = re.compile(
    r"^(?:\s*\**\s*)?(VULNERABLE|INSUFFICIENT_INFO|INSUFFICIENT|SAFE)(?:\.|[A-Z\s]|$)",
    re.I,
)


def _normalize_verdict_token(token: str) -> str | None:
    token = token.upper().replace("-", "_").rstrip(".:;")
    if token in {"VULNERABLE", "VULN"}:
        return "VULNERABLE"
    if token == "SAFE":
        return "SAFE"
    if token in {"INSUFFICIENT", "INSUFFICIENT_INFO", "INSUFFICIENTINFO"}:
        return "INSUFFICIENT"
    return None


def parse_verdict(flow_text: str) -> str:
    if not (flow_text or "").strip():
        return "UNPARSED"

    # Glued outputs: SAFECompeting hypotheses… (no newline after verdict).
    m = _VERDICT_HEAD.match((flow_text or "").lstrip()[:400])
    if m:
        verdict = _normalize_verdict_token(m.group(1))
        if verdict:
            return verdict

    raw = next((ln for ln in flow_text.splitlines() if ln.strip()), "")
    head = _HEAD.sub("", raw).strip().strip('"').strip("'")
    head = head.replace("*", " ").replace("`", " ")
    token = head.split()[0].upper().rstrip(".:;") if head else ""
    verdict = _normalize_verdict_token(token)
    if verdict:
        return verdict
    return "UNPARSED"
