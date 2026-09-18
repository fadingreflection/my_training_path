"""First-line verdict parser. UNPARSED is a parser defect, never silenced."""
from __future__ import annotations

import re

ALLOWED = ("VULNERABLE", "SAFE", "INSUFFICIENT")
_HEAD = re.compile(r"^[\s#>*-]*")


def parse_verdict(flow_text: str) -> str:
    if not (flow_text or "").strip():
        return "UNPARSED"
    raw = flow_text.splitlines()[0]
    head = _HEAD.sub("", raw).strip().strip('"').strip("'")
    token = head.split()[0].upper().rstrip(".:;") if head else ""
    token = token.replace("-", "_")
    if token in {"VULNERABLE", "VULN"}:
        return "VULNERABLE"
    if token == "SAFE":
        return "SAFE"
    if token in {"INSUFFICIENT", "INSUFFICIENT_INFO", "INSUFFICIENTINFO"}:
        return "INSUFFICIENT"
    return "UNPARSED"
