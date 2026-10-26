import json
import re
from collections import defaultdict
from typing import Optional


CWE_RE = re.compile(r"CWE-?\s*(\d+)", re.IGNORECASE)


def normalize_cwe(x) -> Optional[str]:
    if not isinstance(x, str):
        return None
    m = CWE_RE.search(x)
    return f"CWE-{m.group(1)}" if m else None


def extract_first_json(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start=start):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"' and not esc:
                in_str = not in_str
                continue
            if not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[start:i + 1]
                        try:
                            json.loads(cand)
                            return cand
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    return None


def parse_response(text: str):
    js = extract_first_json(text)
    if js is None:
        return None, None
    try:
        return json.loads(js), js
    except json.JSONDecodeError:
        return None, js


def schema_ok(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("explanation"), str) or not obj["explanation"].strip():
        return False
    if normalize_cwe(obj.get("CWE")) is None:
        return False
    f = obj.get("frame")
    if not isinstance(f, dict) or "pattern" not in f or "bindings" not in f:
        return False
    t = obj.get("triplet")
    if not isinstance(t, list) or not t or not all(isinstance(x, dict) for x in t):
        return False
    return True


def macro_f1(y_true, y_pred) -> dict:
    labels = sorted({y for y in y_true if y} | {y for y in y_pred if y})
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p and t is not None:
            tp[t] += 1
        else:
            if p is not None:
                fp[p] += 1
            if t is not None:
                fn[t] += 1
    f1s, per_class = [], {}
    for lab in labels:
        precision = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        recall = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        per_class[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for y in y_true if y == lab),
        }
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p and t is not None) / max(len(y_true), 1)
    return {"macro_f1": macro, "accuracy": acc, "per_class": per_class}