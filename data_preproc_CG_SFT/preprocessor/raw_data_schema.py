from typing import Optional, List, Dict, Any

def make_raw_data(
    code: str,
    source: str,
    context: Optional[str] = None,
    raw_cwe: Optional[str] = None,
    raw_title: Optional[str] = None,
    raw_description: Optional[str] = None,
    raw_recommendation: Optional[str] = None,
    raw_severity: Optional[str] = None,
    raw_location: Optional[str] = None,
    raw_lines: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Создаёт словарь RawData без домысливания."""
    return {
        "code": code,
        "source": source,
        "context": context or "",
        "raw_cwe": raw_cwe,
        "raw_title": raw_title,
        "raw_description": raw_description,
        "raw_recommendation": raw_recommendation,
        "raw_severity": raw_severity,
        "raw_location": raw_location,
        "raw_lines": raw_lines or [],
        "extra": extra or {}
    }