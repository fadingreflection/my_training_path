"""Load and hash generator manifests. Any field change is a new spec."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED = (
    "version",
    "model_name",
    "checkpoint_hash",
    "tokenizer_version",
    "prompt",
    "sampling",
    "constrained_decoding_version",
    "normalization",
    "selection_seed",
)


def load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"manifest is not a mapping: {path}")
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise SystemExit(f"{path}: missing fields {missing}")
    sampling = data["sampling"]
    for key in ("temperature", "top_p", "max_tokens", "seeds"):
        if key not in sampling:
            raise SystemExit(f"{path}: sampling.{key} required")
    if not sampling["seeds"]:
        raise SystemExit(f"{path}: sampling.seeds must be non-empty")
    if not isinstance(data["normalization"], list) or not data["normalization"]:
        raise SystemExit(f"{path}: normalization must be a non-empty list")
    data["spec_hash"] = spec_hash(data)
    data["_path"] = str(path)
    return data


def _canonical(data: dict[str, Any]) -> dict[str, Any]:
    skip = {"spec_hash", "_path"}
    return {k: data[k] for k in sorted(data) if k not in skip}


def spec_hash(data: dict[str, Any]) -> str:
    blob = json.dumps(_canonical(data), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def normalization_fingerprint(data: dict[str, Any]) -> str:
    rules = list(data["normalization"])
    return hashlib.sha256(json.dumps(rules, ensure_ascii=False).encode("utf-8")).hexdigest()


def version_label(data: dict[str, Any]) -> str:
    """Human version plus spec prefix so a field edit cannot hide as v0."""
    claimed = str(data["version"]).strip()
    short = data["spec_hash"][:8]
    return f"{claimed}-{short}"
