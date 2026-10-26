"""
core/loader.py — загрузка TaskSpec / PromptSpec / ModelSpec из YAML.

Импорты относительные (from .schema import ...), потому что loader.py
и schema.py лежат в одном пакете core/. Запуск — из корня inference_backend/:
    python3 -m scripts.check_normalize
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .schema import (
    ExtractorSpec,
    InputSpec,
    ModelSpec,
    PromptSpec,
    TargetSpec,
    TaskSpec,
)

logger = logging.getLogger(__name__)


def load_task(path: str | Path) -> TaskSpec:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TaskSpec not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    required = {"name", "description", "input", "target", "extractor"}
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"TaskSpec {path}: missing keys {sorted(missing)}")

    return TaskSpec(
        name=d["name"],
        description=d["description"],
        input=InputSpec(**d["input"]),
        target=TargetSpec(**d["target"]),
        extractor=ExtractorSpec(**d["extractor"]),
        sample_id_expr=d.get(
            "sample_id_expr",
            "record.get('sample_id') or record.get('id')",
        ),
    )


def load_prompt(path: str | Path) -> PromptSpec:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PromptSpec not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    required = {"name", "system", "user_template"}
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"PromptSpec {path}: missing keys {sorted(missing)}")

    return PromptSpec(
        name=d["name"],
        system=d["system"],
        user_template=d["user_template"],
        output_schema=d.get("output_schema", {}),
    )


def load_model(path: str | Path, name: str | None = None) -> ModelSpec:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ModelSpec not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    if name is not None:
        if name not in d:
            raise KeyError(f"Model '{name}' not found in {path}. Available: {sorted(d.keys())}")
        d = d[name]
    elif all(isinstance(v, dict) and "backend" in v for v in d.values()) and "backend" not in d:
        raise ValueError(
            f"{path} looks like a registry; pass model name. Available: {sorted(d.keys())}"
        )

    required = {"name", "backend"}
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"ModelSpec {path}: missing keys {sorted(missing)}")

    return ModelSpec(
        name=d["name"],
        backend=d["backend"],
        path=d.get("path"),
        provider=d.get("provider"),
        model=d.get("model"),
        gen_kwargs=d.get("gen_kwargs", {}),
        backend_kwargs=d.get("backend_kwargs", {}),
    )


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(base: Path, p: str | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()