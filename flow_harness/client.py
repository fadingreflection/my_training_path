"""Thin generator wrapper: OpenRouter (explain_cwe_map/run.py) or frozen replay."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GenResult:
    flow_text: str
    tokens_used: int
    elapsed_ms: int
    truncated_flag: bool
    backend: str
    error: str = ""


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def api_key_available() -> bool:
    load_env()
    return bool(os.environ.get("OPENROUTER_API_KEY"))


class ReplayGenerator:
    """v0 freeze: replay teacher review traces already on disk."""

    def __init__(self, traces_path: Path):
        self.traces_path = traces_path
        self.by_id: dict[str, dict] = {}
        with traces_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.by_id[rec["id"]] = rec

    def generate(
        self,
        rec: dict[str, Any],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        prompt: str,
    ) -> GenResult:
        del seed, temperature, max_tokens, top_p, prompt
        src = self.by_id.get(rec["id"])
        if src is None:
            return GenResult("", 0, 0, False, "replay", error=f"no frozen trace for {rec['id']}")
        text = src.get("explanation") or ""
        tokens = len(text.split())
        return GenResult(text, tokens, 0, False, "replay")


class OpenRouterGenerator:
    """Same OpenRouter path as explain_cwe_map/run.py: reasoning effort + visible text."""

    RETRIES = 4
    REASONING_MAX_TOKENS = 2048

    def __init__(self, model_name: str):
        import sys

        sys.path.insert(0, str(ROOT / "explain_cwe_map"))
        from run import OPENROUTER_BASE_URL, _visible_text, make_client  # noqa: WPS433

        self.model_name = model_name
        self._make_client = make_client
        self._visible_text = _visible_text
        self._base = OPENROUTER_BASE_URL
        self.client = make_client()

    def generate(
        self,
        rec: dict[str, Any],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        prompt: str,
    ) -> GenResult:
        messages = [{"role": "user", "content": f"{prompt.rstrip()}\n\n{rec['code'].strip()}\n"}]
        reasoning = {"effort": "low", "max_tokens": min(self.REASONING_MAX_TOKENS, int(max_tokens))}
        last_err = ""
        t0 = time.monotonic()
        for attempt in range(self.RETRIES):
            try:
                extra_body: dict[str, Any] = {
                    "models": [self.model_name],
                    "reasoning": reasoning,
                }
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens),
                    seed=int(seed),
                    extra_body=extra_body,
                )
                choice = resp.choices[0]
                text, src = self._visible_text(choice.message)
                usage = getattr(resp, "usage", None)
                tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else len(text.split())
                finish = (getattr(choice, "finish_reason", None) or "").lower()
                truncated = finish in {"length", "max_tokens"}
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                if not text:
                    last_err = f"empty content after reasoning src={src}"
                    time.sleep(min(2 ** attempt, 20))
                    continue
                result = GenResult(text, tokens, elapsed_ms, truncated, "openrouter")
                result.error = ""
                return result
            except Exception as exc:
                err = str(exc)
                last_err = err[:300]
                if "reasoning" in err.lower() and "max_tokens" in reasoning:
                    reasoning = {"effort": "low"}
                wait = min(2 ** attempt, 20)
                if "429" in err:
                    wait = min(15 * (attempt + 1), 90)
                time.sleep(wait)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return GenResult("", 0, elapsed_ms, False, "openrouter", error=last_err or "empty content")


def make_generator(manifest: dict[str, Any]) -> ReplayGenerator | OpenRouterGenerator:
    backend = (manifest.get("backend") or "auto").lower()
    traces = manifest.get("replay_traces")
    traces_path = ROOT / traces if traces else None
    if backend == "replay" or (backend == "auto" and not api_key_available()):
        if not traces_path or not traces_path.exists():
            raise SystemExit("replay backend needs replay_traces path that exists")
        return ReplayGenerator(traces_path)
    return OpenRouterGenerator(manifest["model_name"])
