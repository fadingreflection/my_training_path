#!/usr/bin/env python3
"""Explain BigVul func_before, then map the explanation onto a CWE hierarchy.

Only vulnerable bodies (func_before). Two chat turns on DeepSeek V4 Flash
via the native API. Success = gold CWE matches the predicted chain at raw,
canonical, or family level.

Locked to DeepSeek V4 Flash on the native API (https://api.deepseek.com).

  # put the key in .env (gitignored + cursorignored), then:
  python3 explain_cwe_map/run.py --limit 20 --tag pilot
  python3 explain_cwe_map/run.py --limit 5 --dry-run
  python3 explain_cwe_map/run.py --limit 0 --tag full   # whole pool
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sft_datasets_ready"))
sys.path.insert(0, str(ROOT / "explain_cwe_map"))

from cwe_taxonomy import map_cwe  # noqa: E402
from match import score  # noqa: E402
from prompts import MAP_CWE, is_safe_verdict, review_user_message  # noqa: E402

PAIRS = ROOT / "sft_datasets_ready" / "bigvul_func_after_pairs.jsonl"
OUT_DIR = ROOT / "explain_cwe_map" / "runs"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
ENV_FILE = ROOT / ".env"


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs into os.environ without printing values."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def row_id(rec: dict) -> str:
    blob = f"{rec.get('cve')}|{rec.get('commit_id')}|{rec.get('func_before')[:200]}"
    return hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()[:16]


def load_before(path: Path, *, require_canonical: bool, max_chars: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            code = (rec.get("func_before") or "").strip()
            if not code:
                continue
            if max_chars and len(code) > max_chars:
                continue
            gold = rec.get("cwe") or ""
            if require_canonical and not map_cwe(gold):
                continue
            rows.append(
                {
                    "id": row_id(rec),
                    "cve": rec.get("cve") or "",
                    "cwe": gold,
                    "project": rec.get("project") or "",
                    "commit_id": rec.get("commit_id") or "",
                    "func_before": code,
                    "split": rec.get("split") or "",
                }
            )
    return rows


def sample_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if limit <= 0 or limit >= len(rows):
        out = list(rows)
        rng.shuffle(out)
        return out
    return rng.sample(rows, limit)


def load_done(path: Path) -> set[str]:
    """Ids that already have a successful (non-error) result. Failed rows are retried."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec["id"]
            if rec.get("error"):
                done.discard(rid)
            else:
                done.add(rid)
    return done


def latest_by_id(rows: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for rec in rows:
        by_id[rec["id"]] = rec
    return list(by_id.values())


def make_client() -> OpenAI:
    load_env_file(ENV_FILE)
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    if not key:
        raise SystemExit(
            "Missing DEEPSEEK_API_KEY. Copy .env.example to .env and put the key there."
        )
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL, timeout=120.0)


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    retries: int = 4,
) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"chat failed after {retries} retries: {last}") from last


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    n = max(len(ok), 1)
    keys = ("said_safe", "hit_raw", "hit_canonical", "hit_family", "any_level_hit")
    out = {"n": len(ok), "n_error": sum(1 for r in rows if r.get("error"))}
    for k in keys:
        out[k] = sum(1 for r in ok if r["score"][k]) / n if ok else 0.0
    out["success"] = out["any_level_hit"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two-step LLM: explain func_before, then map CWE hierarchy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Model: deepseek-v4-flash at https://api.deepseek.com\n"
            "Key: .env (DEEPSEEK_API_KEY) or the same variable in the environment.\n"
            "Outputs: explain_cwe_map/runs/{tag}.jsonl and {tag}_summary.json\n"
            "Resume: re-run the same --tag; already scored ids are skipped.\n"
            "Success metric: any_level_hit (raw CWE or canonical class or family)."
        ),
    )
    ap.add_argument("--data", type=Path, default=PAIRS)
    ap.add_argument("--limit", type=int, default=20, help="sample size; 0 = whole pool")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=6000, help="skip longer func_before")
    ap.add_argument("--all-cwe", action="store_true", help="keep gold CWE that we cannot canonicalize")
    ap.add_argument("--dry-run", action="store_true", help="print pool/sample, no API calls")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--sleep", type=float, default=0.4, help="pause between samples, seconds")
    ap.add_argument("--tag", default="pilot", help="output file stem")
    args = ap.parse_args()

    pool = load_before(
        args.data,
        require_canonical=not args.all_cwe,
        max_chars=args.max_chars,
    )
    rows = sample_rows(pool, args.limit, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.tag}.jsonl"
    summary_path = OUT_DIR / f"{args.tag}_summary.json"

    print(
        f"pool={len(pool)} sample={len(rows)} require_canonical={not args.all_cwe} "
        f"out={out_path}"
    )
    if args.dry_run:
        from collections import Counter

        golds = Counter(r["cwe"] for r in rows)
        print("gold CWE:", golds.most_common(12))
        print("sample ids:", [r["id"] for r in rows[:8]])
        return

    client = make_client()
    print(f"model={MODEL} base={DEEPSEEK_BASE_URL}")
    done = load_done(out_path)
    results = []
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    results.append(json.loads(line))

    with out_path.open("a", encoding="utf-8") as fh:
        for rec in tqdm(rows, desc="explain-map"):
            if rec["id"] in done:
                continue
            messages = [{"role": "user", "content": review_user_message(rec["func_before"])}]
            try:
                explanation = chat(
                    client, MODEL, messages, args.temperature, args.max_tokens
                )
                safe = is_safe_verdict(explanation)
                mapping = ""
                if not safe:
                    messages.append({"role": "assistant", "content": explanation})
                    messages.append({"role": "user", "content": MAP_CWE})
                    mapping = chat(
                        client, MODEL, messages, args.temperature, args.max_tokens
                    )
                error = ""
            except Exception as exc:
                explanation, mapping, safe, error = "", "", False, str(exc)
            scored = score(rec["cwe"], mapping, safe)
            if error:
                scored["error"] = True
            row = {
                "id": rec["id"],
                "cve": rec["cve"],
                "project": rec["project"],
                "commit_id": rec["commit_id"],
                "gold_cwe": rec["cwe"],
                "model": MODEL,
                "explanation": explanation,
                "mapping": mapping,
                "error": error,
                "score": scored,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(row)
            done.add(rec["id"])
            if args.sleep:
                time.sleep(args.sleep)

    # summary over this sample's ids only (resume-safe; last row per id wins)
    want = {r["id"] for r in rows}
    sample_results = latest_by_id([r for r in results if r["id"] in want])
    summary = {
        "model": MODEL if not args.dry_run else None,
        "tag": args.tag,
        "pool": len(pool),
        "n_requested": len(rows),
        "n_scored": len(sample_results),
        "metrics": summarize(sample_results),
        "out": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
