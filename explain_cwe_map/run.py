#!/usr/bin/env python3
"""Explain BigVul func_before, then map the explanation onto a CWE hierarchy.

Only vulnerable bodies (func_before). Two chat turns via OpenRouter.
Success = gold CWE matches the predicted chain at raw, canonical, or family.

Default model: GLM 5.3 Flash. Grok 4.6, Gemini 3.8 Flash, and DeepSeek V4.1 Flash
are allowed via --model. Full DeepSeek runs must use a distinct --tag so GLM
outputs are never overwritten.

  python3 explain_cwe_map/make_balanced14.py
  python3 explain_cwe_map/run.py --tag smoke
  python3 explain_cwe_map/run.py --limit 5 --dry-run
  python3 explain_cwe_map/run.py --limit 0 --tag full   # full equal-count 14-way set

Logs: explain_cwe_map/runs/{tag}.log (also stderr). One line before each API
call so a hang is visible while OpenRouter is still thinking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Lock, Thread

from openai import OpenAI
from tqdm import tqdm

log = logging.getLogger("explain_cwe_map")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain_cwe_map"))

from balanced import LARGE_CWE, balance_equal, keep_large, stratified_sample  # noqa: E402
from cwe_taxonomy import map_cwe  # noqa: E402
from match import score  # noqa: E402
from prompts import (  # noqa: E402
    MAP_CWE,
    is_insufficient_verdict,
    is_safe_verdict,
    review_user_message,
    skip_cwe_mapping,
)

PAIRS = ROOT / "raw_data" / "bigvul_func_before_balanced14.jsonl"
SRC_ALL = ROOT / "raw_data" / "bigvul_func_before.jsonl"
OUT_DIR = ROOT / "explain_cwe_map" / "runs"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-5.3-flash"
ALLOWED_MODELS = (
    DEFAULT_MODEL,
    "x-ai/grok-4.6",
    "google/gemini-3.8-flash",
    "deepseek/deepseek-v4.1-flash",
)
MODEL = DEFAULT_MODEL
ENV_FILE = ROOT / ".env"


class _TqdmHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            pass


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = _TqdmHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def _safe_err(exc: BaseException) -> str:
    text = str(exc)
    for token in ("sk-", "Bearer ", "OPENROUTER_API_KEY="):
        idx = text.find(token)
        if idx >= 0:
            text = text[: idx + len(token)] + "[REDACTED]"
    return text[:500]


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
            if gold not in LARGE_CWE:
                continue
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
    return stratified_sample(rows, limit, seed)


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


def load_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def load_id_order(path: Path) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    if not path.exists():
        return order
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rid = json.loads(line)["id"]
            if rid not in seen:
                seen.add(rid)
                order.append(rid)
    return order


def latest_by_id(rows: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for rec in rows:
        by_id[rec["id"]] = rec
    return list(by_id.values())


def make_client() -> OpenAI:
    load_env_file(ENV_FILE)
    key = os.environ.get("OPENROUTER_API_KEY") or ""
    if not key:
        raise SystemExit(
            "Missing OPENROUTER_API_KEY. Put it in .env (OPENROUTER_API_KEY=...) "
            "or export it in the shell."
        )
    timeout = 600.0 if any(x in MODEL for x in ("grok", "gemini", "deepseek")) else 180.0
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, timeout=timeout)


def _dump_msg(msg) -> dict:
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    return {
        "content": getattr(msg, "content", None),
        "reasoning": getattr(msg, "reasoning", None),
        "reasoning_details": getattr(msg, "reasoning_details", None),
    }


def _reasoning_blob(dumped: dict) -> str:
    parts: list[str] = []
    if dumped.get("reasoning"):
        parts.append(str(dumped["reasoning"]))
    details = dumped.get("reasoning_details") or []
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                for key in ("text", "summary", "content"):
                    if item.get(key):
                        parts.append(str(item[key]))
                        break
            elif item:
                parts.append(str(item))
    return "\n".join(parts).strip()


def _slice_from_verdict(text: str) -> str:
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        head = line.strip().lstrip("#").strip().upper()
        if (
            head.startswith("VULNERABLE")
            or head.startswith("INSUFFICIENT_INFO")
            or head.startswith("SAFE")
            or "CWE-" in line.upper()
        ):
            return "\n".join(lines[i:]).strip()
    return (text or "").strip()


def _visible_text(msg) -> tuple[str, str]:
    dumped = _dump_msg(msg)
    content = (dumped.get("content") or "").strip()
    reasoning = _reasoning_blob(dumped)
    if content:
        return content, "content"
    if reasoning:
        return _slice_from_verdict(reasoning), "reasoning"
    return "", "empty"


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    retries: int = 4,
    step: str = "chat",
) -> str:
    if model not in ALLOWED_MODELS:
        raise RuntimeError(f"refusing chat with {model!r}; allowed={ALLOWED_MODELS}")
    last: Exception | None = None
    if "grok" in model or "gemini" in model:
        effort, wait_s = "high", 600
    elif "deepseek" in model:
        effort, wait_s = "low", 600
    else:
        effort, wait_s = "low", 180
    for attempt in range(retries):
        t0 = time.monotonic()
        log.info("%s attempt %d/%d waiting OpenRouter timeout=%ds", step, attempt + 1, retries, wait_s)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "models": [model],
                    "reasoning": {"effort": effort},
                },
            )
            used = (getattr(resp, "model", None) or "").strip()
            slug = model.split("/")[-1]
            if used and slug not in used and model not in used:
                raise RuntimeError(f"OpenRouter returned unexpected model {used!r} (wanted {model!r})")
            msg = resp.choices[0].message
            text, src = _visible_text(msg)
            dt = time.monotonic() - t0
            usage = getattr(resp, "usage", None)
            log.info(
                "%s ok in %.1fs chars=%d src=%s model=%s usage=%s",
                step,
                dt,
                len(text),
                src,
                used or "?",
                usage,
            )
            if not text:
                dumped = _dump_msg(msg)
                log.warning(
                    "%s empty payload keys=%s",
                    step,
                    {k: (len(str(v)) if v is not None else 0) for k, v in dumped.items()},
                )
                raise RuntimeError("empty content after reasoning; retry")
            return text
        except Exception as exc:
            last = exc
            err = _safe_err(exc)
            wait = min(2 ** attempt, 20)
            if "429" in err:
                wait = min(15 * (attempt + 1), 90)
            log.warning(
                "%s fail attempt %d/%d after %.1fs: %s; sleep %.0fs",
                step,
                attempt + 1,
                retries,
                time.monotonic() - t0,
                _safe_err(exc),
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"chat failed after {retries} retries: {last}") from last


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    n = max(len(ok), 1)
    keys = ("said_safe", "said_insufficient", "hit_raw", "hit_canonical", "hit_family", "any_level_hit")
    out = {"n": len(ok), "n_error": sum(1 for r in rows if r.get("error"))}
    for k in keys:
        out[k] = sum(1 for r in ok if r["score"][k]) / n if ok else 0.0
    out["success"] = out["any_level_hit"]
    return out


def process_sample(
    client: OpenAI,
    rec: dict,
    *,
    temperature: float,
    max_tokens: int,
    index: int,
    n: int,
    verbose: bool,
) -> dict:
    t_sample = time.monotonic()
    log.info(
        "[%d/%d] start id=%s cve=%s gold=%s project=%s code_chars=%d",
        index,
        n,
        rec["id"],
        rec["cve"],
        rec["cwe"],
        rec["project"],
        len(rec["func_before"]),
    )
    messages = [{"role": "user", "content": review_user_message(rec["func_before"])}]
    try:
        explanation = chat(
            client,
            MODEL,
            messages,
            temperature,
            max_tokens,
            step=f"review {rec['id']}",
        )
        safe = is_safe_verdict(explanation)
        insufficient = is_insufficient_verdict(explanation)
        skip = skip_cwe_mapping(explanation)
        if verbose:
            tqdm.write(
                f"\n===== [{index}/{n}] {rec['id']} gold={rec['cwe']} REVIEW =====\n"
                f"{explanation or '[empty]'}\n"
            )
        log.info(
            "[%d/%d] review done safe=%s insufficient=%s skip_map=%s",
            index,
            n,
            safe,
            insufficient,
            skip,
        )
        mapping = ""
        if not skip:
            messages.append({"role": "assistant", "content": explanation})
            messages.append({"role": "user", "content": MAP_CWE})
            mapping = chat(
                client,
                MODEL,
                messages,
                temperature,
                max_tokens,
                step=f"map {rec['id']}",
            )
            if verbose:
                tqdm.write(
                    f"===== [{index}/{n}] {rec['id']} gold={rec['cwe']} MAP =====\n"
                    f"{mapping or '[empty]'}\n"
                )
        elif verbose:
            tqdm.write(
                f"===== [{index}/{n}] {rec['id']} skip MAP "
                f"({'SAFE' if safe else 'INSUFFICIENT_INFO'})\n"
            )
        error = ""
    except Exception as exc:
        explanation, mapping, safe, insufficient, error = (
            "",
            "",
            False,
            False,
            _safe_err(exc),
        )
        log.error("[%d/%d] sample failed: %s", index, n, error)
    scored = score(
        rec["cwe"],
        mapping,
        said_safe=safe,
        said_insufficient=insufficient,
    )
    if error:
        scored["error"] = True
    return {
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
        "_elapsed": time.monotonic() - t_sample,
        "_index": index,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two-step LLM: explain func_before, then map CWE hierarchy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default model: z-ai/glm-5.3-flash at https://openrouter.ai/api/v1\n"
            "Override with --model. Key: .env (OPENROUTER_API_KEY).\n"
            "Outputs: explain_cwe_map/runs/{tag}.jsonl, {tag}_summary.json, {tag}.log\n"
            "Resume: re-run the same --tag; already scored ids are skipped.\n"
            "Holdout eval: --only-from <ids.jsonl> --limit 0 --tag <name>\n"
            "Success metric: any_level_hit (raw CWE or canonical class or family)."
        ),
    )
    ap.add_argument("--data", type=Path, default=PAIRS)
    ap.add_argument("--limit", type=int, default=5, help="sample size, stratified by CWE; 0 = whole balanced pool")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=6000, help="skip longer func_before before balancing")
    ap.add_argument(
        "--canonical-only",
        action="store_true",
        help="drop gold CWE that map_cwe cannot canonicalize (off: keep all 14 labels)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print pool/sample, no API calls")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause after each sample (sequential only)")
    ap.add_argument("--workers", type=int, default=16, help="parallel OpenRouter requests")
    ap.add_argument("--tag", default="pilot", help="output file stem")
    ap.add_argument(
        "--skip-from",
        type=Path,
        action="append",
        default=[],
        help="jsonl whose ids to exclude from the sample (repeatable)",
    )
    ap.add_argument(
        "--only-from",
        type=Path,
        help="reuse ids from this jsonl in the same order (for prompt A/B on the same functions)",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model id (default: z-ai/glm-5.3-flash)",
    )
    args = ap.parse_args()
    global MODEL  # noqa: PLW0603
    MODEL = args.model

    src = args.data if args.data.exists() else SRC_ALL
    loaded = load_before(
        src,
        require_canonical=args.canonical_only,
        max_chars=args.max_chars,
    )
    loaded = keep_large(loaded)
    pool, per_class = balance_equal(loaded, args.seed)
    skip_ids: set[str] = set()
    for path in args.skip_from:
        skip_ids |= load_ids(path)
    if skip_ids:
        pool = [r for r in pool if r["id"] not in skip_ids]
    if args.only_from:
        by_id = {r["id"]: r for r in pool}
        missing = [i for i in load_id_order(args.only_from) if i not in by_id]
        if missing:
            raise SystemExit(f"--only-from ids not in pool: {missing[:8]}")
        rows = [by_id[i] for i in load_id_order(args.only_from) if i in by_id]
        if args.limit > 0:
            rows = rows[: args.limit]
    else:
        rows = sample_rows(pool, args.limit, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.tag}.jsonl"
    summary_path = OUT_DIR / f"{args.tag}_summary.json"
    log_path = OUT_DIR / f"{args.tag}.log"
    setup_logging(log_path)

    print(
        f"src={src} loaded={len(loaded)} balanced_pool={len(pool)} "
        f"per_class={per_class} sample={len(rows)} canonical_only={args.canonical_only} "
        f"out={out_path} log={log_path}"
    )
    log.info(
        "pid=%s model=%s temperature=%s src=%s loaded=%d pool=%d per_class=%d sample=%d "
        "canonical_only=%s dry_run=%s workers=%s out=%s",
        os.getpid(),
        MODEL,
        args.temperature,
        src,
        len(loaded),
        len(pool),
        per_class,
        len(rows),
        args.canonical_only,
        args.dry_run,
        args.workers,
        out_path,
    )
    if args.dry_run:
        from collections import Counter

        golds = Counter(r["cwe"] for r in rows)
        print("gold CWE:", golds.most_common(20))
        print("sample ids:", [r["id"] for r in rows[:8]])
        log.info("dry-run golds=%s ids=%s", golds.most_common(20), [r["id"] for r in rows[:8]])
        return

    client = make_client()
    print(f"model={MODEL} base={OPENROUTER_BASE_URL}")
    done = load_done(out_path)
    n = len(rows)
    n_skip = sum(1 for rec in rows if rec["id"] in done)
    log.info("openrouter=%s already_ok=%d todo=%d", OPENROUTER_BASE_URL, n_skip, n - n_skip)
    results = []
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    results.append(json.loads(line))

    with out_path.open("a", encoding="utf-8") as fh:
        pending = [(i, rec) for i, rec in enumerate(rows, 1) if rec["id"] not in done]
        workers = max(1, args.workers)
        verbose = workers == 1
        write_lock = Lock()
        want = {r["id"] for r in rows}
        stats = {"written": 0, "hits": 0, "errors": 0, "last_id": ""}
        t0 = time.monotonic()
        stop_hb = Event()
        progress_path = OUT_DIR / f"{args.tag}_progress.json"

        def emit_progress() -> None:
            elapsed = time.monotonic() - t0
            with write_lock:
                written = stats["written"]
                hits = stats["hits"]
                errors = stats["errors"]
                last_id = stats["last_id"]
            scored = n_skip + written
            remain = n - scored
            rate = written / elapsed if elapsed > 0 else 0.0
            eta = remain / rate if rate > 0 else None
            payload = {
                "model": MODEL,
                "tag": args.tag,
                "scored": scored,
                "n": n,
                "new": written,
                "skip": n_skip,
                "hits": hits,
                "err": errors,
                "remain": remain,
                "elapsed_s": round(elapsed, 1),
                "rate_per_min": round(rate * 60, 2),
                "eta_s": None if eta is None else round(eta),
                "last_id": last_id,
                "out": str(out_path),
            }
            log.info(
                "progress scored=%d/%d new=%d skip=%d hits=%d err=%d remain=%d "
                "elapsed=%.0fs rate=%.2f/min eta=%s last=%s",
                scored,
                n,
                written,
                n_skip,
                hits,
                errors,
                remain,
                elapsed,
                rate * 60,
                "?" if eta is None else f"{eta:.0f}s",
                last_id or "-",
            )
            progress_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        def heartbeat() -> None:
            while not stop_hb.is_set():
                emit_progress()
                if stop_hb.wait(20.0):
                    break
            emit_progress()

        def consume(row: dict) -> None:
            elapsed = row.pop("_elapsed", 0.0)
            index = row.pop("_index", 0)
            with write_lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                results.append(row)
                stats["written"] += 1
                stats["last_id"] = row["id"]
                if row.get("error"):
                    stats["errors"] += 1
                elif row["score"].get("any_level_hit"):
                    stats["hits"] += 1
                scored_n = len(latest_by_id([r for r in results if r["id"] in want]))
                if scored_n % 50 == 0:
                    summary_path.write_text(
                        json.dumps(
                            {
                                "model": MODEL,
                                "tag": args.tag,
                                "pool": len(pool),
                                "n_requested": len(rows),
                                "n_scored": scored_n,
                                "metrics": summarize(latest_by_id([r for r in results if r["id"] in want])),
                                "out": str(out_path),
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            log.info(
                "[%d/%d] wrote id=%s err=%s hit=%s elapsed=%.1fs expl_chars=%d map_chars=%d",
                index,
                n,
                row["id"],
                bool(row.get("error")),
                bool(row["score"].get("any_level_hit")),
                elapsed,
                len(row.get("explanation") or ""),
                len(row.get("mapping") or ""),
            )

        log.info("workers=%d pending=%d", workers, len(pending))
        print(f"workers={workers} pending={len(pending)}")
        hb = Thread(target=heartbeat, name="progress-heartbeat", daemon=True)
        hb.start()
        try:
            if workers == 1:
                for i, rec in tqdm(pending, desc="explain-map"):
                    consume(
                        process_sample(
                            client,
                            rec,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            index=i,
                            n=n,
                            verbose=verbose,
                        )
                    )
                    if args.sleep:
                        time.sleep(args.sleep)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool_ex:
                    futs = {
                        pool_ex.submit(
                            process_sample,
                            client,
                            rec,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            index=i,
                            n=n,
                            verbose=False,
                        ): rec["id"]
                        for i, rec in pending
                    }
                    for fut in tqdm(as_completed(futs), total=len(futs), desc="explain-map"):
                        consume(fut.result())
        finally:
            stop_hb.set()
            hb.join(timeout=5)

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
    log.info("summary %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
