#!/usr/bin/env python3
"""Log throughput of an explain_cwe_map/run.py OpenRouter run every N seconds.

Reads the run's progress json plus the output jsonl so the instantaneous rate is
visible even when the tqdm bar in the other window is a stale frame.

  python3 explain_cwe_map/watch_rate.py --tag v41flash_full --every 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def read_progress(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v41flash_full")
    ap.add_argument("--every", type=float, default=20.0)
    args = ap.parse_args()

    out_path = RUNS / f"{args.tag}.jsonl"
    prog_path = RUNS / f"{args.tag}_progress.json"
    log_path = RUNS / f"{args.tag}.log"

    prev_rows = count_lines(out_path)
    prev_t = time.monotonic()
    t0 = prev_t
    start_rows = prev_rows
    print(f"watch tag={args.tag} every={args.every:.0f}s rows_now={prev_rows} out={out_path}", flush=True)

    while True:
        time.sleep(args.every)
        now = time.monotonic()
        rows = count_lines(out_path)
        prog = read_progress(prog_path)
        dt = now - prev_t
        inst = (rows - prev_rows) / dt * 60 if dt > 0 else 0.0
        avg = (rows - start_rows) / (now - t0) * 60 if now > t0 else 0.0
        total = int(prog.get("n") or 0)
        scored = int(prog.get("scored") or 0)
        hits = int(prog.get("hits") or 0)
        err = int(prog.get("err") or 0)
        remain = total - scored if total else 0
        eta = remain / inst * 60 if inst > 0 and remain else 0
        errs_403 = 0
        if log_path.is_file():
            try:
                errs_403 = log_path.read_text(encoding="utf-8", errors="ignore").count("Key limit exceeded")
            except Exception:
                pass
        print(
            f"{time.strftime('%H:%M:%S')} rows={rows} scored={scored}/{total} "
            f"hits={hits} err={err} keylimit_403={errs_403} "
            f"inst={inst:.1f}/min avg={avg:.1f}/min eta={fmt_eta(eta)}",
            flush=True,
        )
        prev_rows, prev_t = rows, now


if __name__ == "__main__":
    main()
