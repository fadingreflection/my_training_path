"""Stratified 200-input battery. Margins only, fixed seed, no reserved ids."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ast_util import parse_function
from verdict import parse_verdict

ROOT = Path(__file__).resolve().parents[1]
SINK_NAMES = (
    "memcpy", "memmove", "memset", "strcpy", "strncpy", "strcat", "strncat",
    "sprintf", "snprintf", "vsprintf", "gets", "scanf", "sscanf",
    "malloc", "calloc", "realloc", "free", "alloca",
    "kmalloc", "kfree", "copy_from_user", "copy_to_user",
    "recv", "recvfrom", "send", "read", "write", "pread", "pwrite",
    "system", "exec", "execl", "execv", "popen", "eval",
    "executeQuery", "ExecuteQuery",
)
_SINK_RE = re.compile(r"\b(" + "|".join(re.escape(s) for s in SINK_NAMES) + r")\b")

N_BATTERY = 200
LENGTH_QUOTAS = (65, 70, 65)
PER_TOP_CWE = 18
N_OTHER = 20
SINK_YES = 100


def row_id(rec: dict) -> str:
    blob = f"{rec.get('cve')}|{rec.get('commit_id')}|{rec.get('func_before')[:200]}"
    return hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()[:16]


def load_ids_file(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            ids.add(json.loads(line)["id"])
        else:
            ids.add(line)
    return ids


def load_reserved(harness_dir: Path) -> set[str]:
    names = (
        "reserved_heldout.ids",
        "reserved_mechanism_gt.ids",
        "reserved_claims_calib.ids",
    )
    out: set[str] = set()
    for name in names:
        out |= load_ids_file(harness_dir / name)
    return out


def load_dataset(path: Path) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            code = (rec.get("func_before") or rec.get("code") or "").strip()
            if not code:
                continue
            rid = rec.get("id") or row_id(rec)
            if rid in by_id:
                continue
            by_id[rid] = {
                "id": rid,
                "code": code,
                "cwe": rec.get("cwe") or rec.get("gold_cwe") or "",
                "project": rec.get("project") or "",
                "lang": (rec.get("lang") or "").upper() or "C",
                "cve": rec.get("cve") or "",
                "commit_id": rec.get("commit_id") or "",
                "code_len": len(code),
                "has_sink": bool(_SINK_RE.search(code)),
            }
    return list(by_id.values())


def tertile_edges(lengths: list[int]) -> tuple[int, int]:
    xs = sorted(lengths)
    n = len(xs)
    return xs[n // 3], xs[(2 * n) // 3]


def length_stratum(n: int, lo: int, hi: int) -> int:
    if n <= lo:
        return 0
    if n <= hi:
        return 1
    return 2


def top_cwe_classes(rows: list[dict], k: int = 10) -> tuple[list[str], float]:
    counts = Counter(r["cwe"] for r in rows if r["cwe"])
    ranked = [c for c, _ in counts.most_common()]
    total = sum(counts.values()) or 1
    top = ranked[:k]
    coverage = sum(counts[c] for c in top) / total
    return top, coverage


def annotate(rows: list[dict], top: list[str], lo: int, hi: int) -> None:
    top_set = set(top)
    for rec in rows:
        rec["length_stratum"] = length_stratum(rec["code_len"], lo, hi)
        rec["cwe_bucket"] = rec["cwe"] if rec["cwe"] in top_set else "other"


def _can_take(item: dict, counts: dict, *, relax_sink: bool, relax_len: bool) -> bool:
    ls = item["length_stratum"]
    if not relax_len and counts["length"][ls] >= LENGTH_QUOTAS[ls]:
        return False
    bucket = item["cwe_bucket"]
    cap = PER_TOP_CWE if bucket != "other" else N_OTHER
    if counts["cwe"][bucket] >= cap:
        return False
    sink = bool(item["has_sink"])
    sink_cap = SINK_YES if sink else (N_BATTERY - SINK_YES)
    if not relax_sink and counts["sink"][sink] >= sink_cap:
        return False
    return True


def select_battery(
    eligible: list[dict],
    rng: random.Random,
) -> tuple[list[dict], dict]:
    remaining = list(eligible)
    rng.shuffle(remaining)
    selected: list[dict] = []
    counts = {
        "length": [0, 0, 0],
        "cwe": Counter(),
        "sink": {True: 0, False: 0},
    }

    def add(item: dict) -> None:
        selected.append(item)
        counts["length"][item["length_stratum"]] += 1
        counts["cwe"][item["cwe_bucket"]] += 1
        counts["sink"][bool(item["has_sink"])] += 1

    for relax_sink, relax_len in ((False, False), (True, False), (True, True)):
        still = []
        for item in remaining:
            if len(selected) >= N_BATTERY:
                still.append(item)
                continue
            if _can_take(item, counts, relax_sink=relax_sink, relax_len=relax_len):
                add(item)
            else:
                still.append(item)
        remaining = still
        if len(selected) >= N_BATTERY:
            break
    if len(selected) < N_BATTERY:
        for item in remaining:
            if len(selected) >= N_BATTERY:
                break
            add(item)
    return selected[:N_BATTERY], {
        "length": list(counts["length"]),
        "cwe": dict(counts["cwe"]),
        "sink_yes": counts["sink"][True],
        "sink_no": counts["sink"][False],
        "n": len(selected[:N_BATTERY]),
    }


def _margins_of(selected: list[dict]) -> dict:
    length = [0, 0, 0]
    cwe: Counter = Counter()
    sink_yes = 0
    for rec in selected:
        length[rec["length_stratum"]] += 1
        cwe[rec["cwe_bucket"]] += 1
        sink_yes += int(bool(rec["has_sink"]))
    return {
        "length": length,
        "cwe": dict(cwe),
        "sink_yes": sink_yes,
        "sink_no": len(selected) - sink_yes,
        "n": len(selected),
    }


def rebalance_sink(
    selected: list[dict],
    eligible: list[dict],
    target_yes: int = SINK_YES,
) -> tuple[list[dict], dict]:
    """Swap within length×CWE bucket to hit sink yes/no without breaking other margins."""
    selected = list(selected)
    sel_ids = {r["id"] for r in selected}
    pool = [r for r in eligible if r["id"] not in sel_ids]

    def same_bucket(a: dict, b: dict) -> bool:
        return a["length_stratum"] == b["length_stratum"] and a["cwe_bucket"] == b["cwe_bucket"]

    def sink_yes_n() -> int:
        return sum(1 for r in selected if r["has_sink"])

    def swap(old: dict, new: dict) -> None:
        idx = next(i for i, r in enumerate(selected) if r["id"] == old["id"])
        selected[idx] = new
        sel_ids.discard(old["id"])
        sel_ids.add(new["id"])
        pool.append(old)
        pool.remove(new)

    guard = 0
    while sink_yes_n() < target_yes and guard < 400:
        guard += 1
        nos = [r for r in selected if not r["has_sink"]]
        yes_pool = [r for r in pool if r["has_sink"]]
        pair = next(
            ((n, y) for n in nos for y in yes_pool if same_bucket(n, y)),
            None,
        )
        if pair is None:
            break
        swap(*pair)
    while sink_yes_n() > target_yes and guard < 800:
        guard += 1
        yes = [r for r in selected if r["has_sink"]]
        no_pool = [r for r in pool if not r["has_sink"]]
        pair = next(
            ((y, n) for y in yes for n in no_pool if same_bucket(y, n)),
            None,
        )
        if pair is None:
            break
        swap(*pair)
    return selected, _margins_of(selected)


def build_battery(harness_dir: Path, dataset_path: Path, selection_seed: int) -> dict:
    rows = load_dataset(dataset_path)
    reserved = load_reserved(harness_dir)
    lo, hi = tertile_edges([r["code_len"] for r in rows])
    top, coverage = top_cwe_classes(rows, k=10)
    annotate(rows, top, lo, hi)

    langs = Counter(r["lang"] for r in rows)
    dominant, dom_n = langs.most_common(1)[0]
    monolingual = (dom_n / max(len(rows), 1)) >= 0.95

    parse_ok_rows: list[dict] = []
    parse_fail = 0
    ast_diag_fail_examples = []
    for rec in rows:
        parsed = parse_function(rec["code"])
        rec["parse_ok"] = parsed.parse_ok
        rec["parser"] = parsed.parser
        rec["ast_identifiers"] = parsed.identifiers
        if parsed.parse_ok:
            parse_ok_rows.append(rec)
        else:
            parse_fail += 1
            if len(ast_diag_fail_examples) < 8:
                ast_diag_fail_examples.append(rec["id"])

    fail_rate = parse_fail / max(len(rows), 1)
    traces_path = ROOT / "explain_cwe_map" / "runs" / "glm53flash_full.jsonl"
    teacher_verdict: dict[str, str] = {}
    if traces_path.exists():
        with traces_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                teacher_verdict[rec["id"]] = parse_verdict(rec.get("explanation") or "")

    skipped_reserved = sum(1 for r in parse_ok_rows if r["id"] in reserved)
    skipped_unparsed = 0
    eligible = []
    for rec in parse_ok_rows:
        if rec["id"] in reserved:
            continue
        verd = teacher_verdict.get(rec["id"])
        if verd == "UNPARSED":
            skipped_unparsed += 1
            continue
        eligible.append(rec)

    rng = random.Random(selection_seed)
    selected, margins = select_battery(eligible, rng)
    selected, margins = rebalance_sink(selected, eligible, target_yes=SINK_YES)
    if len(selected) != N_BATTERY:
        raise SystemExit(f"battery size {len(selected)} != {N_BATTERY}")

    lang_note = (
        f"monolingual {dominant} ({dom_n}/{len(rows)}); single language stratum"
        if monolingual
        else f"multilingual {dict(langs)}"
    )

    inputs_path = harness_dir / "fixed_200_inputs.jsonl"
    ast_path = harness_dir / "ast_check.jsonl"
    ast_json = harness_dir / "ast_check.json"
    with inputs_path.open("w", encoding="utf-8") as fh, ast_path.open("w", encoding="utf-8") as ah:
        for rec in selected:
            fh.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "code": rec["code"],
                        "cwe": rec["cwe"],
                        "project": rec["project"],
                        "lang": rec["lang"],
                        "cve": rec["cve"],
                        "commit_id": rec["commit_id"],
                        "code_len": rec["code_len"],
                        "length_stratum": rec["length_stratum"],
                        "cwe_bucket": rec["cwe_bucket"],
                        "has_sink": rec["has_sink"],
                        "ast_identifiers": rec["ast_identifiers"],
                        "parser": rec["parser"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ah.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "parse_ok": True,
                        "parser": rec["parser"],
                        "n_identifiers": len(rec["ast_identifiers"]),
                        "replaced": False,
                        "length_stratum": rec["length_stratum"],
                        "cwe_bucket": rec["cwe_bucket"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    diagnosis = {
        "n_unique_dataset": len(rows),
        "n_parse_ok": len(parse_ok_rows),
        "n_parse_fail": parse_fail,
        "fail_rate": round(fail_rate, 4),
        "fail_rate_gt_5pct": fail_rate > 0.05,
        "fail_examples": ast_diag_fail_examples,
        "n_reserved": len(reserved),
        "n_skipped_reserved_parse_ok": skipped_reserved,
        "n_skipped_unparsed_teacher": skipped_unparsed,
        "n_eligible": len(eligible),
        "length_tertiles": {"p33": lo, "p66": hi},
        "top_cwe": top,
        "top_cwe_coverage": round(coverage, 4),
        "top_cwe_coverage_note": (
            "14-way equal-count dataset: 10 classes cover 10/14≈0.714, below 0.80; "
            "using top-10 anyway, remainder in other"
        ),
        "language": lang_note,
        "selection_seed": selection_seed,
        "margins": margins,
        "sampling": (
            "from parse_ok ∩ not reserved ∩ teacher first-line verdict parseable; "
            "parse failures never enter the 200"
        ),
    }
    ast_json.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return diagnosis
