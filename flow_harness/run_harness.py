#!/usr/bin/env python3
"""Phase 0 step 1: freeze a flow-explanation generator and compare later versions.

  python3 flow_harness/run_harness.py generate --manifest flow_harness/v0_manifest.yaml
  python3 flow_harness/run_harness.py evaluate --outputs flow_harness/v0_outputs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from battery import N_BATTERY, build_battery, load_dataset  # noqa: E402
from client import GenResult, api_key_available, make_generator  # noqa: E402
from manifest import load_manifest, normalization_fingerprint, version_label  # noqa: E402
from metrics import evaluate_outputs  # noqa: E402
from verdict import parse_verdict  # noqa: E402

DATASET = ROOT / "raw_data" / "bigvul_func_before_balanced14.jsonl"


def _jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ensure_reserved_stubs() -> None:
    mapping = {
        "reserved_mechanism_gt.ids": "ids reserved for mechanism ground-truth labeling; empty stub",
        "reserved_claims_calib.ids": "ids reserved for claim calibration; empty stub",
        "reserved_heldout.ids": "ids reserved as CWE-map holdout; filled from glm53flash holdout",
    }
    holdout = ROOT / "explain_cwe_map" / "sft" / "glm53flash_hits" / "holdout_ids.jsonl"
    held_path = HERE / "reserved_heldout.ids"
    if not held_path.exists():
        if holdout.exists():
            ids = []
            seen = set()
            for rec in _jsonl(holdout):
                if rec["id"] not in seen:
                    seen.add(rec["id"])
                    ids.append({"id": rec["id"]})
            _write_jsonl(held_path, ids)
        else:
            held_path.write_text("", encoding="utf-8")
    for name, note in mapping.items():
        path = HERE / name
        if not path.exists():
            path.write_text(f"# {note}\n", encoding="utf-8")


def run_seed_check(manifest: dict, force: bool = False) -> dict:
    path = HERE / "seed_check.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    rows = load_dataset(DATASET)
    if not rows:
        raise SystemExit("dataset empty; cannot seed-check")
    rec = rows[0]
    rec = {**rec, "code": rec["code"]}
    gen = make_generator(manifest)
    sampling = manifest["sampling"]
    seeds = list(sampling["seeds"])
    s0, s1 = int(seeds[0]), int(seeds[1] if len(seeds) > 1 else seeds[0] + 1)
    kwargs = dict(
        max_tokens=int(sampling["max_tokens"]),
        top_p=float(sampling["top_p"]),
        prompt=manifest["prompt"],
    )
    a = gen.generate(rec, seed=s0, temperature=float(sampling["temperature"]), **kwargs)
    b = gen.generate(rec, seed=s1, temperature=float(sampling["temperature"]), **kwargs)
    functional = (a.flow_text or "") != (b.flow_text or "") and bool(a.flow_text) and bool(b.flow_text)
    temps = list(sampling.get("fallback_temperatures") or [0.0, 0.4, 0.8])
    report = {
        "sample_id": rec["id"],
        "backend": a.backend,
        "seed_a": s0,
        "seed_b": s1,
        "seeds_functional": functional,
        "identical_outputs": a.flow_text == b.flow_text,
        "api_key_available": api_key_available(),
        "fallback": None if functional else "temperature_x3",
        "fallback_temperatures": None if functional else temps,
        "note": (
            "OpenRouter seeds changed the text."
            if functional
            else (
                "Seeds produced identical text. Dispersion mode is temperature×3 "
                f"{temps}. v0 replay backend is a frozen temperature=0 teacher "
                "trace: the three slots replay the same flow (honest freeze, "
                "zero extra variance until live API generate --force)."
            )
        ),
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def dispersion_slots(manifest: dict, seed_check: dict) -> list[dict]:
    sampling = manifest["sampling"]
    seeds = [int(s) for s in sampling["seeds"]]
    base_t = float(sampling["temperature"])
    if seed_check.get("seeds_functional"):
        return [{"seed": s, "temperature": base_t, "mode": "seed"} for s in seeds]
    temps = list(seed_check.get("fallback_temperatures") or sampling.get("fallback_temperatures") or [0.0, 0.4, 0.8])
    while len(temps) < len(seeds):
        temps.append(temps[-1])
    return [
        {"seed": seeds[i], "temperature": float(temps[i]), "mode": "temperature_fallback"}
        for i in range(len(seeds))
    ]


def outputs_path(manifest: dict) -> Path:
    return HERE / f"{manifest['version']}_outputs.jsonl"


def outputs_meta_path(manifest: dict) -> Path:
    return HERE / f"{manifest['version']}_outputs.meta.json"


def ensure_battery(manifest: dict, force: bool = False) -> dict:
    inputs = HERE / "fixed_200_inputs.jsonl"
    if inputs.exists() and not force:
        return {"n": sum(1 for _ in _jsonl(inputs)), "cached": True}
    return build_battery(HERE, DATASET, int(manifest["selection_seed"]))


def cmd_generate(args: argparse.Namespace) -> None:
    ensure_reserved_stubs()
    manifest = load_manifest(args.manifest)
    seed_check = run_seed_check(manifest, force=args.force_seed_check)
    ensure_battery(manifest, force=args.force_battery)
    inputs = _jsonl(HERE / "fixed_200_inputs.jsonl")
    if len(inputs) != N_BATTERY:
        raise SystemExit(f"fixed_200_inputs.jsonl has {len(inputs)} rows, want {N_BATTERY}")
    slots = dispersion_slots(manifest, seed_check)
    out_path = outputs_path(manifest)
    meta_path = outputs_meta_path(manifest)
    spec = manifest["spec_hash"]
    label = version_label(manifest)

    existing = _jsonl(out_path)
    if existing and not args.force:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if meta.get("spec_hash") != spec:
            alt = HERE / f"{label}_outputs.jsonl"
            raise SystemExit(
                f"manifest spec_hash changed ({meta.get('spec_hash')} → {spec}). "
                f"This is a new version {label}. Refusing to overwrite {out_path.name}. "
                f"Rename version in the yaml and write {alt.name} (or pass --force)."
            )
        have = {(r["input_id"], r["seed"]) for r in existing}
        want = {(rec["id"], slot["seed"]) for rec in inputs for slot in slots}
        if have >= want:
            print(f"cache hit {out_path} n={len(existing)} spec={spec[:8]} (no generate)")
            return
        print(f"resume {out_path} have={len(have)} want={len(want)}")

    if args.force and out_path.exists():
        out_path.unlink()
        existing = []

    have = {(r["input_id"], r["seed"]) for r in existing}
    gen = make_generator(manifest)
    sampling = manifest["sampling"]
    n_written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for rec in inputs:
            for slot in slots:
                key = (rec["id"], slot["seed"])
                if key in have:
                    continue
                result: GenResult = gen.generate(
                    rec,
                    seed=slot["seed"],
                    temperature=slot["temperature"],
                    max_tokens=int(sampling["max_tokens"]),
                    top_p=float(sampling["top_p"]),
                    prompt=manifest["prompt"],
                )
                verd = parse_verdict(result.flow_text) if not result.error else "UNPARSED"
                row = {
                    "input_id": rec["id"],
                    "seed": slot["seed"],
                    "temperature": slot["temperature"],
                    "dispersion_mode": slot["mode"],
                    "flow_text": result.flow_text,
                    "tokens_used": result.tokens_used,
                    "elapsed_ms": result.elapsed_ms,
                    "truncated_flag": bool(result.truncated_flag),
                    "verdict": verd,
                    "backend": result.backend,
                    "error": result.error,
                    "spec_hash": spec,
                    "version_label": label,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                have.add(key)
    meta = {
        "spec_hash": spec,
        "version": manifest["version"],
        "version_label": label,
        "normalization": list(manifest["normalization"]),
        "n_written_this_call": n_written,
        "n_total": len(have),
        "seed_check": seed_check,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {n_written} new rows → {out_path} total_keys={len(have)}")


def _fmt_ci(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['value']:.4f}  [{lo:.4f}, {hi:.4f}]"


def write_report(current: dict, v0: dict | None, path_md: Path, path_json: Path, refused: str | None) -> None:
    path_json.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Flow-explanation harness report",
        "",
        f"version_label: `{current.get('version_label')}`",
        f"spec_hash: `{current.get('spec_hash')}`",
        f"n_rows: {current['metrics']['n_rows']}  n_inputs: {current['metrics']['n_inputs']}",
        f"dispersion: {current.get('dispersion_mode')}",
        "",
        "## Metrics (95% bootstrap CI, resample input_id, n=1000)",
        "",
        "| metric | value | 95% CI |",
        "|---|---:|---|",
    ]
    m = current["metrics"]
    for key in ("truncation_rate", "repetition_ratio", "identifier_precision", "quote_precision"):
        lines.append(f"| {key} | {m[key]['value']:.4f} | [{m[key]['ci95'][0]:.4f}, {m[key]['ci95'][1]:.4f}] |")
    lines += ["", "## verdict_distribution", "", "| verdict | share | count | 95% CI |", "|---|---:|---:|---|"]
    for lab in ("VULNERABLE", "SAFE", "INSUFFICIENT", "UNPARSED"):
        share = m["verdict_distribution"].get(lab, 0.0)
        cnt = m["verdict_counts"].get(lab, 0)
        ci = m["verdict_distribution_ci"][lab]["ci95"]
        lines.append(f"| {lab} | {share:.4f} | {cnt} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    if refused:
        lines += ["", "## v0 comparison", "", f"**refused:** {refused}"]
    elif v0:
        lines += ["", "## v0 vs current", "", "| metric | v0 | current |", "|---|---:|---:|"]
        vm = v0["metrics"]
        for key in ("truncation_rate", "repetition_ratio", "identifier_precision", "quote_precision"):
            lines.append(f"| {key} | {_fmt_ci(vm[key])} | {_fmt_ci(m[key])} |")
        lines.append("| UNPARSED | " f"{vm['unparsed_count']} | {m['unparsed_count']} |")
    else:
        lines += ["", "## v0 vs current", "", "This run is v0 (no prior baseline to diff)."]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_evaluate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    rows = _jsonl(args.outputs)
    if not rows:
        raise SystemExit(f"no rows in {args.outputs}")
    inputs = {r["id"]: r for r in _jsonl(HERE / "fixed_200_inputs.jsonl")}
    missing = [r["input_id"] for r in rows if r["input_id"] not in inputs]
    if missing:
        raise SystemExit(f"outputs reference unknown ids e.g. {missing[:5]}")
    t0 = time.monotonic()
    metrics = evaluate_outputs(
        rows,
        inputs,
        norm_rules=list(manifest["normalization"]),
        bootstrap_seed=int(manifest.get("bootstrap_seed", 20260918)),
        n_boot=int(manifest.get("bootstrap_resamples", 1000)),
    )
    elapsed = time.monotonic() - t0
    current = {
        "version": manifest["version"],
        "version_label": version_label(manifest),
        "spec_hash": manifest["spec_hash"],
        "normalization": list(manifest["normalization"]),
        "normalization_fingerprint": normalization_fingerprint(manifest),
        "dispersion_mode": rows[0].get("dispersion_mode"),
        "evaluate_seconds": round(elapsed, 3),
        "metrics": metrics,
    }
    baseline_path = HERE / "v0_baseline.json"
    v0 = None
    refused = None
    same_v0 = args.outputs.resolve() == (HERE / "v0_outputs.jsonl").resolve()
    if baseline_path.exists() and not same_v0:
        v0 = json.loads(baseline_path.read_text(encoding="utf-8"))
        if v0.get("normalization_fingerprint") != current["normalization_fingerprint"]:
            refused = (
                "normalization spec differs from v0; harness refuses the comparison "
                f"(v0={v0.get('normalization_fingerprint')} current={current['normalization_fingerprint']})"
            )
            v0 = None
    write_report(current, v0, HERE / "harness_report.md", HERE / "harness_report.json", refused)
    if args.write_baseline or (manifest["version"] == "v0" and not baseline_path.exists()):
        baseline_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {baseline_path}")
    print(
        f"evaluate n={metrics['n_rows']} UNPARSED={metrics['unparsed_count']} "
        f"in {elapsed:.2f}s → {HERE / 'harness_report.md'}"
    )
    if metrics["n_rows"] != 600 and manifest["version"] == "v0":
        print(f"warning: v0 expected 600 rows, got {metrics['n_rows']}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flow-explanation regression harness (Phase 0 / step 1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--manifest", type=Path, default=HERE / "v0_manifest.yaml")
    g.add_argument("--force", action="store_true")
    g.add_argument("--force-battery", action="store_true")
    g.add_argument("--force-seed-check", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--outputs", type=Path, default=HERE / "v0_outputs.jsonl")
    e.add_argument("--manifest", type=Path, default=HERE / "v0_manifest.yaml")
    e.add_argument("--write-baseline", action="store_true")
    s = sub.add_parser("seed-check")
    s.add_argument("--manifest", type=Path, default=HERE / "v0_manifest.yaml")
    s.add_argument("--force", action="store_true")
    b = sub.add_parser("build-battery")
    b.add_argument("--manifest", type=Path, default=HERE / "v0_manifest.yaml")
    b.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "seed-check":
        ensure_reserved_stubs()
        man = load_manifest(args.manifest)
        print(json.dumps(run_seed_check(man, force=args.force), indent=2))
    elif args.cmd == "build-battery":
        ensure_reserved_stubs()
        man = load_manifest(args.manifest)
        print(json.dumps(ensure_battery(man, force=args.force), indent=2))


if __name__ == "__main__":
    main()
