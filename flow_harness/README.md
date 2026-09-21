# Flow-explanation harness — Phase 0, step 1

Freeze generator **v0** and compare later versions in one command. Metrics are CPU-only. No training.

v0 is the frozen GLM-5.3-flash teacher review traces (`explain_cwe_map/runs/glm53flash_full.jsonl`, sha256 in the manifest). Live OpenRouter generation is used when `OPENROUTER_API_KEY` is set and `backend: auto`.

## v0-fixed baseline (frozen 2026-09-21)

Live GLM-5.3-flash, 200×3 (`temperature_fallback`), 600 rows in `v0_outputs.jsonl`.
Metrics in `v0_baseline.json` / `harness_report.md`. **Do not re-generate** to refresh numbers; re-run `evaluate` only if the metric code changes.

### Gate decisions (Phase 0 / step 1)

| gate | metric | threshold | v0 value | status |
|---|---|---:|---:|---|
| Gate 2 (identifier) | `identifier_precision` (code-shaped vs body AST) | ≥ 0.85 | **0.852** [0.841, 0.863] | **conditional pass** — point ≥ 0.85, CI crosses threshold; **macros weak (0.48)** in category audit |
| Quote | `quote_precision` (code-shaped backticks) | — | **0.866** [0.850, 0.881] | **informational** — proposed threshold ≥ 0.85 for Phase 1; no false-quote audit yet |
| Repetition | `repetition_ratio` | — | 0.999 | **informational** — not a gate; long CoT keeps 8-grams nearly unique |

**Gate 2 identifier** is the freeze gate for “explanation is grounded in this function’s symbols.” It is **not** a formal pass until macro weakness is accepted or fixed in Phase 1.

**Quote** is tracked but **not** a Phase 0 blocker. Point estimate clears a proposed 0.85 bar; promote to gate after a quote miss audit (like `ident_audit.md`).

Legacy **`identifier_precision_raw`** (~0.19) and **`quote_precision_raw`** (~0.80) stay in the report for regression only — never use for gates.

### Metric definitions

- **`identifier_precision`** — code-shaped tokens vs fresh function-body AST (`primitive_type`, `#define`, fields, callees).
- **`identifier_precision_expanded`** — same pred vs expanded snippet gold (diagnostic).
- **`quote_precision`** — code-shaped backticks + expression token fallback.
- Verdict parser accepts glued outputs (`SAFECompeting…`). **UNPARSED = 0/600** on v0-fixed.
- **Parse fail 17.4%** is a gate record (`ast_check.json`). Battery = parseable 82.6% only. Representativeness: Phase 1.

```bash
python3 flow_harness/ident_audit.py
```

## Commands

```bash
# full v0 freeze (seed check → 200-battery → 600 outputs → metrics + baseline)
python3 flow_harness/run_harness.py generate --manifest flow_harness/v0_manifest.yaml
python3 flow_harness/run_harness.py evaluate --outputs flow_harness/v0_outputs.jsonl --write-baseline

# second generate must be a cache hit (no API / no rewrite of flows)
python3 flow_harness/run_harness.py generate --manifest flow_harness/v0_manifest.yaml

# later version
python3 flow_harness/run_harness.py generate --manifest flow_harness/v1_manifest.yaml
python3 flow_harness/run_harness.py evaluate --outputs flow_harness/v1_outputs.jsonl --manifest flow_harness/v1_manifest.yaml
```

Changing any manifest field changes `spec_hash`. Generate will refuse to overwrite `v0_outputs.jsonl` for a drifted spec unless `--force`. Evaluate refuses a v0 comparison if `normalization` differs.

## Layout

| file | role |
|---|---|
| `v0_manifest.yaml` | frozen generator spec |
| `fixed_200_inputs.jsonl` | battery (do not edit without a major version bump) |
| `v0_outputs.jsonl` | 200 × 3 runs |
| `v0_baseline.json` | metrics snapshot (code-like + raw identifier_precision) |
| `code_idents.py` | Gate 2 code-shaped token extractor |
| `ident_audit.md` / `ident_audit.json` | false-token audit; category split |
| `seed_check.json` | whether seeds change the text |
| `ast_check.jsonl` / `ast_check.json` | parse_ok battery + dataset fail-rate gate record |
| `reserved_*.ids` | ids excluded from the battery |
| `harness_report.md` | human table |

Reserved holdout ids are copied from `explain_cwe_map/sft/glm53flash_hits/holdout_ids.jsonl`. The other two reserved files are empty interface stubs.
