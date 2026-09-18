# Flow-explanation harness — Phase 0, step 1

Freeze generator **v0** and compare later versions in one command. Metrics are CPU-only. No training.

v0 is the frozen GLM-5.3-flash teacher review traces (`explain_cwe_map/runs/glm53flash_full.jsonl`, sha256 in the manifest). Live OpenRouter generation is used when `OPENROUTER_API_KEY` is set and `backend: auto`.

## Gate notes (read before acting on numbers)

- **`identifier_precision` 0.20 in `v0_baseline.json` is the raw frozen number.** It is **under metric audit**. Gate 2 is **not set**. Do not start constrained decoding or a model swap on the back of 0.20.
- Audit (`ident_audit.md`): 0.20 is mostly English prose counted as identifiers (`attacker`, `path`, `sink`). Code-like tokens vs function-body AST are **0.91**; vs expanded snippet gold **0.93**. Locals/params and fields are high; **macros** stay the weak class (~0.57–0.64). 50-token review: 48/50 ast-gold limitation, 2 invented.
- **Parse fail 17.4%** is a gate record, not a footnote (`ast_check.json` → `gate_decision`). The battery is the parseable 82.6% only. Phase 0 conclusions do not cover the other 17%. Cause still **TBD** (provisional: chopped signatures, preprocessor, C++/kernel macros). Representativeness: evaluate in Phase 1.

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
| `v0_baseline.json` | metrics snapshot (raw identifier_precision kept) |
| `ident_audit.md` / `ident_audit.json` | metric audit; gate 2 not set |
| `seed_check.json` | whether seeds change the text |
| `ast_check.jsonl` / `ast_check.json` | parse_ok battery + dataset fail-rate gate record |
| `reserved_*.ids` | ids excluded from the battery |
| `harness_report.md` | human table |

Reserved holdout ids are copied from `explain_cwe_map/sft/glm53flash_hits/holdout_ids.jsonl`. The other two reserved files are empty interface stubs.
