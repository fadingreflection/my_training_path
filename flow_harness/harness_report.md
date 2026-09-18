# Flow-explanation harness report

**Gate 2 is not set.** Raw `identifier_precision` 0.2027 is under metric audit — see `ident_audit.md`. Do not treat 0.20 as a generator fail.

Parse-fail 17.4% is a gate record in `ast_check.json` (`gate_decision`). Battery = parseable subset only.

version_label: `v0-9641eead`
spec_hash: `9641eead0b859b16441bfb27ac6c8be44c24361da66d2fc0ef7c83d53d7a7830`
n_rows: 600  n_inputs: 200
dispersion: temperature_fallback

## Metrics (95% bootstrap CI, resample input_id, n=1000)

| metric | value | 95% CI |
|---|---:|---|
| truncation_rate | 0.0000 | [0.0000, 0.0000] |
| repetition_ratio | 0.9999 | [0.9998, 1.0000] |
| identifier_precision | 0.2027 | [0.1921, 0.2123] |
| quote_precision | 0.7924 | [0.7551, 0.8293] |

## verdict_distribution

| verdict | share | count | 95% CI |
|---|---:|---:|---|
| VULNERABLE | 0.8700 | 522 | [0.8200, 0.9150] |
| SAFE | 0.0950 | 57 | [0.0599, 0.1350] |
| INSUFFICIENT | 0.0350 | 21 | [0.0150, 0.0650] |
| UNPARSED | 0.0000 | 0 | [0.0000, 0.0000] |

## v0 vs current

This run is v0 (no prior baseline to diff).
