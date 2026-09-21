# Flow-explanation harness report

version_label: `v0-fc884d0d`
spec_hash: `fc884d0d179f7bbe39e0365167ce644f368b843258579d58554aed467e16f948`
n_rows: 600  n_inputs: 200
dispersion: temperature_fallback

## Metrics (95% bootstrap CI, resample input_id, n=1000)

| metric | value | 95% CI |
|---|---:|---|
| truncation_rate | 0.0283 | [0.0150, 0.0417] |
| repetition_ratio | 0.9991 | [0.9974, 1.0000] |
| repetition_ratio_nonempty | 0.9991 | [0.9974, 1.0000] |
| identifier_precision | 0.8520 | [0.8406, 0.8629] |
| identifier_precision_raw | 0.1920 | [0.1824, 0.2008] |
| identifier_precision_expanded | 0.8842 | [0.8744, 0.8943] |
| quote_precision | 0.8658 | [0.8497, 0.8812] |
| quote_precision_raw | 0.8008 | [0.7772, 0.8233] |

_identifier_precision = code-shaped tokens vs function-body AST (Gate 2 candidate). identifier_precision_raw = legacy all-words metric. quote_precision = code-shaped backticks only; quote_precision_raw = all backticks ≥8 chars._

rows with zero code-shaped idents: 1 / 600
rows with zero code-shaped quotes: 2 / 600
repetition_ratio_nonempty: 0.9991 [0.9974, 1.0000] (same as all-rows when no empty flows)
code-shaped quote spans: 9206 / raw 11586

## verdict_distribution

| verdict | share | count | 95% CI |
|---|---:|---:|---|
| VULNERABLE | 0.7067 | 424 | [0.6550, 0.7584] |
| SAFE | 0.1700 | 102 | [0.1283, 0.2117] |
| INSUFFICIENT | 0.1233 | 74 | [0.0900, 0.1583] |
| UNPARSED | 0.0000 | 0 | [0.0000, 0.0000] |

## Gate status (v0-fixed)

| gate | value | 95% CI | status |
|---|---:|---|---|
| identifier_precision | 0.8520 | [0.8406, 0.8629] | conditional pass (≥0.85 point; macros weak) |
| quote_precision | 0.8658 | [0.8497, 0.8812] | informational (proposed ≥0.85 in Phase 1) |
| repetition_ratio | 0.9991 | [0.9974, 1.0000] | informational |

## v0 vs current

This run is v0-fixed baseline (no prior baseline to diff).
