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
| identifier_precision | 0.1909 | [0.1815, 0.1997] |
| quote_precision | 0.7388 | [0.7131, 0.7654] |

## verdict_distribution

| verdict | share | count | 95% CI |
|---|---:|---:|---|
| VULNERABLE | 0.7067 | 424 | [0.6550, 0.7584] |
| SAFE | 0.1650 | 99 | [0.1250, 0.2067] |
| INSUFFICIENT | 0.1233 | 74 | [0.0900, 0.1583] |
| UNPARSED | 0.0050 | 3 | [0.0000, 0.0100] |

## v0 vs current

This run is v0 (no prior baseline to diff).
