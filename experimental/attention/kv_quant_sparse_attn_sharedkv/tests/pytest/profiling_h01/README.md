# h01 Profiling Results — KvQuantSparseAttnSharedkv SCFA

Best-kept config: **h01 (v0 dequant lead-by-1 deeper pipeline)**.
See `../../SCFA_h01_Design.md` for the design (pipeline + buffer allocation, CN/EN).

## Test case (`my_test`, the only ENABLED_PARAMS)

| field | value |
|---|---|
| layout_q / layout_kv | BSND / PA_ND |
| q_type / kv_type | bf16 / fp8_e4m3fn |
| B, N1, N2 | 48, 128, 1 |
| S1, S2 | 2, 8192 |
| D, K (sparse blocks), rope | 512, 1024, 64 |
| template_run_mode | SCFA |

`gSize = N1/N2 = 128` → `IS_SPLIT_G` path (M split across 2 AIC cores sharing dequant KV via GM).

Verified against the captured `op_summary_run*.csv`: query Input Shape `48,2,128,512`, scalar K `1024`.

## Profiling setup

- Tool: `msprof --aic-metrics=PipeUtilization`
- Metric: `Task Duration(us)` of `Op Name == KvQuantSparseAttnSharedkv`
- 5 runs, no frequency lock (run-to-run variance is from unlocked AICore freq).

## Results (5 runs)

| run | Task Duration (us) | aic_mac | aic_mte2 | aic_mte1 | aic_fix | aiv_vec | aiv_mte2 | aiv_scalar |
|---|---|---|---|---|---|---|---|---|
| 1 | 107.681 | 0.696 | 0.600 | 0.551 | 0.300 | 0.570 | 0.684 | 0.733 |
| 2 | 115.217 | 0.650 | 0.588 | 0.509 | 0.279 | 0.533 | 0.720 | 0.695 |
| 3 | 108.091 | 0.693 | 0.599 | 0.546 | 0.297 | 0.568 | 0.690 | 0.733 |
| 4 | 108.360 | 0.690 | 0.593 | 0.546 | 0.294 | 0.566 | 0.680 | 0.729 |
| 5 | 117.880 | 0.634 | 0.579 | 0.495 | 0.271 | 0.520 | 0.720 | 0.684 |

- **median: 108.360 us**, min: 107.681 us, max: 117.880 us
- avg: 111.446 us, stdev: 4.256 us (3.82%)

> Representative value = **median ~108.4 us** (consistent with the original h01 floor 108.947 us).
> avg is pulled up by two unlocked-frequency outliers (run 2, run 5, lower `aic_mac`).
> The clean runs (1/3/4) sit at ~108 us with `aic_mac` ≈ 0.69.

## Accuracy

`PctRlt = 99.994818%` (threshold 99.50%), Max-RelativeError 1.387 (threshold 10) → **PASS**.

## vs baseline

| config | duration (us) | aic_mac |
|---|---|---|
| a05 (before) | 131.463 | 0.53 |
| **h01 (this)** | **~108.4 (median)** | **~0.69** |

−17% latency; the cube stops stalling on the AIV dequant producer (`aic_mac` 0.53 → 0.69).

## Files

- `durations.txt` — the 5 raw Task Durations
- `ratios.csv` — `run, duration, aic_mac, aic_mte2, aic_mte1, aic_fix, aiv_vec, aiv_mte2, aiv_scalar`
- `op_summary_run{1..5}.csv` — raw msprof op_summary (full per-pipe metrics, per run)
