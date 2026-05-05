# 2026-05-05 Baseline/Halflocal Performance Summary

Scope: same original normal profiling root, only `B=8` selected S cases: `S=16/64/125/2048`. Ratios are `ChunkGDRNeumann / ChunkGDR` task duration from msprof measurement row.

Precision status before profiling: baseline `3*2` and halflocal `3*2` all built and `compare` precision passed (`20 passed`).

| suite | variant | S=16 | S=64 | S=125 | S=2048 | avg |
|---|---:|---:|---:|---:|---:|---:|
| baseline | `C16_FP32` | 1.142x | 2.262x | 2.491x | 4.117x | 2.503x |
| baseline | `C16_FP16` | 1.103x | 2.213x | 2.480x | 4.110x | 2.477x |
| baseline | `C64_1616_FP32` | 5.267x | 31.423x | 43.902x | 133.261x | 53.463x |
| baseline | `C64_1616_FP16` | 5.254x | 31.369x | 43.919x | 133.273x | 53.454x |
| baseline | `C64_3232_FP32` | 1.290x | 1.967x | 2.237x | 2.847x | 2.085x |
| baseline | `C64_3232_FP16` | 1.303x | 1.933x | 2.266x | 2.846x | 2.087x |
| halflocal | `C16_FP32` | 1.038x | 2.171x | 2.382x | 4.054x | 2.411x |
| halflocal | `C16_FP16` | 1.089x | 2.131x | 2.392x | 4.089x | 2.425x |
| halflocal | `C64_1616_FP32` | 5.240x | 31.341x | 43.869x | 133.229x | 53.420x |
| halflocal | `C64_1616_FP16` | 5.215x | 31.325x | 43.870x | 133.238x | 53.412x |
| halflocal | `C64_3232_FP32` | 1.305x | 1.651x | 1.873x | 2.332x | 1.790x |
| halflocal | `C64_3232_FP16` | 1.383x | 1.647x | 1.864x | 2.319x | 1.803x |

Best among measured variants: `halflocal/C64_3232_FP32` average `1.790x`, with ratios `1.305x / 1.651x / 1.873x / 2.332x` for `S=16/64/125/2048`.

Notes:
- `baseline` here means AIV only constructs initial `L` and `I-L`; Neumann rounds use AIC Matmul/Fixpipe atomic add, not the older AIV add baseline.
- `halflocal` means L0C/CMatrix accumulation for `old_INV + INV@P`, then ordinary Fixpipe writes updated INV back to GM before the next round.
- `halflocal/C16_FP16/S=2048` first msprof run hit a transient AICore `507015` L0A read/write conflict during warmup; rerun succeeded and is included in the table.

Raw CSV: `tests/pytest/prof_out/stage1_baseline_halflocal_s_selected_summary_20260505.csv`.
