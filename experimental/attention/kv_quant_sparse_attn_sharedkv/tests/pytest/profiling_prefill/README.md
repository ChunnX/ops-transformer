# SCFA Prefill Performance Profiling — KvQuantSparseAttnSharedkv

3 kernel variants, **prefill** case, **2 runs each**. Accuracy already verified separately
(this is perf-only; `QSAS_PERF_ONLY=1` skips the CPU golden forward, msprof still reads the
real NPU kernel time).

## Test case (`scfa_prefill`)

| field | value |
|---|---|
| layout_q / layout_kv | BSND / PA_ND |
| q_type / kv_type | bf16 / fp8_e4m3fn |
| B, N1, N2 | 1, 128, 1 |
| S1, S2 | 8192, 32768 |
| D, K (sparse blocks), rope | 512, 1024, 64 |
| cmp_ratio | 4 |
| template_run_mode | SCFA |

## Variants

All three share the current (vectorized + prefill-fix) `block_vector.h` / `block_cube.h`;
**only `kv_quant_sparse_attn_sharedkv_scfa_kernel.h` differs.**

| tag | meaning | kernel source | md5 |
|---|---|---|---|
| baseline | 未做流水优化 (a05) | `/tmp/kernel_before_reorder.h` | 96b57d0c… |
| h01 | 流水优化 + loadlast (committed default) | `/tmp/h01_kernel_keep.h` | 66f743d6… |
| leadby1 | 只有流水优化 (lead-by-1, no loadlast) | `..._scfa_kernel_leadby1_noloadlast.h` | 113415a4… |

## Profiling setup

- Tool: `msprof --aic-metrics=PipeUtilization`
- Metric: `Task Duration(us)` of `Op Name == KvQuantSparseAttnSharedkv`
- 2 runs each, no frequency lock.
- Env: `scfa_env.sh` (venv python 3.12 + torch_npu 2.9.0), install → `/home/w00927905/custom_opp`.

## Results (2 runs)

| variant | run1 (us) | run2 (us) | avg (us) | aic_mac | aic_mte2 | aic_mte1 | aic_fix | aiv_vec | aiv_mte2 | aiv_scalar |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 11188.692 | 11194.203 | **11191.447** | 0.545 | 0.490 | 0.424 | 0.237 | 0.461 | 0.269 | 0.690 |
| h01      | 8139.524  | 8146.094  | **8142.809**  | 0.767 | 0.668 | 0.615 | 0.341 | 0.632 | 0.392 | 0.886 |
| leadby1  | 8321.910  | 8323.237  | **8322.573**  | 0.750 | 0.642 | 0.596 | 0.331 | 0.619 | 0.403 | 0.890 |

## Comparison

| variant | avg (us) | vs baseline | aic_mac |
|---|---|---|---|
| baseline (no pipeline opt) | 11191.447 | — | 0.545 |
| **h01 (pipeline + loadlast)** | **8142.809** | **−27.24%** | 0.767 |
| leadby1 (pipeline only) | 8322.573 | −25.63% | 0.750 |

- Pipeline optimization (leadby1) alone: **−25.6%** (11191 → 8323 us), aic_mac 0.545 → 0.750.
- Adding loadlast on top (h01): a further **−2.2%** vs leadby1 (8323 → 8143 us),
  aic_mac 0.750 → 0.767 — h01 is the best of the three.
- Run-to-run variance is tiny here (<0.1%), so 2 runs is sufficient for this case.

## Files (all retained)

Under `profiling_prefill/<variant>/`:
- `op_summary_run{1,2}.csv` — raw msprof op_summary (full per-pipe metrics, per run)
- `durations.txt` — the 2 raw Task Durations
- `ratios.csv` — run, duration, pipe-utilization ratios
- `build.log` / `install.log` / `msprof_run_{1,2}.log` — build/install/profiling logs
- `run_{1,2}/PROF_*/` — full raw msprof profiling output dirs
