# 性能评估结果

- **Operator**: pdist
- **Device**: npu:0 (source=cli)
- **Warmup**: 3
- **Repeats**: 3
- **Seed**: 42
- **Timing method**: msprof.op_summary.Task_Duration
- **Timing source**: PipeUtilization
- **Target Op Name**: `_Z12pdist_kernelPhS_S_`
- **Case selector**: `function_adapter:PERFORMANCE_CASES.test_function`
- **Profiling mode**: compare
- **Collection ID**: compare_20260902_140605_651332
- **Performance cases**: `case_manifest`; source=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/PERFORMANCE_CASES.json`; sha256=`755702fbb0ead384697e8e43903ed810a3d630e6afbf82f26c3cfe9846418cd0`
- **Golden target source status**: `joined`
- **Default target**: `golden_per_iteration_e2e_us / pypto_target_kernel_us >= 1.0` for every P0 case
- **Default target status**: `not_met` (valid_for_target_met=true)
- **Comparison scope**: `golden_e2e_to_pypto_target_kernel`
- **Golden source**: `torch_npu.kernel_details.all_golden_npu_kernel_sum` (iterations=1, normalized per iteration)
- **Ratio semantics**: this ratio is valid for the Stage 5 default Golden target; it is not baseline-to-final optimization speedup.

## Golden 默认目标对比

| Case | Shape | DType | PyPTO目标kernel(us) | Golden每迭代E2E(us) | 默认目标比值(E2E/kernel) |
| ---- | ----- | ----- | ------------- | -------- | -------------- |
| perf_p0_euclidean | [4096,256] | float32 | 25658.54 | 45514.20 | 1.774 |
| perf_p0_fp16_large | [8192,512] | float16 | 963019.06 | 344973.57 | 0.358 |
| func_p0_hamming_p0 | [1000,200] | float32 | 1286.67 | 3351.90 | 2.605 |
| func_p0_manhattan_p1 | [1000,200] | float32 | 1267.35 | 3062.74 | 2.417 |
| func_p0_euclidean_p2 | [1000,200] | float32 | 1278.08 | 3073.32 | 2.405 |
| func_p0_fractional_p05 | [1000,200] | float32 | 1270.38 | 3063.93 | 2.412 |
| func_p0_chebyshev_pinf | [1000,200] | float32 | 1297.62 | 2043.36 | 1.575 |
| func_p0_min_shape | [2,1] | float32 | 4.08 | 15.81 | 3.875 |
| func_p0_large_m | [64,8192] | float32 | 608.36 | 696.96 | 1.146 |
| func_p0_partial_block | [777,300] | float32 | 1146.23 | 2840.67 | 2.478 |

### 重复采样证据

- `perf_p0_euclidean`: samples_us=[25658.537, 25658.919, 25657.83]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_perf_p0_euclidean_9fe06310`
- `perf_p0_fp16_large`: samples_us=[963019.063, 963004.221, 963030.832]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_perf_p0_fp16_large_d976a20e`
- `func_p0_hamming_p0`: samples_us=[1286.886, 1286.257, 1286.667]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_hamming_p0_efe8f899`
- `func_p0_manhattan_p1`: samples_us=[1267.35, 1267.535, 1266.647]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_manhattan_p1_c63b8542`
- `func_p0_euclidean_p2`: samples_us=[1279.373, 1278.08, 1277.918]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_euclidean_p2_d9eb7b23`
- `func_p0_fractional_p05`: samples_us=[1271.138, 1269.592, 1270.384]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_fractional_p05_556b4f27`
- `func_p0_chebyshev_pinf`: samples_us=[1297.946, 1297.444, 1297.62]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_chebyshev_pinf_087b5c27`
- `func_p0_min_shape`: samples_us=[4.172, 4.059, 4.079]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_min_shape_f3e2e13b`
- `func_p0_large_m`: samples_us=[608.36, 608.694, 608.309]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_large_m_dcf93cde`
- `func_p0_partial_block`: samples_us=[1145.577, 1146.364, 1146.234]; selected_op_names=['_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_', '_Z12pdist_kernelPhS_S_']; evidence=`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_partial_block_22fbf781`

## Pipe / Bound 路由（诊断，不是最终 Roofline 结论）

| Case | Repeat 路由 | Scalar 候选 | Roofline 结论 | 搬运/计算流水证据 |
| ---- | ----------- | ----------- | ------------- | ----------------- |
| perf_p0_euclidean | ['no_dominant_pipe', 'compute_candidate', 'compute_candidate'] | false | ['insufficient_evidence'] | ['unverified'] |
| perf_p0_fp16_large | ['compute_candidate', 'no_dominant_pipe', 'compute_candidate'] | false | ['insufficient_evidence'] | ['unverified'] |
| func_p0_hamming_p0 | ['compute_candidate', 'compute_candidate', 'compute_candidate'] | false | ['insufficient_evidence'] | ['unverified'] |
| func_p0_manhattan_p1 | ['compute_candidate', 'compute_candidate', 'compute_candidate'] | false | ['insufficient_evidence'] | ['unverified'] |
| func_p0_euclidean_p2 | ['compute_candidate', 'compute_candidate', 'compute_candidate'] | false | ['insufficient_evidence'] | ['unverified'] |
| func_p0_fractional_p05 | ['compute_candidate', 'compute_candidate', 'compute_candidate'] | false | ['insufficient_evidence'] | ['unverified'] |
| func_p0_chebyshev_pinf | ['compute_candidate', 'compute_candidate', 'compute_candidate'] | false | ['insufficient_evidence'] | ['unverified'] |
| func_p0_min_shape | ['no_dominant_pipe', 'no_dominant_pipe', 'no_dominant_pipe'] | false | ['insufficient_evidence'] | ['unverified'] |
| func_p0_large_m | ['compute_candidate', 'compute_candidate', 'compute_candidate'] | false | ['insufficient_evidence'] | ['unverified'] |
| func_p0_partial_block | ['compute_candidate', 'compute_candidate', 'compute_candidate'] | false | ['insufficient_evidence'] | ['unverified'] |

> `op_summary` 的 pipe ratio 只用于选择下一项实验；它不包含事件先后关系，不能单独证明 compute/movement bound 或搬运与计算已经重叠。最终结论必须在 `PERFORMANCE_REPORT.md` 中引用工作量/流量模型及 timeline 或等价依赖证据。

## PyPTO measurement summary

| Metric | Value |
| ---- | -- |
| Cases | 10 |
| Valid PyPTO measurements | 10 |
| Mean PyPTO target-kernel (us) | 99683.637 |
| Median PyPTO target-kernel (us) | 1274.232 |
| Total PyPTO target-kernel (us) | 996836.374 |
| Golden target cases | 10 |
| Default target threshold | 1.0 |
| Default target status | not_met |
| Mean Golden-E2E/PyPTO-kernel target ratio | 2.104 |
| Target-met cases (ratio >=1) | 9 |
| Target-missed cases (ratio <1) | 1 |

### 按数据类型汇总

| DType | 用例数 | 平均默认目标比值 | 达标(>=1) | 未达标(<1) |
| ----- | ------ | ------------------- | ------------- | -------- |
| float16 | 1 | 0.358 | 0 | 1 |
| float32 | 9 | 2.298 | 9 | 0 |

## 简短分析

- Golden 每迭代 E2E / PyPTO target-kernel 平均目标比值为 2.104；默认目标要求每个 P0 case 均不低于 1.0。该比值不是 baseline→final 优化加速比。
- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。

## 深度瓶颈分析

- 深度采集轮次：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002`
- `perf_p0_euclidean`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_perf_p0_euclidean_9fe06310/summary.txt` 及七组 `op_summary_*.csv`
- `perf_p0_fp16_large`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_perf_p0_fp16_large_d976a20e/summary.txt` 及七组 `op_summary_*.csv`
- `func_p0_hamming_p0`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_hamming_p0_efe8f899/summary.txt` 及七组 `op_summary_*.csv`
- `func_p0_manhattan_p1`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_manhattan_p1_c63b8542/summary.txt` 及七组 `op_summary_*.csv`
- `func_p0_euclidean_p2`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_euclidean_p2_d9eb7b23/summary.txt` 及七组 `op_summary_*.csv`
- `func_p0_fractional_p05`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_fractional_p05_556b4f27/summary.txt` 及七组 `op_summary_*.csv`
- `func_p0_chebyshev_pinf`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_chebyshev_pinf_087b5c27/summary.txt` 及七组 `op_summary_*.csv`
- `func_p0_min_shape`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_min_shape_f3e2e13b/summary.txt` 及七组 `op_summary_*.csv`
- `func_p0_large_m`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_large_m_dcf93cde/summary.txt` 及七组 `op_summary_*.csv`
- `func_p0_partial_block`：`/data/z00954450/zzz/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist/docs/perf/round_002/case_func_p0_partial_block_22fbf781/summary.txt` 及七组 `op_summary_*.csv`


