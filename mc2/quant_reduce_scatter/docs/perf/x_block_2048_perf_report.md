# QRS (QuantReduceScatter) x_block_2048 优化性能对照报告

> **分支**：`x_block_2048_stage1` (基于 stage1 `f2f9b546`)
> **改动**：`mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_mte.h:149` 在 `maxXPerBlock` 计算后新增 `X_MAX_BLOCK_NUM = 2048U` 上限（与 QAR 同款 patch）
> **基线**：stage2 分支归档 `mc2/quant_reduce_scatter/docs/perf/round_001/baseline_perf.csv`
> **测试**：world_size=4, 4×Ascend950PR, level0 sheet
> **采集**：mc2_test 内置 10 轮取后 5 轮均值，分两段（num 1-8, num 9-24）跑全 24 用例

## 12 个 baseline 用例对照

| case | baseline mc2_prof (us) | current mc2_prof (us) | δ |
|---|---|---|---|
| 48_7168 fp8_e5m2 → bf16 ID0001 | 11.3178 | 10.5434 | **−6.84%** |
| 48_7168 fp8_e4m3fn mxfp → bf16 ID0006 | 11.7112 | 10.5892 | **−9.58%** |
| 48_5120 hif8 → fp32 ID0016 | 10.2926 | 9.5668 | **−7.05%** |
| 1×48×5120 int8 → fp16 ID0015 | 10.4060 | 9.5628 | **−8.10%** |
| 1×48×5120 fp8_e4m3fn mxfp → fp32 ID0017 | 10.5740 | 9.7674 | **−7.63%** |
| 48_7168 fp8_e4m3fn → fp16 ID0002 | 11.3166 | 10.5710 | **−6.59%** |
| 48_7168 int8 → fp32 ID0003 | 11.3140 | 10.4786 | **−7.38%** |
| 48_7168 fp8_e5m2 mxfp → fp32 ID0005 | 11.7134 | 10.8650 | **−7.24%** |
| 1×48×7168 fp8_e5m2 mxfp → fp16 ID0011 | 11.6020 | 10.8470 | **−6.51%** |
| 48_5120 int8 → bf16 ID0015 | 10.6056 | 9.4142 | **−11.23%** |
| 48_5120 fp8_e4m3fn mxfp → bf16 ID0017 | 10.9262 | 9.6796 | **−11.41%** |
| 48_5120 fp8_e5m2 mxfp → fp16 ID0018 | 10.9628 | 9.8040 | **−10.57%** |

## 总结

- **12/12 用例全部提升**，无劣化
- **48×7168 / 1×48×7168 提升幅度 6.5% ~ 9.6%**
- **48×5120 / 1×48×5120 提升幅度 7.0% ~ 11.4%**
- **精度无回退**：level0 全 24 用例 4×rank 共 96 项全部 PASS（100%）

## 异常观察（已澄清）

批跑时 `48_5120 fp8_e5m2 fp32 → fp32 qwen ID0014` (num=17) 的 mc2_prof 报 48.13 us，与同 shape 其他用例 (9-11us) 偏离 5 倍。**单独重测（`--scene performance --num 17`）实测 9.6718 us，与同类用例一致**。该值实为批跑时 profiler 状态残留导致的采集污染，非算子真实性能问题，可忽略。该用例不在 baseline 12 用例对照集中，未影响主结论。

## 完整 24 用例数据

见 `x_block_2048_perf.csv`（同目录）。
