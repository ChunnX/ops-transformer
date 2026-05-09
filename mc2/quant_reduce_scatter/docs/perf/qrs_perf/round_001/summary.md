# QuantReduceScatter 阶段 1 性能验证（round 001）

**日期**：2026-04-30 | **Baseline**：`fdbcd206` | **Optimized**：`f2f9b546`
**环境**：4 卡 Ascend950PR，CANN 9.0.0，--scene one_hccl --world_size 4
**用例**：level0 24 用例去重到 12 条（按 `logical_shape × quant_mode × outdtype`）

## 1. 结论

**阶段 1 整体回退，平均 +2.14%**（12 用例中 11 慢 1 快）。回退幅度比 QuantAllReduce（+5.27%）小，因 ReduceScatter 数据量是 AllReduce 的 1/4，但根因相同：MTE2-bound，阶段 1 优化方向（减 Vec/Scalar）与瓶颈不匹配。

详细 profiling 根因见 `../../quant_all_reduce/docs/perf/round_001/summary.md` §3。

## 2. mc2_prof 端到端对比（us）

| ID（特征） | baseline | optimized | 变化 |
|---|---:|---:|---:|
| 0001 (48_7168 T-G→bf16) | 11.318 | 11.759 | ↑ 3.90% |
| 0002 (48_7168 T-G→fp16) | 11.317 | 11.551 | ↑ 2.07% |
| 0003 (48_7168 T-G int8→fp32) | 11.314 | 11.479 | ↑ 1.46% |
| 0005 (48_7168 MX→fp32) | 11.713 | 11.831 | ↑ 1.01% |
| 0006 (48_7168 MX→bf16) | 11.711 | 12.046 | ↑ 2.86% |
| 0011 (1_48_7168 MX→fp16) | 11.602 | 12.148 | ↑ 4.71% |
| 0015 (1_48_5120 T-G int8→fp16) | 10.406 | 10.671 | ↑ 2.55% |
| 0015b (48_5120 T-G int8→bf16) | 10.606 | 10.932 | ↑ 3.08% |
| 0016 (48_5120 T-G hif8→fp32) | 10.293 | 9.934 | ↓ 3.49% |
| 0017 (1_48_5120 MX→fp32) | 10.574 | 10.587 | ↑ 0.12% |
| 0017b (48_5120 MX→bf16) | 10.926 | 11.256 | ↑ 3.02% |
| 0018 (48_5120 MX→fp16) | 10.963 | 11.387 | ↑ 3.87% |
| **avg** | **11.062** | **11.299** | **↑ 2.14%** |

## 3. 数据清单

| 文件 | 内容 |
|---|---|
| `baseline_perf.csv` / `optimized_perf.csv` | 12 用例 performance_result |
| `baseline_mc2_pro.csv` / `optimized_mc2_pro.csv` | 简版 MC2 性能 |

profiling 未单独采集（`quant_all_reduce` 的 profiling 已能代表两算子的 MTE2-bound 特性）。

阶段 2/3 建议与 (a)(b)(c) 耦合关系详见 `quant_all_reduce/docs/perf/round_001/summary.md` §4 与 plan §5.2.0。复现命令见 plan §10。
