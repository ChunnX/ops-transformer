# QuantAllReduce 阶段 1 性能验证（round 001）

**日期**：2026-04-30 | **Baseline**：`fdbcd206` | **Optimized**：`f2f9b546`
**环境**：4 卡 Ascend950PR，CANN 9.0.0，--scene one_hccl --world_size 4
**用例**：level0 24 用例去重到 10 条（按 `logical_shape × quant_mode × outdtype`，2D/3D 等价）

## 1. 结论

**阶段 1 整体回退，平均 +5.27%**（10 用例中 9 慢 1 快）。最坏 ID0007（[96,7168] T-G int8→bf16）+11.26%。

**根因**：算子 MTE2-bound（aiv_mte2_ratio ≈ 87%），Vec/Scalar 已被 DoubleBuffer 掩盖。阶段 1 让它们提前结束反而让 MTE2 在尾段暴露 → 净变慢。

## 2. mc2_prof 端到端对比（us）

| ID | baseline | optimized | 变化 |
|---|---:|---:|---:|
| 0001 (96_7168 T-G int8→fp32) | 37.635 | 38.416 | ↑ 2.08% |
| 0005 (96_7168 MX e4m3→fp32) | 38.324 | 37.950 | ↓ 0.97% |
| 0006 (96_7168 MX e5m2→fp16) | 38.241 | 42.099 | **↑10.09%** |
| 0007 (96_7168 T-G int8→bf16) | 37.687 | 41.930 | **↑11.26%** |
| 0012 (96_7168 MX e5m2→bf16) | 38.345 | 41.316 | ↑ 7.75% |
| 0013 (96_4096 T-G int8→fp16) | 24.007 | 25.315 | ↑ 5.45% |
| 0014 (96_4096 T-G hif8→bf16) | 24.116 | 25.365 | ↑ 5.18% |
| 0017 (96_4096 MX e4m3→fp16) | 24.700 | 25.882 | ↑ 4.79% |
| 0018 (96_4096 MX e5m2→bf16) | 24.682 | 25.463 | ↑ 3.17% |
| 0024 (96_4096 MX e5m2→fp32) | 24.587 | 25.034 | ↑ 1.82% |
| **avg** | **34.702** | **36.530** | **↑ 5.27%** |

[96,7168] 系列回退最重（数据量大→MTE2 占比更高→Vec 优化的反作用更显著）。

## 3. ID0007 流水分解（rank 0，iter 3-10 稳态均值）

| 指标 | Baseline | Optimized | Δ |
|---|---:|---:|---:|
| Total Duration | 37.50 us | 41.22 us | **+3.73 us (+9.95%)** |
| aiv_vec_time | 7.53 us (20.2%) | 5.80 us (14.2%) | −1.73 us (−23%) |
| aiv_scalar_time | 7.28 us (19.6%) | 2.98 us (7.3%) | −4.30 us (−59%) |
| **aiv_mte2_time** | 32.29 us (86.7%) | 35.76 us (87.3%) | **+3.47 us (+10.7%)** |
| aiv_mte3_time | 3.52 us | 1.65 us | −1.87 us |

**量化**：vec+scalar 节省 6.03us，mte2 净增 3.47us，mte3 减 1.87us，再加 ~1us 同步开销 → duration +3.73us（与实测吻合）。

## 4. 给阶段 2/3 的建议

1. **阶段 2（每核独占 rank）应优先做**：当前所有核竞争所有 rank Win Area → MTE2 串行化；每核独占后路径独立，能直接降 MTE2。这才是 MTE2-bound 算子的优化方向。
2. **阶段 3（SubBlock 主辅核）对本算子收益有限**：不涉及 AIC，辅核空转浪费算力；主核 Vec 工作量翻倍，掩盖 MTE2 反而变差。除非阶段 2 改成 Vec bound 才考虑。
3. **阶段 1 是否回滚**：(a)(b)(c) 是耦合包（(c) 修 (a)(b) 引入的 bug，不是原始 bug；原始 fdbcd206 一致 1024B 分块没问题）。**建议从 fdbcd206 开新分支做阶段 2**，归因清晰。详见 plan §5.2.0。

## 5. 数据清单

| 文件 | 内容 |
|---|---|
| `baseline_perf.csv` / `optimized_perf.csv` | 10 用例 performance_result（mc2_prof 第 9 列） |
| `baseline_mc2_pro.csv` / `optimized_mc2_pro.csv` | 简版 MC2 性能（case, rank, memory, mc2_prof） |
| `profiling_id0007/{baseline,opt}_rank{0..3}_kernel_details.csv` | ID0007 全 4 rank torch_npu profiler 输出 |
| `experiment_log.md` | 过程踩坑与方法论 |

复现命令见 plan §10。
