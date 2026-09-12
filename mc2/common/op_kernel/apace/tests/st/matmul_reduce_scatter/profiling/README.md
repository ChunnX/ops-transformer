# profiling 目录 — quant_matmul_reduce_scatter (apace ST) 性能分析总览

> **代码版本**: ops-transformer upstream-apace @ ed7d74662
> **环境**: Ascend950PR ×4, CANN 9.2.0, bisheng (dav-3510)
> **数据采集**: 2026-09-10 ~ 2026-09-12

## 报告索引

| 报告 | 位置 | 内容 |
|---|---|---|
| **性能对比** | [reports/comparison_report.md](reports/comparison_report.md) | 97 用例 ST vs 级联基线 vs MC2 融合算子完整对比 + 劣势专项分析 |
| **深入分析** | [reports/deep_analysis_report.md](reports/deep_analysis_report.md) | Bound 判定（R 值）、多维加速比分布、通信效率、rank2/4 配对分析 |
| **隔离测试** | [reports/isolation_metrics_report.md](reports/isolation_metrics_report.md) | 12 代表用例四分量分解、加速比与掩盖率（修正口径）、负值验证 |

## 综合结论

### 1. 整体性能优势

| 对比维度 | 结果 |
|---|---|
| **vs 小算子级联基线**（npu_quant_matmul + reduce_scatter） | **93/97 更快**，中位 **1.23x**，avg 1.25x，max 1.65x |
| **vs MC2 融合算子**（ACLNN npu_quant_mm_reduce_scatter） | **88/97 更快**，中位 **1.14x**，avg 1.17x，max 1.69x |
| **通算融合加速比**（隔离测试 T_serial/T_fused，12 代表用例） | 中位 **1.20x**，avg 1.22x，max 1.51x |
| **流水掩盖率**（修正口径，12 代表用例） | 中位 **0.70**，max 0.91（通信 Bound 场景中位 0.83） |

### 2. Bound 判定与优势来源

**70% 用例为通信 Bound**（R < 0.67，基于级联基线分解）：级联基线中通信耗时远超计算。
这正是 apace ST 融合方案优势的来源 —— 通信 Bound 场景下流水掩盖收益最大。

| Bound 类型 | 用例数 | vs 级联中位加速比 | 隔离测试掩盖率中位 |
|---|---|---|---|
| 通信 Bound | 68 (70%) | 1.23x | 0.83 |
| 均衡 | 28 (29%) | 1.26x | 0.38 |
| 计算 Bound | 1 (1%) | 1.04x | — |

### 3. 已知劣势与根因（4/97 vs 级联慢）

| 模式 | 用例 | 根因 | 性质 |
|---|---|---|---|
| **极窄 N=640** | row16/17 | URMA PUT 小消息固定开销 > HCCL；隔离测试证实融合同步开销 > 重叠收益 | **架构性 trade-off** |
| **K=5472/6144 大 K rank4** | row31/51/81 等 | matmul 效率差距 ~10%（K%64==32 padding + tiling 参数） | **可优化项** |

劣势用例重跑验证：偏差全部 < 2%，数据可靠。row17 隔离测试负掩盖率(-0.63)经同二进制 4 次复测确认为**真实回退**（融合同步开销导致，小 tile 场景限制）。

### 4. 多维规律

- **rank4 vs 融合算子加速比（1.32x）高于 rank2（1.13x）**：rank4 通信开销更大，融合掩盖收益更显著
- **M=6144/8192 家族最优**（中位 1.34x/1.31x）：通信/计算比最均衡
- **K > 4608 时 vs 融合收窄至 1.07x**：大 K matmul 效率成为瓶颈（优化方向）
- **N ≤ 1024 时 vs 级联降至 0.83x**：小消息场景劣势（架构限制）

### 5. 后续优化方向

1. **大 K matmul 效率**（K > 4608，~10% 差距）: 调优 baseK/stepK/scaleKL1 tiling 参数
2. **小 N 通信效率**（N ≤ 1024）: 评估 URMA PUT batching 或小消息优化模式
3. **rank2 场景**（vs 融合仅 1.13x）: 通信掩盖空间有限，重点提升 matmul 本身

## 代码质量记录

- **冗余代码清理**（已合入 PR ed7d74662）: 删除 ExecuteReduceSumBatch 死代码 124 行 + DoReduceSumBatch 重复方法 + 未用字段/别名/变量/冗余 include，共 **-175 行（-20%）**，编译验证 + 精度冒烟通过

## 目录结构

```
profiling/
├── README.md                          # 本文件（总览入口）
│
├── reports/                           # 📊 分析报告（读者入口）
│   ├── comparison_report.md           #   性能对比（97 用例 + 劣势专项）
│   ├── deep_analysis_report.md        #   深入分析（Bound/R值/多维分布/通信效率）
│   └── isolation_metrics_report.md    #   隔离测试指标（加速比/掩盖率/负值验证）
│
└── data/                              # 📁 原始数据（按采集来源分，复现用）
    ├── st_perf/                       #   ST 97 用例（365 PROF，op_summary 等）
    ├── baseline/                      #   级联基线 + MC2 融合算子
    │   ├── hccl_mm_..._pro_result.csv #     级联结果 CSV（97 case）
    │   ├── mc2_..._pro_result.csv     #     融合结果 CSV（97 case）
    │   ├── st_perf_summary.txt        #     ST 时延解析汇总
    │   ├── logs/                      #     基线 NPU 日志（99 个）
    │   └── raw_profiler/              #     幸存 profiler 原始 CSV（32 个，4 类 × 4 rank × 2 路径）
    └── isolation/                     #   隔离测试（12 case × 4 mode）
        ├── isolation_metrics.csv      #     指标汇总
        └── row*/prof_*/PROF_*/       #     op_summary 等
```

> **数据口径**: `data/` 保存 msprof 分析级输出（op_summary/op_statistic/task_time/api_statistic CSV +
> case_info），已剥离 .db/trace/日志等重文件（原始完整目录 4.4G，分析数据 18M）。完整 msprof 原始目录保留在
> `../prof/`（ST 97 用例）和 `/home/y00845930/dev/apace_perf_probe/run/`（隔离测试）。

## 数据源清单

| 数据 | 位置 | 说明 |
|---|---|---|
| ST 97 用例分析数据 | `data/st_perf/`（365 PROF） | op_summary + case_info + 统计 CSV |
| ST msprof 完整目录 | `../prof/`（490M，仓库内） | 含 .db/trace/日志（可重新生成分析数据） |
| ST 时延解析汇总 | `data/baseline/st_perf_summary.txt` | parse_prof.py --all 完整输出 |
| 级联基线 CSV | `data/baseline/hccl_mm_aclnnMatmulReduceScatterV2_pro_result.csv` | 97 case 小算子级联性能（含通信/计算分解） |
| MC2 融合 CSV | `data/baseline/mc2_aclnnMatmulReduceScatterV2_pro_result.csv` | 97 case ACLNN 融合算子性能 |
| 基线 NPU 日志 | `data/baseline/logs/`（99 个） | mc2_test 框架逐 case 日志（95 个含完整两级性能打印） |
| 基线 profiler 原始样本 | `data/baseline/raw_profiler/`（32 CSV） | torch_npu profiler 输出样本（仅 row81 重跑幸存，4 类关键 CSV × 4 rank × 2 路径） |
| 隔离测试分析数据 | `data/isolation/`（12 用例 × 4 模式） | op_summary CSV |
| 隔离测试指标汇总 | `data/isolation/isolation_metrics.csv` | 12 用例四分量指标（机器可读） |
| 隔离测试完整目录 | `/home/y00845930/dev/apace_perf_probe/`（仓外，3.9G） | probe 源码 + 完整 msprof 数据 + 核算脚本 |
| 基线 xlsx | `david_excel/aclnnMatmulReduceScatterV2.xlsx` 行 106-203（仓外） | 97 用例定义（备份 .bak_20260911） |

> **基线原始数据说明**: mc2_test 框架的 torch_npu profiler 原始 trace（`Profiling_Result_*` 目录）在框架
> 采集后仅保留提取结果（CSV），原始目录已被框架清理。本目录保存的分析级数据为：
> ① 两份 per-case 结果 CSV（97/97 完整，含 avg_hccl/avg_matmul/通信/内存/performance 六列分解）；
> ② 99 个 NPU 日志（框架打印的完整性能摘要，95 个完整 + 2 个重试覆盖，数据可从 CSV 恢复）；
> ③ 2 个幸存的 profiler 原始目录（row81 重跑，32 个分析 CSV）。
> 如需其余用例的逐 iter 原始 trace，需用 mc2_test 框架重跑（约 2h）。
