# profiling 目录 — quant_matmul_reduce_scatter (apace ST) 性能分析总览

> **代码版本**: ops-transformer upstream-apace @ 6d51651f4 (v8 + 检视修复)
> **环境**: Ascend950PR ×4, CANN 9.2.0 (`/home/l00947526/Ascend915/cann-9.2.0`), bisheng (dav-3510)
> **数据采集**: 2026-09-10 ~ 2026-09-23 (v1 基线 ~ v8 全量)
> **当前版本精度**: 101/101 case PASS

## 报告索引

| 报告 | 位置 | 版本 | 内容 |
|---|---|---|---|
| **★ 全量 v5 vs v8 对比** | [reports/full_suite_v5_vs_v8_report.md](reports/full_suite_v5_vs_v8_report.md) | v8 | **101/101 用例** v5/v8 同日同机交替 A/B，v8 优化收益与归因分析 |
| 性能对比（外部基线） | [reports/comparison_report.md](reports/comparison_report.md) | v1 | 97 用例 ST vs 级联基线 vs MC2 融合算子完整对比 + 劣势专项分析 |
| 深入分析 | [reports/deep_analysis_report.md](reports/deep_analysis_report.md) | v1 | Bound 判定（R 值）、多维加速比分布、通信效率、rank2/4 配对分析 |
| 隔离测试 | [reports/isolation_metrics_report.md](reports/isolation_metrics_report.md) | v1 | 12 代表用例四分量分解、加速比与掩盖率（修正口径）、负值验证 |
| v6 单 barrier 化 | [reports/sync_barrier_opt_v6_report.md](reports/sync_barrier_opt_v6_report.md) | v6 | 每轮 SyncAll 3→1 消减，fence 移出关键路径的 A/B 验证 |
| v7 fence 重叠化探测 | [reports/p1_pipeline_v7_p0_probe_report.md](reports/p1_pipeline_v7_p0_probe_report.md) | v7 | fence 移入 PUT 传输窗口 + 计算核本地轮询，P0/P1 管线探测 |
| v8 reduceSum 优化 | [reports/p2_rs_optimization_report.md](reports/p2_rs_optimization_report.md) | v8 | 尾声批量加载 + UB 分区竞态修复的 A/B 验证 |
| 后续优化方向 | [reports/further_optimization_directions.md](reports/further_optimization_directions.md) | v8 | 基于 v6 A/B 与 op_summary 流水指标的证据链分析 |

## 当前版本性能总览 (v8, 2026-09-23)

### v8 vs v5 同日 A/B（全量 101/101 case）

| 指标 | 结果 |
|---|---|
| 提升用例 | **101/101 (零回退)** |
| 几何平均 | **-5.7% (0.9431x)** |
| 最大单 case | row17 (4096,5120,640,4): 159.2→95.7us, **-39.9%** |

### v8 优化贡献归因

| 优化项 | 提交 | 主要收益族 | 单项收益 |
|---|---|---|---|
| v6 单 barrier 化 (3T→T+1) | ee7c257b5 | 全部 (小 tile 更敏感) | -0.3% geomean |
| v7 fence 重叠化 + 本地轮询 | 1a1a587ee | 通信 bound / rank4 | -3.7% geomean |
| P1-b tile 策略 (cases.csv) | 03c4f545c | 小 N(640): -7~-16% | tiling 参数 |
| v8 尾声批量 RS + UB 分区 | 5567b7ecd | T=1 尾声主导 case | -4.9% geomean |

### 族维度规律 (v8 vs v5)

- **小 N (640) 提升最大 (-27.7%)**: P1-b tiling + v8 尾声批量 RS 叠加，**原融合回退场景反转为最优场景**
- **小 M / 少 tile (T=2, -8.6%)**: 逐轮同步开销占比高，v6/v7 barrier 消减与 fence 重叠化收益最直接
- **rank4 (-6.1%) 略优于 rank2 (-5.2%)**: 通信核多，v7 fence 重叠化收益放大
- **大 M (≥8192) 收窄至 -4.1%**: 通信量主导，受单通道 ~40GB/s 链路上限约束 (P0 已证不可破)

## v1 期外部基线结论（历史参考，2026-09-12）

> 以下为 v1 版本 (ed7d74662) 的 97 用例对比数据。v8 在此基础上全量提升 5.7%+（v5→v8），
> 外部相对优势进一步扩大。

| 对比维度 | 结果 |
|---|---|
| **vs 小算子级联基线** | **93/97 更快**，中位 **1.23x**，avg 1.25x，max 1.65x |
| **vs MC2 融合算子** | **88/97 更快**，中位 **1.14x**，avg 1.17x，max 1.69x |
| **通算融合加速比** (隔离) | 中位 **1.20x**，avg 1.22x，max 1.51x |
| **流水掩盖率** (修正) | 中位 **0.70**，通信 Bound 场景中位 0.83 |

### v1 期已知劣势与 v8 修复状态

| 模式 | v1 结论 | v8 状态 |
|---|---|---|
| 极窄 N=640 (row16/17) | 融合回退 0.83x | **已修复**: v8 P1-b + 批量 RS 反转为 -27.7% 提升 |
| K=5472/6144 大 K rank4 | matmul 效率差 ~10% | 部分缓解: v6/v7 同步消减贡献 ~3-5% |

## 后续优化方向 (v8 基线)

1. **大 K matmul 效率**（K > 4608）: 调优 baseK/stepK/scaleKL1 tiling 参数
2. **大 M 通信链路上限**: 单通道 ~40GB/s 已证不可破 (P0 probe)，需评估多通道/链路聚合
3. **rank2 场景**: 通信掩盖空间有限，重点提升 matmul 本身

## 目录结构

```
profiling/
├── README.md                          # 本文件（总览入口）
├── reports/                           # 📊 分析报告（读者入口）
│   ├── full_suite_v5_vs_v8_report.md  #   ★ v8 全量 101 case A/B + 归因 (最新)
│   ├── comparison_report.md           #   v1 期 97 用例外部基线对比
│   ├── deep_analysis_report.md        #   v1 期深入分析
│   ├── isolation_metrics_report.md    #   v1 期隔离测试
│   ├── sync_barrier_opt_v6_report.md  #   v6 单 barrier 化验证
│   ├── p1_pipeline_v7_p0_probe_report.md # v7 fence 重叠化探测
│   ├── p2_rs_optimization_report.md   #   v8 reduceSum 优化验证
│   └── further_optimization_directions.md # 后续优化方向分析
├── data/                              # 📁 v1 期原始数据（复现用）
│   ├── st_perf/                       #   ST 97 用例 PROF
│   ├── baseline/                      #   级联基线 + MC2 融合算子
│   └── isolation/                     #   隔离测试
└── (v5/v8 全量 PROF 数据: prof_v5_*/, prof_v8_*/ 见仓外或 prof/ 目录)
```

> **数据口径**: v8 报告采用同日同机交替双二进制 A/B (v5 md5=1883722f / v8 md5=7abed458)，
> 消除跨日环境漂移（09-12 报告级联基线跨日漂移达 29%，不作直接对比）。
