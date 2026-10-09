# profiling 目录 — quant_matmul_reduce_scatter (apace ST) 性能分析总览

> **代码版本**: ops-transformer upstream-apace @ 20aec923b (v8 + 检视修复 + 注释清理 + L1 排空条件化)
> **环境**: Ascend950PR ×4, CANN 9.2.0 (`/home/l00947526/Ascend915/cann-9.2.0`), bisheng (dav-3510)
> **数据采集**: 2026-09-10 ~ 2026-09-28 (v1 基线 ~ v8 全量 + 隔离指标重测)
> **当前版本精度**: 101/101 case PASS

## 报告索引

| 报告 | 位置 | 内容 |
|---|---|---|
| **★ 全量 v5 vs v8 对比** | [reports/full_suite_v5_vs_v8_report.md](reports/full_suite_v5_vs_v8_report.md) | **101/101 用例** v5/v8 同日同机交替 A/B；优化收益归因、机制要点、已验证不可行/关闭方向 (P0/P3/P2-AIC) |
| 隔离测试（当前代码重测） | [reports/isolation_metrics_report.md](reports/isolation_metrics_report.md) | 四分量分解 (T_mm/T_a2a/T_rs/T_before/T_after)、融合加速比与掩盖率（12 代表 case），与 v1 期对比 |
| tileM 公式与组合策略 | [reports/formula_tiling_scenario_report.md](reports/formula_tiling_scenario_report.md) | 公式场景 56 case A/B + 掩盖率 + 2×2 组合矩阵 + **组合表落地验证**（21 case 全新 A/B geomean −9.65%，101 全集 ≈ −2.0%，精度 21/21） |
| 策略实验全记录 | [reports/tiling_strategy_experiment_report.md](reports/tiling_strategy_experiment_report.md) | 四阶段实验完整记录：公式/顺序/组合矩阵/落地，含过程事件、数据修正与全量逐 case 附录 |
| 组件归因分析 | [reports/component_attribution_report.md](reports/component_attribution_report.md) | 大额收益来源定位：公式=T_mm 单一来源(103~133%, Bound 迁移, shape 分簇)；row27 顺序=fixpipe 尾暴露(T_mm −7.6%, 机制闭合) |

> v1 期外部基线对比 (97 用例 vs 级联/MC2 融合)、v6/v7/v8 分轮专项 A/B 报告已移除:
> 核心结论分别保留于本文件「v1 期外部基线结论」小节与 full_suite 报告第三/四节, 明细可查 git 历史。

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
| v6 单 barrier 化 (3T→T+1) | 7543af04a | 全部 (小 tile 更敏感) | -0.3% geomean |
| v7 fence 重叠化 + 本地轮询 | 865dc1ee4 | 通信 bound / rank4 | -3.7% geomean |
| P1-b tile 策略 (cases.csv) | 6f8c056ca | 小 N(640): -7~-16% | tiling 参数 |
| v8 尾声批量 RS + UB 分区 | 4d3de33a3 | T=1 尾声主导 case | -4.9% geomean |

### 族维度规律 (v8 vs v5)

- **小 N (640) 提升最大 (-27.7%)**: P1-b tiling + v8 尾声批量 RS 叠加，**原融合回退场景反转为最优场景**
- **小 M / 少 tile (T=2, -8.6%)**: 逐轮同步开销占比高，v6/v7 barrier 消减与 fence 重叠化收益最直接
- **rank4 (-6.1%) 略优于 rank2 (-5.2%)**: 通信核多，v7 fence 重叠化收益放大
- **大 M (≥8192) 收窄至 -4.1%**: 通信量主导，受单通道 ~40GB/s 链路上限约束 (P0 已证不可破)

### 组合 tiling 表（2026-09-29 落地）

cases.csv 逐 case 组合（17 case 公式 tiling + 4 case localScopeLast=1）：21 变更 case 全新 A/B
**geomean −9.65%**（折算 101 全集 ≈ **−2.0%**），精度 21/21 零回退——见
[formula_tiling_scenario_report.md](reports/formula_tiling_scenario_report.md) 第五/六节。

### 隔离指标（2026-09-28 当前代码重测，12 代表 case）

| 指标 | 结果 |
|---|---|
| 融合加速比 | 中位 **1.29x**（min 1.02x / max 1.48x，全部 ≥ 1 无融合回退） |
| 掩盖率（修正） | 中位 **0.68**（通信 Bound 族 0.83 / 均衡族 0.57 / 计算 Bound 0.14） |
| 明细 | [reports/isolation_metrics_report.md](reports/isolation_metrics_report.md) |

## v1 期外部基线结论（历史参考，2026-09-12）

> 以下为 v1 版本 (ed7d74662) 的 97 用例对比数据结论（报告明细已移除, git 历史可查）。
> v8 在此基础上全量提升 5.7%+（v5→v8），外部相对优势进一步扩大。

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
├── README.md                             # 本文件（总览入口 + v1 期外部基线结论留存）
└── reports/
    ├── full_suite_v5_vs_v8_report.md     # ★ v8 全量 101 case A/B + 归因 + 机制要点 + 已关闭方向
    ├── isolation_metrics_report.md       # 隔离测试（当前代码重测）：四分量分解 + 加速比/掩盖率
    ├── formula_tiling_scenario_report.md # tileM 公式化场景验证：56 case A/B + 掩盖率对比
    ├── tiling_strategy_experiment_report.md # 策略实验全记录（四阶段 + 附录）
    └── component_attribution_report.md    # 大额收益组件归因（机制闭合）
```

> **数据口径**: v8 报告采用同日同机交替双二进制 A/B (v5 md5=1883722f / v8 md5=7abed458)，
> 消除跨日环境漂移（09-12 报告级联基线跨日漂移达 29%，不作直接对比）。
> 原始 msprof PROF 数据体积较大不入库，各报告表格为最终汇总；复现路径:
> `../scripts/gen_data.py` → `../run.sh` + msprof 采集 → `../scripts/parse_prof.py`。
