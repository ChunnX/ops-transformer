# CGDR arch35 重构进度报告

> 更新时间：2026-08-17
> PR：https://gitcode.com/cann/ops-transformer/merge_requests/9968
> 分支：`perf/cgdr-arch35-refactor`（基于 upstream/master `94663c2af`）

---

## 目标
- 重构 arch35 平台 CGDR 算子：分析 R1-R7 需求，设计并实现重构方案，编写文档，提交 PR。

## 限制与偏好
- 基于 upstream `cann/ops-transformer` master 最新代码
- PR 提交到 upstream 仓
- 仅涉及 `op_kernel/arch35/`，arch22 不受影响
- TilingKey 不变，接口不变
- 编译验证：ascend950 opkernel + ophost + opapi 全部通过

## 进度

### 已完成
- R2 实现：Stage1 matmul 直接读 strided GM，删除 QKPreProcess/queryConGm_/keyConGm_，删除 sync0
- R3 分析：g_exp 在 matmul 内维不可移，**不可行**
- R4 实现：beta 列缩放 `A_beta = A @ diag(beta)`，删除 VBetaCompute/vBetaWsGm_，新增 ColumnScaleBeta + sync 0xB
- R5 实现：g_exp 列缩放 `A_beta_gexp = A_beta @ diag(g_exp)`，删除 GBKCompute/gBKWsGm_，新增 attnWsNegGm_
- R6 确认：原始代码已满足（gCum_ 输出到 GM + 后续重算）
- R7 实现：d 移到 v_new 行缩放 `v_new *= d[:,None]`，删除 KgCompute/kg_，Stage2 CalStateNew 用 strided K
- 编译通过：ascend950 opkernel + ophost + opapi
- 数学验证：Python 验证误差 < 1e-12
- RDV 精度验证：288 用例全部 PASS（100%）
- 性能验证：与 baseline 持平（±3%）

## 重构方案

| 编号 | 状态 | 方案 |
|:-----|:-----|:-----|
| R2 | **已实现** | matmul 直接读 strided GM（SetOrgShape Ka/Kb） |
| R3 | **不可行** | g_exp 在 A 右边（内维），A 非对角不可移到结果行 |
| R4 | **已实现** | `A_beta = A @ diag(beta)`（列缩放），V 不乘 beta，AIC 直接读 V |
| R5 | **已实现** | `A_beta_gexp = A_beta @ diag(g_exp)`（第二步列缩放），K 不乘 beta/g_exp，AIC 直接读 K |
| R6 | **已满足** | 原始代码已用 gCum_ 输出 + 后续重算 |
| R7 | **已实现** | `v_new *= d[:,None]`（d 从 gCum_ 重算），K 不乘 d，AIC 直接读 K |

数学验证（C=8, float64）：
```
v_inner diff:    6.94e-18
k_cumdecay diff: 1.39e-17
v_new diff:      1.39e-17
state_new diff:  1.39e-17
```

## RDV 精度验证

### 总览

| 用例集 | 总数 | PASS | FAIL | 通过率 |
|:-------|:-----|:-----|:-----|:-------|
| REDLINE (ARC-001~038) | 38 | 38 | 0 | 100% |
| REDLINE-FP32 | 38 | 38 | 0 | 100% |
| REDLINE-NC | 38 | 38 | 0 | 100% |
| REDLINE-FP32-NC | 38 | 38 | 0 | 100% |
| STC (ASC-001~034) | 34 | 34 | 0 | 100% |
| STC-FP32 | 34 | 34 | 0 | 100% |
| STC-NC | 34 | 34 | 0 | 100% |
| STC-FP32-NC | 34 | 34 | 0 | 100% |
| **总计** | **288** | **288** | **0** | **100%** |

### 覆盖路径
- gOptional / !gOptional
- BF16 / FP32 state
- 连续 / 非连续 state
- GQA (Nk≠Nv)
- 各种 shape 组合（S=128~262144, nv=2~64, dk/dv=1~128）

### 关键精度指标
所有用例的 `max_re_npu ≈ max_re_high_type`（NPU 误差 = benchmark 误差），确认重构未引入额外精度损失。

## 性能验证

| 用例 | Baseline (ms) | Refactored (ms) | 变化 |
|:-----|:-------------|:----------------|:-----|
| S=2048 nv=32 | 0.72 | 0.74 | +2.8% |
| S=4096 nv=32 | 1.45 | 1.44 | -0.7% |
| S=8192 nv=32 | 2.83 | 2.82 | -0.4% |
| S=32768 nv=32 | 11.39 | 11.43 | +0.4% |
| S=65536 nv=32 | 22.84 | 22.88 | +0.2% |
| S=131072 nv=32 | 46.11 | 46.05 | -0.1% |
| S=2048 nv=2 | 0.20 | 0.20 | 0% |
| S=2048 nv=32 GQA | 0.73 | 0.72 | -1.4% |
| S=10240 nv=64 GQA | 6.05 | 6.04 | -0.2% |
| S=2048 nv=32 nog | 0.68 | 0.68 | 0% |
| B=4 S=32768 nv=32 | 46.22 | 46.15 | -0.2% |
| S=2048 FP32-state | 0.96 | 0.98 | +2.1% |

**结论：** 性能与 baseline 持平（±3% 以内，测量噪声范围）。
R4+R5+R7 把计算从 AIC matmul 前的 AIV 向量计算移到了 AIC matmul 后的 AIV 列缩放，
同时新增 sync flag 0xB 串行等待。计算量削减被新增同步开销抵消。
主要收益在 **workspace 削减（~15.7MB）**，对大 batch/长序列场景的内存压力有实质帮助。

## WS 削减

| 删除项 | 大小 | 来源 |
|:-------|:-----|:-----|
| queryConGm_ + keyConGm_ | ~3MB | R2 |
| vBetaWsGm_ | ~2.6MB | R4 |
| gBKWsGm_ | ~2.6MB | R5 |
| kg_ | ~8MB | R7 |
| 新增 attnWsNegGm_ | ~0.5MB | R5 |
| **净削减** | **~15.7MB** | |

## 关键上下文
- 分支：`perf/cgdr-arch35-refactor`，基于 `94663c2af`
- 编译命令：`bash build.sh --soc=ascend950 --ops=chunk_gated_delta_rule -j16`
- Stage1 WS：kkWs + attnWs + attnWsNeg（3 个 C×C buffer）
- `MatmulShapeParams` 新增 `ka`/`kb`（默认0=连续，>0=strided）
- `StageTwoParams` 新增 `keyGm`/`cgStartPos`（R7 strided K）
- Stage2 新增 `CalD`/`CastGmToGmWithScale`/`ScaleVInnerBf16`（R7 v_new 行缩放）

## 相关文件
- `op_kernel/arch35/chunk_gated_delta_rule_stage1.h` — R2/R4/R5 实现
- `op_kernel/arch35/chunk_gated_delta_rule_stage2.h` — R7 实现
- `op_kernel/arch35/chunk_gated_delta_rule.h` — 主入口，传参
- `op_host/chunk_gated_delta_rule_tiling.cpp` — WS 大小计算（3 个 C×C）
- `docs/CGDR_arch35_重构前数据流伪代码.md` — 重构前文档
- `docs/CGDR_arch35_重构后数据流与改动点.md` — 重构后文档
- `tests/pytest/run_all_rdv.py` — RDV 批量测试脚本
- `tests/pytest/output/rdv_report_r4_r5_r7.csv` — RDV 测试报告
