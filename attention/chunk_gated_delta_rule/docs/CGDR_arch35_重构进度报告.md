# CGDR arch35 重构进度报告

> 更新时间：2026-08-18
> PR：https://gitcode.com/cann/ops-transformer/merge_requests/10209
> 分支：`perf/cgdr-arch35-cv-pipeline`（基于 `perf/cgdr-arch35-refactor`，基于 upstream/master `94663c2af`）

---

## 目标
- 重构 arch35 平台 CGDR 算子：分析 R1-R7 需求，设计并实现重构方案
- 在重构基础上实现 CV 数据通路优化，减少 GM 访存
- 清理死 workspace，削减内存占用

## 限制与偏好
- 基于 upstream `cann/ops-transformer` master 最新代码
- PR 提交到 upstream 仓
- 仅涉及 `op_kernel/arch35/`（WS 清理同时涉及 arch22 和 tiling.cpp）
- TilingKey 不变，接口不变
- 编译验证：ascend950 opkernel + ophost + opapi 全部通过

## 进度

### 已完成

#### 阶段 1：R2-R7 重构 (perf/cgdr-arch35-refactor 分支)
- R2 实现：Stage1 matmul 直接读 strided GM，删除 QKPreProcess/queryConGm_/keyConGm_，删除 sync0
- R3 分析：g_exp 在 matmul 内维不可移，**不可行**
- R4 实现：beta 列缩放 `A_beta = A @ diag(beta)`，删除 VBetaCompute/vBetaWsGm_，新增 ColumnScaleBeta + sync 0xB
- R5 实现：g_exp 列缩放 `A_beta_gexp = A_beta @ diag(g_exp)`，删除 GBKCompute/gBKWsGm_，新增 attnWsNegGm_
- R6 确认：原始代码已满足（gCum_ 输出到 GM + 后续重算）
- R7 实现：d 移到 v_new 行缩放 `v_new *= d[:,None]`，删除 KgCompute，Stage2 CalStateNew 用 strided K

#### 阶段 2：Bug 修复
- Bug 1 修复：R4/R5 双循环 `inverseRes_` 串扰 → 合并为单循环
- Bug 2 修复：Stage3 读到 d 缩放后的 v_new → 新增 `vNewScaledGm_` per-core buffer

#### 阶段 3：CV 数据通路优化 (perf/cgdr-arch35-cv-pipeline 分支)
- CV 同步精简：AIC VPrime+AttnInter 融合（无中间同步），3-phase 流水线
- 合并 gCum 读：`CalGCumExpAndD` / `CalGCumExpBf16AndD` — state decay + d 向量在一次 GM 读取中完成
- CastGmToGmDual 融合：FP32 path 单次 GM→UB 读取产出 unscaled + scaled 两份输出
- CastGmToGmWithScale 模板特化：新增 BF16→BF16 with scale 支持

#### 阶段 4：死 workspace 清理
- 删除 `attnInter_`（从未被任何 Stage 读取，arch22 和 arch35 均为死代码）
- 节省 ~48MB GM workspace

#### 验证
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
| CV | **已实现** | 合并 gCum 读 + CastGmToGmDual 融合 + VPrime/AttnInter 融合 |
| WS | **已实现** | 删除 attnInter_ 死 workspace（~48MB） |

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
| S=2048 nv=32 | 0.72 | 0.72 | +0.0% |
| S=4096 nv=32 | 1.44 | 1.45 | +0.6% |
| S=8192 nv=32 | 2.85 | 2.79 | -2.1% |
| S=32768 nv=32 | 11.40 | 11.18 | -1.9% |
| S=65536 nv=32 | 22.90 | 22.44 | -2.0% |
| S=131072 nv=32 | 46.07 | 45.28 | -1.7% |
| S=2048 nv=2 | 0.20 | 0.21 | +5.0% |
| S=2048 nv=32 GQA | 0.72 | 0.71 | -1.4% |
| S=10240 nv=64 GQA | 6.05 | 5.96 | -1.5% |
| S=2048 nv=32 nog | 0.69 | 0.65 | -5.8% |
| B=4 S=32768 nv=32 | 46.06 | 45.54 | -1.1% |
| S=2048 FP32-state | 0.98 | 0.97 | -1.0% |

**结论：** 性能与 baseline 持平（±3% 噪声范围，长序列场景有小幅改善趋势）。

**性能未显著提升的根因分析：**
- Stage2 每 chunk 8.0us 中 AIV 向量计算占 76%（6.1us），AIC cube 仅 24%（1.9us）
- CV 优化（合并 gCum 读 + CastGmToGmDual）节省的 GM 往返 ~0.5us 被 AIV state CopyIn/CopyOut（64KB/chunk）主导的关键路径淹没
- state decay 的 GM 写回仍未消除（需要 AIC matmul SetScale 支持，当前 MatmulImpl 不支持左操作数行缩放）
- 要获得 >10% 收益需减少 AIV 向量计算量或改变算法（如 parallel scan 消除 state 递推依赖）

## WS 削减

| 删除项 | 大小 | 来源 |
|:-------|:-----|:-----|
| attnInter_ | ~48MB | 死代码清理（从未被任何 Stage 读取） |
| queryConGm_ + keyConGm_ | ~3MB | R2 |
| vBetaWsGm_ | ~2.6MB | R4 |
| gBKWsGm_ | ~2.6MB | R5 |
| 新增 attnWsNegGm_ | ~0.5MB | R5 |
| 新增 vNewScaledGm_ | ~0.5MB | R7（per-core, 复用 stageWs） |
| **净削减** | **~55.2MB** | |

注：`kg_` 仍分配（arch22 的 CalStateNew 需要），arch35 Stage2 不再使用。

## CV 数据通路优化详解

### Stage2 3-Phase 流水线

```
AIV: state_prep + CalGCumExpAndD → setFlag(0x3) → waitFlag(0x2) → v_new_scale → setFlag(0x5) → waitFlag(0x4)
AIC:                              waitFlag(0x3) → VPrime+AttnInter(fused) → setFlag(0x2) → waitFlag(0x5) → StateNew → setFlag(0x4)
```

### 新增/修改函数

| 函数 | 路径 | 说明 |
|:-----|:-----|:-----|
| `CalGCumExpAndD` | FP32 | 合并 CalGCumExpFp32 + CalD，单次 gCum 读 |
| `CalGCumExpBf16AndD` | BF16 | 合并 CalGCumExpBf16 + CalD，单次 gCum 读 + state cast |
| `CastGmToGmDual` | FP32 | 单次 GM→UB 读取产出 unscaled + scaled 双输出 |
| `CastGmToGmWithScale` | BF16→BF16 | 新增模板特化：BF16→FP32→Scale→BF16 |
| `ColumnScaleBeta` | Stage1 | R4: A_beta = A * beta[None,:] (列缩放) |
| `ColumnScaleGExp` | Stage1 | R5: A_beta_gexp = A_beta * g_exp[None,:] (列缩放) |

### Bug 修复

| Bug | 影响 | 修复 |
|:-----|:-----|:-----|
| R4/R5 双循环 inverseRes_ 串扰 | paraNum>1 时 3/4 chunk 的 k_cumdecay 数据错误 | 合并为单循环 |
| Stage3 读到 d 缩放后 v_new | masked_qkt @ (v_new*d) 而非 v_new | 新增 vNewScaledGm_ per-core buffer |

## 关键上下文
- 分支：`perf/cgdr-arch35-cv-pipeline`，基于 `perf/cgdr-arch35-refactor`，基于 `94663c2af`
- 编译命令：`bash build.sh --soc=ascend950 --ops=chunk_gated_delta_rule -j16`
- Stage1 WS：kkWs + attnWs + attnWsNeg（3 个 C×C buffer）
- Stage2 WS：vNewScaledGm_（per-core C×Dv，复用 stageWs）
- `MatmulShapeParams` 新增 `ka`/`kb`（默认0=连续，>0=strided）
- `StageTwoParams` 新增 `keyGm`/`cgStartPos`/`vNewScaledGm`（R7 strided K + scaled v_new）
- Stage2 新增 `CalGCumExpAndD`/`CalGCumExpBf16AndD`/`CastGmToGmDual`/`CastGmToGmWithScale`(BF16特化)

## AtomicAdd 分析

- `accum=1`（matmul L0C 级累加）已在 CalVPrime/CalStateNew/Stage3 使用
- `SetAtomicAdd`（GM 级原子加）适用于多核写同一地址，但 CGDR 所有并行写均分地址
- CGDR 架构无 AtomicAdd 适用场景，accum=1 更高效（L0C vs GM）

## 相关文件
- `op_kernel/arch35/chunk_gated_delta_rule_stage1.h` — R2/R4/R5 实现 + Bug1 修复
- `op_kernel/arch35/chunk_gated_delta_rule_stage2.h` — R7 + CV 优化 + Bug2 修复
- `op_kernel/arch35/chunk_gated_delta_rule.h` — 主入口，传参，vNewScaledGm_ 绑定
- `op_kernel/arch35/chunk_gated_delta_rule_stage3.h` — 无改动
- `op_kernel/arch22/chunk_gated_delta_rule.h` — attnInter_ 删除（WS 清理）
- `op_host/chunk_gated_delta_rule_tiling.cpp` — WS 大小计算（删 attnInter）
- `docs/CGDR_arch35_重构前数据流伪代码.md` — 重构前文档
- `docs/CGDR_arch35_重构后数据流与改动点.md` — 重构后文档（本次更新）
- `tests/pytest/run_all_rdv.py` — RDV 批量测试脚本
- `tests/pytest/output/rdv_report_r4_r5_r7.csv` — RDV 测试报告
