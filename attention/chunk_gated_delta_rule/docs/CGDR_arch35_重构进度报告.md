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
- R6 确认：原始代码已满足（gCum_ 输出到 GM + 后续重算）
- R4/R5/R7 数学方案验证：Python 验证误差 < 1e-12（`/tmp/opencode/verify_r4_r5_r7.py`）
- 文档：重构前/后数据流文档、进度报告

### 待实现
- R4：beta 列缩放 `A_beta = A @ diag(beta)`，删除 VBetaCompute/vBetaWsGm_，新增 ColumnScale + sync 0xB
- R5：g_exp 列缩放 `A_beta_gexp = A_beta @ diag(g_exp)`，删除 GBKCompute K缩放/gBKWsGm_，新增 attnWsNegGm_
- R7：d 移到 v_new 行缩放 `v_new *= d[:,None]`，删除 KgCompute/kg_，Stage2 CalStateNew 用 strided K

### 受阻
- RDV 用例验证：torch_extension 调用路径与 op-plugin 差异，AI Core Error

## 重构方案

| 编号 | 状态 | 方案 |
|:-----|:-----|:-----|
| R2 | **已实现** | matmul 直接读 strided GM（SetOrgShape Ka/Kb） |
| R3 | **不可行** | g_exp 在 A 右边（内维），A 非对角不可移到结果行 |
| R4 | **待实现** | `A_beta = A @ diag(beta)`（列缩放），V 不乘 beta，AIC 直接读 V |
| R5 | **待实现** | `A_beta_gexp = A_beta @ diag(g_exp)`（第二步列缩放），K 不乘 beta/g_exp，AIC 直接读 K |
| R6 | **已满足** | 原始代码已用 gCum_ 输出 + 后续重算 |
| R7 | **待实现** | `v_new *= d[:,None]`（d 从 gCum_ 重算），K 不乘 d，AIC 直接读 K |

数学验证（C=8, float64）：
```
v_inner diff:    6.94e-18
k_cumdecay diff: 1.39e-17
v_new diff:      1.39e-17
state_new diff:  1.39e-17
```

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
- Stage1 WS（R2 后）：kkWs + attnWs + vBetaWs（R4/R5 后改为 kkWs + attnWs + attnWsNeg）
- `MatmulShapeParams` 新增 `ka`/`kb`（默认0=连续，>0=strided）
- `STAGE_ONE_THREE`: 3 → 1

## 相关文件
- `op_kernel/arch35/chunk_gated_delta_rule_stage1.h` — R2 已实现, R4/R5/R7 待实现
- `op_kernel/arch35/chunk_gated_delta_rule_stage2.h` — R7 待实现
- `op_kernel/arch35/chunk_gated_delta_rule.h` — 主入口
- `op_host/chunk_gated_delta_rule_tiling.cpp` — WS 大小计算
- `docs/CGDR_arch35_重构前数据流伪代码.md` — 重构前文档
- `docs/CGDR_arch35_重构后数据流与改动点.md` — 重构后文档
