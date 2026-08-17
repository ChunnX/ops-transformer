# CGDR arch35 重构进度报告

> 更新时间：2026-08-17
> PR：https://gitcode.com/cann/ops-transformer/merge_requests/9968
> 分支：`perf/cgdr-arch35-refactor`（基于 upstream/master `94663c2af`）

---

## 目标
- 重构 arch35 平台 CGDR (ChunkGatedDeltaRule) 算子：分析 R1-R7 需求，设计并实现重构方案，编写重构前/后数据流文档，提交 PR 到 upstream `cann/ops-transformer` master 分支。

## 限制与偏好
- 必须基于 upstream `cann/ops-transformer` 仓 master 分支最新代码
- PR 必须提交到 upstream 仓（非 fork 内部 MR）
- 仅涉及 `op_kernel/arch35/` 目录，arch22 不受影响
- TilingKey 不变，接口不变
- 编译验证：ascend950 opkernel + ophost + opapi 全部通过
- NPU 环境：Ascend950PR, CANN 9.1.0, torch_npu 2.7.1

## 进度
### 已完成
- R1 分析：建议保留批处理+去除 Stage 间 SyncAll（仅分析结论，无代码变更）
- R2 实现：Stage1 移除 QKPreProcess，AIC matmul 直接读 strided GM（`SetOrgShape(M,N,Ka,Kb,N)`），删除 `queryConGm_`/`keyConGm_` WS，`!gOptional` 路径 kg_ 写入移至 `GBKCompute`，删除 sync0 (`0x9`)
- **R3/R6 回退**：经与参考实现 (`chunk_gated_delta_rule_golden.py`) 比对，R3 的 g_exp 在 matmul 内维不可移，R6 存在 strided 读取 bug
- **R4 待实现**：原方案用行缩放错误，修正为列缩放 `A_beta = A @ diag(beta)`（第 j 列乘 beta[j]），数学等价
- **R5 待实现**：原方案用逐元素乘 C×C 矩阵 gamma 错误，修正为列缩放 C 维向量 g_exp `A_beta_gexp = A_beta @ diag(g_exp)`，数学等价
- **R7 待实现**：原方案标量缩放错误，修正为逐行向量缩放 `v_new * d[:,None]`（`d[j]=exp(g[-1]-g_cum[j])`），数学等价
- 文档：`CGDR_arch35_重构前数据流伪代码.md`（638 行）— GM/WS/UB 布局、AIC/AIV 伪代码、同步图、存储总表
- 文档：`CGDR_arch35_重构后数据流与改动点.md` — R2 改动详解、数学错误分析、WS 布局对比、伪代码、同步图、公式对比、数据流图
- 编译通过：ascend950 opkernel + ophost + opapi

### R3/R6 回退原因 + R4/R5/R7 修正方案

基于参考实现 `chunk_gated_delta_rule_golden.py:102-149` 的正确公式：

| 编号 | 状态 | 正确公式 | 原方案错误 |
|:-----|:-----|:---------|:---------------|
| R3 | 回退 | g_exp 在 `A@diag(g)@K` 中位于 A 右边(内维), 不可移 | 移到结果行缩放, A 非对角混合后丢失 |
| R4 | **待实现** | `A_beta=A@diag(β)`(列缩放β), `v_inner=A_beta@V` | 原方案: 从KKT删beta + 行缩放(两个错误) |
| R5 | **待实现** | `A_beta_gexp=A_beta@diag(g_exp)`(列缩放g_exp), `k_cumdecay=-A_beta_gexp@K` | 原方案: 逐元素乘C×C矩阵gamma ≠ 列缩放C维向量g_exp |
| R6 | 回退 | `DataCopyInFp32WithStride(len,1,src,nv)` | g(T,Nv)非连续, DataCopyPad读跨head垃圾 |
| R7 | **待实现** | `v_new^T@diag(d)@K = (v_new·d[:,None])^T@K` (d在v^T右边可移) | 原方案: 标量exp(g[-1]) ≠ 逐行exp(g[-1]-g_cum[j]) |

**R4 正确方案要点：**
- 利用结合律 `A@diag(β)@V = (A@diag(β))@V`，beta 从 V/K 左边移到 A 右边
- `A_beta = A @ diag(beta)`：**列缩放**（第 j 列乘 beta[j]），非行缩放
- KKT/求逆不变（beta 仍在求逆内部），AttnInverseMMCompute 后做列缩放
- 删除 VBetaCompute + vBetaWsGm_ WS，GBKCompute 去掉 beta，AIC 直接读 V
- 新增同步 flag 0xB（AIC AttnInverseMMCompute 完成后通知 AIV 列缩放）

**R5 正确方案要点（基于 R4）：**
- 在 A_beta 基础上再做 g_exp 列缩放：`A_beta_gexp = A_beta @ diag(g_exp)`
- 两步列缩放等价：`diag(β)·diag(g_exp) = diag(β·g_exp)`
- `k_cumdecay = -A_beta_gexp @ K`（K 不再乘 β·g_exp，直接用 strided GM）
- `v_inner = A_beta @ V`（不变，v_inner 不需要 g_exp）
- 澄清：g_exp（C 维向量 `exp(g_cum[j])`）≠ gamma（C×C 矩阵 decay_mask）
- 删除 gBKWsGm_ WS + GBKCompute 的 K 行缩放，新增 attnWsNegGm_ WS（存 A_beta_gexp）

**R7 正确方案要点：**
- 利用结合律 `v^T@diag(d)@K = (v·d)^T@K`，d 从 K 左边移到 v_new 行
- `d[j] = exp(g[-1]-g_cum[j])`：**逐行向量**（每行 j 不同），非标量
- d 在 `v^T@diag(d)@K` 中位于 v^T 右边（v 的行=求和下标），可直接移
- 与 R3 的 g_exp 区别：g_exp 在 A 右边（A 的列），A 非对角混合后不可移
- 删除 KgCompute + kg_ GM WS（~8MB），Stage2 AIV 做 v_new 行缩放，AIC 直接用 strided K

### 进行中
- 无

### 受阻
- RDV 用例验证未通过：`torch_npu` 2.7.1 未注册 `npu_chunk_gated_delta_rule` 接口；改用 `torch.ops.cann_ops_transformer` 调用方式后出现 AI Core Error（`timeout or trap error`，errno 507015），根因是 torch_extension ACLNN_CMD 调用路径与正式 op-plugin 路径存在差异

## 关键决策
- **R3/R6 全部回退**：R3 的 g_exp 在 matmul 内维不可移，R6 strided 读取 bug
- **R4 修正后待实现**：原方案行缩放错误，改为列缩放 `A@diag(beta)` 数学等价
- **R5 修正后待实现**：原方案逐元素乘 C×C 矩阵错误，改为列缩放 C 维向量 g_exp，基于 R4 两步列缩放
- **R7 修正后待实现**：原方案标量缩放错误，改为逐行向量缩放 `v_new*d[:,None]` 数学等价
- **R2 已实现并修复**：纯数据流优化（strided GM 直读），不改变数学公式
- R2 SetOrgShape 修复：strided GM 用 5 参数 `SetOrgShape(m,n,ka,kb,n)` (orgKc=n 匹配 3 参数行为)，连续矩阵仍用 3 参数 `SetOrgShape(m,n,k)`
- R2 `!gOptional` 路径：QKPreProcess 删除后，`kg = K` 写入逻辑移至 GBKCompute 的 else 分支
- R2 sync0 删除：AIC 直接读 strided GM，不再需要等待 AIV 的 QKPreProcess 同步

## 后续步骤
- 实现 R4 正确方案（beta 列缩放）：删除 VBetaCompute/vBetaWsGm_，新增列缩放 pass + sync flag 0xB
- 实现 R5 正确方案（g_exp 列缩放，基于 R4）：第二步列缩放 → A_beta_gexp，删除 gBKWsGm_/GBKCompute K缩放
- 实现 R7 正确方案（kg 系数移到 v_new）：删除 KgCompute/kg_，Stage2 AIV 做 v_new 行缩放
- 在正式环境（有 op-plugin 的 torch_npu）验证 RDV 用例精度
- 验证 gOptional 和 !gOptional 路径
- 验证 FP32 state path 和非连续 state

## 关键上下文
- Git fork: `https://gitcode.com/zzy__/ops-transformer.git`，upstream: `https://gitcode.com/cann/ops-transformer.git`
- 当前分支：`perf/cgdr-arch35-refactor`，基于 `94663c2af`（upstream/master 最新）
- 编译命令：`bash build.sh --opkernel --soc=ascend950 --ops=chunk_gated_delta_rule -j16`
- 安装命令：`bash build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --force`
- Stage1 WS 布局（R2 后）：gBKWs(C*Dk) + kkWs(C*C) + attnWs(C*C) + vBetaWs(C*Dv) = 4 个 WS buffer
- `STAGE_ONE_THREE`: 3 → 1（删除 queryCon/keyCon）
- `MatmulShapeParams` 结构体新增 `ka`/`kb` 字段（默认0=连续，>0=strided）
- `AICProcess`：`ka>0` 用 5 参数 SetOrgShape，`ka==0` 用 3 参数 SetOrgShape

## 相关文件
- `attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage1.h` — R2 已实现, R4/R5/R7 待实现
- `attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule.h` — 主入口（回退到原始）
- `attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage2.h` — R7 待实现
- `attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage3.h` — 回退到原始
- `attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp` — `STAGE_ONE_THREE` 3→1
- `attention/chunk_gated_delta_rule/docs/CGDR_arch35_重构前数据流伪代码.md` — 重构前文档
- `attention/chunk_gated_delta_rule/docs/CGDR_arch35_重构后数据流与改动点.md` — 重构后文档（R2 已实现 + R4/R5/R7 待实现）
