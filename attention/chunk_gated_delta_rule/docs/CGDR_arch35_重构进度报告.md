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
- **R3-R7 回退**：经与参考实现 (`chunk_gated_delta_rule_golden.py`) 比对，R3/R4/R5/R7 存在数学公式错误，R6 存在 strided 读取 bug，全部回退到原始算法
- 文档：`CGDR_arch35_重构前数据流伪代码.md`（638 行）— GM/WS/UB 布局、AIC/AIV 伪代码、同步图、存储总表
- 文档：`CGDR_arch35_重构后数据流与改动点.md` — R2 改动详解、数学错误分析、WS 布局对比、伪代码、同步图、公式对比、数据流图
- 编译通过：ascend950 opkernel + ophost + opapi

### R3-R7 回退原因详解

基于参考实现 `chunk_gated_delta_rule_golden.py:102-124` 的正确公式：

| 编号 | 错误 | 正确公式 | R3-R7 错误公式 |
|:-----|:-----|:---------|:---------------|
| R4 | 求逆不分配行缩放 | `A = (I - diag(β)·kk·γ)⁻¹` | `A' = (I-kk·γ)⁻¹`, `A_β = diag(β)·A'` |
| R3 | A@V_beta 项多了 g_exp | `v_inner = A@V_beta` (无g_exp) | `v_inner = (A@V_beta + ...)·g_exp` |
| R5 | 逐元素γ ≠ 矩阵乘对角g_exp | `A@diag(-K·β·g_exp)` | `(β·A·(-γ))@K` |
| R7 | 逐行向量 ≠ 标量 | `v_new^T @ (K·exp(g[-1]-g_cum))` | `(v_new·exp(g[-1]))^T @ K` |
| R6 | g(T,Nv)非连续读取bug | `DataCopyInFp32WithStride(len,1,src,nv)` | `DataCopyPad(src, {1, len*4, 0,0,0})` |

### 进行中
- 无

### 受阻
- RDV 用例验证未通过：`torch_npu` 2.7.1 未注册 `npu_chunk_gated_delta_rule` 接口；改用 `torch.ops.cann_ops_transformer` 调用方式后出现 AI Core Error（`timeout or trap error`，errno 507015），根因是 torch_extension ACLNN_CMD 调用路径与正式 op-plugin 路径存在差异

## 关键决策
- **R3-R7 全部回退**：数学公式错误无法通过代码修复，只能回退算法
- **R2 保留并修复**：纯数据流优化（strided GM 直读），不改变数学公式
- R2 SetOrgShape 修复：strided GM 用 5 参数 `SetOrgShape(m,n,ka,kb,n)` (orgKc=n 匹配 3 参数行为)，连续矩阵仍用 3 参数 `SetOrgShape(m,n,k)`
- R2 `!gOptional` 路径：QKPreProcess 删除后，`kg = K` 写入逻辑移至 GBKCompute 的 else 分支
- R2 sync0 删除：AIC 直接读 strided GM，不再需要等待 AIV 的 QKPreProcess 同步

## 后续步骤
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
- `attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage1.h` — R2 核心改动
- `attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule.h` — 主入口（回退到原始）
- `attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage2.h` — 回退到原始
- `attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage3.h` — 回退到原始
- `attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp` — `STAGE_ONE_THREE` 3→1
- `attention/chunk_gated_delta_rule/docs/CGDR_arch35_重构前数据流伪代码.md` — 重构前文档
- `attention/chunk_gated_delta_rule/docs/CGDR_arch35_重构后数据流与改动点.md` — 重构后文档（仅 R2）
