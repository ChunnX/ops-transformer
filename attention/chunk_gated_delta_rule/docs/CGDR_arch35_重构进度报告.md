# CGDR arch35 重构进度报告

> 更新时间：2026-08-20
> PR：https://gitcode.com/cann/ops-transformer/merge_requests/10209

---

## 目标
- 分析 PR10209 CGDR 重构方案中的问题
- 修正问题，确保 RDV 288 用例全部通过

## 分析结果

### 重构方案中的 Bug（6个致命问题）

| Bug | 严重程度 | 根因 | 修复方式 |
|:-----|:---------|:-----|:-----|
| R4/R5 Broadcast 方向错误 | 致命 | `Broadcast<axis=1>`(行广播) 实际是 `diag(β)@A`(行缩放) 而非 `A@diag(β)`(列缩放) | 回退原始 V/K 行缩放 |
| R4/R5 beta 读取不完整 | 致命 | 只读 halfChunkSize 个 beta（半块），列缩放不完整 | 回退原始 V/K 行缩放 |
| R2 strided-A matmul 不可行 | 致命 | `SetOrgShape(orgKa)` 同时控制 K 迭代计数和 A 地址步进，非转置 strided A 两者冲突 | 回退原始 contiguous QK copy |
| R7 strided-B K 不可行 | 致命 | transB=false 的 strided B 同样不支持 | 回退原始 kg_ 路径 |
| R5 FIX→MTE2 竞态 | 致命 | AIC `AttnInverseMMCompute`(PIPE_FIX) 修改 attnWs 后 AIV `ColumnScaleGExp`(PIPE_MTE2) 读取，跨管道同步不支持 | 回退原始 GBKCompute |
| tiling attnInter_ 缺失 | 致命 | 重构删除了 attnInter_ workspace 分配，但 arch22 兼容代码仍引用 | 恢复 attnInter_ 分配 |

### 预存在 Bug（1个，本次修复）

| Bug | 严重程度 | 根因 | 修复方式 |
|:-----|:---------|:-----|:-----|
| tiling stateFp32Wk_ 缺失 | 致命 | BF16-state on ASCEND950 未分配 stateFp32Wk_，workspace 溢出→AICore 崩溃 | 移除 `socVersion != ASCEND950` 条件 |

### R4/R5 不可行的技术限制

1. **精度回归**：R4 将 β 从 V 行缩放移到 A 列缩放，BF16 舍入路径改变导致 ~47/288 用例精度超阈值
2. **FIX→MTE2 同步**：R5 需要在 AIV 读取 AIC 修改后的 attnWsGm_（PIPE_FIX→PIPE_MTE2），Ascend950 不支持跨管道数据可见性同步

## 修正方案

回退所有 R2/R4/R5/R7/CV 重构改动至 upstream 原始代码，仅保留 tiling 修复：
1. `stateFp32Wk_` 分配：移除 `socVersion != ASCEND950` 条件，BF16-state 始终分配
2. `attnInter_` 分配：恢复被重构删除的 workspace 分配

## RDV 精度验证

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

## 相关文件
- `op_kernel/arch35/chunk_gated_delta_rule_stage1.h` — 回退至 upstream 原始
- `op_kernel/arch35/chunk_gated_delta_rule_stage2.h` — 回退至 upstream 原始
- `op_kernel/arch35/chunk_gated_delta_rule_stage3.h` — 回退至 upstream 原始
- `op_kernel/arch35/chunk_gated_delta_rule.h` — 回退至 upstream 原始
- `op_kernel/arch35/chunk_gated_delta_rule_utils.h` — 移除 PREFETCH_MTE2_V_EVENT
- `op_host/chunk_gated_delta_rule_tiling.cpp` — stateFp32Wk_ + attnInter_ 分配修复
- `tests/pytest/run_all_rdv.py` — 新增可选固定种子支持
