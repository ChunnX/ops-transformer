# PR #10552 代码检视报告

## PR 概述

| 项目 | 内容 |
|------|------|
| 标题 | 恢复A5 FullMeshV2 原分核公式 |
| 作者 | zhong-zixin |
| 分支 | `no_all_core_cnt` → `master` |
| 变更量 | 2 files changed, +23 / -52 |
| 类型 | 性能回退（撤销优化 commit `2dcd311ea`） |

### 变更意图

撤销 commit `2dcd311ea`（"A5 FullMeshV2全核分摊专家计数并减少CumSum控制核"）引入的 `useAllCoreCount_` 小 BS 优化路径。该优化在 `axisBS < 128` 时让全部 AIV 核参与专家计数，但在某些多 P 场景下导致性能劣化。本 PR 恢复为仅由 CumSum 核（最多 8 核）负责专家计数的原始策略。

### 涉及文件

1. `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp` (+11 / -11)
2. `mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h` (+12 / -41)

---

## 变更分析

### 1. Kernel 侧 (`moe_distribute_dispatch_v2_a5_full_mesh.h`)

**移除的内容：**
- `useAllCoreCount_` 成员变量及 `ALL_CORE_COUNT_BS_THRESHOLD`、`CUMSUM_STATUS_PER_CORE` 常量
- `CalAndSendCnt()` 分发函数（内联到 `CalCumSum()` 中）
- 发送路径中 `if (useAllCoreCount_)` 块（初始化 `statusBuf_` 并调用计数）

**恢复的内容：**
- `aivUsedCumSum_ = totalExpertNum_ / 16` 始终使用该公式（不再分小 BS / 大 BS 两条路径）
- `CalAndSendCntByRankAndExpert` 和 `SendExpertCntRange` 恢复为**绝对索引** `expertId * UB_ALIGN_DATA_COUNT`（与原始 `CalAndSendCntByRank` 一致）
- `CalAndSendCntByExp` 保持**相对索引** `(curExpertId - startExpertId) * 8`（与原始一致）

**验证结论：索引变更正确。** 通过对比 commit `2dcd311ea` 之前的原始代码确认：
- 原始 `CalAndSendCntByExp` 使用相对索引 → PR 保持相对索引 ✓
- 原始 `CalAndSendCntByRank` 使用绝对索引 → PR 的 `CalAndSendCntByRankAndExpert` + `SendExpertCntRange` 恢复为绝对索引 ✓
- 两个函数的写/读索引各自内部一致，无越界风险
- `statusBuf_` 在 `CalCumSum()`（line 1201）中初始化后再使用，无 use-before-init

### 2. Tiling 侧 (`moe_distribute_dispatch_v2_tiling.cpp`)

**变更：**
- `aivUsedCumSum` 从硬编码 `A5_CUMSUM_MAX_CORE_NUM`（8）改为完整公式，与 kernel 侧对齐
- 新增 `moeExpertRankNum`、`moeExpertNumPerRank`、`rscvStatusNum` 计算
- 移除 `bs < A5_ALL_CORE_COUNT_BS_THRESHOLD` 时的 `statusBuf_` UB 预估块（发送路径不再使用 `statusBuf_`）

**验证结论：tiling 与 kernel 公式一致。** 新增的 clamp 逻辑（`A5_CUMSUM_MAX_CORE_NUM`、`rscvStatusNum`）与 kernel 侧完全对应，UB 预估与实际 kernel 行为一致。

---

## 未发现问题的方面

| 检查项 | 结论 |
|--------|------|
| 索引变更（相对→绝对） | 正确，与原始 `CalAndSendCntByRank` 一致，写/读位置匹配 |
| `statusBuf_` 初始化时序 | 正确，在 `CalCumSum()` 中先 `InitBuffer` 再使用 |
| Buffer 大小与索引匹配 | `statusCntAlign_ = Ceil(totalExpertNum_, 8) * 8` 足以容纳绝对索引 |
| 发送路径 `statusBuf_` 移除 | 正确，发送核不再做专家计数，UB 预估同步移除 |
| Tiling-kernel 公式一致性 | `aivUsedCumSum` 计算、clamp 逻辑均一致 |
| `SplitToCore` 的 `isFront` 参数 | 始终传 `false`，正确对应"仅 CumSum 核计数" |
| `CalExpertCnt` 逻辑 | 与原始内联逻辑完全一致，仅做了函数提取 |
| `moeExpertNum / moeExpertRankNum` 除法 | 安全，API 文档约束 `sharedExpertRankNum ∈ [0, epWorldSize)`，`moeExpertRankNum` 恒 ≥ 1 |

---

## 总结

本 PR 是对 commit `2dcd311ea` 的干净回退，正确恢复了原始的分核策略和索引方式。核心变更（索引从相对恢复为绝对、移除全核计数路径）均与 commit `2dcd311ea` 之前的原始代码一致，无功能性 Bug。

**判定：可通过，直接合并。**
