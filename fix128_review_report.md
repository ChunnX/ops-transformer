# fix128 分支检视报告

**分支**：`fix128`
**Commit**：`21bdaca5d` — `fix(mc2): A5 FullMeshV2全核计数阈值使用最大BS`
**作者**：zhong-zixin
**日期**：2026-08-22
**改动规模**：2 文件，+4/-3 行

## 改动概述

将 A5 FullMeshV2 dispatch 算子中"全核计数"阈值判断从使用本地 BS（`axisBS_`/`bs`）改为使用最大 BS（`axisMaxBS_`/`maxBs = globalBs / epWorldSize`），确保 kernel 侧与 tiling 侧的判断逻辑一致。

## 改动详情

### 1. Kernel 侧

**文件**：`mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h:396`

```diff
-    useAllCoreCount_ = axisBS_ < ALL_CORE_COUNT_BS_THRESHOLD;
+    useAllCoreCount_ = axisMaxBS_ < ALL_CORE_COUNT_BS_THRESHOLD;
```

- `axisBS_`（第331行）= `tilingData->moeDistributeDispatchV2Info.bs`，是本 rank 的 BS
- `axisMaxBS_`（第353行）= `globalBS_ / epWorldSizeOriginal_`，是全局 BS 均分到每 rank 的最大值

### 2. Tiling 侧

**文件**：`mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp:1609-1630`

```diff
-    uint32_t bs = tilingData.moeDistributeDispatchV2Info.bs;
+    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
+    uint32_t maxBs = tilingData.moeDistributeDispatchV2Info.globalBs / epWorldSize;
     ...
-    if (bs < A5_ALL_CORE_COUNT_BS_THRESHOLD) {
+    if (maxBs < A5_ALL_CORE_COUNT_BS_THRESHOLD) {
```

`SendToMoeExpertUsedBuffer` 函数中估算 UB 占用时，阈值判断同步改为 `maxBs`。

## 正确性分析

### 修复的问题

原代码用本地 `axisBS_`（per-rank BS）判断是否启用全核计数模式。但 `useAllCoreCount_` 影响的是 UB buffer 分配策略（第775-778行：全核模式额外分配 `statusBuf_`）和核数分配（第859-934行）。这些策略的 UB 占用量取决于 window 大小，而 window 大小由 `axisMaxBS_` 决定（第393行：`expertPerSizeOnWin_ = axisMaxBS_ * hCommuSize_`）。

当不同 rank 的 BS 不均匀时（elastic scaling 场景），本地 `axisBS_` 可能小于阈值但 `axisMaxBS_` 大于阈值，导致：
- kernel 侧某些 rank 启用全核计数（多分配 `statusBuf_`），某些 rank 不启用
- tiling 侧 UB 估算与 kernel 实际用量不一致，可能 UB 溢出或浪费

### 一致性验证

| 维度 | Kernel 侧 | Tiling 侧 | 一致？ |
|------|-----------|-----------|--------|
| 阈值常量 | `ALL_CORE_COUNT_BS_THRESHOLD = 128` | `A5_ALL_CORE_COUNT_BS_THRESHOLD = 128` | 是 |
| 比较值 | `axisMaxBS_ = globalBS_ / epWorldSizeOriginal_` | `maxBs = globalBs / epWorldSize` | 是（`epWorldSizeOriginal_` = `epWorldSize`） |
| 除零风险 | `epWorldSizeOriginal_` 来自 tiling，已校验 >= 2 | `epWorldSize` 来自 tiling，已校验 >= 2（第684行） | 无风险 |

### globalBs=0 的边界

当 `globalBs=0` 时（inferShape 第244行），`globalBs` 实际被计算为 `bs * epWorldSize`，此时 `maxBs = (bs * epWorldSize) / epWorldSize = bs`，与原行为一致。无回归风险。

## 检视结论

### 无问题

改动正确、简洁、一致。kernel 侧和 tiling 侧同步修改，阈值常量相同，比较值等价，无除零风险，`globalBs=0` 边界无回归。修复了 BS 不均匀场景下 kernel 与 tiling UB 估算不一致的问题。

### 建议（非阻塞）

- 建议在 commit message 中补充具体场景说明（如 elastic scaling 下 BS 不均匀导致的问题现象），便于后续追溯。
