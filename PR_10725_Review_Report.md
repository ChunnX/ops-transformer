# PR #10725 检视报告（第二轮 — 全面复检）

> 更新时间：2026-08-27  
> 检视范围：全部 4 个 commit（含最新 commit `30044e3c8`），完整 diff + 完整文件上下文 + 流水线同步审计

## PR 信息

- **标题**: [WIP]combine收发并行
- **作者**: zhong-zixin
- **分支**: `comb_par_wave` -> `master`
- **修改文件**: 2 个
  - `mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp` (+170 行变动)
  - `mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h` (+854 行变动)
- **Commit 数**: 4
  - `5a3481991` Optimize combine send and receive parallelism
  - `3247a1855` Optimize combine processing by expert slot order
  - `50d360ab3` Optimize combine receive token scheduling by core ratio
  - `30044e3c8` Refactor combine receive pipeline helpers **（本轮新增）**

---

## 变更概述

### 前 3 个 commit（第一轮已检视）

1. **核间切分**：AIV 分为发送核（`sendAivNum_`）和接收核（`recvAivNum_`），接收核提前开始接收处理
2. **双缓冲处理**：`LocalWindowCopy` 中两个 token 之间轮转处理，`sumFloatBuf_` 扩容为 2 倍
3. **Slot 批处理**：`WaitDispatch` 返回已到达 slot 数量，支持部分到达时处理
4. **队列合并**：移除 `moeSumQueue_` 和 `moeCopySumQueue_`，统一使用 `moeMainSumQueue_`
5. **Tiling 重构**：`UbUsedCal` 更精确地计算 UB 用量

### 新增 commit `30044e3c8`（本轮重点检视）

该 commit 为**纯重构**，不引入新功能，核心变化：

**Tiling 侧**：
- 拆分 `UbUsedCal` 为 4 个聚焦函数：`GetFloatAlignSize`、`GetComputeBufferSize`、`GetCommBufferSize`、`GetReceiveBufferSize`
- **修复 `sizeof(DTYPE_SIZE_HALF)` bug**：旧代码 `sizeof(DTYPE_SIZE_HALF)` = `sizeof(uint32_t)` = 4，新代码直接用 `DTYPE_SIZE_HALF` = 2。旧代码高估部分 buffer 尺寸 2 倍

**Kernel 侧**：
- 拆分 `LocalWindowCopy` → `LocalWindowInit` + `TokenInit` + `ProcessToken` + 薄 `LocalWindowCopy`，局部变量提升为成员变量
- 提取 `GetMoeExpertSlotCount` 和 `ProcessSpecialExpert` 从 `ProcessMoeExpertsLoop` 中
- 拆分 `ProcessConstantExpert` → `ProcessConstantExpert`（alpha 计算）+ `AddConstantExpert`（v 向量计算）
- 移除 `CalcDispatchBufferNum`，逻辑内联到 `DispatchBufferInit`
- `ClearPackedTokenFlags` 签名变更：移除 `flagTensor` 参数
- **添加显式 `SyncFunc<MTE3_V>`**：在 `TokenInit` 和 `ProcessToken` 中，当 `outputCopyPending_` 为 true 时显式同步

---

## 第一轮问题追踪

| # | 问题 | 原严重度 | 当前状态 | 说明 |
|---|------|---------|---------|------|
| 1 | Pipeline 同步依赖 `SyncFunc<MTE3_S>` 副作用 | 中 | **已修复** | `TokenInit`/`ProcessToken` 中显式添加 `SyncFunc<MTE3_V>`，不再依赖排空副作用 |
| 2 | `CheckPackedTokenArrive` slot 按序到达约束 | 低 | 仍存在 | 逻辑未变，`CheckPackedFlagRangeArrive` 仍从位置 0 统计连续到达 flag |
| 3 | `ClearPackedTokenFlags` `isInputExpertMaskFlag_` 映射变更 | 低 | **确认正确** | `tokenIndex` 已是实际 token 索引，窗口布局按实际索引存储，无需映射 |
| 4 | `ExpertScaleCopy` 稀疏场景多余读取 | 低 | 仍存在 | strided DataCopyPad 逻辑未变 |
| 5 | Tiling `bufferNum` 极端场景 | 低 | 仍存在（改进） | `DTYPE_SIZE_HALF` 修复使 `totalBufferSize` 估计更精确，但极端场景未增加 `* 1` 检查 |

### 问题 1 修复详情（已解决）

`TokenInit()` (kernel:1666-1670) 和 `ProcessToken()` (kernel:1694-1697) 中：

```cpp
if (outputCopyPending_) {
    // 上一个token搬出完成后，V流水才能复用tokenBuf_
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    outputCopyPending_ = false;
}
```

即使后续维护者移除 `SyncFunc<MTE3_S>`，`tokenBuf_` 上的 MTE3→V 竞争也有显式同步保护。第一轮报告的核心关注点已解决。

---

## 本轮新发现问题

### 新问题 1：`CheckPackedFlagRangeArrive` 缺少 `V_MTE2` 同步

**严重程度**: 中（预存问题，非本 PR 引入）

**位置**: `CheckPackedFlagRangeArrive()`, kernel:1054-1060

**描述**:

```cpp
Duplicate<float>(flagTensor, float(0), compareCount);  // V 写 → packedCheckFlagBuf_
PipeBarrier<PIPE_V>();                                  // 仅排序 V 内部
DataCopyPad(flagTensor, dataFlagGlobal, ...);            // MTE2 写 → 同一 buffer
```

`Duplicate`（V）和 `DataCopyPad`（MTE2）都写同一个 UB buffer `packedCheckFlagBuf_`。`PipeBarrier<PIPE_V>` 只保证 V 指令间有序，不保证 V 完成后 MTE2 才开始。缺少 `V_MTE2` 同步。

**风险**: 若并发执行，部分区域可能残留零值而非 GM 数据，导致 `arriveFlagNum` 计算偏差，`WaitDispatch` 可能返回错误的 slot 数量，使未就绪的 slot 被当作已到达处理。

**属性**: 预存问题。本 PR 仅将返回值从 `bool` 改为 `uint32_t`，未修改同步逻辑。

---

### 新问题 2：`AlltoAllBuffInitAndMaskCal` 中 `V_S` 应为 `MTE2_S`

**严重程度**: 中（预存问题，非本 PR 引入）

**位置**: `AlltoAllBuffInitAndMaskCal()`, kernel:844-845

**描述**:

```cpp
DataCopyPad(expertMaskTensor_, xActiveMaskGM_, ...);   // MTE2 写 → expertMaskBuf_
SyncFunc<AscendC::HardEvent::V_S>();                    // S 等待 V — 但需要的是 MTE2_S
```

`DataCopyPad` 是 MTE2 操作写入 `expertMaskBuf_`，后续 `TokenInit` 中的 `expertMaskTensor_.GetValue()` 是 S 读取同一 buffer。`V_S` 让 S 等待 V 流水线排空，但不保证 MTE2 的 DataCopyPad 已完成。

**补充**: 此处 `V_S` 可能是为了等待 `ExpertMaskCalCnt()` 中的 V 操作（Cast/Sum/GatherMask）完成。但 `ExpertMaskCalCnt` 的 V 操作作用于 `tokenBuf_`/`rowTmpFloatBuf_`/`sumFloatBuf_`，不作用于 `expertMaskBuf_`。对 `expertMaskBuf_` 而言，需要的同步是 `MTE2_S`。若要同时等待 V 和 MTE2，需要两个 SyncFunc 或先 `MTE2_V` 再 `V_S`。

**风险**: 若 MTE2 未完成时 S 读取 `expertMaskTensor_`，可能读到未初始化或部分写入的数据，导致 `validToken` 判断错误。

**属性**: 预存问题。

---

### 新问题 3：`ProcessToken` 中 `MTE3_V` 而非 `MTE3_MTE2`

**严重程度**: 低（预存模式，本 PR 沿用）

**位置**: `ProcessToken()` kernel:1694-1697; `TokenInit()` kernel:1666-1669

**描述**:

前一个 `ProcessToken` 中 MTE3 从 `tokenBuf_` 读取写回 GM，随后 `ProcessExpert` 中 MTE2 向 `tokenBuf_` 写入新数据：

```cpp
// 前一轮: DataCopyPad(GM, sumBufLocal) → MTE3 读 tokenBuf_
// 当前轮: SyncFunc<MTE3_V>() → 随后 MTE2 写 tokenBuf_
```

严格来说，MTE3 读后 MTE2 写同一 buffer 应使用 `MTE3_MTE2` 同步。当前使用 `MTE3_V`，虽然 `SyncFunc<MTE3_V>` 会完全排空 MTE3 pipeline（副作用使 buffer 对 MTE2 也安全），但从编程模型角度不够严谨。

**风险**: 在当前硬件实现上不会出问题（`SyncFunc` 排空整条流水线），但依赖实现细节而非编程模型保证。

---

### 新问题 4：`ProcessConstantExpert` 中 `PipeBarrier<PIPE_ALL>` 过度同步

**严重程度**: 低-中（预存问题）

**位置**: `ProcessConstantExpert()`, kernel:1259

**描述**:

```cpp
PipeBarrier<PIPE_ALL>();  // 全流水线屏障
```

此时 `tokenBuf_` 上的 MTE3 操作已由 `ProcessToken` 中的 `outputCopyPending_` → `MTE3_V` 同步保证完成。`PIPE_ALL` 会排空所有流水线（包括不需要排空的 MTE2/MTE3），影响性能。

**替代方案**: 可移除此 PipeBarrier，依赖已有的 `MTE3_V` 同步。若需额外保护 `tokenBuf_` 的 MTE2 写入，可用更轻量的 `SyncFunc<MTE2_V>`。

**影响**: 该函数调用频率不高（仅常量专家场景），性能影响有限。

---

### 新问题 5：`SetWaitTpStatusAndDisPatch` 中 `MTE3_S` 与后续 `PIPE_ALL` 冗余

**严重程度**: 低（预存问题）

**位置**: `SetWaitTpStatusAndDisPatch()` kernel:901; `Process()` kernel:1780

**描述**:

```cpp
// SetWaitTpStatusAndDisPatch:
ExpertAlltoAllDispatchCopyAdd();
SyncFunc<AscendC::HardEvent::MTE3_S>();  // S 等待 MTE3

// Process:
SetWaitTpStatusAndDisPatch();
PipeBarrier<PIPE_ALL>();  // 紧接着的全屏障，使上面的 MTE3_S 冗余
```

`PipeBarrier<PIPE_ALL>` 会排空所有流水线（包括 MTE3），使前一个 `MTE3_S` 完全冗余。

---

### 新问题 6：`ExpertAlltoAllDispatchCopyAdd` 首轮 `S_MTE2` 为空操作

**严重程度**: 低（预存问题）

**位置**: `ExpertAlltoAllDispatchCopyAdd()`, kernel:974

**描述**:

```cpp
for (uint32_t tokenLoopId = 0U; tokenLoopId < tokenLoopNum; tokenLoopId++) {
    SyncFunc<AscendC::HardEvent::S_MTE2>();  // 首轮无前置 S 读取，此同步为空操作
    DataCopyPad(expandIdxLocal, ...);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // ...
}
```

首轮循环中无前置 S 读取操作，`S_MTE2` 同步为空操作。仅从第二轮起才有意义。

---

### 新问题 7：`DispatchBufferInit` 中 `fixedEndBuf` 浪费 32B UB

**严重程度**: 极低（资源浪费）

**位置**: `DispatchBufferInit()`, kernel:650-660 区域

**描述**:

```cpp
TBuf<> fixedEndBuf;
tpipe_->InitBuffer(fixedEndBuf, UB_ALIGN);
uint64_t fixedEndUbAddr = fixedEndBuf.Get<uint8_t>().GetPhyAddr();
```

`fixedEndBuf` 仅用于探测 UB 地址以计算 `fixedUbSize`，之后不再使用。其 32B 空间被永久占用。`fixedUbSize` 计算已包含此空间，后续 `remainUbSize` 计算正确，无功能影响。

---

## 流水线同步审计总览

| 函数 | 同步类型 | 作用 | 状态 |
|------|---------|------|------|
| `TokenMaskCalCnt` | MTE2_V, V_S | DataCopyPad→Cast→Sum→S | 正确 |
| `ExpertMaskCalCnt` | MTE2_V | DataCopyPad→Cast/Sum/GatherMask | 正确 |
| `MaskAlign` | V_MTE3, MTE3_MTE2, MTE2_V, MTE2_S | V→MTE3(GM)→MTE2(GM)→V/S | 正确 |
| `MaskSpecialExpert` | MTE2_V, MTE2_S, V_S | MTE2→V→S | 正确 |
| `SetWaitTpStatusAndDisPatch` | MTE3_S | dispatch 后 S 等待 MTE3 | 冗余（见新问题 5） |
| `ExpertAlltoAllDispatchCopyAdd` | MTE2_S, S_MTE2 | 批次循环同步 | 首轮空操作（见新问题 6） |
| `ExpertAlltoAllDispatchInnerCopyAdd` | V_MTE2 | Duplicate→DataCopyPad 同 buffer | 正确 |
| `CheckPackedFlagRangeArrive` | PipeBarrier_V, MTE2_V, V_S | Duplicate→DataCopyPad→Compare→S | **缺少 V_MTE2**（见新问题 1） |
| `CalConstExpertAlpha` | EnQue/DeQue, V_S | MTE2→V→S | 正确 |
| `ProcessConstantExpert` | PIPE_ALL, MTE2_V, S_V, V_S | 全屏障+多级同步 | **过度同步**（见新问题 4） |
| `AddConstantExpert` | EnQue/DeQue, S_V | MTE2→V→S→V | 正确 |
| `ProcessCopyExpert` | EnQue/DeQue | MTE2→V | 正确 |
| `ProcessMoeExpert` | EnQue/DeQue | MTE2→V | 正确 |
| `ExpertScaleCopy` | MTE2_S | MTE2→S(GetValue) | 正确 |
| `AddRmsNormAddCompute` | V_MTE2, MTE2_V | V→MTE2→V on tokenBuf_ | 正确 |
| `AddRmsNormRmsNormCompute` | V_MTE3, V_S, S_V | V→MTE3(GM), V→S→V→MTE3(GM) | 正确 |
| `AddSharedExpertX` | V_MTE2, MTE2_V | V→MTE2→V on tokenBuf_ | 正确 |
| `LocalWindowInit` | MTE2_V, V_S | gamma MTE2→V; flag Duplicate→S | 正确 |
| `TokenInit` | MTE3_V, MTE3_S, S_MTE3 | outputCopyPending_ 保护; statePos GM 写 | **已修复**（第一轮问题 1） |
| `ProcessToken` | MTE3_V, V_MTE3, MTE3_V | outputCopyPending_ 保护; Cast→DataCopyPad; RmsNorm | 基本正确（见新问题 3） |
| `LocalWindowCopy` | V_MTE3 | performanceInfo V→MTE3(GM) | 正确 |

---

## 非问题确认

以下变更经分析确认正确，无需关注：

1. **`DTYPE_SIZE_HALF` 修复**：`sizeof(DTYPE_SIZE_HALF)`（=4）→ `DTYPE_SIZE_HALF`（=2），修复了旧代码高估 buffer 尺寸的 bug
2. **`ProcessMoeExpertsLoop` 重构**：`enableSpecialExpert_` 为 false 时所有 expert 走 Moe 路径并 `continue`，非 Moe 专家分支不可达，逻辑正确
3. **Slot 批处理与部分到达**：`ProcessMoeExpertSlots` 返回已处理 slot 数，`topkId` 和 `index` 按引用传递，支持断点续处理
4. **`ProcessSharedExpertsLoop` 续处理**：`topkId` 从 `axisK_` 开始，`while (topkId < axisK_)` 立即退出后进入 shared expert 处理
5. **`sumFloatBufOffset` 对齐**：`hFloatAlign32Size_ / sizeof(float)` 为 float 元素数，与 `sumFloatBuf_` 的 `hFloatAlign32Size_ * 2U` 分配一致
6. **`CheckPackedTokenArrive` 整除截断**：`arriveFlagNum / flagCountPerSlot` 向下取整，保守报告已到达 slot 数，正确
7. **`ExpertAlltoAllDispatchBatchCopyAdd` 批次过滤**：批次范围 `[batchBegin, batchEnd)` 互斥覆盖 0 到 `axisBS_`，所有 token 最终都会被发送
8. **`permStride` 互质性**：与旧代码逻辑一致，保证遍历所有索引
9. **接收核跳过 `BuffInit`**：`tpipe_` 为每核独立实例，初始状态干净
10. **`LocalWindowCopy` 拆分**：拆分为 `LocalWindowInit` + `TokenInit` + `ProcessToken` 后，局部变量提升为成员变量，逻辑等价
11. **`ProcessConstantExpert` 拆分**：拆分为 alpha 计算 + v 向量计算两步，逻辑等价
12. **`ClearPackedTokenFlags` 签名变更**：移除 `flagTensor` 参数，内部从 `packedClearFlagBuf_` 获取，行为不变

---

## 总结

### 改进点

- **第一轮核心问题（Pipeline 同步）已修复**：新增 commit 在 `TokenInit`/`ProcessToken` 中显式添加 `SyncFunc<MTE3_V>`，不再依赖 `SyncFunc<MTE3_S>` 排空副作用
- **修复隐藏 bug**：`sizeof(DTYPE_SIZE_HALF)` 高估 buffer 尺寸 2 倍的问题已修复
- **代码质量提升**：大函数拆分为小函数，可读性和可维护性显著改善

### 遗留问题

- 2 个中等预存同步问题（`CheckPackedFlagRangeArrive` 缺少 `V_MTE2`、`AlltoAllBuffInitAndMaskCal` `V_S` 应为 `MTE2_S`）——非本 PR 引入，但影响正确性健壮性
- 3 个低严重度预存过度同步——性能影响有限
- 3 个第一轮遗留低严重度问题（slot 按序到达、ExpertScaleCopy 多余读取、bufferNum 极端场景）——未恶化也未修复

### 判定

**PASS WITH NOTES**

核心逻辑正确，新 commit 的重构没有引入新 bug，且修复了第一轮发现的同步问题和隐藏的 `sizeof` bug。遗留的同步问题均为预存问题（非本 PR 引入），建议后续单独处理。

**评分**: 88/100

| 维度 | 得分 | 说明 |
|------|------|------|
| 功能正确性 | 28/30 | 核心逻辑正确，2 个预存同步问题扣分 |
| 代码质量 | 25/25 | 重构后可读性显著提升，函数职责清晰 |
| 同步设计 | 20/25 | 第一轮问题已修复，但仍有预存同步缺陷 |
| Tiling 正确性 | 10/10 | DTYPE_SIZE_HALF bug 修复，计算更精确 |
| 性能 | 5/10 | 过度同步问题存在，但影响有限 |
