# arch35 CrossCore flag ID 溢出根因分析

## 1. 现象

在 arch35 (Ascend950/DAV_3510, `KERNEL_TYPE_MIX_AIC_1_2`) ESL 仿真平台上运行 CGDR 算子时，报错：

```
set_cross_core flag ID overflows!
```

该错误来自仿真器 `libpem_davinci.so` 中的 `PemIssQue::release_set_cross_core` 函数。

## 2. 根因分析

### 2.1 CrossCoreWaitFlag 的默认模板参数

`CrossCoreWaitFlag` 的 API 声明（`kernel_operator_block_sync_intf.h:85-86`）：

```cpp
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

CGDR 代码中所有 `CrossCoreWaitFlag(flagId)` 调用**未指定模板参数**，因此默认为 `<0, PIPE_S>`。

而对应的 `CrossCoreSetFlag` 使用 `<0x2, PIPE_MTE3>` (AIV侧) 和 `<0x2, PIPE_FIX>` (AIC侧)。

### 2.2 arch35 的 IsSplitCubePipe / IsSplitVectorPipe 差异

`kernel_event.h:262-272` 中 arch35 的管道分类：

```cpp
// arch35 (3510)
IsSplitVectorPipe = { PIPE_S, PIPE_V, PIPE_MTE2, PIPE_MTE3 }
IsSplitCubePipe   = { PIPE_S, PIPE_MTE1, PIPE_MTE2, PIPE_FIX, PIPE_M }
```

对比 arch22 (c220)：

```cpp
// arch22 (2002)
IsSplitCubePipe   = { PIPE_S, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3, PIPE_M }
```

**关键差异**：arch35 的 `IsSplitCubePipe` **不包含 `PIPE_MTE3`**，而 arch22 包含。

### 2.3 NotifyEventImpl / WaitEventImpl 的两条执行路径

`kernel_operator_sync_impl.h:801-855`：

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void NotifyEventImpl(uint16_t flagId) {
    if constexpr (modeId == 4) {
        // 路径A: intra-block 同步
        // set_intra_block(pipe, flagId) — block内AIC↔AIV硬件flag
    } else {
        // 路径B: cross-core 同步
        // ffts_cross_core_sync(pipe, GetffstMsg(modeId, flagId))
    }
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitEventImpl(uint16_t flagId) {
    if constexpr (modeId == 4) {
        // 路径A: wait_intra_block(pipe, flagId)
    } else {
        // 路径B: wait_flag_dev(pipe, flagId)
    }
}
```

### 2.4 modeId=0x2 (cross-core) 在 1:2 模式下的问题

SDK 内部 `SyncAllImpl` (line 183-212) 根据 `__MIX_CORE_AIC_RATION__` 选择不同同步策略：

**1:1 模式** (`__MIX_CORE_AIC_RATION__ == 1`)：
- AIC↔AIV 用 `ffts_cross_core_sync` / `wait_flag_dev` (cross-core, modeId=0x2)

**1:2 模式** (else 分支)：
- AIC↔AIV 用 `set_intra_block` / `wait_intra_block` (intra-block, modeId=4)
- 仅 C→C 同步用 `ffts_cross_core_sync` (modeId=0x0)

CGDR 使用 `modeId=0x2` (cross-core) 进行 AIC↔AIV 同步，在 1:2 模式下**选择了错误的同步路径**。

### 2.5 SET 与 WAIT 不匹配导致 flag 累积

以 stage1.h 同步0 为例：

| 操作 | 模板参数 | 实际执行 | modeId | pipe |
|------|---------|---------|--------|------|
| AIV Set | `<0x2, PIPE_MTE3>(0x9)` | `ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x2, 0x9))` | 0x2 | PIPE_MTE3 |
| AIC Wait | `(0x9)` → 默认`<0, PIPE_S>` | `wait_flag_dev(PIPE_S, 0x9)` | 0x0 | PIPE_S |

**三重不匹配**：

1. **modeId 不匹配**：Set=0x2, Wait=0x0。`GetffstMsg` 编码 `msg = 0x1 + (mode<<4) + (flagId<<8)`，不同 mode 映射到不同 flag slot。
2. **pipe 不匹配**：Set=PIPE_MTE3, Wait=PIPE_S。仿真器 `HSetWaitEventTable` 按 `(pipe, flagId)` 组合索引 flag 条目（见 `pipe_dependency_ffts_analyzer.py:69-72` 的 `get_flag_index`）。
3. **同步路径不匹配**：Set 走 cross-core (`ffts_cross_core_sync`)，Wait 也走 cross-core (`wait_flag_dev`)，但在 1:2 模式下应走 intra-block (`set_intra_block`/`wait_intra_block`)。

结果：AIV 不断执行 Set，flag 被推入 `HSetWaitEventTable` 但**永远找不到匹配的 Wait**。每次循环迭代增加一个未消耗的 flag 条目，当累积超过 `SYNC_FLAG_ID_MAX=16` 时，仿真器在 `release_set_cross_core` 中报 `set_cross_core flag ID overflows!`。

### 2.6 仿真器 flag 生命周期

从 `libpem_davinci.so` 的符号表推断的 flag 管理流程：

```
SET 流程:
  PemIfu::decode_set_wait_flag  →  解码指令
  PemCcu::dispatch_Hset_flag    →  分发到 HSetWaitEventTable
  PemIssQue::push_set_flag_to_tab  →  推入 flag 等待表
  PemIssQue::release_set_cross_core  →  释放到目标核 [溢出检查点]
  PemCube/PemVec::recv_set_cross_core  →  目标核接收

WAIT 流程:
  PemCcu::dispatch_wait_flag    →  分发 wait
  PemCcu::exec_wait_flag_dev    →  执行 wait_flag_dev
  HSetWaitEventTable::wait_flag_success  →  匹配成功则消耗 SET
  PemIssQue::retire_set_flag    →  退役已消耗的 SET
```

当 SET 无法被 WAIT 匹配时，`HSetWaitEventTable` 中的条目持续累积，最终触发溢出。

### 2.7 flagId 0x5 溢出最快的原因

flagId 0x5 在两个 stage 的循环中均被 Set：
- stage1.h: `CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5)` — 在 `ParaChunkAIV` 的 `VBetaCompute` 循环中
- stage2.h: `CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5)` — 在 `ProcessAivFp32`/`ProcessAivBf16` 的 chunk 循环中

累积速度是其他 flagId 的 2 倍，因此最先达到溢出阈值。

## 3. 修复方案

### 3.1 正确的 arch35 CrossCore 用法（来自 GMM weight_quant 参考）

`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_basic_block.h:417-453`：

```cpp
// AIV → AIC: AIV Set 用 PIPE_MTE3, AIC Wait 用 PIPE_MTE1
CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);     // AIV Set
CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);    // AIC Wait

// AIC → AIV: AIC Set 用 PIPE_MTE1, AIV Wait 用 PIPE_MTE3
CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);     // AIC Set
CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);    // AIV Wait
```

其中 `SYNC_MODE4 = 4` (intra-block 同步)。

### 3.2 具体修改

| 修改项 | 修改前 | 修改后 | 原因 |
|--------|--------|--------|------|
| modeId | 0x2 | 0x4 | 1:2 模式应用 intra-block 同步 |
| AIC pipe | PIPE_FIX | PIPE_MTE1 | 匹配 GMM 参考 + arch35 IsSplitCubePipe |
| WaitFlag 模板参数 | 缺失(默认`<0,PIPE_S>`) | 显式指定 | 确保 Set/Wait modeId+pipe 配对 |

AIC 侧统一使用 `<0x4, PIPE_MTE1>`，AIV 侧统一使用 `<0x4, PIPE_MTE3>`。

### 3.3 V→S 同步修复（附加）

`stage2.h` 中 `CalGCumExpFp32` 和 `CalGCumExpBf16` 在 `Exp` 后直接调用 `lastGCum_.GetValue(0)`，但 `Exp` 在 V 管道执行，`GetValue` 在 S 管道执行，缺少 V→S 同步。在 `Exp` 后添加 `SetFlag/WaitFlag<V_S>` 修复此问题。

## 4. 验证方法

在 arch35 ESL 仿真平台上运行 CGDR 算子，确认 `set_cross_core flag ID overflows!` 错误不再出现。
