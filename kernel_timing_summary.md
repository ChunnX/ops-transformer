# Kernel 内部打点计时功能总结

## 概述

`KernelTiming` 是一个通用的 Ascend C Kernel 内部细粒度计时工具，适用于任何需要在核内对各执行阶段进行耗时分析的算子。

它提供 16 个独立计时 slot，支持**顺序自增**和**显式指定**两种 slot 寻址方式。Kernel 侧记录每个 slot 的**原始 `GetSystemCycle()` 起止时间戳**，由 Python 侧计算 duration 并进行可视化。支持纯 AIV 和 AIC+AIV 混合执行模式。

## 架构分层

```
┌──────────────────────────────────────────────────────────────┐
│  Python 侧 (extract_kernel_timing)                           │
│  - 从输出 tensor 头部提取 timing 数据                         │
│  - 输出: raw CSV + duration CSV + 热力图 + 堆叠条形图         │
├──────────────────────────────────────────────────────────────┤
│  算子 Kernel（任意算子）                                      │
│  - Process() 中各阶段插入 StartTiming / StopTiming            │
│  - WriteTimingToOutput() 将原始时间戳写入输出 tensor 头部     │
├──────────────────────────────────────────────────────────────┤
│  通用计时工具类 (mc2/common/utils/mc2_kernel_timing.h)        │
│  - KernelTiming: 16 slot，记录原始 start/stop 时间戳          │
│  - 基于 AscendC::GetSystemCycle() 硬件周期计数                │
└──────────────────────────────────────────────────────────────┘
```

## Kernel 侧：计时工具类

**文件**: `mc2/common/utils/mc2_kernel_timing.h`
**命名空间**: `Mc2Kernel`

### 核心特性

| 特性 | 说明 |
|------|------|
| Slot 数量 | 16 个（`MAX_SLOTS = 16`），ID 范围 `[0, 15]` |
| 计时单位 | 硬件 cycle（`AscendC::GetSystemCycle()` 原始值） |
| 记录方式 | 每个 slot 记录 start 和 stop 两个原始时间戳，不做差值计算 |
| Slot 寻址 | 显式指定 `slotId` 或默认自增（`slotId = -1`） |
| 核类型支持 | 纯 AIV 模式（默认）+ AIC+AIV 混合模式（`isAicActivated = true`） |
| 跨核同步 | `WriteTimingToOutput` 内置 `PipeBarrier<PIPE_ALL>` + `SyncAll` |
| 输出方式 | 通过 `GlobalTensor<uint64_t>::SetValue` 直接写 GM，每个核写 32 个 `uint64_t`（16 slots × 2 timestamps） |

### 数据结构

```cpp
class KernelTiming {
private:
    uint64_t startCycles_[MAX_SLOTS];    // 每个 slot 的 Start 原始时间戳
    uint64_t stopCycles_[MAX_SLOTS];     // 每个 slot 的 Stop 原始时间戳
    int32_t  currentId_;                 // 当前活跃的 slot ID
    GM_ADDR  gmOutputAddr_;             // 输出 GM 地址
};
```

### 内存布局

每个核写入 32 个 `uint64_t`（16 slots × 2 timestamps），整体在 GM 中的布局为 `(num_cores, 16, 2)`：

```
core_0: [s0, e0, s1, e1, ..., s15, e15]
core_1: [s0, e0, s1, e1, ..., s15, e15]
  ...
core_N: [s0, e0, s1, e1, ..., s15, e15]
```

其中 `s` = start timestamp，`e` = stop timestamp。每个核占 `32 × 8 = 256` 字节。

### API 详细说明

#### `Init(GM_ADDR outputAddr)`

初始化计时器，清零所有 slot 的 start/stop 时间戳，并记录输出 GM 地址。

- **参数**: `outputAddr` — 输出 tensor 的 GM 地址。计时数据将从该地址的 offset 0 开始写入。
- **调用时机**: 在 Kernel 的 `Init` 函数中调用一次。

#### `StartTiming(int32_t slotId = -1)`

记录指定 slot 的 `GetSystemCycle()` 原始起始时间戳。

- **`slotId >= 0`**: 显式指定 slot ID，`currentId_` 切换到该 slot。
- **`slotId = -1`（默认）**: `currentId_` 在当前值基础上 **自增 1**，然后对新 slot 记录起始时间戳。初始值为 0，因此首次调用 `StartTiming()` 时实际计时 slot 1。若需要首个 slot 为 0，应显式传入 `StartTiming(0)`。
- **越界保护**: 若 `currentId_` 超出 `[0, MAX_SLOTS)` 范围，直接返回不做操作。

#### `StopTiming(int32_t slotId = -1)`

记录指定 slot 的 `GetSystemCycle()` 原始结束时间戳。

- **`slotId >= 0`**: 显式指定要停止的 slot ID。
- **`slotId = -1`（默认）**: 停止 `currentId_` 当前指向的 slot。
- **记录行为**: 直接将当前 `GetSystemCycle()` 值写入 `stopCycles_[slotId]`，不做差值计算，duration 由 Python 侧计算。

#### `GetCurrentSlotId() const`

返回当前活跃的 slot ID，用于调试或动态判断当前计时位置。

#### `WriteTimingToOutput(int32_t coreId, bool isAicActivated = false, int32_t aicNum = 0)`

将所有 slot 的 start/stop 原始时间戳写入 GM。

- **`coreId`**: 当前核的 ID，决定写入 GM 的偏移位置（`coreId * 32`）。
- **`isAicActivated`**: 是否启用了 AIC 核。
  - `false`（默认）: 纯 AIV 模式，`SyncAll<true>()` 同步所有 AIV 核。
  - `true`: AIC+AIV 混合模式，`SyncAll<false>()` 同步所有核，且 AIV 核的 `coreId` 会自动加上 `aicNum` 偏移。
- **`aicNum`**: AIC 核的数量，仅在 `isAicActivated = true` 时有效。
- **写入格式**: 每个 slot 写入 2 个 `uint64_t`，依次为 start 和 stop 时间戳。

### Slot 寻址模式

#### 模式一：显式指定 Slot ID

适用于需要精确控制每个阶段对应哪个 slot 的场景，slot 可以按任意顺序打点：

```cpp
kernelTiming_.StartTiming(0);
PhaseA();
kernelTiming_.StopTiming(0);

kernelTiming_.StartTiming(5);
PhaseB();
kernelTiming_.StopTiming(5);

kernelTiming_.StartTiming(2);
PhaseC();
kernelTiming_.StopTiming(2);
```

#### 模式二：默认自增 Slot ID

适用于顺序打点、不需要关心具体 slot 编号的场景。首次调用需显式指定起始 ID，后续自动递增：

```cpp
kernelTiming_.StartTiming(0);   // slot 0
PhaseA();
kernelTiming_.StopTiming();

kernelTiming_.StartTiming();    // 自增到 slot 1
PhaseB();
kernelTiming_.StopTiming();

kernelTiming_.StartTiming();    // 自增到 slot 2
PhaseC();
kernelTiming_.StopTiming();
```

### AIC+AIV 混合模式

当算子同时使用 AIC 和 AIV 核时，两类核各自独立计时并写入 GM 的不同区域：

```cpp
// AIC 核：coreId 直接使用原始值（如 0, 1, 2...）
// AIV 核：coreId 自动加上 aicNum 偏移（如 aicNum=8，则 AIV core 0 实际写入 core 8 的位置）
kernelTiming_.WriteTimingToOutput(coreId, /*isAicActivated=*/true, /*aicNum=*/8);
```

GM 布局变为：

```
core_0 (AIC): [s0, e0, s1, e1, ..., s15, e15]
core_1 (AIC): [s0, e0, s1, e1, ..., s15, e15]
...
core_7 (AIC): [s0, e0, s1, e1, ..., s15, e15]
core_8 (AIV): [s0, e0, s1, e1, ..., s15, e15]   ← AIV core 0 写入此处
...
```

## Python 侧：数据提取与可视化

**文件**: `dispatch_combine_v2_test.py` — `extract_kernel_timing()`

### 函数签名

```python
def extract_kernel_timing(output_tensor, num_cores, rank, out_dir=None):
```

| 参数 | 说明 |
|------|------|
| `output_tensor` | 包含 timing 数据的输出 tensor（timing 数据位于 offset 0） |
| `num_cores` | 参与计时的核数 |
| `rank` | 当前 rank ID，用于文件命名 |
| `out_dir` | 输出目录，默认当前目录 |

### 数据提取

```python
timing_bytes = num_cores * 16 * 2 * 8
timing = output_tensor.contiguous().view(torch.uint8).reshape(-1)[:timing_bytes] \
         .cpu().view(torch.int64).reshape(num_cores, 16, 2)
```

从 tensor 头部按 `uint8` 展平后截取前 `timing_bytes` 字节，再按 `int64` 重塑为 `(num_cores, 16, 2)` 矩阵。输入 tensor 可以是任意 dtype（如 bf16、fp16 等），无需提前转换。

### Duration 计算

```python
cycles_to_us = 50   # NPU 时钟频率 50MHz，1 cycle = 1/50 us
duration = stop - start
duration_us = duration / cycles_to_us
```

### 输出产物

| 文件 | 说明 |
|------|------|
| `timing_rank_{rank}_raw.csv` | 原始数据：每行一个 core，每 slot 两列（`slot_N_start`, `slot_N_stop`），值为原始 `GetSystemCycle()` 时间戳（uint64） |
| `timing_rank_{rank}_duration.csv` | 耗时数据：每行一个 core，每 slot 一列（`slot_N`），值为 us（保留 2 位小数） |
| `timing_rank_{rank}_heatmap.png` | 热力图：横轴为 active slot，纵轴为 core，YlOrRd 色图，颜色深浅表示 duration (us) |
| `timing_rank_{rank}_stack.png` | 水平堆叠条形图：每个 core 一行，不同颜色代表不同 slot 的 duration 占比，段内标注 us 值 |

### 可视化逻辑

1. **Duration 计算**: `duration_us = (stop - start) / cycles_to_us`
2. **Active Slot 筛选**: `np.any(duration_us > 0, axis=0)` — 只展示至少有一个核有非零 duration 的 slot
3. **热力图**: `imshow` 渲染 `duration_us[:, active_slots]`，自适应 figsize
4. **堆叠条形图**: 每个 active slot 作为一层水平堆叠，宽度 > 总 duration 1% 的段内标注 us 值（2 位小数），图例显示 slot 名称和颜色

## 接入新算子指南

`KernelTiming` 是通用工具类，可接入任何 Ascend C 算子，步骤如下：

### 1. 引入头文件

```cpp
#include "mc2_kernel_timing.h"  // 根据实际相对路径调整
using namespace Mc2Kernel;
```

### 2. 定义该算子的 Timing Slot 枚举（可选，也可直接用数字）

```cpp
enum MyOpTimingSlot {
    TIMING_PHASE_A = 0,
    TIMING_PHASE_B = 1,
    TIMING_PHASE_C = 2,
    // ... 最多 16 个
};
```

### 3. 在 Kernel 类中添加成员

```cpp
class MyOpKernel {
private:
    Mc2Kernel::KernelTiming kernelTiming_;
};
```

### 4. 在 Init 中初始化

```cpp
kernelTiming_.Init(outputTensor);  // 复用某个输出 tensor 的 GM 地址
```

### 5. 在 Process 中插入打点

```cpp
// 显式指定 slot ID
kernelTiming_.StartTiming(TIMING_PHASE_A);
PhaseA();
kernelTiming_.StopTiming(TIMING_PHASE_A);

// 自增模式
kernelTiming_.StartTiming();  // 自动递增到下一个 slot
PhaseB();
kernelTiming_.StopTiming();

// 写出（纯 AIV 模式）
kernelTiming_.WriteTimingToOutput(coreId);

// 写出（AIC+AIV 混合模式）
kernelTiming_.WriteTimingToOutput(coreId, true, aicCoreNum);
```

### 6. Python 侧提取

```python
extract_kernel_timing(output_tensor, num_cores, rank, out_dir="timing_output")
```

Python 侧的 `extract_kernel_timing` 函数与具体算子无关，只要 timing 数据写在输出 tensor 头部即可直接使用。函数内部会先将 tensor 按 `uint8` 展平再重新解释为 `int32`，因此输入 tensor 可以是任意 dtype（如 bf16、fp16 等），无需提前转换。
