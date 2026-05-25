# Flash Attention 标量瓶颈优化汇总

> 目标用例：`TND_MIXED_02`（B=8, N1=N2=4, D=128, layout=TND, mask_mode=3）  
> 优化范围：`c:\yf\master\ops-transformer_caculate\attention\flash_attn\op_kernel\arch35\`

---

## 背景：性能瓶颈来源

Flash Attention 内核的四个流水线阶段（C1/V1/C2/V2）本身计算量很小（小序列），
但 profiler 显示**相邻阶段之间存在大量等待时间**，根因是：

1. **AIV 标量开销**在每个任务的硬件派发前占用大量标量周期（`GetTaskDealMode`、`CalcParams`、`UpdateAxisInfo`）。
2. **MTE2 DMA 与 Cube 矩阵乘串行**：AttenMask 从 GM 搬运到 UB 的 DMA 排在 Mm1 结束之后，完全浪费了 Mm1 运行时的 MTE2 空闲时间。
3. **VF 函数内部冗余标量指令**：`LastDivNewVF` 每次内层循环都重新计算谓词寄存器，D=128 时产生大量无效标量指令。

---

## 修改一：`vf_flashupdate_new_local.h` — 引入本地 VF 文件并修复 `LastDivNewVF`

### 1.1 为什么需要本地文件

原始代码 include 的是公共路径的 `vf_flashupdate_new.h`，无法针对 flash_attn 单独修改。
为此在本算子目录下创建 `vf_flashupdate_new_local.h`，并在 `flash_attn_block_vec_noquant_gqa.h` 中改为引用本地版本：

```cpp
// 修改前
#include "../../common/op_kernel/arch35/vf_flashupdate_new.h"

// 修改后
#include "vf_flashupdate_new_local.h"   // 本地可修改的副本
```

### 1.2 `LastDivNewVF` 的问题

`LastDivNewVF` 是 V2 阶段的核心归一化函数，执行 `dst[i][j] = cur[i][j] / expSum[i]`。
D=128 时 `dLoops = 128/64 = 2`，m 最大为 `mBaseSize = 128`，即总调用次数 = 128×2 = **256 次内层循环**。

**修改前**（公共文件中的原始实现）：

```cpp
for (uint16_t i = 0; i < m; ++i) {
    LoadAlign<T, ...>(vreg_exp_sum, expSumUb + i * REDUCE_SIZE);
    for (uint16_t j = 0; j < dLoops; ++j) {
        // ▼ 每次内层循环都重新计算全1谓词寄存器 — 冗余标量指令
        MaskReg preg = CreateMask<float, MaskPattern::ALL>();
        LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
        Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg);
        Div(vreg_div, vreg_input_cur, vreg_exp_sum, preg);
        StoreAlign(dstUb + i * d + j * floatRepSize, vreg_div, preg);
    }
}
```

**修改后**（`vf_flashupdate_new_local.h`，第 405 行）：

```cpp
// ▼ 提到循环外，只创建一次 — 消除 (m × dLoops - 1) 次冗余标量操作
MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();

for (uint16_t i = 0; i < m; ++i) {
    LoadAlign<T, ...>(vreg_exp_sum, expSumUb + i * REDUCE_SIZE);
    for (uint16_t j = 0; j < dLoops; ++j) {
        LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
        Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
        Div(vreg_div, vreg_input_cur, vreg_exp_sum, preg_all);
        StoreAlign(dstUb + i * d + j * floatRepSize, vreg_div, preg_all);
    }
}
```

### 1.3 性能分析

| 项目 | 修改前 | 修改后 | 节省 |
|------|--------|--------|------|
| `CreateMask` 调用次数（D=128, m=128） | 256 次/V2调用 | 1 次/V2调用 | **255 次** |
| 每次 `CreateMask` 的标量指令数 | ~3–5条 | — | ~765–1275 条标量指令/V2 |
| 两个 subBlock 合计 | 512 次 | 2 次 | 510 次 |

> `CreateMask` 在 Ascend 310B/910B VF 核上需要执行 MOV/AND 等多条标量汇编指令构造谓词寄存器，
> 属于纯标量开销，**不在 VF 流水线上**，会直接阻塞 VF 指令发射。

---

## 修改二：`CalcCurS2StartEndWithSparse` — `gSize==1` 快速路径

### 2.1 问题

TND_MIXED_02 中 `N1==N2==4`，故 `gSize = N1/N2 = 1`。
原代码在 `UbFormat::S1G` 路径下无论 `gSize` 是否为 1 都执行整除：

```cpp
// 修改前：gSize=1 时 /1 仍然是一次整除指令
s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
s1LastToken  = static_cast<int64_t>(s1GLastToken  / constInfo.gSize);
```

### 2.2 修改后

```cpp
// 修改后（flash_attn_kernel_base_gqa_template.h，第 607 行）
if (constInfo.gSize == 1) {          // ← 编译期无法消除但运行期只比较一次
    s1FirstToken = s1GFirstToken;    // 直接赋值，无除法
    s1LastToken  = s1GLastToken;
} else {
    s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
    s1LastToken  = static_cast<int64_t>(s1GLastToken  / constInfo.gSize);
}
```

### 2.3 性能分析

`CalcCurS2StartEndWithSparse` 在每次换行（`bN2Cur` 或 `gS1Cur` 变化）时调用一次。
TND_MIXED_02 每核约 1–2 次调用，整除指令本身在标量流水线上约 **10–20 cycles**。
虽然绝对节省量不大，但消除了 hot-path 上的不必要除法。

---

## 修改三：`CalcParams` — 复用 `s2Idx` 消除重复乘法

### 3.1 问题

原代码在计算 `isFirstS2Loop` 和 `isLastS2Loop` 时对 `s2Cur * s2BaseSize` 重复计算：

```cpp
// 修改前（示意）
info.s2Idx = s2Cur * s2BaseSize;
// ...
bool isFirst = (s2Cur * s2BaseSize == 0);             // 重复乘法 ①
bool isLast  = ((s2Cur + 1) * s2BaseSize >= actS2Size); // 重复乘法 ②
```

### 3.2 修改后

```cpp
// 修改后（第 801 行）
info.s2Idx = s2Cur * s2BaseSize;   // 只算一次

// 后续直接复用 info.s2Idx
bool isFirst = (info.s2Idx == 0);
bool isLast  = (info.s2Idx + s2BaseSize >= info.actS2Size);
```

### 3.3 性能分析

每次 `CreateTask` 调用节省 1–2 次乘法（约 2–4 cycles/任务）。

---

## 修改四：`ProcessVec1` — AttenMask DMA 提前到 `WaitCrossCore` 之前（**Change A**）

### 4.1 问题根因

AIV 在进入 `ProcessVec1` 后的执行顺序（修改前）：

```
① bmm1ResBuf.WaitCrossCore()   — 等待 AIC Cube Mm1 完成（阻塞）
② AttenMaskCopyIn()            — MTE2 DMA 从 GM 搬运 mask 到 UB（阻塞）
③ WaitFlag<MTE2_V>             — 等待 DMA 完成（内嵌于 AttentionmaskCopyIn）
④ V1 VF 计算
```

AIC 的 Cube 和 AIV 的 MTE2 是**独立的硬件单元**，完全可以并行，
但原代码将 MTE2 DMA 排在 Mm1 之后，白白浪费了 Mm1 运行时的 MTE2 空闲时间。

### 4.2 修改后（`flash_attn_block_vec_noquant_gqa.h`，第 257 行）

```cpp
// ── ProcessVec1 修改后 ──────────────────────────────────────────────
LocalTensor<uint8_t> attenMaskUb;
if constexpr (!useDn && hasAtten) {
    if (likely(runInfo.actVecMSize != 0)) {
        // ▼ 在 WaitCrossCore 之前启动 MTE2 DMA — 与 AIC Cube Mm1 并行
        set_mark(0x50010000);
        attenMaskUb = this->attenMaskInQue[...].AllocTensor<uint8_t>();
        AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo);
        set_mark(0x500100FF);
    }
}

bmm1ResBuf.WaitCrossCore();  // ← 此时 mask DMA 已完成（与 Mm1 并行运行）
// ...
ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo, attenMaskUb);  // ← 传入已就绪的 mask
```

`ProcessVec1Nd` 相应修改：去掉内部的 `AllocTensor + AttenMaskCopyIn`，改为接收参数。

### 4.3 时序分析

```
修改前时序（串行）：
  AIV: [scalar] → [WaitCrossCore ████████] → [MTE2 DMA ████] → [V1 VF]
  AIC:           [Mm1 ████████████]
  MTE2:                                       [DMA ████]        ← DMA 排在 Mm1 之后

修改后时序（并行）：
  AIV: [scalar] → [MTE2 DMA ████] → [WaitCrossCore ████] → [V1 VF]
  AIC:           [Mm1 ████████████]
  MTE2:          [DMA ████]                               ← DMA 与 Mm1 并行
```

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| V1 总等待时间 | `T_Mm1 + T_mask_DMA` | `max(T_Mm1, T_mask_DMA)` |
| **节省时间** | — | `min(T_Mm1, T_mask_DMA)` ≈ **T_mask_DMA** |

TND_MIXED_02 中 mask 大小 = 64行 × 128列 = 8 KB（uint8），MTE2 DMA 约耗时 **1–2 μs**。
每核 2 任务 × 2 subBlock = 4 次 V1 调用，预计节省 **4–8 μs/核**。

---

## 修改五：两阶段主循环（**Change B**）

### 5.1 问题根因

原循环将 `CalcParams`（标量）与硬件执行**交织**在同一迭代中：

```
原循环每次迭代：
  [CalcParams(N)] → [ExecuteTask(N): Mm1/Vec1(N) + Mm2/Vec2(N-2)]
       ↑ 标量开销                  ↑ 硬件执行（阻塞等待）
```

profiler 上的表现：
```
C1(0)开始 → C1(0)结束 → ████ 标量空档 ████ → C1(1)开始 → C1(1)结束 → ...
                          CalcParams(1) 占用
```

即：**每个相邻 C1/V1/C2/V2 之间都存在一个 CalcParams 大小的标量空档**。

### 5.2 修改后（`flash_attn_kernel_base_gqa_template.h`，第 438 行）

```cpp
// ══ 阶段一：预分发（纯标量，无硬件）══════════════════════════════
// 将 taskRunInfo 缓冲区填满后再启动硬件，消除首批阶段的前置标量延迟
while (ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur) &&
       createdTaskCount - executedTaskCount < PRELOAD_TASK_CACHE_SIZE) {
    TASK_DEAL_MODE mode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
    if (mode == CREATE_TASK) {
        CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
        createdTaskCount++;
    } else if (mode == DEAL_ZERO) {
        if ASCEND_IS_AIV { vecFaBlock.DealZeroActSeqLen(bN2Cur); }
    }
    UpdateAxisInfo(mode, bN2Cur, gS1Cur, s2Cur);
}

// ══ 阶段二：执行优先（先硬件，后标量）══════════════════════════════
// 每轮先派发硬件（无前置标量），硬件返回后再补充分发下一批任务
bool shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
bool shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
while (shouldExecuteTask || shouldDispatchTask) {
    if (shouldExecuteTask) {
        ExecuteTask(executedTaskCount, taskRunInfo);   // ← 硬件优先
        executedTaskCount++;
    }
    // 硬件结束后补充分发（标量紧跟硬件，不占用下一轮硬件的前置时间）
    while (ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur) &&
           createdTaskCount - executedTaskCount < PRELOAD_TASK_CACHE_SIZE) {
        TASK_DEAL_MODE mode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
        if (mode == CREATE_TASK) {
            CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
            createdTaskCount++;
        } else if (mode == DEAL_ZERO) {
            if ASCEND_IS_AIV { vecFaBlock.DealZeroActSeqLen(bN2Cur); }
        }
        UpdateAxisInfo(mode, bN2Cur, gS1Cur, s2Cur);
    }
    shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
    shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
}
```

### 5.3 时序对比

**修改前（N=2 任务/核）：**
```
标量: [CalcParams(0)] [CalcParams(1)]
硬件:                 [C1(0)]        [scalar空档] [C1(1)]        [C2(0)] [C2(1)]
                                     ↑ CalcParams(1) 时间
```

**修改后（N=2 任务/核）：**
```
标量: [CalcParams(0)] [CalcParams(1)]   （预分发阶段，硬件启动前）
硬件:                                  [C1(0)] [C1(1)] [C2(0)] [C2(1)]
                                                ↑ 无空档！
```

### 5.4 定量分析

设 `X` = 单次 CalcParams 及相关标量开销（GetTaskDealMode + CalcCurS2StartEnd + CalcParams + UpdateAxisInfo），
估计 `X` ≈ **500–2000 ns**（约 500–2000 cycles @ 1GHz）。

| 场景 | 修改前硬件间空档总时间 | 修改后 | 节省 |
|------|----------------------|--------|------|
| N=2 任务/核（TND_MIXED_02） | `1 × X` | `0` | **100%** |
| N=4 任务/核 | `3 × X` | `0` | **100%** |
| N=8 任务/核 | `7 × X` | `3 × X` | **57%** |
| N=16 任务/核 | `15 × X` | `11 × X` | **27%** |

> 预分发阶段深度 = `PRELOAD_TASK_CACHE_SIZE = 4`，覆盖 ≤4 任务/核时消除全部空档。

### 5.5 正确性说明

- `DEAL_ZERO`（actSeqLen=0 的批次）在预分发阶段调用 `DealZeroActSeqLen`，写出独立的输出区域，不影响流水线硬件事件。
- `SKIP / SKIP_ZERO / S2_END` 模式只调用 `UpdateAxisInfo`，不创建任务，缓冲区计数不变。
- 总工作量与原循环完全相同（标量工作被前移而非消除），correctness 不变。

---

## 总体收益估算（TND_MIXED_02，每核 2 任务）

| 优化点 | 节省量（估算） | 类型 |
|--------|--------------|------|
| `LastDivNewVF` 去冗余 `CreateMask` | ~510 次标量指令/V2调用 × 2 核阶段 | 减少 VF 核标量指令 |
| `gSize==1` 跳过整除 | ~10–20 cycles/换行 | 减少标量除法 |
| `CalcParams` 复用 `s2Idx` | ~2–4 cycles/任务 | 减少标量乘法 |
| **AttenMask DMA 并行化（Change A）** | **4–8 μs/核**（≈1–2μs × 4次V1） | 隐藏 MTE2 DMA 延迟 |
| **两阶段主循环（Change B）** | **0.5–2 μs/核**（1次CalcParams空档 × X） | 消除硬件间标量空档 |

> Change A 和 Change B 是本次优化中影响最大的两项，针对 profiler 反映的"C1/V1/C2/V2 之间等待时间过长"问题直接解决。

---

## 修改文件清单

| 文件 | 改动摘要 |
|------|---------|
| `vf_flashupdate_new_local.h` | 新增本地副本；`LastDivNewVF` 将 `CreateMask` 提到循环外 |
| `flash_attn_block_vec_noquant_gqa.h` | `#include` 改为本地文件；`ProcessVec1` 提前 AttenMask DMA；`ProcessVec1Nd` 接收预取参数 |
| `flash_attn_kernel_base_gqa_template.h` | `CalcCurS2StartEndWithSparse` 增加 `gSize==1` 快速路径；`CalcParams` 复用 `s2Idx`；`FlashAttention()` 改为两阶段主循环 |
