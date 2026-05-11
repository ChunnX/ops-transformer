# GMMFR MxA8W4 同步机制分析

> 本文档详细分析 GMMFR（GroupedMatmulFinalizeRouting）算子在 MxA8W4 场景下的跨核同步机制，涵盖 AIC↔AIV 跨核同步信号、Pipe 内部 Event 同步、双缓冲/四缓冲流水线、以及完整的时序分析。

## 目录

- [1. 同步机制总览](#1-同步机制总览)
- [2. 跨核同步信号详解](#2-跨核同步信号详解)
  - [2.1 四路同步信号](#21-四路同步信号)
  - [2.2 双 Flag 机制](#22-双-flag-机制)
  - [2.3 同步原语实现](#23-同步原语实现)
- [3. Pipe Event 定义](#3-pipe-event-定义)
  - [3.1 AIC 侧 Event](#31-aic-侧-event)
  - [3.2 AIV 侧 Event](#32-aiv-侧-event)
  - [3.3 Event 配对关系总表](#33-event-配对关系总表)
- [4. AIV 路径同步时序](#4-aiv-路径同步时序)
  - [4.1 Weight 反量化流水线](#41-weight-反量化流水线)
  - [4.2 Routing 后处理流水线](#42-routing-后处理流水线)
- [5. AIC 路径同步时序](#5-aic-路径同步时序)
  - [5.1 MMA 计算流水线](#51-mma-计算流水线)
  - [5.2 Fixpipe 同步](#52-fixpipe-同步)
- [6. 双缓冲与多缓冲机制](#6-双缓冲与多缓冲机制)
  - [6.1 各缓冲区总览](#61-各缓冲区总览)
  - [6.2 weightHighBit 四缓冲](#62-weighthighbit-四缓冲)
  - [6.3 weightLowBit 双缓冲/四缓冲](#63-weightlowbit-双缓冲四缓冲)
  - [6.4 L0A/L0B 双缓冲](#64-l0al0b-双缓冲)
  - [6.5 L1 双缓冲](#65-l1-双缓冲)
  - [6.6 索引计算逻辑](#66-索引计算逻辑)
- [7. 完整数据流时序图](#7-完整数据流时序图)
  - [7.1 初始化阶段](#71-初始化阶段)
  - [7.2 稳态计算阶段（单个 BasicBlock 内）](#72-稳态计算阶段单个-basicblock-内)
  - [7.3 BasicBlock 边界阶段](#73-basicblock-边界阶段)
  - [7.4 结束阶段](#74-结束阶段)
  - [7.5 多 BasicBlock 流水线全景](#75-多-basicblock-流水线全景)
- [8. 同步点标注汇总](#8-同步点标注汇总)

---

## 1. 同步机制总览

GMMFR MxA8W4 场景采用 **AIC（Cube Core）+ AIV（Vector Core）异构双核** 协同工作模式。同步机制分为两个层次：

| 层次 | 机制 | 作用范围 | 用途 |
|------|------|----------|------|
| **跨核同步** | `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIC ↔ AIV | 协调 AIC 的 MMA 计算与 AIV 的 Weight 反量化、Routing 后处理 |
| **核内 Pipe 同步** | `SetFlag` / `WaitFlag` (HardEvent) | 单核内部各 Pipe | 协调 MTE1/MTE2/MTE3/VF/MMA/Fixpipe 之间的数据依赖 |

**核心设计思想**：AIC 和 AIV 形成 **乒乓流水线** —— AIV 先反量化 Weight 供 AIC 做 MMA，AIC 算完后通过 Fixpipe 将结果输出到 UB，AIV 再做 MulLogits + Routing 后处理。两个核通过 4 路跨核同步信号协调，核内通过多级 Event 实现搬运与计算的深度重叠。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        同步机制全景                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────┐       ┌──────────────────────────┐   │
│  │       AIC (Cube Core)    │       │       AIV (Vector Core)  │   │
│  │                          │       │                          │   │
│  │  MTE1 ──Event──► MTE2    │       │  MTE2 ──Event──► VF     │   │
│  │  MTE1 ──Event──► MMA     │       │  VF   ──Event──► MTE2   │   │
│  │  MMA  ──Event──► MTE1    │       │  VF   ──Event──► MTE3   │   │
│  │  MTE2 ──Event──► MTE1    │       │  MTE3 ──Event──► VF     │   │
│  │                          │       │                          │   │
│  │       CrossCore ◄────────┼───────┼──► CrossCore             │   │
│  │   SetAicToAiv / WaitAiv  │       │   SetAivToAic / WaitAic  │   │
│  └──────────────────────────┘       └──────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 跨核同步信号详解

### 2.1 四路同步信号

跨核同步信号定义在 `tool.h` 中，使用 `SYNC_MODE4` 模式：

```cpp
// tool.h
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;          // AIV → AIC
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;          // AIC → AIV
static constexpr uint64_t SYNC_AIC_FIX_AIV_VF_FLAG = 3;   // AIC → AIV (Fixpipe)
static constexpr uint64_t SYNC_AIV_MTE3_AIC_FIX_FLAG = 4; // AIV → AIC (Fixpipe)
static constexpr uint64_t SYNC_MODE4 = 4;
static constexpr uint64_t FLAG_ID_MAX = 16;
```

| 信号 | 值 | 方向 | 关联 Pipe | 语义 |
|------|-----|------|-----------|------|
| `SYNC_AIV_AIC_FLAG` | 8 | AIV → AIC | PIPE_MTE1 / PIPE_MTE3 | AIV 完成 Weight 反量化并写入 L1，AIC 可以开始 MMA |
| `SYNC_AIC_AIV_FLAG` | 9 | AIC → AIV | PIPE_MTE1 / PIPE_MTE3 | AIC 完成 MMA 计算，AIV 可以继续搬运下一块 Weight |
| `SYNC_AIV_MTE3_AIC_FIX_FLAG` | 4 | AIV → AIC | PIPE_MTE3 / PIPE_FIX | AIV 已取走上一个 BasicBlock 的 UB 结果，AIC 可以 Fixpipe 输出新结果 |
| `SYNC_AIC_FIX_AIV_VF_FLAG` | 3 | AIC → AIV | PIPE_FIX / PIPE_V | AIC Fixpipe 完成，UB 上的 MMA 结果已就绪，AIV 可以做 MulLogits + Routing |

**四路信号的分工**：

- **FLAG 8/9**：控制 **K 循环内的 Weight 反量化 ↔ MMA 计算** 流水线。每个 K 块迭代中，AIV 反量化完 Weight 后通知 AIC（FLAG 8），AIC 做完 MMA 后通知 AIV（FLAG 9）。
- **FLAG 4/3**：控制 **BasicBlock 边界的 Fixpipe ↔ Routing 后处理** 流水线。AIV 取走上一个 BasicBlock 的结果后通知 AIC（FLAG 4），AIC Fixpipe 输出新结果后通知 AIV（FLAG 3）。

### 2.2 双 Flag 机制

GMMFR 使用 **不对称双 Flag 机制** 确保跨核同步的可靠性：

```
AIC 通知 AIV（SetAicToAiv）:
  CrossCoreSetFlag<SYNC_MODE4, PIPE>(syncFlag + FLAG_ID_MAX)  // 先设 flag+16
  CrossCoreSetFlag<SYNC_MODE4, PIPE>(syncFlag)                // 再设 flag

AIV 等待 AIC（WaitAicToAiv）:
  CrossCoreWaitFlag<SYNC_MODE4, PIPE>(syncFlag)               // 只等 flag

AIV 通知 AIC（SetAivToAic）:
  CrossCoreSetFlag<SYNC_MODE4, PIPE>(syncFlag)                // 只设 flag

AIC 等待 AIV（WaitAivToAic）:
  CrossCoreWaitFlag<SYNC_MODE4, PIPE>(syncFlag + FLAG_ID_MAX) // 先等 flag+16
  CrossCoreWaitFlag<SYNC_MODE4, PIPE>(syncFlag)               // 再等 flag
```

**设计原理**：

```
┌──────────────────────────────────────────────────────────────────┐
│                     双 Flag 机制时序                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AIC 侧 (SetAicToAiv):                                          │
│    SetFlag(syncFlag + 16) ─────┐                                 │
│    SetFlag(syncFlag) ─────┐    │                                 │
│                            │    │                                 │
│  AIV 侧 (WaitAicToAiv):   │    │                                 │
│    WaitFlag(syncFlag) ◄────┘    │  ← AIV 看到 syncFlag 即认为    │
│    (不等待 syncFlag+16)         │    通知完成                     │
│                                 │                                 │
│  AIV 侧 (SetAivToAic):         │                                 │
│    SetFlag(syncFlag) ──────────┼──┐                              │
│                                │  │                              │
│  AIC 侧 (WaitAivToAic):       │  │                              │
│    WaitFlag(syncFlag + 16) ◄───┘  │  ← AIC 先确认 flag+16 已设   │
│    WaitFlag(syncFlag) ◄───────────┘  ← 再确认 flag 已设          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**不对称设计的原因**：

- **AIC → AIV 方向**：AIC 先设 `flag+16` 再设 `flag`，AIV 只等 `flag`。AIV 看到 `flag` 时，`flag+16` 必然已经设置（因为 AIC 先设 `flag+16`），所以 AIV 只需等一个 flag 即可。
- **AIV → AIC 方向**：AIV 只设 `flag`，AIC 先等 `flag+16` 再等 `flag`。AIC 通过等待 `flag+16` 确保不会读到上一轮的残留 flag（因为上一轮 AIC 设了 `flag+16`），然后等待 `flag` 确认 AIV 已完成。

这种设计避免了跨核同步中的 **flag 残留问题**：如果 AIC 只等 `flag`，可能读到上一轮 AIC 自己设置的 `flag`（因为 AIC→AIV 方向也设了 `flag`），导致误触发。通过先等 `flag+16`（只有 AIC 会设），确保区分 AIC 自己设的 flag 和 AIV 设的 flag。

### 2.3 同步原语实现

```cpp
// gmm_fr_weight_quant_vcv_basic_block.h

// AIV 通知 AIC（单 flag）
template <const pipe_t pipe>
__aicore__ inline void SetAivToAic(uint64_t syncFlag) {
    CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
}

// AIC 等待 AIV（双 flag 等待）
template <const pipe_t pipe>
__aicore__ inline void WaitAivToAic(uint64_t syncFlag) {
    CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
    CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
}

// AIC 通知 AIV（双 flag 设置）
template <const pipe_t pipe>
__aicore__ inline void SetAicToAiv(uint64_t syncFlag) {
    CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
    CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
}

// AIV 等待 AIC（单 flag）
template <const pipe_t pipe>
__aicore__ inline void WaitAicToAiv(uint64_t syncFlag) {
    CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
}
```

**Pipe 参数说明**：跨核同步函数模板参数 `pipe` 指定同步信号关联的硬件 Pipe，确保同步信号与对应 Pipe 的数据流正确关联：

| 调用场景 | Pipe 参数 | 含义 |
|----------|-----------|------|
| `SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG)` | PIPE_MTE1 | AIC 的 MTE1 搬运完成后通知 AIV |
| `WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG)` | PIPE_MTE3 | AIV 在 MTE3 路径上等待 AIC |
| `SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG)` | PIPE_MTE3 | AIV 的 MTE3 搬运完成后通知 AIC |
| `WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG)` | PIPE_MTE1 | AIC 在 MTE1 路径上等待 AIV |
| `SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG)` | PIPE_MTE3 | AIV 后处理就绪，通知 AIC Fixpipe |
| `WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG)` | PIPE_FIX | AIC 在 Fixpipe 路径上等待 AIV |
| `SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG)` | PIPE_FIX | AIC Fixpipe 完成，通知 AIV |
| `WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG)` | PIPE_V | AIV 在 VF 路径上等待 AIC |

---

## 3. Pipe Event 定义

### 3.1 AIC 侧 Event

AIC 侧 Event 定义在 `gmm_fr_weight_quant_cube_compute.h` 中，用于 Cube 核内部的 MTE1↔MTE2↔MMA 流水线同步：

| Event 名称 | Event ID | 方向 | 缓冲数 | 用途 |
|-----------|----------|------|--------|------|
| `EVENT_ID_MTE1_TO_MTE2` | 3 | MTE1 → MTE2 | 2（双缓冲） | A 矩阵 MTE1 搬运（GM→L1）完成后通知 MTE2 可读 |
| `EVENT_ID_SCALE_MTE1_TO_MTE2` | 5 | MTE1 → MTE2 | 2（双缓冲） | MX Scale MTE1 搬运（GM→L1）完成后通知 MTE2 可读 |
| `EVENT_ID_M_TO_MTE1` | 3 | MMA → MTE1 | 2（双缓冲） | MMA 计算完成后通知 MTE1 可以覆盖 L0A/L0B |
| `EVENT_ID_MTE1_TO_M` | 3 | MTE1 → MMA | 1 | MTE1 加载 L0A/L0B 完成后通知 MMA 可以开始计算 |
| `EVENT_ID_MTE2_TO_MTE1` | 3 | MTE2 → MTE1 | 1 | MTE2（AIV 侧 Weight 反量化）完成后通知 MTE1 |

**AIC 侧 Event 初始化**（`MxA8W4Init`）：

```cpp
for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + i);       // ID 3, 4
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + i); // ID 5, 6
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);             // ID 3, 4
}
```

初始化时将所有 Event 置为完成状态（`SetFlag`），确保首次 `WaitFlag` 不会阻塞。

### 3.2 AIV 侧 Event

AIV 侧 Event 定义在 `gmm_fr_weight_quant_vec_compute.h` 中，用于 Vector 核内部的 MTE2↔VF↔MTE3 流水线同步：

| Event 名称 | Event ID | 方向 | 缓冲数 | 用途 |
|-----------|----------|------|--------|------|
| `EVENT_ID_MTE3_TO_V` | 0 | MTE3 → VF | 4（四缓冲） | weightHighBit MTE3 搬运（UB→L1）完成后通知 VF 可覆盖写入 |
| `EVENT_ID_BIAS_FR_MTE3_TO_V` | 4 | MTE3 → VF | 1 | Routing 阶段 MTE3 scatter-add 完成后通知 VF |
| `EVENT_ID_V_TO_MTE2` | 0 | VF → MTE2 | 2/4 | weightLowBit VF 消费完后通知 MTE2 可覆盖写入 |
| `EVENT_ID_FR_V_TO_MTE2` | 4 | VF → MTE2 | 2（双缓冲） | Routing 阶段 VF 处理完 logits 后通知 MTE2 |
| `EVENT_ID_MTE2_TO_V` | 0 | MTE2 → VF | 1 | weightLowBit MTE2 搬运（GM→UB）完成后通知 VF 可读 |
| `EVENT_ID_RL_MTE2_TO_V` | 1 | MTE2 → VF | 2（双缓冲） | Routing 阶段 MTE2 搬运完 rowIndex/logits 后通知 VF |
| `EVENT_ID_V_TO_MTE3` | 0 | VF → MTE3 | 1 | VF 反量化完成后通知 MTE3 可搬运到 L1 |
| `EVENT_ID_S_TO_MTE2` | 0 | Scalar → MTE2 | 2（双缓冲） | Routing 阶段 Scalar 处理完成后通知 MTE2 |
| `EVENT_ID_MTE2_TO_S` | 0 | MTE2 → Scalar | 2（双缓冲） | Routing 阶段 MTE2 搬运完成后通知 Scalar |

**AIV 侧 Event 初始化**（`Init`）：

```cpp
// MTE3→V: 四缓冲，每个 buffer 对应一个 Event
for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);  // ID 0, 1, 2, 3
}
// S→MTE2 和 FR_V→MTE2: 双缓冲
for (uint16_t idx = 0; idx < DOUBLE_BUFFER_NUM; idx++) {
    SetFlag<HardEvent::S_MTE2>(EVENT_ID_S_TO_MTE2 + idx);       // ID 0, 1
    SetFlag<HardEvent::V_MTE2>(EVENT_ID_FR_V_TO_MTE2 + idx);    // ID 4, 5
}
// V→MTE2: ubMte2BufferNum 缓冲
for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
    SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);       // ID 0, 1 (或 0,1,2,3)
}
```

### 3.3 Event 配对关系总表

| 生产者 | 消费者 | Event | Set 位置 | Wait 位置 | 缓冲 |
|--------|--------|-------|----------|-----------|------|
| MTE2 | VF | `MTE2_V(0)` | `CopyGmToUb` 末尾 | `CopyGmToUb` 末尾（同步等待） | 1 |
| VF | MTE2 | `V_MTE2(0..N-1)` | `SetVToMTE2` | `WaitVToMTE2`（下一轮 CopyGmToUb 前） | 2/4 |
| VF | MTE3 | `V_MTE3(0)` | `WeightAntiQuantComputeNzNk` 中 | `WeightAntiQuantComputeNzNk` 中（同步等待） | 1 |
| MTE3 | VF | `MTE3_V(0..3)` | `WeightAntiQuantComputeNzNk` 末尾 | `WeightAntiQuantComputeNzNk` 开头 | 4 |
| MTE2 | VF | `RL_MTE2_V(0..1)` | `SetMte2ToS` | `WaitMte2ToS` | 2 |
| MTE2 | Scalar | `MTE2_S(0..1)` | `SetMte2ToS` | `WaitMte2ToS` | 2 |
| Scalar | MTE2 | `S_MTE2(0..1)` | `SetSToMTE2` | `WaitSToMTE2` | 2 |
| VF | MTE2 | `FR_V_MTE2(0..1)` | `SetFrToMTE2` | `WaitFrVToMTE2` | 2 |
| MTE3 | VF | `MTE3_V(BIAS_FR)` | `RoutingYToGm` 末尾 | `RoutingYToGm` 末尾（同步等待） | 1 |
| MTE1 | MTE2 | `MTE1_MTE2(3,4)` | `SetMTE1ToMTE2` | `WaitMTE1ToMTE2` | 2 |
| MTE1 | MTE2 | `MTE1_MTE2(5,6)` | `SetScaleMTE1ToMTE2` | `WaitScaleMTE1ToMTE2` | 2 |
| MMA | MTE1 | `M_MTE1(3,4)` | `DoMmadCompute` 末尾 | `LaunchMatmul` K 循环开头 | 2 |
| MTE1 | MMA | `MTE1_M(3)` | `DoMmadCompute` 中 | `DoMmadCompute` 中（同步等待） | 1 |
| MTE2 | MTE1 | `MTE2_MTE1(3)` | `LaunchMatmul` 开头 | `LaunchMatmul` 开头（同步等待） | 1 |

---

## 4. AIV 路径同步时序

### 4.1 Weight 反量化流水线

AIV 侧的 Weight 反量化流水线涉及 **MTE2 → VF → MTE3** 三级流水，通过四缓冲实现深度重叠。

**单次迭代的同步流程**（`VecComputeNzNkWithStartLimit` 中每次 K 循环）：

```
步骤 1: WaitVToMTE2()
        └── WaitFlag<V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (ubMte2BufferNum - 1)))
            等待 VF 消费完上一轮 weightLowBit，MTE2 buffer 可覆盖

步骤 2: CopyGmToUb()
        ├── [if hasBias && kGmOffset==0] DataCopyPad2D(bias GM→UB)
        │   ├── SetFlag<MTE2_V>(EVENT_ID_MTE2_TO_V)
        │   └── WaitFlag<MTE2_V>(EVENT_ID_MTE2_TO_V)  ← 同步等待 Bias 搬运完成
        ├── DataCopyPad2D(weight GM→UB)  ← MTE2 搬运 4bit Weight
        ├── SetFlag<MTE2_V>(EVENT_ID_MTE2_TO_V)
        └── WaitFlag<MTE2_V>(EVENT_ID_MTE2_TO_V)  ← 同步等待 Weight 搬运完成

步骤 3: WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG)  ★ 跨核同步点
        └── 等待 AIC 完成上一轮 MMA，确保 L1 上的 Weight 空间可用

步骤 4: WeightAntiQuantComputeNzNk()
        ├── WaitFlag<MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & 3))
        │   └── 等待 MTE3 释放 weightHighBit buffer（四缓冲轮转）
        ├── AntiQuantProcessNzMxA8W4()  ← VF: 4bit→8bit 反量化
        ├── SetFlag<V_MTE3>(EVENT_ID_V_TO_MTE3)
        ├── WaitFlag<V_MTE3>(EVENT_ID_V_TO_MTE3)  ← 同步等待 VF 完成
        ├── CopyWeightHighBitForAligned()  ← MTE3: UB→L1
        ├── [if hasBias] DataCopy(bias UB→L1)
        └── SetFlag<MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & 3))
            └── 释放 weightHighBit buffer 给下一轮

步骤 5: SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG)  ★ 跨核同步点
        └── 通知 AIC：Weight 反量化完成，L1 上的 Weight 已就绪

步骤 6: SetVToMTE2()
        └── SetFlag<V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (ubMte2BufferNum - 1)))
            释放 weightLowBit buffer 给下一轮 MTE2 搬运
```

**四缓冲流水线时序**（weightHighBit）：

```
Buffer 索引:  buf0      buf1      buf2      buf3      buf0      buf1
             ┌────────┬────────┬────────┬────────┬────────┬────────┐
VF:          │ Wait   │ Wait   │ Wait   │ Wait   │ Wait   │ Wait   │
             │MTE3_V  │MTE3_V  │MTE3_V  │MTE3_V  │MTE3_V  │MTE3_V  │
             │ (0)    │ (1)    │ (2)    │ (3)    │ (0)    │ (1)    │
             │ 反量化  │ 反量化  │ 反量化  │ 反量化  │ 反量化  │ 反量化  │
             │ Set    │ Set    │ Set    │ Set    │ Set    │ Set    │
             │V_MTE3  │V_MTE3  │V_MTE3  │V_MTE3  │V_MTE3  │V_MTE3  │
             ├────────┼────────┼────────┼────────┼────────┼────────┤
MTE3:        │        │ Wait   │ Wait   │ Wait   │ Wait   │ Wait   │
             │        │V_MTE3  │V_MTE3  │V_MTE3  │V_MTE3  │V_MTE3  │
             │        │ UB→L1  │ UB→L1  │ UB→L1  │ UB→L1  │ UB→L1  │
             │        │ Set    │ Set    │ Set    │ Set    │ Set    │
             │        │MTE3_V  │MTE3_V  │MTE3_V  │MTE3_V  │MTE3_V  │
             │        │ (0)    │ (1)    │ (2)    │ (3)    │ (0)    │
             └────────┴────────┴────────┴────────┴────────┴────────┘

图例: VF 计算 buf0 时，MTE3 在搬运上一轮的 buf3。
      通过四缓冲 + Event 同步，VF 计算和 MTE3 搬运完全 overlap。
```

### 4.2 Routing 后处理流水线

Routing 后处理阶段（PhaseB）涉及 **MTE2 → VF → MTE3** 的流水线，处理上一个 BasicBlock 的 MMA 结果：

```
步骤 1: WaitSToMTE2(rlLoopIdx_)
        └── WaitFlag<S_MTE2>(EVENT_ID_S_TO_MTE2 + (rlLoopIdx_ & 1))
            等待上一轮 Scalar 处理完成，MTE2 buffer 可用

步骤 2: WaitFrVToMTE2(rlLoopIdx_)
        └── WaitFlag<V_MTE2>(EVENT_ID_FR_V_TO_MTE2 + (rlLoopIdx_ & 1))
            等待上一轮 VF 处理完成

步骤 3: CopyRowIndexLogitsGmToUb()
        ├── DataCopyPad2D(rowIndex GM→UB)
        └── DataCopyPad2D(logits GM→UB)

步骤 4: SetMte2ToS(rlLoopIdx_)
        ├── SetFlag<MTE2_S>(EVENT_ID_MTE2_TO_S + (rlLoopIdx_ & 1))
        └── SetFlag<MTE2_V>(EVENT_ID_RL_MTE2_TO_V + (rlLoopIdx_ & 1))
            通知 Scalar 和 VF：rowIndex/logits 已就绪

步骤 5: SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG)  ★ 跨核同步点
        └── 通知 AIC：AIV 已取走上一个 BasicBlock 的 UB 结果，可以 Fixpipe

步骤 6: WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG)  ★ 跨核同步点
        └── 等待 AIC Fixpipe 完成，UB 上的 MMA 结果已就绪

步骤 7: MulLogits()
        └── VF: y = y * 64.0 * logits[m]

步骤 8: SetFrToMTE2(rlLoopIdx_)
        └── SetFlag<V_MTE2>(EVENT_ID_FR_V_TO_MTE2 + (rlLoopIdx_ & 1))
            通知 MTE2：VF 处理完成

步骤 9: WaitMte2ToS(rlLoopIdx_)
        ├── WaitFlag<MTE2_S>(EVENT_ID_MTE2_TO_S + (rlLoopIdx_ & 1))
        └── WaitFlag<MTE2_V>(EVENT_ID_RL_MTE2_TO_V + (rlLoopIdx_ & 1))
            等待 rowIndex/logits 搬运完成

步骤 10: RoutingYToGm()
         ├── SetFlag<V_MTE3>(EVENT_ID_V_TO_MTE3)
         ├── WaitFlag<V_MTE3>(EVENT_ID_V_TO_MTE3)
         ├── SetAtomicAdd<float>()
         ├── for each mIdx: DataCopyPad2D(UB→GM, atomic add)
         ├── SetAtomicNone()
         ├── SetFlag<MTE3_V>(EVENT_ID_BIAS_FR_MTE3_TO_V)
         └── WaitFlag<MTE3_V>(EVENT_ID_BIAS_FR_MTE3_TO_V)

步骤 11: SetSToMTE2(rlLoopIdx_)
         └── SetFlag<S_MTE2>(EVENT_ID_S_TO_MTE2 + (rlLoopIdx_ & 1))
             通知 MTE2：Scalar 处理完成
```

---

## 5. AIC 路径同步时序

### 5.1 MMA 计算流水线

AIC 侧的 MMA 计算流水线涉及 **MTE1 → MTE2 → MMA** 三级流水，通过 L1/L0 双缓冲实现重叠。

**单次 K 循环迭代的同步流程**（`IterateNzNkWithKAic` 中每次 K 循环）：

```
步骤 1: WaitScaleMTE1ToMTE2(kbL1Offset)
        └── if (kbL1Offset % MX_SCALE_K_L1_SIZE == 0):
            WaitFlag<MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + (mxScaleBufIdx_ & 1))
            等待上一轮 Scale MTE1→MTE2 搬运完成，Scale L1 buffer 可覆盖

步骤 2: CopyMxScaleGmToL1()
        └── if (kbL1Offset % MX_SCALE_K_L1_SIZE == 0):
            ├── DataCopy(ScaleA GM→L1, Dn2Nz)
            └── DataCopy(ScaleB GM→L1, Dn2Nz)

步骤 3: WaitMTE1ToMTE2(kbL1Offset, param)
        └── if (kaGmOffset % param.kaL1Size == 0):
            WaitFlag<MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + (aL1BufIdx_ & 1))
            等待上一轮 A 矩阵 MTE1→MTE2 搬运完成，A L1 buffer 可覆盖

步骤 4: CopyAAndBiasGmToL1()
        └── if (kaGmOffset % param.kaL1Size == 0):
            ├── DataCopy(A 矩阵 GM→L1, Nd2Nz)
            └── PadL1IfNotAlign()  ← K 不对齐时尾部填零

步骤 5: PadL1IfNotAlign(weightL1)
        └── if (kbL1RealSize % K_ALIGNMENT64 != 0):
            FillL1WithZero()  ← Weight L1 尾部填零

步骤 6: WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG)  ★ 跨核同步点
        └── 等待 AIV 完成 Weight 反量化，L1 上的 Weight 已就绪

步骤 7: LaunchMatmul()  ← MMA 计算核心
        ├── SetFlag<MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1)
        ├── WaitFlag<MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1)  ← 同步等待 MTE2→MTE1
        │
        ├── for l1KOffset in [0, kbL1RealSize) step kL0Size:
        │   ├── WaitFlag<M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId)
        │   │   └── 等待上一轮 MMA 完成，L0A/L0B 可覆盖
        │   ├── LoadAAndScaleL1ToL0()  ← MTE1: L1→L0A + MX ScaleA
        │   ├── LoadBAndScaleL1ToL0()  ← MTE1: L1→L0B + MX ScaleB
        │   ├── DoMmadCompute():
        │   │   ├── [if hasBias && isFirstKLoop] LoadBiasToBt()
        │   │   ├── SetFlag<MTE1_M>(EVENT_ID_MTE1_TO_M)
        │   │   ├── WaitFlag<MTE1_M>(EVENT_ID_MTE1_TO_M)  ← 等待 L0A/L0B 加载完成
        │   │   ├── MmadCompute()  ← MMA 计算
        │   │   └── SetFlag<M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId)
        │   │       └── 通知 MTE1：MMA 完成，L0A/L0B 可覆盖
        │   └── l0LoopIdx_++

步骤 8: SetMTE1ToMTE2(kbL1Offset, param)
        └── if (条件满足):
            SetFlag<MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + (aL1BufIdx_ & 1))
            通知 A 矩阵 MTE1→MTE2 搬运完成

步骤 9: SetScaleMTE1ToMTE2(kbL1Offset, param)
        └── if (条件满足):
            SetFlag<MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + (mxScaleBufIdx_ & 1))
            通知 Scale MTE1→MTE2 搬运完成

步骤 10: SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG)  ★ 跨核同步点
         └── 通知 AIV：MMA 计算完成，可以继续搬运下一块 Weight
```

**LaunchMatmul 内部 L0 双缓冲流水线**：

```
L0A/L0B 双缓冲 (loopId = l0LoopIdx_ % 2):

  loopId=0:                          loopId=1:
  ┌─────────────────────┐            ┌─────────────────────┐
  │ WaitFlag<M_MTE1>(3) │            │ WaitFlag<M_MTE1>(4) │
  │ LoadA L1→L0A[0:32K] │            │ LoadA L1→L0A[32:64K]│
  │ LoadB L1→L0B[0:32K] │            │ LoadB L1→L0B[32:64K]│
  │ SetFlag<MTE1_M>     │            │ SetFlag<MTE1_M>     │
  │ WaitFlag<MTE1_M>    │            │ WaitFlag<MTE1_M>    │
  │ MmadCompute         │            │ MmadCompute         │
  │ SetFlag<M_MTE1>(3)  │            │ SetFlag<M_MTE1>(4)  │
  └─────────────────────┘            └─────────────────────┘
        ↑                                  ↑
        └──── 下一轮 WaitFlag<M_MTE1>(3) ──┘

  MTE1 加载 loopId=1 的同时，MMA 在计算 loopId=0 的结果。
```

### 5.2 Fixpipe 同步

Fixpipe 同步发生在 BasicBlock 边界，将 L0C 上的 MMA 累加结果输出到 UB：

```
AIC 侧 (IterateNzNkWithKAic 开头 / End):
  if (cvLoopIdx_ > 0):
    步骤 1: WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG)  ★ 跨核同步点
            └── 等待 AIV 取走上一个 BasicBlock 的 UB 结果

    步骤 2: GetTensorC(ubOutputF32Buffer_, lastOffsetParam)
            └── FixL0CToDst: L0C → UB (ROW_MAJOR, F32)
                l0cOffset_ += CeilAlign(mL0Size, 16) * CeilAlign(nL0Size, 16)

    步骤 3: SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG)  ★ 跨核同步点
            └── 通知 AIV：UB 上的 MMA 结果已就绪
```

**L0C 偏移管理**：

```
L0C 空间 (256×256 float = 256KB):
┌────────────────────────────────────────────────────────────┐
│ BB0 结果  │ BB1 结果  │ BB2 结果  │ ...  │ 空闲/回绕      │
│ M0×N0     │ M1×N1     │ M2×N2     │      │                │
└────────────────────────────────────────────────────────────┘
            ↑ l0cOffset_ 递增
            └── 当剩余空间不足时回绕到 0（kbOffset==0 时检查）
```

---

## 6. 双缓冲与多缓冲机制

### 6.1 各缓冲区总览

| 缓冲区 | 位置 | 缓冲数 | 单缓冲大小 | 切换索引 | 用途 |
|--------|------|--------|-----------|----------|------|
| weightHighBit | UB | 4（四缓冲） | 40KB | `ubComputeLoopIdx_ & 3` | VF 反量化输出 → MTE3 搬运到 L1 |
| weightLowBit | UB | 2 或 4 | 40KB 或 20KB | `weightMte2LoopIdx_ % N` | MTE2 搬运 4bit Weight → VF 消费 |
| bias | UB | 2 或 4 | 0.5KB 或 0.25KB | `weightMte2LoopIdx_ & 3` | MTE2 搬运 Bias → VF 消费 |
| biasRescale | UB | 2 或 4 | 0.5KB 或 0.25KB | `ubComputeLoopIdx_ & 3` | VF Bias 反量化输出 → MTE3 搬运到 L1 |
| logits | UB | 2（双缓冲） | 0.5KB | `rlLoopIdx_ & 1` | MTE2 搬运 Logits → VF MulLogits |
| rowIndex | UB | 2（双缓冲） | 1KB | `rlLoopIdx_ & 1` | MTE2 搬运 RowIndex → Routing |
| WeightL1 | L1 | 2（双缓冲） | 64KB | `cvLoopIdx_ & 1` | AIV MTE3 写入 → AIC MMA 读取 |
| BiasL1 | L1 | 2（双缓冲） | 4KB | `cvLoopIdx_ & 1` | AIV MTE3 写入 → AIC MMA 读取 |
| ScaleAL1 | L1 | 2（双缓冲） | 32KB | `mxScaleBufIdx_ & 1` | AIC MTE1 写入 → AIC L0 加载 |
| ScaleBL1 | L1 | 2（双缓冲） | 32KB | `mxScaleBufIdx_ & 1` | AIC MTE1 写入 → AIC L0 加载 |
| AL1 | L1 | 2（双缓冲） | 124KB/128KB | `aL1BufIdx_ & 1` | AIC MTE1 写入 → AIC L0 加载 |
| L0A | L0 | 2（双缓冲） | 32KB | `l0LoopIdx_ % 2` | MTE1 加载 → MMA 计算 |
| L0B | L0 | 2（双缓冲） | 32KB | `l0LoopIdx_ % 2` | MTE1 加载 → MMA 计算 |
| BiasTable | L0 | 2（双缓冲） | 2KB | `loopId` | MTE1 加载 Bias → MMA 使用 |

### 6.2 weightHighBit 四缓冲

weightHighBit 是 VF 反量化输出和 MTE3 搬运到 L1 之间的桥梁，使用四缓冲（`QUADRUPLE_BUFFER_NUM=4`）实现深度流水。

**缓冲布局**（跳写避免 Bank 冲突）：

```
weightHighBit_ UB 布局（四缓冲跳写，innerDstStride = 256 × 4 = 1024）:

buf0: [N0_K0][  gap  ][N1_K0][  gap  ]...[N0_K1][  gap  ][N1_K1][  gap  ]...
buf1: [  gap  ][N0_K0][  gap  ][N1_K0]...[  gap  ][N0_K1][  gap  ][N1_K1]...
buf2: [  gap  ][  gap  ][N0_K0][  gap  ]...[  gap  ][  gap  ][N0_K1][  gap  ]...
buf3: [  gap  ][  gap  ][  gap  ][N0_K0]...[  gap  ][  gap  ][  gap  ][N0_K1]...
       ←────── innerDstStride = 1024 ──────→
       ←────────────── loopKDstStride ──────────────→

跳写目的：相邻 N 块分散到不同 buffer 的同一偏移位置，避免 UB Bank 冲突。
MTE3 搬运时通过 srcStride = 3 × blockLen 跳过无效数据，只取当前 buffer 的有效部分。
```

**四缓冲同步流程**：

```
索引:  ubComputeLoopIdx_ & 3

VF 侧:
  WaitFlag<MTE3_V>(EVENT_ID_MTE3_TO_V + idx)  ← 等待 MTE3 释放 buffer idx
  AntiQuantProcessNzMxA8W4()                  ← VF 反量化写入 buffer idx
  SetFlag<V_MTE3>(EVENT_ID_V_TO_MTE3)         ← 通知 MTE3 可以搬运
  WaitFlag<V_MTE3>(EVENT_ID_V_TO_MTE3)        ← 同步等待 VF 完成
  CopyWeightHighBitForAligned()               ← MTE3 搬运 buffer idx → L1
  SetFlag<MTE3_V>(EVENT_ID_MTE3_TO_V + idx)   ← 释放 buffer idx
  ubComputeLoopIdx_++

四缓冲确保 VF 计算 buffer N 时，MTE3 在搬运 buffer N-1（或更早的 buffer），
两者完全 overlap。
```

### 6.3 weightLowBit 双缓冲/四缓冲

weightLowBit 的缓冲数由 `vecConfig.ubMte2BufferNum` 动态配置：

- **默认（2）**：`VEC_ANTIQUANT_CONFIG_DEFAULT`，`ubMte2BufferNum = 2`
- **动态（4）**：`VEC_ANTIQUANT_CONFIG_DYNAMIC`，`ubMte2BufferNum = 4`

```
双缓冲模式 (ubMte2BufferNum=2):
  MTE2:  [搬 buf0]        [搬 buf1]        [搬 buf0]        ...
  VF:          [消费 buf0]      [消费 buf1]      [消费 buf0] ...
  Event: Set V_MTE2(0)   Set V_MTE2(1)   Set V_MTE2(0)   ...
         Wait V_MTE2(0)  Wait V_MTE2(1)  Wait V_MTE2(0)  ...

四缓冲模式 (ubMte2BufferNum=4):
  MTE2:  [搬 buf0][搬 buf1][搬 buf2][搬 buf3][搬 buf0]...
  VF:          [消费 buf0][消费 buf1][消费 buf2][消费 buf3]...
  Event: Set V_MTE2(0..3) 循环
         Wait V_MTE2(0..3) 循环
```

### 6.4 L0A/L0B 双缓冲

L0A 和 L0B 各 64KB，分为两个 32KB 的缓冲区：

```
L0A 布局:
┌──────────────────────┬──────────────────────┐
│ L0A buf0 (32KB)      │ L0A buf1 (32KB)      │
│ loopId=0 使用         │ loopId=1 使用         │
└──────────────────────┴──────────────────────┘
  offset=0               offset=32*1024 (L0_BUF_OFFSET_B8)

L0B 布局: 同 L0A

同步流程:
  loopId=0:
    WaitFlag<M_MTE1>(3)     ← 等待 MMA 释放 buf0
    LoadA L1→L0A buf0       ← MTE1 加载
    LoadB L1→L0B buf0
    SetFlag<MTE1_M>(3)      ← 通知 MMA 数据就绪
    WaitFlag<MTE1_M>(3)
    MmadCompute             ← MMA 使用 buf0
    SetFlag<M_MTE1>(3)      ← 释放 buf0

  loopId=1: (同上，使用 buf1 和 Event ID 4)
```

### 6.5 L1 双缓冲

L1 上的 Weight、Bias、ScaleA、ScaleB、A 矩阵均使用双缓冲，通过乒乓切换实现搬运与计算的重叠：

```
L1 双缓冲布局（以 WeightL1 为例）:

┌────────────────────────┬────────────────────────┐
│ WeightL1 P0 (64KB)     │ WeightL1 P1 (64KB)     │
│ cvLoopIdx_=0,2,4... 使用 │ cvLoopIdx_=1,3,5... 使用 │
└────────────────────────┴────────────────────────┘
  offset=0                 offset=weightL1DbOffset_

切换逻辑:
  cvLoopIdx_ & 1 == 0 → 使用 P0
  cvLoopIdx_ & 1 == 1 → 使用 P1

AIC 在 P0 上做 MMA 的同时，AIV 可以将下一块 Weight 反量化结果写入 P1。
```

### 6.6 索引计算逻辑

| 缓冲区 | 索引变量 | 计算方式 | 递增时机 |
|--------|----------|----------|----------|
| weightHighBit | `ubComputeLoopIdx_` | `idx & 3` | 每次 `WeightAntiQuantComputeNzNk` 末尾 `ubComputeLoopIdx_++` |
| weightLowBit | `weightMte2LoopIdx_` | `idx % ubMte2BufferNum` | 每次 `SetVToMTE2` 末尾 `weightMte2LoopIdx_++` |
| bias / biasRescale | `weightMte2LoopIdx_` / `ubComputeLoopIdx_` | `idx & 3` | 同上 |
| logits / rowIndex | `rlLoopIdx_` | `idx & 1` | 每次 PhaseB 末尾 `rlLoopIdx_++` |
| WeightL1 / BiasL1 | `cvLoopIdx_` | `idx & 1` | 每次 K 循环末尾 `cvLoopIdx_++` |
| ScaleAL1 / ScaleBL1 | `mxScaleBufIdx_` | `idx & 1` | `SetScaleMTE1ToMTE2` 条件满足时 `mxScaleBufIdx_++` |
| AL1 | `aL1BufIdx_` | `idx & 1` | `SetMTE1ToMTE2` 条件满足时 `aL1BufIdx_++` |
| L0A / L0B / BiasTable | `l0LoopIdx_` | `idx % 2` | 每次 `DoMmadCompute` 末尾 `l0LoopIdx_++` |

---

## 7. 完整数据流时序图

### 7.1 初始化阶段

```
时间 →

AIC:  [MxA8W4Init: 分配 L1 空间, 初始化 Event]
      [SetAicToAiv<PIPE_MTE1>(FLAG9)] ──────────────────────┐
      [SetAicToAiv<PIPE_MTE1>(FLAG9)] ──────────────────┐   │
                                                         │   │
AIV:  [Init: 分配 UB Buffer, 初始化 Event]               │   │
      [WaitAicToAiv<PIPE_MTE3>(FLAG9)] ◄─────────────────┘   │
      [WaitAicToAiv<PIPE_MTE3>(FLAG9)] ◄─────────────────────┘
      (AIC 初始化完成，AIV 收到通知)

      [InitAtomicGm: GM 清零 + SharedInput 初始化]
      (AIV 独立完成，不涉及跨核同步)
```

**说明**：初始化阶段 AIC 连续发送两次 `FLAG9`（双 flag 机制），AIV 等待两次 `FLAG9`。这确保 AIV 在 AIC 完成 L1 空间分配和 Event 初始化后才开始工作。

### 7.2 稳态计算阶段（单个 BasicBlock 内）

以下时序图展示单个 BasicBlock 内 K 循环的一次迭代（假设 `kbL1Size=256`，`kSize=1024`，共 4 次迭代）：

```
时间 → (单次 K 循环迭代)

AIC (Cube Core)                              AIV (Vector Core)
│                                            │
│  WaitScaleMTE1ToMTE2()                     │  WaitVToMTE2()
│  CopyMxScaleGmToL1()  ← MTE1: Scale→L1    │  CopyGmToUb()  ← MTE2: Weight→UB
│  WaitMTE1ToMTE2()                          │    ├── SetFlag<MTE2_V>(0)
│  CopyAAndBiasGmToL1() ← MTE1: A→L1        │    └── WaitFlag<MTE2_V>(0)
│  PadL1IfNotAlign()                         │
│                                            │  ★ WaitAicToAiv<PIPE_MTE3>(FLAG9)
│  ★ WaitAivToAic<PIPE_MTE1>(FLAG8) ◄───────┤      等待 AIC 完成上一轮 MMA
│      等待 AIV Weight 反量化完成             │
│                                            │  WeightAntiQuantComputeNzNk()
│  LaunchMatmul()                            │    ├── WaitFlag<MTE3_V>(idx)
│    ├── SetFlag<MTE2_MTE1>                  │    ├── VF: 4bit→8bit 反量化
│    ├── WaitFlag<MTE2_MTE1>                 │    ├── SetFlag<V_MTE3>(0)
│    ├── for l1KOffset:                      │    ├── WaitFlag<V_MTE3>(0)
│    │   ├── WaitFlag<M_MTE1>                │    ├── MTE3: UB→L1
│    │   ├── LoadA L1→L0A                    │    └── SetFlag<MTE3_V>(idx)
│    │   ├── LoadB L1→L0B                    │
│    │   ├── SetFlag<MTE1_M>                 │  ★ SetAivToAic<PIPE_MTE3>(FLAG8) ──→
│    │   ├── WaitFlag<MTE1_M>                │      通知 AIC Weight 就绪
│    │   ├── MmadCompute                     │
│    │   └── SetFlag<M_MTE1>                 │  SetVToMTE2()
│    └── end for                             │    └── SetFlag<V_MTE2>(idx)
│                                            │
│  SetMTE1ToMTE2()                           │
│  SetScaleMTE1ToMTE2()                      │
│                                            │
│  ★ SetAicToAiv<PIPE_MTE1>(FLAG9) ────────→│
│      通知 AIV 可继续搬运                    │
│                                            │
│  (下一轮 K 迭代)                            │  (下一轮 K 迭代)
```

**关键同步点标注**：

- **★1** `WaitAivToAic<PIPE_MTE1>(FLAG8)`：AIC 等待 AIV 完成 Weight 反量化。这是 K 循环的核心同步点，确保 AIC 做 MMA 时 L1 上的 Weight 数据有效。
- **★2** `SetAivToAic<PIPE_MTE3>(FLAG8)`：AIV 通知 AIC Weight 反量化完成。
- **★3** `WaitAicToAiv<PIPE_MTE3>(FLAG9)`：AIV 等待 AIC 完成 MMA。确保 AIV 覆盖 L1 上的 Weight 前，AIC 已经消费完上一块 Weight。
- **★4** `SetAicToAiv<PIPE_MTE1>(FLAG9)`：AIC 通知 AIV MMA 完成。

### 7.3 BasicBlock 边界阶段

BasicBlock 边界（`cvLoopIdx_ > 0` 时）需要处理上一个 BasicBlock 的 Fixpipe 输出和 Routing 后处理：

```
时间 → (BasicBlock 边界)

AIC (Cube Core)                              AIV (Vector Core)
│                                            │
│  (进入新 BasicBlock 的 IterateNzNkWithKAic) │  (进入新 BasicBlock 的 IterateNzNkWithAiv)
│                                            │
│  if (cvLoopIdx_ > 0):                      │  Pass1: 预加载前 2 块 Weight
│                                            │    VecComputeNzNkWithStartLimit(0, 2*kbL1Size)
│                                            │    (含 FLAG8/FLAG9 跨核同步)
│                                            │
│    ★ WaitAivToAic<PIPE_FIX>(FLAG4) ◄───────┤  PhaseB:
│        等待 AIV 取走上一个 BB 的 UB 结果      │    WaitSToMTE2(rlLoopIdx_)
│                                            │    WaitFrVToMTE2(rlLoopIdx_)
│    GetTensorC(ubOutputF32Buffer_)           │    CopyRowIndexLogitsGmToUb()
│      └── Fixpipe: L0C → UB (F32)           │    SetMte2ToS(rlLoopIdx_)
│                                            │
│    ★ SetAicToAiv<PIPE_FIX>(FLAG3) ────────→│    ★ SetAivToAic<PIPE_MTE3>(FLAG4) ──→
│        通知 AIV UB 结果已就绪                │        通知 AIC 可以 Fixpipe
│                                            │
│  (继续当前 BB 的 K 循环)                     │    ★ WaitAicToAiv<PIPE_V>(FLAG3) ◄──
│    for kbL1Offset:                         │        等待 AIC Fixpipe 完成
│      ... (含 FLAG8/FLAG9 跨核同步)           │
│                                            │    MulLogits()
│                                            │      └── VF: y *= 64.0 * logits
│                                            │    SetFrToMTE2(rlLoopIdx_)
│                                            │    WaitMte2ToS(rlLoopIdx_)
│                                            │    RoutingYToGm()
│                                            │      └── MTE3: AtomicAdd Scatter→GM
│                                            │    SetSToMTE2(rlLoopIdx_)
│                                            │
│                                            │  Pass2: 继续加载剩余 Weight
│                                            │    VecComputeNzNkWithStartLimit(2*kbL1Size, kSize)
│                                            │    (含 FLAG8/FLAG9 跨核同步)
```

**关键同步点标注**：

- **★5** `WaitAivToAic<PIPE_FIX>(FLAG4)`：AIC 等待 AIV 取走上一个 BasicBlock 的 UB 结果。确保 Fixpipe 覆盖 UB 前，AIV 已经消费完上一轮结果。
- **★6** `SetAicToAiv<PIPE_FIX>(FLAG3)`：AIC 通知 AIV Fixpipe 完成，UB 上的 MMA 结果已就绪。
- **★7** `SetAivToAic<PIPE_MTE3>(FLAG4)`：AIV 通知 AIC 已取走 UB 结果，可以 Fixpipe。
- **★8** `WaitAicToAiv<PIPE_V>(FLAG3)`：AIV 等待 AIC Fixpipe 完成。

### 7.4 结束阶段

```
时间 → (最后一个 BasicBlock 结束后)

AIC (Cube Core)                              AIV (Vector Core)
│                                            │
│  End():                                    │  End():
│    if (cvLoopIdx_ > 0):                    │    if (cvLoopIdx_ > 0):
│      ★ WaitAivToAic<PIPE_FIX>(FLAG4) ◄─────┤      ★ SetAivToAic<PIPE_MTE3>(FLAG4) ──→
│      GetTensorC(ubOutputF32Buffer_)         │      ★ WaitAicToAiv<PIPE_V>(FLAG3) ◄──
│      ★ SetAicToAiv<PIPE_FIX>(FLAG3) ──────→│      MulLogits()
│                                            │      WaitMte2ToS()
│    EndSync():                               │      RoutingYToGm()
│      for i in [0,1]:                       │      SetSToMTE2()
│        WaitFlag<MTE1_MTE2>(5+i)            │
│        WaitFlag<MTE1_MTE2>(3+i)            │    ★ WaitAicToAiv<PIPE_MTE3>(FLAG9)
│        WaitFlag<M_MTE1>(3+i)               │    ★ WaitAicToAiv<PIPE_MTE3>(FLAG9)
│                                            │      (等待 AIC 完成最后两次搬运)
│                                            │
│                                            │    End():
│                                            │      for idx in [0..3]:
│                                            │        WaitFlag<MTE3_V>(idx)
│                                            │      for idx in [0..ubMte2BufferNum-1]:
│                                            │        WaitFlag<V_MTE2>(idx)
```

**说明**：结束阶段 AIV 额外等待两次 `FLAG9`，确保 AIC 的最后两次 MTE1 搬运（A 矩阵和 Scale）完全结束。然后 AIV 等待所有 MTE3→V 和 V→MTE2 的 Event，确保所有流水线排空。

### 7.5 多 BasicBlock 流水线全景

```
时间 → (3 个 BasicBlock 的完整流水线)

BB0:                              BB1:                              BB2:
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ AIC:                 │ │ AIC:                 │ │ AIC:                 │
│  K循环0: CopyA+MMA   │ │  ★ Fixpipe BB0→UB   │ │  ★ Fixpipe BB1→UB   │
│  K循环1: CopyA+MMA   │ │  K循环0: CopyA+MMA   │ │  K循环0: CopyA+MMA   │
│  K循环2: CopyA+MMA   │ │  K循环1: CopyA+MMA   │ │  K循环1: CopyA+MMA   │
│  K循环3: CopyA+MMA   │ │  K循环2: CopyA+MMA   │ │  K循环2: CopyA+MMA   │
│                      │ │  K循环3: CopyA+MMA   │ │  K循环3: CopyA+MMA   │
├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤
│ AIV:                 │ │ AIV:                 │ │ AIV:                 │
│  Pass1: WQ_K0, WQ_K1 │ │  Pass1: WQ_K0, WQ_K1 │ │  Pass1: WQ_K0, WQ_K1 │
│  (无 PhaseB)          │ │  PhaseB:             │ │  PhaseB:             │
│  Pass2: WQ_K2, WQ_K3 │ │    ★ MulLogits+Route │ │    ★ MulLogits+Route │
│                      │ │    (处理 BB0 结果)     │ │    (处理 BB1 结果)     │
│                      │ │  Pass2: WQ_K2, WQ_K3 │ │  Pass2: WQ_K2, WQ_K3 │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
        │                          │                          │
        └── BB0 无 PhaseB ─────────┘── BB1 PhaseB 处理 BB0 ──┘── BB2 PhaseB 处理 BB1

End 阶段: AIC Fixpipe BB2→UB, AIV MulLogits+Route 处理 BB2 结果
```

**流水线重叠分析**：

- **BB0**：AIV 只做 Weight 反量化（Pass1 + Pass2），无 PhaseB（因为还没有 MMA 结果需要后处理）。
- **BB1**：AIV 的 Pass1（预加载 BB1 的 Weight）与 PhaseB（处理 BB0 的 MMA 结果）重叠。AIC 的 Fixpipe（BB0 结果输出）与 K 循环（BB1 的 MMA）重叠。
- **BB2**：同上，PhaseB 处理 BB1 的结果。
- **End**：处理最后一个 BB（BB2）的 PhaseB。

---

## 8. 同步点标注汇总

### 8.1 跨核同步点

| 编号 | 同步点 | 方向 | Pipe | 触发条件 | 语义 |
|------|--------|------|------|----------|------|
| ★1 | `WaitAivToAic<PIPE_MTE1>(FLAG8)` | AIV→AIC | MTE1 | 每次 K 循环 | AIC 等待 AIV Weight 反量化完成 |
| ★2 | `SetAivToAic<PIPE_MTE3>(FLAG8)` | AIV→AIC | MTE3 | 每次 K 循环 | AIV 通知 AIC Weight 反量化完成 |
| ★3 | `WaitAicToAiv<PIPE_MTE3>(FLAG9)` | AIC→AIV | MTE3 | 每次 K 循环 | AIV 等待 AIC MMA 完成 |
| ★4 | `SetAicToAiv<PIPE_MTE1>(FLAG9)` | AIC→AIV | MTE1 | 每次 K 循环 | AIC 通知 AIV MMA 完成 |
| ★5 | `WaitAivToAic<PIPE_FIX>(FLAG4)` | AIV→AIC | FIX | BB 边界 | AIC 等待 AIV 取走 UB 结果 |
| ★6 | `SetAicToAiv<PIPE_FIX>(FLAG3)` | AIC→AIV | FIX | BB 边界 | AIC 通知 AIV Fixpipe 完成 |
| ★7 | `SetAivToAic<PIPE_MTE3>(FLAG4)` | AIV→AIC | MTE3 | BB 边界 | AIV 通知 AIC 可以 Fixpipe |
| ★8 | `WaitAicToAiv<PIPE_V>(FLAG3)` | AIC→AIV | V | BB 边界 | AIV 等待 AIC Fixpipe 完成 |
| ★9 | `SetAicToAiv<PIPE_MTE1>(FLAG9)` ×2 | AIC→AIV | MTE1 | Init | AIC 通知 AIV 初始化完成 |
| ★10 | `WaitAicToAiv<PIPE_MTE3>(FLAG9)` ×2 | AIC→AIV | MTE3 | Init / End | AIV 等待 AIC 初始化/结束 |

### 8.2 核内同步点（AIC 侧）

| 编号 | 同步点 | 方向 | 缓冲 | 语义 |
|------|--------|------|------|------|
| E1 | `WaitFlag<MTE1_MTE2>(5+i)` | MTE1→MTE2 | 2 | 等待 Scale MTE1→MTE2 搬运完成 |
| E2 | `SetFlag<MTE1_MTE2>(5+i)` | MTE1→MTE2 | 2 | 通知 Scale MTE1→MTE2 搬运完成 |
| E3 | `WaitFlag<MTE1_MTE2>(3+i)` | MTE1→MTE2 | 2 | 等待 A 矩阵 MTE1→MTE2 搬运完成 |
| E4 | `SetFlag<MTE1_MTE2>(3+i)` | MTE1→MTE2 | 2 | 通知 A 矩阵 MTE1→MTE2 搬运完成 |
| E5 | `WaitFlag<M_MTE1>(3+i)` | MMA→MTE1 | 2 | 等待 MMA 释放 L0A/L0B |
| E6 | `SetFlag<M_MTE1>(3+i)` | MMA→MTE1 | 2 | 通知 MTE1 L0A/L0B 可覆盖 |
| E7 | `SetFlag<MTE1_M>(3)` | MTE1→MMA | 1 | 通知 MMA L0A/L0B 加载完成 |
| E8 | `WaitFlag<MTE1_M>(3)` | MTE1→MMA | 1 | 等待 MTE1 加载 L0A/L0B 完成 |
| E9 | `SetFlag<MTE2_MTE1>(3)` | MTE2→MTE1 | 1 | 通知 MTE1 MTE2 完成 |
| E10 | `WaitFlag<MTE2_MTE1>(3)` | MTE2→MTE1 | 1 | 等待 MTE2 完成 |

### 8.3 核内同步点（AIV 侧）

| 编号 | 同步点 | 方向 | 缓冲 | 语义 |
|------|--------|------|------|------|
| V1 | `WaitFlag<V_MTE2>(0+i)` | VF→MTE2 | 2/4 | 等待 VF 释放 weightLowBit buffer |
| V2 | `SetFlag<V_MTE2>(0+i)` | VF→MTE2 | 2/4 | 通知 MTE2 weightLowBit buffer 可覆盖 |
| V3 | `SetFlag<MTE2_V>(0)` | MTE2→VF | 1 | 通知 VF weightLowBit 搬运完成 |
| V4 | `WaitFlag<MTE2_V>(0)` | MTE2→VF | 1 | 等待 MTE2 搬运 weightLowBit 完成 |
| V5 | `WaitFlag<MTE3_V>(0+i)` | MTE3→VF | 4 | 等待 MTE3 释放 weightHighBit buffer |
| V6 | `SetFlag<MTE3_V>(0+i)` | MTE3→VF | 4 | 通知 VF weightHighBit buffer 可覆盖 |
| V7 | `SetFlag<V_MTE3>(0)` | VF→MTE3 | 1 | 通知 MTE3 VF 反量化完成 |
| V8 | `WaitFlag<V_MTE3>(0)` | VF→MTE3 | 1 | 等待 VF 反量化完成 |
| V9 | `WaitFlag<S_MTE2>(0+i)` | Scalar→MTE2 | 2 | 等待 Scalar 释放 MTE2 buffer |
| V10 | `SetFlag<S_MTE2>(0+i)` | Scalar→MTE2 | 2 | 通知 MTE2 Scalar 处理完成 |
| V11 | `WaitFlag<V_MTE2>(4+i)` | VF→MTE2 | 2 | 等待 VF 释放 Routing buffer |
| V12 | `SetFlag<V_MTE2>(4+i)` | VF→MTE2 | 2 | 通知 MTE2 VF Routing 完成 |
| V13 | `SetFlag<MTE2_S>(0+i)` | MTE2→Scalar | 2 | 通知 Scalar rowIndex/logits 就绪 |
| V14 | `WaitFlag<MTE2_S>(0+i)` | MTE2→Scalar | 2 | 等待 MTE2 搬运 rowIndex/logits 完成 |
| V15 | `SetFlag<MTE2_V>(1+i)` | MTE2→VF | 2 | 通知 VF rowIndex/logits 就绪 |
| V16 | `WaitFlag<MTE2_V>(1+i)` | MTE2→VF | 2 | 等待 MTE2 搬运 rowIndex/logits 完成 |
| V17 | `SetFlag<MTE3_V>(4)` | MTE3→VF | 1 | 通知 VF Routing scatter-add 完成 |
| V18 | `WaitFlag<MTE3_V>(4)` | MTE3→VF | 1 | 等待 MTE3 Routing scatter-add 完成 |

### 8.4 同步信号生命周期

```
FLAG9 (AIC→AIV) 生命周期:
  Init:     SetAicToAiv ×2  ────→  WaitAicToAiv ×2  (AIV 确认 AIC 初始化完成)
  K循环:    SetAicToAiv     ────→  WaitAicToAiv     (AIV 确认 MMA 完成，可继续搬运)
  End:      (无 Set)              WaitAicToAiv ×2  (AIV 等待 AIC 最后两次搬运完成)

FLAG8 (AIV→AIC) 生命周期:
  K循环:    SetAivToAic     ────→  WaitAivToAic     (AIC 确认 Weight 反量化完成)

FLAG4 (AIV→AIC) 生命周期:
  BB边界:   SetAivToAic     ────→  WaitAivToAic     (AIC 确认可以 Fixpipe)
  End:      SetAivToAic     ────→  WaitAivToAic     (最后一个 BB 的 Fixpipe)

FLAG3 (AIC→AIV) 生命周期:
  BB边界:   SetAicToAiv     ────→  WaitAicToAiv     (AIV 确认 Fixpipe 完成)
  End:      SetAicToAiv     ────→  WaitAicToAiv     (最后一个 BB 的 Fixpipe)
```

---

## 附录：关键常量速查

| 常量 | 值 | 定义位置 | 含义 |
|------|-----|----------|------|
| `SYNC_AIV_AIC_FLAG` | 8 | `tool.h` | AIV→AIC 跨核同步信号 |
| `SYNC_AIC_AIV_FLAG` | 9 | `tool.h` | AIC→AIV 跨核同步信号 |
| `SYNC_AIC_FIX_AIV_VF_FLAG` | 3 | `tool.h` | AIC Fixpipe→AIV VF 跨核同步信号 |
| `SYNC_AIV_MTE3_AIC_FIX_FLAG` | 4 | `tool.h` | AIV MTE3→AIC Fixpipe 跨核同步信号 |
| `SYNC_MODE4` | 4 | `tool.h` | 跨核同步模式 |
| `FLAG_ID_MAX` | 16 | `tool.h` | 双 Flag 机制的偏移量 |
| `QUADRUPLE_BUFFER_NUM` | 4 | `tool.h` | 四缓冲数量 |
| `DOUBLE_BUFFER_NUM` | 2 | `tool.h` | 双缓冲数量 |
| `MX_GROUPSIZE` | 32 | `tool.h` | MX 量化 Group Size |
| `MX_SCALE_K_L1_SIZE` | 4096 | `tool.h` | Scale L1 覆盖的 K 方向元素数 |
| `K_ALIGNMENT64` | 64 | `tool.h` | K 轴对齐粒度 |
| `BLOCK_CUBE` | 16 | AscendC | Cube 分形大小 |
| `VECTOR_REG_WIDTH` | 256 | AscendC | 向量寄存器宽度（字节） |
| `L0_BUF_OFFSET_B8` | 32×1024 | `cube_compute.h` | L0A/L0B 双缓冲偏移 |
| `L0C_BUF_SIZE` | 256×256 | `cube_compute.h` | L0C 总容量 |