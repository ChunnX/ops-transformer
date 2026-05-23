# Flash Attention Cube 核同步优化说明

## 背景

Flash Attention 在 Ascend 芯片上的 Cube 路径由 **AIC（矩阵计算核）** 和 **AIV（向量计算核）** 协同完成：

- **AIC**：负责 BMM1（Q×K^T）、BMM2（P×V）矩阵乘，结果通过 Fixpipe 写到 L1 或 UB
- **AIV**：负责 Softmax、Mask、Online Update 等向量操作

两核通过 `WaitCrossCore` / `SetCrossCore` 交换中间结果（P 矩阵）。AIC 核内部通过 `HardEvent` 管理 DMA（MTE）与矩阵单元（Cube）之间的同步。

---

## 关键同步原语

| 原语 | 含义 |
|---|---|
| `buf.Wait<MTE1_MTE2>()` | 等待 MTE1（L1→L0 搬运）完成，再让 MTE2（GM→L1 搬运）使用此 buffer |
| `buf.Set<MTE2_MTE1>()` | MTE2 完成 GM→L1 搬运，通知 MTE1 可以使用 |
| `buf.Wait<MTE2_MTE1>()` | 等待 MTE2 搬运到 L1 完成 |
| `l0c.Set<M_FIX>()` | Cube 矩阵乘完成，通知 Fixpipe 可以将 L0C 搬运到 UB |
| `l0c.Wait<M_FIX>()` | 等待 Fixpipe 将 L0C 结果搬运到 UB 完成 |
| `l0c.Wait<FIX_M>()` | 等待前一轮 Fixpipe 完成并释放 L0C，本轮才可以重用 L0C |
| `buf.WaitCrossCore()` | AIC 等待 AIV 释放跨核 buffer（P 矩阵 UB）的占用 |
| `buf.SetCrossCore()` | AIC 通知 AIV：跨核 buffer 中的数据已就绪 |

---

## 优化点详解

### 优化 1：`IterateBmm1NdL0Split` — L0C 等待前移 + WaitCrossCore 提前

**适用场景**：`D > 128`，L1 全载，L0 切 D，双 Buffer（最常见路径）

#### 基线执行序

```
CopyQ → CopyK → Wait(Q,K) → Get L0C → Wait<FIX_M>
→ Matmul → Set<M_FIX> → Wait<M_FIX>
→ WaitCrossCore → Fixpipe → SetCrossCore
```

**问题**：
1. `Get L0C + Wait<FIX_M>` 在 DMA 完成后才开始等，L0C 空闲时间浪费在 DMA 之后
2. `WaitCrossCore`（等 AIV 释放 UB）在 Fixpipe 完成之后，即在 AIC 已空闲时才等，浪费等待时间

#### 优化后执行序

```
Get L0C → Wait<FIX_M>         ← 提前到函数入口，与 Q/K DMA 重叠
→ CopyQ（MTE2）
→ CopyK（MTE2）
→ WaitCrossCore                ← 提前到 K DMA 运行期间，等待 AIV 与 K 搬运重叠
→ Wait(Q,K) → Matmul
→ Set<M_FIX> → Wait<M_FIX>
→ Fixpipe → SetCrossCore
```

**收益**：
- **L0C 等待与 DMA 重叠**：上一轮 Fixpipe 对 L0C 的释放延迟被 Q/K DMA 时间掩盖
- **WaitCrossCore 与 K DMA 重叠**：AIV 处理向量的延迟被 K 搬运时间掩盖，AIC 无需在 Fixpipe 后再额外等待 AIV

---

### 优化 2：`IterateBmm1DnSplitK` / `IterateBmm1Dn` — WaitCrossCore 夹入 Set/Wait 之间

**适用场景**：`D > 256`（L1 切 K）或 `useDn=true, 128 < D <= 256`

#### 基线执行序

```
... Matmul ...
→ Set<M_FIX>
→ Wait<M_FIX>      ← Fixpipe 硬件运行，CPU 在此阻塞等待
→ WaitCrossCore    ← Fixpipe 完成后，再阻塞等 AIV
→ Fixpipe
```

**问题**：`WaitCrossCore` 在 `Wait<M_FIX>` 之后，两次阻塞**串行**，总延迟 = Fixpipe 时间 + AIV 响应时间

#### 优化后执行序

```
... Matmul ...
→ Set<M_FIX>       ← 触发 Fixpipe 硬件开始执行
→ WaitCrossCore    ← Fixpipe 硬件运行期间，同步等待 AIV 释放 UB
→ Wait<M_FIX>      ← Fixpipe 很可能已完成
→ Fixpipe
```

**收益**：将 `WaitCrossCore` 夹入 `Set<M_FIX>` 和 `Wait<M_FIX>` 之间，**两次等待并行化**：
- CPU 等 AIV 的时间 与 Fixpipe 硬件执行时间 重叠
- 理想情况下，若 AIV 响应时间 ≤ Fixpipe 时间，`WaitCrossCore` 的延迟完全被隐藏
- 总延迟 ≈ `max(Fixpipe 时间, AIV 响应时间)`（原来是两者之和）

---

### 优化 3：`IterateBmm2l0Split` — WaitCrossCore 移至 V DMA 期间

**适用场景**：BMM2（P×V），`dVTemplateType <= 256`，`bmm2Write2Ub=true`

#### 基线执行序

```
WaitCrossCore（等 P 矩阵）
→ CopyV（MTE2）
→ Matmul
→ Set<M_FIX> → Wait<M_FIX>
→ WaitCrossCore（等 UB V2 释放）← 在 Fixpipe 完成后
→ Fixpipe → SetCrossCore
```

#### 优化后执行序

```
WaitCrossCore（等 P 矩阵）
→ CopyV（MTE2）
→ WaitCrossCore（等 UB V2 释放）← 提前到 V DMA 运行期间
→ Matmul
→ Set<M_FIX> → Wait<M_FIX>
→ Fixpipe → SetCrossCore
```

**收益**：等待 AIV 释放上一轮输出 UB（V2 双 Buffer 另一半）的时间，被 V 从 GM 搬运到 L1 的 DMA 时间掩盖。V 的搬运通常需要数百个周期，足以完全覆盖 AIV 的响应延迟。

---

## 优化效果汇总

| 函数 | 优化手段 | 隐藏的延迟来源 |
|---|---|---|
| `IterateBmm1NdL0Split` | L0C `Wait<FIX_M>` 前移至函数入口 | 前一轮 Fixpipe 对 L0C 的释放延迟 |
| `IterateBmm1NdL0Split` | `WaitCrossCore` 移至 K DMA 期间 | AIV 向量处理延迟 |
| `IterateBmm1DnSplitK` | `WaitCrossCore` 夹入 `Set/Wait<M_FIX>` 之间 | AIV 向量处理延迟 |
| `IterateBmm1Dn` | `WaitCrossCore` 夹入 `Set/Wait<M_FIX>` 之间 | AIV 向量处理延迟 |
| `IterateBmm2l0Split` | `WaitCrossCore` 移至 V DMA 期间 | AIV 处理上一轮输出的延迟 |

---

## 关键原则

> **等待（Wait）应当夹在两个独立的硬件操作启动之间，而不是放在所有操作完成之后。**

CPU 执行同步原语（`WaitCrossCore`、`Wait<HardEvent>`）时，硬件单元（MTE、Cube、Fixpipe）仍在并行工作。将 CPU 等待时间与硬件执行时间重叠，是提升 AIC 核利用率的核心方法。

---

## 调试标记（set_mark）

优化版本在关键阶段加入了 `set_mark` 打点，用于 profiling 分析：

| 标记区间 | 含义 |
|---|---|
| `0x40010000 ~ 0x400100FF` | BMM1 CopyQ 阶段 |
| `0x40020000 ~ 0x400200FF` | BMM1 CopyK 阶段 |
| `0x40030000 ~ 0x400300FF` | BMM1 Matmul 阶段 |
| `0x40040000 ~ 0x400400FF` | BMM1 Fixpipe 阶段 |
| `0x60010000 ~ 0x600100FF` | BMM2 CopyV 阶段 |
| `0x60020000 ~ 0x600200FF` | BMM2 Matmul 阶段 |
| `0x60030000 ~ 0x600300FF` | BMM2 Fixpipe 阶段 |
