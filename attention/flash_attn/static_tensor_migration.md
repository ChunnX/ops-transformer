# Flash Attention 静态 Tensor 编程迁移说明

> 版本：`ops-transformer_caculate`  
> 改动范围：`attention/flash_attn/op_kernel/arch35/`  
> 对应原版（基线）：`ops-transformer` (master 分支)

---

## 目录

1. [背景：为什么要做这次迁移](#1-背景)
2. [基础概念：UB、TBuf/TQue、LocalTensor 三者的关系](#2-基础概念)
3. [Ping-Pong 缓冲区——从零讲解](#3-ping-pong-缓冲区)
4. [事件（Event）同步机制](#4-事件同步机制)
   - 4.1 为什么需要事件同步
   - 4.2 动态 vs 静态事件 ID
   - 4.3 全量静态事件 ID 对照表（VEC 侧 + Cube 侧）
5. [具体修改内容（修改前后对比）](#5-具体修改内容)
   - 5.1 成员变量类型替换
   - 5.2 `InitBuffers` 重写（VEC 侧）
   - 5.3 `AllocEventID / FreeEventID` 静态化（VEC 侧）
   - 5.4 Vec2 内部事件 ID 静态化
   - 5.5 新建 `fa_vec1_vf_static.h`（vselrIndexesBuf 参数替换）
   - 5.6 Cube 侧（C 核）静态化修改
   - 5.7 `fa_static_buffer.h`（消除 common buffer 文件依赖）
6. [Ping-Pong 实现细节](#6-ping-pong-实现细节)
7. [性能影响分析](#7-性能影响分析)

---

## 1. 背景

### 1.1 问题起点

原始 `flash_attn` 算子大量使用了昇腾 CANN 的 **TPipe 框架接口**：

| 接口 | 功能 |
|------|------|
| `TPipe` | 统一管道管理器，负责内存分配和事件ID发放 |
| `TBuf<>` | 通过 TPipe 动态分配的 UB 缓冲区句柄 |
| `TQue<>` | 通过 TPipe 动态分配的 UB 队列句柄 |
| `AllocEventID` | 运行时动态分配事件ID |
| `FetchEventID` | 每次调用时从 TPipe 拿一个事件ID |

这些接口在**动态 Tensor 编程模型**下工作正常，但有以下问题：
- 运行时需要查表、原子计数，引入不可忽视的 CPU 开销
- 不满足**静态 Tensor 编程模型**规范（昇腾极限性能优化要求）
- 在某些特殊路径下（如无 TPipe 初始化时）会触发空指针崩溃

### 1.2 改造目标

用**静态 Tensor 编程模型**全面替代：
- `TBuf<>` → `LocalTensor<uint8_t>`（直接管理内存地址）
- `TQue<>` → `LocalTensor<T>`（直接的数组式访问）
- 动态事件 ID → 编译期 `static constexpr uint32_t` 常量
- `tPipe->InitBuffer()` → `LocalMemAllocator::Alloc()`（编译期固定偏移，中间态）
- `LocalMemAllocator::Alloc()` → `LocalTensor<uint8_t>(pos, offset, size)`（直接构造，终极态）

---

## 2. 基础概念

### 2.1 UB（Unified Buffer）是什么

昇腾 AI Core 内部有一块高速片上存储，称为 **UB（Unified Buffer）**，类似于 GPU 的 shared memory。所有向量计算（VEC 单元）的中间结果都必须放在 UB 里。

```
AI Core 内部存储层次（从快到慢）：
  寄存器
  UB（Unified Buffer）  ←── 所有中间计算结果放这里
  L1 Buffer            ←── 矩阵乘法输入暂存
  L2 / DDR（全局内存）
```

**UB 是有限资源**（典型大小 256KB），所有缓冲区都在这块固定内存里划分。

### 2.2 TBuf / TQue 是什么（旧方式）

在动态 Tensor 编程模型中，开发者通过 `TPipe` 这个管理器来"申请"一块 UB 内存：

```cpp
// 旧方式：通过 TPipe 动态申请
TPipe *tPipe;
TBuf<> softmaxSumBuf[3];   // 声明句柄，但还没有内存地址
TQue<QuePosition::VECOUT, 1> stage1OutQue[2];

// 初始化时，TPipe 负责分配内存地址
tPipe->InitBuffer(softmaxSumBuf[0], 256);  // 给 softmaxSumBuf[0] 分配 256 字节
tPipe->InitBuffer(softmaxSumBuf[1], 256);
tPipe->InitBuffer(softmaxSumBuf[2], 256);

// 使用时需要取出 LocalTensor
LocalTensor<float> sumTensor = softmaxSumBuf[0].template Get<float>();
```

**TBuf 的问题**：
- `tPipe->InitBuffer` 运行时动态维护一个偏移量，有额外开销
- `Get<T>()` 每次调用也有间接层
- 一旦 TPipe 未初始化，直接崩溃

### 2.3 LocalTensor 是什么（新方式）

`LocalTensor<T>` 本质上就是一个**指向 UB 某地址的指针 + 长度**：

```cpp
template <typename T>
struct LocalTensor {
    uint8_t *address_;   // UB 内的起始地址（编译期或运行时确定）
    uint32_t size_;      // 大小（字节）
};
```

用 `LocalMemAllocator` 分配时，偏移量在**编译期**就确定（模板参数），运行时零开销：

```cpp
// 中间方式：LocalMemAllocator 编译期固定偏移
LocalMemAllocator<Hardware::UB> allocator;
LocalTensor<float> sumBuf0 = allocator.Alloc<TPosition::VECIN, float, 64>();
// address = UB基址 + 固定偏移(编译期), size = 64*4 = 256字节
```

终极方式是直接用 `LocalTensor` 构造函数，彻底绕过分配器：

```cpp
// 终极方式：LocalTensor 构造函数直接指定绝对字节偏移
constexpr uint32_t OFF = 49152U;  // 编译期常量
LocalTensor<float> sumBuf0 = LocalTensor<uint8_t>(TPosition::VECIN, OFF, 256U)
                                  .template ReinterpretCast<float>();
// 无任何运行时状态，地址 = UB基址 + OFF（硬编码到指令流）
```

**三级方式对比：**

| | TBuf（旧） | LocalMemAllocator（中间）| LocalTensor 直接构造（终极）|
|--|--|--|--|
| 内存地址确定时机 | 运行时（InitBuffer 调用）| 编译期（Alloc 模板参数）| 编译期（constexpr 偏移）|
| 需要 TPipe | 是 | 否 | 否 |
| 需要分配器状态 | 是 | 是（内部偏移指针）| 否 |
| 访问开销 | 有（Get<T>() 间接层）| 无 | 无 |
| 是否允许使用 | 已禁用 | 中间态，可接受 | 推荐（最终形态）|

---

## 3. Ping-Pong 缓冲区

### 3.1 什么是 Ping-Pong？

Ping-Pong（乒乓缓冲）是**流水线计算**中的一种经典技术，用于让"写入"和"读取"两个操作**同时进行**，互不干扰。

用一个生活中的例子来理解：

> 想象一个工厂的**传送带有两个货箱（Ping 和 Pong）**：
> - **工人 A（生产者）**：往货箱里装零件
> - **工人 B（消费者）**：从货箱里取零件
>
> 如果只有一个货箱：A 装的时候 B 要等，B 取的时候 A 要等 → **串行，效率低**
>
> 如果有两个货箱交替使用：
> - 时刻 1：A 往 Ping 里装 → B 从 Pong 里取（**同时进行！**）
> - 时刻 2：A 往 Pong 里装 → B 从 Ping 里取
>
> → **并行，效率翻倍**

在 Flash Attention 中：
- **生产者** = VEC 单元（执行 softmax 等向量计算，写入 `stage1OutBuf`）
- **消费者** = MTE3（DMA 引擎，把 `stage1OutBuf` 的数据搬到 L1）

### 3.2 Flash Attention 中的三级流水线

Flash Attention 的主循环（S2 维度循环）有三个重叠的阶段：

```
时间轴 →
─────────────────────────────────────────────────────────────
CUBE单元  │ Mm1[N]  │ Mm1[N+1] │ Mm1[N+2] │ Mm2[N-1] │ ...
VEC单元   │ Vec1[N] │ Vec1[N+1]│ Vec2[N-1]│ Vec1[N+2]│ ...
MTE3      │  Copy[N-1]  │  Copy[N]  │  Copy[N+1]  │ ...
─────────────────────────────────────────────────────────────
```

- **CUBE** 做矩阵乘法（Mm1/Mm2）
- **VEC** 做 softmax 和 flash update（Vec1/Vec2）
- **MTE3** 把 Vec1 的结果从 UB 搬到 L1

要让这三个单元**真正并行**，就需要 Ping-Pong：当 VEC 在写 `stage1OutBuf[0]` 时，MTE3 可以同时在读 `stage1OutBuf[1]`。

### 3.3 为什么需要 PRELOAD_N = 2 个任务的超前量？

```cpp
static constexpr uint32_t PRELOAD_N = 2;
```

这意味着系统会超前创建 2 个任务（Mm1 + Vec1），然后才开始执行（Mm2 + Vec2）。这保证了：
- 执行 Task(N) 时，Task(N+1) 和 Task(N+2) 的 Mm1 已经在 CUBE 上排队
- CUBE 和 VEC 始终有活干，没有空闲等待

---

## 4. 事件同步机制

### 4.1 为什么需要事件同步？

CUBE、VEC、MTE 是**三个独立的硬件单元**，并发执行。当它们需要交换数据时，必须用**事件（Event）**来协调：

```
VEC 算完 stage1OutBuf → SetFlag<V_MTE3>(eventId)   ← "我写完了，MTE3 可以读了"
MTE3 开始搬运之前      → WaitFlag<V_MTE3>(eventId)  ← "我在等 VEC 写完"
```

### 4.2 动态 vs 静态事件 ID

**动态方式（旧）**：每次需要事件 ID 时，调用 `FetchEventID` 向 TPipe 动态申请：

```cpp
// 每次 Vec2 被调用都执行这两行 ← 运行时开销！
event_t mte2ToV = static_cast<event_t>(tPipe->FetchEventID(HardEvent::MTE2_V));
event_t vToMte2 = static_cast<event_t>(tPipe->FetchEventID(HardEvent::V_MTE2));
```

`FetchEventID` 内部需要查事件表 + 原子计数器递增，每次约 5~20ns。

**静态方式（新）**：编译期确定所有事件 ID，运行时直接用常量：

```cpp
// 编译期常量，运行时零开销
static constexpr uint32_t FA_MTE2_TO_V_ID    = 2U;
static constexpr uint32_t FA_V_TO_MTE2_ID    = 2U;
static constexpr uint32_t FA_MTE3_TO_MTE2_ID = 2U;

// 使用时
constexpr uint32_t mte2ToV = FA_MTE2_TO_V_ID;   // 编译期替换为字面量 2
constexpr uint32_t vToMte2 = FA_V_TO_MTE2_ID;
```

**注意**：不同硬件事件类型（`MTE2_V` 和 `V_MTE2`）的事件 ID 空间是**独立的**，所以它们可以都取值 `2` 而不冲突。

---

### 4.3 全量静态事件 ID 对照表（VEC 侧 + Cube 侧）

#### 4.3.1 硬件事件类型命名规则

昇腾硬件事件名称格式为 `SOURCE_DEST`，含义是：**SOURCE 单元完成操作，DEST 单元可以继续**。

```
AI Core 数据流方向（Cube 路径）：
  GM → [MTE2] → L1 → [MTE1] → L0A/L0B → [M 单元/CUBE] → L0C → [FixPipe] → UB/L1/GM

  MTE2_MTE1 : MTE2 写完 L1     → MTE1 可以读 L1 (p2c for L1)
  MTE1_MTE2 : MTE1 读完 L1     → MTE2 可以覆写 L1 (c2p for L1) ← 初始化时预置
  MTE1_M    : MTE1 写完 L0A/B  → CUBE 可以读 L0A/B (p2c for L0A/B)
  M_MTE1    : CUBE 读完 L0A/B  → MTE1 可以覆写 L0A/B (c2p for L0A/B) ← 初始化时预置
  M_FIX     : CUBE 写完 L0C    → FixPipe 可以读 L0C (p2c for L0C)
  FIX_M     : FixPipe 读完 L0C → CUBE 可以写 L0C (c2p for L0C) ← 初始化时预置

AI Core 数据流方向（Vector 路径）：
  GM → [MTE2] → UB → [VEC] → UB → [MTE3] → L1/GM

  MTE2_V    : MTE2 写完 UB   → VEC 可以读 UB (p2c)
  V_MTE2    : VEC 读完 UB    → MTE2 可以覆写 UB (c2p)
  V_MTE3    : VEC 写完 UB    → MTE3 可以读 UB (p2c for stage1OutBuf) ← 计算时设置
  MTE3_V    : MTE3 读完 UB   → VEC 可以覆写 UB (c2p for stage1OutBuf) ← 初始化时预置
  MTE3_MTE2 : MTE3 搬运结束  → MTE2 可以继续 (vec2 内部协调)
```

#### 4.3.2 为什么预置 c2p 事件（初始化时 SetFlag）？

c2p 事件的语义是"消费者已处理完，生产者可以重新写入"。在流水线启动之前，所有槽都是空的（消费者还没开始工作），所以需要**预置**这些事件，告诉生产者"槽是空的，你可以立即开始写"。如果不预置，生产者第一次 `WaitFlag<c2p>` 会永久阻塞。

p2c 事件由生产者在每次写完后自然 `SetFlag`，不需要预置。

#### 4.3.3 VEC 侧事件 ID 一览（`flash_attn_block_vec_noquant_gqa.h`）

| 常量名 | 值 | HardEvent 类型 | 方向 | 含义 |
|--------|-----|---------------|------|------|
| `FA_MTE3_TO_V_0` | `0` | `MTE3_V` | c2p | MTE3 读完 stage1OutBuf[0]，VEC 可覆写（**预置**） |
| `FA_MTE3_TO_V_1` | `1` | `MTE3_V` | c2p | MTE3 读完 stage1OutBuf[1]，VEC 可覆写（**预置**） |
| `FA_V_TO_MTE3_0` | `0` | `V_MTE3` | p2c | VEC 写完 stage1OutBuf[0]，MTE3 可读（计算时 SetFlag）|
| `FA_V_TO_MTE3_1` | `1` | `V_MTE3` | p2c | VEC 写完 stage1OutBuf[1]，MTE3 可读（计算时 SetFlag）|
| `FA_MTE2_TO_V_ID` | `2` | `MTE2_V` | p2c | MTE2 写完 UB 局部区域，VEC 可读（Vec2 内部）|
| `FA_V_TO_MTE2_ID` | `2` | `V_MTE2` | c2p | VEC 读完，MTE2 可覆写（Vec2 内部）|
| `FA_MTE3_TO_MTE2_ID` | `2` | `MTE3_MTE2` | - | MTE3 完成，MTE2 可继续（Vec2 内部）|
| `FA_MTE2_TO_V_SINK` | `4` | `MTE2_V` | p2c | MTE2 加载 sink 数据完毕，VEC 可读 |

> **ID 复用说明**：`MTE3_V`、`V_MTE3`、`MTE2_V` 等不同 HardEvent 类型拥有**完全独立的 ID 空间**。因此 `FA_MTE3_TO_V_0 = 0`（MTE3_V 空间中的 ID 0）与 `FA_V_TO_MTE3_0 = 0`（V_MTE3 空间中的 ID 0）互不干扰。同理，Vec2 内部使用的三个 ID=2 事件均来自不同类型，不存在冲突。

#### 4.3.4 Cube 侧事件 ID 一览（`flash_attn_block_cube_noquant_gqa.h`）

各缓冲区均仅使用 **c2p 方向**的事件（初始化时预置，计算过程中由硬件/API 内部管理）。

**L1 缓冲区 — HardEvent::MTE1_MTE2（MTE1 读完 L1，MTE2 可覆写）**

| 常量名 | 值 | 适用场景 | 对应槽位 |
|--------|-----|---------|---------|
| `FA_L1_Q_C2P_0` | `0` | l1QBuffers ping（或 SingleBuffer 唯一槽）| L1Q ping/single |
| `FA_L1_Q_C2P_1` | `1` | l1QBuffers pong（仅 DB 模式，dBaseSize≤256）| L1Q pong |
| `FA_L1_K_C2P_0` | `2` | l1KBuffers ping/single | L1K ping/single |
| `FA_L1_K_C2P_1` | `3` | l1KBuffers pong（DB 模式）| L1K pong |
| `FA_L1_V_C2P_0` | `4` | l1VBuffers ping/single | L1V ping/single |
| `FA_L1_V_C2P_1` | `5` | l1VBuffers pong（DB 模式）| L1V pong |

> L1Q/K/V 三组缓冲区共享同一 HardEvent 类型（`MTE1_MTE2`），所以必须使用不同的 ID 值（0~5）来区分。ID 6、7 预留未用。

**L0A 缓冲区 — HardEvent::M_MTE1（CUBE 读完 L0A，MTE1 可覆写）**

| 常量名 | 值 | 对应槽位 |
|--------|-----|---------|
| `FA_L0A_C2P_0` | `0` | L0A ping（DB 模式，始终为 DB）|
| `FA_L0A_C2P_1` | `1` | L0A pong |

**L0B 缓冲区 — HardEvent::M_MTE1（同类型，ID 空间与 L0A 共享）**

| 常量名 | 值 | 适用场景 | 对应槽位 |
|--------|-----|---------|---------|
| `FA_L0B_C2P_0` | `2` | L0B ping（或 Single）| L0B ping/single |
| `FA_L0B_C2P_1` | `3` | L0B pong（DB 模式）| L0B pong |

> L0A 和 L0B 共享 `M_MTE1` 事件类型，因此 L0B 的 ID 从 2 开始（避免与 L0A 的 0/1 冲突）。

**L0C 缓冲区 — HardEvent::FIX_M（FixPipe 读完 L0C，CUBE 可写入）**

| 常量名 | 值 | 适用场景 | 对应槽位 |
|--------|-----|---------|---------|
| `FA_L0C_C2P_0` | `0` | slot 0（4buff 或 DB 模式）|
| `FA_L0C_C2P_1` | `1` | slot 1 |
| `FA_L0C_C2P_2` | `2` | slot 2（仅 4buff 模式）|
| `FA_L0C_C2P_3` | `3` | slot 3（仅 4buff 模式）|

> L0C 使用独立的 `FIX_M` 事件类型，ID 空间与 L0A/L0B 的 `M_MTE1` 完全隔离，所以从 0 重新起编。

#### 4.3.5 各缓冲区使用的 Buffer 策略类型

缓冲区选择 Single 还是 DB（Double Buffer，乒乓）取决于编译期模板参数：

| 缓冲区 | 策略类型 | 条件 |
|--------|---------|------|
| `l1QBuffers` | `BuffersPolicySingleBuffer` | `dBaseSize > 256`（Q 矩阵宽度大，L1 放不下两份）|
| `l1QBuffers` | `BuffersPolicyDB` | `dBaseSize ≤ 256` |
| `l1KBuffers` | `BuffersPolicySingleBuffer` | `s2BaseSize == 256 && dBaseSize > 128` |
| `l1KBuffers` | `BuffersPolicyDB` | 其他情况 |
| `l1VBuffers` | `BuffersPolicySingleBuffer` | `s2BaseSize == 256 && dBaseSize > 128` |
| `l1VBuffers` | `BuffersPolicyDB` | 其他情况 |
| `mmL0ABuffers` | `BuffersPolicyDB` | **始终** |
| `mmL0BBuffers` | `BuffersPolicySingleBuffer` | `s2BaseSize == 256 && dBaseSize > 128` |
| `mmL0BBuffers` | `BuffersPolicyDB` | 其他情况 |
| `mmL0CBuffers` | `BuffersPolicy4Buff` | 两个方向矩阵均可 4 等分 L0C |
| `mmL0CBuffers` | `BuffersPolicyDB` | 否则 |

---

## 5. 具体修改内容

### 5.1 成员变量类型替换

**位置**：`flash_attn_block_vec_noquant_gqa.h`，类成员变量区域

#### 修改前（master 版本）

```cpp
// ====== 旧版：基于 TPipe 框架 ======
TPipe *tPipe;                                        // 需要外部传入管道对象

TQue<QuePosition::VECOUT, 1> stage1OutQue[2];        // Vec1 输出的 ping-pong 队列
TQue<QuePosition::VECIN,  1> attenMaskInQue[2];      // attenMask 的 ping-pong 队列
TBuf<> stage2OutBuf;                                 // Vec2 输出缓冲区

TBuf<> softmaxMaxBuf[PRELOAD_N + 1];                 // softmax 最大值（3份）
TBuf<> softmaxSumBuf[PRELOAD_N + 1];                 // softmax 累加和（3份）
TBuf<> softmaxExpBuf[PRELOAD_N + 1];                 // softmax exp 结果（3份）
TBuf<> vselrIndexesBuf[4];                           // 量化索引重映射缓冲区
TBuf<> mm2InBuf;                                     // BMM2 输入临时缓冲区
TBuf<> commonTBuf;                                   // Vec1 Nd 路径临时空间

TQue<QuePosition::VECOUT, 1> maxBrdcst;              // max 广播临时缓冲
TQue<QuePosition::VECOUT, 1> sumBrdcst;              // sum 广播临时缓冲
TQue<QuePosition::VECOUT, 1> softmaxLseQueue;        // LSE 输出队列
TQue<QuePosition::VECIN,  1> sinkQue;                // AttentionSink 输入队列

TEventID mte3ToVId[2];  // 动态分配的事件 ID
TEventID vToMte3Id[2];  // 动态分配的事件 ID
```

#### 修改后（tensor 静态版本）

```cpp
// ====== 新版：纯 LocalTensor 静态管理 ======
// 注：无需 TPipe 成员

LocalTensor<INPUT_T> stage1OutBuf[2];                // Vec1 输出 ping-pong（直接内存）
LocalTensor<uint8_t> attenMaskBuf[2];                // attenMask ping-pong（直接内存）
LocalTensor<T>       stage2OutBuf;                   // Vec2 输出缓冲区

LocalTensor<float>   softmaxMaxBuf[PRELOAD_N + 1];   // softmax 最大值（3份）
LocalTensor<float>   softmaxSumBuf[PRELOAD_N + 1];   // softmax 累加和（3份）
LocalTensor<T>       softmaxExpBuf[PRELOAD_N + 1];   // softmax exp 结果（3份）
LocalTensor<uint8_t> vselrIndexesBuf[4];             // 量化索引重映射缓冲区
LocalTensor<T>       mm2InBuf;                       // BMM2 输入临时缓冲区
LocalTensor<uint8_t> commonBuf;                      // Vec1 Nd 路径临时空间

LocalTensor<float>   maxBrdcstBuf;                   // max 广播临时缓冲
LocalTensor<float>   sumBrdcstBuf;                   // sum 广播临时缓冲
LocalTensor<float>   lseBuf;                         // LSE 输出缓冲
LocalTensor<INPUT_T> sinkBuf;                        // AttentionSink 输入缓冲

uint32_t mte3ToVId[2];   // 静态事件 ID（运行时赋值为编译期常量）
uint32_t vToMte3Id[2];   // 静态事件 ID

// 编译期静态事件 ID 常量（无运行时开销）
static constexpr uint32_t FA_MTE3_TO_V_0     = 0U;
static constexpr uint32_t FA_MTE3_TO_V_1     = 1U;
static constexpr uint32_t FA_V_TO_MTE3_0     = 0U;
static constexpr uint32_t FA_V_TO_MTE3_1     = 1U;
static constexpr uint32_t FA_MTE2_TO_V_ID    = 2U;
static constexpr uint32_t FA_V_TO_MTE2_ID    = 2U;
static constexpr uint32_t FA_MTE3_TO_MTE2_ID = 2U;
static constexpr uint32_t FA_MTE2_TO_V_SINK  = 4U;
```

**关键理解**：  
`TBuf<>` 只是一个带"句柄"的包装壳，它的内部还是 `LocalTensor<uint8_t>`；直接使用 `LocalTensor` 去掉了这层壳的运行时开销，让编译器能更好地优化。

---

### 5.2 `InitBuffers` 重写

**位置**：`flash_attn_block_vec_noquant_gqa.h` → `InitBuffers()` 函数

#### 修改前（SoftmaxInitBuffer + InitBuffers）

```cpp
// 旧版：全部通过 tPipe->InitBuffer 动态分配
__aicore__ inline void SoftmaxInitBuffer()
{
    tPipe->InitBuffer(softmaxSumBuf[0], 256);  // 运行时动态分配，维护内部偏移量
    tPipe->InitBuffer(softmaxSumBuf[1], 256);
    tPipe->InitBuffer(softmaxSumBuf[2], 256);
    tPipe->InitBuffer(softmaxMaxBuf[0], 256);
    tPipe->InitBuffer(softmaxMaxBuf[1], 256);
    tPipe->InitBuffer(softmaxMaxBuf[2], 256);
    tPipe->InitBuffer(softmaxExpBuf[0], 256);
    tPipe->InitBuffer(softmaxExpBuf[1], 256);
    tPipe->InitBuffer(softmaxExpBuf[2], 256);
    tPipe->InitBuffer(maxBrdcst, 1, 2048);
    tPipe->InitBuffer(sumBrdcst, 1, 2048);
}

// 接收外部共享缓冲区（由 kernel 级别的 TPipe 统一分配后传入）
__aicore__ inline void InitBuffers(TQue<QuePosition::VECOUT, 1> SharedBuffer1[],
                                   TQue<QuePosition::VECIN,  1> SharedBuffer2[],
                                   TBuf<> &SharedBuffer3)
{
    if constexpr (!bmm2Write2Ub) {
        tPipe->InitBuffer(mm2InBuf, 32768);
    }
    tPipe->InitBuffer(SharedBuffer2[0], 1, 8192);   // attenMask ping
    tPipe->InitBuffer(SharedBuffer2[1], 1, 8192);   // attenMask pong
    tPipe->InitBuffer(SharedBuffer3,    32768);     // stage2 输出
    tPipe->InitBuffer(SharedBuffer1[0], 1, 33024);  // stage1 输出 ping
    tPipe->InitBuffer(SharedBuffer1[1], 1, 33024);  // stage1 输出 pong
    SoftmaxInitBuffer();
    if constexpr (!useDn) {
        tPipe->InitBuffer(commonTBuf, 512);
    }
    // 关联共享缓冲区到成员变量
    stage1OutQue[0]   = SharedBuffer1[0];
    stage1OutQue[1]   = SharedBuffer1[1];
    attenMaskInQue[0] = SharedBuffer2[0];
    attenMaskInQue[1] = SharedBuffer2[1];
    stage2OutBuf      = SharedBuffer3;
}
```

#### 修改后（最终版：完全消除 LocalMemAllocator）

```cpp
// 最终版：SoftmaxInitBuffer 接受基础偏移，所有 LocalTensor 直接构造
__aicore__ inline void SoftmaxInitBuffer(uint32_t off_sm)
{
    // 11 个缓冲区紧密排列：Sum×3 + Max×3 + Exp×3（各 256 bytes）
    //                      + maxBrdcst + sumBrdcst（各 2048 bytes）= 6400 bytes 总计
    softmaxSumBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm,          256U).template ReinterpretCast<float>();
    softmaxSumBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 256U,   256U).template ReinterpretCast<float>();
    softmaxSumBuf[2] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 512U,   256U).template ReinterpretCast<float>();
    softmaxMaxBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 768U,   256U).template ReinterpretCast<float>();
    softmaxMaxBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 1024U,  256U).template ReinterpretCast<float>();
    softmaxMaxBuf[2] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 1280U,  256U).template ReinterpretCast<float>();
    softmaxExpBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 1536U,  256U).template ReinterpretCast<T>();
    softmaxExpBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 1792U,  256U).template ReinterpretCast<T>();
    softmaxExpBuf[2] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 2048U,  256U).template ReinterpretCast<T>();
    maxBrdcstBuf     = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 2304U, 2048U).template ReinterpretCast<float>();
    sumBrdcstBuf     = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 4352U, 2048U).template ReinterpretCast<float>();
}

// 最终版 InitBuffers：无任何 LocalMemAllocator，全部使用 LocalTensor 构造函数
__aicore__ inline void InitBuffers()
{
    // 所有偏移为编译期 constexpr，运行时零开销
    constexpr uint32_t mm1Sz = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    constexpr uint32_t mm2Sz = bmm2Write2Ub ?
        (mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T)) : 0U;

    // mm2InBuf：仅在 !bmm2Write2Ub 时需要，紧跟 bmm1/bmm2 区末尾
    if constexpr (!bmm2Write2Ub) {
        mm2InBuf = LocalTensor<uint8_t>(TPosition::VECIN,
                       mm1Sz * 2U + mm2Sz * 2U, 32768U).template ReinterpretCast<T>();
    }

    // vec block 共享区基址 = bmm1区 + bmm2区 + mm2In区
    constexpr uint32_t mm2InBufSz = bmm2Write2Ub ? 0U : 32768U;
    constexpr uint32_t BASE = mm1Sz * 2U + mm2Sz * 2U + mm2InBufSz;

    // attenMask ping/pong（各 8192 bytes）
    attenMaskBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, BASE,          8192U);
    attenMaskBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 8192U,  8192U);
    // stage2 输出（32768 bytes）
    stage2OutBuf    = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 16384U, 32768U)
                          .template ReinterpretCast<T>();
    // stage1 输出 ping/pong（各 33024 bytes）← Vec1 写 / MTE3 读
    stage1OutBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 49152U, 33024U)
                          .template ReinterpretCast<INPUT_T>();
    stage1OutBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 82176U, 33024U)
                          .template ReinterpretCast<INPUT_T>();

    // softmax 中间值（6400 bytes，3-slot 旋转）
    SoftmaxInitBuffer(BASE + 115200U);

    // commonBuf（512 bytes，编译期条件）
    if constexpr (!useDn) {
        commonBuf = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 121600U, 512U);
    }

    // lseBuf / sinkBuf：地址编译期固定，是否初始化为运行时条件
    constexpr uint32_t commonBufSz = useDn ? 0U : 512U;
    constexpr uint32_t OFF_LSE = BASE + 121600U + commonBufSz;
    constexpr uint32_t LSE_BUF_SZ = (mBaseSize >> 1U) * 8U * sizeof(float);
    if (constInfo.isSoftmaxLseEnable) {
        lseBuf = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSE,
                     LSE_BUF_SZ).template ReinterpretCast<float>();
    }
    if (constInfo.learnableSinkFlag) {
        sinkBuf = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSE + LSE_BUF_SZ,
                      256U).template ReinterpretCast<INPUT_T>();
    }
}
```

**关键改变说明**：

1. **彻底消除 LocalMemAllocator** — 旧版（中间态）使用 `(void)allocator.Alloc<...>()` 推进偏移指针；新版（终极态）直接用 `constexpr BASE` 计算所有偏移，无需任何分配器状态。

2. **SoftmaxInitBuffer 参数变化** — 从接受 `LocalMemAllocator &allocator` 改为接受 `uint32_t off_sm`（基础偏移），各缓冲区在此基础上叠加固定增量，完全无状态。

3. **ReinterpretCast 用法不变** — 先以 `uint8_t` 描述字节大小，再强制转换成实际类型，语义与旧版完全等价。

4. **UB 布局完全在编译期确定** — `BASE`、各偏移量均为 `constexpr`，编译器可将其内联为字面量，无运行时计算。

---

### 5.3 `AllocEventID / FreeEventID` 静态化

**位置**：`flash_attn_block_vec_noquant_gqa.h`

#### 修改前

```cpp
// 旧版：运行时动态申请事件 ID
__aicore__ inline void AllocEventID()
{
    // AllocEventID 需要查全局事件表，有运行时开销
    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}

__aicore__ inline void FreeEventID()
{
    WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
    WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
    // 必须归还给 TPipe，否则泄漏
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[1]);
}
```

#### 修改后

```cpp
// 新版：直接赋值预定义常量，无需查表
__aicore__ inline void AllocEventID()
{
    mte3ToVId[0] = FA_MTE3_TO_V_0;   // = 0U (编译期常量)
    mte3ToVId[1] = FA_MTE3_TO_V_1;   // = 1U
    vToMte3Id[0] = FA_V_TO_MTE3_0;   // = 0U
    vToMte3Id[1] = FA_V_TO_MTE3_1;   // = 1U
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);  // 预置流水：初始状态 MTE3 可立即消费
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}

__aicore__ inline void FreeEventID()
{
    WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);  // 排干流水
    WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
    // 无需 ReleaseEventID：静态事件 ID 不需要归还
}
```

---

### 5.4 Vec2 内部事件 ID 静态化（核心热路径优化）

**位置**：`ProcessVec2DSplit()` 和 `ProcessVec2_C2GmUpdateUb()` 函数体内

这两个函数在每次 S2 任务的 Vec2 阶段被调用一次，是真正的**热路径**。

#### 修改前

```cpp
template <typename VEC2_RES_T>
__aicore__ inline void ProcessVec2DSplit(GlobalTensor<T> &mmRes, const RunInfoX &runInfo)
{
    // ...
    // 每次调用 ProcessVec2DSplit 都执行这三行：运行时查表 + 原子操作，约 5~20ns/次
    event_t mte2ToV    = static_cast<event_t>(tPipe->FetchEventID(HardEvent::MTE2_V));
    event_t vToMte2    = static_cast<event_t>(tPipe->FetchEventID(HardEvent::V_MTE2));
    event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        // ... DataCopy ...
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        // ... VEC 计算 ...
    }
}
```

#### 修改后

```cpp
template <typename VEC2_RES_T>
__aicore__ inline void ProcessVec2DSplit(GlobalTensor<T> &mmRes, const RunInfoX &runInfo)
{
    // ...
    // 编译期常量，运行时直接内联为字面量，零开销
    constexpr uint32_t mte2ToV    = FA_MTE2_TO_V_ID;    // = 2U
    constexpr uint32_t vToMte2    = FA_V_TO_MTE2_ID;    // = 2U
    constexpr uint32_t mte3ToMte2 = FA_MTE3_TO_MTE2_ID; // = 2U

    for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        // ... DataCopy ...
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        // ... VEC 计算 ...
    }
}
```

**为什么这里的改动效果最显著**：

`FetchEventID` 不是在 S2 内层循环里（那样改动会更大），但它是**每个 Vec2 任务**都要调用的。对于典型配置（S2=1024, s2BaseSize=128），有 8 个任务，每任务 3 次 `FetchEventID`，节省 24 次 × 10ns = **240ns ≈ 0.25µs**，与实测吻合。

---

### 5.5 新建 `fa_vec1_vf_static.h`（vselrIndexesBuf 参数替换）

**位置**：新建文件 `flash_attn/op_kernel/arch35/fa_vec1_vf_static.h`

原始的 `ProcessVec1Vf` 系列函数定义在公共库中，参数接受 `TBuf<>*`：

```cpp
// 原始接口（公共库中，不能修改）
namespace FaVectorApi {
    template <...>
    __aicore__ inline void ProcessVec1Vf(
        const LocalTensor<T2>& dstTensor,
        TBuf<> *vselrIndexesBuf,   // ← TBuf 指针，禁用接口
        ...
    );
}
```

#### 解决方案：薄封装层（Thin Wrapper）

新建 `fa_vec1_vf_static.h`，在 `fa_static` 命名空间中创建同名的封装函数，把 `LocalTensor<uint8_t>*` 参数直接传给底层实现（底层实现本身不依赖 TBuf，只依赖 LocalTensor）：

```cpp
// fa_vec1_vf_static.h
namespace fa_static {
using namespace AscendC;
using namespace FaVectorApi;

template <typename T, typename T2, typename pseShiftType, ...>
__aicore__ inline void ProcessVec1NoUpdate(
    const LocalTensor<T2>& dstTensor,
    LocalTensor<uint8_t> *vselrIndexesBuf,  // ← 直接 LocalTensor，无需 TBuf
    ...
) {
    // 从数组中取出对应索引的 LocalTensor
    LocalTensor<uint8_t> indexesTensor =
        vselrIndexesBuf[static_cast<int>(VselrIndexEnum::NZ_INDEX)];
    // 直接调用底层实现（底层接受 LocalTensor<uint8_t>，无 TBuf 依赖）
    ProcessVec1NoUpdateGeneralImpl512GqaFullquant<T, T2, ...>(
        dstTensor, srcTensor, ..., indexesTensor, ...);
}

// ProcessVec1Update、ProcessVec1Vf、ProcessVec1VfDn、ProcessVec1VfDnCausal 同理
} // namespace fa_static
```

**调用处的修改**（`flash_attn_block_vec_noquant_gqa.h`）：

```cpp
// 旧版调用
FaVectorApi::ProcessVec1VfDn<T, INPUT_T, false, false, s2BaseSize>(
    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, ...);

// 新版调用（加 fa_static:: 前缀，vselrIndexesBuf 已是 LocalTensor<uint8_t>[4]）
fa_static::ProcessVec1VfDn<T, INPUT_T, false, false, s2BaseSize>(
    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, ...);
```

---

### 5.6 Cube 侧（C 核）静态化修改

**位置**：`flash_attn_block_cube_noquant_gqa.h`（`FANoQuantGqaBlockCube` 类）

#### 5.6.1 问题根因

在静态 Tensor 编程模型中，Cube Block 没有 `TPipe` 对象。但原有 `Buffer::Init()` 接口内部调用了 `GetTPipePtr()->AllocEventID()`。当 TPipe 指针为 NULL 时，触发设备错误 507015（空指针解引用）。

```
原始调用链（触发崩溃）：
  cubeBlock.InitBuffers()
    → l1QBuffers.Init(*l1BufferManagerPtr, mm1LeftSize)
      → Buffer::Init()
        → GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>()   ← TPipe=NULL → CRASH
```

**错误修复方式**（被否决）：在静态环境中添加一个 `TPipe tPipe` 成员——这与静态 Tensor 规范矛盾，引入未定义行为。

**正确修复方式**：为 `Buffer` 和所有 `BuffersPolicy*` 类添加 `InitStaticDirect` 接口，使用预定义的静态事件 ID 直接初始化，完全绕过 `TPipe`。

#### 5.6.2 Cube 侧数据流与 Buffer 布局

Cube 侧矩阵计算的 L1/L0 缓冲区布局如下：

```
L1 内存布局（从 offset=0 起，总大小约 512KB）：
┌─────────────────────────────────────────┐  offset = 0
│  l1PBuffers（3-slot 旋转）               │  3 × mBaseSize × s2BaseSize × sizeof(Q_T)
│  用于 BMM2 的左矩阵中间结果               │  由 InitMMResBuf() 初始化
├─────────────────────────────────────────┤  FA_L1_CUBE_BASE = 3 × 上述大小
│  l1QBuffers（Q 矩阵，Single 或 DB）      │  mBaseSize × dBaseSize × sizeof(Q_T) [× 1 或 2]
│  l1KBuffers（K 矩阵，Single 或 DB）      │  s2BaseSize × dBaseSize × sizeof(KV_T) [× 1 或 2]
│  l1VBuffers（V 矩阵，Single 或 DB）      │  s2BaseSize × dBaseSize × sizeof(KV_T) [× 1 或 2]
└─────────────────────────────────────────┘

L0A（独立硬件内存，64KB 总）：
  ping: offset 0,     大小 32KB  ← BMM1 左矩阵 slot 0
  pong: offset 32KB,  大小 32KB  ← BMM1 左矩阵 slot 1

L0B（独立硬件内存，64KB 总）：
  ping: offset 0,     大小 32KB  ← BMM1/BMM2 右矩阵 slot 0
  pong: offset 32KB,  大小 32KB  ← 右矩阵 slot 1（DB 模式）

L0C（独立硬件内存，256KB 总）：
  4buff 模式：4 × 64KB  ← 矩阵乘积输出槽
  DB    模式：2 × 128KB ← 矩阵乘积输出槽
```

#### 5.6.3 成员变量对比（去除 TPipe/BufferManager 依赖）

**修改前（master 版本）**：

```cpp
class FANoQuantGqaBlockCube {
    TPipe *tPipe;                              // 外部传入的管道对象
    BufferManager<BufferType::L1> *l1BufferManagerPtr;  // 外部 L1 管理器指针
    BufferManager<BufferType::L0A> l0aBufferManager;    // 本地 L0A 管理器
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    // 缓冲区策略对象（地址由 BufferManager 运行时分配）
    L1QType   l1QBuffers;   // e.g. BuffersPolicySingleBuffer<L1>
    L1KvType  l1KBuffers;
    L1KvType  l1VBuffers;
    L0AType   mmL0ABuffers;
    L0BType   mmL0BBuffers;
    L0CType   mmL0CBuffers;
};
```

**修改后（tensor 静态版本）**：

```cpp
class FANoQuantGqaBlockCube {
    // 无 TPipe、无 BufferManager —— 所有地址由静态 constexpr 偏移决定

    L1QType   l1QBuffers;   // 同类型，但通过 InitStaticDirect 初始化
    L1KvType  l1KBuffers;
    L1KvType  l1VBuffers;
    L0AType   mmL0ABuffers;
    L0BType   mmL0BBuffers;
    L0CType   mmL0CBuffers;

    // 编译期静态事件 ID 常量（完整列表见 4.3.4 节）
    static constexpr TEventID FA_L1_Q_C2P_0 = static_cast<TEventID>(0);
    static constexpr TEventID FA_L1_Q_C2P_1 = static_cast<TEventID>(1);
    // ... 共 12 个事件 ID 常量（L1×6 + L0A×2 + L0B×2 + L0C×4）

    static constexpr uint32_t FA_L1_CUBE_BASE = 3U * mBaseSize * s2BaseSize * sizeof(Q_T);
};
```

`InitCubeBlock` 函数签名也相应简化，去掉了 `tPipe` 和 `l1BufferManagerPtr` 参数：

```cpp
// master 版本（旧）
__aicore__ inline void InitCubeBlock(TPipe *pipe,
                                     BufferManager<BufferType::L1> *l1BuffMgr,
                                     __gm__ uint8_t *query, ...);

// tensor 静态版本（新）：无需 TPipe 和 BufferManager
__aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, ...);
```

#### 5.6.4 `InitBuffers` 重写（纯地址初始化，不含同步）

与 VEC 侧完全对称：`InitBuffers` **只负责建立缓冲区地址**，不插入任何 `SetFlag` 同步原语。

**修改前（master 版本）**：

```cpp
__aicore__ inline void InitBuffers()
{
    // 全部通过 BufferManager 动态分配地址 + 动态 AllocEventID + 隐式 SetFlag
    if constexpr (dBaseSize > 256) {
        l1QBuffers.Init(*l1BufferManagerPtr, mm1LeftSize);   // → AllocEventID + SetFlag(c2p)
        l1KBuffers.Init(*l1BufferManagerPtr, mm1RightSize);  // → AllocEventID + SetFlag(c2p)
        l1VBuffers.Init(*l1BufferManagerPtr, mm2RightSize);  // → AllocEventID + SetFlag(c2p)
    } else { /* 类似 */ }
    l0aBufferManager.Init(tPipe, 65536);
    mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);  // → AllocEventID + SetFlag(c2p)
    mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
    mmL0CBuffers.Init(l0cBufferManager, slotSize);
}
```

**修改后（tensor 静态版本）**：

```cpp
__aicore__ inline void InitBuffers()
{
    // 与 VEC 侧保持一致：仅初始化地址，不插入任何同步原语
    // SetFlag 流水预置统一由 AllocEventID() 完成
    if constexpr (dBaseSize > 256) {
        constexpr uint32_t mm1LeftSize  = mBaseSize * dBaseSize * sizeof(Q_T);
        constexpr uint32_t mm1RightSize = s2BaseSize * l1BaseD * sizeof(KV_T);
        constexpr uint32_t l1QOff       = FA_L1_CUBE_BASE;
        constexpr uint32_t l1KOff       = l1QOff + mm1LeftSize;

        if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
            // dBaseSize>256, s2==256&&d>128 → Single + Single + Single
            l1QBuffers.InitStaticDirect(l1QOff, mm1LeftSize, FA_L1_Q_C2P_0, FA_L1_Q_C2P_0);
            l1KBuffers.InitStaticDirect(l1KOff, mm1RightSize, FA_L1_K_C2P_0, FA_L1_K_C2P_0);
            l1VBuffers.InitStaticDirect(l1KOff + mm1RightSize, mm2RightSize,
                                        FA_L1_V_C2P_0, FA_L1_V_C2P_0);
        } else {
            // dBaseSize>256, else → Single + DB + DB
            l1QBuffers.InitStaticDirect(l1QOff, mm1LeftSize, FA_L1_Q_C2P_0, FA_L1_Q_C2P_0);
            l1KBuffers.InitStaticDirect(l1KOff, l1KOff + mm1RightSize, mm1RightSize,
                                        FA_L1_K_C2P_0, FA_L1_K_C2P_0,
                                        FA_L1_K_C2P_1, FA_L1_K_C2P_1);
            // l1V 类似...
        }
    } else { /* dBaseSize<=256 情况：l1Q 用 DB */ }

    // L0A/B/C：LocalTensor 直接构造，固定偏移 0 起始
    constexpr uint32_t FA_L0_BUF_SIZE = 32U * 1024U;
    mmL0ABuffers.InitStaticDirect(0U, FA_L0_BUF_SIZE, FA_L0_BUF_SIZE,
                                  FA_L0A_C2P_0, FA_L0A_C2P_0,
                                  FA_L0A_C2P_1, FA_L0A_C2P_1);
    // L0B、L0C 类似，按策略类型选择 Single/DB/4buff 重载
}
```

**关键改变**：`InitStaticDirect` 接口由 `buffer.h` 中新增，语义与 VEC 侧的 `LocalTensor(pos, offset, size)` 等价——仅存储地址，不调用任何 TPipe 方法，不发出任何硬件事件。

#### 5.6.5 `AllocEventID` 重写（统一流水预置，与 VEC 侧对称）

**修改前**：空实现（SetFlag 错误地放在 InitBuffers 中）。  
**修改后**：所有 `SetFlag` 集中于此，与 VEC 侧的 `AllocEventID` 结构完全对称。

```cpp
__aicore__ inline void AllocEventID()
{
    // L1 c2p 预置：MTE1 告知 MTE2 当前 slot 空闲，可以重新从 GM 加载
    if constexpr (dBaseSize > 256) {
        SetFlag<HardEvent::MTE1_MTE2>(FA_L1_Q_C2P_0);  // L1Q: Single
        if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
            SetFlag<HardEvent::MTE1_MTE2>(FA_L1_K_C2P_0);    // Single + Single + Single
            SetFlag<HardEvent::MTE1_MTE2>(FA_L1_V_C2P_0);
        } else {
            SetFlag<HardEvent::MTE1_MTE2>(FA_L1_K_C2P_0);    // Single + DB + DB
            SetFlag<HardEvent::MTE1_MTE2>(FA_L1_K_C2P_1);
            SetFlag<HardEvent::MTE1_MTE2>(FA_L1_V_C2P_0);
            SetFlag<HardEvent::MTE1_MTE2>(FA_L1_V_C2P_1);
        }
    } else {
        SetFlag<HardEvent::MTE1_MTE2>(FA_L1_Q_C2P_0);   // L1Q: DB ping
        SetFlag<HardEvent::MTE1_MTE2>(FA_L1_Q_C2P_1);   // L1Q: DB pong
        // L1K/V 类似...
    }
    // L0A c2p 预置：CUBE 告知 MTE1 当前 slot 空闲，可重新从 L1 加载（始终 DB）
    SetFlag<HardEvent::M_MTE1>(FA_L0A_C2P_0);
    SetFlag<HardEvent::M_MTE1>(FA_L0A_C2P_1);
    // L0B c2p 预置（Single 或 DB）
    if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
        SetFlag<HardEvent::M_MTE1>(FA_L0B_C2P_0);
    } else {
        SetFlag<HardEvent::M_MTE1>(FA_L0B_C2P_0);
        SetFlag<HardEvent::M_MTE1>(FA_L0B_C2P_1);
    }
    // L0C c2p 预置：FixPipe 告知 CUBE 当前 slot 空闲，可写入新矩阵结果（4buff 或 DB）
    if constexpr (/* 4buff 条件 */) {
        SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_0);
        SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_1);
        SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_2);
        SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_3);
    } else {
        SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_0);
        SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_1);
    }
}
```

#### 5.6.6 `FreeEventID` 重写（排干流水，与 VEC 侧对称）

**修改前（master 版本）**：

```cpp
__aicore__ inline void FreeEventID()
{
    // Uninit 内部：WaitFlag<EventC2P>(id) + ReleaseEventID（需要 TPipe）
    l1QBuffers.Uninit(*l1BufferManagerPtr);
    l1KBuffers.Uninit(*l1BufferManagerPtr);
    l1VBuffers.Uninit(*l1BufferManagerPtr);
    mmL0ABuffers.Uninit(l0aBufferManager);
    mmL0BBuffers.Uninit(l0bBufferManager);
    mmL0CBuffers.Uninit(l0cBufferManager);
}
```

**修改后（tensor 静态版本）**：

```cpp
__aicore__ inline void FreeEventID()
{
    // UninitStaticDirect 内部仅调用 Buffer::UnInitStatic()：
    // → WaitFlag<EventC2P>(c2pEventId_)  排干流水
    // 无需 ReleaseEventID（静态 ID 不需要归还给 TPipe）
    l1QBuffers.UninitStaticDirect();
    l1KBuffers.UninitStaticDirect();
    l1VBuffers.UninitStaticDirect();
    mmL0ABuffers.UninitStaticDirect();
    mmL0BBuffers.UninitStaticDirect();
    mmL0CBuffers.UninitStaticDirect();
}
```

#### 5.6.7 修改前后同步开销对比

两种版本的 `SetFlag`/`WaitFlag` 调用数量**完全相同**——master 版本的 `Buffer::Init()` 内部也在调用 `SetFlag`，只是被封装隐藏了。

| | master 版本 | tensor 静态版本 |
|--|--|--|
| `SetFlag` 数量 | 与缓冲区槽位数相同（隐藏在 Init 内部）| **相同**（显式写在 AllocEventID 中）|
| `WaitFlag` 数量 | 与缓冲区槽位数相同（隐藏在 Uninit 内部）| **相同**（显式写在 UninitStaticDirect 内部）|
| `AllocEventID` 调用 | 每个 Buffer 槽位一次（运行时 TPipe 查表）| **零次**（全部静态 constexpr）|
| `ReleaseEventID` 调用 | 每个 Buffer 槽位一次（运行时 TPipe 操作）| **零次** |
| `BufferManager` 分配 | 每个 Buffer 槽位一次（运行时指针算术）| **零次** |

> 结论：tensor 静态版本在同步语义上与 master 完全等价，**不引入任何额外的 WaitFlag/SetFlag**，同时消除了所有运行时 TPipe/BufferManager 开销。

#### 5.6.8 与 VEC 侧修改的对比

| 对比维度 | VEC 侧（`flash_attn_block_vec_noquant_gqa.h`）| Cube 侧（`flash_attn_block_cube_noquant_gqa.h`）|
|---------|---------------------------------------------|-----------------------------------------------|
| 缓冲区对象 | `LocalTensor<T>`（完全绕过 Buffer 框架）| `BuffersPolicy*<L1/L0A/B/C>`（保留，供 Bmm API 使用）|
| `InitBuffers` | 直接 `LocalTensor(pos, offset, size)` 构造 | `InitStaticDirect(offset, size, p2cId, c2pId)` 构造 |
| `AllocEventID` | `SetFlag<MTE3_V>(id0/1)`（stage1OutBuf c2p 预置）| 全部 L1/L0 缓冲区 c2p 预置（对应多个 HardEvent 类型）|
| `FreeEventID` | `WaitFlag<MTE3_V>(id0/1)` | `UninitStaticDirect()`（各 Policy 类内部 WaitFlag）|
| 去除的依赖 | `TPipe`、`TBuf`、`TQue`、`LocalMemAllocator` | `TPipe`、`BufferManager`（L1/L0A/B/C 全部）|
| 保留的依赖 | 无（纯 LocalTensor）| `BuffersPolicyDB/Single/4Buff` 接口（Bmm API 要求）|

---

## 6. Ping-Pong 实现细节

### 6.1 UB 内存中的物理布局

所有缓冲区在 UB 中的位置完全由 `InitBuffers` 中的 `constexpr` 偏移决定（以 `mBaseSize=128, s2BaseSize=128, T=bfloat16` 为典型示例，其中 `BASE = mm1Sz*2 + mm2Sz*2`）：

```
UB 地址 (相对 BASE)
┌─────────────────────────────────────┐  BASE + 0
│  attenMaskBuf[0]    8192 bytes      │  ← Ping：奇数迭代用
│  attenMaskBuf[1]    8192 bytes      │  ← Pong：偶数迭代用
├─────────────────────────────────────┤  BASE + 16384
│  stage2OutBuf      32768 bytes      │  ← Vec2 输出（每 S2 覆盖写）
├─────────────────────────────────────┤  BASE + 49152
│  stage1OutBuf[0]   33024 bytes      │  ← Ping：Vec1 写 / MTE3 读（偶数迭代）
├─────────────────────────────────────┤  BASE + 82176
│  stage1OutBuf[1]   33024 bytes      │  ← Pong：Vec1 写 / MTE3 读（奇数迭代）
├─────────────────────────────────────┤  BASE + 115200
│  softmaxSumBuf[3]   768 bytes       │  ← 3-slot 旋转（各 256 bytes）
│  softmaxMaxBuf[3]   768 bytes       │  ← 3-slot 旋转
│  softmaxExpBuf[3]   768 bytes       │  ← 3-slot 旋转
│  maxBrdcstBuf      2048 bytes       │
│  sumBrdcstBuf      2048 bytes       │
├─────────────────────────────────────┤  BASE + 121600
│  commonBuf          512 bytes       │  (条件编译，!useDn 时存在)
│  lseBuf            (可选)           │  (运行时条件)
│  sinkBuf           (可选)           │  (运行时条件)
└─────────────────────────────────────┘
```

### 6.2 stage1OutBuf 2-slot Ping-Pong 完整流程

#### 6.2.1 初始化：AllocEventID 的预热作用

```cpp
__aicore__ inline void AllocEventID()
{
    mte3ToVId[0] = FA_MTE3_TO_V_0;  // = 0U（静态常量）
    mte3ToVId[1] = FA_MTE3_TO_V_1;  // = 1U
    vToMte3Id[0] = FA_V_TO_MTE3_0;  // = 0U
    vToMte3Id[1] = FA_V_TO_MTE3_1;  // = 1U
    // 关键：预置"MTE3 已消费完"标志，让第一次 Vec1 能立即写入
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);  // slot[0] 初始可写
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);  // slot[1] 初始可写
}
```

若不预置这两个标志，第一次 `WaitFlag<MTE3_V>(id=0)` 会永久阻塞——因为 MTE3 从未发出过"读完 slot[0]"的信号。预置等同于告诉硬件："流水线启动前，两个槽都是空的，VEC 随时可以写。"

#### 6.2.2 S2 主循环事件时序（PRELOAD_N=2）

两条执行流并发推进：`CreateTask`（AIV 主流，执行 Vec1）和 `ExecuteTask`（AIV 执行流，执行 MTE3 DataCopy）。`ExecuteTask` 始终比 `CreateTask` 落后 `PRELOAD_N=2` 个任务。

```
                CreateTask(mloop)                     ExecuteTask(loop = mloop - 2)
                ─────────────────                     ─────────────────────────────
mloop=0:
  slot = 0 % 2 = 0
  WaitFlag<MTE3_V>(id=0)   ← 初始已预置，立即通过
  Vec1 → 写入 stage1OutBuf[0]
  SetFlag<V_MTE3>(id=0)    → "slot[0] 数据已就绪"

mloop=1:
  slot = 1 % 2 = 1
  WaitFlag<MTE3_V>(id=1)   ← 初始已预置，立即通过
  Vec1 → 写入 stage1OutBuf[1]
  SetFlag<V_MTE3>(id=1)    → "slot[1] 数据已就绪"

mloop=2:                               loop=0, slot=0:
  slot = 2 % 2 = 0                       WaitFlag<V_MTE3>(id=0)  ← 等 mloop=0 写完
  WaitFlag<MTE3_V>(id=0) ←──────────     DataCopy(L1 ← stage1OutBuf[0])
  （阻塞，直到 MTE3 搬完）               SetFlag<MTE3_V>(id=0)   → "slot[0] 已搬完，可复用"
  Vec1 → 写入 stage1OutBuf[0]（复用）
  SetFlag<V_MTE3>(id=0)

mloop=3:                               loop=1, slot=1:
  slot = 1                               WaitFlag<V_MTE3>(id=1)
  WaitFlag<MTE3_V>(id=1) ←──────────     DataCopy(L1 ← stage1OutBuf[1])
  Vec1 → 写入 stage1OutBuf[1]            SetFlag<MTE3_V>(id=1)
  SetFlag<V_MTE3>(id=1)
                                       loop=2, slot=0: ...（以此类推）
```

**槽位一致性保证**：`slot = mloop % 2 = loop % 2`，因为 `loop = mloop - 2`，而 `2 % 2 = 0`，所以两侧始终使用相同槽位，不存在错位。

#### 6.2.3 收尾：FreeEventID 排干流水

```cpp
__aicore__ inline void FreeEventID()
{
    // S2 循环结束后，CreateTask 已完成，但最后 2 次 DataCopy 可能还在进行
    // 必须等待两个槽都"被 MTE3 读完"，才能安全释放缓冲区
    WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
    WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
    // 静态事件 ID 无需 ReleaseEventID
}
```

### 6.3 softmax 中间值的 3-slot 旋转机制

#### 6.3.1 为什么 2 个槽不够

`PRELOAD_N=2` 意味着任意时刻，最多有 3 个任务的 softmax 中间值（sum/max/exp）同时处于"活跃"状态：

```
时刻 T（mloop=N 的 Vec1 刚开始）：
  CreateTask(N)   正在写入  softmaxSumBuf[N%3]     ← 槽 A 被 Vec1 占用（写）
  CreateTask(N-1) 的结果留在 softmaxSumBuf[(N-1)%3] ← 槽 B 等待 Vec2 读
  ExecuteTask(N-2) 正在读取 softmaxSumBuf[(N-2)%3]  ← 槽 C 被 Vec2 占用（读）
```

若只有 2 个槽：当 `N=2` 时，CreateTask 写 `slot 0`，ExecuteTask(0) 读 `slot 0`——**写读冲突**。3 个槽使得三者始终互不重叠（`N%3 != (N-1)%3 != (N-2)%3` 恒成立）。

#### 6.3.2 代码对应关系

```cpp
// CreateTask 中（Vec1 写入）
uint32_t smSlot = runInfo.mloop % (PRELOAD_N + 1);  // mloop % 3
LocalTensor<float> sumUb = this->softmaxSumBuf[smSlot];  // 写本 task 的槽
LocalTensor<float> maxUb = this->softmaxMaxBuf[smSlot];
LocalTensor<T>     expUb = this->softmaxExpBuf[smSlot];
// Vec1 计算：bmm1结果 → softmax(sum,max,exp) → 写入 stage1OutBuf[mloop%2]

// ExecuteTask 中（Vec2 读取，loop = mloop - PRELOAD_N）
uint32_t smSlot2 = runInfo.loop % (PRELOAD_N + 1);   // loop % 3
LocalTensor<T>     expUb2 = softmaxExpBuf[smSlot2];  // 读 2 个任务前 Vec1 的结果
LocalTensor<float> sumUb2 = softmaxSumBuf[smSlot2];
// Vec2 计算：softmax 累加 + flash attention 增量更新
```

#### 6.3.3 3-slot 在 UB 中的物理地址（以 BASE=0 简化）

```
softmaxSumBuf[0]:  BASE+115200        (256 bytes)  task 0,3,6... 使用
softmaxSumBuf[1]:  BASE+115456        (256 bytes)  task 1,4,7... 使用
softmaxSumBuf[2]:  BASE+115712        (256 bytes)  task 2,5,8... 使用
softmaxMaxBuf[0]:  BASE+115968        (256 bytes)
...
softmaxExpBuf[2]:  BASE+117248        (256 bytes)
maxBrdcstBuf:      BASE+117504        (2048 bytes)  Vec1 广播 max 值用
sumBrdcstBuf:      BASE+119552        (2048 bytes)  Vec1 广播 sum 值用
```

### 6.4 attenMaskBuf 的 Ping-Pong

`attenMaskBuf[2]` 由 MTE2 异步预取（DataCopy）填充，Vec1 同步读取：

```cpp
// CreateTask 中（Vec1 使用当前 mask）
uint32_t maskSlot = runInfo.loop % DB;  // DB=2，当前迭代用的槽
LocalTensor<uint8_t> attenMaskUb = attenMaskBuf[maskSlot];
AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo);

// 当 Vec1 在使用 attenMaskBuf[0] 时，
// 下一次迭代的 AttenMaskCopyIn 会预取到 attenMaskBuf[1]（或由外部逻辑提前预取）
// 两个槽交替，读写互不干扰
```

### 6.5 全流水线并发时序总图

```
时间轴  →
──────────────────────────────────────────────────────────────────────────
CUBE    │Mm1[0]│Mm1[1]│Mm1[2]│Mm1[3]│Mm2[0]│Mm2[1]│Mm2[2]│...
VEC     │Vec1[0]│Vec1[1]│Vec1[2]│Vec1[3]│Vec2[0]│Vec2[1]│Vec2[2]│...
MTE3    │       │ Copy[0]    │ Copy[1]    │ Copy[2]    │...
MTE2    │AttenMask[1]│AttenMask[2]│AttenMask[3]│...
──────────────────────────────────────────────────────────────────────────
stage1 槽│  [0]  │  [1]  │  [0]  │  [1]  │读[0]  │读[1]  │读[0]  │
softmax槽│sm[0]  │sm[1]  │sm[2]  │sm[0]  │读sm[0]│读sm[1]│读sm[2]│
──────────────────────────────────────────────────────────────────────────
```

- **CUBE 和 VEC 真正并行**：CUBE 做 Mm1[N+2] 的矩阵乘时，VEC 同时做 Vec1[N] 的 softmax
- **VEC 和 MTE3 真正并行**：VEC 写 stage1OutBuf[0] 时，MTE3 读 stage1OutBuf[1]（或反之）
- **三单元均无空闲等待**：任何时刻都有工作在进行，流水满载

---

## 7. 性能影响分析

### 7.1 实测结果

| 服务器 | 改动前 | 改动后 | 提升 |
|--------|--------|--------|------|
| 32 核  | 基线   | -0.5µs | 约 1% |
| 28 核  | 基线   | -0.1µs | 约 0.2% |

### 7.2 为什么提升幅度有限？

**有效改动（热路径）**：

```
FetchEventID → constexpr（Vec2 内）
节省量 = 2~3次/任务 × N任务 × ~10ns/次
典型值: 3 × 8 × 10ns = 240ns ≈ 0.25µs（S2=1024）
```

**无效改动（初始化路径，每 kernel 调用一次，摊销后可忽略）**：

| 改动 | 每 kernel 节省 | 摊销到每任务 |
|------|---------------|-------------|
| `tPipe->InitBuffer` ×15 | ~75ns | <1ns/任务 |
| `AllocEventID` ×4 | ~40ns | <0.5ns/任务 |
| `TQue→LocalTensor` | ≈0（硬件等价）| 0 |

**真正的计算主体（未改动）**：

```
kernel 总时间 ≈ BMM1(CUBE) + Vec1(VEC) + BMM2(CUBE) + Vec2(VEC) + DataCopy(MTE)
              ≈  40µs       +   5µs     +  40µs       +  5µs     +  2µs
```

CUBE 矩阵计算（Mm1 + Mm2）占总时间 ~80%，这部分未被本次改动触及。

### 7.3 32核 vs 28核差距的原因

- 32 核服务器测试用例的 S2 更长 → 每核任务数更多 → FetchEventID 节省更多
- 28 核时 S2/28 不整除常见序列长度（1024/28 ≈ 36.6，负载不均衡），部分核任务数少于平均值，节省量减少

### 7.4 进一步提升性能的方向

| 优化方向 | 潜在收益 | 难度 |
|----------|---------|------|
| **AttenMask 异步预取**（在上一迭代末尾发出下一次的 DMA，隐藏 MTE2 延迟） | ~0.25µs × N任务 | 中 |
| **CUBE tiling 优化**（提升矩阵乘法 FLOPS 利用率） | 5~30µs | 高 |
| **减少 AIC/AIV 跨核同步气泡** | 1~5µs | 高 |
| 通过 profiling 定位 CUBE/VEC/MTE 真实瓶颈后针对优化 | 未知 | 中 |

---

## 附录：修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `flash_attn/op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h` | 成员变量类型替换；`SoftmaxInitBuffer` 参数从 `LocalMemAllocator&` 改为 `uint32_t off_sm`；`InitBuffers` 完全重写为 `LocalTensor` 直接构造（消除 `LocalMemAllocator`）；`AllocEventID/FreeEventID` 静态化；Vec2 事件 ID 静态化；调用前缀改为 `fa_static::` |
| `flash_attn/op_kernel/arch35/flash_attention_noquant_block_vec_flashdecode.h` | `InitBuffers` 完全重写：所有共享/专用缓冲区使用 `LocalTensor<uint8_t>(TPosition::VECIN, offset, size)` 直接构造，消除 `LocalMemAllocator`；所有偏移为 `constexpr` |
| `flash_attn/op_kernel/arch35/fa_vec1_vf_static.h` | **新建**：`fa_static` 命名空间封装 `ProcessVec1Vf/VfDn/VfDnCausal`，`TBuf<>*` → `LocalTensor<uint8_t>*` |
| `common/op_kernel/buffer.h` | `InitStaticDirect` 参数类型修正（`event_t` → `TEventID`），移除 `SetStaticEventIds`（无调用方）|
| `common/op_kernel/buffers_policy.h` | 移除 `InitStatic/UninitStatic/AllocStaticAddr`（无调用方的死代码）|
| `common/op_kernel/buffer_manager.h` | `InitStatic()` 修正：使用 `LocalTensor(Position, 0, maxSize)` 替代 `TBuf` |
| `flash_attn/op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h` | 去除 `TPipe*`/`BufferManager` 成员；`InitBuffers` 改用 `InitStaticDirect`（纯地址初始化，无同步）；`AllocEventID` 统一预置所有 L1/L0A/L0B/L0C 的 c2p `SetFlag`；`FreeEventID` 改用 `UninitStaticDirect`；新增 12 个 `static constexpr TEventID` 常量；**新增** `#include "fa_static_buffer.h"` 置于 `matmul.h` 之前，通过 include-guard 遮蔽阻止 common 下的 `buffer.h`/`buffer_manager.h`/`buffers_policy.h` 被拉入 |
| `flash_attn/op_kernel/arch35/fa_static_buffer.h` | **新建**：将 `common/op_kernel/` 下 `buffer.h`、`buffer_manager.h`、`buffers_policy.h` 的全部类型定义（`Buffer`、`BufferInfo`、`BufferManager`、`BuffersPolicySingleBuffer`/`DB`/`3buff`/`4buff`/`Matrix2x2BufferPolicy`）合并到此文件；预定义 `BUFFER_H`/`BUFFER_MANAGER_H`/`BUFFERS_POLICY_H` include-guard 宏，使 `matmul.h` 的 `#include "buffers_policy.h"` 自动跳过 common 版本；保留 `InitStaticDirect`/`UninitStaticDirect`/`SetCrossCore`/`WaitCrossCore` 等全部接口，确保 matmul API 和核间同步功能不受影响 |
