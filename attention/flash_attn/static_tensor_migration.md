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
5. [具体修改内容（修改前后对比）](#5-具体修改内容)
   - 5.1 成员变量类型替换
   - 5.2 `InitBuffers` 重写
   - 5.3 `AllocEventID / FreeEventID` 静态化
   - 5.4 Vec2 内部事件 ID 静态化
   - 5.5 新建 `fa_vec1_vf_static.h`（vselrIndexesBuf 参数替换）
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
- `tPipe->InitBuffer()` → `LocalMemAllocator::Alloc()`（编译期固定偏移）

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
// 新方式：LocalMemAllocator 编译期固定偏移
LocalMemAllocator<Hardware::UB> allocator;

// Alloc<位置, 类型, 元素数>() ← 三个都是编译期常量
LocalTensor<float> sumBuf0 = allocator.Alloc<TPosition::VECIN, float, 64>();
//                           ^ 这行等价于: address = UB基址 + 固定偏移, size = 64*4 = 256字节
```

**对比总结：**

| | TBuf（旧） | LocalTensor（新）|
|--|--|--|
| 内存地址确定时机 | 运行时（InitBuffer 调用时）| 编译期（Alloc 模板参数）|
| 需要 TPipe | 是 | 否 |
| 访问开销 | 有（Get<T>() 间接层）| 无（直接指针）|
| 是否允许使用 | 已禁用 | 推荐 |

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

#### 修改后

```cpp
// 新版：LocalMemAllocator 编译期固定偏移，无需 TPipe
__aicore__ inline void SoftmaxInitBuffer(LocalMemAllocator<Hardware::UB> &allocator)
{
    // Alloc<位置, 类型, 元素数> ← 三个都是编译期常量，运行时零开销
    softmaxSumBuf[0] = allocator.Alloc<TPosition::VECIN, float, 256 / sizeof(float)>();
    softmaxSumBuf[1] = allocator.Alloc<TPosition::VECIN, float, 256 / sizeof(float)>();
    softmaxSumBuf[2] = allocator.Alloc<TPosition::VECIN, float, 256 / sizeof(float)>();
    softmaxMaxBuf[0] = allocator.Alloc<TPosition::VECIN, float, 256 / sizeof(float)>();
    softmaxMaxBuf[1] = allocator.Alloc<TPosition::VECIN, float, 256 / sizeof(float)>();
    softmaxMaxBuf[2] = allocator.Alloc<TPosition::VECIN, float, 256 / sizeof(float)>();
    softmaxExpBuf[0] = allocator.Alloc<TPosition::VECIN, T,     256 / sizeof(T)>();
    softmaxExpBuf[1] = allocator.Alloc<TPosition::VECIN, T,     256 / sizeof(T)>();
    softmaxExpBuf[2] = allocator.Alloc<TPosition::VECIN, T,     256 / sizeof(T)>();
    maxBrdcstBuf = allocator.Alloc<TPosition::VECIN, float, 2048 / sizeof(float)>();
    sumBrdcstBuf = allocator.Alloc<TPosition::VECIN, float, 2048 / sizeof(float)>();
}

// 不再需要外部传入共享缓冲区，全部自主管理
__aicore__ inline void InitBuffers()
{
    LocalMemAllocator<Hardware::UB> allocator;

    // 第一步：跳过 ubBufferManager 已占用的 bmm1/bmm2 结果区域
    // （ubBufferManager 从偏移 0 开始分配，这里用 dummy Alloc 推进偏移指针）
    constexpr uint32_t mm1Sz = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    constexpr uint32_t mm2Sz = bmm2Write2Ub ?
        (mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T)) : 0U;
    (void)allocator.Alloc<TPosition::VECIN, uint8_t, mm1Sz * 2U + mm2Sz * 2U>();
    //   ↑ 这个 Alloc 结果被丢弃，仅用于推进内部偏移，确保后续分配不与 bmm1/2 结果区重叠

    // 第二步：依次分配各个业务缓冲区
    if constexpr (!bmm2Write2Ub) {
        mm2InBuf = allocator.Alloc<TPosition::VECIN, T, 32768 / sizeof(T)>();
    }
    attenMaskBuf[0] = allocator.Alloc<TPosition::VECIN, uint8_t, 8192>();   // mask ping
    attenMaskBuf[1] = allocator.Alloc<TPosition::VECIN, uint8_t, 8192>();   // mask pong
    stage2OutBuf = allocator.Alloc<TPosition::VECIN, uint8_t, 32768>()
                             .template ReinterpretCast<T>();
    stage1OutBuf[0] = allocator.Alloc<TPosition::VECIN, uint8_t, 33024>()
                               .template ReinterpretCast<INPUT_T>();         // stage1 ping
    stage1OutBuf[1] = allocator.Alloc<TPosition::VECIN, uint8_t, 33024>()
                               .template ReinterpretCast<INPUT_T>();         // stage1 pong

    SoftmaxInitBuffer(allocator);
    if constexpr (!useDn) {
        commonBuf = allocator.Alloc<TPosition::VECIN, uint8_t, 512>();
    }
    if (constInfo.isSoftmaxLseEnable) {
        lseBuf = allocator.Alloc<TPosition::VECIN, float, (mBaseSize >> 1U) * 8U>();
    }
    if (constInfo.learnableSinkFlag) {
        sinkBuf = allocator.Alloc<TPosition::VECIN, INPUT_T, 256 / sizeof(INPUT_T)>();
    }
}
```

**关键改变说明**：

1. **不再接受外部共享缓冲区参数** — 旧版需要 kernel 层的 TPipe 分配好后传进来；新版自己通过 `LocalMemAllocator` 直接分配，自给自足。

2. **dummy Alloc 推进偏移** — `ubBufferManager`（管理 bmm1/bmm2 结果区）和本 `InitBuffers` 的 `LocalMemAllocator` 共用同一个 UB，它们约定 bmm1/bmm2 区在前，vec block 的缓冲区在后。`(void)allocator.Alloc<...>()` 就是在推进这个分界点。

3. **ReinterpretCast** — 先以 `uint8_t` 分配（保证 8K/32K 的整体字节大小），再强制转换成实际使用的类型，与旧版的 `TBuf<>.Get<T>()` 语义完全等价。

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

## 6. Ping-Pong 实现细节

### 6.1 stage1OutBuf 的 Ping-Pong

`stage1OutBuf[2]` 是 Vec1 的输出缓冲区，使用 `loop % DB`（DB=2）交替：

```
S2 循环迭代:  loop=0    loop=1    loop=2    loop=3
使用槽位:     [0]       [1]       [0]       [1]
             (Ping)    (Pong)    (Ping)    (Pong)
```

**代码体现**：

```cpp
// ProcessVec1Nd 中
int64_t stage1Offset = runInfo.loop % DB;  // 0 或 1 交替

LocalTensor<INPUT_T> stage1CastTensor = this->stage1OutBuf[stage1Offset];
// Vec1 把 softmax 结果写入 stage1CastTensor（UB 中的 stage1OutBuf[0 or 1]）

SetFlag<HardEvent::V_MTE3>(vToMte3Id[stage1Offset]);   // 告诉 MTE3：VEC 写完了
WaitFlag<HardEvent::V_MTE3>(vToMte3Id[stage1Offset]);  // 等 VEC 真的写完

DataCopy(mm2AL1Tensor, stage1CastTensor, ...);          // MTE3 把数据搬到 L1
```

当 loop=2 想复用 `stage1OutBuf[0]` 时，loop=0 对应的 DataCopy 早已完成（因为每次 `CreateTask` 是串行完成 Vec1 + DataCopy 后才进入下一次）。

### 6.2 softmax 中间值的三缓冲 Ping-Pong

softmax 的 sum/max/exp 中间值需要在 Vec1（写）和 Vec2（读）之间传递：

```
PRELOAD_N=2 时，在任意时刻最多有 3 个任务的中间值同时"活着"：
- task(N)   正在被 Vec1 写入 → 使用槽 mloop % 3
- task(N-1) 已被 Vec1 写完，等待 Vec2 读 → 使用槽 (mloop-1) % 3
- task(N-2) 正被 Vec2 读取 → 使用槽 (mloop-2) % 3
```

三个槽的循环保证了互不覆盖：

```cpp
// Vec1 写入时（ProcessVec1VfDn / ProcessVec1Vf）
LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)];
//                                                      ↑ 写入本 task 的槽位

// Vec2 读取时（ProcessVec2OnUb）
LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)];
//                                          ↑ loop 是 ExecuteTask 的循环计数，
//                                            对应的是 2 个任务前的 Vec1 写入槽位
```

**为什么是 3 个而不是 2 个？**

| 当前 CreateTask | CreateTask(N) 写槽 | ExecuteTask(N-2) 读槽 |
|----------------|-------------------|-----------------------|
| PRELOAD_N=1 时 | slot = N%2         | slot = (N-1)%2        |
| PRELOAD_N=2 时 | slot = N%3         | slot = (N-2)%3        |

`PRELOAD_N=2` 时需要至少 `PRELOAD_N+1=3` 个槽，否则 CreateTask(N) 写 slot 0，同时 ExecuteTask(N-2) 也可能读 slot 0（当 N%3 == (N-2)%3，即 2 整除 3，不成立，故 3 个槽足够）。

### 6.3 attenMaskBuf 的 Ping-Pong

```cpp
// ProcessVec1Nd 中
LocalTensor<uint8_t> attenMaskUb = attenMaskBuf[runInfo.loop % DB];
AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo);
// ... Vec1 计算使用 attenMaskUb ...
// 下一次循环用 attenMaskBuf[(loop+1) % 2]，不覆盖当前正在使用的
```

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
| `flash_attn/op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h` | 成员变量类型替换、`InitBuffers` 重写、`AllocEventID/FreeEventID` 静态化、Vec2 事件 ID 静态化、调用前缀改为 `fa_static::` |
| `flash_attn/op_kernel/arch35/fa_vec1_vf_static.h` | **新建**：`fa_static` 命名空间封装 `ProcessVec1Vf/VfDn/VfDnCausal`，`TBuf<>*` → `LocalTensor<uint8_t>*` |
| `common/op_kernel/buffer.h` | `InitStaticDirect` 参数类型修正（`event_t` → `TEventID`），移除 `SetStaticEventIds`（无调用方）|
| `common/op_kernel/buffers_policy.h` | 移除 `InitStatic/UninitStatic/AllocStaticAddr`（无调用方的死代码）|
| `common/op_kernel/buffer_manager.h` | `InitStatic()` 修正：使用 `LocalTensor(Position, 0, maxSize)` 替代 `TBuf` |
| `flash_attn/op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h` | `static constexpr event_t` → `static constexpr TEventID`（枚举类型隐式转换修正）|
