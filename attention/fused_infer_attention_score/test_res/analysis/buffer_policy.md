# FIA NoQuant GQA Buffer Policy 详解

## 1. 核心 Pipeline 中的 Buffer

```
BMM1(AIC) ──→ Vec1(AIV) ──→ BMM2(AIC) ──→ Vec2(AIV)
   │              │              │              │
   │  Q×K 写入UB   │ P=softmax    │  P×V 写入GM   │ FlashUpdate
   │  SetCrossCore │ 写入L1       │ SetCrossCore  │ 读取BMM2结果
   │              │ SetCrossCore  │              │
   └──bmm1Buffers─┘              └──bmm2ResGm───┘
         DB, BOTH    l1PBuffers      3buff, FORWARD
                     3buff, FORWARD
```

```cpp
// fia_kernel_noquant_gqa.h:72-75
BuffersPolicy3buff<GM, CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;  // BMM2→Vec2
BuffersPolicyDB<UB, CROSS_CORE_SYNC_BOTH>        bmm1Buffers;      // BMM1→Vec1
BuffersPolicyDB<UB, CROSS_CORE_SYNC_BOTH>        bmm2Buffers;      // BMM2→Vec2 (TOUB)
BuffersPolicy3buff<L1, CROSS_CORE_SYNC_FORWARD>  l1PBuffers;       // Vec1→BMM2
```

## 2. BuffersPolicyDB — 双 Buffer (DB)

### 工作原理

```
池: [buffer A] [buffer B]

iter0: Get()→A   │  AIC写A, SetCrossCore → AIV Get()→A, WaitCrossCore, 读, SetCrossCore
                 │  ring自动切到B
iter1: Get()→B   │  AIC写B, SetCrossCore → AIV Get()→B, WaitCrossCore, 读, SetCrossCore
iter2: Get()→A   │  ...
```

### 同步模式: CROSS_CORE_SYNC_BOTH

```
AIC (生产者)               AIV (消费者)
─────────────────         ─────────────────
写数据到 buffer[i]
SetCrossCore() ──────────→ WaitCrossCore()
                           读取数据
                           SetCrossCore()  ← 归还 buffer 到池
```

**关键**: BOTH 模式要求生产者和消费者**都调 SetCrossCore**。生产者 Set 通知"数据就绪"，消费者 Set 通知"已读完可以复用"。

### 使用场景

- **bmm1Buffers**: BMM1 结果 (Q×K) → Vec1 处理
- **bmm2Buffers**: BMM2 结果写入 UB 时 (BMM2_TOUB=true)

### 为什么 BMM1 需要 DB 不是 3buff？

BMM1 和 Vec1 是**同一时刻**执行的（都在 ExecuteTask(N)）。不需要 PRELOAD_N 延迟。1 个 write + 1 个 read 交替，2 槽够用。

## 3. BuffersPolicy3buff — 三 Buffer (Ring)

### 工作原理

```
池: [A] [B] [C], 内部 ring 计数器从 0 开始

ring 计数器:  0    1    2    0    1    2   ...
             ↓    ↓    ↓    ↓    ↓    ↓
Get()返回:   A    B    C    A    B    C   ...
```

每次 `Get()` 返回值递增: A→B→C→A→...，ring 自动循环。

### 同步模式: CROSS_CORE_SYNC_FORWARD

```
生产者 (Vec1)              消费者 (BMM2)
─────────────────         ─────────────────
写数据到 L1
outputBuf.SetCrossCore() → 释放 buffer N (通知消费者)
                            Get() → 返回 buffer N
                            WaitCrossCore() → 等待生产者 Set
                            读取数据
                            (不需要 SetCrossCore)
```

**关键**: FORWARD 模式下消费者**不需要 SetCrossCore**归还。ring 的回收由策略内部管理：只要生产者 SetCrossCore 了当前槽，该槽就被标记为可消费。消费者 Get/WaitCrossCore 后自动推进 ring。

### 为什么用 3buff 而不是 2buff？

```
PRELOAD_N=2 意味着 Vec1(N) 写的数据被 BMM2(N+2) 消费。
loop0: Vec1写slot0 → BMM2不消费 (PRELOAD_N延迟)
loop1: Vec1写slot1 → BMM2不消费
loop2: Vec1写slot2 → BMM2读slot0 ← PRELOAD_N=2
loop3: Vec1写slot0 → BMM2读slot1

同时活跃的 slot 数 = 1(正在写) + PRELOAD_N(待消费) = 1+2 = 3
```

如果只有 2buff:
```
loop2: Vec1写slot0 → BMM2读slot0  ← 同时读写！数据竞争
```

### 使用场景

- **l1PBuffers**: Vec1 输出 (softmax P) → BMM2 输入
- **bmm2ResGmBuffers**: BMM2 输出 (P×V) → Vec2 FlashUpdate

## 4. Ring 错位问题

### 为什么跳过 Get() 会导致死锁

```
l1PBuffers (3buff FORWARD ring):

正常:
  Vec1(loop0): SetCrossCore(A)       ──→ ring标记 A 可用
  BMM2(loop0): Get()→A, WaitCrossCore, 读A, ring→B
  Vec1(loop1): SetCrossCore(B)       ──→ ring标记 B 可用
  BMM2(loop1): Get()→B, WaitCrossCore, 读B, ring→C

跳过 BMM2(loop1):
  Vec1(loop0): SetCrossCore(A)
  BMM2(loop0): Get()→A, WaitCrossCore, 读A, ring→B ★ 推进到B
  Vec1(loop1): SetCrossCore(B)
  BMM2(loop1): (什么都不做)          ← ring 停在 B！
  Vec1(loop2): SetCrossCore(C)
  BMM2(loop2): Get()→B ★ 拿到的是 loop1 该消费的 B
               WaitCrossCore → 这步可能通过(B已被Set)
               读数据 → 读的是 Vec1(loop1)的数据，不是 Vec1(loop2)！

  后续: ring 彻底错位，某个槽永久被跳过
```

**结论**: `Get()` 是 ring 推进的唯一方式。**永远不能跳过 Get()**。哪怕数据不用，也必须 `Get() + WaitCrossCore` 维护 ring 位置。

## 5. 硬件 L1B / L0C 事件配对

除了 Buffer Policy 层面的 ring，硬件上还有内部的 pipe event 配对：

```
L1B 事件配对 (MTE1↔MTE2):
  Wait<MTE1_MTE2>()  ← 占用 L1B 总线
  CopyValueTile(...)  ← 加载 V 数据
  Set<MTE2_MTE1>()   ← 通知 MatMul 数据就绪
  MatMul(..., mm2B)   ← MatMul 消费 Wait<MTE2_MTE1>
  Set<MTE1_MTE2>()   ← 释放 L1B 总线 ← 必须和 Wait<MTE1_MTE2> 配对！

L0C 事件配对 (FIX↔M):
  Wait<FIX_M>()       ← 占用 L0C
  MatMul → 写入 L0C
  Set<M_FIX>()         ← 通知 Fixpipe MatMul 完成
  Fixpipe → Wait<M_FIX>, 读取 L0C
  Set<FIX_M>()          ← 释放 L0C ← 必须和 Wait<FIX_M> 配对！
```

### SkipBmm2 为什么需要维护这些事件

```
正常 BMM2 的 L1B 链:
  Wait<MTE1_MTE2> → CopyV → Set<MTE2_MTE1> → MatMul → Set<MTE1_MTE2>

跳过 BMM2 的 L1B 链:
  Wait<MTE1_MTE2> → (跳过CopyV) → Set<MTE2_MTE1> → Set<MTE1_MTE2>
                                                    ↑ 不能漏！
```

如果漏掉 `Set<MTE1_MTE2>`，L1B 总线永远被占用，下次 `Wait<MTE1_MTE2>` 永久阻塞。

## 6. 总结: SkipBmm2 的正确实现

```
SkipBmm2l0Split:
  ┌─ L1A: Get() + WaitCrossCore()                     ← 维持 3buff ring
  ├─ L1B: Get() + Wait<MTE1_MTE2> + Set<MTE2_MTE1>
  │       + Set<MTE1_MTE2>                            ← 维持 L1B 事件对
  ├─ L0C: Get() + Wait<FIX_M> + Set<FIX_M>            ← 维持 DB ring + FIX事件对
  ├─ 输出清零                                          ← Vec2 需要有效零值
  └─ outputBuf.SetCrossCore()                          ← 通知 Vec2
```

四个同步链缺一不可。跳过的只有中间的**数据搬运和计算**（CopyValueTile + MatMul + Fixpipe）。
