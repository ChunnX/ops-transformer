# FlashAttention 静态 Tensor 编程迁移文档（arch35）

---

## 一、为什么迁移到静态 Tensor 编程

### 1.1 TPipe 动态编程模型的开销

在基于 `TPipe` 的编程范式下，框架自动完成：
- `TPipe` 实例化（占用寄存器状态、事件 ID 计数器）
- `InitBuffer` 为每个 `TQue`/`TBuf` 分配事件 ID（`AllocEventID()`）
- `EnQue/DeQue` 封装 `SetFlag/WaitFlag` 同步
- `FreeTensor/AllocTensor` 管理缓冲区轮转

这带来了**约数百纳秒的运行时初始化开销**，并且开发者无法精细控制事件 ID 与内存布局。

### 1.2 静态 Tensor 编程模型

Ascend C 提供静态 Tensor 编程方式，由开发者**完全自主管理**：
- AI Core 各内存单元（UB、L1、L0A、L0B、L0C）的布局和地址
- 各流水线（MTE2/MTE1/CUBE/MTE3/VEC）之间的同步事件
- Ping-Pong 双缓冲的建立和拆除

**约束（必须严格遵守）：**
- 不能使用 `TPipe`/`TQue`/`TQueBind`/`TBufPool` 框架接口，混用会产生未定义行为
- 事件 ID **不能使用 6 和 7**（与系统内部保留 ID 冲突）
- Kernel 入口**必须手动调用 `InitSocState()`**（TPipe 框架自动完成此步，静态模式下需手动调用）
- 同步由开发者用 `SetFlag`/`WaitFlag`/`PipeBarrier` 手动插入

### 1.3 FlashAttention 迁移触发原因

原代码在 tensor 版本内核入口（`flash_attn_entry_regbase.h`）中**未实例化 `TPipe`**，
但 `Buffer::Init()` 内部仍调用 `GetTPipePtr()->AllocEventID()`，触发了 NULL 指针解引用，导致：

```
Device error type 3, error code 507015（AIC 核心全部挂起）
```

**完整崩溃调用链：**

```
cubeBlock.InitBuffers()
  → l1QBuffers.Init(*l1BufferManagerPtr, size)    ← BuffersPolicyDB::Init
    → ping_.Init()                                 ← Buffer::Init
      → GetTPipePtr()->AllocEventID<MTE2_MTE1>()   ← TPipe 指针为 NULL
        → NULL 解引用 → device error 507015
```

---

## 二、静态 Tensor 编程核心概念

### 2.1 内存管理

AI Core 各内存单元在静态模式下地址从 0 开始固定，开发者有两种方式创建 `LocalTensor`：

**方式一：`LocalMemAllocator`（线性分配器，推荐初期使用）**

```cpp
AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
// 从偏移 0 开始，按调用顺序线性分配，不支持释放
LocalTensor<half>   stageBuf  = ubAllocator.Alloc<half, 4096>();
LocalTensor<float>  sumBuf    = ubAllocator.Alloc<float, 256>();
LocalTensor<uint8_t> maskBuf  = ubAllocator.Alloc<uint8_t, 8192>();
```

**方式二：`LocalTensor` 构造函数（直接指定地址，极致性能）**

```cpp
// 直接指定硬件位置和偏移地址，可精确控制 Bank 冲突规避
LocalTensor<half> pingBuf(AscendC::TPosition::VECCALC, pingAddr, pingLen);
LocalTensor<half> pongBuf(AscendC::TPosition::VECCALC, pongAddr, pongLen);
```

### 2.2 同步管理（SetFlag / WaitFlag）

AI Core 内部多条流水并行运行，数据依赖必须通过同步事件协调：

| 同步方向 | 接口 | 语义 |
|---|---|---|
| 正向（本次数据搬入 → 本次计算） | `SetFlag<MTE2_V>` → `WaitFlag<MTE2_V>` | 等 DMA 完成再计算 |
| 正向（本次计算 → 本次搬出） | `SetFlag<V_MTE3>` → `WaitFlag<V_MTE3>` | 等计算完成再搬出 |
| 反向（上次计算 → 本次搬入） | `SetFlag<V_MTE2>` → `WaitFlag<V_MTE2>` | 防止本次 DMA 覆盖上次未计算完数据 |
| 反向（上次搬出 → 本次计算） | `SetFlag<MTE3_V>` → `WaitFlag<MTE3_V>` | 防止本次计算覆盖上次未搬出数据 |

### 2.3 Ping-Pong 双缓冲原理

在 TPipe 编程中，`InitBuffer(que, depth=2)` 自动开启双缓冲。  
静态模式下需**手动声明两组 Tensor，并手动管理两套事件 ID**：

```
  迭代 0: DMA → ping,  COMPUTE ping
  迭代 1: DMA → pong,  COMPUTE pong   ← DMA(pong) 与 COMPUTE(ping) 重叠
  迭代 2: DMA → ping,  COMPUTE ping   ← DMA(ping) 与 COMPUTE(pong) 重叠
  ...
```

官方文档示例（简化版）：

```cpp
// 声明 ping/pong 两组 Tensor
LocalTensor<half> xPing(...), xPong(...);

// 初始化：两个 buffer 均处于"空闲可写"状态
SetFlag<V_MTE2>(EVENT_ID0);   // ping 的反向同步初始化
SetFlag<V_MTE2>(EVENT_ID1);   // pong 的反向同步初始化
SetFlag<MTE3_V>(EVENT_ID0);   // ping 的搬出后同步初始化
SetFlag<MTE3_V>(EVENT_ID1);   // pong 的搬出后同步初始化

for (uint32_t i = 0; i < totalBlocks; i++) {
    event_t eId      = (i & 1) ? EVENT_ID1 : EVENT_ID0;
    auto&   xLocal   = (i & 1) ? xPong : xPing;

    WaitFlag<V_MTE2>(eId);         // 等上一次同组计算完成（反向同步）
    DataCopy(xLocal, xGm[...]);    // DMA 搬入当前组

    SetFlag<MTE2_V>(eId);
    WaitFlag<MTE2_V>(eId);         // 等本次 DMA 完成（正向同步）
    WaitFlag<MTE3_V>(eId);         // 等本次搬出完成（反向同步）

    Add(zLocal, xLocal, yLocal);   // 计算

    SetFlag<V_MTE2>(eId);          // 通知 DMA 本组缓冲已用完
    SetFlag<V_MTE3>(eId);
    WaitFlag<V_MTE3>(eId);
    DataCopy(zGm[...], zLocal);    // 搬出
    SetFlag<MTE3_V>(eId);
}

// 结束：排干所有流水
WaitFlag<V_MTE2>(EVENT_ID0);
WaitFlag<V_MTE2>(EVENT_ID1);
WaitFlag<MTE3_V>(EVENT_ID0);
WaitFlag<MTE3_V>(EVENT_ID1);
```

---

## 三、内核入口初始化

### 3.1 修改文件：`flash_attn_entry_regbase.h`

**动态编程模式（旧）：**

```cpp
TPipe tPipe;          // 实例化 TPipe，自动初始化寄存器状态
Kernel op;
op.Init(..., &tPipe); // 将 TPipe 传入算子，用于分配事件 ID
op.Process();
```

**静态 Tensor 编程模式（新）：**

```cpp
InitSocState();       // ① 手动初始化全局状态寄存器（静态模式必须）
Kernel op;
op.Init(...);         // ② 无 TPipe 参数
op.Process();
```

**为什么必须调用 `InitSocState()`：**  
全局状态寄存器（包括向量掩码、饱和模式等）在内核启动时处于不确定状态。TPipe 框架会在构造时完成初始化；静态模式无 TPipe，必须手动调用，否则计算结果未定义。

---

## 四、UB 内存静态管理（AIV Vec Block）

### 4.1 原有方式：TQue + TBuf

```cpp
// 旧：框架管理缓冲区，开发者通过 AllocTensor/FreeTensor 轮转
TQue<VECOUT, 1> stage1OutQue[2];   // 双缓冲输出队列
TBuf<> softmaxSumBuf;              // Softmax 中间结果
TQue<VECIN, 1> attenMaskInQue[2]; // Mask 输入队列

// 使用时：
LocalTensor<T> t = stage1OutQue[i].AllocTensor<T>();
// ... 计算 ...
stage1OutQue[i].EnQue(t);           // → 内部调用 SetFlag
LocalTensor<T> t2 = stage1OutQue[i].DeQue<T>(); // → 内部调用 WaitFlag
stage1OutQue[i].FreeTensor(t2);
```

### 4.2 静态方式：LocalTensor + LocalMemAllocator

**成员变量替换：**

| 旧类型 | 新类型 | 说明 |
|---|---|---|
| `TQue<VECOUT,1> stage1OutQue[2]` | `LocalTensor<INPUT_T> stage1OutBuf[2]` | Stage1 输出双缓冲 |
| `TQue<VECIN,1> attenMaskInQue[2]` | `LocalTensor<uint8_t> attenMaskBuf[2]` | Attention Mask |
| `TBuf<> stage2OutBuf` | `LocalTensor<T> stage2OutBuf` | Stage2 中间结果 |
| `TBuf<> softmaxSumBuf[3]` | `LocalTensor<float> softmaxSumBuf[3]` | Softmax Sum |
| `TBuf<> softmaxMaxBuf[3]` | `LocalTensor<float> softmaxMaxBuf[3]` | Softmax Max |
| `TBuf<> softmaxExpBuf[3]` | `LocalTensor<T> softmaxExpBuf[3]` | Softmax Exp |
| `TQue<VECOUT,1> maxBrdcst` | `LocalTensor<float> maxBrdcstBuf` | Max 广播 |
| `TQue<VECOUT,1> sumBrdcst` | `LocalTensor<float> sumBrdcstBuf` | Sum 广播 |
| `TQue<VECOUT,1> softmaxLseQueue` | `LocalTensor<float> lseBuf` | LSE 输出 |
| `TQue<VECIN,1> sinkQue` | `LocalTensor<INPUT_T> sinkBuf` | Learnable Sink |

**`InitBuffers()` 核心：用 `LocalMemAllocator` 线性分配所有 UB 缓冲区**

```cpp
// flash_attn_block_vec_noquant_gqa.h → InitBuffers()
__aicore__ inline void InitBuffers() {
    LocalMemAllocator<Hardware::UB> allocator;

    // ① 跳过 ubBufferManager 已占用的 bmm1/bmm2 区域
    //    bmm1: mBaseSize/CV_RATIO * s2BaseSize * sizeof(T) * 2（ping+pong）
    //    bmm2: (可选) mBaseSize/CV_RATIO * dV * sizeof(T) * 2
    constexpr uint32_t mm1Sz = ...;
    (void)allocator.Alloc<uint8_t, mm1Sz * 2 + mm2Sz * 2>();

    // ② 按固定顺序分配剩余缓冲（顺序与 FD block 保持一致，两者共享同一段 UB）
    if constexpr (!bmm2Write2Ub)
        mm2InBuf       = allocator.Alloc<T, 32768 / sizeof(T)>();
    attenMaskBuf[0]    = allocator.Alloc<uint8_t, 8192>();
    attenMaskBuf[1]    = allocator.Alloc<uint8_t, 8192>();
    stage2OutBuf       = allocator.Alloc<uint8_t, 32768>().ReinterpretCast<T>();
    stage1OutBuf[0]    = allocator.Alloc<uint8_t, 33024>().ReinterpretCast<INPUT_T>();
    stage1OutBuf[1]    = allocator.Alloc<uint8_t, 33024>().ReinterpretCast<INPUT_T>();

    // ③ Softmax、广播、LSE、Sink 等辅助缓冲
    SoftmaxInitBuffer(allocator);  // 分配 sum/max/exp × 3 + brdcst × 2 + lse + sink
}
```

**操作替换规则：**

| 旧 TQue/TBuf 操作 | 新静态 Tensor 操作 |
|---|---|
| `stage1OutQue[i].AllocTensor<T>()` | `stage1OutBuf[i]`（直接使用，无需分配）|
| `EnQue(t)` + `DeQue<T>()` | `PipeBarrier<PIPE_V>()`（确保向量流水可见性）|
| `FreeTensor(t)` | 删除（静态地址，无需释放）|
| `sinkQue.EnQue(t)` | `SetFlag<MTE2_V>(FA_MTE2_TO_V_SINK)` |
| `sinkQue.DeQue<T>()` | `WaitFlag<MTE2_V>(FA_MTE2_TO_V_SINK)` |
| `tPipe->FetchEventID(MTE2_V)` | `constexpr event_t id = FA_MTE2_TO_V_ID` |

### 4.3 AIV 的静态事件 ID（`AllocEventID` / `FreeEventID`）

vec block 需要管理 MTE3→VEC 和 VEC→MTE3 两组双缓冲事件（用于 bmm2 结果的搬运流水）：

```cpp
// 静态常量定义
static constexpr event_t FA_MTE3_TO_V_0    = 0U;
static constexpr event_t FA_MTE3_TO_V_1    = 1U;
static constexpr event_t FA_V_TO_MTE3_0    = 0U;
static constexpr event_t FA_V_TO_MTE3_1    = 1U;
static constexpr event_t FA_MTE2_TO_V_ID   = 2U;
static constexpr event_t FA_V_TO_MTE2_ID   = 2U;
static constexpr event_t FA_MTE3_TO_MTE2_ID= 2U;
static constexpr event_t FA_MTE2_TO_V_SINK = 4U;

// AllocEventID：赋值静态 ID + 预置双缓冲初始状态
__aicore__ inline void AllocEventID() {
    mte3ToVId[0] = FA_MTE3_TO_V_0;
    mte3ToVId[1] = FA_MTE3_TO_V_1;
    vToMte3Id[0] = FA_V_TO_MTE3_0;
    vToMte3Id[1] = FA_V_TO_MTE3_1;
    // 预置："MTE3 搬出已完成，VEC 可以读取" → 两个 slot 都可立即处理
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}

// FreeEventID：排干所有流水（等待最后一次搬出完成）
__aicore__ inline void FreeEventID() {
    WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
    WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
    // 静态 ID 无需 ReleaseEventID（无 TPipe 管理）
}
```

**Ping-Pong 在 vec block 中的实际使用（`ProcessVec2OnUb` 简化）：**

```cpp
__aicore__ inline void ProcessVec2OnUb(RunInfoX &runInfo) {
    LocalTensor<T> vec2ResUb = this->stage2OutBuf;
    LocalTensor<T> mmRes     = bmm2ResBuf.template GetTensor<T>();

    // 等本 slot 的搬出完成（反向同步），防止覆盖上一次未搬出数据
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);   // 两 slot 轮流使用 [0] 和 [1]

    if (runInfo.isFirstS2Loop) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        FlashUpdateNew(...);   // online softmax update
    }

    bmm2ResBuf.SetCrossCore();  // 通知 AIC 本次 bmm2 结果已消费

    if (runInfo.isLastS2Loop) {
        CopyOutAttentionOut(...);
    }

    // 标记本次搬出已完成，下一轮可重用此 slot
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}
```

---

## 五、L1/L0 内存静态管理（AIC Cube Block）

### 5.1 背景：Buffer 类与 BuffersPolicy 类

FlashAttention 的 AIC 侧通过以下层次结构管理 L1/L0 缓冲区：

```
BuffersPolicy（策略层）
  └── Buffer<bufferType, syncType>（单个 buffer slot）
        └── BufferManager<Hardware>（硬件内存线性分配器）
```

**`Buffer::Init()`（原有）** 在内部调用 `GetTPipePtr()->AllocEventID()`，依赖 TPipe。  
**`Buffer::InitStatic()`（新增）** 接受调用方预定义的事件 ID，不依赖 TPipe。

### 5.2 `Buffer` 类静态接口设计

```cpp
// buffer.h 新增

// 初始化：直接赋值静态 ID，SetFlag 预置流水初始状态
__aicore__ inline void InitStatic(uint32_t p2cEventId, uint32_t c2pEventId) {
    if ASCEND_IS_AIC {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            p2cEventId_ = p2cEventId;   // 生产者→消费者事件（DMA完成→CUBE可读）
            c2pEventId_ = c2pEventId;   // 消费者→生产者事件（CUBE完成→DMA可写）
            // 预置："消费者已完成，缓冲区空"，允许第一次 DMA 立即写入
            SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
        }
    }
}

// 去初始化：WaitFlag 排干流水，不 ReleaseEventID（静态 ID 无需归还）
__aicore__ inline void UnInitStatic() {
    if ASCEND_IS_AIC {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
        }
    }
}
```

| 方法 | AllocEventID | SetFlag C2P | WaitFlag C2P | ReleaseEventID |
|---|:---:|:---:|:---:|:---:|
| `Init()`（原有，依赖 TPipe） | ✅ | ✅ | — | — |
| `InitStatic()`（新增） | ❌ | ✅ | — | — |
| `UnInit()`（原有，依赖 TPipe） | — | — | ✅ | ✅ |
| `UnInitStatic()`（新增） | — | — | ✅ | ❌ |

### 5.3 BuffersPolicy 类静态接口设计

四种策略类均新增 `InitStatic` / `UninitStatic`，以 `BuffersPolicyDB`（双缓冲）为核心讲解：

```cpp
// buffers_policy.h — BuffersPolicyDB 新增

// 分配 ping/pong 两个 buffer slot 并分别初始化
// ping 使用 p2cBase/c2pBase，pong 使用 +1 偏移，确保两个 slot 事件 ID 互不干扰
__aicore__ inline void InitStatic(BufferManager<bufferType> &bufferManager, uint32_t size,
                                  uint32_t p2cEventIdBase, uint32_t c2pEventIdBase) {
    ping_ = bufferManager.template AllocBuffer<syncType>(size);  // 线性分配 ping slot
    pong_ = bufferManager.template AllocBuffer<syncType>(size);  // 线性分配 pong slot
    ping_.InitStatic(p2cEventIdBase,     c2pEventIdBase);        // ping: ID base
    pong_.InitStatic(p2cEventIdBase + 1, c2pEventIdBase + 1);    // pong: ID base+1
}

// 排干两个 slot 的流水，归还内存（无需归还事件 ID）
__aicore__ inline void UninitStatic(BufferManager<bufferType> &bufferManager) {
    ping_.UnInitStatic();
    pong_.UnInitStatic();
    bufferManager.FreeBuffer(ping_);
    bufferManager.FreeBuffer(pong_);
}
```

其他策略类的 ID 步进规则：

| 策略类 | Buffer 数量 | ID 分配 |
|---|---|---|
| `BuffersPolicySingleBuffer` | 1 | `p2cBase`, `c2pBase` |
| `BuffersPolicyDB` | 2（ping/pong） | `Base`, `Base+1` |
| `BuffersPolicy3buff` | 3（a/b/c） | `Base`, `Base+1`, `Base+2` |
| `BuffersPolicy4buff` | 4（a/b/c/d） | `Base`, `Base+1`, `Base+2`, `Base+3` |

### 5.4 Ping-Pong 在 L1 缓冲区中的实现

以 `l1KBuffers`（K 矩阵 L1 双缓冲）为例，说明完整 Ping-Pong 流水：

```
初始化后状态（InitStatic 调用后）：
  ping.c2p = ID2 已 SetFlag → "ping 缓冲区空，DMA 可立即写入 K0"
  pong.c2p = ID3 已 SetFlag → "pong 缓冲区空，DMA 可立即写入"

迭代流水：
┌─────────────────────────────────────────────────────┐
│ 迭代 0: WaitFlag(c2p=ID2) → DMA(K0→ping) →         │
│         SetFlag(p2c=ID2) → CUBE WaitFlag(p2c=ID2) → │
│         CUBE 使用 ping[K0] 计算                      │
│         CUBE 完成 → SetFlag(c2p=ID2)                 │
├─────────────────────────────────────────────────────┤
│ 迭代 1: DMA: WaitFlag(c2p=ID3) → DMA(K1→pong) →    │  ← 与 CUBE 并行
│              SetFlag(p2c=ID3)                        │
│         CUBE: WaitFlag(p2c=ID3) → 使用 pong[K1]     │
│              → SetFlag(c2p=ID3)                      │
└─────────────────────────────────────────────────────┘
```

**`InitBuffers()` 中的静态 ID 分配（`flash_attn_block_cube_noquant_gqa.h`）：**

```cpp
// L1 缓冲区（MTE2_MTE1 / MTE1_MTE2 事件类型）
l1QBuffers.InitStatic(*l1BufferManagerPtr, mm1LeftSize,  0, 0);
//   ping: p2c=0, c2p=0; pong: p2c=1, c2p=1
l1KBuffers.InitStatic(*l1BufferManagerPtr, mm1RightSize, 2, 2);
//   ping: p2c=2, c2p=2; pong: p2c=3, c2p=3
l1VBuffers.InitStatic(*l1BufferManagerPtr, mm2RightSize, 4, 4);
//   ping: p2c=4, c2p=4; pong: p2c=5, c2p=5

l0aBufferManager.InitStatic();
l0bBufferManager.InitStatic();
l0cBufferManager.InitStatic();

// L0A/L0B 缓冲区（MTE1_M / M_MTE1 事件类型，与 L1 类型不同，ID 可从 0 重新分配）
mmL0ABuffers.InitStatic(l0aBufferManager, 32 * 1024, 0, 0);
//   ping: p2c=0, c2p=0; pong: p2c=1, c2p=1
mmL0BBuffers.InitStatic(l0bBufferManager, 32 * 1024, 2, 2);
//   ping: p2c=2, c2p=2; pong: p2c=3, c2p=3

// L0C 缓冲区（M_FIX / FIX_M 事件类型，4buff，a/b/c/d 使用 ID 0..3）
mmL0CBuffers.InitStatic(l0cBufferManager, sizePerBuf, 0, 0);
```

### 5.5 静态事件 ID 完整分配表

各硬件单元的 DMA pipeline 事件类型相互独立，不同类型之间的 ID 可重复使用；**同一事件类型内 ID 不能重叠**。

| 缓冲区 | 事件类型 (P2C → C2P) | ping p2c/c2p | pong p2c/c2p |
|---|---|---|---|
| `l1QBuffers` | MTE2_MTE1 / MTE1_MTE2 | 0 / 0 | 1 / 1 |
| `l1KBuffers` | MTE2_MTE1 / MTE1_MTE2 | 2 / 2 | 3 / 3 |
| `l1VBuffers` | MTE2_MTE1 / MTE1_MTE2 | 4 / 4 | 5 / 5 |
| `mmL0ABuffers` | MTE1_M / M_MTE1 | 0 / 0 | 1 / 1 |
| `mmL0BBuffers` | MTE1_M / M_MTE1 | 2 / 2 | 3 / 3 |
| `mmL0CBuffers` | M_FIX / FIX_M（4buff） | 0/0 ~ 3/3 | — |
| AIV mte3ToV[0]/[1] | MTE3_V / V_MTE3 | 0 | 1 |
| AIV mte2ToV / sink | MTE2_V / V_MTE2 | 2 | 4 |

> 所有 ID ≤ 5，严格遵守"不使用 6 和 7"的约束。

### 5.6 `FreeEventID()` — 拆除时排干流水

```cpp
// flash_attn_block_cube_noquant_gqa.h → FreeEventID()
__aicore__ inline void FreeEventID() {
    // 等待各缓冲区最后一次"消费完成"信号，确保所有 DMA/CUBE 流水排干
    l1QBuffers.UninitStatic(*l1BufferManagerPtr);
    l1KBuffers.UninitStatic(*l1BufferManagerPtr);
    l1VBuffers.UninitStatic(*l1BufferManagerPtr);
    mmL0ABuffers.UninitStatic(l0aBufferManager);
    mmL0BBuffers.UninitStatic(l0bBufferManager);
    mmL0CBuffers.UninitStatic(l0cBufferManager);
    // 无需 ReleaseEventID：静态 ID 不由 TPipe 管理
}
```

---

## 六、修改文件汇总

| 文件 | 修改内容 |
|---|---|
| `flash_attn_entry_regbase.h` | 删除 `TPipe tPipe;`，删除 `Init()` 中的 `&tPipe`，新增 `InitSocState()` |
| `flash_attn_kernel_base_gqa_template.h` | 删除 `TPipe *pipe` 成员及 `SharedBuffer1/2/3`；`InitMMResBuf` 改用静态 BufferManager；`InitVecBlock`/`InitCubeBlock` 去掉 pipe 参数；`FlashDecode()` 去掉共享缓冲参数 |
| `flash_attn_block_vec_noquant_gqa.h` | `TQue`/`TBuf`→`LocalTensor`；`InitBuffers()` 改用 `LocalMemAllocator`；`AllocEventID`/`FreeEventID` 使用静态常量；`tPipe->FetchEventID` 改为 `constexpr` 常量 |
| `flash_attn_block_cube_noquant_gqa.h` | 删除 `TPipe *tPipe`；`InitCubeBlock` 去掉 pipe 参数；`InitBuffers()` 改用 `InitStatic(bufMgr, size, p2cBase, c2pBase)`；`FreeEventID()` 改用 `UninitStatic()` |
| `flash_attention_noquant_block_vec_flashdecode.h` | `InitBuffers()` 改为无参，内部用 `LocalMemAllocator` 与 FA block 保持相同 UB 布局；删除 `FreeBuffers()` 调用 |
| `buffer.h` | 新增 `InitStatic(p2cId, c2pId)` 和 `UnInitStatic()`（原有接口保留） |
| `buffers_policy.h` | 4 个策略类各新增 `InitStatic(bufMgr, size, p2cBase, c2pBase)` 和 `UninitStatic(bufMgr)`（原有接口保留） |

---

## 七、关键约束遵守情况

| 约束 | 状态 |
|---|---|
| 不使用 `TPipe`/`TQue`/`TBufPool` 框架接口 | ✅ 全部移除，原有接口保留但 FA 路径不再调用 |
| 事件 ID 不使用 6 和 7 | ✅ cube block 最大 ID=5；vec block 使用 0,1,2,4；FD block 使用 8-13 |
| Kernel 入口调用 `InitSocState()` | ✅ `flash_attn_entry_regbase.h` 已添加 |
| Ping-Pong 双缓冲手动管理 | ✅ `BuffersPolicyDB::InitStatic` ping=Base，pong=Base+1，各有独立同步 ID |
| FA/FD block 共享同一段 UB | ✅ 两者 `InitBuffers()` 使用相同 `LocalMemAllocator` 分配顺序 |
| 原有动态接口不破坏 | ✅ `Buffer::Init`/`UnInit`、`BuffersPolicy::Init`/`Uninit` 均保留 |

---

## 八、遗留说明

- `vselrIndexesBuf[4]` 保留 `TBuf<>` 类型：作为数组传给 `ProcessVec1Vf` API，待该 API 签名迁移后统一改为 `LocalTensor<uint8_t>[4]`。
- `BufferManager<L1/L0>::InitStatic()` 以 offset=0 静态初始化；`AllocBuffer(size)` 按调用顺序线性递增偏移，各 slot 地址自动不重叠。
- Ping-Pong 对性能的实际收益依赖 S2 外层循环次数：S2 较小时（如 TND_MIXED_02 的 S2=128）流水暖机代价占比大，收益有限；大 S2 场景下双缓冲可有效隐藏 DMA 延迟。
