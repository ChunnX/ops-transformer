# Flash Attention 静态 Tensor 编程迁移文档

## 概述

将 Flash Attention 内核代码从基于 `TPipe` 的动态缓冲区管理迁移到**静态 Tensor 编程**模型。

- **目标**：彻底移除 `TPipe`、`TQue`、`TBuf` 相关的缓冲区申请与同步机制
- **替换为**：`LocalTensor` 静态声明、`LocalMemAllocator` 手动分配、`SetFlag`/`WaitFlag`/`PipeBarrier` 显式同步
- **约束**：避免使用 event ID 6 和 7；在内核入口调用 `InitSocState()`

---

## 修改文件汇总

### 1. `flash_attn_entry_regbase.h`

**修改位置**：函数 `flash_attn_kernel_run`（原约 110-114 行）

| 原代码 | 新代码 |
|--------|--------|
| `TPipe tPipe;` | 删除 |
| `op.Init(..., &tPipe);` | `op.Init(...)` （去掉 `&tPipe`） |
| — | 新增 `InitSocState();` 在 `Kernel op;` 前 |

```cpp
// 修改后
InitSocState();
Kernel op;
op.Init(query, key, value, attenMask, cuSeqLensQ, cuSeqLensKv,
        blocktable, learnableSink, softmaxLse, attentionOut,
        workspace, runtimeMetaData, sequsedQ, sequsedKv, tilingData);
op.Process();
```

---

### 2. `flash_attn_kernel_base_gqa_template.h`（`FlashAttentionNoQuantGqaKernel`）

#### 2.1 成员变量

| 删除 | 说明 |
|------|------|
| `TPipe *pipe;` | 动态管道指针 |
| `TQue<QuePosition::VECOUT, 1> SharedBuffer1[2];` | 共享输出缓冲队列 |
| `TQue<QuePosition::VECIN, 1> SharedBuffer2[2];`  | 共享输入缓冲队列 |
| `TBuf<> SharedBuffer3;` | 共享临时缓冲 |

#### 2.2 `Init()` 签名

```cpp
// 旧：
__aicore__ inline void Init(..., TPipe *tPipe)

// 新：
__aicore__ inline void Init(...)   // 去掉 TPipe 参数
```

内部去除：`this->pipe = tPipe;`

#### 2.3 `InitMMResBuf()`

```cpp
// 旧：
l1BufferManager.Init(pipe, 524288);

// 新：
l1BufferManager.Init(524288);      // BufferManager 静态模式接口
```

#### 2.4 `InitVecBlock` / `InitCubeBlock` 调用

```cpp
// 旧：
vecFaBlock.InitVecBlock(tPipe, ...);
cubeBlock.InitCubeBlock(pipe, ...);

// 新：
vecFaBlock.InitVecBlock(...);      // 无 pipe 参数
cubeBlock.InitCubeBlock(...);
```

#### 2.5 `FlashDecode()`

```cpp
// 旧：
vecFdBlock.InitBuffers(this->pipe, SharedBuffer1, SharedBuffer2, SharedBuffer3);
vecFdBlock.FreeBuffers(SharedBuffer1, SharedBuffer2);

// 新：
vecFdBlock.InitBuffers();          // 无共享缓冲参数
// FreeBuffers 调用删除（静态分配无需释放）
```

#### 2.6 `Process()`

```cpp
// 旧：
vecFaBlock.InitBuffers(SharedBuffer1, SharedBuffer2, SharedBuffer3);

// 新：
vecFaBlock.InitBuffers();
```

---

### 3. `flash_attn_block_vec_noquant_gqa.h`（`FANoQuantGqaBlockVec`）

#### 3.1 成员变量替换

| 旧类型 | 新类型 | 说明 |
|--------|--------|------|
| `TPipe *tPipe;` | 删除 | — |
| `TBuf<> commonTBuf;` | `LocalTensor<uint8_t> commonBuf;` | 临时工作区 |
| `TQue<VECOUT,1> stage1OutQue[2];` | `LocalTensor<INPUT_T> stage1OutBuf[2];` | Stage1 输出双缓冲 |
| `TQue<VECIN,1> attenMaskInQue[2];` | `LocalTensor<uint8_t> attenMaskBuf[2];` | AttentionMask 输入 |
| `TBuf<> stage2OutBuf;` | `LocalTensor<T> stage2OutBuf;` | Stage2 中间结果 |
| `TEventID mte3ToVId[2];` | `event_t mte3ToVId[2];` | 静态赋值 {0,1} |
| `TEventID vToMte3Id[2];` | `event_t vToMte3Id[2];` | 静态赋值 {0,1} |
| `TBuf<> softmaxMaxBuf[PRELOAD_N+1];` | `LocalTensor<float> softmaxMaxBuf[3];` | Softmax max |
| `TBuf<> softmaxSumBuf[PRELOAD_N+1];` | `LocalTensor<float> softmaxSumBuf[3];` | Softmax sum |
| `TBuf<> softmaxExpBuf[PRELOAD_N+1];` | `LocalTensor<T> softmaxExpBuf[3];` | Softmax exp |
| `TBuf<> mm2InBuf;` | `LocalTensor<T> mm2InBuf;` | Bmm2 输入缓冲 |
| `TQue<VECOUT,1> maxBrdcst;` | `LocalTensor<float> maxBrdcstBuf;` | Max 广播 |
| `TQue<VECOUT,1> sumBrdcst;` | `LocalTensor<float> sumBrdcstBuf;` | Sum 广播 |
| `TQue<VECOUT,1> softmaxLseQueue;` | `LocalTensor<float> lseBuf;` | LSE 输出 |
| `TQue<VECIN,1> sinkQue;` | `LocalTensor<INPUT_T> sinkBuf;` | Learnable Sink |
| `TBuf<> vselrIndexesBuf[4];` | **保留** `TBuf<>` | 兼容 ProcessVec1Vf API |

#### 3.2 静态 Event ID 常量（新增）

```cpp
static constexpr event_t FA_MTE3_TO_V_0     = 0U;
static constexpr event_t FA_MTE3_TO_V_1     = 1U;
static constexpr event_t FA_V_TO_MTE3_0     = 0U;
static constexpr event_t FA_V_TO_MTE3_1     = 1U;
static constexpr event_t FA_MTE2_TO_V_ID    = 2U;
static constexpr event_t FA_V_TO_MTE2_ID    = 2U;
static constexpr event_t FA_MTE3_TO_MTE2_ID = 2U;
static constexpr event_t FA_MTE2_TO_V_SINK  = 4U;
```

> 同一 HardEvent 通道下的 event ID 相互独立，因此 `FA_MTE2_TO_V_ID` 与 `FA_V_TO_MTE2_ID` 可安全复用值 2。

#### 3.3 `InitVecBlock` 签名

```cpp
// 旧：
void InitVecBlock(TPipe *pipe, ...)  { tPipe = pipe; ... }

// 新：
void InitVecBlock(...)               { ... }  // 去掉 pipe 参数和赋值
```

#### 3.4 `SoftmaxInitBuffer` → 接受 `LocalMemAllocator &`

```cpp
void SoftmaxInitBuffer(LocalMemAllocator &allocator) {
    softmaxSumBuf[i] = allocator.alloc<float>(256 / sizeof(float));  // × 3
    softmaxMaxBuf[i] = allocator.alloc<float>(256 / sizeof(float));  // × 3
    softmaxExpBuf[i] = allocator.alloc<T>(256 / sizeof(T));          // × 3
    maxBrdcstBuf     = allocator.alloc<float>(2048 / sizeof(float));
    sumBrdcstBuf     = allocator.alloc<float>(2048 / sizeof(float));
}
```

#### 3.5 `InitBuffers()` — 核心 UB 静态分配

```cpp
void InitBuffers() {
    LocalMemAllocator allocator;
    // 1. 跳过内核 ubBufferManager 占用的 bmm1/bmm2 区域
    constexpr uint32_t mm1Sz = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    constexpr uint32_t mm2Sz = bmm2Write2Ub ?
        (mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T)) : 0U;
    (void)allocator.alloc<uint8_t>(mm1Sz * 2U + mm2Sz * 2U);

    // 2. 按原共享缓冲顺序分配（与 FD block 共享同一 UB 区域）
    if constexpr (!bmm2Write2Ub)
        mm2InBuf       = allocator.alloc<T>(32768 / sizeof(T));
    attenMaskBuf[0]    = allocator.alloc<uint8_t>(8192);
    attenMaskBuf[1]    = allocator.alloc<uint8_t>(8192);
    stage2OutBuf       = allocator.alloc<uint8_t>(32768).ReinterpretCast<T>();
    stage1OutBuf[0]    = allocator.alloc<uint8_t>(33024).ReinterpretCast<INPUT_T>();
    stage1OutBuf[1]    = allocator.alloc<uint8_t>(33024).ReinterpretCast<INPUT_T>();
    // 3. Softmax / 广播 / vselr / LSE / Sink 缓冲
    SoftmaxInitBuffer(allocator);
    ...
}
```

#### 3.6 `AllocEventID` / `FreeEventID`

```cpp
void AllocEventID() {
    mte3ToVId[0] = FA_MTE3_TO_V_0;  mte3ToVId[1] = FA_MTE3_TO_V_1;
    vToMte3Id[0] = FA_V_TO_MTE3_0;  vToMte3Id[1] = FA_V_TO_MTE3_1;
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}
void FreeEventID() {
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
    // 无需 ReleaseEventID（静态 ID）
}
```

#### 3.7 队列操作替换模式

| 旧 TQue 操作 | 新静态 Tensor 操作 |
|---|---|
| `stage1OutQue[i].AllocTensor<T>()` | `stage1OutBuf[i]`（直接使用）|
| `stage1OutQue[i].EnQue(t)` + `DeQue<T>()` | `PipeBarrier<PIPE_V>()` |
| `stage1OutQue[i].FreeTensor(t)` | 删除 |
| `attenMaskInQue[i].AllocTensor<uint8_t>()` | `attenMaskBuf[i]` |
| `attenMaskInQue[i].FreeTensor(t)` | 删除 |
| `softmaxLseQueue.AllocTensor<float>()` | 直接使用 `lseBuf` |
| `softmaxLseQueue.EnQue(t)` + `DeQue()` | `PipeBarrier<PIPE_V>()` |
| `softmaxLseQueue.FreeTensor(t)` | 删除 |
| `sumBrdcst/maxBrdcst` 的 AllocTensor/EnQue/DeQue/FreeTensor | `PipeBarrier<PIPE_V>()` + 直接用 `sumBrdcstBuf`/`maxBrdcstBuf` |
| `sinkQue.AllocTensor<INPUT_T>()` | 直接用 `sinkBuf` |
| `sinkQue.EnQue(t)` | `SetFlag<MTE2_V>(FA_MTE2_TO_V_SINK)` |
| `sinkQue.DeQue<INPUT_T>()` | `WaitFlag<MTE2_V>(FA_MTE2_TO_V_SINK)` |
| `TBuf::Get<T>()` | 直接用对应 `LocalTensor<T>` 成员 |

#### 3.8 `tPipe->FetchEventID` 替换

```cpp
// ProcessVec2DSplit / ProcessVec2_C2GmUpdateUb 中：
// 旧：
event_t mte2ToV   = (event_t)tPipe->FetchEventID(HardEvent::MTE2_V);
event_t vToMte2   = (event_t)tPipe->FetchEventID(HardEvent::V_MTE2);
event_t mte3ToMte2 = (event_t)GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);

// 新：
constexpr event_t mte2ToV    = FA_MTE2_TO_V_ID;      // = 2
constexpr event_t vToMte2    = FA_V_TO_MTE2_ID;      // = 2
constexpr event_t mte3ToMte2 = FA_MTE3_TO_MTE2_ID;   // = 2
```

---

### 4. `flash_attention_noquant_block_vec_flashdecode.h`（`FiaBlockVecFlashDecode`）

#### 4.1 `InitBuffers()` 签名变更

```cpp
// 旧：
void InitBuffers(TPipe *pipe,
                 TQue<VECOUT,1> SharedBuffer1[],
                 TQue<VECIN,1>  SharedBuffer2[],
                 TBuf<>        &SharedBuffer3)

// 新：
void InitBuffers()  // 无参，内部使用 LocalMemAllocator
```

#### 4.2 UB 布局与 FA block 保持一致

`FiaBlockVecFlashDecode::InitBuffers()` 内部也跳过 bmm1/bmm2 区域，然后按照与 FA block **相同的分配顺序**定位各缓冲（FA 与 FD 共享同一段 UB，两者不同时运行）：

```
attenMaskBuf[0] 等价地址 ↔ fdLseMaxUbBuf 基地址  (8K)
attenMaskBuf[1]（跳过）                           (8K)
stage2OutBuf    等价地址 ↔ fdSumBuf 基地址        (32K)
stage1OutBuf[0] 等价地址 ↔ fdMm2ResBuf 基地址    (33K)
stage1OutBuf[1] 等价地址 ↔ fdReduceBuf 基地址    (33K)
```

#### 4.3 `FreeBuffers` 删除

静态分配无需 `FreeBuffers`，对应调用从 `FlashDecode()` 中删除。

#### 4.4 `AllocEventID` / `FreeEventID` 保持不变

FD block 的 event ID（8–13）使用硬编码常量 `SYNC_*_FLAG`，已是静态模式，无需改动。

#### 4.5 `FiaBlockVecFlashDecodeDummy` 同步更新

Dummy 类的 `InitBuffers` 签名同步改为无参。

---

### 5. `flash_attn_block_cube_noquant_gqa.h`（`FANoQuantGqaBlockCube`）

#### 5.1 成员变量

| 删除 | 说明 |
|------|------|
| `TPipe *tPipe;` | 动态管道指针 |

#### 5.2 `InitCubeBlock` 签名

```cpp
// 旧：
void InitCubeBlock(TPipe *pipe, BufferManager<L1> *l1BuffMgr, ...)
{ tPipe = pipe; ... }

// 新：
void InitCubeBlock(BufferManager<L1> *l1BuffMgr, ...)
{ ... }
```

#### 5.3 `InitBuffers()` 内的 BufferManager 初始化

```cpp
// 旧：
l0aBufferManager.Init(tPipe, 65536);
l0bBufferManager.Init(tPipe, 65536);
l0cBufferManager.Init(tPipe, 262144);

// 新：
l0aBufferManager.Init(65536);
l0bBufferManager.Init(65536);
l0cBufferManager.Init(262144);
```

---

## 主体修改框架

```
┌─────────────────────────────────────────────────────────┐
│  flash_attn_entry_regbase.h                             │
│  InitSocState();   ← 新增                               │
│  op.Init(...);     ← 去掉 &tPipe 参数                    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  flash_attn_kernel_base_gqa_template.h                  │
│  • 删除 TPipe *pipe; SharedBuffer1/2/3 成员             │
│  • Init() 去掉 TPipe 参数                               │
│  • InitMMResBuf: l1/ubBufferManager.Init(size)          │
│  • InitVecBlock/InitCubeBlock 无 pipe 参数              │
│  • InitBuffers()/AllocEventID()/FreeEventID() 无参      │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐  ┌────────────▼───────────────────┐
│ flash_attn_block_   │  │ flash_attn_block_cube_         │
│ vec_noquant_gqa.h   │  │ noquant_gqa.h                  │
│ • TQue/TBuf/TPipe   │  │ • TPipe *tPipe 删除            │
│   → LocalTensor     │  │ • InitCubeBlock 无 pipe        │
│ • LocalMemAllocator │  │ • l0[abc]BufferManager.Init    │
│   静态分配UB缓冲    │  │   (size) 无 pipe               │
│ • 静态 event ID     │  └────────────────────────────────┘
│ • PipeBarrier 替代  │
│   EnQue/DeQue       │
└──────────┬──────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│ flash_attention_noquant_block_vec_flashdecode.h         │
│ • InitBuffers() 无参，LocalMemAllocator + UB 共享布局   │
│ • FreeBuffers 删除                                      │
│ • AllocEventID/FreeEventID 已是静态 event 模式          │
└─────────────────────────────────────────────────────────┘
```

## 关键约束遵守情况

| 约束 | 状态 |
|------|------|
| 不使用 event ID 6 和 7 | ✅ FA block 使用 0,1,2,4；FD block 使用 8-13 |
| 内核入口调用 `InitSocState()` | ✅ `flash_attn_entry_regbase.h` 已添加 |
| 不使用 `TPipe`/`TQue`/`TBuf` 的 TPipe 初始化 API | ✅ 全部移除 |
| FA/FD block 共享同一段 UB 内存 | ✅ 两者 `InitBuffers()` 使用相同分配顺序和偏移 |

## 遗留说明

- `vselrIndexesBuf[4]` 保留 `TBuf<>` 类型：其作为数组传递给 `ProcessVec1Vf`/`ProcessVec1VfDn` API，该 API 签名暂未更新，待 VF 接口迁移后可统一替换为 `LocalTensor<uint8_t>[4]`。
- `BufferManager<L1/UB>::Init(size)` 的静态模式 API 需确认框架版本支持（原 `Init(pipe, size)` 已改为 `Init(size)`）。
