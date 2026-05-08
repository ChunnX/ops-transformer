# IncreFlashAttention Preload vs Preload-DD 深度对比分析

## 1. 演进背景

`incre_flash_attention_preload_dd.h` 是在 `incre_flash_attention_preload.h` 基础上进行的**激进流水线重构**。两者共享同一个 Tiling 文件 (`incre_flash_attention_tiling.cpp`)，但 Kernel 侧的实现策略有本质差异。

核心驱动力：**减少 Cube 侧 MTE2（GM→L1）和 V（Vector 计算）的串行等待，通过 L1 独占 Buffer 和更细粒度的 M/K 方向 tiling、Fixpipe 内反量化融合等手段，将 Vector 与 Cube 的计算重叠度推向极致。**

---

## 2. 模板类与宏定义差异

### 2.1 类名变化

| 版本 | 类名 |
|------|------|
| Preload | `IncreFlashAttentionAttenPreload<IFAT>` |
| Preload-DD | `IncreFlashAttentionAttenPreloadDD<IFAT>` |

### 2.2 预加载数量

| 版本 | 宏定义 | 值 | 含义 |
|------|--------|----|------|
| Preload | `PRE_LOAD_NUM` | 4 | workspace 预加载 slot 数 |
| Preload-DD | `PRE_LOAD_NUM_DD` | 4 | workspace 预加载 slot 数 |
| Preload-DD | `TASK_NUM` | 2 | CV 间流水深度（两个任务并发） |
| Preload-DD | `EXTRA_NUM` | 6 | ExtraInfo 环形缓冲区大小 |

DD 版将 4-slot 批量预加载改为 2-task 流水深度 + 6-ring ExtraInfo，引入 `TaskContext` 结构体管理任务状态。

### 2.3 新增模板参数（DD 独有）

```cpp
// Preload-DD 独有
static constexpr uint32_t KV_LAYOUT_T = IFAT::kvLayout;    // KV 独立 layout
static constexpr uint32_t ATTENMASK_STRIDE = 2048;          // attention mask 对齐约束
static constexpr bool DEQF16 = false;                        // Fixpipe 反量化到 F16 标志
```

### 2.4 ExtraInfo 结构体差异

| 字段 | Preload | Preload-DD |
|------|---------|------------|
| `loop` | ✓ | ✓ |
| `bIdx` / `n2Idx` / `s2Idx` | ✓ | ✓ |
| `bn2IdxInCurCore` | ✓ | ✗ (由 TaskContext 管理) |
| `curSInnerLoopTimes` | ✓ | ✗ (由 TaskContext 管理) |
| `isFirstSInnerLoop` | ✓ | ✗ |
| `needSetOrgShape` | ✗ | ✓ |
| `s2BatchOffset` | ✗ | ✓ |
| `batOffset` / `n2Offset` | 各字段展开 | 组合为 `aivTensorParams` → delegate |
| `antiqKeyParamOffsetToken / antiqValueParamOffsetPerToken` | ✓ | ✗ (合并到 perChannelQuantOffset) |
| `mm2Completed` | ✗ | ✓ (flashDecode 各段完成标志) |

---

## 3. L1 Buffer 分配 —— 最核心的架构差异

### 3.1 Preload 架构：双通道共享 L1

```
L1 布局 (Preload):
┌─────────────────────────────────────────────────────┐
│  TQue<A1,1> inputQueL1A  — 2 slots × 32×KV_PRE_LOAD_L1_ROW_NUM × sizeof(KV_T)
│    用途：BMM1 载入 Q (A矩阵) + BMM2 载入 P (A矩阵)  — 分时复用
├─────────────────────────────────────────────────────┤
│  TQue<B1,1> inputQueL1B  — 2 slots × KV_PRE_LOAD_L1_ROW_NUM × headDim × sizeof(KV_T)
│    用途：BMM1 载入 K (B矩阵) + BMM2 载入 V (B矩阵)  — 分时复用
└─────────────────────────────────────────────────────┘
        总容量：约 128KB (两个 64KB 级别双缓冲队列)
        通过 EnQue/DeQue 自动管理同步
```

**特点**：
- L1 仅 2 个 Queue（A1 和 B1），BMM1 和 BMM2 分时使用
- Q 和 P 共享 A1 Queue，K 和 V 共享 B1 Queue
- 每次 Cube Compute 都需重新加载数据
- `EnQue/DeQue` 自动流水，但受 Queue 深度=2 限制，无法实现多轮预加载

### 3.2 Preload-DD 架构：三区独占 L1

```
L1 布局 (Preload-DD):
┌─────────────────────────────────────────────────────────────────┐
│  TBuf tmpBufL1Q  = 1 × 64KB   → qL1Buffers[0]
│    用途：仅 BMM1 载入 Q (A矩阵)，s2Idx==0 时加载后常驻
├─────────────────────────────────────────────────────────────────┤
│  TBuf tmpBufL1KP = 3 × 64KB   → kpL1Buffers[0..2]
│    用途：BMM1 载入 K (B矩阵) + BMM2 载入 P (A矩阵)
│          — 三重缓冲，BMM2 可提前预加载 P 而 BMM1 同时使用 K
├─────────────────────────────────────────────────────────────────┤
│  TBuf tmpBufL1V  = 4 × 64KB   → vL1Buffers[0..3]
│    用途：BMM2 载入 V (B矩阵)
│          — 四重缓冲，支持 BMM2 多轮复用
└─────────────────────────────────────────────────────────────────┘
        总容量：1×64K + 3×64K + 4×64K = 512KB (恰好用完 L1)
```

**特点**：
- L1 被切为三个独立区域，各自独立管理同步（通过 `HardEvent` 标志）
- **Q L1 常驻优化**：Q 矩阵在首次 s2Idx==0 时加载后，整个 sInnerLoop 不再重复加载，节省频繁的 GM→L1 搬运
- **K/P L1 三重缓冲**：BMM1 的 K 矩阵与 BMM2 的 P 矩阵可以交错使用，通过 `l1KpBufIter` 轮转
- **V L1 四重缓冲**：BMM2 的 V 矩阵可以多轮预加载，V 的 `l1VBufIter` 独立管理
- 所有同步从自动 `EnQue/DeQue` 改为手动 `WaitFlag/SetFlag<HardEvent::MTE1_MTE2>`

### 3.3 L0 Buffer 同步差异

| 项目 | Preload | Preload-DD |
|------|---------|------------|
| L0A 大小 | `2×32×256 = 16KB` (每个 16×128 INT8) | `2×32KB = 64KB` (ping-pong，容量 4×) |
| L0A 同步 | `BufferManager` 自动 | `WaitFlag/SetFlag L0A_EVENT0/1` 手动 |
| L0B 大小 | `2×256×128 = 64KB` (每个 256×128 INT8) | `2×32KB = 64KB` (ping-pong) |
| L0B 同步 | `BufferManager` 自动 | `WaitFlag/SetFlag L0B_EVENT0/1` 手动 |
| L0C 大小 | `2×32×512×4 = 128KB` (INT32) | `2×64KB = 128KB` (ping-pong, 容量同) |
| L0C 同步 | `BufferManager` 自动 | `WaitFlag/SetFlag L0C_EVENT0/1` 手动 |

**优势分析**：
- DD 的 L0A 从 16KB 增至 64KB，支持 BMM1/BMM2 的 A 矩阵一次性加载更多行（最多 `64K / sizeof(KV_T)` 行），减少 L1→L0 的迭代次数
- 手动 Event 同步消除 `EnQue/DeQue` 的开销，允许更灵活的流水排布
- L0A/L0B/L0C 的 ping-pong 标志各自独立，三路流水互不干扰

---

## 4. UB Buffer 分配差异

### 4.1 容量对比总表

| Buffer | Preload | Preload-DD | 差异说明 |
|--------|---------|------------|---------|
| inputQue1 | 32KB | 32KB | 同 |
| inputQue2 | 16KB | 16KB | 同 |
| outputQue1 | 32KB | 16KB | **DD 减半** |
| outputQue2 | 8KB | 8KB | 同 |
| tmpBuff1 | 32KB | 32KB | 同 |
| tmpBuff2 | 32KB | 32KB | 同 |
| tmpBuff3 | 2KB | 2KB | 同 |
| vec2ResBuff | ✗ | **16KB (新增)** | 反量化 BMM2 中间结果暂存 |
| qAmaxBuff[i] | 4×(256+256)B, 数组 | 4×(2K+256)B, flat | 容量增大 |
| softmaxResAmaxBuff[i] | 4×(256+256)B, 数组 | (2K+256)B × PRE_LOAD_NUM_DD, flat | DD: 单块存储 |
| qRowSumBuff[i] | 4×(256+256)B, 数组 | **✗ 移除** | PerToken 需要 qRowSum |
| softmaxResRowSumBuff[i] | 4×(256+256)B, 数组 | **单块 P_AMAX_BUF_BYTES** | DD 单块，preload 4 槽 |
| softmaxMaxBuff[i] | 4×256B, 数组 | (512B) × PRE_LOAD_NUM_DD, flat | |
| softmaxExpBuff[i] | 4×256B, 数组 | (512B) × PRE_LOAD_NUM_DD, flat | |
| softmaxSumBuff[i] | 4×256B, 数组 | (512B) × PRE_LOAD_NUM_DD, flat | |
| softmaxMaxDefaultBuff | 256B | 512B | 容量加倍 |
| softmaxSumDefaultBuff | 256B | 512B | 容量加倍 |
| **UB 总计** | ~153KB | ~165KB | DD 多 12KB |

### 4.2 Per-slot 数组 vs Flat Buffer

Preload 版本使用 C++ 数组存储每槽的 Buffer：
```cpp
TBuf<> softmaxMaxBuff[PRE_LOAD_NUM];    // 4个独立 TBuf
LocalTensor<T> softmaxMaxUb[PRE_LOAD_NUM]; // 4个独立 LocalTensor
```

Preload-DD 版本使用单块 flat Buffer + 偏移访问：
```cpp
TBuf<> softmaxMaxBuff;   // 1个TBuf, 大小 = 512B * PRE_LOAD_NUM_DD
LocalTensor<T> softmaxMaxUb; // 单 LocalTensor, 通过偏移访问
// 使用方式:
uint32_t idx = loop % PRE_LOAD_NUM_DD;
LocalTensor<T> myMax = softmaxMaxUb[idx * SOFTMAX_BUF_SIZE_PER_LOOP];
```

**原因**：DD 的 TASK_NUM=2 流水深度取代了原来的 4 路预加载，softmax 状态改为在 Tiling 阶段预计算的 `softmaxTmpUb` 中保留，per-slot 的概念被 per-task 取代。

### 4.3 vec2ResBuff（DD 独有）

```cpp
// Preload-DD: 16KB UB 矢量缓冲区
TBuf<> vec2ResBuff; // 用于 BMM2 反量化中间结果暂存
LocalTensor<half> vec2ResUb;
```

**用途**：DD 的 BMM2 输出在 AIV 侧处理时，不直接将反量化结果写回 GM，而是先暂存于 `vec2ResUb`，待完成 scaleV / offsetV 后再从 `vec2ResBuff` 转出到 `vec2ResGm`。这减少了一次 GM→UB 的读入操作。

---

## 5. 计算流水线对比

### 5.1 Preload 流水结构

```
时间轴 ──────────────────────────────────────────────────────────>
         Preload 4 tasks                     Process 4 tasks
  ┌──────┬──────┬──────┬──────┐     ┌──────┬──────┬──────┬──────┐
  │ T0   │ T1   │ T2   │ T3   │     │ T0   │ T1   │ T2   │ T3   │  (PRE_LOAD_NUM=4)
  │ Calc │ Calc │ Calc │ Calc │     │ Vec1 │ Vec1 │ Vec1 │ Vec1 │
  │+MM1  │+MM1  │+MM1  │+MM1  │     │+MM2  │+MM2  │+MM2  │+MM2  │
  └──────┴──────┴──────┴──────┘     └──────┴──────┴──────┴──────┘
              全并发 (4-task batch)           全并发
                                              尾部: 4×Vec2 (单步)
```

**特点**：
- 批量模式：一次预加载 4 个任务，再统一处理
- `CrossCoreWaitFlag(SYNC_V0_C1_FLAG)` 等待所有 4 个任务的 Q PreProcess 完成
- 尾部 Vec2 单独执行，不与后续任务重叠
- 适合 seqLen 较大的场景，4 个任务同时跑有助于掩盖延迟

### 5.2 Preload-DD 流水结构

```
时间轴 ────────────────────────────────────────────────────────────────>
  S Inner Loop 内:
  
  任务构建            Vec0 + QPre     Mm1 + Vec1       Mm2 + Vec2
  (TASK_NUM=2)        (AIV)           (AIC+AIV)        (AIC+AIV)
  ┌──────────┐    ┌──────┬──────┐ ┌──────┬──────┐ ┌──────┬──────┐
  │ TaskCtx  │    │ T0   │ T1   │ │ T0   │ T1   │ │ T0   │ T1   │
  │ [0..1]   │    │Calc  │Calc  │ │BMM1  │BMM1  │ │BMM2  │BMM2  │
  │          │    │+QPre │+QPre │ │+Vec1 │+Vec1 │ │+Vec2 │+Vec2 │
  └──────────┘    └──────┴──────┘ └──────┴──────┘ └──────┴──────┘
     ↑               ↑              ↑              ↑
  待tasks==2      SetFlag(V0_C1)  SetFlag(C1_V1)   SetFlag(V1_C2)
  或isLast时                                          SetFlag(C2_V2)
  才发射                                          并释放 taskCtx
```

**特点**：
- **渐进式流水**：不等待 4 个任务全部就绪，每凑齐 2 个 (TASK_NUM) 就发射
- gLoop/gtasks/gtasksC1V1/gtasksC2V2 四个计数器精确跟踪流水状态
- Vec2 与前一轮的 Mm2 可以重叠（不同的 TaskContext 槽位）
- `taskCtxRelease` 标志在 Vec2 完成后释放 taskContext
- **Q L1 常驻**：sInnerLoop 内 Q 只在 s2Idx==0 时加载，后续轮次直接复用

### 5.3 ExtraInfo 环形缓冲区

```
Preload:             Preload-DD:
extraInfo[0..3]      extraInfo[0..5]   (EXTRA_NUM = 6)
  4 slots               6 slots (ring buffer)
  idx = loop % 4        idx = loop % 6
```

DD 的 `EXTRA_NUM=6` 大于 `PRE_LOAD_NUM_DD=4`，目的是为 2 级流水（TASK_NUM=2）提供足够的缓冲空间，使 Vec2 阶段仍能读取上一轮的 extraInfo 参数而不被覆盖。

---

## 6. BMM1 计算流程对比

### 6.1 Preload 的 ComputeMm1

```
for each N-iteration (copy K into L1):       // N = sInner direction
    1. Load Q → L1A (AllocTensor + DataCopy) // 每次都重新加载
    2. Load K → L1B
    3. L1A→L0A, L1B→L0B, Mmad→L0C
    4. L0C→Fixpipe→mm1ResGm (INT32 写入 workspace)
    5. CrossCoreSetFlag C1_V1
```

### 6.2 Preload-DD 的 ComputeMm1

```
if (s2Idx == 0):
    Load Q → L1_Q (常驻, 仅一次)           // Q L1 常驻优化

for each M-iteration (baseM=64/sizeof(KV_T)): // 外层: M方向tiling
    ┌─────────────────────────────────────────┐
    │ Load Q[this_m] → L0A                    │ // L1_Q → L0A
    │                                         │
    │ for each N-iteration (copy K into L1): │ // 内层: K/N方向
    │     Load K → L1_KP (triple buffer)      │
    │     L1_KP → L0B                         │
    │     Mmad→L0C                            │
    │     Fixpipe→mm1ResGm:                   │
    │       scale1: IS_INT32_2_FP16(×1/1024)  │ // DEQF16反量化融合
    │       scale2: IS_INT32_2_FP16(×1/1024/254) + AtomicAdd
    │     isHead? → DataCopy                  │
    │     isMid?  → AtomicAdd                 │ // msd合并
    │     isTail? → DataCopy                  │
    └─────────────────────────────────────────┘
```

**关键差异**：
1. **Q L1 常驻**：减少 Q 的反复 GM→L1 搬运（对长 sequence 场景收益明显）
2. **M 方向 tiling**：将 Q 的 M 维度切分为 64-elem 的块，支持 BMM1 用更大的 L0A
3. **Fixpipe 反量化融合**：`quantPre = DEQF16` 直接将 INT32 反量化为 FP16 输出，无需 AIV 做额外反量化
4. **MSD 合并 inline**：通过 `SetAtomicAdd(1)` 将两段 MSD 在 Fixpipe 阶段合并到 GM
5. **head/mid/tail 三段写入**：M-tiling 引入后，中间段 (mid) 需要 AtomicAdd 而非覆盖写入

### 6.3 Q 常驻优化的 Tiling 支持

DD 版因为 Q 常驻 L1，在 Tiling 阶段有对应的 **BMM1 Buffer Size 检查**：

```cpp
// Tiling::CalcInnerSize
if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
    auto bmm1BufferSize = 40U * 1024U;  // 40KB 限制(含 Q 常驻 overhead)
    sInnerSize_ = min(bmm1BufferSize / nNumOfQInOneGroup_ / qSeqSize_, MAX_SPLIT_SIZE);
}
```

---

## 7. BMM2 计算流程对比

### 7.1 Preload 的 ComputeMm2

```
for each K-iteration (copy V into L1):       // K = sInner size direction
    1. Load P → L1A (from vec1ResGm)          // 每次都重新加载
    2. Load V → L1B (from valueGm)
    3. L1A→L0A, L1B→L0B, Mmad→L0C
    4. Fixpipe→mm2ResGm (INT32 写入 workspace)
    5. CrossCoreSetFlag C2_V2
```

### 7.2 Preload-DD 的 ComputeMm2

```
for each M-iteration (mCopyRowCount = P_LOAD_TO_L1_ROW_NUM = 128):
┌────────────────────────────────────────────┐
│ if (mCopyIdx == 0):                        │
│     Load V → L1_V (quadruple buffer)       │ // 仅首次加载一次！
│                                            │
│ Load P[mCopyIdx*128, 128] → L1_KP         │ // 复用 KP buffer!
│                                            │
│ for each K-iteration (baseK=128/sizeof):   │
│     L1_KP → L0A                            │
│     L1_V  → L0B                            │
│     Mmad→L0C                               │
│     Fixpipe→mm2ResGm (DEQF16 + AtomicAdd)  │
│                                            │
│ SetFlag V1_C2 (M-iteration间也可发)        │
└────────────────────────────────────────────┘
```

**关键差异**：
1. **V L1 常驻**：V 矩阵仅在每个 sInner loop 的 `mCopyIdx==0` 时加载一次到 L1_V，后续 M-iteration 直接复用，大幅减少 GM→L1 的 V 搬运
2. **KP L1 复用**：P 矩阵通过 `kpL1Buffers` 载入 L1_KP，此后立即转做 L0A 数据源。KP 和 Q 的 L1 分离避免了 BMM1 和 BMM2 的 L1 冲突
3. **双层 Tiling**：M 方向 128 行 × K 方向 128 列的子块迭代，覆盖 M×V 的大矩阵乘
4. **Fixpipe DEQF16**：与 BMM1 相同的 inline 反量化

### 7.3 L1 Buffer 在 BMM1/BMM2 间的联动

```
时间轴 ─────────────────────────────────────────────────────→
         BMM1 阶段                      BMM2 阶段
        
L1_Q:   [  Q数据 常驻  ............................... ]
L1_KP:  [  K_BUF0  ] [ K_BUF1  ] [ P_BUF0 ] [ P_BUF1 ]  ...
L1_V:                  [  V_BUF0 ] [  V_BUF1 ] [ V_BUF2 ] ...
```

KP L1 的三重缓冲在 BMM1 和 BMM2 之间平滑切换：BMM1 使用其加载 K，BMM2 使用其加载 P。V L1 的四重缓冲允许 BMM2 提前完成预加载。

---

## 8. Shape 变化追踪

### 8.1 Preload 版本 Shape 变换

```
输入:
Q:  [Batch, N_num, QSeqLen, HeadDim]    → BSND (或 BSH/BSND/TND)
K:  [Batch, N_kv,  SeqLen,   HeadDim]   → BSND
V:  [Batch, N_kv,  SeqLen,   HeadDim]   → BSND

阶段:
┌──────────────────────────────────────────────────────────┐
│ Q PreProcess (ANTIQUANT)                                 │
│   Q [B,N,S1,D] → queryPreProcessResGm [int8]:           │
│     msdIterNum × (gSize/gSizeCube × S1) × D             │
│     实际: 2 × gSizeCube × HeadDim (s1Split)              │
├──────────────────────────────────────────────────────────┤
│ BMM1: Q × K^T                                            │
│   Q: [M=gSizeCube, K=headDim] × K^T: [N=sInnerSize, K=headDim] │
│   → mm1ResGm: [M, N] = [gSizeCube, sInnerSize] (INT32)  │
│   M = gSizeCube = gSize (preload G轴不切在Cube侧)        │
├──────────────────────────────────────────────────────────┤
│ Vec1: Softmax + P后处理                                  │
│   mm1ResGm [M,N](INT32) → vec1ResGm [M,N](KV_T)         │
│   shape 不变, dtype INT32→KV_T (通过反量化)              │
├──────────────────────────────────────────────────────────┤
│ BMM2: P × V                                              │
│   P: [M=gSizeCube, K=sInnerSize] × V: [K=sInnerSize, N=headDim] │
│   → mm2ResGm: [M, N] = [gSizeCube, headDim] (INT32)     │
├──────────────────────────────────────────────────────────┤
│ Vec2: 累积 + 后处理                                      │
│   mm2ResGm [M,N](INT32) → vec2ResGm [M,N](T/F16)→ attentionOutGm │
│   M = gSizeCube, N = headDim                             │
└──────────────────────────────────────────────────────────┘
```

### 8.2 Preload-DD 版本 Shape 变换

```
┌──────────────────────────────────────────────────────────────┐
│ BMM1: Q × K^T (M-tiling 引入)                                │
│   outer loop M (baseM = 64/sizeof(KV_T)):                    │
│     Q sub: [baseM, headDim]                                  │
│     K sub: [sInnerSize_sub, headDim]                         │
│   → mm1ResGm sub: [baseM, sInnerSize_sub] (H16 via DEQF16)  │
│                                                              │
│   单次 BMM1 总输出: [gSizeCube, sInnerSize] × msdIterNum    │
│   但通过 Fixpipe 的头/中/尾 AtomicAdd 分三段写入              │
├──────────────────────────────────────────────────────────────┤
│ Vec1: Softmax + P后处理 (M方向与BMM1一致)                    │
│   mm1ResGm → vec1ResGm                                       │
│   额外输出: qAmaxBuff, softmaxResAmaxBuff (perToken模式)     │
├──────────────────────────────────────────────────────────────┤
│ BMM2: P × V (M+K双层 tiling)                                │
│   outer M (P_LOAD_TO_L1_ROW_NUM = 128):                      │
│     inner K (baseK = 128/sizeof(KV_T)):                      │
│       P sub: [P_LOAD_TO_L1_ROW_NUM, sInnerSize_sub]          │
│       V sub: [sInnerSize_sub, headDim]                       │
│     → mm2ResGm sub: [P_LOAD_TO_L1_ROW_NUM, headDim] (H16)   │
│                                                              │
│   单次 BMM2 总输出: [gSizeCube, headDim] × msdIterNum        │
├──────────────────────────────────────────────────────────────┤
│ Vec2: 通过 vec2ResUb → vec2ResGm → 累积 → attentionOutGm    │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. C-V 间同步机制对比

### 9.1 CrossCore 同步标志（两者编号相同）

| 标志 | 值 | 方向 | 含义 |
|------|----|------|------|
| SYNC_V0_C1_FLAG | 6 | AIV→AIC | Q PreProcess 完成，Cube 可以开始 BMM1 |
| SYNC_C1_V1_FLAG | 7 | AIC→AIV | BMM1 完成，AIV 可以开始 Vec1 |
| SYNC_V1_C2_FLAG | 8 | AIV→AIC | Vec1 完成，Cube 可以开始 BMM2 |
| SYNC_C2_V2_FLAG | 9 | AIC→AIV | BMM2 完成，AIV 可以开始 Vec2 |

### 9.2 Preload-DD 独有：内部 L1/L0 Event 标志

DD 引入 11 个 `HardEvent` 标志管理 L1/L0 数据搬运：

| Event | 用途 |
|-------|------|
| L1Q_EVENT0 | Q L1 缓冲区就绪 |
| L1KP_EVENT0/1/2 | KP L1 三重缓冲就绪 |
| L1V_EVENT0/1/2/3 | V L1 四重缓冲就绪 |
| L0A_EVENT0/1 | L0A ping-pong 就绪 |
| L0B_EVENT0/1 | L0B ping-pong 就绪 |
| L0C_EVENT0/1 | L0C ping-pong 就绪 |
| SYNC_INPUT_BUF1_FLAG (2) | AIV inputQue1 搬运完成 |
| SYNC_INPUT_BUF2_FLAG (3) | AIV inputQue2 搬运完成 |
| SYNC_OUTPUT_BUF1_FLAG (4) | AIV outputQue1 写出完成 |
| SYNC_OUTPUT_BUF2_FLAG (5) | AIV outputQue2 写出完成 |

### 9.3 同步粒度对比

```
Preload:                        Preload-DD:
                                
AIV → SetFlag(C1)               AIV → SetFlag(C1) + SetFlag(input_buf)
AIC → WaitFlag(C1)             AIC → WaitFlag(C1)
AIC → Bmm1                      AIC → WaitFlag(L1Q)
AIC → SetFlag(V1)               AIC → Bmm1(M-loop, subm-loop)
AIV → WaitFlag(V1)              AIC → SetFlag(L1KP) + SetFlag(L0A/B/C) × N
AIV → Vec1                      AIC → SetFlag(V1)
AIV → SetFlag(C2)               AIV → WaitFlag(V1)
AIC → WaitFlag(C2)             AIV → Vec1 (Vec1→C2 per M-iteration)
AIC → Bmm2                      AIC → WaitFlag(C2)
AIC → SetFlag(V2)               AIC → Bmm2(双层M,K loop)
AIV → WaitFlag(V2)              AIC → SetFlag(L1KP, L1V, L0A/B/C) × N
AIV → Vec2                      AIC → SetFlag(V2)
                                AIV → WaitFlag(V2)
                                AIV → Vec2 (Vec2→SetFlag时释放taskCtx)
```

DD 的 BMM1 和 BMM2 内部有大量 L1/L0 Event 等待和释放，同步粒度为每个子块（sub-tile），而 Preload 仅跨核同步。

---

## 10. 量化模式处理差异

### 10.1 ANTIQUANT 处理对比

| 处理 | Preload | Preload-DD |
|------|---------|------------|
| Q PreProcess (MSD) | FP32→FP16→INT8, 写入 queryPreProcessResGm | 相同流程 |
| BMM1 反量化 | AIV 在 Vec1 阶段做 `AntiquantMatmulResCombine` | **Fixpipe DEQF16 inline 反量化**，AIV 不做 |
| BMM2 反量化 | AIV 在 Vec2 阶段做 Combine | **Fixpipe DEQF16 inline 反量化** |
| msdIterNum=2 合并 | AIV Vec1/Vec2 阶段分别做两段 Combine | Fixpipe 内通过 `SetAtomicAdd(1)` 自动合并 |
| perToken Scale/Offset | `CopyAntiquantParamsPerToken` + `FusedMulAdd` | 相同流程，但 DD 使用 vec2ResUb 暂存 |

### 10.2 Fixpipe DEQF16 收益

**Preload 版本**：
```
Mmad(INT8) → L0C(INT32) → Fixpipe → mm1ResGm(INT32) → [AIV] AntiquantMatmulResCombine → F32
```
需要在 AIV 侧做两次 INT32→F32 转换和 MSD 合并，中间通过 GM 周转。

**Preload-DD 版本**：
```
Mmad(INT8) → L0C(INT32) → Fixpipe(DEQF16) → mm1ResGm(FP16)
   scale1: × quantScaleC1S1 → FP16     } 
   scale2: × quantScaleC1S2 → FP16 + AtomicAdd → FP16
```
两步 MSD 反量化直接在 Fixpipe 中完成，输出到 GM 即为 FP16，AIV 侧直接 Cast 为 F32 使用，省去一次完整的 GM→UB→GM 往返。

**量化 scale 配置**：
```cpp
// Preload-DD
quantScaleC1S1 = 1.0f / 1024;          // INT8 → FP16 第一段
quantScaleC1S2 = 1.0f / (1024 * 254);  // INT8 → FP16 第二段
quantScaleC2O1 = 1.0f / (1024 * 127);  // INT32 → FP16 BMM2 第一段
quantScaleC2O2 = 1.0f / (1024 * 254 * 127); // INT32 → FP16 BMM2 第二段
```

---

## 11. Workspace 空间对比

### 11.1 两者共用的 Workspace 结构

```
+0:          queryPreProcessResGm      (ANTIQUANT only, bmm2ResUbSize × actCoreNum × preLoadNum)
+offset1:    mm1ResGm                  (mmResUbSize × actCoreNum × preLoadNum)
+offset2:    vec1ResGm                 (mmResUbSize × actCoreNum × preLoadNum × kvDtypeRatio)
+offset3:    mm2ResGm                  (bmm2ResUbSize × actCoreNum × preLoadNum)
+offset4:    vec2ResGm                 (bmm2ResUbSize × actCoreNum × preLoadNum × F32)
```

### 11.2 Preload-DD 额外 Workspace

```
+offset5:    softmaxRescaleGm          (bmm2ResUbSize × actCoreNum × preLoadNum × F32)
             为 Vec2 的 perToken 反量化 rescale 预留中间空间
```

### 11.3 Tiling 阶段差异（同一文件，不同分支）

两者的 Tiling 在同一文件中通过 `perfMode_` 分支：

```
perfMode_:
  BMM_ALL_BY_VEC      → preload 未使用
  CUBE_VIEW_MM        → preload
  CUBE_VIEW_MM_DD     → preload_dd   ← DD 走此分支
  CUBE_VIEW_MM_FULL_LOAD → preload variant
  CUBE_VIEW_MM_MLA    → preload variant (MLA)
```

DD 模式在 Tiling 的关键差异：
- `preLoadNum = BUF_NUM (6)` vs Preload 的 `PRE_LOAD_NUM (4)`
- workspace 按 6× 而非 4× 倍数计算，但 DD Kernel 内部仍用 4 级 (PRE_LOAD_NUM_DD)
- 多出来的 2 个 slot 为 AIV 侧的 FLASH_DECODE 和 system prefix 逻辑预留

---

## 12. 核间核内流水对比总结

### 12.1 Preload 流水模型

```
   AIV Core 0              AIC Core 0
┌──────────────┐        ┌──────────────┐
│ QPre  QPre   │  V0C1  │  BMM1  BMM1  │
│ QPre  QPre   │------->│  BMM1  BMM1  │
│              │  C1V1  │              │
│ Vec1  Vec1   │<-------│              │
│ Vec1  Vec1   │  V1C2  │  BMM2  BMM2  │
│              │------->│  BMM2  BMM2  │
│ Vec2  Vec2   │  C2V2  │              │
│ Vec2  Vec2   │<-------│              │
└──────────────┘        └──────────────┘
      4级流水, Sequential Phase Execution
      Vec2 为尾部单独执行
```

### 12.2 Preload-DD 流水模型

```
   AIV Core 0              AIC Core 0
┌──────────────┐        ┌──────────────┐
│ QPre  QPre   │  V0C1  │  BMM1        │  ← s2Idx==0 Q→L1Q常驻
│              │------->│  (M-loop)    │  ← L1Q→L0A per M
│              │  C1V1  │  BMM1        │  ← L1KP→L0B per N
│ Vec1  Vec1   │<-------│              │  ← Fixpipe DEQF16
│              │  V1C2  │              │
│              │------->│  BMM2        │  ← L1KP←P, L1V常驻
│              │  C2V2  │  (M.K loop)  │  ← Fixpipe DEQF16
│ Vec2  Vec2   │<-------│              │
└──────────────┘        └──────────────┘
   TASK_NUM=2 深流水: BMM2(T0) 与 Vec1(T1) 可重叠
   M-tiling 使每步耗时均等, 减少流水气泡
```

### 12.3 流水气泡对比

**Preload**：
- 四个阶段的边界处存在气泡：QPre 全部完成 → BMM1 才开始，BMM1 全部完成 → Vec1 才开始
- Vec2 单独位于尾部，无法与下一轮重叠

**Preload-DD**：
- BMM1 按照 M 方向切成小块后，每完成一个小块即可发送 C1_V1 信号，AIV 不需要等整个 BMM1 完成
- BMM2 的 M-tiling 同样可以提前释放 V1_C2 信号
- 通过 `taskCtxRelease` 机制，Vec2 完成即释放一个 TASK_NUM 槽位，新的任务可以提前进入

---

## 13. 分核与 Tiling 差异

两者的分核逻辑完全相同（共享 tiling.cpp 文件），但以下参数在 DD 模式下有不同取值：

| 参数 | Preload 典型值 | Preload-DD 典型值 | 差异原因 |
|------|--------------|------------------|---------|
| preLoadNum | 4 (PRE_LOAD_NUM) | 6 (BUF_NUM) | DD 多 2 slot 给 AIV 端 |
| sInnerSize_ (非 GQA) | 8192 | 8192 (非 GQA) / 2048 (n/5三档) | DD 考虑 32MB workspace 约束 |
| sInnerSize_ (GQA+antiq) | 1024/2048 | 1024 (DD mode) | DD 模式强制 1024 |
| sInnerSize_ (MLA rope) | 512 | 512 | 一致 |
| cubeMSize | gSize×s1SplitSize | gSize×s1SplitSize (min to gMax) | MLA 可按 gMax 截断 |

DD 更激进地控制 workspace 大小（≤ 32MB per core × preLoadNum），通过下调 `sInnerSize_` 实现。

---

## 14. 优化收益分析

### 14.1 为什么 Q L1 常驻收益大

对于典型长序列场景（seqLen=8192+），`sInnerLoopTimes` 可能达到 8~16 次。每次 BMM1 如果都要重新加载 Q（M×headDim = 64×128 = 8K elements × 2B = 16KB），累计 GM→L1 带宽消耗：
- **Preload**：16KB × 8 loops × 4 cores = 512KB L1 读带宽（每个 sInnerLoop）
- **Preload-DD**：16KB × 1 load (s2Idx==0) = 16KB

提升比例 ≈ 8~16×，对于内存带宽密集的 Attention 算子至关重要。

### 14.2 Fixpipe DEQF16 消除 AIV 反量化

每次 BMM1 需要在 AIV 侧做 `AntiquantMatmulResCombine`（INT32→F32），操作量：
- M × N × 2(msdIterNum) 元素 × 2 (Mul + Add)
- 以 M=64, N=2048, msdIterNum=2 计算：64×2048×2 = 262K 元素运算

DD 将此计算融合进 Cube 的 Fixpipe 硬件单元，AIV 侧只需 Cast FP16→F32，大幅解放 AIV 计算资源。

### 14.3 L0A 容量翻倍 (16K→64K) 减少迭代

```
Preload: L0A = 16KB → 加载 16×128/sizeof(KV_T) 行 Q
         需要迭代 gSizeCube / (16×128/sizeof) 次

DD:      L0A = 64KB → 加载 64×128/sizeof(KV_T) 行 Q  
         迭代次数为 Preload 的 1/4
```

迭代次数减少直接减少 L1Q→L0A 的 DataCopy 指令发射次数。

### 14.4 V L1 常驻与四重缓冲

与 Q 常驻类似，DD 的 V 矩阵在 BMM2 阶段也只加载一次（`mCopyIdx==0`）。配合 4-slot L1_V，可以提前完成所有 V 的预加载而无需等待 BMM2 本身的数据消耗。这在 SeqLen 大而 HeadDim 小（如 128）的场景效果显著。

---

## 15. 修改点清单

从 Preload 到 Preload-DD 涉及的主要修改：

| # | 修改项 | 影响范围 | 复杂度 |
|---|--------|---------|--------|
| 1 | L1 分区：Queues → TBufs | InitBuffers, 所有 BMM 函数 | 高 |
| 2 | L1/L0 手动同步 Event | 所有 LoadData/DataCopy 调用处 | 高 |
| 3 | BMM1 M-tiling (外层 loop) | ComputeMm1, CopyInMm1A | 高 |
| 4 | BMM2 M/K 双层 tiling | ComputeMm2, CopyInMm2A/B | 高 |
| 5 | Fixpipe DEQF16 + AtomicAdd | ComputeMm1, ComputeMm2 | 中 |
| 6 | Process 流水重构 (2-task) | Process, CalcParams, TaskContext | 高 |
| 7 | ExtraInfo 环形缓冲 (6-slot) | CalcParams, TaskContext | 中 |
| 8 | vec2ResBuff / vec2ResUb | InitBuffers, Vec2 处理 | 低 |
| 9 | Q 常驻 L1 判断 (s2Idx==0) | CopyInMm1AToL1 | 低 |
| 10 | V 常驻 L1 判断 (mCopyIdx==0) | CopyInMm2BToL1 | 低 |
| 11 | per-slot arrays → flat buffers | InitBuffers, Softmax, Vec1/Vec2 | 低 |
| 12 | Tiling: preLoadNum 4→6 | CalcWorkSpace | 中 |
| 13 | Tiling: sInnerSize 约束新增 | CalcInnerSize | 中 |
| 14 | KV_LAYOUT_T 独立支持 | Init, CopyIn | 低 |
| 15 | ATTENMASK_STRIDE 2048 | AttenMaskCopyIn | 低 |

---

## 16. 设计理念总结

Preload 到 Preload-DD 的演进体现了 AscendC 算子优化的核心方法论：

1. **从分时复用到空间独占**：将 L1 从"分时共享 Queue"改为"空间独占 TBuf"，消除 BMM1/BMM2 的 L1 竞争
2. **从 Coarse-Grained 到 Fine-Grained Tiling**：M/K 双层 tiling 使每步计算更均匀，流水气泡更少
3. **从 Kernel-Side Dequant 到 Hardware-Assisted Dequant**：Fixpipe DEQF16 将反量化从 AIV 移至 Cube 硬件，释放 AIV 算力
4. **从批量预加载到渐进式流水**：TASK_NUM=2 的深流水 + taskCtx 轮转机制，比 4-task 批量更灵活
5. **从 auto-sync 到 manual-sync**：EnQue/DeQue → WaitFlag/SetFlag，赋予开发者精确控制同步粒度的能力

**DD 的代价**：代码复杂度显著增加（手动同步、更多 Indirection）、需要 L1 512KB 全量占用（无预留空间）、ExtraInfo 环形缓冲增加额外边界处理逻辑。
