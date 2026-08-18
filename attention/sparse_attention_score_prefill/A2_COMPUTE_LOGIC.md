# A2 计算过程：循环、偏移、同步机制

## 一、循环处理逻辑

### 1.1 顶层结构

```
Kernel::operator()
├── InitWorkspaceStats()          // GM workspace 初始化 (-inf / 0)
├── InitSyncFlags()               // 预置流水线同步标志
├── SyncAll()                     // 全局屏障
├── Phase1KvCentricCompute()      // Phase 1: QK + Softmax + PV
│   └── for taskIdx stride loop   // 按 taskIdx 跨核遍历
│       └── for bi pipeline loop  // 三级流水线主循环
├── SyncAll()                     // Phase 1 完成屏障
└── Phase2CombineScale()          // Phase 2: 合并 topK (仅 VEC)
    └── for taskIdx stride loop   // 按 taskIdx 跨核遍历
```

### 1.2 Phase 1 — taskIdx stride 循环

```cpp
// kernel_a2.h:916
for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNumP1_; taskIdx += coreNum)
```

| 属性 | 值 |
|---|---|
| **循环变量** | `taskIdx` |
| **步长** | `coreNum = GetBlockNum() = 20` (AIC 数) |
| **起始** | `coreIdx = GetBlockIdx() / GetSubBlockNum()` (MIX block index) |
| **总任务数** | `totalTaskNumP1_ = totalKvRows_ * kvHeads_` |
| **任务映射** | `packedRow = taskIdx / kvHeads_`, `kvHeadIdx = taskIdx % kvHeads_` |
| **语义** | 每个 task 处理一个 (KV block, kvHead) 组合 |

**packedRow → (batchIdx, kvBlockIdx) 状态机：**

```cpp
// 增量步进，跨 task 持久化
while (curPackedRow < packedRow) {
    AdvancePackedRowCoord(batchIdx, kvBlockIdx);
    curPackedRow++;
}
```

### 1.3 Phase 1 — 三级流水线 bi 循环

```cpp
// kernel_a2.h:1013
for (uint32_t bi = 0U; bi < numBatches + PRE_LAUNCH; bi++)
```

| 属性 | 值 |
|---|---|
| **循环范围** | `[0, numBatches + PRE_LAUNCH)` |
| **PRE_LAUNCH** | `2` (PV 滞后 QK 2 个 batch) |
| **numBatches** | `CeilDiv(numQTokens, batchGroupsMax)` |
| **BATCH_RING_STAGES** | `PRE_LAUNCH + 1 = 3` (ring buffer 级数) |

**bi 循环内的条件分支：**

```
bi < numBatches:           QK + Softmax (收集 + 计算)
bi >= PRE_LAUNCH:          PV (滞后 PRE_LAUNCH 个 batch)
PRE_LAUNCH ≤ bi < numBatches:  QK + Softmax + PV 同时执行 (流水线满载)
bi ≥ numBatches:           只剩 PV 排空
```

**时序图：**

```
bi:     0    1    2    3    4    5    6
        ├─QK─┤├─QK─┤├─QK─┤├─QK─┤├─QK─┤
        ├─SM─┤├─SM─┤├─SM─┤├─SM─┤├─SM─┤
                  ├─PV─┤├─PV─┤├─PV─┤├─PV─┤├─PV─┤
        ←PRE_LAUNCH=2→
        numBatches=5, 循环范围=[0, 5+2)=[0,7)
```

### 1.4 QK matmul — L0 三重循环

```cpp
// block_mmad_qk_prefill_a2.hpp:293
for (nL0Itr = 0; nL0Itr < nL0Num; nL0Itr++)      // N 维 (validSize, 通常=1)
  for (mL0Itr = 0; mL0Itr < mL0Num; mL0Itr++)    // M 维 (groupCount*groupSize)
    for (kL0Itr = 0; kL0Itr < kL0Num; kL0Itr++)  // K 维 (head_dim=128)
      tileMmad(l0CTile, l0ATile, l0BTile, initMmad)
```

| 切分维度 | 参数 | 典型值 |
|---|---|---|
| N | `L0_TILE_N = 128` | nL0Num = 1 |
| M | `L0_TILE_M = 128` | mL0Num = 1 (M=4) |
| K | `L0_TILE_K = 128` | kL0Num = 1 (K=128) |

### 1.5 PV matmul — L0 三重循环

```cpp
// block_mmad_pv_prefill_a2.hpp:397
for (nL0Itr = 0; nL0Itr < nL0Num; nL0Itr++)      // N 维 (head_dim=128)
  for (mL0Itr = 0; mL0Itr < mL0Num; mL0Itr++)    // M 维 (groupCount*groupSize)
    for (kL0Itr = 0; kL0Itr < kL0Num; kL0Itr++)  // K 维 (validSize)
      tileMmad(l0CTile, l0ATile, l0BTile, initMmad)
```

### 1.6 Phase 2 — taskIdx stride 循环

```cpp
// kernel_a2.h:1286
for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNumP2_; taskIdx += coreNum)
```

| 属性 | 值 |
|---|---|
| **coreIdx** | `GetBlockIdx()` (线性 AIV id, 0..39) |
| **coreNum** | `GetBlockNum() * GetSubBlockNum() = 20 * 2 = 40` |
| **totalTaskNumP2_** | `totalQTokens * kvHeads_` |
| **任务映射** | `qToken = taskIdx / kvHeads_`, `kvHeadIdx = taskIdx % kvHeads_` |
| **语义** | 每个 task 合并一个 (Q token, kvHead) 的 topK 个 partial 结果 |

---

## 二、偏移量计算逻辑

### 2.1 GM 输入数据偏移

#### Q (Query) — GM → L1 gather

```cpp
// block_mmad_qk_prefill_a2.hpp:254
qBase = qToken * perQTokenStrideQ + qHeadStartBase
      = qToken * (qHeads * D) + (kvHeadIdx * groupSize * D)
```

| 变量 | 含义 |
|---|---|
| `perQTokenStrideQ` | `qHeads * embed` (一个 Q token 的 stride) |
| `qHeadStartBase` | `kvHeadIdx * groupSize * embed` (KV head 对应的 Q head 起点) |

#### K/V (Key/Value) — GM → L1 resident

```cpp
// kernel_a2.h:971
kvBlockBase = physicalBlockId * blockSize * kvHeads * embed
            + kvHeadIdx * embed
```

| 变量 | 含义 |
|---|---|
| `physicalBlockId` | `gBlockTable[batchIdx * maxBlocksPerBatch + localBlockIdx]` |
| `kvHeadIdx` | 当前 KV head 编号 |

### 2.2 GM S/P Workspace 偏移 (A2 跨核 staging)

#### S workspace (CUBE fixpipe → GM, VEC DataCopyPad 读)

```cpp
// kernel_a2.h:459
gmSWorkspace_ = params.workSpace + gmStagingBaseOffset
              + workspaceCoreIdx * gmSWorkspaceSize_

// block_mmad_qk_prefill_a2.hpp:391 (CUBE 写)
gmStageOffset = (bi % 2) * gmSStageElems_          // 双缓冲
gmSStageElems_ = A2_L0_TILE_M * blockSize = 128 * 128 = 16384

// block_epilogue_online_softmax_prefill_a2.hpp:606 (VEC 读)
gmSubOffset = gmStageOffset                          // subBlock 0
            = gmStageOffset + mCopyOffset * nRound   // subBlock 1
```

#### P workspace (VEC DataCopyPad → GM, CUBE Nd2Nz 读)

```cpp
// kernel_a2.h:468
gmPWorkspace_ = params.workSpace + gmStagingBaseOffset
              + totalGmSBytes + workspaceCoreIdx * gmPWorkspaceSize_

// block_epilogue_online_softmax_prefill_a2.hpp:504 (VEC 写)
gmStageOffset = (bi % 3) * gmPStageElems_            // 三缓冲
gmPStageElems_ = A2_L0_TILE_M * blockSize = 128 * 128 = 16384

// block_mmad_pv_prefill_a2.hpp:362 (CUBE 读)
gmPStageOffset = (bDe % 3) * gmPStageElems_          // 三缓冲
nRoundP = RoundUp(K, C0_SIZE)                        // K 对齐到 16
```

**关键问题：** `workspaceCoreIdx = coreIdx`，同一核上多个 task 共享同一 GM staging slot。

### 2.3 GM O_partial (accumOut) 偏移

```cpp
// block_mmad_pv_prefill_a2.hpp:479
wsOOff = qToken * perQTokenStride + kvHeadBase + slotK * slotOElems
       = qToken * (kvHeads * topK * slotOElems)
         + kvHeadIdx * (topK * slotOElems)
         + slotK * slotOElems
```

| 变量 | 含义 |
|---|---|
| `perQTokenStride` | `kvHeads * topK * slotOElems` |
| `kvHeadBase` | `kvHeadIdx * topK * slotOElems` |
| `slotOElems` | `groupSize * D` (一个 slot 的元素数) |
| `slotK` | 该 Q token 选中的第几个 KV block |

**ndNum=2 优化：** 一次写 2 个 qToken 的 slot

```cpp
srcNdStride = groupRows * C0_SIZE        // L0C zN 格式: 一个 group 占 16 行
dstNdStride = (nextQ - q) * perQTokenStride + (nextSlot - slot) * slotOElems
```

### 2.4 GM Stats (rowMax/rowSum) 偏移

```cpp
// kernel_a2.h:222
SlotStatOffset(qToken, kvHeadIdx, slotK)
  = TaskLinearIdx(qToken, kvHeadIdx) * topK * slotStatElems
    + slotK * slotStatElems

TaskLinearIdx(qToken, kvHeadIdx) = qToken * kvHeads + kvHeadIdx
slotStatElems_ = groupSize
```

**GM 布局：** `[totalQ * kvHeads, topK, groupSize]` (padded to 8-float stride)

```cpp
statStride = RoundUp(groupSize, 8)    // 8-float 对齐
taskStatBase = taskIdx * topK * statStride
paddedOff = taskStatBase + slot * statStride + group
```

### 2.5 GM 最终输出 (attentionOut) 偏移

```cpp
// kernel_a2.h:1195
attenOutOffset = qToken * kvHeads * groupSize * embed
               + kvHeadIdx * groupSize * embed
```

**GM 布局：** `[totalQ, kvHeads, groupSize, D]` = `[totalQ, q_heads, D]`

### 2.6 CSR 数据偏移

```cpp
// kernel_a2.h:955
rowPtrBase = kvHeadIdx * (totalKvRows + 1)
csrStart = k2q_row_ptr[rowPtrBase + packedRow]
csrEnd   = k2q_row_ptr[rowPtrBase + packedRow + 1]
numQTokens = csrEnd - csrStart

// kernel_a2.h:965
csrDataBase = kvHeadIdx * k2qNnzUpperBound
qToken  = k2q_q_indices[csrDataBase + csrStart + bi * batchGroupsMax + g]
slotK   = k2q_slot_indices[csrDataBase + csrStart + bi * batchGroupsMax + g]
```

### 2.7 UB subBlock 分区偏移

```cpp
// epilogue online softmax
gSplit = CeilDiv(groupCount, 2)         // group 分两半
mCopyOffset = gSplit * groupRows        // 行偏移
m = (subBlockIdx == 0) ? mHalf : (batchM - mHalf)

ubSubOffset = subBlockIdx * mCopyOffset * nRound       // UB S/P 分区
gmSubOffset = gmStageOffset + subBlockIdx * mCopyOffset * nRound  // GM S 分区
```

### 2.8 UB Stats 分区偏移

```cpp
// kernel_a2.h:768
subBlockStatsOff = subBlockIdx * 32 * sizeof(float)    // 每个 subBlock 32 floats
gmUbAlias = ubBuf[SM_UB_GM_OFFSET + ubSBufId * SM_UB_STAGE_BYTES + subBlockStatsOff]
glUbAlias = ubBuf[SM_UB_GL_OFFSET + ubSBufId * SM_UB_STAGE_BYTES + subBlockStatsOff]

grpStride = RoundUp(groupRows, 8)       // 8-float 对齐
localOff = (g - gLo) * grpStride        // group 内偏移
```

---

## 三、同步机制逻辑

### 3.1 同步机制总览

```
┌─────────────────────────────────────────────────────────┐
│  同步层级                                                │
├──────────┬──────────┬──────────────┬────────────────────┤
│ 核内流水线 │ 跨核同步  │ 全局屏障      │ 事件标志管理        │
│ HardEvent │ CrossCore │ SyncAll      │ InitSyncFlags      │
│ (同核不同  │ (AIC↔AIV │ (所有核)     │ ReleaseSyncFlags   │
│  pipe 之间)│  同一MIX) │              │                    │
└──────────┴──────────┴──────────────┴────────────────────┘
```

### 3.2 核内流水线同步 (HardEvent)

#### CUBE 侧

| 事件 | 含义 | 预置 |
|---|---|---|
| `MTE1_MTE2` | K/Q L1→L0 (MTE1) 完成 → GM→L1 (MTE2) 可覆盖 L1 | EVENT_ID0,1,2,3 |
| `M_MTE1` | MMAD 完成 → MTE1 可加载下一批 L0A/L0B | EVENT_ID0,1,2,3 |
| `FIX_M` | Fixpipe (L0C→GM) 完成 → MMAD 可写下一批 L0C | EVENT_ID0,1,2,3 |
| `M_FIX` | MMAD 完成 → Fixpipe 可读 L0C | (运行时 Set/Wait) |

#### VEC 侧

| 事件 | 含义 | 预置 |
|---|---|---|
| `V_MTE2` | V 计算 → MTE2 (GM→UB DataCopyPad) 可搬运 | EVENT_ID0,1 |
| `MTE2_V` | MTE2 (GM→UB) 完成 → V 可读 UB | (运行时 Set/Wait) |
| `V_MTE3` | V 计算 → MTE3 (UB→GM DataCopyPad) 可搬运 | (运行时 Set/Wait) |
| `MTE3_V` | MTE3 (UB→GM) 完成 → V 可覆盖 UB | EVENT_ID0,1,2,3,4 |
| `S_MTE3` | Stats V → MTE3 可搬运 | EVENT_ID0 |
| `MTE3_S` | Stats MTE3 完成 → S 可覆盖 | EVENT_ID0,1 |

### 3.3 跨核同步 (CrossCore MODE_2)

#### 机制

MODE_2 (0x2) 跨核同步：AIC SET → 两个 AIV WAIT 都完成 → AIV SET → AIC WAIT 完成。

**关键规则：**
1. 两个 AIV subBlock **必须都参与**（一个都不能跳过，否则 AIC WAIT 永远挂起）
2. 两个 AIV subBlock **必须用相同的 pipe** SET 同一个 flag ID
3. Flag ID 是 per-MIX-block scoped（每个 MIX block 独立的 0-7 ID 空间）

#### Flag ID 分配

| Flag ID | 名称 | 用途 | SET 方 | WAIT 方 |
|---|---|---|---|---|
| 0, 1 | `mm1ToSmFlag` | QK→Softmax 同步 | AIC: PIPE_FIX; VEC: PIPE_V | AIC: PIPE_FIX; VEC: PIPE_V |
| 2, 3, 4 | `smToMm2Flag` | Softmax→PV 同步 | VEC: PIPE_MTE3 | AIC: PIPE_FIX; VEC: PIPE_MTE3 |

```cpp
// kernel_a2.h:1044
mm1ToSmFlagId = ubSBufId;                    // 0 或 1 (UB_S_OTMP_BUF_STAGES=2)
smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;  // 2, 3, 4 (P_L1_BUF_NUM=3)
```

#### InitSyncFlags 预置

```cpp
// CUBE: 预置 smToMm2Flag (IDs 2,3,4) on PIPE_FIX
CrossCoreSetFlag<0x2, PIPE_FIX>(2);
CrossCoreSetFlag<0x2, PIPE_FIX>(3);
CrossCoreSetFlag<0x2, PIPE_FIX>(4);

// VEC: 预置 mm1ToSmFlag (IDs 0,1) on PIPE_V
CrossCoreSetFlag<0x2, PIPE_V>(0);
CrossCoreSetFlag<0x2, PIPE_V>(1);
```

**预置原因：** 流水线启动时第一批数据没有前置依赖，预置 flag 使第一批不被阻塞。

#### 跨核同步时序（每个 bi 循环）

```
CUBE (QK):                          VEC (Softmax):                    CUBE (PV):
─────────────                       ─────────────                     ─────────
                    ─── mm1ToSmFlag (ID=ubSBufId) ───
Wait mm1ToSmFlag                    Wait mm1ToSmFlag
  [if nL0Itr==0]                      [PIPE_V]
  [PIPE_FIX]
QK matmul...                        Read S from GM (MTE2)
Fixpipe S→GM                        Compute softmax
Set mm1ToSmFlag                     Write P→GM (MTE3)
  [PIPE_FIX]                         Set mm1ToSmFlag [PIPE_V]
                                                                      ─── smToMm2Flag (ID=l1PBufId+2) ───
                                                                      Wait smToMm2Flag [PIPE_FIX]
                                                                      Read P from GM (Nd2Nz)
                                                                      PV matmul...
                                                                      Set smToMm2Flag [PIPE_FIX] (for next)
                    Wait smToMm2Flag [PIPE_MTE3]
                    Set smToMm2Flag [PIPE_MTE3]
```

#### m=0 early-return 路径

当某个 subBlock 的 m=0（无数据）时，**仍必须执行全部 4 个跨核操作**：

```cpp
// block_epilogue_online_softmax_prefill_a2.hpp:570
if (m == 0) {
    WaitCrossCoreSync<0x2, PIPE_V>(mm1ToSmFlag);
    SetCrossCoreSync<0x2, PIPE_V>(mm1ToSmFlag);
    WaitCrossCoreSync<0x2, PIPE_MTE3>(smToMm2Flag);
    SetCrossCoreSync<0x2, PIPE_MTE3>(smToMm2Flag);
    return;
}
```

### 3.4 全局屏障 (SyncAll)

| 位置 | 用途 |
|---|---|
| InitSyncFlags 之后 | 确保所有核的同步标志预置完成 |
| Phase 1 结束 | 确保所有 Phase 1 task 完成，workspace 数据就绪 |
| (Phase 2 无 SyncAll) | Phase 2 仅 VEC 执行，各 task 独立无依赖 |

### 3.5 VEC 末尾 drain

```cpp
// kernel_a2.h:1131
// Phase 1 最后一批 stats scatter 是异步 UB→GM (MTE3)
// SyncAll 只同步控制流，不等 MTE3 传输完成
// 必须显式 drain，否则 Phase 2 可能读到半写的 rowMax/rowSum
SetFlag<V_MTE3>(EVENT_ID3);
WaitFlag<V_MTE3>(EVENT_ID3);
```

### 3.6 同步完整时序图

```
时间 →
bi:    0         1         2         3         4
       │         │         │         │         │
CUBE:  ┌─QK(0)─┐ ┌─QK(1)─┐ ┌─QK(2)─┐ ┌─QK(3)─┐ ┌─QK(4)─┐
       │  ↓    │ │  ↓    │ │  ↓    │ │  ↓    │ │  ↓    │
       │ flag0 │ │ flag1 │ │ flag0 │ │ flag1 │ │ flag0 │
       │  ↓    │ │  ↓    │ │  ↓    │ │  ↓    │ │  ↓    │
VEC:   └─SM(0)─┘ └─SM(1)─┘ └─SM(2)─┘ └─SM(3)─┘ └─SM(4)─┘
                    │         │         │         │         │
                    │ flag2   │ flag3   │ flag2   │ flag3   │ flag2
                    ↓         ↓         ↓         ↓         ↓
CUBE:            ┌─PV(0)─┐ ┌─PV(1)─┐ ┌─PV(2)─┐ ┌─PV(3)─┐ ┌─PV(4)─┐
                 └───────┘ └───────┘ └───────┘ └───────┘ └───────┘

flag0 = mm1ToSmFlag ID 0 (bi%2==0)
flag1 = mm1ToSmFlag ID 1 (bi%2==1)
flag2 = smToMm2Flag ID 2 (bDe%3==0)
flag3 = smToMm2Flag ID 3 (bDe%3==1)
flag4 = smToMm2Flag ID 4 (bDe%3==2)

PRE_LAUNCH=2: PV(0) 在 bi=2 开始，滞后 QK(0) 两个 batch
```

### 3.7 同步标志生命周期

```
InitSyncFlags()
  CUBE: Set MTE1_MTE2(0-3), M_MTE1(0-3), FIX_M(0-3)
        CrossCoreSet PIPE_FIX(2,3,4)    ← 预置 smToMm2
  VEC:  Set V_MTE2(0,1), MTE3_V(0-4), S_MTE3(0), MTE3_S(0,1)
        CrossCoreSet PIPE_V(0,1)        ← 预置 mm1ToSm

  ↓ Phase 1 循环 (运行时 Set/Wait)

ReleaseSyncFlags()
  CUBE: Wait MTE1_MTE2(0-3), M_MTE1(0-3), FIX_M(0-3)
  VEC:  Wait V_MTE2(0,1), MTE3_V(0-4), S_MTE3(0), MTE3_S(0,1)
```
