# arch35 FIA 非量化 MLA D512 算子详细设计文档

> **目标平台**: Ascend DAV_3510 (arch35)
> **算子**: FusedInferAttentionScore (FIA)
> **场景**: MLA (Multi-head Latent Attention) D512 非量化 Decode
> **数据类型**: fp16 / bf16
> **文档日期**: 2026-08-11

---

## 1. 概述

### 1.1 MLA 算法背景

MLA (Multi-head Latent Attention) 是 DeepSeek-V2/V3 提出的注意力机制，通过将 KV Cache 压缩到低维隐空间（latent dimension）来大幅减少 KV Cache 的内存占用。其核心特征：

- **QK 维度**: D = 576（512 基础维度 + 64 RoPE 维度），其中 Q/K 的前 512 维为 latent 压缩表示，后 64 维为解耦的 RoPE 部分
- **V 维度**: DV = 512（与 latent 维度一致）
- **K = V**: 在 MLA 场景下，Key 和 Value 共享同一份 latent 数据（GM 上同一指针），RoPE 部分单独存储
- **N2 = 1**: KV head 数为 1（MLA 的 GQA 膨胀通过 G 轴实现，N1 = G）

### 1.2 模板链路总览

| 层级 | 文件 | 核心类 | 职责 |
|------|------|--------|------|
| **Kernel 主调度** | `fia_kernel_noquant_mla.h` | `FlashAttentionNoQuantMlaKernel` | AIC/AIV 协同调度、任务分发、流水编排 |
| **Cube 块** | `fia_block_cube_noquant_mla.h` | `FANoQuantMlaBlockCube` | BMM1 (QK^T) / BMM2 (PV) Cube 计算、GM↔L1 搬运 |
| **Vec 块** | `fia_block_vec_noquant_mla.h` | `FANoQuantMlaBlockVec` | Softmax / FlashUpdate / 行无效 / 输出搬运 |
| **FD 块** | `fia_block_vec_flashdecode_mla.h` | `FiaBlockVecFlashDecodeMla` | FlashDecode 规约（LSE 合并 + 累加输出） |
| **Kernel 路由** | `fia_arch35_noquant.h` | `fia_noquant_regbase` | 根据 tilingkey 路由到 MLA/GQA kernel |
| **模板分发** | `fia_template_dispatcher.h` | `run_fia_noquant_mla_kernel` | 模板参数推导、Block 类型组装 |
| **Host Tiling** | `fia_tiling_nonquant_mla.cpp/.h` | `FiaTilingNonQuantMlaArch35` | 能力判定、分核、Tiling 数据生成 |
| **Checker** | `common_checker.cpp`, `rope_checker.cpp` | `CommonChecker`, `RopeChecker` | 输入约束校验 |

### 1.3 两条模板链路

| 链路 | 模板标识 | 路由条件 | 状态 |
|------|----------|----------|------|
| 老模板 | `isReconstructTemp=false` | `dTemplateType==Aligned576` → `FAKernelNoquantMla` | 线上在用 |
| **新模板（重构版）** | `isReconstructTemp=true` | `config==Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 && hasRope` → `run_fia_noquant_mla_kernel` | **本次解锁** |

新模板在 `fia_arch35_noquant.h:88-97` 中通过 `isReconstructTemp` 标志位区分，走 `run_fia_noquant_mla_kernel` 路径。

---

## 2. 数据维度与基本块定义

### 2.1 固定维度参数

| 参数 | 值 | 说明 |
|------|----|------|
| `mBaseSize` (S1 外切) | 64 | Q 的 GS1 方向基本块，由 `S1TemplateType::Aligned64` 编译期确定 |
| `s2BaseSize` (S2 内切) | 128 | KV 的 S2 方向基本块，由 `S2TemplateType::Aligned128` 确定 |
| `dBaseSize` (QK D) | 576 | 512 + 64 rope，由 `DTemplateType::Aligned576` 确定 |
| `dVBaseSize` (V D) | 512 | 由 `DTemplateType::Aligned512` 确定 |
| `dSizeRope` | 64 | RoPE 维度，kernel 侧当前硬编码 (`constInfo_.dSizeRope = 64`) |
| `CV_RATIO` | 2 | AIC:AIV = 1:2，M 维对半拆分给两个 Vector 核 |
| `PRELOAD_N` | 1 | 主循环预取深度：C1→C1→C2 三级流水 |
| `PRE_LOAD_NUM_MLA_ARCH35` | 2 | FD 阶段预取深度 |

### 2.2 Tiling 切分策略

| 切分维度 | 参数 | 值 | 说明 |
|----------|------|----|------|
| S2 外切 (SOuter) | `sOuterFactor_` | 32 | GS1 方向每次处理 32 个 token（对应 mBaseSize=64 = 32×2 CV_RATIO） |
| S2 内切 (SInner) | `sInnerFactor_` | 128 | KV S2 方向每次处理 128 个 token |
| FD M 切分 | `fdBalanceMBaseSize` | 8 | FD 阶段 M 方向每次处理 8 行 |

> **注意**: MLA D512 不支持 S1 外切（`enableS1OutSplit` 固定 false），直接走 `SplitCore` 分核。

---

## 3. Kernel 架构设计

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────────┐
│              FlashAttentionNoQuantMlaKernel              │
│  (fia_kernel_noquant_mla.h)                              │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐   ┌────────────┐  │
│  │  CubeBlock   │    │  VecFaBlock  │   │ VecFdBlock │  │
│  │ (AIC 核)     │    │ (AIV 核)     │   │ (AIV 核)   │  │
│  │              │    │              │   │            │  │
│  │ BMM1: QK^T  │───▶│ Softmax      │   │ FlashDecode│  │
│  │ BMM2: PV    │───▶│ FlashUpdate  │   │ Reduction  │  │
│  │ Fixpipe→UB  │    │ CopyOut      │   │            │  │
│  └──────────────┘    └──────────────┘   └────────────┘  │
│         │                  ↑                  ↑          │
│         └── CrossCore ─────┘──────────────────┘          │
│            Buffer (UB)                                    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 AIC/AIV 协同模型

arch35 采用 **AIC:AIV = 1:2** 的混合核模式（`KERNEL_TASK_TYPE_MIX_AIC_1_2`）：

- 每个 AIC（Cube 核）负责 BMM1/BMM2 的 Cube 计算
- 对应 2 个 AIV（Vector 核）负责 Softmax/FlashUpdate 等 Vector 计算
- M 维按 `CV_RATIO=2` 对半拆分：`subBlockIdx==0` 处理前半 M，`subBlockIdx==1` 处理后半 M
- AIC 与 AIV 通过 **CrossCore Buffer**（UB 层级）同步 BMM1/BMM2 结果

```cpp
// M 维拆分逻辑 (fia_kernel_noquant_mla.h:653-658)
info.actVecMSize = (info.actMSize + 1) >> 1;  // 前半
info.vecMbaseIdx = 0;
if (constInfo_.subBlockIdx == 1) {
    info.vecMbaseIdx = info.actVecMSize;
    info.actVecMSize = info.actMSize - info.actVecMSize;  // 后半
}
```

### 3.3 Buffer 管理与内存层级

```
┌─────── GM ───────┐   ┌─────── L1 ───────┐   ┌──── L0A/L0B/L0C ────┐   ┌──── UB ────┐
│                   │   │                   │   │                      │   │            │
│ queryGm           │   │ l1QBuffers (Q)   │   │ mmL0ABuffers (DB)   │   │ bmm1Buffers│
│ keyGm (=valueGm)  │──▶│ l1KvPBuffers     │   │ mmL0BBuffers (DB)   │   │ bmm2Buffers│
│ keyRopeGm         │   │  (K/V/Rope/P     │   │ mmL0CBuffers (DB)   │   │ (CrossCore)│
│ queryRopeGm       │   │   3-buffer 复用) │   │                      │   │            │
│ attenMaskGm       │   │                   │   │                      │   │ softmaxBuf │
│ attentionOutGm    │   │ l1BufferManager   │   │                      │   │ stage1/2   │
│ softmaxLseGm      │   │  (512KB)         │   │                      │   │            │
│ accumOutGm (FD)   │   │                   │   │                      │   │            │
│ softmaxFDSum/Max  │   │                   │   │                      │   │            │
└───────────────────┘   └───────────────────┘   └──────────────────────┘   └────────────┘
```

#### Buffer 分配详情 (`fia_kernel_noquant_mla.h:165-176`)

| Buffer | 层级 | 大小 | 用途 |
|--------|------|------|------|
| `l1BufferManager_` | L1 | 512KB | L1 总池 |
| `l1QBuffers` | L1 | 64×576×2B = 72KB (单 buffer) | Q + Q_rope（D 方向合并） |
| `l1KvPBuffers` | L1 | 128×576×2B × 3 = 432KB | K+K_rope / V / P（BMM2 左矩阵）3-buffer 复用 |
| `l0aBufferManager` | L0A | 64KB | L0A 总池 |
| `l0bBufferManager` | L0B | 64KB | L0B 总池 |
| `l0cBufferManager` | L0C | 256KB | L0C 总池 |
| `mmL0ABuffers` | L0A | 32KB × 2 (DB) | BMM 左矩阵 |
| `mmL0BBuffers` | L0B | 32KB × 2 (DB) | BMM 右矩阵 |
| `mmL0CBuffers` | L0C | 128KB × 2 (DB) | BMM 结果 |
| `bmm1Buffers` | UB | mBaseSize/2 × s2BaseSize × 4B × 2 (DB) | BMM1 结果 (CrossCore) |
| `bmm2Buffers` | UB | mBaseSize/2 × dVBaseSize × 4B × 2 (DB) | BMM2 结果 (CrossCore) |

#### L1 K/V/P 3-Buffer 复用机制

MLA 场景下 K=V，L1 上 K/V/Rope/P 的空间管理为 **3-buffer 前向同步**：

```
l1KvPBuffers (3-buffer, CROSS_CORE_SYNC_FORWARD):
  Buffer 0: K + K_rope (BMM1 右矩阵)
  Buffer 1: V           (BMM2 右矩阵，与 K 数据相同但需重新拷入或复用)
  Buffer 2: P           (BMM2 左矩阵，由 Vec1 计算后写入)
```

- BMM1 阶段：从 GM 拷入 K+K_rope 到当前 buffer
- BMM2 阶段：使用 K 的数据作为 V（偏移 `s2BaseSize * dVsize`），P 复用同一 buffer 空间的另一个 slot
- `GetReused()` 方法获取 P 的复用地址

### 3.4 任务调度与流水

#### 3.4.1 任务调度主循环 (`FlashAttention` 方法)

```cpp
// fia_kernel_noquant_mla.h:329-378
while (shouldDispatchTask || shouldExecuteTask) {
    shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
    if (shouldDispatchTask) {
        TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
        // CREATE_TASK / DEAL_ZERO / SKIP / NOT_START / S2_END / SKIP_S1OUT
        if (taskDealMode == CREATE_TASK) {
            CreateTask(...);   // 创建任务到 taskRunInfo[loop % CACHE_SIZE]
            UpdateAxisInfo(...);
        }
    }
    shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
    if (shouldExecuteTask) {
        ExecuteTask(executedTaskCount, taskRunInfo);
    }
}
```

#### 3.4.2 任务状态机 (`TASK_DEAL_MODE`)

| 状态 | 含义 | 处理 |
|------|------|------|
| `CREATE_TASK` | 有效任务，需创建并执行 | `CreateTask` → `EnableTask` |
| `DEAL_ZERO` | actSeqLen=0 的空 batch | AIV 侧置零输出，跳过 |
| `SKIP` | `curS2Start >= curS2End`，当前 head 无有效 S2 | 跳过 |
| `NOT_START` | `s2Cur < curS2Start` | 直接跳到 `curS2Start` |
| `S2_END` | `s2Cur >= curS2End` | 推进到下一个 GS1/BN2 |
| `SKIP_ZERO` | 非 first task 且 actSeqLen=0 | 跳过 |
| `SKIP_S1OUT` | S1OutSplit 场景非当前核任务 | 跳过 |

#### 3.4.3 PRELOAD 流水 (C1→C1→C2)

```
PRELOAD_N = 1, PRELOAD_TASK_CACHE_SIZE = 2

时间轴:  t0    t1    t2    t3    t4    ...
任务:    T0    T1    T2    T3    T4
AIC:     BMM1  BMM1  BMM1  BMM1  BMM1
                     BMM2  BMM2  BMM2  (T0的BMM2)
AIV:     Vec1  Vec1  Vec1  Vec1  Vec1
                     Vec2  Vec2  Vec2  (T0的Vec2)

→ ExecuteTask(loop):
  本轮: ComputeMm1(runInfo0)  / ComputeVec1(runInfo0)
  上PRELOAD_N轮: ComputeMm2(runInfoNegN) / ComputeVec2(runInfoNegN)
```

- **BMM1 与 Vec1 同步**: BMM1 结果通过 `bmm1Buffers` (CrossCore UB) 从 AIC 传给 AIV
- **BMM2 与 Vec2 同步**: BMM2 结果通过 `bmm2Buffers` (CrossCore UB) 从 AIC 传给 AIV
- **L1 K/V/P 同步**: `l1KvPBuffers` (CrossCore L1) 在 AIC 侧管理，Vec1 通过 `outputBuf` (L1 CrossCore) 接收 stage1 结果

#### 3.4.4 轴遍历顺序

```
for bN2Cur in [bN2Start, bN2End):        // B × N2
    for gS1Cur in [0, gS1LoopTimes):     // G × S1/mBaseSize
        for s2Cur in [curS2Start, curS2End):  // S2/s2BaseSize
            CREATE_TASK → BMM1/BMM2/Vec1/Vec2
```

- `bN2Cur = bIdx * n2Size + n2Idx`
- `gS1Cur` 遍历 `actSeqLensQ * gSize / mBaseSize` 个块
- `s2Cur` 遍历 `actSeqLensKv / s2BaseSize` 个块（受 sparse mode 裁剪）

### 3.5 Mask 与 SparseMode 处理

#### 3.5.1 SparseMode 枚举

```cpp
// vector_common.h:45
enum SparseMode : uint8_t {
    DEFAULT_MASK = 0,   // pre/next 决定 S2 遍历范围，attenMask 全量参与
    ALL_MASK,           // 1: 全量 mask
    LEFT_UP_CAUSAL,     // 2: 因果
    RIGHT_DOWN_CAUSAL,  // 3: 右下因果
    BAND,               // 4: 带状
};
```

#### 3.5.2 S2 范围计算

**无 Mask** (`CalcCurS2StartEndNoSparse`): S2 遍历全量 `[0, ceil(actSeqLensKv/s2BaseSize))`

**有 Mask** (`CalcCurS2StartEndWithSparse`):
1. 计算当前 GS1 块的 S1 首末 token 索引
2. 通过 `GetPreNextTokenLeftUp` + `GetSafeActToken` 钳制 pre/next token
3. BAND 模式: `preTokenLeftUp += actSeqLensQ - actSeqLensKv`
4. RIGHT_DOWN 模式: `nextTokenLeftUp = actSeqLensKv - actSeqLensQ`
5. 将 token 范围转为 S2 块索引: `curS2Start = s2FirstToken / s2BaseSize`, `curS2End = s2LastToken / s2BaseSize + 1`

#### 3.5.3 Mask 加载与应用

**Vec 块** (`fia_block_vec_noquant_mla.h`):
- `AttenMaskCopyIn`: 填充 `MaskInfo`（含 sparseMode、batch stride、S1/S2 stride、maskValue=-1e12、maskFormat=SG/GS）
- `IsSkipAttentionmask(DEFAULT_MASK)=false` → 全量拷贝
- `ProcessVec1Vf<..., hasAtten=true, ...>`: 在 softmax 前将 mask 叠加到 QK 结果
- `RowInvalid`: 对无效行（pre/next 负值导致的越界行）置 `-inf`

**FD 块** (`fia_block_vec_flashdecode_mla.h`):
- `CalcPreNextTokens`: `DEFAULT_MASK && attenMaskFlag` 分支计算 pre/next
- `DealInvalidRows` / `DealInvalidMaskRows`: FD 规约阶段处理无效行
- FD 不重复加载 mask（prefill 阶段已作用于累加结果）

---

## 4. Cube 块设计 (`FANoQuantMlaBlockCube`)

### 4.1 BMM1 (QK^T) 计算

```
BMM1: [M=64, K=576(512+64rope)] × [K=576, N=128] = [M=64, N=128]
```

**流程** (`IterateBmm1`):
1. 首次 S2 循环: 从 GM 拷入 Q+Q_rope 到 L1 (`CopyQueryAndRopeTile`)
2. 从 GM 拷入 K+K_rope 到 L1 (`CopyKeyAndRopeTile`)
3. `MatmulK<Q_T, KV_T, T, 64, 128, 128, MK, KN>`: L1→L0A/L0B→L0C 计算
4. `FixpipeMm1`: L0C→UB (NZ2ND, dual-dst M/2 拆分)
5. `SetCrossCore()`: 通知 AIV 侧 BMM1 结果就绪

**Q+Q_rope L1 合并存储**:
```
L1 Q buffer layout (NZ format):
  [0, dSize)         → Q (512维)
  [dSize*stride, +dSizeRope) → Q_rope (64维)
```

### 4.2 BMM2 (PV) 计算

```
BMM2: [M=64, K=128] × [K=128, N=512] = [M=64, N=512]
```

**流程** (`IterateBmm2`):
1. 使用 `kvpSharedBuf.GetReused()` 获取 P 矩阵（Vec1 计算后写入 L1）
2. V 矩阵复用 K 的数据（偏移 `s2BaseSize * VDsize`）
3. `MatmulN<Q_T, KV_T, T, 64, 128, 128, MK, KN>`: 计算
4. `FixpipeMm2PartialN`: L0C→UB (按实际 dVSize fixpipe)
5. `SetCrossCore()`: 通知 AIV 侧 BMM2 结果就绪
6. 释放 K/V buffer: `SetFlag<MTE1_MTE2>`

### 4.3 Fixpipe 配置

```cpp
// BMM1 Fixpipe (fia_block_cube_noquant_mla.h:268-285)
FixpipeParamsC310<CO2Layout::ROW_MAJOR>:
  nSize = (actSingleLoopS2Size + 7) >> 3 << 3   // 8对齐
  mSize = (actMSize + 1) >> 1 << 1              // 2对齐
  dualDstCtl = 1                                // 双目标: M/2 拆分到两个 UB
  dstStride = s2BaseSize                        // UB 行间隔

// BMM2 Fixpipe (fia_block_cube_noquant_mla.h:332-346)
  nSize = (realN + 7) >> 3 << 3                 // dV 对齐
  dstStride = (dVTemplateType + 15) >> 4 << 4   // 16对齐
  dualDstCtl = 1
```

---

## 5. Vector 块设计 (`FANoQuantMlaBlockVec`)

### 5.1 ProcessVec1 (Softmax 阶段)

**输入**: BMM1 结果 (UB, CrossCore), AttenMask (GM)
**输出**: P 矩阵 (L1, CrossCore), Softmax Max/Sum/Exp (UB)

**流程** (`ProcessVec1Nd`):
1. `AttenMaskCopyIn`: 拷入 mask 到 UB (`attenMaskInQue[loop%2]`, ping-pong)
2. `ProcessVec1Vf<T, INPUT_T, ..., hasAtten, ...>`:
   - 按 `actSingleLoopS2Size` 分支: EQ_128 / LTE_64 / GT_64_LTE_128 / GT_128_LTE_256
   - 首/非首 S2 循环区分: `isFirstS2Loop` 控制 softmax 初始化 vs 更新
   - 计算: `scale * QK + mask → softmax → cast to INPUT_T`
3. `DataCopy` stage1 结果到 L1 (`mm2AL1Tensor`)
4. `UpdateExpSumAndExpMax`: 非首循环更新 softmax 统计量
5. `SoftmaxDataCopyOut`: 末循环写出 LSE (FD/非FD 分别处理)

### 5.2 ProcessVec2 (FlashUpdate 阶段)

**输入**: BMM2 结果 (UB, CrossCore)
**输出**: AttentionOut (GM)

**流程** (`ProcessVec2OnUb`):
1. 首 S2 循环: `DataCopy` BMM2 结果到 `vec2ResUb`
2. 中间循环: `FlashUpdateNew<P, exp, vec2ResUb>`: `vec2Res = BMM2 * exp + vec2Res`
3. 末循环: `FlashUpdateLastNew<..., sumUb>`: `vec2Res = (BMM2 * exp + vec2Res) / sum`
4. 末循环特例 (首=末): `LastDivNew`: `vec2Res = vec2Res / sum`
5. `CopyOutAttentionOut`: Cast + RowInvalid + DataCopyOut

### 5.3 输出布局支持

| 输出布局 | GmFormat | OffsetCalculator | 说明 |
|----------|----------|------------------|------|
| BSH | BSNGD | bSize, n2Size, gSize, s1Size, dSizeV | B-S-N-G-D |
| BNSD | BNGSD | bSize, n2Size, gSize, s1Size, dSizeV | B-N-G-S-D |
| TND | TNGD | n2Size, gSize, dSizeV | T-N-G-D (累积S) |
| NTD | NGTD | n2Size, gSize, dSizeV | N-G-T-D (累积S) |
| NBSD | NGBSD | bSize, n2Size, gSize, s1Size, dSizeV | N-G-B-S-D |

### 5.4 Softmax Buffer 管理

```
softmaxSumBuf[PRELOAD_N+1]  = [0], [1]   (2 slots, 256B 对齐)
softmaxMaxBuf[PRELOAD_N+1]  = [0], [1]
softmaxExpBuf[PRELOAD_N+1]  = [0], [1]
maxBrdcst / sumBrdcst                   (Broadcast [S1,1]→[S2,8])
```

- 按 `mloop % (PRELOAD_N+1)` 索引，支持流水重叠
- `ACT_BYTES = (mBaseSize/CV_RATIO * sizeof(float) + 255) / 256 * 256` → 256B 对齐

---

## 6. FlashDecode 块设计 (`FiaBlockVecFlashDecodeMla`)

### 6.1 FD 触发条件

当 S2 被多核切分（`isS2SplitCore=true`）时，各核的中间结果需通过 FlashDecode 合并：

1. 各核写出 `accumOut` (M×DV float) + `softmaxFDSum` + `softmaxFDMax` 到 workspace
2. FD 阶段读取这些中间结果，按 S2 split 份数加权合并

### 6.2 FD 流程 (`FlashDecode`)

```
for fdS1gOuterMIdx in [0, fdBalanceMSplitNum):    // M 方向按 8 行切分
    LoadLseAndPreloadData(...)                      // 加载 LSE sum/max + 预取 accumOut
    ComputeScaleValue(lseExp, ...)                  // 计算每份的 exp(lseMax_i - globalMax)
    CalcPreNextTokens(...)                          // 计算 pre/next 用于行无效
    CopySoftmaxLseOutput(...)                       // 写出 LSE
    for i in [0, actualCombineLoopSize):            // 遍历 S2 split 份数
        WaitMm2Res / CopyAccumOutIn(...)            // 加载第 i 份 accumOut
        ReduceFinalRes(reduceOut, mm2Res, lseExp)   // 加权累加
    CopyFinalResOut(reduceOut, ...)                  // Cast + 行无效处理 + 写出
```

### 6.3 FD Buffer 与同步

| Buffer | 大小 | 用途 |
|--------|------|------|
| `fdSumBuf1/2` | 6KB | LSE sum (ping-pong) |
| `fdMaxBuf1/2` | 6KB | LSE max (ping-pong) |
| `fdLseExpBuf` | 6KB | exp 计算临时 |
| `fdMm2ResBuf1/2` | 16KB | accumOut 预取 (ping-pong) |
| `fdReduceBuf` | 16KB | 规约结果 |
| `fdOutputBuf` | 16KB | Cast 输出 |
| `fdLseMaxUbBuf1/2` | 256B | 全局 max (ping-pong) |
| `fdLseUbBuf` | 256B | LSE 输出 |

同步事件 ID (0-7):
- `SYNC_LSE_MAX_SUM_BUF1/2_FLAG` (0,1): LSE sum/max MTE2↔V
- `SYNC_MM2RES_BUF1/2_FLAG` (2,3): accumOut MTE2↔V
- `SYNC_FDOUTPUT_BUF_FLAG` (4): FD 输出 MTE3↔V
- `SYNC_LSEOUTPUT_BUF_FLAG` (5): LSE 输出 MTE3↔V
- `SYNC_SINK_BUF1/2_FLAG` (6,7): Sink 相关

### 6.4 FD MetaData

FD 任务的元数据通过 `fiaMetaDataGm_` 传递：

| MetaData 字段 | 含义 |
|---------------|------|
| `FD_CORE_ENABLE_INDEX` | 当前 AIV 是否参与 FD |
| `FD_BN2_IDX_INDEX` | 当前 FD 任务的 BN2 索引 |
| `FD_M_IDX_INDEX` | 当前 FD 任务的 M 块索引 |
| `FD_S2_SPLIT_NUM_INDEX` | S2 切分份数（需合并的份数） |
| `FD_M_START_INDEX` | M 方向起始行 |
| `FD_M_NUM_INDEX` | M 方向处理行数 |
| `FD_WORKSPACE_IDX_INDEX` | workspace 偏移索引 |

---

## 7. Host Tiling 设计 (`FiaTilingNonQuantMlaArch35`)

### 7.1 能力判定 (`IsCapable`)

```
IsCapable()
  ├── IsCapableBasicCheckMla()
  │   ├── emptyTensorFlag → false
  │   ├── quantMode != NO_QUANT → false
  │   └── mlaMode != ROPE_SPLIT_D512 → false
  ├── IsCapableFeatureCheckMla()
  │   ├── sysPrefixFlag / pseShiftFlag / alibiPse → false
  │   ├── qPaddingSizeFlag / kvPaddingSizeFlag → false
  │   ├── isOutQuantEnable / learnableSinkFlag → false
  │   ├── isQKVDDifferent → false
  │   └── kvStorageMode == TENSOR_LIST → false
  └── IsCapableSparseLayoutCheckMla()
      ├── sparseMode ∉ {0, 3, 4} → false    (NO_MASK / RIGHT_DOWN / BAND)
      ├── qkHeadDim != 512 → false
      ├── ropeHeadDim != 64 → false
      ├── vHeadDim != 512 → false
      ├── n2Size != 1 → false
      └── n1Size ∉ [1, 128] → false
```

> **关键设计**: MLA host 不拒绝 `sparseMode=0 && attenMaskFlag=true` 组合（与 GQA 不同），因为 MLA kernel 已支持该场景。

### 7.2 分核策略 (`SplitPolicy`)

```
SplitPolicy()
  ├── AdjustSinnerAndSouter()        // 固定 sOuter=32, sInner=128
  ├── CreateSplitInput(baseInfo, splitParam)
  │   ├── mBaseSize = sOuter * CV_RATIO = 64
  │   ├── s2BaseSize = sInner = 128
  │   ├── gS1BaseSizeOfFd = 8
  │   ├── streamK = true
  │   └── fdTolerance = 9, fdLeastBlock = 0
  ├── SplitCore(aicNum, baseInfo, splitParam, result)  // 共享分核算法
  ├── SetSplitOutput(result)          // 生成 FA/FD MetaData
  ├── CalcNumBlocks(usedCoreNum)
  ├── flashDecodeFlag_ = (fdNum > 0)
  └── isExistRowInvalid = IsExistRowInvalid(baseInfo)
```

### 7.3 TilingKey 生成

```cpp
// fia_tiling_nonquant_mla.cpp:460-503
UpdateTilingKeyInfo():
  inputLayout: BNSD_BNSD / BSH_BSH / TND_TND / NTD_NTD
  config: Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 (= 9)
  pseMode: PSE_MODE_PSE_NONE_TYPE (= 9)
  quantMode: NoQuantMode (= 31)
  hasAttenMask: fiaInfo_->attenMaskFlag (false / true)
  hasRope: true
  kvLayoutType: 0(无PA) / 1(BBH) / 2(BNBD) / 3(NZ)
  isFd: flashDecodeFlag_
  emptyTensor: false
  enableKvPrefix: false
  isReconstructTemp: true    ← 关键标志，区分新/老模板

tilingKey = GET_TPL_TILING_KEY(inputLayout, config, pseMode, quantMode,
                                hasAttenMask, hasRope, kvLayoutType, isFd,
                                emptyTensor, enableKvPrefix, false, isReconstructTemp)
```

### 7.4 Workspace 计算

```cpp
// fia_tiling_nonquant_mla.cpp:514-534
workspaceSize = defaultSysWorkspaceSize  // 系统预留

if (flashDecodeFlag_):
  faTmpAttenGmSize = numBlocks_ * PRE_LOAD_NUM(2) * mSize(64) * dSize(512)
  faTmpResLseGmSize = numBlocks_ * PRE_LOAD_NUM(2) * mSize(64) * lseSize(8)
  workspaceSize += (faTmpAttenGmSize + 2 * faTmpResLseGmSize) * sizeof(float)
  // 2份 LSE: sum 和 max
```

### 7.5 模板注册

```cpp
// fia_tiling_nonquant_mla.cpp:760-762
REGISTER_TILING_TEMPLATE_FIA(FusedInferAttentionScore,
    FiaTilingNonQuantMlaArch35,
    {NpuArch::DAV_3510},
    5);  // 优先级 005: 0xx-非量化, x0x-mla, 个位5
```

优先级编码规则：百位=量化模式(0非量化)，十位=算子类型(0=mla)，个位=特化程度。

---

## 8. Checker 修改设计

### 8.1 修改背景

原 checker 对 MLA D512 的 G 轴和 N1(head_num) 限制为 **2 的幂集合** `{1, 2, 4, 8, 16, 32, 64, 128}`。arch35 非量化模板的 host `IsCapableSparseLayoutCheckMla` 放行 N1 ∈ [1, 128] 任意整数，导致 checker 与 host 约束不一致。

### 8.2 修改内容

#### 8.2.1 `common_checker.cpp` - G 轴检查 (`CheckAxis`)

```cpp
// 修改前: 仅允许 2 的幂
if (SUPPORT_G_IN_IFAMLA.find(fiaInfo.gSize) == SUPPORT_G_IN_IFAMLA.end()) → reject

// 修改后: arch35 非量化放宽
bool isArch35NonQuant = (enableNonQuant_ && fiaInfo.npuArch == NpuArch::DAV_3510);
if (isArch35NonQuant) {
    if (fiaInfo.gSize < 1U || fiaInfo.gSize > 128U) → reject  // [1, 128] 任意整数
} else {
    // 原有 2 的幂检查
}
```

#### 8.2.2 `common_checker.cpp` - HeadNum 检查 (`CheckHeadNum`)

同上模式，N1 从 `{1,2,4,8,16,32,64,128}` 放宽到 `[1, 128]` 任意整数。

#### 8.2.3 `rope_checker.cpp` - N 轴检查 (`CheckNSize`)

同上模式，N1 从 `{1,2,4,8,16,32,64,128}` 放宽到 `[1, 128]` 任意整数。

### 8.3 判定条件设计

```cpp
bool isArch35NonQuant = (enableNonQuant_ && fiaInfo.npuArch == NpuArch::DAV_3510);
```

- **正向判定**: 使用 `npuArch == DAV_3510` 而非 `socVersion != 910B`，与 checkers 既有惯例一致
- **与其他 checker 一致**: `enableNonQuant_` 由 `FiaChecker` 在初始化时根据量化模式设置
- **MLA decode 下 N2==1**: 故 G==N1，两处检查同步放宽

---

## 9. TilingKey 模板恢复设计

### 9.1 修改内容

`fused_infer_attention_score_template_tiling_key.h` 约 1457-1473 行，取消注释"重构MLA模板tilingkey"段：

```cpp
ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST,
        InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH,
        InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),       // 4 layouts
    ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST,
        Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),  // 1 config
    ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
    ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
    ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),              // 2 mask
    ASCENDC_TPL_BOOL_SEL(HasRope, true),
    ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST,
        KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),  // 4 KV
    ASCENDC_TPL_BOOL_SEL(IsFd, false, true),                      // 2 FD
    ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
    ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
    ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
    ASCENDC_TPL_BOOL_SEL(IsReconstructTemp, true),
    ASCENDC_TPL_TILING_STRUCT_SEL(FusedInferAttentionScoreTilingData)
)
```

### 9.2 实例化矩阵

**全量恢复**: 4 layout × 2 mask × 4 KV × 2 isFd = **64 个实例** × 2 dtype(half/bf16) = **128 个 kernel 实例**

### 9.3 路由一致性保证

| 层级 | 标志位 | 位置 |
|------|--------|------|
| Host `GenTilingKey` | `isReconstructTemp=true`, `hasRope=true`, `config=9` | `fia_tiling_nonquant_mla.cpp:484,468,429` |
| Kernel 路由 | `isReconstructTemp==true && config==9 && hasRope` | `fia_arch35_noquant.h:88-89` |
| TilingKey 表 | `IsReconstructTemp=true, HasRope=true, Config=9` | `fused_infer_attention_score_template_tiling_key.h:1457` |

三处必须一一对应，任何枚举不一致会导致 tilingkey 查表失败。

---

## 10. 执行路径示例

### 10.1 sparseMode=0 + mask 场景（本次解锁的核心场景）

```
Host:
  IsCapable() → true (sparseMode=0 放行)
  SplitPolicy() → SplitCore(DEFAULT_MASK 已支持)
  IsExistRowInvalid() → 置位
  FillTiling(sparseMode=0, pre/next=INT_MAX, mask维度, isRowInvalidOpen)
  GenTilingKey(hasAttenMask=1, hasRope=1, isReconstructTemp=1)

Kernel:
  run_fia_noquant_mla_kernel<half/bf16, ..., hasAttenMask=true, ...>
  → FlashAttentionNoQuantMlaKernel::InitConstInfo(sparseMode=0, mask维度)
  → GetTaskDealMode: HAS_MASK → CalcCurS2StartEndWithSparse
      pre/next 经 GetSafeActToken 钳制 → S2 范围≈全量(INT_MAX)
  → ProcessVec1: AttenMaskCopyIn(IsSkipAttentionmask=false → 全量拷贝)
      → ProcessVec1Vf<hasAtten=true> 叠加 mask(-1e12) 后 softmax
  → ProcessVec2 → RowInvalid(必要时) → CopyOut
```

### 10.2 FD 场景 (S2 多核切分 + mask)

```
Host:
  SplitCore → fdRes.fdNum > 0 → flashDecodeFlag_=true
  Workspace: accumOut + softmaxFDSum/Max

Kernel FA 阶段:
  各核独立计算 → isS2SplitCore=true
  → Bmm2ResForFDCopyOut: 写 accumOut 到 workspace
  → ComputeLogSumExpAndCopyToGm: 写 softmaxFDSum/Max

Kernel FD 阶段:
  FlashDecode() → 读取 FD MetaData
  → LoadLseAndPreloadData: 加载各核 LSE
  → ComputeScaleValue: 计算 exp(max_i - globalMax) 权重
  → ReduceFinalRes: 加权累加各核 accumOut
  → DealInvalidRows / DealInvalidMaskRows: 行无效处理
  → CopyFinalResOut: Cast + 输出
```

---

## 11. 风险与注意事项

| 风险项 | 说明 | 缓解措施 |
|--------|------|----------|
| **编译产物膨胀** | 全量恢复 128 个 kernel 实例 | 首期可裁剪 layout/KV/isFd 组合，验证后逐步扩展 |
| **TilingKey 枚举漂移** | host `GenTilingKey` 与 kernel 模板表必须一一对应 | 恢复时逐位核对 11 个字段；建议写单测断言 tilingkey 数值 |
| **Mask UB 缓冲容量** | `attenMaskInQue` 各 4096B = 32行×128列，仅适配当前 config | 若未来扩展 S2=256 需扩到 8192B（GQA 模板即为 8192B） |
| **与老模板路由竞争** | 解锁后 fia\*mla 与 `flash_attention_*_mla` 并存 | 核对模板注册表优先级，按灰度策略切换 |
| **dSizeRope 硬编码** | `constInfo_.dSizeRope = 64` 写死在 kernel | 需改为从 tiling 传入 |
| **PRELOAD_N=1** | MLA 老模板使用 preload 1，新模板也是 1（GQA 用 2） | 需评估能否切到 2 提升流水重叠 |
| **Checker 正向判定** | `npuArch == DAV_3510` 正向判定 | 确保新芯片型号不会误入放宽分支 |

---

## 12. 文件变更清单

### 12.1 已修改文件（工作区，未提交）

| 文件 | 变更类型 | 核心改动 |
|------|----------|----------|
| `op_host/arch35/fia_tiling_nonquant_mla.cpp` | 修改 | 删除 `IsCapable()` 中的 `return false;` 硬禁用 |
| `op_kernel/arch35/fused_infer_attention_score_template_tiling_key.h` | 修改 | 取消注释"重构MLA模板tilingkey"段（64实例全量恢复） |
| `op_host/checkers/common_checker.cpp` | 修改 | arch35 非量化 G 轴/N1 放宽到 [1,128] 任意整数 |
| `op_host/checkers/rope_checker.cpp` | 修改 | arch35 非量化 N1 轴放宽到 [1,128] 任意整数 |

### 12.2 新增文件（未跟踪）

| 文件 | 说明 |
|------|------|
| `op_kernel/arch35/fia_mla_sparsemode0_mask_design.md` | sparseMode=0 + mask 方案设计文档（前置分析） |
| `build_mla.sh` | MLA 编译脚本 |

### 12.3 未修改但已就绪的 Kernel 文件

| 文件 | 说明 |
|------|------|
| `fia_kernel_noquant_mla.h` | Kernel 主调度，mask 机制已就绪 |
| `fia_block_cube_noquant_mla.h` | Cube 块，与 mask 无关 |
| `fia_block_vec_noquant_mla.h` | Vec 块，mask 加载/应用/行无效已实现 |
| `fia_block_vec_flashdecode_mla.h` | FD 块，行无效处理已实现 |
| `fia_template_dispatcher.h` | 模板分发，`hasAttenMask` 已透传 |
| `fia_arch35_noquant.h` | Kernel 路由，reconstruct 分支已就绪 |

---

## 13. 支持范围

### 13.1 平台与量化模式

| 项目 | 支持值 | 来源 |
|------|--------|------|
| NPU 架构 | DAV_3510 (arch35) | 模板注册 `REGISTER_TILING_TEMPLATE_FIA` |
| 量化模式 | 非量化 (NoQuant) | `IsCapableBasicCheckMla`: `quantMode != NO_QUANT → false` |
| MLA 模式 | ROPE_SPLIT_D512 | `IsCapableBasicCheckMla`: `mlaMode != ROPE_SPLIT_D512 → false` |
| 数据类型 (Q/K/V) | fp16 / bf16 | `fia_arch35_noquant.h` dtype 条件编译 |
| 输出数据类型 | fp16 / bf16 / int8 / hifloat8 / fp8_e4m3 | `fia_arch35_noquant.h` 各 `ORIG_DTYPE_ATTENTION_OUT` 分支 |

### 13.2 维度约束

| 维度 | 支持值 | 约束来源 |
|------|--------|----------|
| QK HeadDim (D) | 512 | `IsCapableSparseLayoutCheckMla`: `qkHeadDim != DSIZE_512 → false` |
| RoPE HeadDim | 64 | `ropeHeadDim != MLA_ROPE_D_DIM_64 → false` |
| V HeadDim (DV) | 512 | `vHeadDim != DSIZE_512 → false` |
| 实际 BMM1 K | 576 (512 + 64 rope) | 编译期 `DTemplateType::Aligned576` |
| N2 (KV head) | 1 | `n2Size != 1 → false` |
| N1 / G (Q head) | [1, 128] 任意整数 (arch35 非量化) | checker 放宽: `isArch35NonQuant` 分支 |
| S1 (Q seq len) | >= 1 | `common_checker.cpp:624`: `s1Size < 1 → reject` |
| S2 (KV seq len) | >= 0 (0 为空 batch) | `IsExistRowInvalid` 空 tensor 处理 |
| B (batch) | (0, 65536] | `common_checker.cpp:569`: `bSize > B_LIMIT(65536) → reject` |

### 13.3 布局支持

#### 输入布局 (Q/K/V)

| 布局 | 支持 | TilingKey 枚举 |
|------|------|----------------|
| BNSD | yes | `InOutLayoutType_BNSD_BNSD` (0) |
| BSH / BSND | yes | `InOutLayoutType_BSH_BSH` (1) |
| TND | yes | `InOutLayoutType_TND_TND` (2) |
| NTD | yes | `InOutLayoutType_NTD_NTD` (4) |

#### 输出布局

| 输出布局 | 支持 | GmFormat |
|----------|------|----------|
| BSH | yes | BSNGD |
| BNSD | yes | BNGSD |
| TND | yes | TNGD |
| NTD | yes | NGTD |
| NBSD | yes | NGBSD |

#### KV 存储布局

| KV 布局 | TilingKey 枚举 | 支持 | 说明 |
|---------|----------------|------|------|
| 非 PA (连续) | `KvLayoutType_NO_PA` (0) | yes | 标准 BNSD/TND 连续 KV |
| PA BBH | `KvLayoutType_PA_BBH` (1) | yes | Paged Attention, [B, B, H] |
| PA BNBD | `KvLayoutType_PA_BNBD` (2) | yes | Paged Attention, [B, N, B, D] |
| PA NZ | `KvLayoutType_PA_NZ` (3) | yes | Paged Attention, NZ 格式 |
| TensorList | — | **no** | `kvStorageMode == TENSOR_LIST → false` |

> PA blockSize 约束: [16, 1024]，且 16 对齐 (`BLOCK_SIZE_MAX_FOR_NO_QUANT=1024`, `BLOCK_SIZE_ALIGN_SIZE_16=16`)

### 13.4 SparseMode 与 Mask 支持

| SparseMode | 枚举值 | 支持 | Mask 要求 |
|------------|--------|------|-----------|
| NO_MASK / DEFAULT | 0 | yes | 可不传 mask；传 mask 则全量参与（pre/next 决定 S2 窗口） |
| ALL_MASK | 1 | **no** | `IsCapableSparseLayoutCheckMla` 拒绝 |
| LEFT_UP_CAUSAL | 2 | **no** | `IsCapableSparseLayoutCheckMla` 拒绝 |
| RIGHT_DOWN_CAUSAL | 3 | yes | mask 必须 [2048, 2048] 精确尺寸 |
| BAND | 4 | yes | mask 必须 [2048, 2048] 精确尺寸 |
| TREE | 9 | **no** | `IsCapableSparseLayoutCheckMla` 拒绝 |

#### Mask 参数约束（sparseMode=0 + mask 场景）

| 约束项 | 要求 | 来源 |
|--------|------|------|
| Mask dtype | bool / int8 / uint8 | `mask_checker.cpp` |
| Mask format | ND / NCHW / NHWC / NCDHW | `mask_checker.cpp` |
| Mask 维数 | 3D `[B或1, >=S1, >=S2]` 或 4D `[B或1, 1, >=S1, >=S2]` | arch35 拒绝 2D mask |
| Mask batch | {1, B} | `mask_checker.cpp` |
| Mask 4D N 维 | == 1 | `mask_checker.cpp` |
| Mask S1 下限 | >= s1Size (decode 即 >=1) | `mask_checker.cpp:538` |
| Mask S2 下限 | >= maxActualseq (非 PA) 或 s2Size (PA) | `mask_checker.cpp:538` |
| preTokens + nextTokens | >= 0 | `mask_checker.cpp:308` |
| preTokens 默认 | INT_MAX | parser 未传时置 `SPARSE_MODE_INT_MAX` |
| nextTokens 默认 | INT_MAX | 同上 |

> sparseMode=0 不传 mask: 合法，跳过全部 mask 校验，pre/next 不生效（走 `CalcCurS2StartEndNoSparse`，S2 固定全量）。

### 13.5 功能特性支持

| 特性 | 支持 | 约束来源 |
|------|------|----------|
| AttenMask | yes | `hasAttenMask: false / true` 双路径 |
| RoPE (query_rope / key_rope) | yes (必须) | `hasRope=true` 固定，`UpdateTilingKeyHasRope` |
| FlashDecode (isFd) | yes | S2 多核切分时自动触发 |
| softmaxLse 输出 | yes | `isSoftmaxLseEnable` 从入参解析 |
| S1 外切 (enableS1OutSplit) | **no** | 固定 false，走 `SplitCore` 而非 `SplitS1OutCore` |
| 空 Tensor (emptyTensor) | **no** | `IsCapableBasicCheckMla`: `emptyTensorFlag → false` |
| 公共前缀 (sysPrefix) | **no** | `IsCapableFeatureCheckMla`: `sysPrefixFlag → false` |
| PSE Shift | **no** | `pseShiftFlag → false` |
| Alibi PSE | **no** | `enableAlibiPse → false` |
| Q 左 Padding | **no** | `qPaddingSizeFlag → false` |
| KV 左 Padding | **no** | `kvPaddingSizeFlag → false` |
| 后量化 (PostQuant) | **no** | `isOutQuantEnable → false` |
| Learnable Sink | **no** | `learnableSinkFlag → false` |
| D 不等长 (isQKVDDifferent) | **no** | `isQKVDDifferent → false` |
| KV TensorList | **no** | `kvStorageMode == TENSOR_LIST → false` |
| 行无效 (RowInvalid) | yes | `isRowInvalidOpen = (innerPrecise >> 1) & 1` |
| ViewStride (非连续 KV) | yes | `fiaInfo_->hasViewStride` 时写入 strides |

### 13.6 Tiling 切分参数

| 参数 | 值 | 说明 |
|------|----|------|
| SOuter (S1 外切) | 32 | `sOuterFactor_ = SOUTER_32`，GS1 方向每次 32 token |
| SInner (S2 内切) | 128 | `sInnerFactor_ = SINNER_128`，KV S2 方向每次 128 token |
| mBaseSize | 64 | `sOuter * CV_RATIO = 32 * 2`，Cube M 基本块 |
| s2BaseSize | 128 | Cube N 基本块 |
| FD M 切分 | 8 | `fdBalanceMBaseSize = 8`，FD 阶段 M 方向每次 8 行 |
| FD gS1BaseSize | 8 | `splitParam.gS1BaseSizeOfFd = 8` |
| PRE_LOAD_NUM | 2 | FD 阶段预取深度 |
| StreamK | true | `splitParam.streamK = true`，流式 K 分配 |

### 13.7 编译期实例化矩阵

| 维度 | 取值数 | 值 |
|------|--------|----|
| InOutLayoutType | 4 | BNSD, BSH, TND, NTD |
| Config | 1 | S1Aligned64_S2Aligned128_DAligned576_DVAligned512 (9) |
| PseMode | 1 | PSE_NONE_TYPE (9) |
| QuantMode | 1 | NoQuantMode (31) |
| HasAttenMask | 2 | false, true |
| HasRope | 1 | true (固定) |
| KvLayoutType | 4 | NO_PA, PA_BBH, PA_BNBD, PA_NZ |
| IsFd | 2 | false, true |
| EmptyTensor | 1 | false (固定) |
| EnableKVPrefix | 1 | false (固定) |
| EnableS1OutSplit | 1 | false (固定) |
| IsReconstructTemp | 1 | true (固定) |
| **单 dtype 实例数** | **64** | 4 × 2 × 4 × 2 |
| **总实例数 (含 half/bf16)** | **128** | 64 × 2 |

---

## 14. 待完成事项

1. **测试用例补充**: sparseMode=0 + mask（3D/4D mask）、显式 pre/next（如 0/0 对角窗口）、innerPrecise 行无效、isFd+mask 组合
2. **回归验证**: sparseMode=0 无 mask（保持现状行为不劣化）、sparseMode=3/4（已有放行）不劣化
3. **精度对照**: 与老模板（`flash_attention_*_mla`）同参数结果对齐
4. **编译验证**: 验证 128 个实例的编译时间与二进制增量可控
5. **dSizeRope 参数化**: 将 `constInfo_.dSizeRope = 64` 改为从 tiling 数据传入
6. **PRELOAD_N 评估**: 评估 MLA 模板能否切换到 PRELOAD_N=2 以提升流水重叠

---

## 附录 A: 关键数据结构

### A.1 RunInfoX (任务描述)

```cpp
struct RunInfoX {
    uint32_t loop;           // 全局循环计数
    uint32_t mloop;          // M 方向循环计数（softmax buffer 索引）
    bool isValid;            // 任务是否有效
    bool isFirstS2Loop;      // 是否为当前 head 的首个 S2 内循环
    bool isLastS2Loop;       // 是否为末个 S2 内循环
    uint32_t bIdx, n2Idx;    // batch / N2 索引
    uint32_t gS1Idx, s1Idx;  // GS1 / S1 索引
    uint32_t s2Idx;          // S2 索引
    uint32_t actMSize;       // 实际 M 大小（可能 < mBaseSize，尾块）
    uint32_t actVecMSize;    // Vector 视角 M 大小（actMSize / 2）
    uint32_t vecMbaseIdx;    // Vector M 起始偏移（0 或 actVecMSize）
    uint32_t actSingleLoopS2Size;  // 实际 S2 大小（可能 < s2BaseSize，尾块）
    bool isS2SplitCore;      // S2 是否被多核切分（触发 FD）
    uint32_t faTmpOutWsPos;  // workspace 写出位置索引
    int64_t preTokensLeftUp, nextTokensLeftUp;  // mask 窗口参数
};
```

### A.2 FiaMlaTilingKeyInfo (TilingKey 字段)

```cpp
struct FiaMlaTilingKeyInfo {
    uint64_t inputLayout;     // InOutLayoutType (0-5)
    uint64_t config;          // Config (9 = MLA D512)
    uint64_t pseMode;         // PSE_MODE_PSE_NONE_TYPE (9)
    uint64_t quantMode;       // NoQuantMode (31)
    bool hasAttenMask;        // false / true
    bool hasRope;             // true (固定)
    uint64_t kvLayoutType;    // 0/1/2/3
    bool isFd;                // false / true
    bool emptyTensor;         // false
    bool enableKvPrefix;      // false
    bool isReconstructTemp;   // true (固定，区分新/老模板)
};
```

### A.3 FiaMetaData (核间元数据)

```
FA Metadata (per AIC, 8 fields):
  [0] FA_CORE_ENABLE_INDEX
  [1] FA_BN2_START_INDEX
  [2] FA_M_START_INDEX
  [3] FA_S2_START_INDEX
  [4] FA_BN2_END_INDEX
  [5] FA_M_END_INDEX
  [6] FA_S2_END_INDEX
  [7] FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX

FD Metadata (per AIV, 8 fields):
  [0] FD_CORE_ENABLE_INDEX
  [1] FD_BN2_IDX_INDEX
  [2] FD_M_IDX_INDEX
  [3] FD_S2_SPLIT_NUM_INDEX
  [4] FD_WORKSPACE_IDX_INDEX
  [5] FD_M_START_INDEX
  [6] FD_M_NUM_INDEX
  [7] (reserved)
```

---

## 附录 B: Config 索引 9 的含义

```cpp
// fia_arch35_template_tiling_key_enum.h:136-137
ConfigValue[9] = {
    inferS1TemplateType::Aligned64,     // s1 = 64
    inferS2TemplateType::Aligned128,    // s2 = 128
    inferDTemplateType::Aligned576,     // d = 576 (512 + 64 rope)
    inferDTemplateType::Aligned512      // dv = 512
};
```

这是 MLA D512 的唯一 config，与其他 GQA/全量化 config 区分。`fia_arch35_noquant.h:89` 的路由条件 `config==9 && hasRope` 精确匹配此 config。
