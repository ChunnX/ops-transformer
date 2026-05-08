# Incre Flash Attention Preload DD Kernel 计算流程与优化分析

## 1. 概述

`IncreFlashAttentionAttenPreloadDD` 是增量式 Flash Attention 算子在 Ascend 910B（arch32 / DAV_2201）NPU 上的核心 Kernel，整体采用 **Vector+Cube 异构流水** 架构，AIV 与 AIC 核心通过 CrossCore 同步机制协同工作完成 Attention 计算。

中间计算数据类型均为 `float`（高精度模式），外部输入/输出 Quant 类型通过 IFAT 模板参数在编译期决议，消除运行时分支开销。

**适用场景**：GQA + PageAttention + BF16 量化（Q=B F16, KV=INT8）+ NZ 布局 KV Cache 的反量化特性场景。该场景下 Cube 侧矩阵乘发块必须精细控制，通过与 Vector 侧的流水重叠掩盖反量化开销。

## 2. 整体计算流程

```
           AIV (Vector)                        AIC (Cube)
    ┌──────────────────────┐          ┌──────────────────────┐
    │  1. Q PreProcess     │          │                      │
    │  (反量化 Q 到 GM)     │ ──sync──>│                      │
    │                      │          │  2. BMM1: Q × K^T   │
    │                      │   sync<──│  (Mmad + Fixpipe)    │
    │  3. Vec1 Process     │          │                      │
    │  - 读出 BMM1 结果     │          │                      │
    │  - Scale + AttnMask  │          │                      │
    │  - SoftmaxFlashV2    │          │                      │
    │  - Antiquant P 到 GM │ ──sync──>│                      │
    │                      │          │  4. BMM2: P × V     │
    │                      │   sync<──│  (Mmad + Fixpipe)    │
    │  5. Vec2 Process     │          │                      │
    │  - 累积更新 O         │          │                      │
    │  - ScaleV / Output   │          │                      │
    └──────────────────────┘          └──────────────────────┘
```

### 2.1 详细阶段描述

#### 阶段 0: Q 预量化 (QueryPreProcess) — AIV
- 仅在每个 Batch 的第一个 S Inner Loop 执行
- 从 GM 读取 Q 矩阵，根据反量化模式执行：
  - **PerChannel (ANTIQUANT_PER_CHANNEL)**：单 scale 参数，逐 channel 反量化
  - **PerToken (ANTIQUANT_PER_TOKEN)**：逐 token scale 参数反量化
- 使用 AbsRowMax 计算每行最大值，然后用 `127/(1.001*Amax)` 做归一化缩放
- MSD 迭代展开：将 FP32 → FP16 → INT8，通过两次迭代（`msdIterNum=2`）实现高精度转换，写入 workspace 的 `queryPreProcessResGm`
- **非 ANTIQUANT 场景**：Q 直接使用原始类型参与 Cube 计算（不经过预量化）

#### 阶段 1: BMM1 (Q × K^T) — AIC (Cube)
- **矩阵规模**：Q [mSize, headDim] × K^T [sInnerSize, headDim]^T → S [mSize, sInnerSize]
- **双 MSD 反量化**（仅在 ANTIQUANT 模式）：
  - K 矩阵从 GM 载入 L1，L1→L0 通过 `LoadData` 完成
  - Q 矩阵（已预量化存储在 workspace）载入 L1，L1→L0 通过 `LoadData`
  - Mmad INT8 矩阵乘（m×k, k×n）, 硬件 INT8 乘结果为 INT32
  - Fixpipe 反量化输出为 F16（两个 MSD 段分别输出 with/without AtomicAdd）
  - 量化因子：`scaleC1S1 = 1/1024`, `scaleC1S2 = 1/(1024*254)`
- **非 ANTIQUANT 模式**：直接使用原始数据类型做 Mmad

#### 阶段 2: Vec1 (Softmax + P 后处理) — AIV
- 从 workspace 的 `mm1ResGm` 读取 BMM1 的 F16 中间结果，Cast 为 F32
- 反量化还原：`S = S_cube * scaleC1 * attnScale * Amax`
- **ElewiseCompute**：
  - 乘 attention scale（非 ANTIQUANT 场景）
  - Attention Mask 处理（`SelectWithBytesMask`，mask 值填充为 -1e12）
- **SoftmaxFlashV2Compute**：在线安全 Softmax
  - 使用上一轮的 max/sum 作为在线更新初值
  - 输出 exp(S)*scale 覆盖回原输入
  - 保存新 max/sum（4 个 slot 轮转）
- **P 后处理**：
  - ANTIQUANT_PER_CHANNEL：`AntiquantSoftmaxResPreProcess` → 写入 `vec1ResGm`
  - ANTIQUANT_PER_TOKEN：乘 key/value scale + `AntiquantMatmulPreProcess` → 写入 `vec1ResGm`
  - 非 ANTIQUANT：直接 Cast F32→KV_T，写入 `vec1ResGm`
- Flash Decode 模式：最后一轮额外调用 `ComputeLogSumExpAndCopyToGm` 输出 LSE

#### 阶段 3: BMM2 (P × V) — AIC (Cube)
- **矩阵规模**：P [mSize, sInnerSize] × V [sInnerSize, headDim] → O [mSize, headDim]
- 与 BMM1 类似的 L1/L0 数据搬运 + Mmad + Fixpipe 流程
- 量化因子：`scaleC2O1 = 1/(1024*127)`, `scaleC2O2 = 1/(1024*254*127)`
- 同样支持双 MSD 反量化输出

#### 阶段 4: Vec2 (O 累积与输出) — AIV
- 从 workspace 的 `mm2ResGm` 读取 BMM2 的 F16 中间结果
- **在线累积**（Flash Attention 递推公式）：
  - 第 1 轮：`O_current = BMM2_result`
  - 第 i 轮：`O_current = O_prev * exp(Amax_old - Amax_new) + BMM2_result`
  - exp_sum_ratio 来自 softmax 阶段保存的 exp(Amax_old - Amax_new)
- 最后一轮执行输出归一化：`O = O / softmax_sum`
- **后处理**：
  - 反量化：乘 scaleV + 加 offsetV + scaleC2
  - Cast + DealInvalidRows（处理 s1 < s2 的无效位置填 0）+ DataCopy 输出
- 中间结果暂存：非最后一轮时 `O_current` 保存在 `vec2ResUb` 中供下一轮使用

### 2.2 Flash Decode 模式

Flash Decode 是 Split-KV 的特殊模式：
- 每个 AICore 处理一个 (batch, head) 对的不同 S 段
- 所有段做完后 `SyncAll()` 多核同步
- AIV 负责 Combine 阶段：
  - 从 workspace 读取各段的 `accumOut` 和 LSE
  - `ComputeScaleValue`：合并 LSE（Max + Exp + Sum）
  - `ReduceFinalRes`：加权累加各段输出
  - `CopyFinalResOut`：Cast + DataCopy 最终结果

## 3. Tiling 切块与核分配逻辑

> 以下分析基于 `ops-transformer/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp`

### 3.1 多级切块策略

Tiling 在 Host 侧完成，核心目标是将 Attention 计算分配到多个 Core 并控制单次计算规模在 UB/L1 限制内。主要维度的切分策略：

```
Attention 原始规模: [BatchSize, numHeads(Q), kvHeads(KV), QSeqLen, KVSeqLen, headDim]
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              G 轴切分              S1(Q) 轴切分         S2(KV) 轴切分
          (groupSplitSize_)      (s1SplitSize_)        (sInnerSize_)
```

#### 3.1.1 S2(KV) 方向切分 —— `sInnerSize_`

S2 切分是控制 **BMM1 矩阵乘规模**（N=K 维）的关键参数，直接影响单次 MM 的 L0C 容量和向量临时缓冲大小。

调用路径：`CalcInnerSize(seqSize)` → 以下分支逻辑：

| 场景 | sInnerSize | 理由 |
|------|-----------|------|
| 非 GQA，非伪量化 | **8192** | Vector 基本块 8×1024，Cube 发大块减少通信 |
| GQA (n>1) + 伪量化 + CUBE_VIEW_MM | **2048** | Vector 较重，Cube 发中块 |
| GQA (n>1) + 伪量化 + 其他模式 | **1024** | Vector 很重，Cube 发小块掩盖 Vector 延迟 |
| n==1 + 伪量化 + DD 模式 | **1024** | DD 模式下 sInner 固定 1024 |
| GQA (n>1) + 非伪量化 | **8192/4096/2048** | 根据 `nNumOfQInOneGroup_/5` 分三档，控制 workspace ≤32MB |
| MLA (rope) + 非伪量化 | **512** | MLA 场景基础切分 |
| MLA (rope) + 伪量化 | **1024** | 伪量化向量计算重 |
| 310P + M>1 | 计算值 `min(40K/g/s1/4B, 8192)`，128 对齐 | Cube buffer 限制 |
| Page Attention | 向下对齐到 blockSize 倍数 | 避免跨 block 搬运复杂度 |

```
sInnerLoopTimes = ceil(seqSize / sInnerSize)
sInnerSizeTail  = seqSize - (sInnerLoopTimes-1) * sInnerSize
sInnerSizeAlign = Align(sInnerSize, 32)   // INT4: Align(sInnerSize, 64)
```

#### 3.1.2 S1(Q) 方向切分 —— `s1SplitSize_`

在 BSH/BSND/TND 等 layout 下生效，切分 Q 的 S 维度。

- **MLA 模式**：`s1SplitSize_ = gMax_ / nNumOfQInOneGroup_`
  - MLA 3buf: `gMax_ = 512`；MTP: `gMax_ = 128(伪量化) / 256(非伪量化)`
  - 限制 `s1SplitSize_ ≤ qSeqSize`
- **Balanced 负载均衡**：`s1SplitSize_ = FIA_BALANCE_SG_BASIC_SIZE / gSize`
- **非 MLA/均衡**：`s1SplitSize_ = qSeqSize`（不切）
- `s1Outer_ = ceil(qSeqSize / s1SplitSize_)`

#### 3.1.3 G 轴切分 —— `groupSplitSize_`

在 BNSD 等 layout 下生效，切分 Q 的 Group 维度。

- **MLA 模式**：`groupSplitSize_ = gMax_ / qSeqSize_`，向下对齐到偶数
  - 限制 `groupSplitSize_ ≤ nNumOfQInOneGroup_`
- **非 MLA**：`groupSplitSize_ = nNumOfQInOneGroup_`（不切，默认）
- `gOuter_ = ceil(nNumOfQInOneGroup_ / groupSplitSize_)`

> 每个 Core 上处理的 M 维大小 `cubeMSize = groupSplitSize × s1SplitSize`（MLA 按需取 min 到 gMax）

### 3.2 核分配策略

#### 3.2.1 可用 Core 数量确定

```
核心数 = AIV 数量（默认，Vector 核作为调度主体）
       = AIC 数量（CUBE_VIEW_MM / DD / MLA / BALANCE 模式）
```

| 模式 | 核类型 | 说明 |
|------|--------|------|
| BMM_ALL_BY_VEC | AIV | 纯向量计算 |
| CUBE_VIEW_MM / DD | AIC | Cube 侧主导调度，AIV 伴随 |
| CUBE_VIEW_MM_MLA / BALANCE | AIC | MLA/均衡分核 |

#### 3.2.2 任务总数计算

```
总任务数 bn = batchSize × kvHeadNum × outer
其中 outer = s1Outer（切 S1 时）或 gOuter（切 G 时）或 1（不切时）
```

#### 3.2.3 分核算法 —— `SplitBN()`

1. **均衡分配** (SplitBN_V0)：当所有 batch 实际 seqLen 相同、或 bn ≤ coreNum 时，每个 core 分配 `ceil(bn / coreNum)` 个任务
2. **稀疏分配** (Sparse Split)：当 batch 之间 seqLen 差异大时，按各 batch 的实际负载（`estimatedLoad`）作为权重，贪心分配到各 core
3. **负载均衡分配** (SplitBalanced)：适用于 MLA + TND layout，基于 `gS1SplitNumOfFdHead` 的精细分配
4. Flash Decode 场景 → `SplitBNS()`：额外进行 S2 维度的 Split-KV 切分，`kvSplitPart` 记录分裂份数

#### 3.2.4 核心启动范围

```cpp
usedCoreNum_    = 有实际任务的 core 数
formerCoreNum_  = coreSidxEnd[aiCoreIdx] （每个 core 处理的任务起始索引）
bn2LoopTimes    = coreSidxEnd[aiCoreIdx+1] - coreSidxEnd[aiCoreIdx] （当前 core 的任务数）
```

> `coreSidxEnd[]` 数组（50 元素）通过 tiling 数据下发到 kernel，kernel 内通过 `copy_data_align64` 从 UB 的 tiling 数据拷贝到栈。

### 3.3 MM 中间结果 Buffer 大小

Tiling 阶段预计算每个 Core 的 UB Buffer 大小：

```cpp
// cubeMSize = groupSplitSize × s1SplitSize（已按 gMax 截断）
mmResUbSize_   = sInnerSizeAlign × cubeMSize × msdIterNum   // BMM1 S 结果: [M, N]
bmm2ResUbSize_ = headDimAlign   × cubeMSize × msdIterNum   // BMM2 O 结果: [M, D]
```

rope 场景下 `msdIterNum` 不参与乘法（直接 float 计算），UB 不含 MSD 膨胀。

### 3.4 Workspace 空间计算

Tiling 阶段在 `CalcWorkSpace()` 中计算 workspace 总需求，最终写入 `ifaContext_->workSpaceSize`。

**预加载倍数 `preLoadNum`**：

| PerfMode | preLoadNum | 含义 |
|----------|-----------|------|
| 默认 | 1 | BMM_ALL_BY_VEC，不需要预加载 |
| CUBE_VIEW_MM / FULL_LOAD | **4** | PRE_LOAD_NUM，4 级预加载 |
| CUBE_VIEW_MM_DD | **6** | BUF_NUM，6 级预加载（DD 模式最激进） |
| CUBE_VIEW_MM_MLA | **2** | MLA 模式 2 级 |

**有效核数 `actCoreNum`**：默认等于 `coreNum_`，MLA 模式下 tiling 下沉时取 `aicNum_`。

#### Workspace 组成明细（CUBE_VIEW_MM_DD 为例）

```
workspaceSize = libapiSize_  (基础库空间, ~几KB)
              // === 中间结果 buffer (每核 × 预加载倍数) ===
              + preLoadNum × mmResUbSize  × actCoreNum × elemSize(mmRes)
              + preLoadNum × mmResUbSize  × actCoreNum × elemSize(vec1Res)  × kvDtypeRatio
              + preLoadNum × bmm2ResUbSize × actCoreNum × elemSize(mm2Res)
              + preLoadNum × bmm2ResUbSize × actCoreNum × elemSize(vec2Res)
              + preLoadNum × bmm2ResUbSize × actCoreNum × elemSize(qPreProcRes) × kvDtypeRatio
              // === MLA 辅助空间 ===
              + preLoadNum × gMax_  × actCoreNum × elemSize(nUpdate)
              + preLoadNum × gMax_  × actCoreNum × elemSize(softmaxSum)
              // === FULL_LOAD 额外空间 ===
              + preLoadNum × actCoreNum × dealBN2Num × gSize × 32B × tmpWsNum
              + preLoadNum × actCoreNum × dealBN2Num × gSize × 32B (softmaxSum)
              // === Page Attention 回调节省 ===
              + actCoreNum × 64B × 2
              // === Flash Decode ===
              + accumOutSize × blockTypeSize
              + logSumExpSize × blockTypeSize
              // === System Prefix ===
              + batchSize × gSize × headDim × 2 × actCoreNum × 2B (prefix中间结果)
```

**elemSize 含义**：
| 字段 | ANTIQUANT mode (无 Rope) | ANTIQUANT mode (Rope) | 非 ANTIQUANT |
|------|--------------------------|----------------------|-------------|
| mmResElemSize | 4B (int32) | 2B (fp16) | 原始类型 |
| vec1ResElemSize | 1B (int8) | 1B (int8) | 原始类型 |
| bmm2ResElemSize | 4B (int32) | 2B (fp16) | 原始类型 |
| vec2ResElemSize | 4B (float) | 2B (fp16) | 原始类型 |
| qPreProcResElemSize | 1B (int8) | 1B (int8) | 原始类型 |
| kvDtypeRatio | INT4: 0.5, 其他: 1.0 | | |

**Flash Decode 空间**：
```
accumOutSize   = FDParamNums × headDimAlign
logSumExpSize  = 2 × FDParamNums × (32B / sizeof(float))
```
其中 `FDParamNums` = batchSize × kvHeadNum × 外层数 × sInnerLoopTimes（非均衡）或按实际分核统计（均衡）。

#### Workspace 总量估算（典型 GQA 场景）

以 nNumOfQInOneGroup=8, cubeMSize=64, headDim=128, sInnerSizeAlign=2048, actCoreNum=24, msdIterNum=2 为例：

| 组分 | 计算 | 大小 |
|------|------|------|
| mm1ResGm | 6×24×(2048×64×2)×4B | ≈75MB |
| vec1ResGm | 6×24×(2048×64×2)×1B | ≈19MB |
| mm2ResGm | 6×24×(128×64×2)×4B | ≈9.4MB |
| vec2ResGm | 6×24×(128×64×2)×4B | ≈9.4MB |
| qPreProcRes | 6×24×(128×64×2)×1B | ≈2.4MB |
| nUpdate + softmaxSum | 6×128×24×(4+8)B | ≈0.2MB |
| 其他 | PA/SP/LibApi | ≈几MB |
| **合计（控制目标）** | | **≤ 200MB**（典型值） |

> GQA 场景下 `sInnerSize_` 分档切分（8192→4096→2048）的目标是控制单核 workspace ≤ `32MB ÷ preLoadNum`，保证总 workspace 在合理范围。

## 4. 存储分配

### 4.1 L1 Buffer（AIC Cube 侧，总 512KB —— 恰好装满）

L1 总容量为 512KB，本 Kernel 将其 100% 利用：

| Buffer | 大小 | 数量 | 总容量 | 用途 | 备注 |
|--------|------|------|--------|------|------|
| Q L1 | 64KB | 1 | 64KB | BMM1 A 矩阵（Q） | 单 buffer，s2Idx=0 时载入 |
| K/P L1 | 64KB | 3 | 192KB | BMM1 B 矩阵（K）/ BMM2 A 矩阵（P） | triple buffer，支持 ND→NZ 转置 |
| V L1 | 64KB | 4 | 256KB | BMM2 B 矩阵（V） | quadruple buffer，支持 BMM2 多轮复用 |
| **合计** | | | **512KB** | | |

> 64 + 192 + 256 = 512KB 恰好用完 L1 容量，无浪费。K/P 的三重缓冲和 V 的四重缓冲深度由流水需要而定，tiling 阶段通过 UB 空间检查 (`CheckUbSpace`) 保证不超。

### 4.2 L0 Buffer（Cube 本地，总约 256KB）

| Buffer | 大小 | 数量 | 总容量 | 用途 |
|--------|------|------|--------|------|
| L0A | 32KB | 2 | 64KB | BMM1/BMM2 A 矩阵 ping-pong |
| L0B | 32KB | 2 | 64KB | BMM1/BMM2 B 矩阵 ping-pong |
| L0C | 64KB | 2 | 128KB | BMM1/BMM2 C 结果 ping-pong |

- L0C 大小为 L0A/L0B 的 2 倍（64KB vs 32KB），因为 INT8 乘累加需要 INT32 输出（4×）
- 在 ANTIQUANT 场景 L0C_T = int32_t，非 ANTIQUANT 场景 L0C_T = float

### 4.3 UB Buffer（AIV Vector 侧，总约 165KB）

| Buffer | 大小 | 数量 | 用途 |
|--------|------|------|------|
| inputBuf1 | 32KB | 1 | Vector InQueue（读取 BMM 结果/attnMask） |
| inputBuf2 | 16KB | 1 | Vector InQueue（辅助数据搬运） |
| outputBuf1 | 16KB | 1 | Vector OutQueue（写出 BMM/量化结果） |
| outputBuf2 | 8KB | 1 | Vector OutQueue（辅助写出） |
| tmpBuff1 | 32KB | 1 | 临时计算空间（mmResUb/accumOutLocal 等） |
| tmpBuff2 | 32KB | 1 | 临时计算空间（aFloorUb/softmaxTmpUb 等） |
| tmpBuff3 | 2KB | 1 | 临时计算空间（RowMax/Brcb 辅助） |
| vec2ResBuff | 16KB | 1 | BMM2 中间结果暂存（ANTIQUANT 模式） |
| antiqScaleBuff | 4KB | 1 | 反量化 scale 暂存 |
| antiqOffsetBuff | 4KB | 1 | 反量化 offset 暂存 |
| qAmaxBuff | ~9KB | 4 | 预量化 Q 的 Amax（4 slot × (2K+256B)） |
| softmaxResAmaxBuff | ~9KB | 4 | Softmax P 的 Amax（perToken 模式） |
| softmaxMaxBuff | 2KB | 4 | Softmax Max（4 slot × 512B） |
| softmaxExpBuff | 2KB | 4 | Softmax Exp（4 slot × 512B） |
| softmaxSumBuff | 2KB | 4 | Softmax Sum（4 slot × 512B） |
| softmaxMaxDefaultBuff | 512B | 1 | Softmax Max 初始值（第一轮用） |
| softmaxSumDefaultBuff | 512B | 1 | Softmax Sum 初始值（第一轮用） |

> 注：UB 总量通过 `IFAT::totalUb` 控制，上述分配尺寸在 InitBuffer 时由 Pipe 管理，确保不超出硬件限制。

## 5. 关键优化技术深度分析

### 5.1 四级软件流水 + CrossCore 双引擎异构调度

#### 优化原理

这是本 Kernel 最核心的架构级优化。Ascend 910B 的 DaVinci 架构在同一物理 Core 中包含两个异构计算引擎：
- **AIC (AI Core / Cube 引擎)**：专用于密集矩阵乘（Mmad），含 L0A/L0B/L0C 本地缓存
- **AIV (AI Vector 引擎)**：专用于逐元素向量计算（Mul, Add, Exp, Cast 等）

两个引擎通过 CrossCore Flag 机制进行软件同步，形成 4 阶段交错流水线：

```
Timeline  ────────────────────────────────────────────────────────►

AIV:  QPreProcess(loop N)  │ Vec1(loop N)           │ Vec2(loop N-2)          │ QPreProcess(loop N+2)
                           │                        │                         │
AIC:                       │ BMM1(loop N)           │ BMM2(loop N-1)          │ BMM1(loop N+2)
                           │                        │                         │
Sync:   V0→C1              C1→V1                   V1→C2                   C2→V2
```

四个 CrossCore Flag 定义（`:298-302`）：

```cpp
static constexpr uint64_t SYNC_V0_C1_FLAG = 6;  // AIV QPreProcess 完成 → AIC 可开始 BMM1
static constexpr uint64_t SYNC_C1_V1_FLAG = 7;  // AIC BMM1 完成 → AIV 可开始 Vec1
static constexpr uint64_t SYNC_V1_C2_FLAG = 8;  // AIV Vec1 完成 → AIC 可开始 BMM2
static constexpr uint64_t SYNC_C2_V2_FLAG = 9;  // AIC BMM2 完成 → AIV 可开始 Vec2
```

#### 与硬件体系结构的关系

Ascend 910B 的 DaVinci Core 中：
- Cube 引擎和 Vector 引擎共享 L1、UB 和 GM 带宽，但拥有独立的指令流水线和执行单元
- CrossCore Flag 是硬件支持的轻量级同步原语（基于 Barrier 寄存器），延迟约几十个 cycle
- 没有硬件 Cache Coherence，需要通过 Flag 保证数据生产者-消费者顺序
- `CrossCoreSetFlag<SYNC_MODE2, PIPE_xxx>` 中的 `SYNC_MODE2` 指跨 Core 双模式同步：既清流水线又设 Flag

#### 优化效果

- 将 Vector 侧的 Softmax/反量化开销与 Cube 侧的矩阵乘开销完全重叠
- 在理想流水状态下，总耗时 ≈ max(Σ Vector 耗时, Σ Cube 耗时)，而非两者之和
- 对于 GQA+反量化场景（Vector 侧计算量大），通过 Cube 发小块（sInnerSize=1024）进一步平衡两者负载

#### 代码实现

Process() 中的主循环（`:3778-3919`）：

```cpp
// 阶段分发逻辑
for (uint32_t i = 0; i < tasks; i++) {
    uint32_t loop = gtasks;
    CalcParams(loop, extraInfo[loop % EXTRA_NUM], taskContext[i]);
    if ASCEND_IS_AIV {
        QueryPreProcessL(extraInfo[loop % EXTRA_NUM]);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V0_C1_FLAG);  // V→C
    }
    gtasks++;
}
// BMM1 + Vec1 阶段
for (uint32_t i = 0; i < TASK_NUM; i++) {
    if (gloop > 0 && gtasksC1V1 < gtasks) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(SYNC_V0_C1_FLAG);
            ComputeMm1(extraInfo[loop % EXTRA_NUM]);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);  // C→V
        }
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
            ProcessVec1L(extraInfo[loop % EXTRA_NUM]);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG);  // V→C
        }
    }
}
// BMM2 + Vec2 阶段（同理，同步 V1→C2→V2）
```

#### 亮点

- **零浪费的 L1 分配**：Q L1(64KB) + K/P Triple(192KB) + V Quad(256KB) = 512KB 恰好占满 L1
- **`SYNC_MODE2` 的双重语义**：既确保 Cube/Vector 流水线清空，又设置跨 Core Flag，比单纯 Flag 更安全
- **TASK_NUM=2 批量派发**：一次派发 2 个 [B,N2,S1] 任务的参数，减少空泡

---

### 5.2 DD (Double-Double) 预加载与 TASK_NUM 流水深度

#### 优化原理

DD 模式通过 `PRE_LOAD_NUM_DD = 4` 在 workspace 中为每核预留 4 份中间结果 buffer，支持最多 4 个 batch/(n2,g) 子任务的结果同时在 GM 中存活、在不同流水阶段被消费。

配合 `TASK_NUM = 2` 的流水深度，形成"2 路输入 + 4 路轮转"的缓冲体系。softmax 中间状态（max/sum）同样按 4 slot 轮转（`:236-239`）：

```cpp
TBuf<> softmaxMaxBuff;  // PRE_LOAD_NUM_DD * 2K
TBuf<> softmaxExpBuff;  // PRE_LOAD_NUM_DD * 2K
TBuf<> softmaxSumBuff;  // PRE_LOAD_NUM_DD * 2K
TBuf<> qAmaxBuff;       // PRE_LOAD_NUM_DD * (2K + 256B)
```

#### 与硬件体系结构的关系

- 每份 workspace buffer 按 `aiCoreIdx * dbWorkspaceRatio * size` 偏移分配到独立 GM 区域，多核间互不干扰，无需锁
- 4 slot 轮转索引 = `loop % PRE_LOAD_NUM_DD`，在 GM 中利用空间换时间消除同步等待
- TASK_NUM=2 配合内层循环的 `curS2LoopTimes = sInnerLoopTimes + 2`（最后 batch 尾部多 2 次排空）

#### 优化效果

- 将 Cube 引擎的 MM 启动延迟与 Vector 引擎的 Softmax/反量化处理延迟完全解耦
- 多 batch 场景下，前一个 batch 的 Vec1 处理可与后一个 batch 的 QPreProcess 并行
- DD 模式的 preLoadNum 为 6（tiling 侧），比标准预加载的 4 更激进，进一步增大流水窗口

#### 亮点

- **ExtraInfo 环形缓冲区**（`EXTRA_NUM = 6`）：`:3816` 中 `ExtraInfo extraInfo[EXTRA_NUM]` 在栈上分配 6 个 slot 的上下文信息，支持 6 个在飞迭代的状态管理
- **Deducing Active Vector Core**：通过 `info.mSizeV = (info.mSize + 1) / 2` 将 M 维对半拆分给两个 AIV Core，`tmpBlockIdx % 2 == 1` 的 Core 处理下半部分

---

### 5.3 L1 Triple/Quadruple Buffer 与 L0 Ping-Pong

#### 优化原理

L1 作为 Cube 引擎与 GM 之间的中间缓存，采用多重缓冲消除数据搬运与计算的依赖：

- **K/P L1：Triple Buffer (3 × 64KB)**
  - BMM1 阶段：K 矩阵的 N 方向按 512 行切分 (`KV_LOAD_TO_L1_ROW_NUM`)，每次载入一块到 L1
  - BMM2 阶段：P 矩阵（Vec1 输出）按 128 行切分 (`P_LOAD_TO_L1_ROW_NUM`)，复用同一物理 buffer
  - 3 份 buffer 支持：1 份正在写入（MTE2→L1）、1 份正在读入 L0（L1→MTE1→M）、1 份空闲/就绪

- **V L1：Quadruple Buffer (4 × 64KB)**
  - BMM2 阶段 V 矩阵的关键优化：因为 BMM2 的 M 维可能多次循环（`mCopyTimes`），而 K 维只需载入一次
  - 4 份 buffer 支持 V 矩阵多轮复用，避免重复从 GM 搬运

- **L0A/L0B/L0C：Double Buffer (ping-pong)**
  - 每份 buffer 32KB/64KB
  - M 方向按 64 元素切分 (`baseM = 64/sizeof(KV_T)`)
  - 事件链：`MTE1→M(LoadData) → M→FIX(Mmad) → FIX→M(Fixpipe)`

#### 与硬件体系结构的关系

Ascend 910B DaVinci 架构的存储层次：
```
GM (HBM, ~32-64GB) → L2 Cache → L1 (512KB/Core) → L0 (256KB/Cube) → 寄存器
```

- L1→L0 的 `LoadData` 是 MTE1（Memory Transfer Engine 1）硬件指令，支持 2D Block 传输
- `LoadDataWithTranspose` 在搬运同时完成矩阵转置（K^T 场景关键）
- L0C 到 GM 的 Fixpipe 是专门的量化/反量化硬件通路

事件管理代码示例（`:3071-3072`）：
```cpp
// 触发 MTE2 将 Q 从 GM 搬运到 L1
WaitFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0 + qL1BufIter);
CopyInMm1AToL1(aL1Tensor, info);
SetFlag<HardEvent::MTE2_MTE1>(L1Q_EVENT0 + qL1BufIter);
```

#### 优化效果

- K 矩阵 GM→L1 搬运与上一块 K 的 L1→L0→Mmad 重叠
- V 矩阵四重缓冲使多轮 BMM2 中 V 无需重新搬运
- L0 双缓冲隐藏 LoadData 延迟（约 64 cycles for 32KB）

#### 亮点

- **K/P 复用 L1 物理空间**：BMM1 用 K、BMM2 用 P，不同阶段复用 `tmpBufL1KP`，节省 128KB L1
- **V 智能复用**：`mCopyIdx == 0` 时才触发 V 搬运（`:3576`），后续迭代直接引用已有 buffer（`:3621`）

---

### 5.4 ND→NZ 布局转换与 Fracture NZ 格式

#### 优化原理

Ascend DaVinci 架构的 Cube 引擎对 NZ (Nd_Z) 数据布局有最优访问模式：
- **ND 布局**：标准的行列主序，同一行的元素连续存储
- **NZ 布局**：将矩阵分成 16×16 的小块（Block），Block 内以列主序排列，Block 间以行主序排列

NZ 布局的优势：
1. `LoadData` 每次搬运 16 行的连续块，与 Cube 的 16×16 MAC 阵列天然对齐
2. 减少 L0 读取时的 Bank Conflict
3. 转置操作可通过 `LoadDataWithTranspose` 直接在搬运时完成

`Nd2NzParams` 结构体定义搬运参数（`:2790-2797`）：
```cpp
Nd2NzParams mm1Nd2NzParamsForA;
mm1Nd2NzParamsForA.ndNum = 1;    // ND 矩阵个数
mm1Nd2NzParamsForA.nValue = 16;  // 每个 ND 矩阵的行数（16 对齐）
mm1Nd2NzParamsForA.dValue = headDimAlign;  // 列数
mm1Nd2NzParamsForA.srcDValue = headDim;    // 源 ND 矩阵行间隔
mm1Nd2NzParamsForA.dstNzC0Stride = 16;    // NZ Block 间 C0 方向步长
mm1Nd2NzParamsForA.dstNzNStride = 1;      // NZ Block 间 N 方向步长
DataCopy(aL1Tensor[i * copyStrideL1],  /* 目标 L1 地址 */
         vec1ResGm[srcOffset],          /* 源 GM 地址 */
         mm1Nd2NzParamsForA);           /* ND→NZ 转换参数 */
```

#### 与硬件体系结构的关系

- DaVinci Cube 的 MAC 阵列是 16×16 的脉动阵列（Systolic Array）
- NZ 格式天然将矩阵分块为 16×16 的 Block，与 MAC 阵列维度完全匹配
- `L1_LAYOUT_zN` 宏启用后（`:3180`），L1 内数据也按 NZ 组织，`LoadData` 可一次读入一个完整 Block

#### 优化效果

- 消除 ND→NZ 在线转换开销（传统方案需要在每个 Kernel 内做布局转换）
- 对于 Attention 这种多次矩阵乘的场景，NZ 布局使 L1→L0 搬运效率提升约 30-50%

#### 亮点

- **Stribe 边界处理**：BSH/BSND 格式下，当 `baseN * step > 65535` 时（超过 MTE2 地址空间限制），自动退化为逐个 ND 矩阵搬运（`:3001`），避免硬件溢出
- **Page Attention 下 NZ 适配**：NZ 布局的 Page Attention 通过精确的 `blockElementCnt`（`:3104`）对齐到 Cache Block 内的 Block 边界

---

### 5.5 MSD 双段高精度反量化（Most Significant Digit）

#### 优化原理

INT8 量化本质上是有损压缩。为了将 INT8 矩阵乘结果高精度还原为 FP16，DD 模式采用 MSD（Most Significant Digit）双段分解：

```
原始值 V_int8 = V_MSD × 254 + V_LSB
反量化: V_fp16 = (V_MSD × 1/1024) + (V_LSB × 1/(1024×254))
             = V_MSD × scaleC1S1 + V_LSB × scaleC1S2

其中 scaleC1S1 = 1/1024, scaleC1S2 = 1/(1024×254)
```

MSD 侧通过 `AntiquantMatmulPreProcess`（`:1382-1418`）将 FP32 数据分解：
1. 对源矩阵逐行计算 AbsRowMax
2. 归一化：`A' = A × (127/Amax)`
3. 取整：FP32→FP16→INT8（两轮 Cast）
4. MSD 展开：`AntiquantAIterExpand` 将 INT8 分解为两段

BMM2 的量化因子略有不同：`scaleC2O1 = 1/(1024×127)`, `scaleC2O2 = 1/(1024×254×127)`，多除以 127 是因为 P 矩阵在反量化前先乘了 Scale V 的 127。

#### 与硬件体系结构的关系

- DaVinci 的 Fixpipe 是专用硬件单元，支持 INT32→FP16 的量化反量化转换
- Fixpipe 支持 `SetAtomicAdd` 模式：两段 MSD 反量化结果通过硬件原子加合并到同一 GM 位置，无需额外搬运
- `msdIterNum = 2` 是精度与性能的折中：1 次迭代精度不足（≈7bit），3 次以上边际收益递减

#### 优化效果

- 精度：MSD 双段分解将 INT8→FP16 反量化精度提升到约 14bit 有效精度（vs 直接反量化的 7bit）
- 性能：Fixpipe 硬件原子加消除两段结果合并的额外内存访问，GM 上直接完成合并

#### 代码实现

Fixpipe 的两段输出 + AtomicAdd（`:3212-3235`）：
```cpp
for (uint32_t mIter = 0; mIter < msdIterNum; mIter++) {
    float tmp = (mIter == 0) ? quantScaleC1S1 : quantScaleC1S2;
    FixpipeParamsV220 fixParams;
    fixParams.nSize = actualBaseN;
    fixParams.mSize = actualBaseM / msdIterNum;
    fixParams.quantPre = QuantMode_t::DEQF16;
    fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
    if (mIter == 1) {
        SetAtomicAdd<half>();  // 第二段启用原子加
    }
    Fixpipe(mm1ResGm[workspace_offset], cL0Tensor[mIter * fixParams.mSize * 16], fixParams);
    if (mIter == 1) {
        SetAtomicNone();
    }
}
```

#### 亮点

- **硬件原子加**：`SetAtomicAdd<half>()` 利用 Fixpipe 的原子写特性直接在 GM 上合并 MSD 两段，消除了额外的读写往返
- **中间段特殊处理**（`:3264-3284`）：当 M 维循环出现中间块时（既不是 Head 也不是 Tail），该块同时跨越 MSD 第一段和第二段，使用精确定位 + `cL0Tensor[(info.mSize - i * baseM) * 16]` 偏移来处理

---

### 5.6 Online Safe Softmax (Flash V2) + Brcb 优化

#### 优化原理

传统 Softmax 需要三遍遍历（Max → Exp → Sum → Div），Flash Attention 将其在线化：

```
算法（每轮 sInnerLoop）：
  m_new = max(m_old, rowmax(S_cur))
  S_cur = exp(S_cur - m_new)
  O = O_old * exp(m_old - m_new) + S_cur × V_cur
  sum_new = sum_old * exp(m_old - m_new) + sum(S_cur)

最后一轮: O = O / sum_final
```

DD Kernel 中通过 `IFA_SOFTMAX_WITHOUT_BRC_PRELOAD` 宏启用进一步优化。该宏利用 Brcb（Broadcast Block）指令替代显式 DataCopy：

`:3963-3964` 中 `CopyFixedUbToGm`：
```cpp
// WITH Brcb: 将 m×1 的 1D 数据扩展为 m×8 的 2D 数据
Brcb(tmp, src, (mSizeVector + 7) / 8, {1, 8});
// WITHOUT Brcb: 显式 DataCopy
DataCopy(tmp, src, size);
```

#### 与硬件体系结构的关系

- Brcb 是 Vector 引擎的硬件指令，直接从寄存器广播标量到多行，延迟远低于 DataCopy
- 使用 Brcb 后，softmax 的 max/sum buffer 从 `BLOCK_ELEMENT_NUM` 大小缩减为 1 个元素（`:2014` 的 `baseOffset = startRow` 而非 `startRow * BLOCK_ELEMENT_NUM`）
- 节省约 87.5% 的 softmax 状态缓冲区（每 slot 从 2KB 降至 256B）

#### 优化效果

- Softmax 状态缓冲区总大小减少：`4 × 3 × (2048 - 256) ≈ 21KB` UB 空间节省
- Brcb 指令延迟 ≈ 8 cycles（vs DataCopy 的 ≈ 32+ cycles for 2KB），Softmax 入口延迟降低约 75%
- 节省的 UB 空间可用于更大的 M split size，减少循环次数

#### 亮点

- **Buffer 尺寸宏自动切换**：`:666-672` 中根据宏开关自动选择 512B 或 2KB 的 default buffer 大小
- **Softmax 切分 8 对齐**：`:2357-2361` 中 `mSplitSize = mSplitSize / 8 * 8` 确保 Softmax Flash V2 的 M 维是 8 的倍数，匹配 `Brcb` 的 `{1, 8}` repeat pattern

---

### 5.7 Attention Mask 优化

#### 优化原理

Attention Mask 是 Decode/Sparse 场景的频繁操作，DD Kernel 实现了三层优化：

1. **Skip 判定**：`IsSkipAttenMask`（`:1834-1860`）在 s2 位置未进入 mask 区域时直接跳过整个 mask 流程
2. **Broadcast 复用**：qSeqSize=1 且非 TND 布局时，mask 只拷贝一行，其余通过 `DataCopy(dst, src, stride)` 原地复制
3. **按需排列 + 硬件加速**：
   - BNSD 布局：通过 head-mid-tail 三段拆解，将 mask 按 G 维度重新排列
   - BSH/BSND/TND：通过 `Copy<int16_t>` 的 Mask Counter 模式（`:1962-1964`）高效重排

#### 与硬件体系结构的关系

- `SelectWithBytesMask` 是 Vector 引擎的硬件指令，在一次操作完成 Compare + Select + Fill，代替软件三分支实现
- Mask Counter 模式的 `Copy<int16_t>` 利用 Vector Mask 寄存器加速不规则数据搬运
- `ATTENMASK_STRIDE = 2048` 固定步长简化了 2D→1D 的地址计算

#### 优化效果

- Skip 判定将不必要 Mask 区域的 softmax 迭代完全跳过，节省约 30-70% 的 mask 开销（取决于序列长度比）
- `SelectWithBytesMask` 一次完成三段操作，延迟约为软件实现的 1/5
- qSeqSize=1 时的 Broadcast 复用节省 (dealRowCount-1) × attenMaskSizeAlign 的数据搬运

#### 代码实现

IsSkipAttenMask 的判定逻辑（`:1834-1860`）：
```cpp
// s2<s1时，必然走mask
if (info.s2Size < actualSeqQ) { return false; }
// s2结束位置未超过 mask 区域起点时，跳过
uint32_t s2EndPos = info.s2Idx * singleProcessSInnerSize + info.actualSingleProcessSInnerSize;
if (s2EndPos <= (info.s2Size - actualSeqQ + 1)) { return true; }
// BSH/BSND/TND: 最后一个 s1 位置不需要 mask
if (s1StartIdx == s1EndIdx && s1StartIdx == actualSeqQ - 1) { return true; }
```

#### 亮点

- **Mask Counter 的高效重排**：`:1960-1964` 中 `Copy<int16_t>` 配合 `SetMaskCount()` / `SetVectorMask<MaskMode::COUNTER>` 实现带掩码的不规则 stride 拷贝，一步完成 head/mid/tail 三段排列

---

### 5.8 Page Attention 支持

#### 优化原理

Page Attention（PagedAttention / vLLM 风格）中，KV Cache 不是连续存储在 GM 中，而是通过 Block Table 间接寻址。DD Kernel 实现了完整的 Page Attention 支持：

- `CopyInMm1BToL1ForPA` / `CopyInMm2BToL1ForPA`：按 block 粒度逐块搬运，处理 block 内偏移和跨 block 边界
- Block Table 查询：`blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset)` 获取物理 block ID
- 三段式处理（`:3088-3122`, `:3580-3613`）：head（起始不完整 block）、mid（完整 block）、tail（尾部不完整 block）

#### 与硬件体系结构的关系

- 每次 `blockTableGm.GetValue` 是一次标量 GM 读取，延迟约 200 cycles
- BNSD vs BSH/BSND 的物理地址计算公式不同（`:3103-3114`）：
  - BNSD: `offset = id × blockSize × headDim × kvHeadNum + n2Idx × headDim × blockSize + remainRowCnt × headDim`
  - BSH/BSND: `offset = id × blockSize × headDim × kvHeadNum + n2Idx × headDim + remainRowCnt × headDim × kvHeadNum`

#### 优化效果

- 完全解耦了逻辑 KV 序列与物理存储，支持任意 block 排列
- 三段式处理避免逐元素的 `if` 分支，减少了地址计算开销
- 在 Page Attention 场景下，BMM1/BMM2 的 K/V 搬运路径与连续存储路径共用了 `Nd2NzParams` 转换

#### 亮点

- **blockK 对齐**：`:3408` 中 `blockK = 32 / sizeof(KV_T)` 将搬运粒度对齐到 L1 的 32B 行，避免跨 stride 的超额搬运
- **NZ 布局下的 Page Attention**：`:3046-3047` 中 `blockElementCnt` 在不同量化精度下自动调整，INT4 时为 64（而非 32B）

---

### 5.9 KV Left Padding 处理

#### 优化原理

在 Batch 变长场景中，某些 batch 的 KV 序列可能有左侧填充（Left Padding）。DD Kernel 通过 `ComputeKVPaddingBeginOffset`（`:861-879`）计算有效 KV 起始偏移：

```cpp
int64_t paddingSize = kvPaddingSizeGm.GetValue(0);
int64_t startPosition = kvSeqSize - paddingSize - curActualSeqLen;
kvPaddingBeginOffset = static_cast<uint64_t>(startPosition);
```

后续所有 K/V 数据搬运（tensorBCoreOffset, attenMaskOffset, antiqKeyParamOffset 等）统一加 `kvPaddingBeginOffset`，实现"逻辑上跳跃 Padding 区域，物理上连续寻址"。

#### 优化效果

- 避免为每个 batch 创建不同的 K/V 起始地址，统一在 offset 中加减
- 无效 Padding 区域的 K/V 数据根本不会被 MTE2 搬运，节省带宽

---

### 5.10 Coupled/DeCoupled Batch 模式

DD Kernel 支持连续 batch（Coupled）和非连续 batch（DeCoupled）两种模式：

- **Coupled**（`batchContinuous = 1`）：只初始化一次 `InitKeyGm(0)` / `InitValueGm(0)`，所有 batch 共享同一 `GlobalTensor` 基址
- **DeCoupled**（`batchContinuous = 0`）：每个 batch 动态获取 `keyGm` / `valueGm` 基址（`:1009-1011`），通过 `SeqLenFromTensorList` 从 `ListTensorDesc` 获取实际序列长度

#### 与硬件体系结构的关系

- `ListTensorDesc` 是 AscendC 提供的动态形状描述符，允许不同 batch 的 Tensor 有不同的物理地址和形状
- `SeqLenFromTensorList`（`:847-858`）通过 `ListTensorDesc::GetDesc` + `TensorDesc::GetShape` 获取运行时序列长度

---

### 5.11 L2 Cache 控制

DD Kernel 显式控制 L2 Cache 行为：

```cpp
if (tilingData->baseParams.l2CacheOffFlag) {
    keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
}
```

#### 优化原理

- Attention 计算中 K/V 矩阵通常只被当前 Core 访问一次（或通过 Page Attention 间接访问），在 L2 中缓存收益为负
- 关闭 K/V 的 L2 Cache 可以避免 L2 被一次性读取的数据污染，将 L2 带宽留给更频繁访问的数据（如中间 workspace）

#### 优化效果

- 在 Decode 场景（小 batch、长序列）中，K/V 数据量大但复用率低，关闭 L2 Cache 可提升约 5-10% 的 GM 带宽利用率

---

### 5.12 INT4 奇数列处理（DealKvInt4ColumnOdd）

#### 优化原理

INT4 量化时每个字节存储 2 个元素。当 KV 序列长度（sInnerSize）为奇数时，最后一个字节只有低位有效、高位为垃圾数据。`DealKvInt4ColumnOdd`（`:3974-3990`）检测此情况并将无效列清零：

```cpp
if constexpr (KVINT4) {
    if (actualColumnCount % 2 == 1) {
        uint64_t mask[2] = {1U << maskIdx, 0};
        Duplicate(mmResUb[i * columnCount + offset], FLOAT_ZERO, mask, 1, 1, 8);
    }
}
```

#### 亮点

- 使用 `Duplicate` 的掩码模式（`mask` 参数），一次指令清零 8 个 float 中的指定位置
- 仅在 ANTIQUANT PerChannel 模式的 Vec1 阶段调用，避免在正常流程中引入额外开销

---

### 5.13 BF16 高精度舍入（RINT 模式）

DD Kernel 在输出为 BF16 格式时使用 `CAST_RINT`（四舍六入五成双）代替默认的 `CAST_ROUND`（`:2197-2198`, `:1729-1730`）：

```cpp
if constexpr (IsSameType<OUT_T, bfloat16_t>::value) {
    Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, ...);
} else {
    Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, ...);
}
```

#### 优化效果

- `CAST_RINT` 是 IEEE 754 标准的最近偶数舍入，在累计场景下无偏（vs `CAST_ROUND` 的四舍五入有 +0.5LSB 偏差）
- 对 Flash Attention 这种多轮累加输出有显著精度收益

---

### 5.14 G 维度切分与 Vector 负载均衡

#### 优化原理

DD Kernel 支持 G 维（Group Query 的 Q 头数）的直接切分（`:614-617`），在 CalcParams 中根据 `gIdx` 和 `gSizeSub` 动态计算 M 维的实际范围。

每个 AIV Core 获得一半的 M 维负载（`:2669-2676`）：
```cpp
if ASCEND_IS_AIV {
    info.mSizeV = (info.mSize + 1) / 2;
    info.mSizeVStart = 0;
    if (tmpBlockIdx % 2 == 1) {
        info.mSizeVStart = info.mSizeV;
        info.mSizeV = info.mSize - info.mSizeV;
    }
}
```

两个 AIV Core 共享 L1/L0（由 AIC Core 统一管理），但各自独立执行 Vector 指令。

#### 优化效果

- 在 GQA 场景（gSize=8，两个 AIV 各处理 4 行），Vector 负载精确均分
- `mSizeVector == 0` 的提前返回（`:2647-2649`）避免尾块 Core 空转

---

### 5.15 Host 侧 TilingKey 编译期路由

#### 优化原理

DD Kernel 通过 `incre_flash_attention_tilingkey.h` 中的 ~320+ 个 TilingKey 宏实现编译期路由。每个 macro 编码了 (layout, Q_T, KV_T, OUT_T, antiquant mode, page attention, flash decode, shared prefix, MLA mode) 的组合。

DD 相关 TilingKey 示例（`PREDD_TILING`）：
```
16000000020222320: BF16/INT8/BF16 per-channel, BNSD, page attention, DD
16000000020222321: 同上, BSH layout
```

Tiling Host 侧在 `SetTilingKey` 中根据参数组合选择对应 Key，Kernel 入口通过 `switch(tilingKey)` 分发到正确的模板实例化。

#### 亮点

- 完全在编译期消除模板参数组合的分支，生成零开销的专用 Kernel
- Host 侧 tiling 可以针对不同组合选择不同的 `sInnerSize_`、`preLoadNum`、`coreNum_`
- DD 模式通过独立的 `EnableCubeViewMMDD()` 决策函数控制是否启用

---

### 5.16 copy_data_align64 栈拷贝优化

#### 优化原理

Tiling 数据从 GM 下发到 UB 后，Kernel 需要读取 `coreSidxEnd[]` 数组。传统做法是将整个 `IncreFlashAttentionCoreParams` 结构体拷贝到栈上，DD Kernel 改用 `copy_data_align64` 仅拷贝需要的 `coreSidxEnd[50]`：

```cpp
uint32_t coreSidxEnd[50];
copy_data_align64((uint8_t *)coreSidxEnd,
    (uint8_t *)(tilingData->increFlashAttentionCoreParams.coreSidxEnd),
    sizeof(coreSidxEnd));
```

#### 优化效果

- 拷贝量从 ~200B 降至 50×4=200B 实际仍有节省，但关键是避免拷贝整个结构体中的不必要字段
- `copy_data_align64` 利用 MTE2 的 64B 对齐搬运，比逐字段赋值快数倍

---

### 5.17 批量流水排空（curS2LoopTimes + 2）

#### 优化原理

Process() 主循环中，最后一个 batch 的 S 循环多执行 2 轮（`:3849`）：
```cpp
uint32_t curS2LoopTimes = (bn2gIdx == bn2LoopTimes - 1) 
    ? sInnerLoopTimes + 2 : sInnerLoopTimes;
```

这多出的 2 轮纯排空（有 `gloop` 守卫，不再生成新任务），使流水线中最后 2 个在飞任务得以完成（`gtasksC1V1 < gtasks`, `gtasksC2V2 < gtasks`）。

#### 亮点

- 无需额外的排空阶段代码，与正常流水线共享同一循环体
- +2 对应 `TASK_NUM=2` 的流水深度，通用性强

---

## 6. 数据流总结

```
GM: Q, K, V, AttnMask, Out, Workspace (mm1Res, vec1Res, mm2Res, vec2Res)
                     │
     ┌───────────────┼─────────────────────────────────────────────┐
     │               │                                             │
  [Vector]        [Cube]                                       [Vector]
  Q PreProcess    BMM1 Q×K^T                                    Vec1
  Q→fp16→int8    (L1 Q→L0A,       ──fixpipe→ mm1ResGm(f16)     Read mm1Res
  →workspace      L1 K→L0B,                                     →F32 + Scale
                  Mmad→L0C)                                     →AttnMask
                                                                 →Softmax
                                                                 →vec1ResGm
                                                                      │
  [Vec2]         [Cube]                                       ←───────┘
  Read mm2Res    BMM2 P×V
  →累积 O        (L1 P→L0A,       ──fixpipe→ mm2ResGm(f16)
  →归一化         L1 V→L0B,
  →输出到 GM      Mmad→L0C)
```

## 7. 参数与配置概要

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `mSize` | Q 矩阵的行数 = gSize × s1Size | ~1024 |
| `headDim` | 每个头的维度 | 128 |
| `gSize` | Group Query 中的 Q 头数 | 1~8 |
| `sInnerLoopTimes` | S 方向循环次数 | 取决于 seqLen |
| `singleProcessSInnerSize` | 每次处理的 S 片段大小 | 由 Tiling 决定 |
| `msdIterNum` | MSD 迭代次数 | 2 |
| `PRE_LOAD_NUM_DD` | DD 预加载 slot 数 | 4 |
| `TASK_NUM` | 流水深度 | 2 |
| `EXTRA_NUM` | ExtraInfo 环形缓冲大小 | 6 |

## 8. 优化技术总览表

| 序号 | 优化技术 | 优化原理 | 硬件关系 | 优化亮点 |
|------|---------|---------|---------|---------|
| 5.1 | 四级软件流水 | AIV/AIC 双引擎通过 CrossCore Flag 交错执行，4 阶段流水重叠 | DaVinci 双引擎架构，CrossCore 硬件同步 | 零浪费的 512KB L1 占满；SYNC_MODE2 双重语义 |
| 5.2 | DD 预加载 + TASK_NUM=2 | 4 slot 轮转 + 2 路任务批处理，解耦 MM 与 Vector 处理 | GM workspace 空间换时间 | ExtraInfo[6] 环形缓冲支持 6 个在飞迭代 |
| 5.3 | L1 多缓冲 + L0 乒乓 | L1 Triple/Quad + L0 Double buffer，掩盖 MTE 搬运延迟 | L1→L0 LoadData 硬件通道 | K/P 复用 L1；V 四重缓冲跨 M 轮复用 |
| 5.4 | ND→NZ 布局转换 | 将 ND 转为 NZ 布局（16×16 Block 主序），匹配 Cube MAC 阵列 | 16×16 脉动阵列天然对齐 | Stribe>65535 边界自动退化为逐块搬运 |
| 5.5 | MSD 双段反量化 | INT32→FP16 分两段：1/1024 + 1/(1024×254)，硬件原子加合并 | Fixpipe + SetAtomicAdd 硬件支持 | 精度从 7bit→14bit，零额外 GM 访问 |
| 5.6 | Online Safe Softmax + Brcb | 在线递推 Max/Sum + Brcb 指令扩展 1D→2D | Brcb 硬件广播指令 | 节省 ~21KB UB，Softmax 延迟降 75% |
| 5.7 | Attention Mask 优化 | Skip 判定 + Broadcast 复用 + SelectWithBytesMask 硬件指令 | Mask Counter + SelectWithBytesMask | Mask 操作节省 30-70% 耗时 |
| 5.8 | Page Attention | Block Table 间接寻址，head/mid/tail 三段式搬运 | blockTableGm.GetValue 标量访存 | 完全解耦逻辑/物理 KV 布局 |
| 5.9 | KV Left Padding | 统一 offset 偏移，跳过无效 Padding | 单次 gmGetValue 查表 | 避免多 batch 重复初始化 |
| 5.10 | Coupled/DeCoupled Batch | ListTensorDesc 动态形状描述 | 动态 Tensor 描述符 | 支持变长 batch |
| 5.11 | L2 Cache 控制 | 关闭 K/V L2 Cache，防止污染 | SetL2CacheHint 硬件接口 | Decode 场景 ≈5-10% GM 带宽提升 |
| 5.12 | INT4 奇数列处理 | Duplicate mask 模式清零无效高位 | Duplicate 硬件掩码 | 一次指令完成 |
| 5.13 | BF16 RINT 舍入 | CAST_RINT 最近偶数舍入 vs CAST_ROUND | IEEE 754 硬件舍入 | 多轮累加无偏 |
| 5.14 | G 维切分 + Vector 均衡 | G 维直接切分 + 两个 AIV 各一半 M | AIV 独立引擎 | mSizeVector==0 提前返回 |
| 5.15 | TilingKey 编译期路由 | ~320+ 个 TilingKey 宏，编译期消除分支 | if constexpr 零开销 | DD 模式独立决策函数 |
| 5.16 | copy_data_align64 | 仅拷贝 coreSidxEnd[50] 到栈，非整个结构体 | MTE2 64B 对齐搬运 | 减拷贝量数倍 |
| 5.17 | 批量流水排空 | 尾 Batch 多执行 2 轮 S 循环（+2=TASK_NUM） | 共享循环体 | 无需专门排空代码 |

## 9. 代码结构索引

| 模块 | 文件名 | 核心内容 |
|------|--------|---------|
| DD Kernel | `op_kernel/arch32/incre_flash_attention_preload_dd.h` | 四级流水主循环、BMM/Mmad/Softmax/Fixpipe/反量化全流程 |
| Tiling | `op_host/incre_flash_attention_tiling.cpp` | 多级切块、核分配、workspace 计算、DD 模式决策 |
| Tiling Key | `op_kernel/incre_flash_attention_tilingkey.h` | ~320+ TilingKey 宏（含 DD 专用 PREDD_TILING） |
| 公共定义 | `op_kernel/ifa_public_define.h` | IFAType 模板参数、BufferManager、向量辅助函数 |
| Tiling 结构 | `op_host/incre_flash_attention_tiling.h` | IncreFlashAttentionTilingData 结构体 |
| Tiling 检查 | `op_host/incre_flash_attention_tiling_check.cpp` | `EnableCubeViewMMDD()` 条件判断 |
