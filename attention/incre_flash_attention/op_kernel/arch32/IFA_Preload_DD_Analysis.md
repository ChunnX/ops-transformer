# Incre Flash Attention Preload DD Kernel 计算流程与优化分析

## 1. 概述

`IncreFlashAttentionAttenPreloadDD` 是增量式 Flash Attention 算子在 Ascend NPU 上的核心 Kernel（arch32/Cube 架构），整体采用 **Vector+Cube 异构流水** 架构，AIV 与 AIC 核心通过 CrossCore 同步机制协同工作完成 Attention 计算。

中间计算数据类型均为 `float`（高精度模式），外部输入/输出 Quant 类型通过 IFAT 模板参数控制。

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
  - 第 i 轮：`O_current = O_prev * exp_sum_ratio + BMM2_result`
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

## 5. 关键优化技术

### 5.1 DD (Double-Double) 四级预加载
- `PRE_LOAD_NUM_DD = 4`：workspace 中预留 4 个 slot，支持最多 4 个 batch/(n2,g) 子任务的流水切换
- softmax 中间状态（max/sum）、Amax 等均 4 路轮转
- ExtraInfo 环形缓冲区（`EXTRA_NUM = 6`）存储每轮的参数上下文

### 5.2 异构流水：AIV ↔ AIC CrossCore 同步
- AIV 和 AIC 之间通过 6 个 CrossCore Flag（V0→C1, C1→V1, V1→C2, C2→V2）实现软件流水
- 流水深度为 2 级（`TASK_NUM = 2`）：同时处理两块数据
- AIV 先做 Q PreProcess + Vec1 → 发信号给 AIC → AIC 做 BMM1 + BMM2 → 发信号给 AIV → AIV 做 Vec2
- 预加载流水（`PRE_LOAD_NUM_DD=4`）在内层循环的尾部 task 控制释放

### 5.3 L1/L0 Ping-Pong / Triple Buffer
- **L1 B**: K 矩阵 triple buffer（3×64KB），配合 BMM1 的 N 方向切分（512 per step）
- **L1 V**: V 矩阵 quadruple buffer（4×64KB），配合 BMM2 的 K 方向复用
- **L0A/L0B/L0C**: 双 buffer ping-pong，掩盖 LoadData 与 Mmad 的延迟

### 5.4 ND→NZ 布局转换
- K/V 矩阵从 GM 通过 `Nd2NzParams` 转换到 L1，将 ND 布局转为 NZ 布局
- NZ 布局有利于 `LoadData` 和 `LoadDataWithTranspose` 的高效搬入 L0
- K 方向切分基于 256 行（BSH/BSND 格式还需处理 stribe 超过 65535 的边界情况）

### 5.5 MSD 双段反量化
- 每个量化矩阵分解为两段（Most Significant Digit 迭代）
- `msdIterNum = 2`：第一段 scale = 1/1024，第二段 scale = 1/(1024×254)
- 两段在 Fixpipe 阶段通过 `AtomicAdd` 合并到 GM workspace
- 实现 INT8→FP16 的高精度还原

### 5.6 Online Safe Softmax (Flash V2)
- 使用 Flash Attention 安全 Softmax 算法，一行一遍历
- 内层 sInnerLoop 循环中维护 running max/sum（4 slot 轮转）
- `IFA_SOFTMAX_WITHOUT_BRC_PRELOAD` 宏启用时，使用无 BRCB 预加载的 Softmax Flash V2 配置

### 5.7 Attention Mask 优化
- **Skip 判定**：当 s2 位置未进入 mask 区域时直接跳过（`IsSkipAttenMask`）
- **Broadcast 复用**：qSeqSize=1 且非 TND 布局时，mask 拷贝一份后通过 DataCopy 广播复用到所有行
- **按需搬运**：BSH/TND 布局下通过 `Copy<int16_t>` 指令按 S1 粒度重新排列 mask 数据
- **SelectWithBytesMask**：硬件加速的选择填充指令

### 5.8 Page Attention 支持
- K、V 不在连续 GM 空间，通过 BlockTable 间接寻址
- `CopyInMm1BToL1ForPA` / `CopyInMm2BToL1ForPA`：按 block 粒度逐块搬运到 L1
- 处理 block 边界跨越（head/mid/tail 三段）

### 5.9 KV Left Padding 处理
- `ComputeKVPaddingBeginOffset`：计算有效的 KV 起始偏移，忽略左侧 padding

### 5.10 Coupled/DeCoupled 模式切换
- `batchContinuous` 标志控制 Batch 是否连续
- 不连续时逐 Batch 重新初始化 keyGm/valueGm（`InitKeyGm`/`InitValueGm`）
- 连续时仅初始化一次

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
