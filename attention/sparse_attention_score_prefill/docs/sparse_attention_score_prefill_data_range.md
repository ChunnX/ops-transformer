# SparseAttentionScorePrefill 算子数据范围约束文档

## 1. 概述

本文档描述 `sparse_attention_score_prefill` 算子的 A2 已验证出版范围。当前范围为
`D=128/256`、分页 `blockSize=16..128`（16 的整数倍）或连续 KV
`blockSize=0`、`qHeads<=64`、`groupSize<=16`、`totalQTokens<=1024K`、
`kvSeqlen<=256K`。这些上界均已完成 A2 真机验证，但不是所有参数上界的笛卡尔积；
实际可运行 shape 仍受 Workspace 和输入张量的 HBM 容量限制。

### 1.1 核心约束摘要

| 约束项 | 限制 | 约束来源 |
|--------|------|----------|
| 输入/输出 dtype | BF16 | 当前 A2 kernel 派发范围 |
| `totalQTokens` (Batch × seq_len) | ≤ 1,048,576 (1024K) | A2 真机边界验证；实际受 HBM 限制 |
| `D` (embeddingSize) | 128 或 256 | A2 真机精度矩阵 |
| `blockSize` | 0，或 16..128 且为 16 的整数倍 | 0 表示连续 KV；正值表示分页 KV |
| `topK` | ≤ 16 | Phase2 UB 容量 |
| `batch` | 未设独立出版上限 | 与 `totalQTokens`、CSR 和 HBM 容量联合约束 |
| `kvHeads` | 1..`numHeads`（因此不超过 64） | `numHeads<=64` 与 GQA 映射 |
| `numHeads` | ≤ 64，且 = kvHeads × groupSize | A2 真机边界验证与 GQA 映射 |
| `groupSize` | 1..16 | A2 真机 group sweep |
| `kvSeqlen` | ≤ 262,144 (256K) | 分页 KV 真机边界验证 |

---

## 2. Workspace 大小分析

### 2.1 计算公式

```
slotOElems    = groupSize × D
slotStatElems = RoundUp(groupSize, 8)
taskSlots     = totalQTokens × kvHeads × topK

accumOutSize  = taskSlots × slotOElems                         // O_partial 元素数
lseStatSize   = taskSlots × slotStatElems                      // max/sum 各自元素数

accumOutBytes = accumOutSize × (innerPrecise==1 ? 2 : 4)       // bf16=2B, fp32=4B
userWorkspace = accumOutBytes + (lseStatSize × 2) × 4          // max + sum 始终 fp32
gmSBytes      = blockDim × 2 × 64 × blockSize × scoreElementBytes
gmPBytes      = blockDim × 3 × 64 × blockSize × 2
workSpaceSize = libapiSize + userWorkspace + gmSBytes + gmPBytes
```

其中 `scoreElementBytes` 在 `innerPrecise=0` 时为 4（FP32），其余路径为 2
（BF16）；A2 的 `gmSBytes/gmPBytes` 是每个 AIC 的 staging 容量乘 `blockDim`。

### 2.2 Workspace 布局

```
┌─────────────────────────────────────────────────────────────────┐
│ libapiWorkspace (libapiSize_)                                   │
├─────────────────────────────────────────────────────────────────┤
│ accumOut    [totalQ × kvHeads × topK × groupSize × D]           │
│   fp32 (innerPrecise≠1) 或 bf16 (innerPrecise==1)               │
├─────────────────────────────────────────────────────────────────┤
│ softmaxMax  [totalQ × kvHeads × topK × groupSize] fp32          │
├─────────────────────────────────────────────────────────────────┤
│ softmaxSum  [totalQ × kvHeads × topK × groupSize] fp32          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 典型配置下的 Workspace 大小

以下表格先列 `accumOut + lseMax + lseSum` 的 user workspace；最终总量还需加
`libapiSize` 和 A2 的 GM S/P staging。对 blockSize=128、blockDim=20，staging
约为 1.56 MiB（BF16 score）或 2.19 MiB（FP32 score）。

#### 配置 A（大规模 Prefill）

| 参数 | 值 |
|------|-----|
| totalQTokens | 131,072 (128K) |
| kvHeads | 4 |
| topK | 16 |
| groupSize | 16 |
| D | 128 |
| numHeads | 64 |

| 项目 | 元素数 | 字节 |
|------|--------|------|
| accumOutSize | 17,179,869,184 | — |
| lseStatSize | 134,217,728 | — |
| accumOut (fp32) | — | ≈ 64 GiB |
| accumOut (bf16) | — | ≈ 32 GiB |
| lseMax + lseSum | — | ≈ 1 GiB |
| **user workspace (fp32)** | — | **≈ 65 GiB** |
| **user workspace (bf16)** | — | **≈ 33 GiB** |

#### 配置 B（中等规模）

| 参数 | 值 |
|------|-----|
| totalQTokens | 131,072 (128K) |
| kvHeads | 8 |
| topK | 8 |
| groupSize | 8 |
| D | 128 |

| 项目 | 字节 |
|------|------|
| accumOut (fp32) | ≈ 32 GiB |
| accumOut (bf16) | ≈ 16 GiB |
| lseMax + lseSum | ≈ 0.5 GiB |
| **user workspace (fp32)** | **≈ 32.5 GiB** |
| **user workspace (bf16)** | **≈ 16.5 GiB** |

#### 配置 C（最小规模，无实际意义）

| 参数 | 值 |
|------|-----|
| totalQTokens | 131,072 (128K) |
| kvHeads | 1 |
| topK | 1 |
| groupSize | 1 |
| D | 128 |

| 项目 | 字节 |
|------|------|
| **user workspace (fp32)** | **≈ 72 MiB** |

### 2.4 实际可用上限

以 64 GiB HBM 为例，`totalQTokens` 的可用上限取决于 workspace dtype 和运行时额外占用：

```
totalQTokens_max ≈ HBM_capacity / (kvHeads × topK × groupSize × D × sizeof(float))
```

| 配置 | per-token workspace (fp32) | 64 GiB HBM 上限 |
|------|---------------------------|----------------|
| kvHeads=4, topK=16, groupSize=16, D=128 | 520 KiB | ≈ 126K tokens |
| kvHeads=8, topK=8, groupSize=8, D=128 | 260 KiB | ≈ 252K tokens |
| kvHeads=1, topK=8, groupSize=8, D=128 | 32.5 KiB | ≈ 2.0M tokens |

**结论**：出版接口的索引验证上限为 `totalQTokens<=1024K`，但这不代表任意
`H=64/G=16/topK=16/D=256` 组合都能在 1024K 下分配成功。1024K 边界采用
`H=1/topK=1/D=128` 常量输入做逐元素精确验证；大 head、topK、D 或 FP32
O_partial 路径必须按上述公式单独核算 HBM。生产配置在 `innerPrecise=1` 下的
128K 真机 workspace 为 35,451,895,808 字节（含 libapi 和 A2 staging）。

---

## 3. uint32_t 溢出风险分析

### 3.1 TilingData 中的 uint32_t 字段

Host 侧 `CalculateTaskSplit` 和 `CalculateReverseIndexMeta` 中的乘法在 `uint32_t` 域内执行：

```cpp
// tiling.cpp
uint32_t totalTaskP1 = numKvBlocks_ * kvHeads_;      // Phase1 任务数
uint32_t totalTaskP2 = totalQTokens_ * kvHeads_;     // Phase2 任务数
uint32_t k2qNnzUpperBound_ = totalQTokens_ * topK_;  // CSR 数据区上界
```

| 表达式 | 溢出条件 (totalQTokens=1024K) | 风险等级 |
|--------|-------------------------------|----------|
| `totalQTokens × kvHeads` | kvHeads ≥ 4096 (1024K×4096=2³²) | 低 |
| `totalQTokens × topK` | topK ≥ 4096 (1024K×4096=2³²) | 低 |
| `numKvBlocks × kvHeads` | 极难达到 | 极低 |

### 3.2 安全的 uint64_t 计算

Workspace 大小计算 (`accumOutSize`, `lseStatSize`, `workSpaceSize`) 均使用 `uint64_t`，无溢出风险。

### 3.3 Kernel 侧 GM 地址计算

Kernel 主体中所有 GM 偏移计算均使用 `uint64_t` 显式转换：

| 位置 | 表达式 | 类型 |
|------|--------|------|
| `SlotStatOffset` | `static_cast<uint64_t>(qToken) × kvHeads + kvHeadIdx` | uint64_t ✓ |
| `perQTokenStride` | `static_cast<uint64_t>(kvHeads) × topK × slotOElems` | uint64_t ✓ |
| `kvHeadBase` | `static_cast<uint64_t>(kvHeadIdx) × topK × slotOElems` | uint64_t ✓ |
| `kvBlockBase` | `static_cast<uint64_t>(physicalBlockId) × blockSize × kvHeads × D` | uint64_t ✓ |
| `csrDataBase` | `static_cast<uint64_t>(kvHeadIdx) × k2qNnzUpperBound` | uint64_t ✓ |
| `btOffset` | `static_cast<int64_t>(batchIdx) × maxBlocksPerBatch + localBlockIdx` | int64_t ✓ |

### 3.4 PV O_partial 地址安全

当前 A2 PV 路径按 group 独立执行一次 Fixpipe scatter，不再用两个目标之间的
`uint32_t dstNdStride` 表达任意 CSR 间距。每个目标偏移均直接按 `uint64_t` 计算：

```cpp
uint64_t wsOOff = static_cast<uint64_t>(qToken) * perQTokenStride
                + kvHeadBase
                + static_cast<uint64_t>(slotK) * slotOElems;
```

该实现牺牲了双目标 Fixpipe 的潜在发射收益，但避免了任意 CSR 顺序下的 group 身份
交换和 32 位 stride 截断。

---

## 4. 各维度约束详情

### 4.1 totalQTokens (Batch × seq_len)

- **类型**：uint32_t (kernel 侧) / uint32_t (host 侧)
- **已验证上限**：1,048,576 (1024K)
- **约束来源**：A2 多 batch 常量边界验证；实际可用值受 Workspace HBM 容量限制
- **校验**：当前 Host 无显式上界拒绝逻辑，由调用方按出版范围和 HBM 保证

### 4.2 D (embeddingSize)

- **类型**：uint32_t
- **支持值**：128、256
- **约束来源**：同一 A2 tiling 路径的模板化 D 处理；两个值均覆盖 `innerPrecise=0/1/4`

### 4.3 blockSize

- **类型**：uint32_t
- **支持值**：0，或 16/32/48/64/80/96/112/128
- **语义**：0 表示连续 TND KV；正值表示分页 KV
- **约束来源**：A2 L1/GM staging 和 16-token 对齐要求
- **隐含上限**：`blockSize × D × 2 ≤ L1_SIZE` (A5/A2: 512KB)

### 4.4 topK

- **类型**：uint32_t
- **默认值**：8
- **上限**：16 (Phase2 UB 容量约束)
- **约束来源**：Phase2 epilogue 中 `lseMaxBuf` / `lseSumBuf` 需容纳 `topK × groupSize × 8` 个 fp32 元素

### 4.5 batch

- **类型**：uint32_t
- **独立出版上限**：未设置
- **约束来源**：与 `totalQTokens`、CSR 大小和 HBM 容量联合约束；增量 packed-row 解码会随 batch 增加控制开销

### 4.6 kvHeads

- **类型**：uint32_t
- **约束范围**：1..`numHeads`，由 `numHeads<=64` 隐含不超过 64
- **隐含上限**：`totalQTokens × kvHeads < 2³²` (uint32_t 溢出)
- **GQA 约束**：`numHeads` 必须能被 `kvHeads` 整除，且 `groupSize<=16`

### 4.7 numHeads

- **类型**：uint32_t
- **已验证上限**：64
- **约束**：必须能被 `kvHeads` 整除，并满足 `groupSize<=16`

### 4.8 groupSize (numHeads / kvHeads)

- **类型**：uint32_t
- **约束来源**：`groupSize × D × topK × sizeof(float)` 须适配 Phase2 UB
- **已验证范围**：1..16（逐值 golden）
- **典型值**：1 (MHA), 4/8/16 (GQA)

### 4.9 kvSeqlen

- **类型**：actual sequence length 为 int32，地址计算使用 uint64_t
- **已验证上限**：262,144 (256K)
- **验证**：三个 `innerPrecise` 模式均有常量全输出精确校验，并补充 GQA16/topK16 随机 sample golden
- **限制**：物理 KV cache 和 block table 仍须完整覆盖实际序列长度

---

## 5. CSR 反向索引维度约束

### 5.1 张量维度

| 张量 | Shape | 约束 |
|------|-------|------|
| `k2qRowPtr` | `[kvHeads, numKvBlocks + 1]` | numKvBlocks 为 uint32_t |
| `k2qQIndices` | `[kvHeads, k2qNnzUpperBound]` | k2qNnzUpperBound = totalQTokens × topK |
| `k2qSlotIndices` | `[kvHeads, k2qNnzUpperBound]` | 同上 |

### 5.2 k2qNnzUpperBound 溢出分析

```
k2qNnzUpperBound = totalQTokens × topK
```

| totalQTokens | topK | k2qNnzUpperBound | 是否溢出 uint32_t |
|--------------|------|------------------|-------------------|
| 131,072 (128K) | 8 | 1,048,576 (1M) | 否 |
| 131,072 (128K) | 16 | 2,097,152 (2M) | 否 |
| 1,048,576 (1024K) | 16 | 16,777,216 (16M) | 否 |
| 131,072 (128K) | 32768 | 4,294,967,296 (4G) | 是 (溢出) |

### 5.3 GM 访问安全

CSR 数据的 GM 基地址计算使用 `uint64_t`：

```cpp
uint64_t rowPtrBase = static_cast<uint64_t>(kvHeadIdx) * (totalKvRows_ + 1);
uint64_t csrDataBase = static_cast<uint64_t>(kvHeadIdx) * k2qNnzUpperBound_;
```

---

## 6. block_table 维度约束

- **Shape**：`[batch, maxBlocksPerBatch]` (int32)
- **maxBlocksPerBatch**：来自 `block_table.shape[1]`，须 ≥ `max_b(ceil(kv_seqlen[b] / blockSize))`
- **GM 访问**：使用 `int64_t` 偏移，无溢出
- **numKvBlocks**：来自 `k2q_row_ptr.shape[1] - 1`，为所有 batch 的 KV block 总数

---

## 7. 硬件平台适配

### 7.1 支持的 AICore 配置

| 平台 | AICore 配置 | __CCE_AICORE__ | L0C | UB | L1 |
|------|-------------|----------------|-----|-----|-----|
| Atlas A5 | ascend950 | 310 | 256 KB | 256 KB | 512 KB |
| Atlas A2 | ascend910b | 220 | 128 KB | 192 KB | 512 KB |

### 7.2 A2 适配要点

| 差异点 | A5 (arch35) | A2 (AtlasA2) |
|--------|-------------|--------------|
| L0C 大小 | 256 KB | 128 KB |
| L0C buffer 分区 | `L0C_HALF_SIZE=128KB`, `L0C_BUF_SIZE=64KB` | `L0C_HALF_SIZE=64KB`, `L0C_BUF_SIZE=32KB` |
| L0C→UB fixpipe | `FixpipeParamsC310` + `CFG_ROW_MAJOR_UB` | `FixpipeParamsV220` + `CFG_ROW_MAJOR` |
| L0C→GM fixpipe | `DataCopyCO12DstParams` + `DataCopy` | `FixpipeParamsV220` + `Fixpipe` |
| Cast+Transdata | `CastTransdataDeconflict` (融合指令) | `Cast` + `DataCopy` (两步) |
| Phase2 reduce | MicroAPI regbase (`ReduceFinalRes_VF`) | 传统 Vector API (`Muls`/`Add`/`Cast`) |
| Phase2 scale | MicroAPI regbase (`ComputeScaleValue_VF`) | 传统 Vector API (`Max`/`Sub`/`Exp`/`Mul`/`Div`) |

### 7.3 Kernel 类复用

Arch35 与 A2 使用独立的顶层 kernel，复用各自架构对应的 block 组件：
- `SasaPrefillRegularKernelA2`：A2 Phase1/Phase2 调度与 staging 生命周期
- `BlockMmadQKPrefillA2` / `BlockMmadPVPrefillA2`：GEMM block
- `BlockEpilogue<EpilogueOnlineSoftmaxPrefillA2, ...>`：Phase1 softmax
- `BlockEpilogue<EpilogueRescaleOPrefillA2, ...>`：Phase2 combine

---

## 8. Tiling Key 派发

| TilingKey | 宏定义 | 含义 | REDtype |
|-----------|--------|------|---------|
| 20001 | `SASA_PREFILL_BF16_D128_TILING` | bf16 输入, `innerPrecise=4`, BF16 score / FP32 O_partial | `float` |
| 20002 | `SASA_PREFILL_BF16_D128_INNER_LOW_TILING` | bf16 输入, `innerPrecise=1`, BF16 O_partial | `bfloat16_t` |
| 20003 | `SASA_PREFILL_BF16_D128_HIGH_SOFTMAX_TILING` | bf16 输入, `innerPrecise=0`, FP32 score / FP32 O_partial | `float` |

选择逻辑（host 侧 `FillTilingData`）：

```cpp
uint64_t tilingKeyVal = SASA_PREFILL_BF16_D128_TILING;
if (innerPrecise_ == 0U) {
    tilingKeyVal = SASA_PREFILL_BF16_D128_HIGH_SOFTMAX_TILING;
} else if (innerPrecise_ == 1U) {
    tilingKeyVal = SASA_PREFILL_BF16_D128_INNER_LOW_TILING;
}
```

---

## 9. 风险点与建议

### 9.1 已识别风险

| 风险 | 严重程度 | 位置 | 说明 |
|------|----------|------|------|
| Workspace 超 HBM 容量 | 高 | tiling.cpp `CalculateWorkSpace` | 参数上界不能直接组合；1024K 仅在轻量边界 shape 验证 |
| totalTaskNumP2 溢出 | 低 | tiling.cpp:206 | uint32_t 域乘法，1024K 时需 kvHeads≥4096 才溢出 |
| k2qNnzUpperBound 溢出 | 低 | tiling.cpp:190 | uint32_t 域乘法，1024K 时需 topK≥4096 才溢出 |

### 9.2 建议

1. **Host 侧增加显式校验**：在 `GetTiling` 中检查出版上界，并在运行时分配失败前给出更明确的 Workspace/HBM 错误日志。
2. **totalTaskNumP2 升级为 uint64_t**：将 `totalTaskNumP2` 字段和计算改为 `uint64_t`，消除大 kvHeads 下的溢出风险。
