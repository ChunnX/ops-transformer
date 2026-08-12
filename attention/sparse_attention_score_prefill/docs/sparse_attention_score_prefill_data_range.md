# SparseAttentionScorePrefill 算子数据范围约束文档

## 1. 概述

本文档描述 `sparse_attention_score_prefill` 算子支持的数据范围，核心约束为 **`Batch × seq_len ≤ 1024K (1,048,576)`**（即 `totalQTokens` 上限）。该限制并非来自单一硬编码校验，而是由 Workspace HBM 容量、uint32_t 溢出风险、GM 地址索引截断等多个因素共同决定。

### 1.1 核心约束摘要

| 约束项 | 限制 | 约束来源 |
|--------|------|----------|
| `totalQTokens` (Batch × seq_len) | ≤ 1,048,576 (1024K) | Workspace HBM 容量 |
| `D` (embeddingSize) | 128 (固定) | TilingKey 硬编码路径 |
| `blockSize` | 128 (默认) | L1 buffer 隐含约束 |
| `topK` | ≤ 16 | Phase2 UB 容量 |
| `batch` | ≤ 4096 (预期) | 线性扫描性能退化 |
| `kvHeads` | ≥ 1 | 无显式上限 |
| `numHeads` | = kvHeads × groupSize | GQA 映射 |

---

## 2. Workspace 大小分析

### 2.1 计算公式

```
slotOElems    = groupSize × D
slotStatElems = groupSize
taskSlots     = totalQTokens × kvHeads × topK

accumOutSize  = taskSlots × slotOElems                         // O_partial 元素数
lseStatSize   = taskSlots × slotStatElems                      // max/sum 各自元素数

accumOutBytes = accumOutSize × (innerPrecise==1 ? 2 : 4)       // bf16=2B, fp32=4B
userWorkspace = accumOutBytes + (lseStatSize × 2) × 4          // max + sum 始终 fp32
workSpaceSize = libapiSize + userWorkspace
```

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

#### 配置 A（大规模 Prefill）

| 参数 | 值 |
|------|-----|
| totalQTokens | 1,048,576 (1024K) |
| kvHeads | 4 |
| topK | 16 |
| groupSize | 16 |
| D | 128 |
| numHeads | 64 |

| 项目 | 元素数 | 字节 |
|------|--------|------|
| accumOutSize | 137,438,953,472 | — |
| lseStatSize | 1,073,741,824 | — |
| accumOut (fp32) | — | ≈ 512 GB |
| accumOut (bf16) | — | ≈ 256 GB |
| lseMax + lseSum | — | ≈ 8 GB |
| **总 workspace (fp32)** | — | **≈ 520 GB** |
| **总 workspace (bf16)** | — | **≈ 264 GB** |

#### 配置 B（中等规模）

| 参数 | 值 |
|------|-----|
| totalQTokens | 1,048,576 (1024K) |
| kvHeads | 8 |
| topK | 8 |
| groupSize | 8 |
| D | 128 |

| 项目 | 字节 |
|------|------|
| accumOut (fp32) | ≈ 256 GB |
| accumOut (bf16) | ≈ 128 GB |
| lseMax + lseSum | ≈ 4 GB |
| **总 workspace (fp32)** | **≈ 260 GB** |
| **总 workspace (bf16)** | **≈ 132 GB** |

#### 配置 C（最小规模，无实际意义）

| 参数 | 值 |
|------|-----|
| totalQTokens | 1,048,576 (1024K) |
| kvHeads | 1 |
| topK | 1 |
| groupSize | 1 |
| D | 128 |

| 项目 | 字节 |
|------|------|
| **总 workspace (fp32)** | **≈ 520 MB** |

### 2.4 实际可用上限

以 Atlas A5/A2 典型 HBM 容量 64 GB 为例，fp32 路径下的 `totalQTokens` 上限：

```
totalQTokens_max ≈ HBM_capacity / (kvHeads × topK × groupSize × D × sizeof(float))
```

| 配置 | per-token workspace (fp32) | 64 GB HBM 上限 |
|------|---------------------------|----------------|
| kvHeads=4, topK=16, groupSize=16, D=128 | 512 KB | ≈ 128K tokens |
| kvHeads=8, topK=8, groupSize=8, D=128 | 32 KB | ≈ 2M tokens |
| kvHeads=1, topK=8, groupSize=8, D=128 | 4 KB | ≈ 16M tokens |

**结论**：在典型 Prefill 配置（大 groupSize + 大 topK）下，Workspace HBM 容量是 `totalQTokens ≤ 1024K` 的首要约束来源。

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
| `totalQTokens × kvHeads` | kvHeads ≥ 4096 (1M×4096=2³²) | 低 |
| `totalQTokens × topK` | topK ≥ 4096 (1M×4096=2³²) | 低 |
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

### 3.4 PV Block 的 dstNdStride 截断风险

`block_mmad_pv_prefill` 中 fixpipe scatter 的 `dstNdStride` 声明为 `uint32_t`，存在截断风险：

```cpp
uint32_t dstNdStride = 1;
dstNdStride = static_cast<int32_t>(nextQ - qTokens[g]) * perQTokenStride
            + static_cast<int32_t>(nextSlot - slotKs[g]) * static_cast<int32_t>(slotOElems);
```

- `perQTokenStride` 是 `uint64_t`，乘法结果为 `uint64_t`
- 赋值给 `uint32_t dstNdStride` 时发生截断
- 典型配置 (kvHeads=8, topK=8, groupSize=8, D=128) 下 `perQTokenStride = 65,536`
- 相邻 qToken 差距 > 65,536 时 `dstNdStride` 溢出
- 实际影响有限：`COPY_GRANULARITY=2` 仅配对相邻 2 个 group，同 batch 内 qToken 差距通常较小

---

## 4. 各维度约束详情

### 4.1 totalQTokens (Batch × seq_len)

- **类型**：uint32_t (kernel 侧) / uint32_t (host 侧)
- **上限**：1,048,576 (1024K)
- **约束来源**：Workspace HBM 容量
- **校验**：无显式校验，由调用方保证

### 4.2 D (embeddingSize)

- **类型**：uint32_t
- **当前值**：128 (固定)
- **约束来源**：TilingKey 硬编码 (`SASA_PREFILL_BF16_D128_TILING` / `SASA_PREFILL_BF16_D128_INNER_LOW_TILING`)
- **扩展**：需新增 TilingKey 路径和对应模板实例化

### 4.3 blockSize

- **类型**：uint32_t
- **默认值**：128
- **约束来源**：L1 buffer 大小 (`blockSize × D × sizeof(bf16)` 须适配 L1)
- **隐含上限**：`blockSize × D × 2 ≤ L1_SIZE` (A5: 512KB, A2: 256KB)

### 4.4 topK

- **类型**：uint32_t
- **默认值**：8
- **上限**：16 (Phase2 UB 容量约束)
- **约束来源**：Phase2 epilogue 中 `lseMaxBuf` / `lseSumBuf` 需容纳 `topK × groupSize × 8` 个 fp32 元素

### 4.5 batch

- **类型**：uint32_t
- **预期上限**：4096
- **约束来源**：`ResolveBatchQSide` 对 batch 做线性扫描，大 batch 下性能退化（非正确性问题）

### 4.6 kvHeads

- **类型**：uint32_t
- **下限**：1
- **隐含上限**：`totalQTokens × kvHeads < 2³²` (uint32_t 溢出)
- **实际范围**：1 ~ 64 (典型 GQA 配置)

### 4.7 groupSize (numHeads / kvHeads)

- **类型**：uint32_t
- **约束来源**：`groupSize × D × topK × sizeof(float)` 须适配 Phase2 UB
- **典型值**：1 (MHA), 4/8/16 (GQA)

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
| 1,048,576 (1024K) | 8 | 8,388,608 (8M) | 否 |
| 1,048,576 (1024K) | 16 | 16,777,216 (16M) | 否 |
| 1,048,576 (1024K) | 4096 | 4,294,967,296 (4G) | 是 (溢出) |

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
| Atlas A2 | ascend910b | 220 | 128 KB | 192 KB | 256 KB |

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

`SasaPrefillRegularKernelArch35` 通过模板参数 `ArchTag` 自动适配 A2：
- `BlockMmadQKPrefillA2` / `BlockMmadPVPrefillA2`：GEMM block
- `BlockEpilogue<EpilogueOnlineSoftmaxPrefillA2, ...>`：Phase1 softmax
- `BlockEpilogue<EpilogueRescaleOPrefillA2, ...>`：Phase2 combine

---

## 8. Tiling Key 派发

| TilingKey | 宏定义 | 含义 | REDtype |
|-----------|--------|------|---------|
| 20001 | `SASA_PREFILL_BF16_D128_TILING` | bf16 输入, D=128, fp32 O_partial | `float` |
| 20002 | `SASA_PREFILL_BF16_D128_INNER_LOW_TILING` | bf16 输入, D=128, bf16 O_partial | `bfloat16_t` |

选择逻辑（host 侧 `FillTilingData`）：

```cpp
uint64_t tilingKeyVal = (innerPrecise_ == 1U)
    ? SASA_PREFILL_BF16_D128_INNER_LOW_TILING   // bf16 路径
    : SASA_PREFILL_BF16_D128_TILING;             // fp32 路径
```

---

## 9. 风险点与建议

### 9.1 已识别风险

| 风险 | 严重程度 | 位置 | 说明 |
|------|----------|------|------|
| Workspace 超 HBM 容量 | 高 | tiling.cpp `CalculateWorkSpace` | 1024K tokens + 大配置下 workspace 达数百 GB |
| PV dstNdStride 截断 | 中 | block_mmad_pv fixpipe scatter | uint64_t→uint32_t 截断，大 qToken 差距下可能写错地址 |
| totalTaskNumP2 溢出 | 低 | tiling.cpp:206 | uint32_t 域乘法，需 kvHeads≥4096 才在 1024K 下溢出 |
| k2qNnzUpperBound 溢出 | 低 | tiling.cpp:190 | uint32_t 域乘法，需 topK≥4096 才在 1024K 下溢出 |

### 9.2 建议

1. **Host 侧增加显式校验**：在 `GetTiling` 中检查 `totalQTokens ≤ 1024K` 和 workspace 大小是否超过设备 HBM 容量，超限时返回 `GRAPH_FAILED` 并输出错误日志。
2. **PV dstNdStride 改为 uint64_t**：将 `block_mmad_pv` 中 `dstNdStride` 类型从 `uint32_t` 改为 `uint64_t`，消除截断风险（需确认 `Fixpipe` / `SetFixpipeNz2ndFlag` 接口是否接受 64 位 stride）。
3. **totalTaskNumP2 升级为 uint64_t**：将 `totalTaskNumP2` 字段和计算改为 `uint64_t`，消除大 kvHeads 下的溢出风险。
