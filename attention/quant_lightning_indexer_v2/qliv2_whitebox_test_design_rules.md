# 白盒用例设计规则 — QuantLightningIndexerV2 (arch35)

## 算子概述

`QuantLightningIndexerV2` 是一个高性能稀疏注意力索引算子，运行于 Ascend 910B 和 Ascend 950 平台。核心功能：

1. **量化 Matmul**：Cube 核心计算 Q×K^T（Q/K 为 INT8/FP8_E4M3FN/HIFLOAT8 量化格式）
2. **加权归约**：Vector 核心执行反量化（qScale/kScale）、逐头权重乘加、跨 Group 头归约
3. **TopK 选择**：在每个 Query 位置上筛选得分最高的 K 个 KV 位置
4. **LD 合并**：当 S2 跨核心切分时，合并各核心的部分 TopK 结果

输出：`sparse_indices` (INT32)、可选的 `sparse_values` (BF16)。

---

## 一、参数维度边界值覆盖

### 1.1 核心维度参数

| 参数 | 最小值 | 典型值 | 最大值 | 特殊边界 |
|------|--------|--------|--------|----------|
| `B` (batch) | 1 | 2, 16 | 大B | **0** — 空batch |
| `S1` (Q seq) | 1 | 2, 8192 | 大S1 | **0** — 空query |
| `S2` (KV seq) | 1 | 8192 | 131072 | **0** — 空KV |
| `N1` (Q heads) | 1 | 64, 128 | 128 | `gSize = N1/N2`，仅64或32（平台相关） |
| `N2` (KV heads) | 1 | 1 | 1 | **固定为1** |
| `D` (head dim) | 128 | 128 | 128 | **固定为128** |
| `topK` (K) | 1 | 2048 | 2048（910B）| **K > validS2Len 时走全排序路径** |
| `cmpRatio` | 1 | 4, 128 | 128 | 910B要求2的幂，950无此要求 |
| `gSize` (=N1/N2) | 32 | 64 | 64 | 910B仅64，950支持32/64 |

### 1.2 量化相关参数

| 参数 | 910B 值 | 950 值 | 说明 |
|------|---------|--------|------|
| `quantMode` | **2** (arch22) | **1** (per-token-head) 或 **4** (per-tensor) | 量化模式 |
| Q/K dtype | INT8 | FP8_E4M3FN / HIFLOAT8 | 量化数据类型 |
| QK_T (matmul输出) | float | bfloat16_t | Cube输出类型 |
| SCORE_T | uint32_t | uint16_t | 得分类型（影响TopK直方图层数） |
| qScale/kScale dtype | half (float16) | float32 | 反量化scale类型 |
| Weight dtype | half (float16) | float32 | 权重类型 |

### 1.3 反量化模式分支

| quantMode | 含义 | 函数路径 |
|-----------|------|----------|
| 1 (per-token-head) | 每个token/head有独立scale | `BatchMulWeightAndReduceSum` |
| 2 (arch22) | 910B INT8路径 | `BatchMulWeightAndReduceSum` (float QK_T) |
| 4 (per-tensor) | 整层共享scale | `BatchMulWeightAndReduceSumPerTensor` |

### 1.4 Attention Mask 参数

| maskMode | 含义 | 行为 |
|----------|------|------|
| 0 | NO_MASK | 全S2范围，无裁剪 |
| 3 | RIGHT_DOWN_CAUSAL | causal mask，`validS2Len` 按三角公式计算 |

mask模式下 `validS2Len` 公式：

```
validS2Len = (s1Offset + actS2SizeOrig - actS1Size + s1BaseSize) / cmpRatio
clamped to [1, actS2Size]
```

### 1.5 Layout 组合

| Q Layout | K Layout | 场景 |
|----------|-----------|------|
| BSND | BSND | 标准4D填充 |
| TND | TND | TND打包（可变长序列） |
| BSND | PA_BBND | BSND Q + PageAttention KV |
| TND | PA_BBND | （不支持的组合） |

**Layout约束：** Q和K layout 必须一致（同为BSND或同为TND），除非K使用PA_BBND。

---

## 二、块切分与尾块覆盖

### 2.1 各轴块大小

| 轴 | 910B 块大小 | 950 块大小 | 尾块条件 |
|----|-------------|-----------|----------|
| **M (S1×G)** | `M_BASE_SIZE = 256` | `M_BASE_SIZE = 256` (`s1BaseSize×gSize=4×gSize`) | `actMBaseSize < 256` |
| **S1** | `S1_BASE_SIZE = 4` | `S1_BASE_SIZE = 4` | `actS1Size < 4` |
| **S2** | `s2BaseSize = 2048` | `s2BaseSize = 128` | `validS2Len < s2BaseSize` |
| **S2 (短Query优化)** | — | `s2BasicBlock = 256` (S1≤4时) | — |
| **D (head dim)** | `D_BASIC_BLOCK = 128` | `D_BASIC_BLOCK = 128` | D=128整除，无尾块 |

### 2.2 S2 长短Query自适应

在 `InitParams` 中：

```
if (maxSeqlenQ <= 4 && maxSeqlenQ >= 0):
    s2BasicBlock_ = 256    // 长S2块 + key cache
else:
    s2BasicBlock_ = 128    // 标准S2块
```

**需覆盖：**

| 场景 | S1 | S2 | s2BasicBlock | 说明 |
|------|----|----|--------------|------|
| S2-01 | 1 | 8192 | 256 | decode场景（长S2块+key cache） |
| S2-02 | 2 | 8192 | 256 | 仍触发key cache |
| S2-03 | 4 | 8192 | 256 | 边界刚好触发 |
| S2-04 | 5 | 8192 | 128 | 标准S2块，不触发cache |
| S2-05 | 8192 | 8192 | 128 | prefill场景 |

**Key cache 行为：** s2BasicBlock=256时，若当前S2范围在已加载范围内，跳过key加载；L1 buffer耗尽或最后S2内循环时cache失效。

### 2.3 M 轴尾块

```
actMBaseSize = Min(M_BASE_SIZE, actS1Size - gS1Idx * S1_BASE_SIZE)
```

| 场景 | S1 | gS1块序 | actMBaseSize | 说明 |
|------|----|---------|-------------|------|
| M-01 | 4 | 0 | 256 | 满块 |
| M-02 | 3 | 0 | 192 (3×64) | M尾块 < 256 |
| M-03 | 5 | 1 | 64 (1×64) | 最后gS1块，尾块 |
| M-04 | 2 | 0 | 128 (2×64) | 小batch |

### 2.4 S2 尾块（validS2Len < s2BaseSize）

影响 TopK 处理路径：

| 场景 | validS2Len | TopK | 路径 |
|------|-----------|------|------|
| TK-01 | ≥ 2048 | 2048 | `validS2Len >= topkCount_` → TopK 选择 |
| TK-02 | < 2048 | 2048 | `validS2Len < topkCount_` → **全排序**（不做TopK筛选） |
| TK-03 | = 0 | — | 填充 -1 / -inf，跳过TopK |
| TK-04 | ≤ 0 且 isNeedLD | — | 填充LD workspace 为 -1/0 |

### 2.5 TopK 多Trunk处理

当 `validS2Len > TRUNK_LEN_16K (16384)` 时，TopK 数据分多个16K trunk处理：

```
if (s2LoopNum == 1):      单trunk TopK，直接调用LiTopKVF
else:                      多trunk：排序→合并→再排序→再合并...
```

**需覆盖**:

| 场景 | validS2Len | trunk数 | 路径 |
|------|-----------|--------|------|
| TK-05 | ≤ 16384 | 1 | 单trunk快速路径 |
| TK-06 | 16385 ~ 32768 | 2 | 多trunk合并TopK |
| TK-07 | 32769 ~ | 3+ | 多次迭代合并 |

### 2.6 TopK 直方图层数

| 平台 | SCORE_T | 直方图方法 | 层数 |
|------|---------|-----------|------|
| 910B | uint32_t | 4层8-bit直方图 | 4 |
| 950 | uint16_t | 2层8-bit直方图 | 2 |

---

## 三、分核场景覆盖（Load Balancing）

### 3.1 分核算法主要场景

| # | 场景 | 触发条件 | 代码路径 |
|---|------|----------|----------|
| 1 | **all-zero捷径** | 所有batch S1==0 或 S2==0 | `BalanceSchedule`→单核心，全部清0 |
| 2 | **单核心分配** | load < avgCost | `usedCoreNum==1` |
| 3 | **batch级分配** | 整BN2代价 ≤ costLimit+容忍度 | `AssignByBatch` |
| 4 | **row级分配** | 整S1G行代价 ≤ costLimit+容忍度 | `AssignByRow` |
| 5 | **block级分配** | supportFd=true，单S2块分配 | `AssignByBlock` |
| 6 | **强制分配** | supportFd=true，core取0块 | `ForceAssign` |
| 7 | **跨核心S2行分裂(LD)** | `curKvSplitPart > 1` | 记录LD信息 + `SplitFD` |

### 3.2 平台差异

| 特性 | ASCEND 910B | ASCEND 950 |
|------|-------------|------------|
| `s2BaseSize` | **2048** | **128** |
| `mBaseSize` | `s1BaseSize × gSize = 4 × gSize` | 同左 |
| `s1BaseSize` | 4 | 4 |
| `supportFd` | 条件启用 | 条件启用 |

**关键影响：**
- 910B 的 s2BaseSize=2048 比 950 的 128 分块粒度粗得多 → 分核精度差异显著
- 相同 workload 在 910B 上可能单核心完成，950 上可能跨多核心

### 3.3 代价计算

```
Cost(M, S2) = COST_WEIGHT_M(6) × align(M/16) + COST_WEIGHT_S2(10) × align(S2/64)
```

**2×2 代价矩阵：**

| M\S2 | NORMAL_BLOCK | TAIL_BLOCK |
|------|-------------|------------|
| NORMAL_BLOCK | `Cost(mBaseSize, s2BaseSize)` | `Cost(mBaseSize, s2TailSize)` |
| TAIL_BLOCK | `Cost(s1GTailSize, s2BaseSize)` | `Cost(s1GTailSize, s2TailSize)` |

零值为0（当对应尾块大小为0时）。

### 3.4 容忍度边界

```
costLimit + lastBlockCost / FA_TOLERANCE_RATIO >= coreCost + assignedCost
```

`FA_TOLERANCE_RATIO = 2`：允许最多超过公平份额 `lastBlockCost/2`。

### 3.5 LD (Lightning Decode) 负载均衡

**LD任务产生条件：** 一个S1G行在FA阶段被跨核心切分 → 需要合并。

**AIV 向量核负载分配：**

```
averageLoad = ceil(totalFDLoad / aivCoreNum)        // 每个AIV核的负载
每个FD task: curFDVectorNum = max(1, taskLoad / averageLoad)  // 至少1个AIV核
M轴均分：curAveMSize = ceil(fdMSize / curFDVectorNum)
```

### 3.6 Metadata 格式

```
[0..AIC_CORE_MAX_NUM*8) = AIC FA metadata (8 INT32s/核)
[AIC_CORE_MAX_NUM*8..1024) = AIV LD metadata (8 INT32s/核)
```

**AIC FA metadata (8 fields):** coreEnable, bN2Start, mStart, s2Start, bN2End, mEnd, s2End, firstLDWorkspaceIdx

**AIV LD metadata (8 fields):** coreEnable, bn2Idx, mIdx, workspaceIdx, workspaceNum, mStart, mNum, (reserved)

---

## 四、Kernel 内部分支场景

### 4.1 Cube 核心 (Matmul) 分支

#### 4.1.1 S2 块自适应

| 条件 | s2BasicBlock | 行为 |
|------|-------------|------|
| `maxSeqlenQ ≤ 4` | 256 | key cache启用，key重用 |
| `maxSeqlenQ > 4` | 128 | 标准，每次重新加载key |

#### 4.1.2 Matmul 尾块处理

```cpp
if ((m / 16) * (n / 16) < 10):  // 小矩阵
    Insert PipeBarrier           // 避免pipeline冲突
```

#### 4.1.3 PageAttention vs 普通Layout

| Layout | Key加载函数 | 说明 |
|--------|-----------|------|
| PA_BBND | `KeyNd2NzForPA` | block table寻址，处理block边界 |
| BSND/TND | `KeyNd2Nz` | 连续寻址 |

### 4.2 Vector 核心分支

#### 4.2.1 加权归约（Vec1）

**核心流程图：**

```
ProcessVec1:
  ├── 计算当前AIV的S1范围 (blockId%2决定处理哪一半)
  ├── curAivS1ProcNum == 0 → 仅sync Cube, return
  ├── 加载 weights (BF16 or FP32)
  ├── [quantMode==4] → BatchMulWeightAndReduceSumPerTensor (per-tensor scale)
  │      ├── batch==1 → MulWeightAndReduceSumPerTensor (单行)
  │      └── batch==2 → MulWeightAndReduceSumPerTensor2 (双行)
  ├── [quantMode!=4] → BatchMulWeightAndReduceSum (per-element scale)
  │      ├── batch==1 → MulWeightAndReduceSum (QK_T=float or bfloat16)
  │      └── batch==2 → MulWeightAndReduceSum2 (双行并行)
  └── CrossCore sync (等Cube完成 → 信号Cube)
```

**batch=1 vs batch=2 区别：**
- S1_BASE_SIZE=4，每个AIV处理2行
- batch=1：单S1行归约
- batch=2：双S1行并行归约（`MulWeightAndReduceSum2`）

#### 4.2.2 TopK 选择（ProcessTopK）

```
ProcessTopK:
  ├── [attenMaskFlag] → 计算mask-adjusted validS2Len
  ├── For each AIV S1 row:
  │   ├── validS2Len<=0 && !isNeedLD → 填充 -1/-inf
  │   ├── validS2Len<=0 && isNeedLD → 填充LD WS为 -1/0
  │   ├── validS2Len >= topkCount_:
  │   │   ├── s2LoopNum==1 → 单trunk TopK (LiTopKVF)
  │   │   └── s2LoopNum>1 → 多trunk合并TopK
  │   └── validS2Len < topkCount_:
  │       └── 全排序 (validS2Len < topkCount_, 无需筛选)
  │   ├── !isNeedLD → 写 final output
  │   └── isNeedLD → 写 LD workspace
  └── End for
```

**TopK 内部关键分支：**

| loopIdx | isNeedLD | 操作 |
|---------|----------|------|
| 0 | false | 首次：直接cast到hisIndexLocal |
| >0 | false | 合并：TopKVF → LiTopKGatherVF merge |
| 0 | true | 首次LD：cast + copy到output |
| >0 | true | 合并LD：TopKVF → LiTopKGatherVF → copy到output |

#### 4.2.3 LD 合并（ProcessLD）

```
ProcessLD:
  ├── [isLdCoreEnable == false] → 跳过
  └── For each output row (mNum):
      └── For each LD workspace chunk (ldProcessNum):
          └── Load partial TopK → LdTopK merge → Write final output
```

#### 4.2.4 outputIdxOffset

| 条件 | 行为 |
|------|------|
| `outputIdxOffset != 0` | 输出索引加上偏移量 |
| `outputIdxOffset == 0` | 不偏移 |

#### 4.2.5 无效输出清理

| 场景 | 操作 |
|------|------|
| `curActSeqLenIsZero` (BSND) | `CleanInvalidOutput` ← 填充 -1 / -inf |
| `needTndPadding` (TND) | `DoTndPadding` ← 填充padding区域 |

### 4.3 编译期分支 (`if constexpr`)

| 模板参数 | 值 | 影响 |
|----------|----|------|
| `PAGE_ATTENTION` | true/false | Key加载方式、kScale加载方式 |
| `Q_LAYOUT_T` | BSND / TND | 序列长度获取方式、offset计算 |
| `K_LAYOUT_T` | BSND / TND / PA_BBND | 同上 |
| `std::is_same_v<QK_T, float>` | true/false | Fixpipe 单/双目标模式 |
| `std::is_same_v<QK_T, bfloat16_t>` | true/false | BF16模式带ReLU激活 |

### 4.4 架构分支 (`#if`)

| 宏 | 平台 | 差异 |
|-----|------|------|
| `__CCE_AICORE__ == 310` | 950 (arch35) | QK_T=bfloat16, SCORE_T=uint16, 2层TopK直方图 |
| `else` | 910B (arch22) | QK_T=float, SCORE_T=uint32, 4层TopK直方图 |
| `ORIG_DTYPE_Q == DT_FLOAT8_E4M3FN` | 950 FP8路径 | `fp8_e4m3fn_t` |
| `ORIG_DTYPE_Q == DT_HIFLOAT8` | 950 HIF8路径 | `hifloat8_t` |
| `ORIG_DTYPE_Q else` | fallback INT8 | `int8_t` |

---

## 五、推荐测试场景矩阵

### 5.1 基本场景

| 编号 | 名称 | 平台 | 关键参数 | 覆盖目标 |
|------|------|------|----------|----------|
| B-01 | 910B decode | 910B | B=1,S1=1,S2=8192,N1=64,quant=2,INT8 | 910B基础路径 |
| B-02 | 950 decode FP8 | 950 | B=1,S1=1,S2=8192,N1=64,quant=1,FP8_E4M3FN | 950 FP8路径 |
| B-03 | 950 decode HIF8 | 950 | B=1,S1=1,S2=8192,N1=64,quant=4,HIFLOAT8 | 950 HIF8+per-tensor |
| B-04 | 910B prefill | 910B | B=1,S1=8192,S2=8192,N1=64,quant=2 | 910B大prefill |
| B-05 | 950 prefill | 950 | B=1,S1=8192,S2=8192,N1=64,quant=1 | 950大prefill |
| B-06 | 950 decode N128 | 950 | B=1,S1=2,N1=128,quant=1,K=2048 | gSize=128 |
| B-07 | 950 decode N32 | 950 | B=1,S1=2,N1=32,quant=1 | gSize=32最小值 |

### 5.2 量化模式场景

| 编号 | quantMode | Q/K dtype | Scale | 覆盖目标 |
|------|-----------|-----------|-------|----------|
| Q-01 | 1 | FP8_E4M3FN | per-token-head float32 | 标准950量化 |
| Q-02 | 4 | HIFLOAT8 | per-tensor float32 | per-tensor反量化 |
| Q-03 | 2 | INT8 | per-token-head half | 910B INT8路径 |

### 5.3 块切分尾块场景

| 编号 | 条件 | 参数 | 覆盖 |
|------|------|------|------|
| T-01 | S2尾块 < 128 (950) | S2 = (128×N + 64) | S2尾块 |
| T-02 | S2尾块 < 2048 (910B) | S2 = (2048×N + 1024) | 910B S2尾块 |
| T-03 | S1尾块 < 4 | S1 = 3, gSize=64 | M尾块 = 192 |
| T-04 | M=256 满块 | S1 ≥ 4 恰好整除 | 满M块 |
| T-05 | M尾块=64 | S1=5, 最后gS1块 | 单行M尾块 |
| T-06 | S1 ≤ 4 → s2BasicBlock=256 | S1=2, S2=8192 | key cache路径 |
| T-07 | S1 > 4 → s2BasicBlock=128 | S1=5, S2=8192 | 标准S2块 |

### 5.4 TopK 专项场景

| 编号 | 条件 | 参数 | 覆盖 |
|------|------|------|------|
| K-01 | validS2Len ≥ topK | validS2Len=8192, K=2048 | TopK标准路径 |
| K-02 | validS2Len < topK | validS2Len=512, K=2048 | 全排序降级路径 |
| K-03 | validS2Len ≤ 0 | validS2Len=0 | 填充无效值 |
| K-04 | 多trunk TopK | validS2Len=32768, K=2048 | 2 trunk合并 |
| K-05 | 单trunk TopK | validS2Len=16384, K=2048 | 单trunk快速路径 |
| K-06 | return_value=1 | 输出values | 值输出路径 |
| K-07 | return_value=0 | 仅输出indices | 无值输出路径 |
| K-08 | outputIdxOffset != 0 | 偏移量 > 0 | 索引偏移 |

### 5.5 分核场景

| 编号 | 条件 | 覆盖 |
|------|------|------|
| C-01 | all-zero (S1=0 或 S2=0) | 单核心, 全清 -1 |
| C-02 | 单batch, 小负载 | 单核心分配 |
| C-03 | 多batch (B=16), 均匀 | batch级分配 |
| C-04 | 单batch, 大S2 | row级/block级分配 |
| C-05 | supportFd + 跨核心S2分裂 | LD任务产生 |
| C-06 | 910B (s2BaseSize=2048) | 粗粒度分核 |
| C-07 | 950 (s2BaseSize=128) | 细粒度分核 |
| C-08 | 极端不平衡 | 1个大batch + N个小batch |

### 5.6 Mask 场景

| 编号 | maskMode | 参数 | 覆盖 |
|------|----------|------|------|
| MK-01 | 0 (NO_MASK) | — | 无掩码全S2 |
| MK-02 | 3 (CAUSAL) | cmpRatio=1 | 因果mask |
| MK-03 | 3 (CAUSAL) | cmpRatio=4 | 因果mask + 压缩KV |
| MK-04 | 3 + validS2Len=0 | S1 ≫ S2 | 超边界裁剪 |

### 5.7 Layout 场景

| 编号 | Q Layout | K Layout | 覆盖 |
|------|----------|-----------|------|
| L-01 | BSND | BSND | 标准4D |
| L-02 | TND | TND | 可变长打包 |
| L-03 | BSND | PA_BBND | PageAttention KV |
| L-04 | TND (padding) | TND | sequsedQ < cuSeqlensQ → 需TND padding处理 |

### 5.8 可选输入场景

| 编号 | 条件 | 覆盖 |
|------|------|------|
| O-01 | seqused_q 提供 | 实际序列长度 |
| O-02 | seqused_q = null | fallback到cuSeqlens或maxSeqlen |
| O-03 | seqused_k 提供 | 实际KV长度 |
| O-04 | seqused_k = null | fallback |
| O-05 | cu_seqlens_q + TND | TND累计序列长度 |
| O-06 | cu_seqlens_k + TND | TND累计序列长度 |
| O-07 | block_table + PA_BBND | PageAttention寻址 |
| O-08 | cmp_residual_k ≠ 0 | cmpRatio≠1时修正压缩后长度 |
| O-09 | output_idx_offset 提供 | 解码阶段索引偏移 |

### 5.9 综合场景

| 编号 | 描述 |
|------|------|
| Z-01 | 950 + BSND+PA_BBND + decode + SCFA + FP8_E4M3FN — 典型生产场景 |
| Z-02 | 950 + BSND+BSND + prefill + S1=8192 + K=2048 — 大prefill |
| Z-03 | 910B + TND+TND + mask + cmpRatio=4 — 910B复合场景 |
| Z-04 | 950 + BSND+PA_BBND + HIFLOAT8 + per-tensor — HIF8专用路径 |
| Z-05 | 多batch变长 + TND + padding — TND padding覆盖 |
| Z-06 | decode + return_value + outputIdxOffset — 全输出特征 |
| Z-07 | S2尾块 + mask + 跨核心LD — 多特征组合 |

---

## 六、用例设计 Checklist

- [ ] `N1` 取 **32/64/128**（950），**64**（910B）
- [ ] `N2` = **1**
- [ ] `D` = **128**
- [ ] `quantMode` 的 **1 / 2 / 4**
- [ ] Q/K dtype: **INT8 / FP8_E4M3FN / HIFLOAT8**
- [ ] `QK_T`: **float (910B) / bfloat16 (950)**
- [ ] `SCORE_T`: **uint32 (910B) / uint16 (950)**
- [ ] `cmpRatio`: **1 / 4 / 128**（910B仅2的幂）
- [ ] `maxSeqlenQ ≤ 4` → s2BasicBlock=256 (key cache)
- [ ] `maxSeqlenQ > 4` → s2BasicBlock=128
- [ ] `S1` 整除/不整除 **S1_BASE_SIZE (4)**
- [ ] `validS2Len` 的 **0 / < K / ≥ K / > 16384(多trunk)**
- [ ] `s2BaseSize` 的 **128 (950) / 2048 (910B)**
- [ ] `maskMode` 的 **0 (NO_MASK) / 3 (CAUSAL)**
- [ ] `PAGE_ATTENTION` 的 **true / false**
- [ ] `Q_LAYOUT_T` 的 **BSND / TND**
- [ ] `K_LAYOUT_T` 的 **BSND / TND / PA_BBND**
- [ ] Layout 组合：BSND+BSND, TND+TND, BSND+PA_BBND
- [ ] `return_value` 的 **0 / 1**
- [ ] `output_idx_offset` 的 **0 / > 0**
- [ ] `curActSeqLenIsZero` → CleanInvalidOutput 路径
- [ ] `needTndPadding` → DoTndPadding 路径
- [ ] `isNeedLD` → LD 路径 (跨核心split)
- [ ] Vec1 归约: **batch=1 / batch=2**
- [ ] TopK: **单trunk / 多trunk / 全排序**
- [ ] coreNum: **1核 / 部分核 / 全部核**
- [ ] 平台: **ASCEND 910B / ASCEND 950**
