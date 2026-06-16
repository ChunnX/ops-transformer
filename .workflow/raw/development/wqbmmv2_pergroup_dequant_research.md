# wqbmmv2 PerGroup 反量化实现调研与 GMM A16W4 适配方案

> **调研日期**: 2026-06-16
> **调研范围**: wqbmmv2 arch35 reg_base 路径的 pergroup 反量化实现
> **目标**: 为 GMM A16W4 INT4 ND pergroup 的 kernel 实现提供设计依据

---

## §1. 调研背景与目标

### 1.1 为什么需要调研 wqbmmv2 的 pergroup 反量化实现

GMM A16W4 pergroup ND 的 kernel 实现需要参照 wqbmmv2（`weight_quant_batch_matmul_v2`）的 pergroup 反量化体系。wqbmmv2 是目前 arch35 平台上 pergroup 反量化实现最完整的参照算子，覆盖了 W4/W8、KN/NK、多种 groupSize（32/64/128/256/Gt128）的全部分支。

GMM 的 `GMMWeightQuantBasicController`（A16W8 多多多）体系已有 perchannel 和 pertensor 的 ND 反量化实现，但 **pergroup ND 路径尚未完整实现**。需要理解 wqbmmv2 的 pergroup 实现细节，才能正确适配到 GMM 框架中。

### 1.2 GMM A16W4 pergroup ND 的核心挑战

**weight 可转置但 scale/offset 始终非转置** — 这是与 wqbmmv2 最关键的差异。

- **wqbmmv2**：scale/offset 的 GM 布局随 `bTrans` 变化（`bTrans=false` → `[G, N]`，`bTrans=true` → `[N, G]`）
- **GMM A16W4 pergroup**：scale/offset 的 GM 布局**始终为 `[E, G, N]`**（多多多）或 `[G, N]`（单单单），不随 `transB` 变化

这意味着当 weight 转置（`transB=true`，NK 路径）时，GMM 的 scale/offset 布局与 wqbmmv2 不同，需要修改 VF 函数中的 scale 地址公式。

---

## §2. wqbmmv2 pergroup 反量化实现分析

### 2.1 整体架构

**入口文件**: `mc2/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2.cpp`

**Process() 主流程**（`weight_quant_batch_matmul_v2_reg_base.h`，第 42-64 行）：

```
Process()
  InitSync()
  while (IterMatmulOut()) {          // 遍历 M/N tile
    for (kFactorIdx = 0; kFactorIdx < kSingleCoreIterNum_; kFactorIdx++) {
      GetAL1(kFactorIdx)             // A 矩阵 tile GM→L1（AIC 流水线）
      GetBL1(kFactorIdx)             // B 矩阵: GM→UB→反量化→L1（AIV 流水线）
      GetBiasL1(kFactorIdx)          // Bias 加载（AIC 流水线）
      IterateMatmul(kFactorIdx)      // Cube 矩阵乘（AIC 流水线）
      PostProcess(kFactorIdx)        // Buffer 索引轮转与同步
    }
    GetTensorC()                     // 结果 L0C→GM
  }
  EndSync()
```

**反量化在 AIV 流水线中执行**，调用链为：

```
GetBL1() (line 2033)
  → CopyUb2L1() (line 2070)
    → VectorProcess() (line 2297)
      → BL1Process()
        → BL1ProcessNK() / BL1ProcessKN()  // 按 bTrans 分派
          → CopyInWeight()
          → CopyInScaleOffset()     // scale/offset GM→UB
          → VectorRegCompute() (line 1701)
            → AntiQuantCompute() (line 1591)  // 反量化核心分派
          → CopyVecOut2L1()
```

**核心实现文件**:
- `weight_quant_batch_matmul_v2_reg_base_common.h` — CopyInScaleOffset (line 987)、AntiQuantCompute (line 1591)、AntiquantComputeKNPerGroup (line 1246)、AntiquantComputeW4NKPerGroup (line 1294)
- `weight_quant_batch_matmul_v2_vf.h` — 所有 VF 函数实现

### 2.2 Scale/Offset GM→UB 搬运（CopyInScaleOffset）

**文件**: `weight_quant_batch_matmul_v2_reg_base_common.h`，第 987-1028 行

**这是最关键的部分，bTrans 对 scale/offset 布局有决定性影响。**

#### bTrans=false（KN 路径，weight 布局为 [K, N]）

```cpp
// 第 1014-1022 行
// GM 布局: [G, N]，即 scale[g][n]，其中 g = k / groupSize
intriParams.blockLen = bubNLen * sizeof(xType);           // 每次搬 N 个元素
intriParams.srcStride = (tiling_->nSize - bubNLen) * sizeof(xType);  // GM 行间距
gmOffset = bubKOffset / tiling_->groupSize * tiling_->nSize + bubNOffset;
ubOffset = ubInBufIdx_ * vecScaleOffsetSize_;
intriParams.blockCount = groupNumBub_;                     // G 个 block
// UB 布局: [G_tile, N_tile]
```

**解读**: scale 在 GM 中按 `[K/groupSize, N]` 排列（N 为内轴）。搬运时按 group 维度分 block，每个 block 搬 `bubNLen` 个连续 N 元素，共 `groupNumBub_` 个 block。

#### bTrans=true（NK 路径，weight 布局为 [N, K]）

```cpp
// 第 1003-1013 行
// GM 布局: [N, G]，即 scale[n][g]  ← 关键差异！
if (bubKLoopIdx != 0) { return; }  // 仅在第一次 K 循环时搬运
intriParams.blockCount = bubNLen;                           // N 个 block
intriParams.blockLen = groupNumBub_ * sizeof(xType);        // 每次搬 G 个元素
intriParams.srcStride = (groupNum_ - groupNumBub_) * sizeof(xType);  // 跳过剩余 group
gmOffset = bubNOffset * CeilDiv(tiling_->kSize, tiling_->groupSize) + bubKOffset / tiling_->groupSize;
ubOffset = ubInBufIdx_ * vecScaleOffsetSize_;
// UB 布局: [N_tile, G_tile_aligned]
```

**解读**: scale 在 GM 中按 `[N, K/groupSize]` 排列（G 为内轴）。搬运时按 N 维度分 block，每个 block 搬 `groupNumBub_` 个连续 group 元素，共 `bubNLen` 个 block。

#### UB Buffer 大小分配（InitUbBuffer，第 853-863 行）

```cpp
if constexpr (bTrans) {
    // NK: UB 布局 [N_tile, G_tile_aligned]
    vecScaleOffsetSize_ = tiling_->nBubSize *
        CeilAlign(CeilDiv(tiling_->kBubSize, tiling_->groupSize), ONE_BLK_X_NUM);
} else {
    // KN: UB 布局 [G_tile, N_tile]
    vecScaleOffsetSize_ = CeilDiv(tiling_->kBubSize, tiling_->groupSize) * tiling_->nBubSize;
}
```

#### 核心发现

> **wqbmmv2 中，当 weight 转置时，scale/offset 在 GM 中也是转置布局 `[N, G]`。这意味着 wqbmmv2 的 weight 和 scale/offset 的转置状态是一致的。**

### 2.3 KN 路径（bTrans=false）：AntiquantComputeKNPerGroup

**文件**: `weight_quant_batch_matmul_v2_reg_base_common.h`，第 1246-1287 行

> **重要**：KN 路径使用**统一的** `AntiquantPerGroupKN` VF 函数处理**所有 groupSize**（32/64/128/256），不按 groupSize 分派不同的 VF 函数。这与 NK 路径（§2.4）形成鲜明对比 — NK 路径按 groupSize 分派 4 个专用 VF 函数。
>
> 原因：KN 路径的 scale 按 `(group, N-tile)` 加载，一个 group 的 scale 在 K 内循环中被复用 groupSize 次。groupSize 的大小只影响 K 内循环的迭代次数，不影响 scale 的加载方式。

**参数构造**（`ParamsKN<xType>`）：

```cpp
// 第 1251-1284 行
param.nBubXTypeAlign = CeilAlign(nBub, BLOCK_CUBE);       // BLOCK_CUBE = 16
param.nBubWTypeAlign = CeilAlign(nBub, ONE_BLK_W_NUM);    // ONE_BLK_W_NUM = 32 (int4)
param.vfElemB16 = VEC_MAX_ELEM_B16;                        // 128 (fp16 向量寄存器元素数)
param.nLoop = CeilDiv(nBubWTypeAlign, VEC_MAX_ELEM_B16);   // N 方向循环次数
param.groupNum = CeilDiv(kBub, groupSize) - 1;              // 主 group 数（减 1 因为 tail 单独处理）
param.groupTail = kBub % groupSize > 0 ? kBub % groupSize : groupSize;
param.dataBlockStride = CeilAlign(kBub, BLOCK_CUBE) + 1;   // NZ 输出跳行避免 bank 冲突
// W4 参数:
param.wNStride = VEC_MAX_ELEM_B16 >> INT4_DTYPE_PARAM;     // 128 >> 1 = 64
param.wKStride = nBubWTypeAlign >> INT4_DTYPE_PARAM;
param.wGroupStride = groupSize * nBubWTypeAlign >> INT4_DTYPE_PARAM;
```

**VF 调用**（第 1286 行）：

```cpp
AscendC::VF_CALL<AntiquantPerGroupKN<xType, wType, hasAntiQuantOffset>>(param);
```

#### VF 函数：AntiquantPerGroupKN

**文件**: `weight_quant_batch_matmul_v2_vf.h`，第 346-443 行

| 维度 | 详情 |
|------|------|
| **Scale 加载** | `DIST_NORM`（直接加载 128 个 FP16 元素） |
| **Scale 地址公式** | `CreateAddrReg<XType>(nIdx, vfElemB16, gIdx, nBubXTypeAlign)` |
| **地址含义** | scale UB 布局 `[G, N_tile_align]`，第 gIdx 行第 nIdx 列 |
| **Weight 加载** | W8: `DIST_UNPACK_B8`; W4: `DIST_UNPACK4_B8` |
| **Weight 地址公式** | `CreateAddrReg<WType>(nIdx, wNStride, gIdx, wGroupStride, kIdx, wKStride)` |
| **循环结构** | `nLoop → groupNum → groupSize`（三层循环） |
| **Cast** | W4: int4→fp16 直接 cast; W8: int8→fp16，若 xType=bf16 则再 fp16→bf16 |
| **输出** | `DATA_BLOCK_COPY + POST_MODE_UPDATE`（NZ 格式，K 为内轴） |
| **Tail 处理** | 最后一个 group 单独处理 `groupTail` 个 K 元素 |

**关键代码片段**（第 356-399 行）：

```cpp
for (uint16_t nIdx = 0; nIdx < param.nLoop; ++nIdx) {
    maskWeight = MicroAPI::UpdateMask<XType>(maskWeightValue);
    for (uint16_t gIdx = 0; gIdx < param.groupNum; ++gIdx) {
        // Scale 加载: 从 UB [G, N_tile_align] 布局中加载第 gIdx 行
        MicroAPI::AddrReg addRegScale =
            MicroAPI::CreateAddrReg<XType>(nIdx, param.vfElemB16, gIdx, param.nBubXTypeAlign);
        MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_NORM>(scale, param.scaleBaseAddr, addRegScale);

        for (uint16_t kIdx = 0; kIdx < param.groupSize; ++kIdx) {
            // Weight 加载 + Cast + Add offset + Mul scale
            MicroAPI::AddrReg addRegWeight = MicroAPI::CreateAddrReg<WType>(
                nIdx, param.wNStride, gIdx, param.wGroupStride, kIdx, param.wKStride);
            MicroAPI::DataCopy<WType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(weightIn, ..., addRegWeight);
            MicroAPI::Cast<XType, WType, castTraitNorm>(weightOut, weightIn, maskWeight);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightOut, weightOut, offset, maskWeight);
            }
            MicroAPI::Mul(weightOut, weightOut, scale, maskWeight);
            MicroAPI::DataCopy<XType, DATA_BLOCK_COPY, POST_MODE_UPDATE>(
                param.weightOutBaseAddr, weightOut, param.dataBlockStride, param.repeatStride, maskWeight);
        }
    }
    // 处理 groupTail ...
    param.weightOutBaseAddr += param.weightOutStride;
}
```

### 2.4 NK 路径（bTrans=true）：AntiquantComputeW4NKPerGroup

**文件**: `weight_quant_batch_matmul_v2_reg_base_common.h`，第 1294-1410 行

**按 groupSize 分派 4 个专用 VF 函数**（加上 Gt128 和 Odd 共 6 个分支）：

| groupSize | VF 函数 | Params 结构体 | 行号 |
|-----------|---------|--------------|------|
| 32 | `AntiquantW4Pergroup32NK` | `ParamsGroupSize32` | 1296-1326 |
| 64 | `AntiquantW4Pergroup64NK` | `ParamsGroupSize64` | 1327-1352 |
| 128 | `AntiquantW4Pergroup128NK` | `ParamsGroupSize128And256` | 1353-1380 |
| 256 | `AntiquantW4Pergroup256NK` | `ParamsGroupSize128And256` | 1381-1410 |
| >128, ≠256 | `AntiquantW4PergroupGt128NKCase1/2` | `ParamsGroupSizeGt128` | — |
| %64≠0 | `AntiquantW4Pergroup32OddNK` | `ParamsGroupSize32OddNK` | — |

**公共参数**（所有 groupSize 共享）：

```cpp
params.outerStrideScale = CeilAlign(groupNumBub_, ONE_BLK_X_NUM);  // scale UB 外轴步长
params.dataBlockStride = CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1;    // NZ 输出跳行
params.repeatStride = 1;
```

**useVag 分派策略**：当循环维度 > 1 时使用 `useVag=true`（AddrReg 硬件地址寄存器），= 1 时使用 `useVag=false`（直接指针运算），避免编译器限制。

#### 2.4.1 gs=128：AntiquantW4Pergroup128NK

**文件**: `weight_quant_batch_matmul_v2_vf.h`，第 639-691 行

| 维度 | 详情 |
|------|------|
| **Scale 加载** | `DIST_BRC_B16`（广播单个 FP16 到 128 元素） |
| **Scale 地址** | `scaleBaseAddr + groupIdx + nBubIdx * outerStrideScale` |
| **Scale AddrReg** | `CreateAddrReg<XType>(groupIdx, 1, nBubIdx, outerStrideScale)` |
| **Weight 加载** | `DIST_UNPACK4_B8`，128 元素 = 1 个向量寄存器 |
| **Weight 地址** | `weightInBaseAddr + groupIdx * innerStrideWeight + nBubIdx * outerStrideWeight` |
| **innerStrideWeight** | 64（= groupSize/2，因为 INT4 每字节 2 个元素） |
| **循环结构** | `group → N`（两层循环，无内层 K 循环） |
| **输出** | `DATA_BLOCK_COPY + POST_MODE_UPDATE`，mask=`bubKLen` |

**关键代码**（第 653-690 行）：

```cpp
for (uint16_t groupIdx = 0; groupIdx < p.groupNum; ++groupIdx) {
    for (uint16_t nBubIdx = 0; nBubIdx < p.bubNLen; ++nBubIdx) {
        // useVag=false 路径:
        MicroAPI::DataCopy<XType, DIST_BRC_B16>(
            scale, p.scaleBaseAddr + groupIdx + nBubIdx * p.outerStrideScale);
        MicroAPI::DataCopy<WType, DIST_UNPACK4_B8>(
            weightB4, p.weightInBaseAddr + groupIdx * p.innerStrideWeight + nBubIdx * p.outerStrideWeight);
        MicroAPI::Cast<XType, WType, castTrait>(weightB16, weightB4, maskAll);
        MicroAPI::Mul(weightB16, weightB16, scale, maskAll);
        MicroAPI::DataCopy<XType, DATA_BLOCK_COPY, POST_MODE_UPDATE>(
            p.weightOutBaseAddr, weightB16, p.dataBlockStride, p.repeatStride, maskWeight);
    }
    p.weightOutBaseAddr += p.weightOutAddrOffset;
}
```

**核心特点**: groupSize=128 恰好等于一个向量寄存器的元素数（128 个 fp16），因此每个 group 只需 1 次 weight 加载，无需内层 K 循环。这是最简单的 NK pergroup VF 函数。

#### 2.4.2 gs=256：AntiquantW4Pergroup256NK

**文件**: `weight_quant_batch_matmul_v2_vf.h`，第 693-761 行

| 维度 | 详情 |
|------|------|
| **Scale 加载** | `DIST_BRC_B16`（同 gs=128） |
| **Scale 地址** | `scaleBaseAddr + groupIdx + nBubIdx * outerStrideScale`（同 gs=128） |
| **Weight 加载** | `DIST_UNPACK4_B8`，**2 次加载**（各 128 元素 = 256 元素 = 1 个 group） |
| **Weight 地址** | `weightInBaseAddr + groupIdx * innerStrideWeight + nBubIdx * outerStrideWeight`（第 1 次）<br>`weightInBaseAddr1 + ...`（第 2 次，偏移 64 字节） |
| **innerStrideWeight** | 128（= groupSize/2） |
| **循环结构** | `group → N`（两层循环，每次处理 2 个 VL） |
| **输出** | 2 次 `DATA_BLOCK_COPY`，分别到 `weightOutBaseAddr` 和 `weightOutBaseAddr1` |

**关键代码**（第 725-754 行）：

```cpp
// useVag=false 路径:
MicroAPI::DataCopy<XType, DIST_BRC_B16>(
    scale, p.scaleBaseAddr + groupIdx + nBubIdx * p.outerStrideScale);
// 2 次 weight 加载
MicroAPI::DataCopy<WType, DIST_UNPACK4_B8>(
    weightB4, p.weightInBaseAddr + groupIdx * p.innerStrideWeight + nBubIdx * p.outerStrideWeight);
MicroAPI::DataCopy<WType, DIST_UNPACK4_B8>(
    weight1B4, p.weightInBaseAddr1 + groupIdx * p.innerStrideWeight + nBubIdx * p.outerStrideWeight);
// Cast + Mul 各 2 次
MicroAPI::Cast<XType, WType, castTrait>(weightB16, weightB4, maskAll);
MicroAPI::Cast<XType, WType, castTrait>(weight1B16, weight1B4, maskAll);
MicroAPI::Mul(weightB16, weightB16, scale, maskAll);
MicroAPI::Mul(weight1B16, weight1B16, scale, maskAll);
// 2 次输出
MicroAPI::DataCopy<XType, DATA_BLOCK_COPY, POST_MODE_UPDATE>(
    p.weightOutBaseAddr, weightB16, p.dataBlockStride, p.repeatStride, maskWeight);
MicroAPI::DataCopy<XType, DATA_BLOCK_COPY, POST_MODE_UPDATE>(
    p.weightOutBaseAddr1, weight1B16, p.dataBlockStride, p.repeatStride, maskWeight1);
```

#### 2.4.3 gs=64：AntiquantW4Pergroup64NK

**文件**: `weight_quant_batch_matmul_v2_vf.h`，第 566-637 行

| 维度 | 详情 |
|------|------|
| **Scale 加载** | `DIST_BRC_B16` × 2 + `Select(maskH)` 拼接 |
| **Scale 地址** | `scaleBaseAddr00 + innerIdx * 2 + outerIdx * outerStrideScale`（第 1 个 group）<br>`scaleBaseAddr01 + innerIdx * 2 + outerIdx * outerStrideScale`（第 2 个 group，偏移 +1） |
| **Scale AddrReg** | `CreateAddrReg<XType>(innerIdx, 2, outerIdx, outerStrideScale)` |
| **Weight 加载** | `DIST_UNPACK4_B8`，64 元素 = 半个向量寄存器 |
| **循环结构** | `inner(K/128) → outer(N)` |
| **Select 拼接** | `Select(scale0, oriScale00, oriScale01, maskH)` — 前半=oriScale00, 后半=oriScale01 |

**关键代码**（第 606-623 行，useVag=false 路径）：

```cpp
// 加载 2 个相邻 group 的 scale（各广播到 128 元素）
MicroAPI::DataCopy<XType, DIST_BRC_B16>(
    oriScale00, p.scaleBaseAddr00 + innerIdx * 2 + outerIdx * p.outerStrideScale);
MicroAPI::DataCopy<XType, DIST_BRC_B16>(
    oriScale01, p.scaleBaseAddr01 + outerIdx * p.outerStrideScale + innerIdx * 2);
// Select 拼接: 前 64 元素 = oriScale00, 后 64 元素 = oriScale01
MicroAPI::Select(scale0, oriScale00, oriScale01, maskRegSelect);

// Weight 加载（64 元素 = 半个 VL）
MicroAPI::DataCopy<WType, DIST_UNPACK4_B8>(
    weight0B4, p.weightInBaseAddr0 + innerIdx * 64 + outerIdx * p.outerStrideWeight);
```

**核心特点**: groupSize=64 意味着 2 个 group 恰好填满一个向量寄存器（128 元素）。通过 `Select` 指令将两个广播结果拼接，实现一次处理 2 个 group。

`scaleBaseAddr00` 和 `scaleBaseAddr01` 相差 1 个元素（第 1338-1342 行）：

```cpp
params.scaleBaseAddr00 = scaleBaseAddr1_ + 0;
params.scaleBaseAddr01 = scaleBaseAddr1_ + 1;
```

#### 2.4.4 gs=32：AntiquantW4Pergroup32NK

**文件**: `weight_quant_batch_matmul_v2_vf.h`，第 445-564 行

| 维度 | 详情 |
|------|------|
| **Scale 加载** | `DIST_E2B_B16` 一次加载 8 个连续 group scale + `Interleave` 扩展 |
| **Scale 地址** | `scaleBaseAddr0 + innerIdx * 8 + outerIdx * outerStrideScale` |
| **Scale AddrReg** | `CreateAddrReg<XType>(innerIdx, 8, outerIdx, outerStrideScale)` |
| **Weight 加载** | `DIST_UNPACK4_B8` × 2（各 128 INT4 = 256 元素 = 8 个 group） |
| **循环结构** | `inner(K/256) → outer(N)` |
| **Interleave** | `Interleave(scale0, scale1, scale0, scale0)` — 将 8 个 scale 值扩展到 256 元素 |

**关键代码**（第 506-539 行，useVag=false 路径）：

```cpp
// DIST_E2B_B16: 加载 8 个连续元素，每个广播到 16 个
// SRC: s0 s1 s2 s3 s4 s5 s6 s7
// DST: s0×16 s1×16 s2×16 s3×16 s4×16 s5×16 s6×16 s7×16
MicroAPI::DataCopy<XType, DIST_E2B_B16>(
    scale0, p.scaleBaseAddr0 + innerIdx * 8 + outerIdx * p.outerStrideScale);

// Interleave 后:
// scale0: s0×32 s1×32 s2×32 s3×32 (groups 0-3 各 32 元素)
// scale1: s4×32 s5×32 s6×32 s7×32 (groups 4-7 各 32 元素)
MicroAPI::Interleave(scale0, scale1, scale0, scale0);

// 2 次 weight 加载（各 128 INT4 = 128 元素）
MicroAPI::DataCopy<WType, DIST_UNPACK4_B8>(
    weightB4Vl0, p.weightInBaseAddr + innerIdx * 128 + outerIdx * p.outerStrideWeight);
MicroAPI::DataCopy<WType, DIST_UNPACK4_B8>(
    weightB4Vl1, p.weightInBaseAddr + 64 + innerIdx * 128 + outerIdx * p.outerStrideWeight);
```

**核心特点**: groupSize=32 意味着 4 个 group 填满一个向量寄存器。利用 `DIST_E2B_B16`（Element-to-Block）指令一次加载 8 个连续 group 的 scale 值，再通过 `Interleave` 将每个值重复 32 次，覆盖 256 个 weight 元素（8 个 group × 32 元素）。

**`DIST_E2B_B16` 要求源地址连续** — 8 个 group 的 scale 值必须在 GM/UB 中连续存放。

### 2.5 KN vs NK 路径对比表

| 维度 | KN (bTrans=false) | NK (bTrans=true) |
|------|-------------------|------------------|
| **GM scale 布局** | `[G, N]`（G 为外轴，N 为内轴） | `[N, G]`（N 为外轴，G 为内轴） |
| **UB scale 布局** | `[G_tile, N_tile_align]` | `[N_tile, G_tile_align]` |
| **Scale 加载指令** | `DIST_NORM`（连续加载 128 元素） | `DIST_BRC_B16`（广播）/ `DIST_E2B_B16`（元素到块） |
| **VF 函数数量** | **1 个统一**（`AntiquantPerGroupKN`，所有 groupSize 共用） | **4+ 个按 groupSize 分派** |
| **gs=32 处理** | 与 gs=128/256 完全相同（K 内循环迭代 32 次，scale 复用 32 次） | 专用 `AntiquantW4Pergroup32NK`，使用 `DIST_E2B_B16` + `Interleave` |
| **输出 NZ 内轴** | K（`dataBlockStride = CeilAlign(kBub, BLOCK_CUBE) + 1`） | N（`dataBlockStride = CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1`） |
| **Scale GM→UB 搬运** | `blockCount=groupNumBub_`, `blockLen=bubNLen` | `blockCount=bubNLen`, `blockLen=groupNumBub_` |
| **UB buffer 大小** | `CeilDiv(kBubSize, groupSize) * nBubSize` | `nBubSize * CeilAlign(CeilDiv(kBubSize, groupSize), ONE_BLK_X_NUM)` |
| **循环外层** | N → group → K | group → N（gs=128/256）或 inner_K → N（gs=32/64） |

---

## §3. GMM A16W4 pergroup 的核心差异

### 3.1 Scale/Offset 布局约束

**wqbmmv2**：scale/offset 的 GM 布局随 `transB` 变化：
- `bTrans=false` → `[G, N]`（即 `[K/groupSize, N]`）
- `bTrans=true` → `[N, G]`（即 `[N, K/groupSize]`）

**GMM A16W4 pergroup**：scale/offset 的 GM 布局**始终为 `[G, N]`**（单单单场景）或 `[E, G, N]`（多多多场景），不随 `transB` 变化。

这一约束来自 GMM 的接口定义（参见 `grouped_matmul/README.md` 中伪量化 pergroup 的 shape 约束表）：

| 使用场景 | shape 限制 |
|---------|-----------|
| 伪量化 pergroup, weight 单 tensor | `[g, G, n]` |
| 伪量化 pergroup, weight 多 tensor | `[G_i, n_i]` |

### 3.2 数学原理

pergroup 反量化公式：

```
w_dequant[k][n] = (w_int4[k][n] + offset[g(k)][n]) × scale[g(k)][n]
其中 g(k) = k / groupSize
```

当 weight 转置时（`transB=true`），weight 在 GM 中为 `[N, K]`，但 scale/offset 仍为 `[G, N]`。

这意味着对于 weight 的每个元素 `w[n][k]`，需要查找 `scale[k/groupSize][n]`。

**在 NK 路径中**，VF 按 `(group, N)` 或 `(inner_K_block, N)` 循环。对于每个 `(groupIdx, nBubIdx)` 组合：

- **weight 地址**（转置 weight 的 NK 布局）：
  ```
  weightBase + groupIdx * innerStrideWeight + nBubIdx * outerStrideWeight
  ```

- **scale 地址（GMM `[G, N]` 布局）**：
  ```
  scaleBase + groupIdx * nTileAlign + nBubIdx
  ```
  其中 `nTileAlign = CeilAlign(nRealLen, BLOCK_CUBE)`

- **scale 地址（wqbmmv2 `[N, G]` 布局）**：
  ```
  scaleBase + groupIdx + nBubIdx * outerStrideScale
  ```
  其中 `outerStrideScale = CeilAlign(groupNumBub_, ONE_BLK_X_NUM)`

**两者地址公式完全不同！** GMM 的 scale 地址以 group 为外轴（步长 `nTileAlign`），N 为内轴；wqbmmv2 的 scale 地址以 N 为外轴（步长 `outerStrideScale`），group 为内轴。

### 3.3 影响分析

| 路径 | 影响 | 原因 |
|------|------|------|
| **KN (bTrans=false)** | **可直接复用** | weight 和 scale 都是 `[G, N]` 相关布局，与 wqbmmv2 一致 |
| **NK (bTrans=true)** | **需要修改 scale/offset 地址公式** | wqbmmv2 的 NK 假设 scale 为 `[N, G]`，但 GMM 的 scale 始终为 `[G, N]` |

**GMM 现有代码的佐证**：

1. **`CopyAntiQuantParamsGmToUb`**（`weight_quant_vec_compute.h`，第 515-519 行）已经正确处理了 pergroup 的 GM→UB 搬运：
   ```cpp
   DataCopyPad2D(ubAntiQuantScaleTotalBuffer_[...],
                 antiQuantScaleGlobal_[ubMte2KOffset / antiQuantGroupSize_ * offsetParam.nSize + ubMte2NOffset],
                 CeilDivide(ubMte2KSize, antiQuantGroupSize_), ubMte2NSize,
                 VEC_MAX_ELEM_B16, offsetParam.nSize);
   ```
   这里 GM 源地址为 `scale[kOffset/groupSize * N + nOffset]`，搬运形状为 `[K/G groups, N columns]`，目标步长为 `VEC_MAX_ELEM_B16`。**UB 布局为 `[G_tile, N_tile_padded]`**，与 wqbmmv2 NK 的 `[N_tile, G_tile_padded]` 不同。

2. **`CalLocalAddrForVf`**（`weight_quant_vec_compute.h`，第 1137-1176 行）中 scale 的 UB 地址计算：
   ```cpp
   uint64_t antiquantParamOffset =
       mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize + nWeightLowBitUbOffset;
   ```
   这里只用了 `nWeightLowBitUbOffset`，**没有考虑 K 方向的 group 偏移** — 这是 perchannel 逻辑，对 pergroup 不完整。

3. **`NdNkLoadScaleOffset`**（`basic_block_vf_nd.h`，第 327-344 行）中 scale 的 UB 加载：
   ```cpp
   MicroAPI::DataCopy<xType, DIST_BRC_B16>(antiQuantScaleVreg,
                                            antiQuantScaleBasePhyAddr + ubLoopNIdx);
   ```
   只按 N 方向偏移 `ubLoopNIdx`，**没有 group 维度的偏移** — 这是 perchannel 逻辑。

4. **`GMMWeightQuantBasicController::SetGmAddr`**（`grouped_matmul_weight_quant_basic_controller.h`，第 236 行）：
   ```cpp
   antiquantParamsBaseOffset += offsetParam.nSize;  // perchannel 逻辑
   ```
   对比 resplit controller（`grouped_matmul_weight_quant_resplit_controller.h`，第 260-262 行）的 pergroup 正确实现：
   ```cpp
   antiquantScaleGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
   ```

---

## §4. GMM A16W4 pergroup 实现方案

### 4.1 KN 路径（bTrans=false）— 直接复用

**结论：直接复用 wqbmmv2 的 `AntiquantPerGroupKN` VF 函数，所有 groupSize（32/64/128/256）均无需特殊处理。**

#### 4.1.1 为什么 KN 路径可以直接复用

KN 路径的核心数据结构：
- **Weight UB 布局**：`[K, N_align]`（K 为外轴，N 为内轴）
- **Scale UB 布局**：`[G, N_align]`（G 为外轴，N 为内轴）
- **GMM 与 wqbmmv2 的 scale GM 布局一致**：都是 `[G, N]`

`AntiquantPerGroupKN` 的三层循环结构：

```
for nIdx in [0, nLoop):           // N 方向，每次 128 元素
    for gIdx in [0, groupNum):     // G 方向，每次 1 个 group
        scale ← DIST_NORM(scaleBase + gIdx * nBubXTypeAlign + nIdx * vfElemB16)
        // scale 寄存器包含当前 group 在 N 方向的 128 个 scale 值
        for kIdx in [0, groupSize):  // K 方向，每次 1 个 K 位置
            weight ← DIST_UNPACK4_B8(weightBase + ...)
            weightOut ← Cast(weight)
            weightOut ← weightOut + offset  // if hasOffset
            weightOut ← weightOut × scale   // scale 复用 groupSize 次！
            DataCopy(weightOutBase, weightOut)
```

**关键**：scale 是按 `(group, N-tile)` 加载的，一个 group 的 scale 在 K 内循环中被**复用 groupSize 次**。groupSize 的大小只影响 K 内循环的迭代次数，不影响 scale 的加载方式。

因此，**gs=32 在 KN 路径下不需要任何特殊构造** — scale 加载方式与 gs=128/256 完全相同，只是 K 内循环迭代 32 次而非 128/256 次。

#### 4.1.2 双 AIV 切分对 KN 路径的影响

KN 路径 AIV 沿 K 切分（§5.3），每个 AIV 处理 K/2：
- kL1=256 → 每个 AIV 处理 K=128
- gs=32 时：每个 AIV 有 128/32 = 4 个 group
- gs=128 时：每个 AIV 有 128/128 = 1 个 group
- gs=256 时：每个 AIV 有 128/256 < 1 → 只有 1 个 group（尾块处理）

这些差异都通过 `param.groupNum` 和 `param.groupTail` 参数自动处理，不需要修改 VF 函数。

#### 4.1.3 需要修改的 GMM 代码

1. **`CalLocalAddrForVf`**（`weight_quant_vec_compute.h`，第 1141-1142 行）：
   ```cpp
   // 当前（perchannel）:
   uint64_t antiquantParamOffset =
       mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize + nWeightLowBitUbOffset;
   // 需要改为（pergroup KN）:
   uint64_t antiquantParamOffset =
       mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize +
       kWeightLowBitUbOffset / antiQuantGroupSize_ * nSize + nWeightLowBitUbOffset;
   ```

2. **`NdKnLoadScaleOffset`**（`basic_block_vf_nd.h`，第 346-368 行）：
   当前 perchannel 逻辑加载 2 个相邻 128 元素的 scale block，pergroup 需要按 group 维度加载。

### 4.2 NK 路径（bTrans=true）— 核心新增代码

#### 4.2.1 Scale/Offset GM→UB 搬运

GMM 的 `CopyAntiQuantParamsGmToUb`（第 515-519 行）已经正确实现了 pergroup 的 GM→UB 搬运：

```text
// GMM: scale GM 布局始终为 [G, N]
gBase = ubMte2KOffset / antiQuantGroupSize_
gTile = CeilDivide(ubMte2KSize, antiQuantGroupSize_)
nTileAlign = VEC_MAX_ELEM_B16  // DataCopyPad2D 的目标步长

GM src = antiquantScale[gBase * N + ubMte2NOffset]
UB dst layout = [gTile, nTileAlign]

DataCopyPad2D(scaleUb, scaleGm[src], gTile, ubMte2NSize, VEC_MAX_ELEM_B16, offsetParam.nSize)
```

**UB 布局为 `[G_tile, N_tile_padded]`**，与 wqbmmv2 NK 的 `[N_tile, G_tile_padded]` 不同。

#### 4.2.2 NK VF 函数适配

##### gs=128：`AntiquantW4Pergroup128NK_GMM`

```text
// scale 地址从 [N,G] 改为 [G,N]
// wqbmmv2: scaleBase + groupIdx + nBubIdx * outerStrideScale
// GMM:     scaleBase + groupIdx * nTileAlign + nBubIdx
scale ← DIST_BRC_B16(scaleBase + groupIdx * nTileAlign + nBubIdx)
```

其余逻辑（weight 加载、Cast、Mul、输出）不变。

**AddrReg 形式**：
```text
// wqbmmv2: CreateAddrReg<XType>(groupIdx, 1, nBubIdx, outerStrideScale)
// GMM:     CreateAddrReg<XType>(nBubIdx, 1, groupIdx, nTileAlign)
//   或:    CreateAddrReg<XType>(groupIdx, nTileAlign, nBubIdx, 1)
```

##### gs=256：`AntiquantW4Pergroup256NK_GMM`

```text
// 同 gs=128，scale 地址改为 [G,N] 布局
scale ← DIST_BRC_B16(scaleBase + groupIdx * nTileAlign + nBubIdx)
// 2 次 weight 加载不变（weight 布局不受 scale 布局影响）
```

##### gs=64：`AntiquantW4Pergroup64NK_GMM`

```text
// wqbmmv2: 从 [N,G] 加载相邻 2 个 group
//   oriScale00 ← scaleBaseAddr00 + innerIdx * 2 + outerIdx * outerStrideScale
//   oriScale01 ← scaleBaseAddr01 + innerIdx * 2 + outerIdx * outerStrideScale
//   其中 scaleBaseAddr01 = scaleBaseAddr00 + 1（相邻 group 在 G 维度连续）
//
// GMM: 从 [G,N] 加载相邻 2 个 group
//   oriScale00 ← scaleBase + (innerIdx * 2) * nTileAlign + outerIdx
//   oriScale01 ← scaleBase + (innerIdx * 2 + 1) * nTileAlign + outerIdx
//   相邻 group 在 G 维度相隔 nTileAlign

oriScale00 ← DIST_BRC_B16(scaleBase + (innerIdx * 2) * nTileAlign + outerIdx)
oriScale01 ← DIST_BRC_B16(scaleBase + (innerIdx * 2 + 1) * nTileAlign + outerIdx)
scale0 ← Select(oriScale00, oriScale01, maskH)
```

##### gs=32：`AntiquantW4Pergroup32NK_GMM`

**双 AIV 切分对 gs=32 的影响**：

根据 §5 的分析，NK 路径（bTrans=true）AIV 沿 N 方向切分，K 方向完整保留。因此每个 AIV 处理的 K 范围仍为 kL1=256，对应 256/32 = 8 个 group。

**NK 路径的 gs=32 实现**（每个 AIV 处理 8 个 group）：

GMM 的 scale/offset 始终为 `[G, N]` 布局。在 UB 中，8 个 group 的 scale 在同一 N 列上相隔 `nTileAlign`，不连续。因此不能使用 `DIST_E2B_B16`（要求源地址连续）。

**方案**：8 次 `DIST_BRC_B16` 广播 + `Interleave` + `Select` 拼接

```text
// 加载 8 个 group 的 scale（每个广播为 128 元素的全同值寄存器）
A ← DIST_BRC_B16(scaleBase + (innerIdx * 8 + 0) * nTileAlign + outerIdx)  // scale[0]
B ← DIST_BRC_B16(scaleBase + (innerIdx * 8 + 1) * nTileAlign + outerIdx)  // scale[1]
C ← DIST_BRC_B16(scaleBase + (innerIdx * 8 + 2) * nTileAlign + outerIdx)  // scale[2]
D ← DIST_BRC_B16(scaleBase + (innerIdx * 8 + 3) * nTileAlign + outerIdx)  // scale[3]
E ← DIST_BRC_B16(scaleBase + (innerIdx * 8 + 4) * nTileAlign + outerIdx)  // scale[4]
F ← DIST_BRC_B16(scaleBase + (innerIdx * 8 + 5) * nTileAlign + outerIdx)  // scale[5]
G ← DIST_BRC_B16(scaleBase + (innerIdx * 8 + 6) * nTileAlign + outerIdx)  // scale[6]
H ← DIST_BRC_B16(scaleBase + (innerIdx * 8 + 7) * nTileAlign + outerIdx)  // scale[7]

// 目标：构造 2 个 128 元素 scale 寄存器
//   scale0 = [A×32, B×32, C×32, D×32]  (groups 0-3)
//   scale1 = [E×32, F×32, G×32, H×32]  (groups 4-7)

// 步骤 1：Interleave 构造 64 元素对
//   Interleave(src0, src1) → [src0[0:64], src1[0:64]]
//   由于 A/B 是全同值寄存器，Interleave(A, B) = [A×64, B×64]
AB ← Interleave(A, B)    // [A×64, B×64]
CD ← Interleave(C, D)    // [C×64, D×64]
EF ← Interleave(E, F)    // [E×64, F×64]
GH ← Interleave(G, H)    // [G×64, H×64]

// 步骤 2：Select 拼接 128 元素寄存器
//   Select(src0, src1, maskH) → [src0[0:64], src1[0:64]]
scale0 ← Select(AB, CD, maskH)   // [A×32, B×32, C×32, D×32]
scale1 ← Select(EF, GH, maskH)   // [E×32, F×32, G×32, H×32]
```

**指令开销**：8 条 `DIST_BRC_B16` + 4 条 `Interleave` + 2 条 `Select` = 14 条指令（仅 scale 部分）

**offset 同理构造**：`hasAntiQuantOffset=true` 时，offset 也需要相同的 14 条指令构造。

> **注意**：以上 `Select` 和 `Interleave` 的具体 mask 构造需要在实现时根据 Ascend C MicroAPI 文档确认。`maskH` 预期为前 64 lane 选 src0、后 64 lane 选 src1 的模式。`Interleave` 预期为取两个寄存器前半部分拼接的模式。

#### 4.2.3 NK 路径地址公式汇总

| groupSize | wqbmmv2 scale 地址 | GMM scale 地址 | 加载方式 |
|-----------|-------------------|---------------|----------|
| 128 | `base + g + n × strideS` | `base + g × nTileAlign + n` | DIST_BRC_B16 |
| 256 | `base + g + n × strideS` | `base + g × nTileAlign + n` | DIST_BRC_B16 |
| 64 | `base + g + n × strideS` (2次, g 和 g+1) | `base + g × nTileAlign + n` (2次) | DIST_BRC_B16 + Select |
| 32 | `base + g×8 + n × strideS` (E2B, 8个连续) | `base + (g×8+i) × nTileAlign + n` (8次) | DIST_BRC_B16 × 8 + Interleave × 4 + Select × 2 |

> **KN 路径**（bTrans=false）所有 groupSize 统一使用 `DIST_NORM` 加载 scale，地址公式为 `base + gIdx × nBubXTypeAlign + nIdx × vfElemB16`，不需要按 groupSize 分派。详见 §4.1。

其中：
- `g` = groupIdx 或 innerIdx（取决于 groupSize）
- `n` = nBubIdx 或 outerIdx
- `strideS` = `CeilAlign(groupNumBub_, ONE_BLK_X_NUM)`（wqbmmv2 NK 的 scale UB 外轴步长）
- `nTileAlign` = `CeilAlign(nRealLen, BLOCK_CUBE)`（GMM 的 scale UB 外轴步长）

#### 4.2.4 gs=32 NK 路径性能影响分析

| 维度 | wqbmmv2 NK | GMM NK |
|------|-----------|--------|
| **Scale 加载指令数** | 1 条 `DIST_E2B_B16` + 1 条 `Interleave` | 8 条 `DIST_BRC_B16` + 4 条 `Interleave` + 2 条 `Select` |
| **指令总数（scale 部分）** | ~2 条 | ~14 条 |
| **性能影响** | 基准 | scale 加载开销增加约 7 倍 |

> **KN 路径无额外开销**：KN 路径所有 groupSize 统一使用 `DIST_NORM` 加载 scale（1 条指令），gs=32 与 gs=128/256 的 scale 加载开销完全相同。

**缓解方案**：

1. **初始实现优先正确性**：先用直接地址计算（`useVag=false`），验证精度
2. **AddrReg 优化**：后续可用 `CreateAddrReg` 减少地址计算指令
3. **实际影响有限**：gs=32 在实际模型中使用较少（MoE 场景通常 gs=128），且 scale 加载只是整个 VF 函数的一小部分
4. **KN 路径开销较低**：KN 路径每个 AIV 仅处理 4 个 group，指令开销约为 NK 路径的一半
5. **潜在优化**：可探索将 `[G, N]` 布局的 scale 在 UB 中转置为 `[N, G]` 布局，从而复用 wqbmmv2 的 VF 函数 — 但这会增加 GM→UB 搬运的复杂度

---

## §5. 双 AIV 并行切分逻辑

### 5.1 GMM 双 AIV 架构

Ascend 950 的 AI Core 包含 1 个 Cube 和 2 个 Vector Core（AIV0、AIV1）。GMM 的 basic_block 架构利用双 AIV 并行执行反量化，提高吞吐量。

### 5.2 wqbmmv2 双 AIV 切分策略

**文件**: `weight_quant_batch_matmul_v2_reg_base_common.h`，`CopyUb2L1` 函数（第 1904-1963 行）

wqbmmv2 根据 `antiQuantType`、`weightNz`、`bTrans` 三个编译期参数决定切分方向：

| 条件 | 切分方向 | AIV0 工作量 | AIV1 工作量 |
|------|----------|-------------|-------------|
| `PER_GROUP && !weightNz && bTrans` (NK) | **N** | 前 `nBubSize` 列 | 剩余 `nBL1Len_ - nBubSize` 列 |
| `PER_GROUP && !weightNz && !bTrans` (KN) | **K** | 前 `kBubSize` 行 | 剩余 `kBL1Len_ - kBubSize` 行 |
| `IS_4BIT_WEIGHT` (NZ weight) | **N 优先，K 兜底** | 前半 N（或 K）bubbles | 后半 N（或 K）bubbles |

**关键代码**（第 1910-1930 行）：
```cpp
// NK 路径（bTrans=true）：沿 N 切分
if (GetSubBlockIdx() == 1) {
    nBL1Offset_ += nBL1Len_ > nBubSize ? nBubSize : 0;
    vecNBL1Len_ = nBL1Len_ > nBubSize ? nBL1Len_ - nBubSize : nBL1Len_;
} else {
    vecNBL1Len_ = Min(nBubSize, nBL1Len_);
}

// KN 路径（bTrans=false）：沿 K 切分
twoVectorCoreSplitK_ = kBL1Len_ > kBubSize;
if (GetSubBlockIdx() == 1) {
    kBL1Offset_ += twoVectorCoreSplitK_ ? kBubSize : 0;
    vecKBL1Len_ = twoVectorCoreSplitK_ ? kBL1Len_ - kBubSize : kBL1Len_;
} else {
    vecKBL1Len_ = Min(kBubSize, kBL1Len_);
}
```

**单任务回退**：当 `nBL1Len_ <= nBubSize`（NK）或 `kBL1Len_ <= kBubSize`（KN）时，两个 AIV 执行相同的工作（冗余执行）。

### 5.3 GMM 双 AIV 切分策略

**文件**: `weight_quant_basic_block.h`，`ComputeBasicBlockAivNdKnNzNk` 和 `ComputeBasicBlockAivNdNkNzKn` 函数

GMM 的切分策略与 wqbmmv2 一致：

| 路径 | bTrans | 切分方向 | AIV0 工作量 | AIV1 工作量 |
|------|--------|----------|-------------|-------------|
| ND KN | false | **K** | 前 `kbL1Size / 2` 行 | 剩余 K 行 |
| ND NK | true | **N** | 前 `nL1Size / 2` 列 | 剩余 N 列 |

**ND KN 路径**（`ComputeBasicBlockAivNdKnNzNk`，第 268-318 行）：
```cpp
uint64_t kMte2BaseSize = offsetParam.kbL1Size / VEC_CORE_NUM;  // K 平分给 2 个 AIV
l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * kMte2BaseSize;

// AIV0: 前 K/2 行，AIV1: 后 K/2 行
uint64_t mte2RealK = GetSubBlockIdx() == 0 
    ? min(kMte2BaseSize, l1ConsumeConfig.l1RealExternalLen)
    : l1RealExternalLen > kMte2BaseSize ? l1RealExternalLen - kMte2BaseSize : 0;

// 两个 AIV 都处理完整的 N
ubConsumeConfig.l1RequireVfComputeRealN = offsetParam.nL1Size;
```

**ND NK 路径**（`ComputeBasicBlockAivNdNkNzKn`，第 181-226 行）：
```cpp
uint64_t curVecCoreMte2RealN = offsetParam.nL1Size / VEC_CORE_NUM;  // N 平分给 2 个 AIV
ubConsumeConfig.l1RequireVfComputeRealN = 
    GetSubBlockIdx() == 0 ? curVecCoreMte2RealN : nL1Size - curVecCoreMte2RealN;
l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * curVecCoreMte2RealN;

// 两个 AIV 都处理完整的 K（外层循环迭代 K）
```

### 5.4 VfConfig 与切分关系

**文件**: `basic_block_config.h`，`GetVfConfig` 函数（第 249-272 行）

```cpp
if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ && wqmmConfig.bTrans) {
    return {.vfNStandardLen = 64, .vfKStandardLen = 256};  // NK: 每次 VF 处理 N=64, K=256
} else if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ && !wqmmConfig.bTrans) {
    return {.vfNStandardLen = 256, .vfKStandardLen = 64};  // KN: 每次 VF 处理 N=256, K=64
}
```

**解读**:
- VfConfig 定义的是**单次 VF 迭代**的标准块大小，不是 per-AIV 总块大小
- per-AIV 总块大小由 kL1/nL1 和 VEC_CORE_NUM 决定
- 每个 AIV 可能需要多次 VF 迭代来完成其分配的块

### 5.5 对 pergroup 反量化的影响

| 路径 | AIV 切分方向 | per-AIV 块大小 | group num per AIV |
|------|-------------|---------------|------------------|
| KN (bTrans=false) | K | N=256, K=128 | 128 / groupSize |
| NK (bTrans=true) | N | N=128, K=256 | 256 / groupSize |

**关键影响**:
- KN 路径：每个 AIV 处理的 K 减半，group 数减半，但 scale 加载方式不变（`DIST_NORM` 按 group 加载，在 K 内循环中复用 groupSize 次）
- NK 路径：每个 AIV 处理的 N 减半，但 K 完整，group 数不变

**对 gs=32 的具体影响**:
- **KN 路径**：每个 AIV 处理 K=128，对应 128/32 = 4 个 group。但 scale 加载方式与 gs=128/256 完全相同（`DIST_NORM` 一次加载 128 个 N 方向的 scale），K 内循环迭代 32 次复用同一 scale。**无需特殊处理。**
- **NK 路径**：每个 AIV 处理 K=256（完整），对应 256/32 = 8 个 group。需要构造 2 个 128 元素 scale 寄存器（8 groups × 32 elements = 256 K elements）。**需要特殊构造**（详见 §4.2.2）。

---

## §6. 实现文件映射

| 文件 | 路径 | 修改内容 |
|------|------|----------|
| `basic_block_vf_nd.h` | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/` | 新增 KN/NK pergroup VF 调用入口；修改 `NdNkLoadScaleOffset` / `NdKnLoadScaleOffset` 增加 PER_GROUP 分支 |
| `weight_quant_vec_compute.h` | 同上 | 修改 `CalLocalAddrForVf` 增加 pergroup scale UB 地址计算；修改 `CopyAntiQuantParamsGmToUb` 补充 offset 的 pergroup 搬运；新增 `AntiQuantComputeKNPerGroup` / `AntiQuantComputeW4NKPerGroup` 参数构造和 VF 调用 |
| `basic_block_config.h` | 同上 | 新增 A16W4 ND pergroup `WqmmConfig` 常量（含 `hasAntiQuantOffset` 变体），如 `A16W4_NDKN_PG` 和 `A16W4_NDNK_PG` |
| `grouped_matmul_weight_quant_basic_controller.h` | 同上 | 修改 `SetGmAddr` 增加 pergroup 感知的 `antiquantScale/Offset` GM 地址更新（参照 resplit_controller.h 第 260-262 行） |

**参照文件**（不修改，仅参考）：

| 文件 | 路径 | 参考价值 |
|------|------|----------|
| `weight_quant_batch_matmul_v2_reg_base_common.h` | `mc2/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/` | CopyInScaleOffset (line 987)、AntiQuantCompute (line 1591)、参数构造 |
| `weight_quant_batch_matmul_v2_vf.h` | 同上 | 所有 VF 函数实现（KN: line 346, NK: line 445-761） |
| `grouped_matmul_weight_quant_resplit_controller.h` | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/` | pergroup GM 地址更新参考（line 260-262） |

---

## §7. 风险与缓解

| 风险 | 影响 | 概率 | 缓解 |
|------|------|------|------|
| **gs=32 NK 性能** | scale 加载指令数增加约 7 倍（从 2 条到 ~14 条：8 BRC + 4 Interleave + 2 Select） | 高 | 初始实现优先正确性，后续优化；gs=32 在实际模型中使用较少 |
| **UB 空间不足** | pergroup scale/offset 需要额外 UB（`[G_tile, N_tile]` 比 perchannel `[N_tile]` 大 G 倍） | 中 | 复用 A16W8 ND 的 UB 分配（4KB scale + 4KB offset）；验证 `antiQuantScaleUbSingleBufferSize` 是否足够 |
| **NZ 输出格式** | KN 和 NK 的 NZ 内轴不同（KN: K 为内轴, NK: N 为内轴） | 低 | 复用 wqbmmv2 的 `DATA_BLOCK_COPY` 参数和 `dataBlockStride` 计算 |
| **`CalLocalAddrForVf` 修改影响范围** | 该函数被 perchannel 和 pergroup 共用，修改可能影响已有功能 | 中 | 使用 `if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP)` 条件编译隔离 |
| **`SetGmAddr` pergroup 适配** | 当前 basic controller 的 GM 地址更新是 perchannel 逻辑 | 高 | 参照 resplit controller 的 pergroup 实现（line 260-262），使用 `if constexpr` 分支 |
| **gs=32 NK `DIST_E2B_B16` 不可用** | GMM `[G,N]` 布局下 group 不连续，无法使用 E2B 批量加载 | 确定 | 改用 `DIST_BRC_B16` + `Interleave` + `Select` 拼接，如 §4.2.2 所述 |
| **双 AIV 切分对 NK gs=32 的影响** | NK 路径每个 AIV 处理 8 个 group（K=256 完整），需要构造 2 个 128 元素 scale 寄存器 | 确定 | 已在 §4.2.2 给出具体构造方案 |

---

## 附录 A：关键常量定义

| 常量 | 值 | 含义 |
|------|-----|------|
| `VEC_MAX_ELEM_B16` | 128 | FP16 向量寄存器最大元素数（256B / 2B） |
| `VECTOR_REG_WIDTH` | 256 | 向量寄存器宽度（字节） |
| `BLOCK_CUBE` | 16 | Cube 块大小 |
| `ONE_BLK_X_NUM` | 16 | FP16 一个 block 的元素数 |
| `ONE_BLK_W_NUM` | 32 | INT4 一个 block 的元素数（按 int8_t 计数） |
| `INT4_DTYPE_PARAM` | 1 | INT4 地址右移位数（每字节存 2 个 INT4） |
| `WEIGHT_F16_UB_NZ_STRIDE` | 65 | NZ 格式 UB 输出跳行步长（避免 bank 冲突） |

## 附录 B：源码文件索引

| 文件 | 完整路径 |
|------|----------|
| wqbmmv2 reg_base_common | `mc2/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h` |
| wqbmmv2 vf | `mc2/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_vf.h` |
| wqbmmv2 reg_base | `mc2/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base.h` |
| GMM basic_block_vf_nd | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h` |
| GMM vec_compute | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h` |
| GMM basic_block_config | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h` |
| GMM basic_controller | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h` |
| GMM resplit_controller | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h` |
| GMM tiling_key | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h` |
| GMM host tiling | `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp` |
