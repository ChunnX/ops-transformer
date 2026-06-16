# 04 Kernel 实现方案 — GMM A16W4 PerGroup ND

> 本文档描述 A16W4 INT4 weight ND pergroup 数据流的 kernel 方案。整体 controller、basic block、L1/UB 分配、cube 计算流程按 A16W8 ND basic-controller 方案复用；新增实现集中在 INT4 pergroup VF 反量化和 pergroup 地址偏移。
> 相关文档：[01 交付件清单](./01_deliverables.md)、[02 与硬件无关方案](./02_platform_independent_design.md)、[03 Tiling 方案](./03_tiling_design.md)

---

## 1. Kernel 入口与模板分发

入口位于 `gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp` 的 `V310_GMM_ANTI_QUANT` 分支，A16W4 pergroup ND 新增 8 个 basic-controller 分支：

```text
2 个 transB 状态 × 2 个 antiquantOffset 状态 × 2 个 bias 状态 = 8
```

分支共同特征：

| 模板参数 | 取值 |
|----------|------|
| `W_TYPE` | `WQGMM_ND` |
| `C_QUANT_TYPE` | `WQGMM_NONE` |
| `W_QUANT_TYPE` | `WQGMM_PER_GROUP` |
| `WQ_A_TRANS` | `WQGMM_NO_TRANS` |
| `WQ_B_TRANS` | `WQGMM_NO_TRANS` / `WQGMM_TRANS` |
| `TEMPLATE_CUSTOM_SC` | `WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4` |
| `ALGORITHM_SUB_CATEGORY` | `WQGMM_N_FIRST_BASIC_BLOCK` |
| `ALGORITHM_CATEGORY` | `WQGMM_VECTOR_ANTIQUANT` |

调用形式复用：

```cpp
INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(
    GMMWeightQuantBasicController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
```

`VEC_ANTIQUANT_CONFIG_3` 保持现有值：

```cpp
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
// ubMte2BufferNum = 4
// ubMte2InnerSize = 256
```

---

## 2. Controller 层适配

文件：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h`

### 2.1 修改点总览

| 函数 | 设计结论 |
|------|----------|
| `Init()` | 复用，`groupSize` 已通过 `wqmmBasicBlock_.Init(gmmBaseTiling_->groupSize, ...)` 传入 |
| `Process()` | 复用 A16W8 M/N 二维分块和 round-robin 分核 |
| `SetMKN()` | 复用，M/K/N 获取逻辑与 pergroup 无关 |
| `SetGmAddr()` | 需要区分 pergroup scale/offset 偏移，并将 bias 偏移独立出来 |
| `GetSplitValueFromGroupList()` | 复用 |

### 2.2 SetGmAddr 偏移

现有 A16W8 perchannel：

```cpp
antiquantParamsBaseOffset += offsetParam.nSize;
```

A16W4 pergroup：

```cpp
uint64_t groupNumK = offsetParam.kSize / gmmBaseTiling_->groupSize;
antiquantParamsBaseOffset += groupNumK * offsetParam.nSize;
```

原因：

```text
perchannel scale: [N]       每 group 占 N 个元素
pergroup scale:   [G, N]    每 group 占 (K/groupSize) * N 个元素
```

### 2.3 Bias 偏移独立

单单单场景下，bias shape 为 `[E,N]`，scale/offset shape 为 `[E,G,N]`。因此 bias 不能继续复用 `antiquantParamsBaseOffset`。

设计：

```cpp
uint64_t biasBaseOffset = 0;

// singleWeight == 1
biasGm = GetTensorAddr<biasType>(0, biasGm_) + biasBaseOffset;

// 每个 group 结束后
biasBaseOffset += offsetParam.nSize;
```

完整偏移更新：

```cpp
xBaseOffset += offsetParam.mSize * offsetParam.kSize;

if constexpr (IsSameType<wType, int4b_t>::value) {
    weightBaseOffset += (offsetParam.nSize * offsetParam.kSize) >> 1;
} else {
    weightBaseOffset += offsetParam.nSize * offsetParam.kSize;
}

if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    antiquantParamsBaseOffset +=
        (offsetParam.kSize / gmmBaseTiling_->groupSize) * offsetParam.nSize;
} else {
    antiquantParamsBaseOffset += offsetParam.nSize;
}

biasBaseOffset += offsetParam.nSize;
yBaseOffset += offsetParam.mSize * offsetParam.nSize;
```

`weightBaseOffset >> 1` 的 INT4 packed 字节偏移与 perchannel/pergroup 无关，保留现有逻辑。

---

## 3. WqmmConfig 与 BasicBlock 分发

文件：
- `basic_block_config.h`
- `weight_quant_basic_block.h`

### 3.1 WqmmConfig

A16W4 pergroup ND 需要覆盖转置/非转置和有无 offset。可以新增命名常量，也可以在 `grouped_matmul_apt.cpp` 分支内局部定义 `static constexpr WqmmConfig`。

```cpp
// B 非转置
static constexpr WqmmConfig A16W4_ND_PG = {
    false, false, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::ND};

static constexpr WqmmConfig A16W4_ND_PG_OFF = {
    false, false, QuantType::PER_GROUP, true, QuantType::NONE, CubeFormat::ND};

// B 转置
static constexpr WqmmConfig A16W4_ND_TRANS_PG = {
    false, true, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::ND};

static constexpr WqmmConfig A16W4_ND_TRANS_PG_OFF = {
    false, true, QuantType::PER_GROUP, true, QuantType::NONE, CubeFormat::ND};
```

### 3.2 BasicBlock 分发

`ComputeBasicBlock()` 的分发逻辑不需要改变：

```cpp
if ASCEND_IS_AIV {
    if constexpr ((wqmmConfig.weightFormat != CubeFormat::NZ && !wqmmConfig.bTrans) ||
                  (wqmmConfig.weightFormat == CubeFormat::NZ && wqmmConfig.bTrans)) {
        ComputeBasicBlockAivNdKnNzNk(offsetParam);
    } else {
        ComputeBasicBlockAivNdNkNzKn(offsetParam);
    }
} else {
    ComputeBasicBlockAic(offsetParam);
}
```

对应关系：

| `bTrans` | AIV 路径 | A16W4 VF 参考 |
|----------|----------|---------------|
| false | `ComputeBasicBlockAivNdKnNzNk` | ops-nn `AntiquantComputeKNPerGroup` |
| true | `ComputeBasicBlockAivNdNkNzKn` | ops-nn `AntiquantComputeW4NKPerGroup` |

---

## 4. L1/UB 空间划分

### 4.1 L1 分配

A16W4 pergroup L1 分配完全复用 A16W8 ND basic block。L1 中存放的是反量化后的 `xType` weight，因此 A16W8 与 A16W4 在 L1 上的数据类型和大小一致。

```cpp
weightL1Space = matmulTiling->baseN * matmulTiling->stepKb * matmulTiling->baseK;
// 256 * 4 * 64 = 65536 xType elements = 128KB when xType is half/bfloat16
```

无 bias：

```text
L1 512KB:
| WeightL1_P0 128KB | AL1_P0 128KB | AL1_P1 128KB | WeightL1_P1 128KB |
```

有 bias：

```text
L1 512KB:
| WeightL1_P0 128KB | Bias_P0 4KB | AL1_P0 124KB |
| AL1_P1 124KB | Bias_P1 4KB | WeightL1_P1 128KB |
```

有 bias 时 host tiling 将 `baseM` 从 256 调整为 240：

```text
A tile = ceilAlign(240,16) * 256 * 2B = 120KB
AL1 single buffer = 124KB
```

因此 A tile 能放入 AL1，L1 分配闭合，无需新增 bias 专用切分策略。

### 4.2 UB 分配

A16W4 pergroup 复用 A16W8 ND 的 `GetNdBufferInfo()`，不新增专用 pergroup buffer 配置。

```cpp
// 当前 A16W8 ND perchannel 配置
{
    .ubWeightOutputHighBitBufferNum = 2,       // 双缓冲
    .weightInputLowbitUbTotalSize = 174KB,     // weight 输入（int4 原始数据）
    .highBitDataUbTotalSize = 66KB,            // 反量化后 fp16 weight
    .antiQuantScaleUbTotalSize = 4KB,          // ★ perchannel scale（仅 1 行）
    .antiQuantScaleAfterCastUbTotalSize = 0,
    .antiQuantOffsetUbTotalSize = 4KB,         // perchannel offset（A16W4 pergroup 不使用，但保留空间分配）
    .weightInputLowBitUbSingleBufferSize = 174KB / ubMte2BufferNum,
    .antiQuantScaleUbSingleBufferSize = 4KB / ubMte2BufferNum,
    .antiQuantOffsetUbSingleBufferSize = 4KB / ubMte2BufferNum,
    .highBitDataUbSingleBufferSize = 66KB / 2,
};
```

具体大小：

| Buffer | 总大小 | 缓冲数 | 单缓冲 | 用途 |
|--------|:------:|:------:|:------:|------|
| `weightInputLowBit` | 174KB | 4 | 43.5KB | INT4 packed weight 输入 |
| `highBitData` | 66KB | 2 | 33KB | VF 输出的 FP16/BF16 weight |
| `antiQuantScale` | 4KB | 4 | 1KB | 当前 VF tile 的 scale |
| `antiQuantOffset` | 4KB | 4 | 1KB | 保留空间，A16W4 pergroup 不使用 |

**UB 空间分配逻辑不变**：复用 `GetNdBufferInfo()`，`antiQuantOffset` 的 4KB 保留。A16W4 pergroup 运行时不使用 offset buffer，但不修改分配代码。

### 4.3 UB 容量与 VF tile

`GetVfConfig()` 对 ND 场景的 VF 标准块为：

| 场景 | `vfNStandardLen` | `vfKStandardLen` |
|------|:----------------:|:----------------:|
| `bTrans=false` | 256 | 64 |
| `bTrans=true` | 64 | 256 |

最小 `groupSize=32` 时，单次 VF tile 的 scale 最大需求：

```text
bTrans=false: ceil(64 / 32) * 256 * 2B = 1024B = 1KB
bTrans=true:  ceil(256 / 32) * 64 * 2B = 1024B = 1KB
```

因此 scale 单缓冲 1KB 足够。A16W4 pergroup 的 scale 搬运和地址组织必须按当前 VF tile 对齐，而不是按完整 `kbL1Size × nL1Size` 一次性展开到单个 scale buffer。

weight 和 high-bit 输出容量：

```text
单次 VF high-bit 输出最大 = 256 * 64 * 2B = 32KB
highBitData 单缓冲 = 33KB

INT4 packed 输入最大低于 weightInputLowBit 单缓冲 43.5KB
```

---

## 5. Vec 反量化实现

文件：
- `basic_block_vf_nd.h`
- `weight_quant_vec_compute.h`
- 参考：`~/Code/ops-nn/matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h`
- 参考：`~/Code/ops-nn/matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_vf.h`

### 5.1 迁移边界

A16W4 pergroup 只在 VF 将 INT4 weight 反量化为 `xType` 时参考 ops-nn wqbmmv2。其余流程复用 GMM A16W8：
- GM 到 UB 的 weight 搬运框架复用。
- UB buffer 分配复用。
- AIV/AIC 同步复用。
- UB 到 L1 搬运复用。
- Cube matmul 复用。

### 5.2 Scale/Offset GM→UB 搬运

#### 5.2.1 现有 pergroup 搬运（scale）

`CopyAntiQuantParamsGmToUb`（`weight_quant_vec_compute.h`，第 515-519 行）已实现 PER_GROUP 的 scale GM→UB 搬运：

```cpp
} else if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    DataCopyPad2D(ubAntiQuantScaleTotalBuffer_[...],
                  antiQuantScaleGlobal_[ubMte2KOffset / antiQuantGroupSize_ * offsetParam.nSize + ubMte2NOffset],
                  CeilDivide(ubMte2KSize, antiQuantGroupSize_), ubMte2NSize,
                  VEC_MAX_ELEM_B16, offsetParam.nSize);
}
```

GM 源地址和搬运形状：

```text
gBase = ubMte2KOffset / antiQuantGroupSize_
gTile = CeilDivide(ubMte2KSize, antiQuantGroupSize_)
nTileAlign = VEC_MAX_ELEM_B16  // DataCopyPad2D 的目标步长

GM src = antiquantScale[gBase * N + ubMte2NOffset]
UB dst layout = [gTile, nTileAlign]

DataCopyPad2D(scaleUb, scaleGm[src], gTile, ubMte2NSize, VEC_MAX_ELEM_B16, offsetParam.nSize)
```

**UB 布局为 `[G_tile, N_tile_padded]`**，与 wqbmmv2 NK 的 `[N_tile, G_tile_padded]` 不同。这是 GMM 与 wqbmmv2 在 NK 路径上的核心差异来源。

#### 5.2.2 需要补充的 offset 搬运

当前 PER_GROUP 分支**只搬运了 scale，未搬运 offset**。需要新增：

```cpp
} else if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    DataCopyPad2D(ubAntiQuantScaleTotalBuffer_[...],
                  antiQuantScaleGlobal_[ubMte2KOffset / antiQuantGroupSize_ * offsetParam.nSize + ubMte2NOffset],
                  CeilDivide(ubMte2KSize, antiQuantGroupSize_), ubMte2NSize,
                  VEC_MAX_ELEM_B16, offsetParam.nSize);
    // ★ 新增：PER_GROUP offset 搬运
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        DataCopyPad2D(ubAntiQuantOffsetTotalBuffer_[...],
                      antiQuantOffsetGlobal_[ubMte2KOffset / antiQuantGroupSize_ * offsetParam.nSize + ubMte2NOffset],
                      CeilDivide(ubMte2KSize, antiQuantGroupSize_), ubMte2NSize,
                      VEC_MAX_ELEM_B16, offsetParam.nSize);
    }
}
```

#### 5.2.3 搬运粒度：VF tile vs MTE2 tile

scale/offset 以 **VF tile 粒度**组织到 UB，而不是按完整 MTE2 weight tile 一次性展开。

原因：`antiQuantScale` 单缓冲仅 1KB。在 `groupSize=32` 时，一个 MTE2 tile（K=256, N=256）的 scale 需要 `(256/32) * 256 * 2B = 4KB`，超过单缓冲容量。而 VF tile（最大 256×64 或 64×256）的 scale 最大仅需 1KB。

因此，该 helper 在 `WeightAntiQuantCompute()` 的每个 VF tile 计算前调用（即 `WeightAntiQuantProcess` 内部），而不是在 `CopyGmToUb()` 中按完整 MTE2 weight tile 调用。

#### 5.2.4 CalLocalAddrForVf 适配

当前 `CalLocalAddrForVf`（`weight_quant_vec_compute.h`，第 1137-1176 行）中 scale 的 UB 地址计算使用 perchannel 逻辑：

```cpp
// 当前（perchannel）:
uint64_t antiquantParamOffset =
    mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize + nWeightLowBitUbOffset;
```

PER_GROUP 需要增加 K 方向的 group 偏移：

```cpp
// 新增（pergroup）:
if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    uint64_t antiquantParamOffset =
        mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize +
        kWeightLowBitUbOffset / antiQuantGroupSize_ * nSize + nWeightLowBitUbOffset;
} else {
    // perchannel 原逻辑
    uint64_t antiquantParamOffset =
        mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize + nWeightLowBitUbOffset;
}
```

使用 `if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP)` 条件编译隔离，不影响已有 perchannel 路径。

### 5.3 非转置：KN 路径（`bTrans=false`）

#### 5.3.1 调用链

```text
ComputeBasicBlockAivNdKnNzNk
  -> CopyGmToUb
  -> WeightAntiQuantCompute
    -> WeightAntiQuantProcess
      -> CopyAntiQuantParamsGmToUb (scale/offset GM→UB, VF tile 粒度)
      -> AntiQuantProcessNd
        -> AntiQuantInt4NdKn 或新增 AntiQuantInt4PerGroupNdKn
          -> VF_CALL<AntiquantPerGroupKN<xType, wType, hasAntiQuantOffset>>
```

#### 5.3.2 可直接复用 wqbmmv2 的 AntiquantPerGroupKN

**结论：直接复用 wqbmmv2 的 `AntiquantPerGroupKN` VF 函数，所有 groupSize（32/64/128/256）均无需特殊处理。**

原因：KN 路径的核心数据结构在 GMM 和 wqbmmv2 中一致：
- Weight UB 布局：`[K, N_align]`（K 为外轴，N 为内轴）
- Scale UB 布局：`[G, N_align]`（G 为外轴，N 为内轴）
- GMM 与 wqbmmv2 的 scale GM 布局一致：都是 `[G, N]`

#### 5.3.3 参数构造

参考 `AntiquantComputeKNPerGroup`（`weight_quant_batch_matmul_v2_reg_base_common.h`，第 1246-1287 行）：

```cpp
ParamsKN<xType> param;
param.kBub = bubKLen;
param.nBub = bubNLen;
param.nBubXTypeAlign = CeilAlign(nBub, BLOCK_CUBE);       // BLOCK_CUBE = 16
param.nBubWTypeAlign = CeilAlign(nBub, ONE_BLK_W_NUM);    // ONE_BLK_W_NUM = 32 (INT4)
param.vfElemB16 = VEC_MAX_ELEM_B16;                        // 128
param.nLoop = CeilDiv(nBubWTypeAlign, VEC_MAX_ELEM_B16);
param.groupSize = tiling->groupSize;
param.groupNum = CeilDiv(kBub, groupSize) - 1;              // 减 1 因为 tail 单独处理
param.groupTail = kBub % groupSize > 0 ? kBub % groupSize : groupSize;
param.dataBlockStride = CeilAlign(kBub, BLOCK_CUBE) + 1;   // NZ 输出跳行避免 bank 冲突
// W4 参数:
param.wNStride = VEC_MAX_ELEM_B16 >> INT4_DTYPE_PARAM;     // 128 >> 1 = 64
param.wKStride = nBubWTypeAlign >> INT4_DTYPE_PARAM;
param.wGroupStride = groupSize * nBubWTypeAlign >> INT4_DTYPE_PARAM;
// UB 地址映射到 GMM buffer:
param.scaleBaseAddr = ubAntiQuantScaleTotalBuffer_ 当前 VF tile 偏移;
param.offsetBaseAddr = ubAntiQuantOffsetTotalBuffer_ 当前 VF tile 偏移;
param.weightInBaseAddr = ubWeightInputLowBitTotalBuffer_ 当前偏移;
param.weightOutBaseAddr = ubHighBitTotalBuffer_ 当前偏移;
```

#### 5.3.4 VF 函数内部循环结构

`AntiquantPerGroupKN` 的三层循环：

```text
for nIdx = 0 to nLoop-1:                          // N 方向外层循环
  maskWeight = UpdateMask(nBubXTypeAlign)          // 更新 mask（尾块可能不满）

  for gIdx = 0 to groupNum-1:                      // 完整 group 遍历
    // 1. 加载 scale（DIST_NORM：直接加载 128 个 FP16 元素）
    addrRegScale = CreateAddrReg(nIdx, 128, gIdx, nBubXTypeAlign)
    scale ← DataCopy<DIST_NORM>(scaleBaseAddr, addrRegScale)

    // 若 hasAntiQuantOffset: 加载 offset
    if hasAntiQuantOffset:
      offset ← DataCopy<DIST_NORM>(offsetBaseAddr, addrRegScale)

    for kIdx = 0 to groupSize-1:                   // group 内逐 K 行处理
      // 2. 加载 INT4 weight（DIST_UNPACK4_B8：从 UB 加载并 unpack）
      addrRegWeight = CreateAddrReg(nIdx, wNStride, gIdx, wGroupStride, kIdx, wKStride)
      weightB4 ← DataCopy<DIST_UNPACK4_B8>(weightInBaseAddr, addrRegWeight)

      // 3. Cast INT4 → FP16/BF16
      weightB16 ← Cast<xType, int4x2_t>(weightB4, maskWeight)

      // 4. 若 hasAntiQuantOffset: 加 offset
      if hasAntiQuantOffset:
        weightB16 ← Add(weightB16, offset, maskWeight)

      // 5. 乘以 scale
      weightB16 ← Mul(weightB16, scale, maskWeight)

      // 6. 写入 UB highBitData（NZ 格式，DATA_BLOCK_COPY + POST_MODE_UPDATE）
      DataCopy<DATA_BLOCK_COPY, POST_MODE_UPDATE>(weightOutBaseAddr, weightB16,
                                                   dataBlockStride, repeatStride, maskWeight)

  // 处理 groupTail（最后一个不完整 group）
  scaleTail ← DataCopy<DIST_NORM>(scaleTailAddr, ...)
  for kIdx = 0 to groupTail-1:
    // 同上：load → unpack → cast → mul → store

  weightOutBaseAddr += weightOutStride              // N 方向步进到下一个 nLoop
```

**关键**：scale 按 `(group, N-tile)` 加载，一个 group 的 scale 在 K 内循环中被**复用 groupSize 次**。groupSize 的大小只影响 K 内循环的迭代次数，不影响 scale 的加载方式。因此，gs=32 在 KN 路径下不需要任何特殊构造。

### 5.4 转置：NK 路径（`bTrans=true`）

#### 5.4.1 调用链

```text
ComputeBasicBlockAivNdNkNzKn
  -> CopyGmToUb
  -> WeightAntiQuantCompute
    -> WeightAntiQuantProcess
      -> CopyAntiQuantParamsGmToUb (scale/offset GM→UB, VF tile 粒度)
      -> AntiQuantProcessNd
        -> AntiQuantInt4NdNk 或新增 AntiQuantInt4PerGroupNdNk
          -> 按 groupSize 分派专用 VF
```

#### 5.4.2 分派逻辑

参考 `AntiquantComputeW4NKPerGroup`（`weight_quant_batch_matmul_v2_reg_base_common.h`，第 1294-1410 行），按 `groupSize` 分发：

```cpp
if (groupSize == 32) {
    VF_CALL<AntiquantW4Pergroup32NK<xType, wType, hasAntiQuantOffset, useVag>>(params);
} else if (groupSize == 64) {
    VF_CALL<AntiquantW4Pergroup64NK<xType, wType, hasAntiQuantOffset, useVag>>(params);
} else if (groupSize == 128) {
    VF_CALL<AntiquantW4Pergroup128NK<xType, wType, hasAntiQuantOffset, useVag>>(params);
} else if (groupSize == 256) {
    VF_CALL<AntiquantW4Pergroup256NK<xType, wType, hasAntiQuantOffset, useVag>>(params);
}
```

本数据流只支持 32/64/128/256，不需要迁移 ops-nn 中 odd 或 greater-than-128 的泛化分支。

`useVag` 分派策略：当循环维度 > 1 时使用 `useVag=true`（AddrReg 硬件地址寄存器），= 1 时使用 `useVag=false`（直接指针运算），避免编译器限制。建议初始实现设为 `false`。

#### 5.4.3 Scale/Offset 地址公式适配（核心差异）

**不能原样迁移 ops-nn scale/offset 索引。**

| 维度 | wqbmmv2 NK | GMM NK |
|------|-----------|--------|
| **GM scale 布局** | `[N, G]` | `[G, N]`（始终，不随 bTrans 变化） |
| **UB scale 布局** | `[N_tile, G_tile_align]` | `[G_tile, N_tile_align]` |
| **scale 地址公式** | `base + groupIdx + nBubIdx × outerStrideScale` | `base + groupIdx × nTileAlign + nBubIdx` |
| **outerStrideScale** | `CeilAlign(groupNumBub_, ONE_BLK_X_NUM)` | — |
| **nTileAlign** | — | `CeilAlign(nRealLen, BLOCK_CUBE)` |

GMM 的 scale/offset 固定为 `[G, N]` 输入布局，直接按 VF tile 从 GM 搬入 UB 后自然布局为 `[G_tile, N_tile_aligned]`。因此 NK 路径只能复用 ops-nn 的 weight unpack、cast、输出 NZ 组织和 groupSize 分派结构；scale/offset 加载必须改为 GMM 专用地址公式。

#### 5.4.4 gs=128：AntiquantW4Pergroup128NK

最简单的 NK 实现，每个 group 恰好 128 个 K 元素 = 1 个向量寄存器宽度。

**参数**（`ParamsGroupSize128And256`）：

| 参数 | 计算方式 | 说明 |
|------|----------|------|
| `groupNum` | `CeilDiv(bubKLen, 128)` | group 数量 |
| `bubNLen` | `bubNLen` | N 方向实际长度 |
| `innerStrideWeight` | `64` | K 方向 weight 步长（128 INT4 = 64 字节） |
| `outerStrideWeight` | `CeilAlign(bubKLen, 32) >> 1` | N 方向 weight 步长（字节） |
| `nTileAlign` | `CeilAlign(bubNLen, BLOCK_CUBE)` | **GMM `[G_tile,N_tile_aligned]` 的 N 方向步长** |
| `dataBlockStride` | `CeilAlign(bubNLen, 128) + 1` | 输出 NZ block 间距 |
| `maskWeight` | `bubKLen` | weight mask（尾块处理） |

**循环结构**（useVag=false 路径）：

```text
for groupIdx = 0 to groupNum-1:                    // 按 group 遍历 K 方向
  for nBubIdx = 0 to bubNLen-1:                    // N 方向逐行

    // 1. GMM 专用 scale/offset 地址：[G_tile, N_tile_aligned]
    //    ★ 差异点：wqbmmv2 用 base + groupIdx + nBubIdx * outerStrideScale
    //              GMM 改为 base + groupIdx * nTileAlign + nBubIdx
    scale ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + groupIdx * nTileAlign + nBubIdx)
    if hasAntiQuantOffset:
      offset ← DataCopy<DIST_BRC_B16>(offsetBaseAddr + groupIdx * nTileAlign + nBubIdx)

    // 2. 加载 INT4 weight（DIST_UNPACK4_B8）— 与 wqbmmv2 一致
    weightB4 ← DataCopy<DIST_UNPACK4_B8>(weightInBaseAddr +
                                          groupIdx * innerStrideWeight +
                                          nBubIdx * outerStrideWeight)

    // 3. Cast INT4 → FP16/BF16
    weightB16 ← Cast<xType, int4x2_t>(weightB16, weightB4, maskAll)

    // 4. 若 hasAntiQuantOffset: 加 offset
    if hasAntiQuantOffset:
      weightB16 ← Add(weightB16, offset, maskAll)

    // 5. 乘以 scale
    weightB16 ← Mul(weightB16, weightB16, scale, maskAll)

    // 6. 写入 UB highBitData（NZ 格式）
    DataCopy<DATA_BLOCK_COPY, POST_MODE_UPDATE>(weightOutBaseAddr, weightB16,
                                                 dataBlockStride, repeatStride, maskWeight)

  weightOutBaseAddr += weightOutAddrOffset          // 下一个 group 的输出偏移
  maskWeight = UpdateMask(maskWeight)               // 重置 mask
```

**AddrReg 形式**：

```text
// wqbmmv2: CreateAddrReg<XType>(groupIdx, 1, nBubIdx, outerStrideScale)
// GMM:     CreateAddrReg<XType>(nBubIdx, 1, groupIdx, nTileAlign)
//   或:    CreateAddrReg<XType>(groupIdx, nTileAlign, nBubIdx, 1)
```

#### 5.4.5 gs=256：AntiquantW4Pergroup256NK

每个 group 有 256 个 K 元素 = 2 个向量寄存器宽度，需要加载 2 次 weight。

**与 gs=128 的差异**：

| 差异点 | gs=128 | gs=256 |
|--------|--------|--------|
| K 元素/group | 128 | 256 |
| weight 加载次数 | 1 次 | 2 次（weightB4 + weight1B4） |
| `innerStrideWeight` | 64 | 128（2×64） |
| 输出寄存器 | 1 个 | 2 个（weightB16 + weight1B16） |
| `weightOutBaseAddr1` | 无 | `weightOutBaseAddr + dataBlockStride × 128` |

**循环结构**（核心差异部分）：

```text
for groupIdx = 0 to groupNum-1:
  for nBubIdx = 0 to bubNLen-1:
    // ★ GMM scale 地址：[G, N] 布局
    scale ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + groupIdx * nTileAlign + nBubIdx)
    if hasAntiQuantOffset:
      offset ← DataCopy<DIST_BRC_B16>(offsetBaseAddr + groupIdx * nTileAlign + nBubIdx)

    // 加载 2 个 128 元素的 INT4 weight — 与 wqbmmv2 一致
    weightB4  ← DataCopy<DIST_UNPACK4_B8>(weightInBaseAddr + ...)
    weight1B4 ← DataCopy<DIST_UNPACK4_B8>(weightInBaseAddr1 + ...)   // 偏移 64 字节

    weightB16  ← Cast(weightB4)
    weight1B16 ← Cast(weight1B4)

    if hasAntiQuantOffset:
      weightB16  ← Add(weightB16, offset)
      weight1B16 ← Add(weight1B16, offset)

    weightB16  ← Mul(weightB16, scale)
    weight1B16 ← Mul(weight1B16, scale)

    // 分别写入两个输出位置
    DataCopy<DATA_BLOCK_COPY>(weightOutBaseAddr, weightB16, ...)
    DataCopy<DATA_BLOCK_COPY>(weightOutBaseAddr1, weight1B16, ...)
```

#### 5.4.6 gs=64：AntiquantW4Pergroup64NK

每个 group 有 64 个 K 元素 = 半个向量寄存器宽度，需要 `Select` 拼接 scale。

**参数**（`ParamsGroupSize64`）：

| 参数 | 计算方式 | 说明 |
|------|----------|------|
| `innerExtend` | `CeilDiv(bubKLen, 128)` | K 方向按 128 元素分块 |
| `outerExtend` | `bubNLen` | N 方向逐行 |
| `nTileAlign` | `CeilAlign(bubNLen, BLOCK_CUBE)` | **GMM `[G_tile,N_tile_aligned]` 的 N 方向步长** |
| `outerStrideWeight` | `CeilAlign(bubKLen, 32) >> 1` | weight N 方向步长（字节） |

**循环结构**（useVag=false 路径）：

```text
for innerIdx = 0 to innerExtend-1:                 // K 方向按 128 元素分块
  for outerIdx = 0 to outerExtend-1:               // N 方向逐行

    // 1. 从 [G_tile, N_tile_aligned] 加载 2 个相邻 group 的 scale，用 Select 拼接
    //    ★ 差异点：wqbmmv2 从 [N,G] 加载，相邻 group 在 G 维度连续（+1）
    //              GMM 从 [G,N] 加载，相邻 group 在 G 维度相隔 nTileAlign
    //
    //    wqbmmv2: oriScale00 ← scaleBaseAddr00 + innerIdx * 2 + outerIdx * outerStrideScale
    //             oriScale01 ← scaleBaseAddr01 + innerIdx * 2 + outerIdx * outerStrideScale
    //             其中 scaleBaseAddr01 = scaleBaseAddr00 + 1
    //
    //    GMM:    oriScale00 ← scaleBase + (innerIdx * 2) * nTileAlign + outerIdx
    //            oriScale01 ← scaleBase + (innerIdx * 2 + 1) * nTileAlign + outerIdx
    oriScale00 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*2)     * nTileAlign + outerIdx)
    oriScale01 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*2 + 1) * nTileAlign + outerIdx)
    scale0 ← Select(oriScale00, oriScale01, maskH)  // 前半=00, 后半=01
    // offset 同理构造

    // 2. 加载 128 个 INT4 weight（覆盖 2 个 group × 64 K）— 与 wqbmmv2 一致
    weightB4 ← DataCopy<DIST_UNPACK4_B8>(weightInBaseAddr + innerIdx*64 + ...)

    // 3. Cast + Add(offset) + Mul
    weightB16 ← Cast(weightB4)
    if hasAntiQuantOffset:
      weightB16 ← Add(weightB16, offset0)
    weightB16 ← Mul(weightB16, scale0)

    // 4. 写入 UB
    DataCopy<DATA_BLOCK_COPY>(weightOutBaseAddr, weightB16, ...)

  weightOutBaseAddr += outerStride
```

#### 5.4.7 gs=32：AntiquantW4Pergroup32NK

每个 group 有 32 个 K 元素 = 1/4 向量寄存器宽度。

**核心差异**：ops-nn 在 `[N,G]` 布局下可用 `DIST_E2B_B16` 一次加载 8 个连续 group scale；GMM 的 scale/offset 固定为 `[G,N]`，同一个 N 下的 8 个 group 在 UB 中相隔 `nTileAlign`，不连续，因此**不能使用 `DIST_E2B_B16`**。

**参数**（`ParamsGroupSize32`）：

| 参数 | 计算方式 | 说明 |
|------|----------|------|
| `innerExtend` | `CeilDiv(bubKLen, 256)` | K 方向按 256 元素分块（8 groups × 32） |
| `outerExtend` | `bubNLen` | N 方向逐行 |
| `nTileAlign` | `CeilAlign(bubNLen, BLOCK_CUBE)` | **GMM `[G_tile,N_tile_aligned]` 的 N 方向步长** |
| `outerStrideWeight` | `CeilAlign(bubKLen, 32) >> 1` | weight N 方向步长（字节） |

**循环结构**（useVag=false 路径）：

```text
for innerIdx = 0 to innerExtend-1:                 // K 方向按 256 元素分块
  for outerIdx = 0 to outerExtend-1:               // N 方向逐行

    // 1. 分别广播 8 个 group 的 scale
    //    ★ wqbmmv2: 1 条 DIST_E2B_B16 加载 8 个连续 group + 1 条 Interleave
    //    ★ GMM:     8 条 DIST_BRC_B16 + 4 条 Interleave + 2 条 Select
    s0 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*8 + 0) * nTileAlign + outerIdx)
    s1 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*8 + 1) * nTileAlign + outerIdx)
    s2 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*8 + 2) * nTileAlign + outerIdx)
    s3 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*8 + 3) * nTileAlign + outerIdx)
    s4 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*8 + 4) * nTileAlign + outerIdx)
    s5 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*8 + 5) * nTileAlign + outerIdx)
    s6 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*8 + 6) * nTileAlign + outerIdx)
    s7 ← DataCopy<DIST_BRC_B16>(scaleBaseAddr + (innerIdx*8 + 7) * nTileAlign + outerIdx)

    // 2. Interleave 构造 64 元素对
    //    由于 s0..s7 是全同值寄存器（DIST_BRC_B16 广播），
    //    Interleave(sA, sB) = [sA×64, sB×64]
    AB ← Interleave(s0, s1)    // [s0×64, s1×64]
    CD ← Interleave(s2, s3)    // [s2×64, s3×64]
    EF ← Interleave(s4, s5)    // [s4×64, s5×64]
    GH ← Interleave(s6, s7)    // [s6×64, s7×64]

    // 3. Select 拼接 128 元素寄存器
    //    Select(src0, src1, maskH) → [src0[0:64], src1[0:64]]
    scale0 ← Select(AB, CD, maskH)   // [s0×32, s1×32, s2×32, s3×32] (groups 0-3)
    scale1 ← Select(EF, GH, maskH)   // [s4×32, s5×32, s6×32, s7×32] (groups 4-7)
    // offset 同理构造；hasAntiQuantOffset=false 时编译期消除

    // 4. 加载 256 个 INT4 weight（分 2 次 × 128 元素）— 与 wqbmmv2 一致
    weightB4Vl0 ← DataCopy<DIST_UNPACK4_B8>(weightInBaseAddr + innerIdx*128 + ...)
    weightB4Vl1 ← DataCopy<DIST_UNPACK4_B8>(weightInBaseAddr + 64 + innerIdx*128 + ...)

    // 5. Cast + Add(offset) + Mul
    weightB16Vl0 ← Cast(weightB4Vl0)
    weightB16Vl1 ← Cast(weightB4Vl1)
    if hasAntiQuantOffset:
      weightB16Vl0 ← Add(weightB16Vl0, offset0)
      weightB16Vl1 ← Add(weightB16Vl1, offset1)
    weightB16Vl0 ← Mul(weightB16Vl0, scale0)
    weightB16Vl1 ← Mul(weightB16Vl1, scale1)

    // 6. 分别写入两个输出位置
    DataCopy<DATA_BLOCK_COPY>(weightOutBaseAddr0, weightB16Vl0, ...)
    DataCopy<DATA_BLOCK_COPY>(weightOutBaseAddr1, weightB16Vl1, ...)

  weightOutBaseAddr0 += outDimOffset
  weightOutBaseAddr1 += outDimOffset
```

**指令开销**：8 条 `DIST_BRC_B16` + 4 条 `Interleave` + 2 条 `Select` = 14 条指令（仅 scale 部分）。offset 同理，`hasAntiQuantOffset=true` 时额外 14 条。

> **注意**：`Select` 和 `Interleave` 的具体 mask 构造需在实现时根据 Ascend C MicroAPI 文档确认。`maskH` 预期为前 64 lane 选 src0、后 64 lane 选 src1 的模式。

### 5.5 INT4 反量化步骤

每个 VF tile 的核心操作：

```text
1. 从 UB 读取 INT4 packed weight
2. unpack INT4（DIST_UNPACK4_B8）
3. cast INT4 -> xType(FP16/BF16)（Cast<xType, int4x2_t, castS4ToF16Trait>）
4. 若 hasAntiQuantOffset: weight += offset（Add）
5. weight *= scale（Mul）
6. 将 xType weight 写入 UB highBitData（DATA_BLOCK_COPY + POST_MODE_UPDATE）
7. 由现有 MTE3 流程搬到 L1 WeightL1
```

`hasAntiQuantOffset` 通过 `WqmmConfig.hasAntiQuantOffset` 做编译期分支。`PER_GROUP` 路径必须和 `PER_CHANNEL` 一样，在 `hasAntiQuantOffset=true` 时把 offset 从 GM 搬到 `ubAntiQuantOffsetTotalBuffer_`。

**关键指令说明**：

| 指令 | 作用 | 数据格式 |
|------|------|----------|
| `DIST_UNPACK4_B8` | 从 UB 加载 INT4 packed 数据并 unpack 到寄存器 | Vn: 每字节 2 个 INT4 → Vd: 每 8 字节存 2 个 INT4 |
| `DIST_NORM` | 从 UB 直接加载 FP16 scale 到寄存器（KN 路径） | 连续 128 个 FP16 元素 |
| `DIST_BRC_B16` | 从 UB 广播单个 FP16 到 128 元素（NK 路径） | 1 个 FP16 → 128 个相同 FP16 |
| `Cast<xType, int4x2_t>` | INT4 → FP16/BF16 类型转换 | 128 个 INT4 → 128 个 FP16 |
| `Mul` | 向量乘法 | weight × scale |
| `DATA_BLOCK_COPY + POST_MODE_UPDATE` | 将寄存器数据写入 UB（NZ 格式） | 每次写 16 个 FP16，间距 dataBlockStride |

### 5.6 Scale/Offset UB 布局

VF 看到的 scale/offset shape 统一为当前 VF tile 的 `[G_tile, N_tile]`：

```text
G_tile = vfKRealLen / groupSize    // K % groupSize == 0
N_tile = vfNRealLen
```

最大数据量为 1KB，见 §4.3。

**UB 布局统一为 `[G_tile, N_tile_padded]`**：

| 路径 | UB 布局 | scale 加载方式 |
|------|---------|--------------|
| KN (bTrans=false) | `[G_tile, N_tile_align]` | `DIST_NORM`（连续加载 128 元素） |
| NK (bTrans=true) | `[G_tile, N_tile_align]` | `DIST_BRC_B16`（广播）/ `Select` 拼接 |

**与 wqbmmv2 的差异**：wqbmmv2 NK 路径的 UB 布局为 `[N_tile, G_tile_align]`（N 为外轴），GMM 始终为 `[G_tile, N_tile_align]`（G 为外轴）。这是 NK 路径需要修改 scale 地址公式的根本原因。

### 5.7 双 AIV 并行切分

Ascend 950 的 AI Core 包含 1 个 Cube 和 2 个 Vector Core（AIV0、AIV1）。GMM 的 basic_block 利用双 AIV 并行执行反量化。

#### 5.7.1 切分策略

| 路径 | bTrans | 切分方向 | AIV0 工作量 | AIV1 工作量 |
|------|--------|----------|-------------|-------------|
| ND KN | false | **K** | 前 `kbL1Size / 2` 行 | 剩余 K 行 |
| ND NK | true | **N** | 前 `nL1Size / 2` 列 | 剩余 N 列 |

**ND KN 路径**（`ComputeBasicBlockAivNdKnNzNk`）：

```cpp
uint64_t kMte2BaseSize = offsetParam.kbL1Size / VEC_CORE_NUM;  // K 平分给 2 个 AIV
l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * kMte2BaseSize;
// AIV0: 前 K/2 行，AIV1: 后 K/2 行
// 两个 AIV 都处理完整的 N
ubConsumeConfig.l1RequireVfComputeRealN = offsetParam.nL1Size;
```

**ND NK 路径**（`ComputeBasicBlockAivNdNkNzKn`）：

```cpp
uint64_t curVecCoreMte2RealN = offsetParam.nL1Size / VEC_CORE_NUM;  // N 平分给 2 个 AIV
ubConsumeConfig.l1RequireVfComputeRealN =
    GetSubBlockIdx() == 0 ? curVecCoreMte2RealN : nL1Size - curVecCoreMte2RealN;
l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * curVecCoreMte2RealN;
// 两个 AIV 都处理完整的 K（外层循环迭代 K）
```

#### 5.7.2 对 pergroup 反量化的影响

| 路径 | AIV 切分方向 | per-AIV 块大小 | group num per AIV |
|------|-------------|---------------|------------------|
| KN (bTrans=false) | K | N=256, K=128 | 128 / groupSize |
| NK (bTrans=true) | N | N=128, K=256 | 256 / groupSize |

- **KN 路径**：每个 AIV 处理的 K 减半，group 数减半，但 scale 加载方式不变（`DIST_NORM` 按 group 加载，在 K 内循环中复用 groupSize 次）。这些差异通过 `param.groupNum` 和 `param.groupTail` 参数自动处理，不需要修改 VF 函数。
- **NK 路径**：每个 AIV 处理的 N 减半，但 K 完整，group 数不变。
- **gs=32 KN**：每个 AIV 处理 K=128，对应 4 个 group。scale 加载方式与 gs=128/256 完全相同。**无需特殊处理。**
- **gs=32 NK**：每个 AIV 处理 K=256（完整），对应 8 个 group。需要构造 2 个 128 元素 scale 寄存器。**需要特殊构造**（详见 §5.4.7）。

**单任务回退**：当 `kBL1Len_ <= kBubSize`（KN）或 `nBL1Len_ <= nBubSize`（NK）时，两个 AIV 执行相同的工作（冗余执行），不影响正确性。

### 5.8 KN vs NK 路径对比

| 维度 | KN 路径（`bTrans=false`） | NK 路径（`bTrans=true`） |
|------|--------------------------|--------------------------|
| VF 函数数量 | 1 个统一函数 | 4 个专用函数（按 gs 分派） |
| 循环结构 | nLoop → group → k（三层） | group → N 或 inner → outer（两层） |
| scale 加载方式 | `DIST_NORM`（直接加载 N 向量） | `DIST_BRC_B16` 广播；gs=64/32 用 `Select` 拼接 |
| K 方向处理 | 逐 K 行（groupSize 次迭代） | 按 group 整体处理（gs 个 K 共享 1 个 scale） |
| weight 加载 | 每次 128 个 INT4（1 个寄存器） | gs=128: 1 次; gs=256: 2 次; gs=64/32: 按 innerExtend 分块 |
| 输出格式 | NZ（K 为内轴，`dataBlockStride = CeilAlign(K,16)+1`） | NZ（N 为内轴，`dataBlockStride = CeilAlign(N,128)+1`） |
| groupTail 处理 | 有（单独循环处理不完整 group） | 无（本规格 K%gs==0，不产生 tail） |
| scale 地址公式 | `base + gIdx × nBubXTypeAlign + nIdx × vfElemB16` | `base + gIdx × nTileAlign + nBubIdx` |
| 与 wqbmmv2 差异 | scale/offset 布局一致，可直接复用 | scale/offset 布局不同（`[G,N]` vs `[N,G]`），需改地址公式 |

### 5.9 NK 路径地址公式汇总

| groupSize | wqbmmv2 scale 地址 | GMM scale 地址 | 加载方式 |
|-----------|-------------------|---------------|----------|
| 128 | `base + g + n × strideS` | `base + g × nTileAlign + n` | DIST_BRC_B16 |
| 256 | `base + g + n × strideS` | `base + g × nTileAlign + n` | DIST_BRC_B16 |
| 64 | `base + g + n × strideS` (2次, g 和 g+1) | `base + g × nTileAlign + n` (2次) | DIST_BRC_B16 + Select |
| 32 | `base + g×8 + n × strideS` (E2B, 8个连续) | `base + (g×8+i) × nTileAlign + n` (8次) | DIST_BRC_B16 × 8 + Interleave × 4 + Select × 2 |

> **KN 路径**（bTrans=false）所有 groupSize 统一使用 `DIST_NORM` 加载 scale，地址公式为 `base + gIdx × nBubXTypeAlign + nIdx × vfElemB16`，不需要按 groupSize 分派。详见 §5.3。

其中：
- `g` = groupIdx 或 innerIdx（取决于 groupSize）
- `n` = nBubIdx 或 outerIdx
- `strideS` = `CeilAlign(groupNumBub_, ONE_BLK_X_NUM)`（wqbmmv2 NK 的 scale UB 外轴步长）
- `nTileAlign` = `CeilAlign(nRealLen, BLOCK_CUBE)`（GMM 的 scale UB 外轴步长）

### 5.10 性能影响分析

#### gs=32 NK 路径 scale 加载开销

| 维度 | wqbmmv2 NK | GMM NK |
|------|-----------|--------|
| **Scale 加载指令数** | 1 条 `DIST_E2B_B16` + 1 条 `Interleave` | 8 条 `DIST_BRC_B16` + 4 条 `Interleave` + 2 条 `Select` |
| **指令总数（scale 部分）** | ~2 条 | ~14 条 |
| **性能影响** | 基准 | scale 加载开销增加约 7 倍 |

**缓解因素**：

1. **初始实现优先正确性**：先用直接地址计算（`useVag=false`），验证精度
2. **AddrReg 优化**：后续可用 `CreateAddrReg` 减少地址计算指令
3. **实际影响有限**：gs=32 在实际模型中使用较少（MoE 场景通常 gs=128），且 scale 加载只是整个 VF 函数的一小部分
4. **KN 路径无额外开销**：KN 路径所有 groupSize 统一使用 `DIST_NORM` 加载 scale（1 条指令），gs=32 与 gs=128/256 的 scale 加载开销完全相同

### 5.11 迁移到 GMM 的适配要点

1. **函数签名适配**：GMM 的 VF 函数需要接收 GMM 风格的参数（`offsetParam`、`localAddressParam`），而非 wqbmmv2 的 `ParamsKN` / `ParamsGroupSize*`。需要在 `weight_quant_vec_compute.h` 中构造参数后调用。

2. **UB 地址映射**：
   - `weightInUbBaseAddr_` → GMM 的 `ubWeightInputLowBitTotalBuffer_` 偏移
   - `weightOutUbAddr_` → GMM 的 `ubHighBitTotalBuffer_` 偏移
   - `scaleBaseAddr1_` → GMM 的 `ubAntiQuantScaleTotalBuffer_` 当前 VF tile 偏移
   - `offsetBaseAddr1_` → GMM 的 `ubAntiQuantOffsetTotalBuffer_` 当前 VF tile 偏移

3. **NK scale/offset 地址公式独立于 ops-nn**：NK 路径中，scale/offset 地址使用 `gLocal * nTileAlign + nLocal`。不要使用 ops-nn 的 `nLocal * outerStrideScale + gLocal`，否则在本算子固定 `[G,N]` 输入下会把不同 N/group 的参数读错。

4. **输出 NZ 格式**：VF 输出到 UB 的 highBitData 使用 NZ 格式（`DATA_BLOCK_COPY + POST_MODE_UPDATE`），后续 MTE3 从 UB 搬到 L1 时需要按 NZ 格式解析。这与 GMM 现有的 MTE3 流程一致，无需修改。

---

## 6. Cube 计算

文件：`weight_quant_cube_compute.h`

Cube 侧完全复用 A16W8：

| 函数 | 结论 |
|------|------|
| `ComputeBasicBlockAic()` | 复用 |
| `CopyAAndBiasGmToL1()` | 复用 |
| `LaunchMatmul()` | 复用 |
| `GetTensorC()` | 复用 |
| `WaitMTE1ToMTE2/SetMTE1ToMTE2` | 复用 |

原因：Cube 只消费 L1 中已反量化为 `xType` 的 weight，不感知 weight 原始是 A16W8 还是 A16W4，也不感知 perchannel/pergroup。

---

## 7. 修改点汇总

### 7.1 需要修改

| 文件 | 修改内容 |
|------|----------|
| `grouped_matmul_apt.cpp` | 新增 A16W4 pergroup ND 的 8 个 basic-controller 分支 |
| `weight_quant_tiling_key.h` | 新增对应 tiling key 组合 |
| `grouped_weight_quant_batch_matmul_tiling.cpp` | A16W4 pergroup shape/groupSize/tiling key 校验与生成 |
| `grouped_matmul_weight_quant_basic_controller.h` | `SetGmAddr` 中 pergroup scale/offset 偏移和 bias 独立偏移 |
| `basic_block_config.h` | 新增或局部定义 A16W4 ND pergroup `WqmmConfig` |
| `weight_quant_vec_compute.h` | ① `CopyAntiQuantParamsGmToUb` 补充 PER_GROUP offset 搬运（`if constexpr (hasAntiQuantOffset)` 分支）；② `CalLocalAddrForVf` 增加 PER_GROUP scale UB 地址计算（含 K 方向 group 偏移）；③ 在 ND + PER_GROUP + INT4 下分发到 A16W4 pergroup VF 参数构造 |
| `basic_block_vf_nd.h` | 新增 KN/NK A16W4 pergroup VF 实现；`NdNkLoadScaleOffset` / `NdKnLoadScaleOffset` 增加 PER_GROUP 分支（NK 按 `[G,N]` 布局构造广播与 Select） |
| `grouped_matmul_def.cpp` | config950 补充 INT4 ND 组合 |

### 7.2 不修改

| 文件/模块 | 结论 |
|-----------|------|
| `grouped_matmul_tiling_data_apt.h` | `groupSize` 字段已存在 |
| `weight_quant_basic_block.h` | BasicBlock AIV/AIC 分发复用 |
| `weight_quant_cube_compute.h` | Cube 计算复用 |
| `GetNdBufferInfo()` | A16W4 pergroup 复用 A16W8 ND UB buffer |
| L1 layout | A16W8 ND layout 复用 |

### 7.3 兼容性保护

| 保护点 | 方案 |
|--------|------|
| A16W8 perchannel | `WqmmConfig.antiQuantType == PER_CHANNEL` 原路径不变 |
| A16W4 pergroup | 新 tiling key + `PER_GROUP` 编译期分支隔离 |
| 有/无 offset | `hasAntiQuantOffset` 编译期分支 |
| 有/无 bias | tiling key 选择 + runtime `hasBias` 传入 BasicBlock |
| A2/A3 | 仅 arch35 路径适配，不触碰 arch30 |
