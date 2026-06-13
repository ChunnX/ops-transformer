# 04 Kernel 实现方案 — GMM A16W4 PerGroup ND

> 本文档设计 A16W4 pergroup ND 场景的 kernel 实现方案，覆盖 Controller、BasicBlock、VF 函数各层。
> 相关文档：[01 交付件清单](./01_deliverables.md)、[02 与硬件无关方案](./02_platform_independent_design.md)、[03 Tiling 方案](./03_tiling_design.md)

---

## 1. Controller 层适配

**文件**：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h`

### 1.1 修改点总览

| 函数 | 修改内容 | 复杂度 |
|------|----------|:------:|
| `Init()` | 无需修改（groupSize 已通过 `gmmBaseTiling_->groupSize` 传递） | 低 |
| `Process()` | 无需修改（主循环框架通用） | 低 |
| `SetMKN()` | 无需修改（MKN 取值逻辑通用） | 低 |
| `SetGmAddr()` | **★ 修改 antiquantScale/Offset 偏移公式 + bias 偏移分离** | 中 |
| `GetSplitValueFromGroupList()` | 无需修改 | 低 |

### 1.2 SetGmAddr() 中 pergroup 地址偏移公式设计

#### 1.2.1 当前 perchannel 实现

```cpp
// 当前 perchannel 代码
antiquantParamsBaseOffset += offsetParam.nSize;  // scale shape = (E, N)
```

#### 1.2.2 pergroup 修改方案

```cpp
// 修改后代码
if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    // pergroup: scale shape = (E, G, N), G = K/gs
    // 每个 group 的 scale 元素数 = (kSize / groupSize) * nSize
    antiquantParamsBaseOffset += (offsetParam.kSize / gmmBaseTiling_->groupSize) * offsetParam.nSize;
} else {
    // perchannel: scale shape = (E, N)
    antiquantParamsBaseOffset += offsetParam.nSize;
}
```

#### 1.2.3 bias 偏移分离

**问题**：现有代码中 bias 地址使用 `antiquantParamsBaseOffset`：
```cpp
biasGm = GetTensorAddr<biasType>(0, biasGm_) + antiquantParamsBaseOffset;
```

在 pergroup 场景下，bias shape 为 $(E, N)$ 而 antiquantScale shape 为 $(E, G, N)$，两者偏移量不同。

**修正方案**：新增独立的 `biasBaseOffset` 变量：

```cpp
// 在 SetGmAddr() 中
if (singleWeight == 0) {
    biasGm = GetTensorAddr<biasType>(groupIdx, biasGm_);
} else {
    biasGm = GetTensorAddr<biasType>(0, biasGm_) + biasBaseOffset;  // ★ 使用独立偏移
}

// 偏移更新
biasBaseOffset += offsetParam.nSize;  // bias shape = (E, N)，每 group 偏移 N
```

#### 1.2.4 int4b_t weight 偏移 >> 1 处理验证

现有代码：
```cpp
if constexpr (IsSameType<wType, int4b_t>::value) {
    weightBaseOffset += (nSize * kSize) >> 1;  // INT4: 字节数 = 元素数 / 2
} else {
    weightBaseOffset += nSize * kSize;
}
```

**pergroup 场景验证**：
- weight shape 为 $(E, N, K)$ 或 $(E, K, N)$
- 每个 group 的元素数 = $N \times K$
- INT4 字节数 = $N \times K / 2$
- `>> 1` 处理正确，**无需修改**

### 1.3 完整 SetGmAddr() 修改伪代码

```cpp
void SetGmAddr(uint64_t groupIdx, uint64_t &xBaseOffset, uint64_t &weightBaseOffset,
               uint64_t &yBaseOffset, uint64_t &antiquantParamsBaseOffset,
               uint64_t &biasBaseOffset,  // ★ 新增参数
               const BasicBlockOffsetParam &offsetParam)
{
    // === x 地址（不变）===
    if (singleX == 0) {
        xGm = GetTensorAddr<xType>(groupIdx, xGm_);
    } else {
        xGm = GetTensorAddr<xType>(0, xGm_) + xBaseOffset;
    }

    // === y 地址（不变）===
    if (singleY == 0) {
        yGm = GetTensorAddr<yType>(groupIdx, yGm_);
    } else {
        yGm = GetTensorAddr<yType>(0, yGm_) + yBaseOffset;
    }

    // === weight/antiquantScale/antiquantOffset/bias 地址 ===
    if (singleWeight == 0) {
        weightGm = GetTensorAddr<wType>(groupIdx, weightGm_);
        antiquantScaleGm = GetTensorAddr<xType>(groupIdx, antiquantScaleGm_);
        antiquantOffsetGm = GetTensorAddr<xType>(groupIdx, antiquantOffsetGm_);
        biasGm = GetTensorAddr<biasType>(groupIdx, biasGm_);
    } else {
        weightGm = GetTensorAddr<wType>(0, weightGm_) + weightBaseOffset;
        antiquantScaleGm = GetTensorAddr<xType>(0, antiquantScaleGm_) + antiquantParamsBaseOffset;
        antiquantOffsetGm = GetTensorAddr<xType>(0, antiquantOffsetGm_) + antiquantParamsBaseOffset;
        biasGm = GetTensorAddr<biasType>(0, biasGm_) + biasBaseOffset;  // ★ 独立偏移
    }

    // 更新全局地址到 BasicBlock（不变）
    wqmmBasicBlock_.UpdateGlobalAddr(xGm, weightGm, antiquantScaleGm, antiquantOffsetGm,
                                      nullptr, nullptr, biasGm, yGm, isBias, true);

    // === 偏移更新 ===
    xBaseOffset += offsetParam.mSize * offsetParam.kSize;

    if constexpr (IsSameType<wType, int4b_t>::value) {
        weightBaseOffset += (offsetParam.nSize * offsetParam.kSize) >> 1;
    } else {
        weightBaseOffset += offsetParam.nSize * offsetParam.kSize;
    }

    // ★ pergroup/perchannel 分支
    if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
        antiquantParamsBaseOffset += (offsetParam.kSize / gmmBaseTiling_->groupSize) * offsetParam.nSize;
    } else {
        antiquantParamsBaseOffset += offsetParam.nSize;
    }

    biasBaseOffset += offsetParam.nSize;  // ★ bias 独立偏移
    yBaseOffset += offsetParam.mSize * offsetParam.nSize;
}
```

---

## 2. BasicBlock 层适配

**文件**：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h`、`weight_quant_basic_block.h`

### 2.1 WqmmConfig 新增配置项设计

在 `basic_block_config.h` 中新增以下 `WqmmConfig` 实例：

```cpp
// A16W4 ND pergroup — B 非转置（weight shape = (N, K)，K 为内轴）
static constexpr WqmmConfig A16W4_NDKN_PG = {
    false,                    // aTrans = false
    false,                    // bTrans = false
    QuantType::PER_GROUP,     // antiQuantType = PER_GROUP
    false,                    // hasAntiQuantOffset = false（对称量化）
    QuantType::NONE,          // quantType = NONE
    CubeFormat::ND            // weightFormat = ND
};

// A16W4 ND pergroup — B 非转置 + 有 offset
static constexpr WqmmConfig A16W4_NDKN_PG_OFF = {
    false, false, QuantType::PER_GROUP, true, QuantType::NONE, CubeFormat::ND
};

// A16W4 ND pergroup — B 转置（weight shape = (K, N)，N 为内轴）
static constexpr WqmmConfig A16W4_NDNK_PG = {
    false, true, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::ND
};

// A16W4 ND pergroup — B 转置 + 有 offset
static constexpr WqmmConfig A16W4_NDNK_PG_OFF = {
    false, true, QuantType::PER_GROUP, true, QuantType::NONE, CubeFormat::ND
};
```

**与已有实例对比**：

| 实例 | aTrans | bTrans | antiQuantType | hasOffset | weightFormat |
|------|:------:|:------:|:-------------:|:---------:|:------------:|
| S8S4_NZKN_G | false | false | PER_GROUP | false | **NZ** |
| A16MXF4_NZKN | false | false | MX | false | NZ |
| MXA8W4_NZNK | false | true | MX | false | NZ |
| **A16W4_NDKN_PG** | false | false | **PER_GROUP** | false | **ND** |
| **A16W4_NDNK_PG** | false | **true** | **PER_GROUP** | false | **ND** |

### 2.2 WeightQuantMatmulBasicBlock 模板参数适配

`WeightQuantMatmulBasicBlock` 的模板参数链：

```cpp
template <typename xType, typename wType, typename biasType, typename yType,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
class WeightQuantMatmulBasicBlock;
```

**适配方式**：新增的 `WqmmConfig` 实例通过模板参数传递，BasicBlock 内部通过 `if constexpr (wqmmConfig.antiQuantType == PER_GROUP)` 分支处理 pergroup 逻辑。

**无需修改 BasicBlock 类定义**，仅需确保内部分支正确处理 pergroup。

### 2.3 ComputeBasicBlockAivNdNkNzKn 中 pergroup 分支设计

**现有分发逻辑**（`weight_quant_basic_block.h`）：

```cpp
void ComputeBasicBlock(const BasicBlockOffsetParam &offsetParam, ...)
{
    if ASCEND_IS_AIV {
        if ((ND && !bTrans) || (NZ && bTrans)) {
            ComputeBasicBlockAivNdKnNzNk(offsetParam);  // K 为内轴
        } else {
            ComputeBasicBlockAivNdNkNzKn(offsetParam);  // N 为内轴
        }
    } else {
        ComputeBasicBlockAic(offsetParam);
    }
}
```

**A16W4 pergroup 场景**：
- B 非转置（ND NK）→ `ComputeBasicBlockAivNdKnNzNk`（K 为内轴）
- B 转置（ND KN）→ `ComputeBasicBlockAivNdNkNzKn`（N 为内轴）

**分发逻辑无需修改**，pergroup 的差异在 VF 函数层处理。

---

## 3. L1/UB 空间划分方案

**文件**：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h`

### 3.1 L1 Buffer 划分

**L1 总量**：512 KB

A16W4 pergroup 的 L1 划分与 A16W8 **完全一致**，无需修改：

```
无 bias 时:
| WeightL1_P0 (128KB) | AL1_P0 (128KB) | AL1_P1 (128KB) | WeightL1_P1 (128KB) |

有 bias 时:
| WeightL1_P0 (128KB) | Bias_P0 (4KB) | AL1_P0 (124KB) | AL1_P1 (124KB) | Bias_P1 (4KB) | WeightL1_P1 (128KB) |
```

**WeightL1 大小**：
```cpp
weightL1Space = baseN * stepKb * baseK;  // 256 * 4 * 64 = 65536 元素 = 128KB (fp16)
```

**说明**：L1 存储的是反量化后的 weight（FP16/BF16），与反量化类型（perchannel/pergroup）无关。

### 3.2 UB Buffer 划分

#### 3.2.1 当前 ND 场景 Buffer 配置（GetNdBufferInfo()）

```cpp
// 当前 A16W8 ND perchannel 配置
{
    .ubWeightOutputHighBitBufferNum = 2,       // 双缓冲
    .weightInputLowbitUbTotalSize = 174KB,     // weight 输入（int4 原始数据）
    .highBitDataUbTotalSize = 66KB,            // 反量化后 fp16 weight
    .antiQuantScaleUbTotalSize = 4KB,          // ★ perchannel scale（仅 1 行）
    .antiQuantScaleAfterCastUbTotalSize = 0,
    .antiQuantOffsetUbTotalSize = 4KB,         // perchannel offset
    .weightInputLowBitUbSingleBufferSize = 174KB / ubMte2BufferNum,
    .antiQuantScaleUbSingleBufferSize = 4KB / ubMte2BufferNum,
    .antiQuantOffsetUbSingleBufferSize = 4KB / ubMte2BufferNum,
    .highBitDataUbSingleBufferSize = 66KB / 2,
};
```

#### 3.2.2 pergroup scale UB 空间计算

**pergroup scale 数据量分析**：
- scale shape 在 UB 上为 $(\lceil kL1/gs \rceil, nL1)$
- 以 baseK=64, baseN=256 为例：
  - gs=32: $\lceil 64/32 \rceil \times 256 = 2 \times 256 = 512$ 个元素 = 1KB
  - gs=64: $\lceil 64/64 \rceil \times 256 = 1 \times 256 = 256$ 个元素 = 0.5KB
  - gs=128: $\lceil 64/128 \rceil \times 256 = 1 \times 256 = 256$ 个元素 = 0.5KB
  - gs=256: $\lceil 64/256 \rceil \times 256 = 1 \times 256 = 256$ 个元素 = 0.5KB

**但实际 kL1 可能更大**（kL1 = kbL1Size，由 Controller 动态决定）：
- 若 kL1=256, gs=32: $\lceil 256/32 \rceil \times 256 = 8 \times 256 = 2048$ 个元素 = 4KB
- 若 kL1=256, gs=64: $\lceil 256/64 \rceil \times 256 = 4 \times 256 = 1024$ 个元素 = 2KB

**结论**：当前 4KB scale buffer 在大多数场景下足够，但 gs=32 + kL1=256 时刚好达到上限。

**方案**：新增 `GetNdPerGroupBufferInfo()` 函数，增大 scale/offset buffer：

```cpp
template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr UbBufferInfo GetNdPerGroupBufferInfo()
{
    return {.ubWeightOutputHighBitBufferNum = DOUBLE_BUFFER_NUM,
            .weightInputLowbitUbTotalSize = 174 * GetKBUnit<int8_t>(),  // 174KB（不变）
            .highBitDataUbTotalSize = 62 * GetKBUnit<half>(),           // 62KB（缩减 4KB）
            .antiQuantScaleUbTotalSize = 8 * GetKBUnit<half>(),         // ★ 8KB（增大）
            .antiQuantScaleAfterCastUbTotalSize = 0,
            .antiQuantOffsetUbTotalSize = 8 * GetKBUnit<half>(),        // ★ 8KB（增大）
            .weightInputLowBitUbSingleBufferSize = 174 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleUbSingleBufferSize = 8 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleAfterCastUbSingleBufferSize = 0,
            .antiQuantOffsetUbSingleBufferSize = 8 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .highBitDataUbSingleBufferSize = 62 * GetKBUnit<half>() / DOUBLE_BUFFER_NUM,
            .antiQuantScaleMaskBufferSize = 0};
}
```

**UB 总量验证**：174 + 62 + 8 + 8 = 252 KB ≤ 256 KB（UB 可用空间约 248~256 KB），满足约束。

#### 3.2.3 GetBufferConfig() 适配

```cpp
template <typename xType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr UbBufferInfo GetBufferConfig()
{
    if constexpr (wqmmConfig.antiQuantType == QuantType::MX) {
        // ... MX 路径不变
    }

    if constexpr (wqmmConfig.weightFormat == CubeFormat::ND) {
        if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
            return GetNdPerGroupBufferInfo<vecConfig>();  // ★ pergroup 专用
        } else {
            return GetNdBufferInfo<vecConfig>();          // perchannel 不变
        }
    } else {
        // ... NZ 路径不变
    }
}
```

### 3.3 VecAntiQuantConfig 参数设置

复用现有 `VEC_ANTIQUANT_CONFIG_3`：

```cpp
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
// ubMte2BufferNum = 4（四缓冲）
// ubMte2InnerSize = 256（每次 MTE2 搬运 256 元素宽度）
```

**说明**：pergroup 场景的 MTE2 搬运配置与 perchannel 一致，scale 的搬运通过 `CopyAntiQuantParamsGmToUb()` 中已有的 `PER_GROUP` 分支处理。

---

## 4. Vec 反量化函数实现

**文件**：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h`、`weight_quant_vec_compute.h`

### 4.1 新增 NK 排布 pergroup ND 反量化 VF 函数

**对应 wqbmmv2**：`AntiquantComputeW4NKPerGroup`（bTrans=true，weight shape = (N, K)）

**GMM 中的调用路径**：
```
ComputeBasicBlockAivNdNkNzKn()  // B 转置 → N 为内轴
  → WeightAntiQuantCompute()
    → AntiQuantProcessNd()
      → AntiQuantInt4NdNk()  // 当前仅支持 PER_TENSOR/PER_CHANNEL
        → NdNkLoadScaleOffset()  // ★ 需新增 PER_GROUP 分支
```

**函数签名**：

```cpp
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdNkPerGroup(
    __local_mem__ xType *antiQuantScaleBasePhyAddr,   // scale UB 基址
    __local_mem__ xType *antiQuantOffsetBasePhyAddr,  // offset UB 基址（可为 nullptr）
    __local_mem__ wType *weightLowBitPhyAddr0,        // weight 输入 UB 基址
    __local_mem__ xType *weightF16PhyAddr0,           // weight 输出 UB 基址
    uint64_t ubLoopN,       // N 方向循环次数（bubNLen）
    uint64_t ubLoopK,       // K 方向循环次数（bubKLen）
    uint64_t groupSize,     // pergroup 大小（从 tiling 传入）
    uint64_t groupNumBub)   // scale 的 K 方向 group 数
```

**核心逻辑**（参照 wqbmmv2 `AntiquantComputeW4NKPerGroup`）：

```
FOR groupIdx = 0 TO groupNum-1:
    FOR nBubIdx = 0 TO ubLoopN-1:
        // 1. 加载 scale/offset（DIST_BRC_B16 广播）
        scale = LoadBroadcast(scaleBase[groupIdx + nBubIdx * outerStrideScale])
        offset = LoadBroadcast(offsetBase[groupIdx + nBubIdx * outerStrideScale])

        // 2. 加载 int4 weight 并 unpack
        weightB4 = LoadUnpack4(weightInBase[groupIdx * innerStrideWeight + nBubIdx * outerStrideWeight])

        // 3. Cast int4 → xType
        weightB16 = Cast(weightB4)

        // 4. 反量化
        IF hasOffset: weightB16 += offset
        weightB16 *= scale

        // 5. 写回 UB
        StoreBlockCopy(weightOutBase, weightB16, dataBlockStride, repeatStride)
```

**按 groupSize 分支处理策略**：

| groupSize | 策略 | 说明 |
|:---------:|------|------|
| 128 | 专用 VF 函数 | 128 个 int4 = 64B = 一个完整 VREG，最高效 |
| 256 | 专用 VF 函数 | 256 个 int4 = 128B = 两个 VREG，需两次加载 |
| 64 | 专用 VF 函数 | 64 个 int4 = 32B = 半个 VREG，需 Select 拼接 |
| 32 | 专用 VF 函数 | 32 个 int4 = 16B，需特殊处理 |

**推荐方案**：运行时分支（与 wqbmmv2 一致），在 `AntiQuantInt4NdNkPerGroup` 内部按 groupSize 分发到不同的专用 VF 函数。

### 4.2 新增 KN 排布 pergroup ND 反量化 VF 函数

**对应 wqbmmv2**：`AntiquantComputeKNPerGroup`（bTrans=false，weight shape = (K, N)）

**GMM 中的调用路径**：
```
ComputeBasicBlockAivNdKnNzNk()  // B 非转置 → K 为内轴
  → WeightAntiQuantCompute()
    → AntiQuantProcessNd()
      → AntiQuantInt4NdKn()  // 当前仅支持 PER_TENSOR/PER_CHANNEL
        → NdKnLoadScaleOffset()  // ★ 需新增 PER_GROUP 分支
```

**函数签名**：

```cpp
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdKnPerGroup(
    __local_mem__ xType *antiQuantScaleBasePhyAddr,
    __local_mem__ xType *antiQuantOffsetBasePhyAddr,
    __local_mem__ wType *weightLowBitPhyAddr0,
    __local_mem__ xType *weightF16PhyAddr0,
    uint64_t ubLoopN,       // N 方向循环次数
    uint64_t ubLoopK,       // K 方向循环次数
    uint64_t groupSize)     // pergroup 大小
```

**核心逻辑**（参照 wqbmmv2 `AntiquantComputeKNPerGroup`）：

```
FOR nIdx = 0 TO nLoop-1:           // N 方向按 128 元素分块
    FOR gIdx = 0 TO groupNum-1:    // 完整 group 遍历
        // 加载 scale/offset（DIST_NORM 直接搬运）
        scale = LoadNorm(scaleBase[nIdx * 128 + gIdx * nBubXTypeAlign])
        offset = LoadNorm(offsetBase[nIdx * 128 + gIdx * nBubXTypeAlign])

        FOR kIdx = 0 TO groupSize-1:  // group 内逐 K 行
            // 加载 int4 weight
            weightIn = LoadUnpack4(weightInBase[nIdx * wNStride + gIdx * wGroupStride + kIdx * wKStride])
            weightOut = Cast(weightIn)
            IF hasOffset: weightOut += offset
            weightOut *= scale
            StoreBlockCopy(weightOutBase, weightOut, dataBlockStride, repeatStride)

    // 处理 groupTail（最后一个不完整 group）
    ...
```

**KN 排布优势**：统一处理所有 groupSize，无需按 gs 值分支（因为 K 方向逐行步进，groupSize 只影响循环次数）。

### 4.3 函数接口对齐

#### 4.3.1 与 GMM 已有函数的对齐

| 维度 | 已有 `AntiQuantInt4NdNk` | 新增 `AntiQuantInt4NdNkPerGroup` |
|------|:------------------------:|:--------------------------------:|
| 模板参数 | `<xType, wType, wqmmConfig, vecConfig>` | 相同 |
| 前 4 个参数 | `scale, offset, weightIn, weightOut` | 相同 |
| 额外参数 | `scaleValue, offsetValue, ubLoopN` | `ubLoopN, ubLoopK, groupSize, groupNumBub` |
| 返回值 | void | void |

#### 4.3.2 与 wqbmmv2 的映射

| wqbmmv2 函数 | GMM 新增函数 | 排布 |
|-------------|-------------|:----:|
| `AntiquantComputeW4NKPerGroup` | `AntiQuantInt4NdNkPerGroup` | NK（B 转置） |
| `AntiquantComputeKNPerGroup` | `AntiQuantInt4NdKnPerGroup` | KN（B 非转置） |

### 4.4 INT4 weight unpack 和 cast 流程

```
1. GM → UB: weight INT4 数据通过 MTE2 搬运到 UB（weightInputLowBit buffer）
   - int4 以 int8 存储，每字节 2 个 int4 元素

2. UB 上 unpack:
   DataCopy<DIST_UNPACK4_B8>(weightS4Vreg, weightLowBitPhyAddr)
   - int4x2_t → int4x2_t（unpack 后每个 int4 占 2 字节）

3. Cast:
   Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg, weightS4Vreg, mask)
   - int4 → fp16/bf16

4. 反量化:
   Add(weightF16Vreg, weightF16Vreg, offsetVreg, mask)  // if hasOffset
   Mul(weightF16Vreg, weightF16Vreg, scaleVreg, mask)

5. UB → L1: 通过 MTE3 搬运到 L1（weightL1 buffer）
```

### 4.5 pergroup scale/offset 加载方式

#### 4.5.1 NK 排布（B 转置）

scale 在 UB 上的布局：$(groupNum, N)$ 的 ND 排布

```
scale[g, n] 地址 = scaleBaseAddr + g + n * outerStrideScale
其中 outerStrideScale = CeilAlign(groupNumBub, ONE_BLK_X_NUM)

加载方式：DIST_BRC_B16（广播单个值到整个 VREG）
- 每个 scale 值广播到 N 方向的所有 weight 元素
```

#### 4.5.2 KN 排布（B 非转置）

scale 在 UB 上的布局：$(groupNum, N)$ 的 ND 排布

```
scale[g, n] 地址 = scaleBaseAddr + nIdx * vfElemB16 + gIdx * nBubXTypeAlign

加载方式：DIST_NORM（直接搬运，不广播）
- 每个 group 的 N 方向 scale 一次性加载
```

### 4.6 按 groupSize 分支处理策略

**NK 排布**：按 groupSize 运行时分支（与 wqbmmv2 一致）

```cpp
__aicore__ inline void AntiQuantInt4NdNkPerGroup(...) {
    if (groupSize == 128) {
        // 调用 gs=128 专用 VF 逻辑
    } else if (groupSize == 256) {
        // 调用 gs=256 专用 VF 逻辑
    } else if (groupSize == 64) {
        // 调用 gs=64 专用 VF 逻辑
    } else if (groupSize == 32) {
        // 调用 gs=32 专用 VF 逻辑
    }
}
```

**KN 排布**：统一处理，无需分支

```cpp
__aicore__ inline void AntiQuantInt4NdKnPerGroup(...) {
    // groupSize 仅影响循环次数，统一逻辑
    for (gIdx = 0; gIdx < groupNum; gIdx++) {
        for (kIdx = 0; kIdx < groupSize; kIdx++) {
            // ...
        }
    }
}
```

---

## 5. Cube 计算

**文件**：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_cube_compute.h`

### 5.1 确认与 A16W8 多多多一致

Cube 计算流程**完全一致**，无需修改：

| 函数 | 说明 | 是否需要修改 |
|------|------|:------------:|
| `ComputeBasicBlockAic()` | AIC 核主循环 | 否 |
| `LaunchMatmul()` | matmul 调用 | 否 |
| `CopyAAndBiasGmToL1()` | A/Bias 搬运 | 否 |
| `GetTensorC()` | C 写回 GM | 否 |
| `WaitMTE1ToMTE2()` | 同步 | 否 |
| `SetMTE1ToMTE2()` | 同步 | 否 |

**原因**：Cube 计算接收的是反量化后的 FP16/BF16 weight 数据（已存储在 L1 中），与反量化类型（perchannel/pergroup）无关。

---

## 6. 核心代码适配点汇总

### 6.1 需要修改的文件和函数

| 序号 | 文件 | 函数 | 修改内容 |
|:----:|------|------|----------|
| 1 | `grouped_matmul_weight_quant_basic_controller.h` | `SetGmAddr()` | pergroup 地址偏移公式 + bias 偏移分离 |
| 2 | `basic_block_config.h` | 全局 | 新增 4 个 `WqmmConfig` 实例 |
| 3 | `basic_block_config.h` | `GetBufferConfig()` | 新增 pergroup 分支调用 `GetNdPerGroupBufferInfo()` |
| 4 | `basic_block_config.h` | `GetNdPerGroupBufferInfo()` | **新增函数**，pergroup 专用 buffer 配置 |
| 5 | `basic_block_vf_nd.h` | `AntiQuantInt4NdNkPerGroup()` | **新增函数**，NK 排布 pergroup VF |
| 6 | `basic_block_vf_nd.h` | `AntiQuantInt4NdKnPerGroup()` | **新增函数**，KN 排布 pergroup VF |
| 7 | `basic_block_vf_nd.h` | `NdNkLoadScaleOffset()` | 新增 `PER_GROUP` 分支 |
| 8 | `basic_block_vf_nd.h` | `NdKnLoadScaleOffset()` | 新增 `PER_GROUP` 分支 |
| 9 | `weight_quant_vec_compute.h` | `AntiQuantProcessNd()` | 新增 pergroup int4 分发 |
| 10 | `weight_quant_vec_compute.h` | `CopyAntiQuantParamsGmToUb()` | 确认 PER_GROUP 分支在 ND 下工作 |
| 11 | `grouped_matmul_apt.cpp` | `grouped_matmul()` | 新增 8 个 `if constexpr` 分支 |
| 12 | `weight_quant_tiling_key.h` | 全局 | 新增 8 个 `ASCENDC_TPL_ARGS_SEL` 组合 |
| 13 | `grouped_weight_quant_batch_matmul_tiling.cpp` | 多个函数 | pergroup 校验/计算分支 |
| 14 | `grouped_matmul_def.cpp` | config950 | 新增 INT4 ND dtype 组合 |

### 6.2 新增文件和函数清单

| 类型 | 名称 | 位置 |
|------|------|------|
| 新增函数 | `GetNdPerGroupBufferInfo()` | `basic_block_config.h` |
| 新增函数 | `AntiQuantInt4NdNkPerGroup()` | `basic_block_vf_nd.h` |
| 新增函数 | `AntiQuantInt4NdKnPerGroup()` | `basic_block_vf_nd.h` |
| 新增常量 | `A16W4_NDKN_PG` | `basic_block_config.h` |
| 新增常量 | `A16W4_NDKN_PG_OFF` | `basic_block_config.h` |
| 新增常量 | `A16W4_NDNK_PG` | `basic_block_config.h` |
| 新增常量 | `A16W4_NDNK_PG_OFF` | `basic_block_config.h` |

### 6.3 不需要修改的部分（确认可复用）

| 模块 | 函数/代码 | 说明 |
|------|-----------|------|
| Controller | `Init()` | 初始化流程通用 |
| Controller | `Process()` | 主循环框架通用 |
| Controller | `SetMKN()` | MKN 取值逻辑通用 |
| Controller | `GetSplitValueFromGroupList()` | groupList 解析通用 |
| Controller | `SetGmAddr()` x/weight/y 偏移 | 与反量化类型无关 |
| BasicBlock | `Init()` | L1 空间分配通用 |
| BasicBlock | `ComputeBasicBlock()` | 分发逻辑基于 weightFormat/bTrans |
| BasicBlock | `ComputeBasicBlockAic()` | Cube 计算流程通用 |
| Cube | `LaunchMatmul()` | matmul 调用通用 |
| Cube | `CopyAAndBiasGmToL1()` | A/Bias 搬运通用 |
| Cube | `GetTensorC()` | C 写回通用 |
| Vec | `CopyGmToUb()` | 搬运框架通用 |
| Vec | `CopyWeightGmToUb()` | weight 搬运通用 |
| Vec | `WeightAntiQuantCompute()` | 反量化主循环框架通用 |
| Vec | `WeightHighBitUbToL1()` | UB→L1 搬运通用 |
| Vec | `CalLocalAddrForVf()` | UB 地址计算通用 |
| Vec | `CopyAntiQuantParamsGmToUb()` PER_GROUP 分支 | 已存在，需确认 ND 适用性 |
| Sync | `SetAivToAic()` / `WaitAivToAic()` | 跨核同步通用 |
| Sync | `SetAicToAiv()` / `WaitAicToAiv()` | 跨核同步通用 |
