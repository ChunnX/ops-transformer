# WQBMMv2 ND INT4 PerGroup 反量化走读文档

> **目标**：走读 `weight_quant_batch_matmul_v2` 中 weight ND int4 pergroup 反量化的完整实现，明确迁移到 GMM `basic_block_vf_nd.h` 的适配方案。

## 目录

1. [常量与基础设施](#1-常量与基础设施)
2. [调度入口：AntiQuantCompute](#2-调度入口antiquantcompute)
3. [AntiquantComputeW4NKPerGroup（NK排布，bTrans=true）](#3-antiquantcomputew4nkpergroupnk排布btranstrue)
4. [AntiquantComputeKNPerGroup（KN排布，bTrans=false）](#4-antiquantcomputeknpergroupkn排布btransfalse)
5. [VF 底层函数实现详解](#5-vf-底层函数实现详解)
6. [NK vs KN 差异对比表](#6-nk-vs-kn-差异对比表)
7. [NZ 格式 pergroup 对比参照](#7-nz-格式-pergroup-对比参照)
8. [GMM basic_block_vf_nd.h 已有接口分析](#8-gmm-basic_block_vf_ndh-已有接口分析)
9. [迁移适配方案](#9-迁移适配方案)

---

## 1. 常量与基础设施

### 1.1 关键常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `VECTOR_REG_WIDTH` | 256 B | 向量寄存器宽度 |
| `ONE_BLK_SIZE` | 32 B | 单个 block 大小 |
| `BLOCK_CUBE` | 16 | NZ 格式 C0 维度大小 |
| `ONE_BLK_X_NUM` | 16 | 一个 block 中 half/bf16 元素数 (32/2) |
| `ONE_BLK_W_NUM` | 32 | 一个 block 中 int8 元素数 (32/1) |
| `VEC_MAX_ELEM_B16` | 128 | 一个向量寄存器中 half 元素数 (256/2) |
| `INT4_DTYPE_PARAM` | 1 | int4 右移参数（int4 每字节存 2 个元素） |

### 1.2 数据类型映射

```
int4b_t → RegTensorActualT → int4x2_t  (UB 上 int4 以 int8 存储，每字节 2 个 int4)
xType   → half / bfloat16_t            (反量化后的输出类型)
```

### 1.3 VF_CALL 机制

wqbmmv2 使用 `AscendC::VF_CALL<VfFunc>(params)` 机制将 VF 函数提交到向量单元执行。VF 函数运行在向量核上，通过 `__local_mem__` 指针访问 UB 缓冲区。

---

## 2. 调度入口：AntiQuantCompute

**文件**: `weight_quant_batch_matmul_v2_reg_base_common.h` L1451-1494

```
AntiQuantCompute(bubKOffset, bubNLen, bubKLen)
  │
  ├── 设置 UB 基地址
  │   ├── weightInUbBaseAddr_  = weightInUb_[offset].GetPhyAddr()
  │   ├── scaleBaseAddr1_      = scaleInUb_[offset].GetPhyAddr()
  │   ├── offsetBaseAddr1_     = offsetInUb_[offset].GetPhyAddr()
  │   └── weightOutUbAddr_     = weightOutUb_[offset].GetPhyAddr()
  │
  ├── bTrans=true (weight 转置，即 NK 排布)
  │   ├── wType=int4 && PER_GROUP && !weightNz → AntiquantComputeW4NKPerGroup(bubNLen, bubKLen)
  │   └── wType=int8 && PER_GROUP && !weightNz → AntiquantComputeW8NKPerGroup(bubNLen, bubKLen)
  │
  └── bTrans=false (weight 非转置，即 KN 排布)
      ├── IS_4BIT && PER_GROUP && weightNz  → AntiQuantComputeKNGroupWeightNz(...)
      └── PER_GROUP && !weightNz            → AntiquantComputeKNPerGroup(bubNLen, bubKLen)
```

**关键理解**：
- **NK 排布**（bTrans=true）：weight shape 为 `(N, K)`，即 N 是外层维度，K 是内层维度
- **KN 排布**（bTrans=false）：weight shape 为 `(K, N)`，即 K 是外层维度，N 是内层维度
- scale/offset shape 统一为 `(k/gs, n)` 的 ND 排布（pergroup 维度在 K 方向）

---

## 3. AntiquantComputeW4NKPerGroup（NK排布，bTrans=true）

**文件**: `weight_quant_batch_matmul_v2_reg_base_common.h` L1185-1409

### 3.1 函数签名

```cpp
TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::AntiquantComputeW4NKPerGroup(int32_t bubNLen, int32_t bubKLen)
```

**参数**：
- `bubNLen`: 当前 bubble 的 N 维度实际长度
- `bubKLen`: 当前 bubble 的 K 维度实际长度

### 3.2 按 groupSize 分支调度

该函数根据 `tiling_->groupSize` 值分为多个分支：

```
AntiquantComputeW4NKPerGroup(bubNLen, bubKLen)
  │
  ├── groupSize == 32  → VF_CALL<AntiquantW4Pergroup32NK>
  ├── groupSize == 64  → VF_CALL<AntiquantW4Pergroup64NK>
  ├── groupSize == 128 → VF_CALL<AntiquantW4Pergroup128NK>
  ├── groupSize == 256 → VF_CALL<AntiquantW4Pergroup256NK>
  ├── groupSize % 64 > 0 (非64对齐) → VF_CALL<AntiquantW4Pergroup32OddNK>
  └── groupSize > 128 && != 256 → VF_CALL<AntiquantW4PergroupGt128NKCase1/2>
```

### 3.3 groupSize=128 详解（典型路径）

**参数结构体**: `ParamsGroupSize128And256<xType, wType>`

```cpp
// 参数设置
params.dataBlockStride = CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1;  // 防 bank conflict
params.repeatStride = 1;
params.weightOutAddrOffset =
    VEC_MAX_ELEM_B16 * (CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1) - bubNLen * ONE_BLK_X_NUM;

params.bubNLen = bubNLen;
params.groupNum = CeilDiv(bubKLen, tiling_->groupSize);
params.outerStrideScale = CeilAlign(groupNumBub_, ONE_BLK_X_NUM);  // scale 的 N 方向步长
params.outerStrideWeight = CeilAlign(bubKLen, ONE_BLK_W_NUM) >> 1; // weight N 方向步长(字节)
params.innerStrideWeight = 64;  // 128个int4 = 64字节

params.weightInBaseAddr = weightInUbBaseAddr_;
params.weightOutBaseAddr = weightOutUbAddr_;
params.scaleBaseAddr = scaleBaseAddr1_;
params.offsetBaseAddr = offsetBaseAddr1_;
params.maskWeight = bubKLen;
```

**伪代码**：

```
FOR groupIdx = 0 TO groupNum-1:
    FOR nBubIdx = 0 TO bubNLen-1:
        // 1. 加载 scale/offset（广播到整个向量寄存器）
        scale  = LoadBroadcast(scaleBase[groupIdx + nBubIdx * outerStrideScale])
        offset = LoadBroadcast(offsetBase[groupIdx + nBubIdx * outerStrideScale])

        // 2. 加载 int4 weight 并 unpack 到 int4x2
        //    地址 = weightInBase + groupIdx * 64 + nBubIdx * outerStrideWeight
        weightB4 = LoadUnpack4(weightInBase[addr])

        // 3. Cast int4 → xType (half/bf16)
        weightB16 = Cast(weightB4)

        // 4. 反量化: (weight + offset) * scale
        IF hasOffset: weightB16 += offset
        weightB16 *= scale

        // 5. 写回 UB（DATA_BLOCK_COPY + POST_MODE_UPDATE）
        StoreBlockCopy(weightOutBase, weightB16, dataBlockStride, repeatStride)

    // N 维度循环结束后，修正输出地址
    weightOutBase += weightOutAddrOffset
```

**数据流图**：

```
                    UB Memory Layout (NK排布)
                    ========================

  weightIn (int4, stored as int8):
  ┌─────────────────────────────────────────────┐
  │ n=0: [k0k1 k2k3 k4k5 ... k_{K-1}k_K]       │ ← outerStrideWeight (bytes)
  │ n=1: [k0k1 k2k3 k4k5 ... k_{K-1}k_K]       │
  │ ...                                         │
  │ n=bubNLen-1: [...]                          │
  └─────────────────────────────────────────────┘

  scale/offset (xType, shape = (groupNum, N)):
  ┌──────────────────────────┐
  │ g=0: [s(n=0) s(n=1) ...] │ ← outerStrideScale
  │ g=1: [s(n=0) s(n=1) ...] │
  │ ...                      │
  └──────────────────────────┘

  weightOut (xType, NZ-like layout in UB):
  ┌──────────────────────────────────────────┐
  │ (k1, n1, n0, k0) format                 │
  │ block = 16 elements (ONE_BLK_X_NUM)     │
  │ dataBlockStride = CeilAlign(N,16)+1     │
  │ → 防 bank conflict                      │
  └──────────────────────────────────────────┘
```

### 3.4 groupSize=64 详解

**参数结构体**: `ParamsGroupSize64<xType, wType>`

核心差异：gs=64 时，一个 group 只有 64 个 int4 = 32 字节 = 半个向量寄存器。因此需要**两次加载 + Select 拼接**来凑满一个完整的向量寄存器。

```
scale/offset 加载方式：
  oriScale00 = Load(scaleBaseAddr00 + ...)  // 偶数位置
  oriScale01 = Load(scaleBaseAddr01 + ...)  // 奇数位置
  scale0 = Select(oriScale00, oriScale01, maskSelect)  // H mask: 取前半+后半

weight 加载：
  weight0B4 = LoadUnpack4(weightInBase + innerIdx*64 + outerIdx*outerStrideWeight)
  // 64个int4 = 32字节 = 1/4 VREG → unpack4 后占半个 VREG

计算：
  weight0B16 = Cast(weight0B4)
  weight0B16 = (weight0B16 + offset0) * scale0
  StoreBlockCopy(weightOut, weight0B16, ...)
```

### 3.5 groupSize=32 详解

**参数结构体**: `ParamsGroupSize32<xType, wType>`

gs=32 时，一个 group 只有 32 个 int4 = 16 字节。需要特殊处理：

```
// 每次处理 2*VEC_MAX_ELEM_B16 = 256 个 K 元素
// 分两次加载 weight（各 128 个 int4 = 64B）
weightB4Vl0 = LoadUnpack4(weightInBase)
weightB4Vl1 = LoadUnpack4(weightInBase + 64)

// scale/offset 使用 DIST_E2B_B16 加载（8个scale广播到8段）
// 然后 Interleave 拆成 scale0/scale1

// 分别计算并写出到 weightOutBaseAddr0 和 weightOutBaseAddr1
```

### 3.6 groupSize=256 详解

**参数结构体**: `ParamsGroupSize128And256<xType, wType>`

gs=256 时，一个 group 有 256 个 int4 = 128 字节 = 一个完整向量寄存器。需要两次加载：

```
weightB4  = LoadUnpack4(weightInBase + groupIdx*innerStrideWeight + ...)
weight1B4 = LoadUnpack4(weightInBase1 + groupIdx*innerStrideWeight + ...)
// innerStrideWeight = 2*64 = 128 bytes

// 分别 cast、add offset、mul scale
// 写出到 weightOutBaseAddr 和 weightOutBaseAddr1
```

---

## 4. AntiquantComputeKNPerGroup（KN排布，bTrans=false）

**文件**: `weight_quant_batch_matmul_v2_reg_base_common.h` L1136-1180

### 4.1 函数签名

```cpp
TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::AntiquantComputeKNPerGroup(int32_t bubNLen, int32_t bubKLen)
```

### 4.2 参数设置

```cpp
ParamsKN<xType> param;
param.kBub = bubKLen;
param.nBub = bubNLen;
param.nBubXTypeAlign = CeilAlign(nBub, BLOCK_CUBE);     // N 对齐到 16
param.nBubWTypeAlign = CeilAlign(nBub, ONE_BLK_W_NUM);   // N 对齐到 32
param.vfElemB16 = VEC_MAX_ELEM_B16;                       // 128
param.nBubTail = nBubWTypeAlign % VEC_MAX_ELEM_B16;
param.nLoop = CeilDiv(nBubWTypeAlign, VEC_MAX_ELEM_B16); // N 方向循环次数
param.groupSize = tiling_->groupSize;
param.groupNum = CeilDiv(kBub, groupSize) - 1;            // 完整 group 数（减1因为尾块单独处理）
param.groupTail = kBub % groupSize > 0 ? kBub % groupSize : groupSize;
param.dataBlockStride = CeilAlign(kBub, BLOCK_CUBE) + 1; // 输出步长
param.weightOutStride = dataBlockStride * vfElemB16 - bubKLen * BLOCK_CUBE;

// int4 特有步长
param.wNStride = VEC_MAX_ELEM_B16 >> INT4_DTYPE_PARAM;   // 128 >> 1 = 64 bytes
param.wKStride = nBubWTypeAlign >> INT4_DTYPE_PARAM;      // N 方向步长(bytes)
param.wGroupStride = groupSize * nBubWTypeAlign >> INT4_DTYPE_PARAM; // group 步长
```

### 4.3 VF 函数调用

```cpp
AscendC::VF_CALL<AntiquantPerGroupKN<xType, wType, hasAntiQuantOffset>>(param);
```

### 4.4 伪代码

```
// KN 排布: weight shape = (K, N), K 是外层, N 是内层
// 输出 shape: (k1, n1, n0, k0) 在 UB 上

FOR nIdx = 0 TO nLoop-1:           // N 方向按 VEC_MAX_ELEM_B16 分块
    maskWeight = UpdateMask(nBubXTypeAlign)

    FOR gIdx = 0 TO groupNum-1:    // 完整 group 循环
        // 加载 scale/offset: shape (k/gs, n) 的 ND 排布
        //   地址 = scaleBase + nIdx*128 + gIdx*nBubXTypeAlign
        scale  = LoadNorm(scaleBase, addrReg)
        offset = LoadNorm(offsetBase, addrReg)

        FOR kIdx = 0 TO groupSize-1:  // group 内 K 方向逐行
            // 加载 weight: KN 排布下地址 = base + nIdx*wNStride + gIdx*wGroupStride + kIdx*wKStride
            weightIn = LoadUnpack4(weightInBase, addrReg)
            weightOut = Cast(weightIn)

            IF hasOffset: weightOut += offset
            weightOut *= scale

            StoreBlockCopy(weightOutBase, weightOut, dataBlockStride, repeatStride)

    // 处理 groupTail（最后一个不完整的 group）
    scale  = LoadNorm(scaleTailAddr, addrReg)
    offset = LoadNorm(offsetTailAddr, addrReg)

    FOR kIdx = 0 TO groupTail-1:
        weightIn = LoadUnpack4(weightInGroupTailAddr, addrReg)
        weightOut = Cast(weightIn)
        IF hasOffset: weightOut += offset
        weightOut *= scale
        StoreBlockCopy(weightOutBase, weightOut, dataBlockStride, repeatStride)

    weightOutBase += weightOutStride  // N 块间跳转
```

**数据流图**：

```
                    UB Memory Layout (KN排布)
                    ========================

  weightIn (int4, stored as int8, shape = K × N):
  ┌─────────────────────────────────────────────┐
  │ k=0: [n0n1 n2n3 n4n5 ... n_{N-1}n_N]       │ ← wKStride = nBubWTypeAlign/2 (bytes)
  │ k=1: [n0n1 n2n3 n4n5 ... n_{N-1}n_N]       │
  │ ...                                         │
  │ k=gs-1: [...]                               │ ← 一个 group 结束
  │ k=gs: [n0n1 ...]                            │ ← wGroupStride 偏移
  │ ...                                         │
  └─────────────────────────────────────────────┘

  scale/offset (xType, shape = (groupNum, N)):
  ┌──────────────────────────────────┐
  │ g=0: [s(n=0..127)] [s(n=128..)] │ ← nBubXTypeAlign stride
  │ g=1: [s(n=0..127)] [s(n=128..)] │
  │ ...                              │
  └──────────────────────────────────┘

  weightOut (xType, NZ-like in UB):
  ┌──────────────────────────────────────────┐
  │ (n1, k1, k0, n0) format                 │
  │ dataBlockStride = CeilAlign(K,16)+1     │
  │ → 输出按 K 方向排列                      │
  └──────────────────────────────────────────┘
```

---

## 5. VF 底层函数实现详解

**文件**: `weight_quant_batch_matmul_v2_vf.h`

### 5.1 AntiquantPerGroupKN（KN 排布 VF 函数）

**签名**: `template <typename XType, typename WType, bool hasAntiquantOffset> AntiquantPerGroupKN(ParamsKN<XType>& param)`

**核心循环结构**：

```
外层循环: nIdx ∈ [0, nLoop)     — N 方向按 128 元素分块
  中层循环: gIdx ∈ [0, groupNum) — 完整 group 遍历
    内层循环: kIdx ∈ [0, groupSize) — group 内逐 K 行处理
```

**MicroAPI 调用序列**：

```cpp
// 1. 构造地址寄存器
AddrReg addRegScale = CreateAddrReg<XType>(nIdx, vfElemB16, gIdx, nBubXTypeAlign);

// 2. 加载 scale/offset（DIST_NORM = 直接搬运，不广播）
DataCopy<DIST_NORM>(offset, offsetBaseAddr, addRegScale);
DataCopy<DIST_NORM>(scale, scaleBaseAddr, addRegScale);

// 3. 加载 weight（DIST_UNPACK4_B8 = int4 unpack）
AddrReg addRegWeight = CreateAddrReg<int4x2_t>(nIdx, wNStride, gIdx, wGroupStride, kIdx, wKStride);
DataCopy<DIST_UNPACK4_B8>(weightIn, weightInBaseAddr, addRegWeight);

// 4. Cast int4 → xType
Cast<xType, int4x2_t, castTraitNorm>(weightOut, weightIn, maskWeight);

// 5. 反量化计算
Add(weightOut, weightOut, offset, maskWeight);  // if hasOffset
Mul(weightOut, weightOut, scale, maskWeight);

// 6. 写回 UB
DataCopy<DATA_BLOCK_COPY, POST_MODE_UPDATE>(weightOutBaseAddr, weightOut, dataBlockStride, repeatStride, maskWeight);
```

### 5.2 AntiquantW4Pergroup128NK（NK 排布 gs=128 VF 函数）

**签名**: `template <typename XType, typename WType, bool hasAntiquantOffset, bool useVag> AntiquantW4Pergroup128NK(ParamsGroupSize128And256<XType, WType>& p)`

**核心循环结构**：

```
外层循环: groupIdx ∈ [0, groupNum) — 按 group 遍历
  内层循环: nBubIdx ∈ [0, bubNLen) — N 方向逐行
```

**MicroAPI 调用序列**：

```cpp
// 1. 加载 scale/offset（DIST_BRC_B16 = 广播到整个 VREG）
AddrReg aregScale = CreateAddrReg<XType>(groupIdx, 1, nBubIdx, outerStrideScale);
DataCopy<DIST_BRC_B16>(scale, scaleBaseAddr, aregScale);

// 2. 加载 weight（DIST_UNPACK4_B8）
AddrReg aregWeight = CreateAddrReg<int4x2_t>(groupIdx, innerStrideWeight, nBubIdx, outerStrideWeight);
DataCopy<DIST_UNPACK4_B8>(weightB4, weightInBaseAddr, aregWeight);

// 3. Cast + 反量化
Cast<xType, int4x2_t, castTrait>(weightB16, weightB4, maskAll);
Add(weightB16, weightB16, offset, maskAll);  // if hasOffset
Mul(weightB16, weightB16, scale, maskAll);

// 4. 写回 UB
DataCopy<DATA_BLOCK_COPY, POST_MODE_UPDATE>(weightOutBaseAddr, weightB16, dataBlockStride, repeatStride, maskWeight);
```

### 5.3 关键 LoadDist 模式说明

| LoadDist | 含义 | 使用场景 |
|----------|------|----------|
| `DIST_NORM` | 直接搬运，src→dst 1:1 | KN 排布加载 scale/offset（N 方向已有连续数据） |
| `DIST_BRC_B16` | 广播单个值到整个 VREG | NK 排布加载 scale/offset（单个 scale 广播到 N 方向） |
| `DIST_BLK` | 读取 32B 广播到 256B | NZ 格式加载 scale |
| `DIST_E2B_B16` | 每 2B 扩展到 16B | gs=32 时加载 8 个 scale 并广播 |
| `DIST_UNPACK_B8` | int8 unpack: 1B→2B | int8 weight 加载 |
| `DIST_UNPACK4_B8` | int4 unpack: 0.5B→2B | int4 weight 加载 |

### 5.4 关键 StoreDist 模式说明

| StoreDist | 含义 | 使用场景 |
|-----------|------|----------|
| `DIST_NORM_B16` | 按地址寄存器写出 | NZ 格式写出 |
| `DATA_BLOCK_COPY` + `POST_MODE_UPDATE` | 按 block 写出，自动更新地址 | ND 格式写出到 UB |

---

## 6. NK vs KN 差异对比表

| 维度 | NK 排布 (bTrans=true) | KN 排布 (bTrans=false) |
|------|----------------------|----------------------|
| **Weight shape** | `(N, K)` — N 外层, K 内层 | `(K, N)` — K 外层, N 内层 |
| **调度函数** | `AntiquantComputeW4NKPerGroup` | `AntiquantComputeKNPerGroup` |
| **VF 函数** | 按 groupSize 分多个专用函数 | 统一使用 `AntiquantPerGroupKN` |
| **循环结构** | `group → n` 或 `inner → outer` | `n → group → k` |
| **scale/offset 加载** | `DIST_BRC_B16`（广播） | `DIST_NORM`（直接搬运） |
| **scale 地址模式** | `scaleBase[groupIdx + nIdx * stride]` | `scaleBase[nIdx * 128 + gIdx * nBubXTypeAlign]` |
| **weight 内层遍历** | K 方向一次性加载（gs 个元素） | K 方向逐行循环（groupSize 次） |
| **weight unpack** | `DIST_UNPACK4_B8` | `DIST_UNPACK4_B8` |
| **输出布局** | `(k1, n1, n0, k0)` | `(n1, k1, k0, n0)` |
| **dataBlockStride** | `CeilAlign(bubNLen, 16) + 1` | `CeilAlign(bubKLen, 16) + 1` |
| **groupSize 适配** | 按 gs 值分 6+ 个专用 VF 函数 | 统一处理，无分支 |
| **代码复杂度** | 高（多分支、多专用函数） | 低（统一逻辑） |
| **UB buffer 数** | 2 | 2 |

### 核心差异总结

1. **循环顺序不同**：
   - NK: 外层按 group 循环，内层按 N 循环 → scale 广播到 N 方向
   - KN: 外层按 N 循环，中层按 group 循环，内层按 K 逐行 → scale 直接加载

2. **weight 访问模式不同**：
   - NK: weight 在 K 方向连续，一次 unpack 整个 group 的数据
   - KN: weight 在 N 方向连续，K 方向需要逐行步进

3. **NK 按 groupSize 分支**：
   - 不同 groupSize 需要不同的数据加载和 scale 广播策略
   - KN 统一处理，因为 K 方向逐行步进，groupSize 只影响循环次数

---

## 7. NZ 格式 pergroup 对比参照

**文件**: `basic_block_vf_nz.h` (ops-nn)

### 7.1 AntiQuantInt4NzKnVf

NZ 格式的 pergroup 实现（`AntiQuantS8S4NzKnGroupVf`）与 ND 的关键差异：

| 维度 | NZ pergroup | ND pergroup |
|------|-------------|-------------|
| **scale 加载** | `DIST_BLK`（32B 广播到 256B） | `DIST_NORM` 或 `DIST_BRC_B16` |
| **scale 来源** | 从 UB 上 NZ 格式的 scale buffer | 从 UB 上 ND 格式的 scale buffer |
| **weight 布局** | NZ 格式 `(n1, k1, k0, n0)` | ND 格式 `(K, N)` 或 `(N, K)` |
| **输出方式** | `DIST_PACK_B16`（pack 后写出） | `DATA_BLOCK_COPY`（直接写出） |
| **输出类型** | int8（cast f16→s8） | xType（half/bf16 直接输出） |
| **group 遍历** | `loopGroupNum` 循环 | `groupNum` 循环 |

### 7.2 NZ pergroup 的 scale 加载方式

```cpp
// NZ pergroup: scale 在 UB 上已经是 NZ 格式
// 每个 n1 块有 loopGroupNum 个 group 的 scale
antiQuantScaleBasePhyAddr = base + loopN1Idx * C0_SIZE_B8 + loopGroupIdx * VEC_MAX_ELEM_B16;
DataCopy<DIST_BLK>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr);  // 32B → 256B 广播
DataCopy<DIST_BLK>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr + ONE_BLK_ELEM_B16);
Select(antiQuantScaleVreg, antiQuantScaleVreg, antiQuantScaleVreg1, maskSelect);
```

---

## 8. GMM basic_block_vf_nd.h 已有接口分析

**文件**: `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h`

### 8.1 已有函数清单

| 函数名 | 排布 | 数据类型 | 量化类型 | 说明 |
|--------|------|----------|----------|------|
| `AntiQuantFP8NdNkVf` | NK | FP8 | PER_TENSOR/PER_CHANNEL | FP8 ND NK 反量化 |
| `AntiQuantFP8NdKnVf` | KN | FP8 | PER_TENSOR/PER_CHANNEL | FP8 ND KN 反量化 |
| `AntiQuantB8CommonNdKn` | KN | int8 | PER_TENSOR/PER_CHANNEL | int8 ND KN 反量化 |
| `AntiQuantB8CommonNdNk` | NK | int8 | PER_TENSOR/PER_CHANNEL | int8 ND NK 反量化 |
| `AntiQuantInt4NdNk` | NK | int4 | PER_TENSOR/PER_CHANNEL | **int4 ND NK 反量化** |
| `AntiQuantInt4NdKn` | KN | int4 | PER_TENSOR/PER_CHANNEL | **int4 ND KN 反量化** |

### 8.2 已有 int4 ND 函数签名

```cpp
// NK 排布
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdNk(
    __local_mem__ xType *antiQuantScaleBasePhyAddr,
    __local_mem__ xType *antiQuantOffsetBasePhyAddr,
    __local_mem__ wType *weightLowBitPhyAddr0,
    __local_mem__ xType *weightF16PhyAddr0,
    xType scaleValue, xType offsetValue,
    uint64_t ubLoopN)

// KN 排布
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdKn(
    __local_mem__ xType *antiQuantScaleBasePhyAddr,
    __local_mem__ xType *antiQuantOffsetBasePhyAddr,
    __local_mem__ wType *weightLowBitPhyAddr0,
    __local_mem__ xType *weightF16PhyAddr0,
    xType scaleValue, xType offsetValue,
    uint64_t ubLoopK)
```

### 8.3 已有 int4 ND 实现逻辑

**AntiQuantInt4NdNk** (NK 排布)：

```cpp
// 循环结构: 按 N 方向循环
for (ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
    // 1. 加载 scale/offset（DIST_BRC_B16 广播）
    NdNkLoadScaleOffset(antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr, ...);

    // 2. 加载 int4 weight（DIST_UNPACK4_B8）
    //    每次处理 128 个 int4 = 64B
    DataCopy<DIST_UNPACK4_B8>(weightS4Vreg0, weightLowBitPhyAddr0 + ...);
    DataCopy<DIST_UNPACK4_B8>(weightS4Vreg1, weightLowBitPhyAddr1 + ...);

    // 3. Cast int4 → xType
    Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
    Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);

    // 4. 反量化
    AddMulWeightF16RegTensorNdNk(...);

    // 5. 写出到 NZ 格式 UB
    WeightF16NdRegTensorToNzUb(...);
}
```

**AntiQuantInt4NdKn** (KN 排布)：

```cpp
// scale/offset 预加载（KN 排布下 scale 在 N 方向连续）
NdKnLoadScaleOffset(...);  // DIST_NORM 直接加载两份

// 循环结构: 按 K 方向循环
for (ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
    // 1. 加载 int4 weight
    DataCopy<DIST_UNPACK4_B8>(weightS4Vreg0, weightLowBitPhyAddr0 + ...);
    DataCopy<DIST_UNPACK4_B8>(weightS4Vreg1, weightLowBitPhyAddr1 + ...);

    // 2. Cast + 反量化
    Cast(...);
    AddMulWeightF16RegTensorNdKn(...);

    // 3. 写出
    WeightF16NdRegTensorToNzUb(...);
}
```

### 8.4 关键差异：已有 GMM 函数 vs wqbmmv2 pergroup

| 维度 | GMM 已有 (PER_CHANNEL/PER_TENSOR) | wqbmmv2 pergroup |
|------|-----------------------------------|------------------|
| **scale shape** | `(N,)` perchannel 或标量 pertensor | `(k/gs, N)` pergroup |
| **scale 加载** | 每个 N 位置一个 scale 或全局标量 | 每个 group 每个 N 位置一个 scale |
| **循环结构** | 简单 N/K 单层循环 | 需要 group 维度循环 |
| **groupSize 适配** | 无 | 按 gs 值分支处理 |
| **参数传递** | `WqmmConfig` + `VecAntiQuantConfig` 编译期常量 | 运行时 tiling 参数 |
| **UB 管理** | 直接指针传递 | 通过 UB buffer 索引管理 |

---

## 9. 迁移适配方案

### 9.1 架构差异总结

| 维度 | wqbmmv2 | GMM |
|------|---------|-----|
| **命名空间** | `WeightQuantBatchMatmulV2::Arch35` | `WeightQuantBatchMatmulV2::Arch35`（相同） |
| **配置传递** | 运行时 `tiling_` 指针 + 模板参数 | 编译期 `WqmmConfig` + `VecAntiQuantConfig` 常量 |
| **UB 管理** | 类成员 `weightInUb_[]`, `weightOutUb_[]` 数组 + 索引 | 直接 `__local_mem__` 指针传递 |
| **VF 调用** | `VF_CALL<VfFunc>(params)` 在类方法中 | `__VEC_SCOPE__` 块内直接调用 MicroAPI |
| **参数结构体** | 定义在 `_vf.h` 中，按 groupSize 分多个 | 可复用或新增 |
| **buffer 配置** | 类成员 `vecWeightInSize_` 等 | `UbBufferInfo` 结构体 |

### 9.2 可复用部分

1. **VF 函数核心逻辑**：`AntiquantPerGroupKN` 和 `AntiquantW4Pergroup*NK` 的 MicroAPI 调用序列可直接复用
2. **参数结构体**：`ParamsKN`, `ParamsGroupSize128And256`, `ParamsGroupSize64`, `ParamsGroupSize32` 等可直接复用或微调
3. **Cast trait 常量**：`castTraitNorm`, `castTraitF162Bf16`, `CAST_S4_TO_F16_TRAIT` 等
4. **RegTensorActualT 模板**：int4b_t → int4x2_t 的映射

### 9.3 需要适配部分

#### 9.3.1 参数传递方式

**wqbmmv2 方式**：
```cpp
// 运行时从 tiling_ 读取
param.groupSize = tiling_->groupSize;
param.groupNum = CeilDiv(bubKLen, param.groupSize) - 1;
```

**GMM 适配方式**：
```cpp
// 方案 A：通过函数参数传递（推荐，灵活性高）
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdNkPerGroup(
    __local_mem__ xType *antiQuantScaleBasePhyAddr,   // shape: (groupNum, N)
    __local_mem__ xType *antiQuantOffsetBasePhyAddr,
    __local_mem__ wType *weightLowBitPhyAddr0,
    __local_mem__ xType *weightF16PhyAddr0,
    uint64_t ubLoopN,       // bubNLen
    uint64_t ubLoopK,       // bubKLen
    uint64_t groupSize,     // 从 tiling 传入
    uint64_t groupNumBub)   // scale 的 N 方向对齐长度

// 方案 B：扩展 WqmmConfig 增加 groupSize（编译期固定，性能更优但灵活性低）
```

#### 9.3.2 UB Buffer 地址计算

**wqbmmv2 方式**：
```cpp
weightInUbBaseAddr_ = weightInUb_[ubInBufIdx_ * vecWeightInSize_].GetPhyAddr();
weightOutUbAddr_ = weightOutUb_[ubOutBufIdx_ * vecWeightOutSize_].GetPhyAddr();
```

**GMM 适配方式**：
```cpp
// GMM 已有函数直接接收 __local_mem__ 指针
// 调用方（weight_quant_vec_compute.h 或 weight_quant_vcv_basic_block.h）负责计算地址
// 新增 pergroup 函数保持相同接口风格即可
```

#### 9.3.3 输出布局对齐

**wqbmmv2**：输出到 UB 使用 `DATA_BLOCK_COPY + POST_MODE_UPDATE`，布局为 `(k1, n1, n0, k0)` 或 `(n1, k1, k0, n0)`

**GMM**：已有函数使用 `WeightF16NdRegTensorToNzUb`，stride 为 `WEIGHT_F16_UB_NZ_STRIDE = 65`

**适配**：pergroup 函数的输出布局需要与 GMM 后续 cube 计算的 L1 输入布局对齐。需要确认 GMM 的 ND weight 反量化后在 UB 上的布局是否与 wqbmmv2 一致。

#### 9.3.4 groupSize 分支处理

**wqbmmv2**：在 `AntiquantComputeW4NKPerGroup` 中按 groupSize 运行时分支

**GMM 适配方案**：
```cpp
// 方案 A：运行时分支（与 wqbmmv2 一致）
__aicore__ inline void AntiQuantInt4NdNkPerGroup(...) {
    if (groupSize == 128) {
        // 调用 gs=128 专用 VF 函数
    } else if (groupSize == 64) {
        // 调用 gs=64 专用 VF 函数
    }
    // ...
}

// 方案 B：统一处理（类似 KN 排布的方式）
// 优点：代码简洁；缺点：可能损失性能
```

### 9.4 接口对齐方式

#### 新增函数签名建议

```cpp
// NK 排布 pergroup（对应 wqbmmv2 的 AntiquantComputeW4NKPerGroup）
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdNkPerGroup(
    __local_mem__ xType *antiQuantScaleBasePhyAddr,
    __local_mem__ xType *antiQuantOffsetBasePhyAddr,
    __local_mem__ wType *weightLowBitPhyAddr0,
    __local_mem__ xType *weightF16PhyAddr0,
    uint64_t ubLoopN,
    uint64_t ubLoopK,
    uint64_t groupSize,
    uint64_t groupNumBub);

// KN 排布 pergroup（对应 wqbmmv2 的 AntiquantComputeKNPerGroup）
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdKnPerGroup(
    __local_mem__ xType *antiQuantScaleBasePhyAddr,
    __local_mem__ xType *antiQuantOffsetBasePhyAddr,
    __local_mem__ wType *weightLowBitPhyAddr0,
    __local_mem__ xType *weightF16PhyAddr0,
    uint64_t ubLoopN,
    uint64_t ubLoopK,
    uint64_t groupSize);
```

#### 与已有函数的接口一致性

| 维度 | 已有 `AntiQuantInt4NdNk` | 新增 `AntiQuantInt4NdNkPerGroup` |
|------|--------------------------|----------------------------------|
| 模板参数 | `<xType, wType, wqmmConfig, vecConfig>` | 相同 |
| 前 4 个参数 | `scale, offset, weightIn, weightOut` | 相同 |
| 额外参数 | `scaleValue, offsetValue, ubLoopN` | `ubLoopN, ubLoopK, groupSize, groupNumBub` |
| 返回值 | void | void |

### 9.5 Buffer 配置适配

GMM 的 `GetNdBufferInfo()` 已有 ND 场景的 buffer 配置：

```cpp
// 当前 ND buffer 配置
weightInputLowbitUbTotalSize = 174 KB   // weight 低比特 UB 总大小
highBitDataUbTotalSize = 66 KB          // 反量化后高比特 UB 总大小
antiQuantScaleUbTotalSize = 4 KB        // scale UB 总大小
antiQuantOffsetUbTotalSize = 4 KB       // offset UB 总大小
ubWeightOutputHighBitBufferNum = 2      // 双缓冲
```

**pergroup 场景需要评估**：
- scale/offset buffer 是否需要扩大（pergroup 的 scale shape 为 `(k/gs, N)`，比 perchannel 的 `(N,)` 大）
- 当前 4KB scale buffer 是否足够（4KB = 2048 个 half，若 groupNum * N_bub ≤ 2048 则够用）

### 9.6 迁移步骤建议

```
Step 1: 在 basic_block_vf_nd.h 中新增 pergroup 函数
  ├── AntiQuantInt4NdNkPerGroup (NK 排布)
  └── AntiQuantInt4NdKnPerGroup (KN 排布)

Step 2: 复用 wqbmmv2 的 VF 函数核心逻辑
  ├── 复制 AntiquantPerGroupKN → 适配为 GMM 风格
  ├── 复制 AntiquantW4Pergroup128NK → 适配为 GMM 风格
  └── 按需添加其他 groupSize 的分支

Step 3: 在调用方（weight_quant_vec_compute.h）中增加 pergroup 分支
  └── 根据 wqmmConfig.antiQuantType == PER_GROUP 调用新函数

Step 4: 更新 buffer 配置（basic_block_config.h）
  └── 评估 pergroup 场景的 scale/offset buffer 大小

Step 5: 更新 VfConfig（basic_block_config.h）
  └── pergroup 场景的 vfNStandardLen / vfKStandardLen
```

---

## 附录 A：关键代码片段

### A.1 NK 排布 gs=128 完整 VF 函数

```cpp
template <typename XType, typename WType, bool hasAntiquantOffset, bool useVag>
__aicore__ inline void AntiquantW4Pergroup128NK(ParamsGroupSize128And256<XType, WType>& p)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
        MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    MicroAPI::RegTensor<XType> offset;
    MicroAPI::RegTensor<XType> scale;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightB4;
    MicroAPI::RegTensor<XType> weightB16;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);

    for (uint16_t groupIdx = 0; groupIdx < static_cast<uint16_t>(p.groupNum); ++groupIdx) {
        for (uint16_t nBubIdx = 0; nBubIdx < static_cast<uint16_t>(p.bubNLen); ++nBubIdx) {
            if constexpr (useVag) {
                // 使用地址寄存器
                MicroAPI::AddrReg aregScale = MicroAPI::CreateAddrReg<XType>(
                    groupIdx, 1, nBubIdx, p.outerStrideScale);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        offset, p.offsetBaseAddr, aregScale);
                }
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    scale, p.scaleBaseAddr, aregScale);

                MicroAPI::AddrReg aregWeight = MicroAPI::CreateAddrReg<
                    typename RegTensorActualT<WType>::T>(
                    groupIdx, p.innerStrideWeight, nBubIdx, p.outerStrideWeight);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T,
                    MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4, (__local_mem__ typename RegTensorActualT<WType>::T*)(
                        p.weightInBaseAddr), aregWeight);
            } else {
                // 直接地址偏移
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        offset, p.offsetBaseAddr + groupIdx +
                        nBubIdx * p.outerStrideScale);
                }
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    scale, p.scaleBaseAddr + groupIdx +
                    nBubIdx * p.outerStrideScale);

                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T,
                    MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4, (__local_mem__ typename RegTensorActualT<WType>::T*)(
                        p.weightInBaseAddr + groupIdx * p.innerStrideWeight +
                        nBubIdx * p.outerStrideWeight));
            }

            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(
                weightB16, weightB4, maskAll);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightB16, weightB16, offset, maskAll);
            }
            MicroAPI::Mul(weightB16, weightB16, scale, maskAll);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr, weightB16, p.dataBlockStride,
                p.repeatStride, maskWeight);
        }
        p.weightOutBaseAddr += p.weightOutAddrOffset;
        maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);
    }
}
```

### A.2 KN 排布完整 VF 函数

```cpp
template <typename XType, typename WType, bool hasAntiquantOffset>
__aicore__ inline void AntiquantPerGroupKN(ParamsKN<XType>& param)
{
    MicroAPI::RegTensor<XType> offset;
    MicroAPI::RegTensor<XType> scale;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightIn;
    MicroAPI::RegTensor<XType> weightOut;
    MicroAPI::MaskReg maskWeight = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::ALL>();
    uint32_t maskWeightValue = param.nBubXTypeAlign;

    for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(param.nLoop); ++nIdx) {
        maskWeight = MicroAPI::UpdateMask<XType>(maskWeightValue);
        for (uint16_t gIdx = 0; gIdx < static_cast<uint16_t>(param.groupNum); ++gIdx) {
            MicroAPI::AddrReg addRegScale = MicroAPI::CreateAddrReg<XType>(
                nIdx, param.vfElemB16, gIdx, param.nBubXTypeAlign);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_NORM>(
                    offset, param.offsetBaseAddr, addRegScale);
            }
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_NORM>(
                scale, param.scaleBaseAddr, addRegScale);

            for (uint16_t kIdx = 0; kIdx < static_cast<uint16_t>(param.groupSize); ++kIdx) {
                // int4 路径
                MicroAPI::AddrReg addRegWeight = MicroAPI::CreateAddrReg<
                    typename RegTensorActualT<WType>::T>(
                    nIdx, param.wNStride, gIdx, param.wGroupStride, kIdx, param.wKStride);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T,
                    MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightIn, (__local_mem__ typename RegTensorActualT<WType>::T*)(
                        param.weightInBaseAddr), addRegWeight);
                MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTraitNorm>(
                    weightOut, weightIn, maskWeight);

                if constexpr (hasAntiquantOffset) {
                    MicroAPI::Add(weightOut, weightOut, offset, maskWeight);
                }
                MicroAPI::Mul(weightOut, weightOut, scale, maskWeight);
                MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    param.weightOutBaseAddr, weightOut,
                    param.dataBlockStride, param.repeatStride, maskWeight);
            }
        }

        // groupTail 处理（最后一个不完整 group）
        MicroAPI::AddrReg addRegScaleTail = MicroAPI::CreateAddrReg<XType>(
            nIdx, param.vfElemB16);
        if constexpr (hasAntiquantOffset) {
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_NORM>(
                offset, param.offsetTailAddr, addRegScaleTail);
        }
        MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_NORM>(
            scale, param.scaleTailAddr, addRegScaleTail);

        for (uint16_t kIdx = 0; kIdx < static_cast<uint16_t>(param.groupTail); ++kIdx) {
            MicroAPI::AddrReg addRegWeightTail = MicroAPI::CreateAddrReg<
                typename RegTensorActualT<WType>::T>(
                nIdx, param.wNStride, kIdx, param.wKStride);
            MicroAPI::DataCopy<typename RegTensorActualT<WType>::T,
                MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightIn, (__local_mem__ typename RegTensorActualT<WType>::T*)(
                    param.weightInGroupTailAddr), addRegWeightTail);
            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTraitNorm>(
                weightOut, weightIn, maskWeight);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightOut, weightOut, offset, maskWeight);
            }
            MicroAPI::Mul(weightOut, weightOut, scale, maskWeight);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                param.weightOutBaseAddr, weightOut,
                param.dataBlockStride, param.repeatStride, maskWeight);
        }
        param.weightOutBaseAddr += param.weightOutStride;
    }
}
```
