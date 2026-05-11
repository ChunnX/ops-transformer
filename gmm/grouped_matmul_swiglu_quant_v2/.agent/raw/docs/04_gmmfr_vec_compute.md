# GMMFR VecCompute 层代码走读

## 概述

`GMMFRWeightQuantVecCompute` 是 GMMFR（GroupedMatmulFinalizeRouting）算子在 AIV（AI Vector）上的核心计算类，负责 **4bit 权重的反量化** 和 **Routing 后处理**（logits 乘加 + scatter add 写出）。该类运行在 Vector 核上，与 Cube 核上的矩阵乘计算形成流水线。

### 关键常量速查

| 常量 | 值 | 含义 |
|------|-----|------|
| `QUADRUPLE_BUFFER_NUM` | 4 | 四缓冲数量，用于 weightHighBit 输出 |
| `DOUBLE_BUFFER_NUM` | 2 | 双缓冲数量，用于 logits/rowIndex |
| `VECTOR_REG_WIDTH` | 256 (bytes) | 向量寄存器宽度 |
| `VEC_MAX_ELEM_B32` | 64 | 每个向量寄存器容纳的 float 数量 |
| `VEC_MAX_ELEM_B16` | 128 | 每个向量寄存器容纳的 half 数量 |
| `C0_SIZE_B8` | 32 | int8 场景下 Cube 基本块大小 |
| `BLOCK_CUBE` | 16 | Cube 基本块大小（half/float 场景） |
| `E2M1_SHIFT_RIGHT_SIZE` | 0x2 | MX 格式低 2bit 右移量 |
| `SHIFT_LEFT_SIZE` | 0x4 | MX 格式高 2bit 左移量 |
| `E2M1_AND_MASK` | 0x9C | MX 格式掩码（10011100b） |

---

## 1. GmmFrUbBufferInfo — UB Buffer 分配策略

### 1.1 结构体定义

```cpp
struct GmmFrUbBufferInfo {
    uint64_t weightHighBitBufferNum;       // weightHighBit 的 buffer 数量（=4）
    uint64_t weightHighBitTotalSize;       // weightHighBit UB 总大小
    uint64_t weightHighBitSingleBufferSize;// weightHighBit 单 buffer 大小
    uint64_t weightLowbitTotalSize;        // weightLowBit UB 总大小
    uint64_t weightLowBitSingleBufferSize; // weightLowBit 单 buffer 大小
    uint64_t biasTotalSize;                // bias UB 总大小
    uint64_t biasSingleBufferSize;         // bias 单 buffer 大小
    uint64_t biasRescaleTotalSize;         // bias 反量化后 UB 总大小
    uint64_t biasRescaleSingleBufferSize;  // bias 反量化后单 buffer 大小
    uint64_t logitsTotalSize;              // logits UB 总大小
    uint64_t logitsSingleBufferSize;       // logits 单 buffer 大小
    uint64_t rowIndexTotalSize;            // rowIndex UB 总大小
    uint64_t rowIndexSingleBufferSize;     // rowIndex 单 buffer 大小
};
```

### 1.2 分配详情表

通过 `GetGMMFRMxA8W4BufferInfo()` 计算，以 `int8_t` 为基准单位（`GetKBUnit<int8_t>() = 1024`）：

| Buffer | 总大小 | 单 Buffer 大小 | 缓冲数 | 说明 |
|--------|--------|---------------|--------|------|
| `weightLowBit_` | 80KB | 80KB / `ubMte2BufferNum` | `ubMte2BufferNum` (2 或 4) | 存放 MTE2 从 GM 搬运来的 4bit 权重 |
| `weightHighBit_` | 160KB | 160KB / 4 = 40KB | `QUADRUPLE_BUFFER_NUM` (4) | 存放 VF 反量化后的 8bit 权重，四缓冲 |
| `bias_` | 1KB | 1KB / `ubMte2BufferNum` | `ubMte2BufferNum` | 存放原始 bias（half 类型） |
| `biasRescale_` | 1KB | 1KB / `ubMte2BufferNum` | `ubMte2BufferNum` | 存放反量化后的 bias（half 类型） |
| `rowIndex_` | 2KB | 2KB / 2 = 1KB | `DOUBLE_BUFFER_NUM` (2) | 存放 scatter add 的行索引（int64） |
| `logits_` | 1KB | 1KB / 2 = 0.5KB | `DOUBLE_BUFFER_NUM` (2) | 存放 logits 值（float） |

**UB 总使用量**：80 + 160 + 1 + 1 + 2 + 1 = **245KB**，在 `UB_AVAILABLE_SIZE = 248KB` 范围内。

### 1.3 缓冲策略说明

- **weightHighBit 使用四缓冲（QUADRUPLE_BUFFER_NUM=4）**：反量化输出是 Cube 核的输入，需要深度流水。VF 计算、MTE3 搬运到 L1、Cube 计算三者通过四缓冲实现 overlap。
- **weightLowBit 使用 `ubMte2BufferNum` 缓冲**：动态配置为 2（默认）或 4（`VEC_ANTIQUANT_CONFIG_DYNAMIC`），控制 MTE2 搬运和 VF 消费之间的流水深度。
- **logits/rowIndex 使用双缓冲（DOUBLE_BUFFER_NUM=2）**：Routing 后处理阶段，MTE2 搬运下一批 logits/rowIndex 的同时 VF 处理当前批。

---

## 2. Init() — 初始化分析

```cpp
__aicore__ inline void Init(float sharedInputWeight, __gm__ yType *yFp32Addr)
{
    sharedInputWeight_ = sharedInputWeight;
    yFp32Global_.SetGlobalBuffer(yFp32Addr);

    // === UB Buffer 初始化（地址连续分配） ===
    weightLowBit_ = LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightLowbitTotalSize);
    uint64_t ubOffset = UB_BUFFER_INFO.weightLowbitTotalSize;
    weightHighBit_ = LocalTensor<xType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.weightHighBitTotalSize);
    ubOffset += UB_BUFFER_INFO.weightHighBitTotalSize;

    bias_ = LocalTensor<biasType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.biasTotalSize);
    ubOffset += UB_BUFFER_INFO.biasTotalSize * sizeof(biasType);
    biasRescale_ = LocalTensor<biasType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.biasRescaleTotalSize);
    ubOffset += UB_BUFFER_INFO.biasRescaleTotalSize * sizeof(biasType);
    rowIndex_ = LocalTensor<rowIndexType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.rowIndexTotalSize);
    ubOffset += UB_BUFFER_INFO.rowIndexTotalSize * sizeof(rowIndexType);
    logits_ = LocalTensor<logitsType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.logitsTotalSize);

    // === Event ID 初始化 ===
    // MTE3→V: 四缓冲，每个 buffer 对应一个 Event
    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
    }
    // S→MTE2 和 FR_V→MTE2: 双缓冲（Routing 阶段使用）
    for (uint16_t idx = 0; idx < DOUBLE_BUFFER_NUM; idx++) {
        SetFlag<HardEvent::S_MTE2>(EVENT_ID_S_TO_MTE2 + idx);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_FR_V_TO_MTE2 + idx);
    }
    // V→MTE2: ubMte2BufferNum 缓冲（weight 搬运阶段使用）
    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}
```

### 2.1 UB 内存布局

```
UB 地址空间 (248KB):
┌──────────────────────────────────────────────┐  0
│  weightLowBit_  (80KB)                       │
│  [buf0][buf1]  (ubMte2BufferNum=2 时)        │
├──────────────────────────────────────────────┤  80KB
│  weightHighBit_ (160KB)                      │
│  [buf0][buf1][buf2][buf3]  (四缓冲)           │
├──────────────────────────────────────────────┤  240KB
│  bias_ (1KB)  |  biasRescale_ (1KB)          │
├──────────────────────────────────────────────┤  242KB
│  rowIndex_ (2KB)                             │
├──────────────────────────────────────────────┤  244KB
│  logits_ (1KB)                               │
└──────────────────────────────────────────────┘  245KB
```

### 2.2 Event ID 分配表

| Event 名称 | 起始 ID | 数量 | 方向 | 用途 |
|-----------|---------|------|------|------|
| `EVENT_ID_MTE3_TO_V` | 0 | 4 | MTE3→V | weightHighBit 四缓冲：MTE3 搬运完成后通知 VF 可读 |
| `EVENT_ID_BIAS_FR_MTE3_TO_V` | 4 | 1 | MTE3→V | Routing 阶段：MTE3 scatter add 完成后通知 VF |
| `EVENT_ID_V_TO_MTE2` | 0 | 2/4 | V→MTE2 | weightLowBit：VF 消费完后通知 MTE2 可覆盖写入 |
| `EVENT_ID_FR_V_TO_MTE2` | 4 | 2 | V→MTE2 | Routing 阶段：VF 处理完 logits 后通知 MTE2 |
| `EVENT_ID_MTE2_TO_V` | 0 | 1 | MTE2→V | weightLowBit：MTE2 搬运完成后通知 VF 可读 |
| `EVENT_ID_RL_MTE2_TO_V` | 1 | 1 | MTE2→V | Routing 阶段：MTE2 搬运完 rowIndex/logits 后通知 VF |
| `EVENT_ID_V_TO_MTE3` | 0 | 1 | V→MTE3 | VF 反量化完成后通知 MTE3 可搬运到 L1 |
| `EVENT_ID_S_TO_MTE2` | 0 | 2 | S→MTE2 | Routing 阶段：Scalar 处理完成后通知 MTE2 |
| `EVENT_ID_MTE2_TO_S` | 0 | 2 | MTE2→S | Routing 阶段：MTE2 搬运完成后通知 Scalar |

---

## 3. CopyGmToUb() — GM 到 UB 的数据搬运

```cpp
__aicore__ inline void CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset, uint64_t kL1Offset,
                                   uint64_t biasGmOffset, uint64_t biasRealSize,
                                   const BasicBlockOffsetParam &offsetParam)
{
    // === Step 1: Bias 搬运（仅首次 kGmOffset==0 时） ===
    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            DataCopyPad2D(
                bias_[(weightMte2LoopIdx_ & QUADRUPLE_BUFFER_MASK) * UB_BUFFER_INFO.biasSingleBufferSize],
                biasGlobal_[biasGmOffset],
                1, biasRealSize, biasRealSize, biasRealSize);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        }
    }

    // === Step 2: 4bit Weight 搬运 ===
    if (offsetParam.nL1Size == 0 || ubMte2KSize == 0) return;

    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) * UB_BUFFER_INFO.weightLowBitSingleBufferSize]
            .template ReinterpretCast<wType>(),
        wGlobal_[(kGmOffset + kL1Offset) * offsetParam.nAlign + offsetParam.nOffset * C0_SIZE],
        CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),                          // blockCount
        CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,      // blockLen
        CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,      // dstInnerLength
        offsetParam.nAlign * C0_SIZE);                                                     // srcInnerLength

    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
}
```

### 3.1 DataCopyPad2D 参数含义（GM→UB 方向）

```cpp
DataCopyPad2D(dstLocal, srcGlobal, blockCount, blockLen, dstInnerLength, srcInnerLength)
```

| 参数 | 含义 | 本例取值 |
|------|------|---------|
| `blockCount` | 搬运的块数（K 方向循环次数） | `CeilDivide(ubMte2KSize, C0_SIZE)` |
| `blockLen` | 每块的数据长度（N 方向） | `CeilAlign(nL1Size, BLOCK_CUBE) * C0_SIZE` |
| `dstInnerLength` | UB 侧一行总长度（含 padding） | 同 blockLen（连续存放） |
| `srcInnerLength` | GM 侧一行总长度（含 stride） | `nAlign * C0_SIZE`（NZ 格式的 N 方向对齐长度） |

**搬运模式**：每次搬运 `blockLen` 个元素，连续搬运 `blockCount` 次。GM 侧按 `srcInnerLength` 跳行（NZ 格式的 stride），UB 侧连续存放。

### 3.2 搬运流程

```
GM (NZ格式):                          UB (连续):
┌──────────────────────────┐          ┌──────────────────┐
│ row0: [data0..dataN] ... │  ─MTE2→  │ [data0..dataN]   │  buf0
│ row1: [data0..dataN] ... │          │ [data0..dataN]   │
│ ...                      │          │ ...              │
│ rowK: [data0..dataN] ... │          └──────────────────┘
└──────────────────────────┘
  srcInnerLength = nAlign*C0           dstInnerLength = blockLen
```

---

## 4. AntiQuantProcessNzMxA8W4() — 反量化参数准备与调用

```cpp
__aicore__ inline void AntiQuantProcessNzMxA8W4(uint64_t biasRealSize, uint64_t ubMte2KSize,
                                                 uint64_t kGmOffset, const BasicBlockOffsetParam &offsetParam)
{
    MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams;
    uint64_t ubMte2BufferIdx = weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1);

    // N 方向对齐到 Cube 基本块
    mxA8W4NzParams.nRealSizeAlign = CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));

    // 输入：4bit 权重在 UB 中的物理地址
    mxA8W4NzParams.weightLowBitPhyAddr =
        (__ubuf__ wType *)weightLowBit_[ubMte2BufferIdx * UB_BUFFER_INFO.weightLowBitSingleBufferSize].GetPhyAddr();

    // 输出：反量化后 8bit 权重在 UB 中的物理地址（四缓冲）
    mxA8W4NzParams.weightHighBitPhyAddr =
        (__ubuf__ xType *)weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1))
                                         * VECTOR_REG_WIDTH].GetPhyAddr();

    // K 方向循环次数
    mxA8W4NzParams.loopKNum = CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE));

    // N 方向内层循环次数
    mxA8W4NzParams.innerLoopNum =
        CeilDivide(CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                   static_cast<uint64_t>(VECTOR_REG_WIDTH));

    // 输出跳写 stride（避免 UB bank 冲突）
    mxA8W4NzParams.innerDstStride = VECTOR_REG_WIDTH * UB_BUFFER_INFO.weightHighBitBufferNum; // 256 * 4 = 1024
    mxA8W4NzParams.loopKDstStride = mxA8W4NzParams.innerLoopNum * mxA8W4NzParams.innerDstStride;

    // Bias 参数（仅首次 kGmOffset==0 时处理）
    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            mxA8W4NzParams.biasInUbAddr = ...;
            mxA8W4NzParams.biasOutUbAddr = ...;
            mxA8W4NzParams.biasLoopNum = CeilDivide(biasRealSize, VEC_MAX_ELEM_B16);
            FrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, true>(mxA8W4NzParams);
        } else {
            FrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, false>(mxA8W4NzParams);
        }
    } else {
        FrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, false>(mxA8W4NzParams);
    }
}
```

### 4.1 参数含义汇总

| 参数 | 含义 |
|------|------|
| `nRealSizeAlign` | N 方向对齐到 BLOCK_CUBE(16) 后的实际大小 |
| `weightLowBitPhyAddr` | 4bit 权重在 UB 中的物理地址（输入） |
| `weightHighBitPhyAddr` | 反量化后 8bit 权重在 UB 中的物理地址（输出） |
| `loopKNum` | K 方向循环次数 = `CeilDivide(K, C0_SIZE_B8)` |
| `innerLoopNum` | N 方向内层循环次数 = `CeilDivide(N_aligned * C0_SIZE, VECTOR_REG_WIDTH)` |
| `innerDstStride` | 输出 N 方向 stride = `256 * 4 = 1024`（四缓冲跳写） |
| `loopKDstStride` | 输出 K 方向 stride = `innerLoopNum * innerDstStride` |
| `biasLoopNum` | Bias 处理循环次数 = `CeilDivide(biasRealSize, 128)` |

### 4.2 输出跳写示意图

```
weightHighBit_ UB 布局（四缓冲跳写）:
buf0: [N0_K0][gap][N1_K0][gap]...[N0_K1][gap][N1_K1][gap]...
buf1: [N0_K0][gap][N1_K0][gap]...[N0_K1][gap][N1_K1][gap]...
buf2: [N0_K0][gap][N1_K0][gap]...[N0_K1][gap][N1_K1][gap]...
buf3: [N0_K0][gap][N1_K0][gap]...[N0_K1][gap][N1_K1][gap]...
       ←── innerDstStride=1024 ──→
       ←────────── loopKDstStride ──────────→
```

跳写 `innerDstStride = VECTOR_REG_WIDTH * 4 = 1024` 的目的是避免同一 bank 内的连续访问冲突，通过将相邻 N 块分散到不同 buffer 的同一偏移位置来减少 bank conflict。

---

## 5. FrAntiQuantMxA8W4NzNkVf — VF 反量化指令详解

这是整个 VecCompute 层最核心的函数，实现 **MX 格式 4bit→8bit 反量化**。

### 5.1 完整实现

```cpp
template <typename xType, typename wType, typename biasType, bool hasBias>
__simd_vf__ inline void FrAntiQuantMxA8W4NzNkVf(MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams)
{
    // === Step 1: Bias 反量化（乘以 MX_BIAS_FACTOR = 0.015625） ===
    if constexpr (hasBias) {
        FrMxA8W4BiasCompute<biasType>(mxA8W4NzParams.biasLoopNum,
                                       mxA8W4NzParams.biasInUbAddr,
                                       mxA8W4NzParams.biasOutUbAddr);
    }

    // === Step 2: 准备常量寄存器 ===
    MicroAPI::RegTensor<int8_t> wShrReg, wShlReg, wAndReg, wLoad, wShl, wShr0, wShr1, wSel, wAnd;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, MaskPattern::ALL>();
    MicroAPI::MaskReg pregVsel = MicroAPI::CreateMask<uint16_t, MaskPattern::ALL>();

    // wShrReg = 2  (右移2位，提取高4bit中的低2bit)
    MicroAPI::Duplicate<int8_t, MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    // wShlReg = 4  (左移4位，提取低4bit)
    MicroAPI::Duplicate<int8_t, MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    // wAndReg = 0x9C (10011100b，MX E2M1 格式掩码)
    MicroAPI::Duplicate<int8_t, MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    // === Step 3: 双层循环处理每个 K/N 块 ===
    for (uint16_t loopKIdx = 0; loopKIdx < mxA8W4NzParams.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < mxA8W4NzParams.innerLoopNum; ++innerLoopIdx) {
            // 3a. 加载 4bit 权重（DIST_US_B8 模式）
            MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * mxA8W4NzParams.nRealSizeAlign) >> 1,
                innerLoopIdx, VECTOR_REG_WIDTH >> 1);
            MicroAPI::LoadAlign<uint8_t, LoadDist::DIST_US_B8>(
                (RegTensor<uint8_t> &)wLoad,
                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightLowBitPhyAddr,
                aregWeightB8In);

            // 3b. 位操作提取高/低 4bit
            MicroAPI::ShiftRight(wShr0, wLoad, wShrReg, preg);  // wLoad >> 2 → 高4bit的低2bit
            MicroAPI::ShiftLeft(wShl, wLoad, wShlReg, preg);    // wLoad << 4 → 低4bit移到高4bit
            MicroAPI::ShiftRight(wShr1, wShl, wShrReg, preg);   // wShl >> 2 → 低4bit归位
            MicroAPI::Select(wSel, wShr1, wShr0, pregVsel);     // 交错选择：wShr1(偶) wShr0(奇)
            MicroAPI::And(wAnd, wSel, wAndReg, preg);           // & 0x9C → MX E2M1 解码

            // 3c. 存储 8bit 结果（跳写避免 bank 冲突）
            MicroAPI::AddrReg aregWeightB8Out = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, mxA8W4NzParams.loopKDstStride,
                innerLoopIdx, mxA8W4NzParams.innerDstStride);
            MicroAPI::StoreAlign<uint8_t, StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightHighBitPhyAddr,
                (RegTensor<uint8_t> &)wAnd, aregWeightB8Out, preg);
        }
    }
}
```

### 5.2 DIST_US_B8 搬运模式详解

`DIST_US_B8`（Unpack Signed 8bit from 4bit）是 4bit→8bit 反量化专用的 Load 模式。

**物理含义**：从 UB 中读取连续的 4bit 数据，每个 4bit 值扩展为 8bit 存入向量寄存器的连续两个位置。

```
UB 中 4bit 物理布局（每个字节存 2 个 4bit 值）:
Byte0: [D0_lo | D0_hi]   Byte1: [D1_lo | D1_hi]   Byte2: [D2_lo | D2_hi]   Byte3: [D3_lo | D3_hi]
        4bit    4bit              4bit    4bit              4bit    4bit              4bit    4bit

DIST_US_B8 Load 后 Vn 中的 8bit 布局:
Vn: [D0_lo][D0_hi][D0_lo][D0_hi][D1_lo][D1_hi][D1_lo][D1_hi][D2_lo][D2_hi][D2_lo][D2_hi]...
     8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit
```

**地址计算**：`4bit 物理地址位移 = 逻辑索引 >> 1`，因为每个字节存 2 个 4bit 值。

### 5.3 MX E2M1 格式解码流程

MX（Microscaling）格式中，每个 8bit 权重实际编码了 2 个 E2M1（2bit 指数 + 1bit 尾数）子元素。解码过程：

```
原始 8bit 数据: [b7 b6 b5 b4 b3 b2 b1 b0]
                  └──高4bit──┘ └──低4bit──┘
                  sub1 (E2M1)  sub0 (E2M1)

Step 1: wLoad >> 2  → wShr0
  提取高 4bit 的低 2bit（sub1 的指数部分）

Step 2: wLoad << 4  → wShl
  将低 4bit 移到高 4bit 位置

Step 3: wShl >> 2   → wShr1
  将低 4bit 归位到高 4bit 的低 2bit（sub0 的指数部分）

Step 4: Select(wShr1, wShr0) → wSel
  交错选择：偶数位置取 wShr1（sub0），奇数位置取 wShr0（sub1）
  结果：每个 8bit 位置对应一个 E2M1 子元素

Step 5: wSel & 0x9C  → wAnd
  掩码 0x9C = 10011100b
  保留 bit7, bit4, bit3, bit2，清零其他位
  这是 MX E2M1 格式的标准解码掩码
```

### 5.4 Bias 反量化

```cpp
template <typename biasType>
__simd_callee__ inline void FrMxA8W4BiasCompute(uint16_t biasLoopNum,
                                                 __ubuf__ biasType *biasInUbAddr,
                                                 __ubuf__ biasType *biasOutUbAddr)
{
    static constexpr biasType MX_BIAS_FACTOR = static_cast<biasType>(0.015625f); // 1/64
    // 每个 bias 元素乘以 1/64，对应 MX 格式的 scale 因子
    for (uint16_t loopBiasIdx = 0; loopBiasIdx < biasLoopNum; ++loopBiasIdx) {
        Load biasInUbAddr → biasVreg
        biasVreg = biasVreg * MX_BIAS_FACTOR
        Store biasVreg → biasOutUbAddr
    }
}
```

`MX_BIAS_FACTOR = 0.015625 = 1/64`，对应 MX 格式中 6bit scale 的缩放因子。

---

## 6. WeightAntiQuantComputeNzNk() — 反量化完整流程

```cpp
__aicore__ inline void WeightAntiQuantComputeNzNk(
    uint64_t kRealSize, uint64_t biasRealSize, uint64_t kGmOffset,
    const LocalTensor<xType> &weightHighBitL1,
    const LocalTensor<biasType> &biasL1,
    const BasicBlockOffsetParam &offsetParam)
{
    // Step 1: 等待 MTE3 释放 weightHighBit buffer
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V +
        (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));

    // Step 2: VF 反量化计算
    AntiQuantProcessNzMxA8W4(biasRealSize, kRealSize, kGmOffset, offsetParam);

    // Step 3: 通知 MTE3 可以搬运，并等待 MTE3 完成
    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    // Step 4: UB→L1 搬运反量化后的 weightHighBit
    if (likely(kRealSize > 0)) {
        CopyWeightHighBitForAligned(offsetParam.nL1Size, kRealSize, weightHighBitL1);
    }

    // Step 5: UB→L1 搬运反量化后的 bias
    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            DataCopy(biasL1,
                     biasRescale_[(ubComputeLoopIdx_ & QUADRUPLE_BUFFER_MASK) *
                                  UB_BUFFER_INFO.biasRescaleSingleBufferSize],
                     biasRealSize);
        }
    }

    // Step 6: 释放 weightHighBit buffer 给下一轮 MTE3 搬运
    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V +
        (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));
    ubComputeLoopIdx_++;
}
```

### 6.1 完整流水线时序

```
时间 →

MTE2:  [搬 weightLowBit buf0] [搬 weightLowBit buf1] [搬 weightLowBit buf0] ...
           │Set MTE2→V              │                        │
VF:        │  [Wait MTE2→V]         │                        │
           │  [Wait MTE3→V]         │                        │
           │  [反量化 buf0→buf0']   │  [反量化 buf1→buf1']   │  [反量化 buf0→buf2'] ...
           │  [Set V→MTE3]          │                        │
MTE3:      │     [Wait V→MTE3]      │                        │
           │     [搬 buf0'→L1]      │     [搬 buf1'→L1]      │     [搬 buf2'→L1] ...
           │     [Set MTE3→V]       │                        │
```

### 6.2 CopyWeightHighBitForAligned — UB→L1 搬运

```cpp
__aicore__ inline void CopyWeightHighBitForAligned(
    uint64_t antiQuantRealN, uint64_t antiQuantRealK,
    const LocalTensor<xType> &weightHighBitL1)
{
    DataCopyParams params;
    // blockCount: 总搬运块数 = (K_aligned * N_aligned) / VEC_REG_ELEM
    params.blockCount = CeilAlign(antiQuantRealK, C0_SIZE) *
                        CeilAlign(antiQuantRealN, BLOCK_CUBE) / VEC_REG_ELEM;
    // blockLen: 每块长度（以 32B 为单位）
    params.blockLen = VEC_REG_ELEM / ONE_BLK_SIZE;
    // srcStride: UB 侧跳读（跳过其他 3 个 buffer 的数据）
    params.srcStride = (UB_BUFFER_INFO.weightHighBitBufferNum - 1) * params.blockLen;
    // dstStride: L1 侧连续存放
    params.dstStride = 0;

    DataCopy(weightHighBitL1,
             weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) * VEC_REG_ELEM],
             params);
}
```

**关键点**：`srcStride = 3 * blockLen`，因为 VF 反量化输出时使用了四缓冲跳写（`innerDstStride = 1024`），UB 中有效数据和无效数据交错存放。搬运到 L1 时需要跳过无效数据，只取当前 buffer 的有效部分。

```
UB 中四缓冲跳写布局:              L1 中连续布局:
buf0: [D0][--][D1][--][D2][--]    [D0][D1][D2]...
buf1: [--][D0][--][D1][--][D2]
buf2: ...
buf3: ...
      ↑srcStride=3*blockLen 跳过     ↑dstStride=0 连续
```

---

## 7. MulLogits() 和 RoutingYToGm() — Routing 后处理

### 7.1 MulLogits — Logits 乘加

```cpp
__aicore__ inline void MulLogits(const LocalTensor<float> &ubOutputF32Buffer,
                                  uint64_t mRealSize,
                                  const BasicBlockOffsetParam &offsetParam,
                                  uint64_t rlLoopIdx)
{
    if (mRealSize == 0) return;

    FrMulLogitsVf(static_cast<uint16_t>(mRealSize),
                  CeilDivide(offsetParam.nL1Size, VEC_MAX_ELEM_B32),
                  (__ubuf__ float *)logits_.GetPhyAddr((rlLoopIdx & 1) * UB_BUFFER_INFO.logitsSingleBufferSize),
                  (__ubuf__ float *)ubOutputF32Buffer.GetPhyAddr());
}
```

VF 实现：

```cpp
__simd_vf__ inline void FrMulLogitsVf(uint16_t mSize, uint64_t nLoopCnt,
                                       __ubuf__ float *logitsUbAddr,
                                       __ubuf__ float *yUbAddr)
{
    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        // 广播加载第 mIdx 个 token 的 logit 值
        MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(logits, logitsUbAddr + mIdx);

        for (uint16_t nLoopIdx = 0; nLoopIdx < nLoopCnt; ++nLoopIdx) {
            // 加载第 mIdx 个 token 的第 nLoopIdx 段输出
            MicroAPI::LoadAlign<float, LoadDist::DIST_NORM>(yFp32, yUbAddr + mIdx * 256 + nLoopIdx * VEC_MAX_ELEM_B32);

            // y = y * 64.0 * logit
            MicroAPI::Muls(yFp32, yFp32, 64.0f, maskAll);
            MicroAPI::Mul(yFp32, yFp32, logits, maskAll);

            // 存回原位置
            MicroAPI::StoreAlign<float, StoreDist::DIST_NORM_B32>(yUbAddr, yFp32, outAddrReg, maskAll);
        }
    }
}
```

**计算逻辑**：对每个 token 的每个 N 段输出，执行 `y = y * 64.0 * logits[m]`。`64.0` 是 MX 格式的 scale 补偿因子。

### 7.2 RoutingYToGm — Scatter Add 写出

```cpp
__aicore__ inline void RoutingYToGm(uint64_t mRealSize,
                                     const LocalTensor<float> &ubOutputF32Buffer,
                                     const BasicBlockOffsetParam &offsetParam,
                                     uint64_t rlLoopIdx)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    AscendC::SetAtomicAdd<float>();  // 开启 AtomicAdd 模式
    for (uint32_t mIdx = 0; mIdx < mRealSize; mIdx++) {
        int64_t mGmOffset = rowIndex_[(rlLoopIdx & 1) * UB_BUFFER_INFO.rowIndexSingleBufferSize].GetValue(mIdx);
        if (mGmOffset < 0) {         // 无效索引，跳过
            continue;
        }
        // Scatter Add: yFp32Global[mGmOffset * nSize + nOffset] += ubOutputF32Buffer[mIdx * 256]
        DataCopyPad2D(yFp32Global_[mGmOffset * offsetParam.nSize + offsetParam.nOffset],
                      ubOutputF32Buffer[mIdx * 256],
                      1, offsetParam.nL1Size, offsetParam.nL1Size, offsetParam.nL1Size);
    }
    AscendC::SetAtomicNone();        // 关闭 AtomicAdd 模式

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_BIAS_FR_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_BIAS_FR_MTE3_TO_V);
}
```

**关键机制**：

1. **AtomicAdd**：多个 expert 的输出可能 scatter 到同一个 token 位置，因此需要原子加。`SetAtomicAdd<float>()` 开启 MTE3 的原子加模式，此后的 `DataCopyPad2D` 写 GM 时会自动执行 atomic add 而非普通写。

2. **Scatter 寻址**：`rowIndex[mIdx]` 给出第 `mIdx` 个 token 在全局输出中的行索引。`mGmOffset < 0` 表示该 token 不属于当前 expert，跳过。

3. **同步**：`EVENT_ID_BIAS_FR_MTE3_TO_V` 确保 scatter add 全部完成后才通知 VF。

---

## 8. 同步 Event 管理

### 8.1 Event 配对关系总表

| 生产者 | 消费者 | Event | Set 位置 | Wait 位置 |
|--------|--------|-------|----------|-----------|
| MTE2 | VF | `MTE2_V(0)` | `CopyGmToUb` 末尾 | `CopyGmToUb` 末尾（同步等待） |
| VF | MTE2 | `V_MTE2(0..N-1)` | `SetVToMTE2` | `WaitVToMTE2`（下一轮 CopyGmToUb 前） |
| VF | MTE3 | `V_MTE3(0)` | `WeightAntiQuantComputeNzNk` 中 | `WeightAntiQuantComputeNzNk` 中（同步等待） |
| MTE3 | VF | `MTE3_V(0..3)` | `WeightAntiQuantComputeNzNk` 末尾 | `WeightAntiQuantComputeNzNk` 开头 |
| MTE2 | VF | `RL_MTE2_V(0..1)` | `SetMte2ToS` | `WaitMte2ToS` |
| MTE2 | Scalar | `MTE2_S(0..1)` | `SetMte2ToS` | `WaitMte2ToS` |
| Scalar | MTE2 | `S_MTE2(0..1)` | `SetSToMTE2` | `WaitSToMTE2` |
| VF | MTE2 | `FR_V_MTE2(0..1)` | `SetFrToMTE2` | `WaitFrVToMTE2` |
| MTE3 | VF | `MTE3_V(BIAS_FR)` | `RoutingYToGm` 末尾 | `RoutingYToGm` 末尾（同步等待） |

### 8.2 四缓冲同步流程（weightHighBit）

```
Buffer 索引: buf0    buf1    buf2    buf3    buf0    buf1    ...
             ┌──────┬──────┬──────┬──────┬──────┬──────┐
VF:          │ Wait │ Wait │ Wait │ Wait │ Wait │ Wait │
             │MTE3_V│MTE3_V│MTE3_V│MTE3_V│MTE3_V│MTE3_V│
             │ (0)  │ (1)  │ (2)  │ (3)  │ (0)  │ (1)  │
             │ 计算  │ 计算  │ 计算  │ 计算  │ 计算  │ 计算  │
             │ Set  │ Set  │ Set  │ Set  │ Set  │ Set  │
             │V_MTE3│V_MTE3│V_MTE3│V_MTE3│V_MTE3│V_MTE3│
             ├──────┼──────┼──────┼──────┼──────┼──────┤
MTE3:        │      │ Wait │ Wait │ Wait │ Wait │ Wait │
             │      │V_MTE3│V_MTE3│V_MTE3│V_MTE3│V_MTE3│
             │      │ UB→L1│ UB→L1│ UB→L1│ UB→L1│ UB→L1│
             │      │ Set  │ Set  │ Set  │ Set  │ Set  │
             │      │MTE3_V│MTE3_V│MTE3_V│MTE3_V│MTE3_V│
             │      │ (0)  │ (1)  │ (2)  │ (3)  │ (0)  │
             └──────┴──────┴──────┴──────┴──────┴──────┘
```

VF 计算 buf0 时，MTE3 在搬运上一轮的 buf3。通过四缓冲 + Event 同步，VF 计算和 MTE3 搬运完全 overlap。

### 8.3 End() — 收尾同步

```cpp
__aicore__ inline void End()
{
    // 等待所有 MTE3→V 完成（确保所有 weightHighBit 搬运到 L1）
    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
    }
    // 等待所有 V→MTE2 完成（确保所有 weightLowBit 被消费）
    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}
```

---

## 9. 完整数据流总结

```
GM (4bit NZ Weight)                    GM (Output F32)
      │                                      ▲
      │ MTE2                                 │ MTE3 (AtomicAdd)
      ▼                                      │
┌─────────────────┐                    ┌──────────────┐
│  weightLowBit_  │                    │ RoutingYToGm │
│  (UB, 80KB)     │                    │ (scatter add)│
└────────┬────────┘                    └──────▲───────┘
         │                                     │
         │ VF: FrAntiQuantMxA8W4NzNkVf         │
         │ 4bit → 8bit MX E2M1 解码             │
         ▼                                     │
┌─────────────────┐                    ┌──────────────┐
│  weightHighBit_ │                    │ MulLogits    │
│  (UB, 160KB)    │                    │ y *= 64*logit│
│  四缓冲跳写      │                    └──────▲───────┘
└────────┬────────┘                           │
         │ MTE3: CopyWeightHighBitForAligned   │
         │ UB → L1 (连续)                      │
         ▼                                     │
┌─────────────────┐                    ┌──────────────┐
│  weightHighBit  │                    │ Cube 输出    │
│  (L1)           │ ──── Cube ──────→  │ (UB, F32)    │
└─────────────────┘                    └──────────────┘
```

**流水线阶段**：
1. **MTE2 阶段**：从 GM 搬运 4bit 权重到 UB（`weightLowBit_`）
2. **VF 阶段**：MX E2M1 反量化（4bit→8bit），输出到 `weightHighBit_`（四缓冲跳写）
3. **MTE3 阶段**：从 UB 搬运反量化后的 8bit 权重到 L1（连续存放）
4. **Cube 阶段**：使用 L1 中的 8bit 权重进行矩阵乘
5. **Routing 阶段**：Cube 输出乘以 logits，通过 AtomicAdd scatter 到 GM 输出