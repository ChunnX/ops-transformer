# GmmSwiGluQuant MxA8W4 SRS

> 基于 `ops-transformer/gmm/common/cgmct/epilogue/block_epilogue_swiglu_mx_quant.h` 实现，覆盖 SwiGLU 计算与 MX Quantization 量化流程。

---

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档标题 | GmmSwiGluQuant MxA8W4 SRS |
| 迭代/特性名称 | GMMSwigluQuant MxA8W4 |
| 算子类型 | 融合算子 |
| 主领域目录 | FakeQuantization |
| 责任人 | 任泽涛 |
| 更新时间 | 2026-05-09 |
| 文档版本 | v1.0 |
| 关联 STC | 无                        |
| 关联里程碑/需求单 | 无 |

---

## 2. 需求背景

### 2.1 背景

某推理框架在部署大规模语言模型时，需要对 MoE 结构中的 GMM（Grouped Matrix Multiplication）+ SwiGLU 激活 + MX Quantization 进行算子融合，以降低访存开销并提升端到端推理性能。当前通路中 GMM、SwiGLU 与量化分步执行，中间结果多次写回 Global Memory，带宽瓶颈明显。

### 2.2 目标

- 提供 GMM + SwiGLU + MX Quantization 的融合 Kernel 实现，支持 A8W4（激活 float8_e4m3、权重量化到 float4_e2m1）精度配置。
- 输出为 MX 格式的 float8_e4m3 量化张量及对应 E8M0 scale。
- 精度标准双千分之一

### 2.3 范围

- **包含**：
  - GMM 矩阵乘（GroupList 分组）
  - SwiGLU 激活计算（Swish × Gate）
  - MX Quantization 三步流水线（MaxExp → Scale → Quantize）
  - Scale 布局转置（从紧凑的排布转为稀疏的）

---

## 3. 方案简述

**方案示意图**：

```
L0C Output (fp32 基本块为256 * 256)
  ├── l0cOutUbFirst_  ──┐  (Swish branch: W[:, 0:N/2])
  └── l0cOutUbSecond_ ──┘  (Gate branch: W[:, N/2:N])
         ↓
    VFDoSwigluAndQuantForMX
         ↓
    ┌─────────────────────────────────────┐
    │ 1. GmmSwigluVf: SwiGLU 计算          │
    │ 2. ComputeMaxExpVf: 最大指数提取       │
    │ 3. ComputeScaleVf: MX Scale 计算      │
    │ 4. ComputeDataForQuantTargetFp8Vf:    │
    │    数据归一化并转 FP8                  │
    │ 5. TransMxScaleLayoutVf: Scale 转置   │
    └─────────────────────────────────────┘
        ↓
    CopyOutputFromUb2Gm
        ↓
    CopyScaleFromUb2Gm
```

代码实现：

```cpp
__aicore__ inline void BlockEpilogueGMMSwigluQuant::VFDoSwigluAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
    __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
    uint16_t mSize, uint16_t nSize)
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint16_t OneRowRepeatTimes =
        CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(sizePerRepeat)); // 一行需要循环的次数，等于n/64
    uint32_t nSrcUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    uint32_t nDstUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE); // out 1B
    const float scalarOne = 1.0;
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();

    // ========== SwiGLU 计算 ==========
    // 输入: firstSrc (Swish branch), secondSrc (Gate branch)
    // 输出: gluResAddr (BF16)
    // 因为这次的N是128对齐的，因此一次值可能作64或128,这里面N方向是一次做64
    GmmSwigluVf(firstSrc, secondSrc, gluResAddr, mSize, nSize, 
                nSrcUbAligned, nDstUbAligned, OneRowRepeatTimes, sizePerRepeat);

    // ========== Quant 量化计算 ==========
    // 计算数据总量和循环次数
    uint32_t totalDataInUb = mSize * nDstUbAligned; // nDstUbAligned aligned by 64
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2);
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;
    
    // Step 1: ComputeMaxExp - 提取每32个数中的最大指数
    // 输入: gluResAddr (BF16)
    // 输出: maxExpAddr (uint16_t, 每32B一个max exponent)
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
    ComputeMaxExpVf(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum, 
                    vlForHalfNumber_, elementAfterReduce_);
    
    // Step 2: ComputeScale - 计算共享MX scale和half-scale
    // 输入: maxExpAddr
    // 输出: scaleDst (mxScale, E8M0), halfScaleLocalAddr (halfScale for data normalization)
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
    ComputeScaleVf(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, 
                   loopScaleNum, fpEmax_, vlForHalfNumber_);
    
    // Step 3: ComputeDataForQuant - 缩放数据并转换为FP8
    // 输入: gluResAddr (BF16), halfScaleLocalAddr
    // 输出: outputDst (FP8 quantized data)
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        // FP8 量化路径: BF16 -> FP32 -> FP8
        ComputeDataForQuantTargetFp8Vf(gluResAddr, halfScaleLocalAddr, outputDst,
                                       totalDataInUb, loopDataNum,
                                       vlForHalfNumber_, elementAfterReduce_);
    }
}
```



---

## 4. 数学定义与语义

### 4.1 SwiGLU

```
SwiGLU(x, gate) = Swish(x) ⊙ gate
Swish(x) = x ⊙ sigmoid(x) = x / (1 + exp(-x))
```

SwiGLU 需要两个矩阵乘法：
- `Y_left  = X · W_left`   (Swish branch)
- `Y_right = X · W_right`  (Gate branch)
- `SwiGLU(X) = Swish(Y_left) ⊙ Y_right`

### 4.2 MX Quantization

三步流水线：

```
gluRes_ (bf16)
  → ComputeMaxExp      : 每 32 个数中提取一个最大值
  → ComputeScale       : 计算共享 MX scale 和 half-scale
  → ComputeDataForQuant: 缩放数据并转换为 FP8
```

核心公式：

```
sharedExp = maxExp - fpEmax
mxScale   = sharedExp >> 7                    (E8M0 块缩放, 0-255)
halfScale = 0x7F00 - sharedExp
          = (254 - mxScale) << 7
          → 表示 2^((-mxScale + 254) - 127)
          → 表示 2^(127 - mxScale) 在标准 BF16 中
```

maxExp：表示的是32个值中的指数位最大的值

fpEmax：表示目标目标量化类型是（flaot8_e4m3）指数位最大的表达值即 `8<<7` 值为 0x0400

---

## 5. 输入输出规格

### 5.1 输入 — 张量参数

| 参数名 | 必选/可选 | dtype | format | shape / 存储语义 | 说明 |
| --- | --- | --- | --- | --- | --- |
| x | 必选 | float8_e4m3 | ND | [M, K] | 激活值输入 |
| x_scale | 必选 | float8_e8m0 | ND | [M, K/64, 2] | 激活值缩放因子 |
| group_list | 必选 | int64 | ND | [G] | 分组列表 |
| weight | 必选 | float4_e2m1 | FRACTAL_NZ | [G, N, K] | 权重输入（NZ 格式） |
| weight_scale | 必选 | float8_e8m0 | ND | [G, N, K/64, 2] | 权重缩放因子 |
| weight_assist_matrix | 可选 | 保留 | 保留 | 保留 | 保留参数（本期不使用） |
| bias | 可选 | bfloat16 | ND | [G, N] | 偏置 |
| smooth_scale | 可选 | 保留 | 保留 | 保留 | 保留参数（本期不使用） |

### 5.2 输入 — 标量 / 属性

| 参数名 | 类型 | 取值 / 约束 | 说明 |
| --- | --- | --- | --- |
| dequant_mode | int | {0} | 反量化模式 |
| dequant_dtype | int | {0} | 反量化数据类型 |
| quant_mode | int | {0} | 量化模式 |
| quant_dtype | int | {0} | 量化数据类型 |
| transpose_weight | bool | {false, true} | 是否转置权重 |
| group_list_type | int | {0, 1} | 0：累加和；1：分组大小 |
| tuning_config | ListInt | {0} | 调优配置 |

### 5.3 输出

| 输出名 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- |
| y | float8_e4m3 | ND | [M, N/2] | 量化输出 |
| y_scale | float8_e8m0 | ND | [M, N/128, 2] | 输出量化参数（MX E8M0 scale） |

---

## 6. 约束条件

| 约束项 | 条件 | 说明 / 校验失败表现 |
| --- | --- | --- |
| K 对齐 | K % 32 == 0 | 非32对齐被拦截 |
| N 对齐 | N % 128 == 0 | 非128对齐被拦截 |
| group_list | G >= 1 | 至少一组 |
| K 和 N 的大小 | K>0 ，N>0    | 要求K和N大于0 |

---

## 7. 交付标准

精度要求：双千分之一

交付范围：GroupedMatmulSwigluQuantV2 MxA8W4 kernel部分的swiglu 和 quant部分

---

## 8. 具体方案描述

### 8.1 UB 内存划分

| Buffer | 偏移量 | 大小 | 用途 |
| --- | --- | --- | --- |
| `weight8Bit_` | 0 KB | 64 KB | Packed 4-bit weight input |
| `weight16Bit_` | 64 KB | 128 KB | Anti-quantized fp8 weight |
| `yF32Buffer_` | 64 KB | 128 KB | MMA fixpipe output（复用 weightHighBit_ 空间） |
| `bias_` | 192 KB | 1 KB | Bias input |
| `biasRescale_` | 193 KB | 1 KB | Rescaled bias |
| `maxB16Buffer_` | 194 KB | 2 KB | Per-block max exponent（ComputeMaxExp）（给halfscale用的） |
| `scaleE8M0Buffer_` | 196 KB | 16 KB | MX E8M0 scales |
| `yFp8Buffer_` | 212 KB | 32 KB | Final quantized FP8 output |

**注意**：`weightHighBit_` 和 `yF32Buffer_` 共用同一块内存空间（64 KB ~ 192 KB），因为权重反量化完成后，该空间可被 MMA 输出复用。 

`yF32Buffer_`：因为后面的计算都是顺序的关系，这个空间会被复用，swiglue的result结果会放置在[0,32KB]的空间内；[32KB,33KB]的空间内存放MxScale；[33KB,34KB]存放halfScale，

### 8.2 执行流程

```
输入参数:
  - outputDst: 量化输出缓冲区 (FP8)
  - scaleDst: MX scale 输出缓冲区 (E8M0)
  - firstSrc: Swish branch 输入（来自 L0C）
  - secondSrc: Gate branch 输入（来自 L0C）
  - mSize: M 维度大小
  - nSize: N 维度大小

执行流程:
  1. SwiGLU 计算
     输入: firstSrc, secondSrc
     输出: gluRes_ (BF16)
     调用: GmmSwigluVf(...)

  2. 计算数据总量(量化参数 + scale的参数)
     totalDataInUb = mSize * nDstUbAligned
     totalScaleInUb = totalDataInUb / 32

  3. ComputeMaxExp (Step 1 of Quant)
     输入: gluRes_ (BF16)
     输出: maxExp_（每 32 个值一个max exponent）
     调用: ComputeMaxExpVf(...)

  4. ComputeScale (Step 2 of Quant)
     输入: maxExp_
     输出: scaleDst (mxScale), halfScale_ (halfScale)
     调用: ComputeScaleVf(...)

  5. ComputeDataForQuant (Step 3 of Quant)
     输入: gluRes_ (BF16), halfScale_
     输出: outputDst (FP8)
     调用: ComputeDataForQuantTargetFp8Vf(...)
     根据 DataTypeOut 类型选择 FP8 E4M3 或 E5M2 路径
```

### 8.3 关键逻辑 — SwiGLU 向量化实现

```cpp
__simd_vf__ inline void GmmSwigluVf(__ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                     __ubuf__ bfloat16_t *gluResAddr, uint16_t mSize, uint16_t nSize,
                                     uint32_t nSrcUbAligned, uint32_t nDstUbAligned,
                                     uint16_t OneRowRepeatTimes, uint16_t sizePerRepeat)
{
    const float scalarOne = 1.0;
    for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
        uint32_t elementNum = nSize;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
            AscendC::MicroAPI::RegTensor<bfloat16_t> verg7;
            AscendC::MicroAPI::RegTensor<float> swishInput, gateInput;
            AscendC::MicroAPI::RegTensor<float> verg1, verg2, verg3, verg4, verg5, verg6, swishOutput;

            uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;
            
            // 1. 加载 Swish branch 数据
            AscendC::MicroAPI::DataCopy(swishInput, firstSrc + l0cOutOffset);
            
            // 2. Swish 计算: x / (1 + exp(-x))
            AscendC::MicroAPI::Muls(verg2, swishInput, -(scalarOne), mask);  // -x
            AscendC::MicroAPI::Exp(verg3, verg2, mask);                       // exp(-x)
            AscendC::MicroAPI::Adds(verg4, verg3, scalarOne, mask);           // 1 + exp(-x)
            AscendC::MicroAPI::Div<float, &DIV_MODE>(swishOutput, swishInput, verg4, mask); // x / (1 + exp(-x))

            // 3. 加载 Gate branch 数据
            AscendC::MicroAPI::DataCopy(gateInput, secondSrc + l0cOutOffset);

            // 4. SwiGLU: Swish(x) * gate
            AscendC::MicroAPI::Mul(verg6, swishOutput, gateInput, mask);

            // 5. 转换为 BF16 并存储
            AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg7, verg6, mask);
            uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                gluResAddr + dstUbOffset, verg7, mask);
        }
    }
}
```

### 8.4 关键逻辑 — Quant 三步流水线

![](figure/获取最大指数位的bit位逻辑.png)

通过两轮Max从而实现32个值中取一个最大值的目的

以 M=128, N=256 的数据块为例，三步流水线数据转换如下：

**Step 1: ComputeMaxExp**

```
gluRes_ (BF16, 128×128 = 32KB)
┌─────────────────────────────────────────────────────────────┐
│ [32个数] [32个数] [32个数] [32个数] │  每行 4 个 Block        │
│    ↓        ↓        ↓        ↓                             │
│  max(0)   max(1)   max(2)   max(3)                          │
├─────────────────────────────────────────────────────────────┤
│              ...          ...          ...                  │
└─────────────────────────────────────────────────────────────┘
         ↓
maxExp_ (uint16_t, 128×4 = 512 个值 ≈ 1KB)
```

每 32 个数中提取一个最大值。输入为 BF16 的 gluRes_，输出为每 32 个数一个 max exponent（uint16_t）。

关键常量：`MAX_EXP_FOR_BF16 = 0x7f80`

```cpp
__simd_vf__ inline void ComputeMaxExpVf(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                         uint32_t totalCountInUB, uint16_t loopNum,
                                         uint32_t vlForHalfNumber, uint16_t elementAfterReduce)
{
    AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
    AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
    AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
    AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);  // 0x7F80

    AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
    AscendC::MicroAPI::MaskReg scaleMask1;
    AscendC::MicroAPI::UnalignReg u1;

    for (uint16_t i = 0; i < loopNum; i++) {
        scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
        
        // 加载两个交错块
        AscendC::MicroAPI::DataCopy<bfloat16_t, 
            AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
            vdExp0, vdExp1, srcAddr, vlForHalfNumber * 2);

        // 掩码提取指数位: data & 0x7F80
        AscendC::MicroAPI::And(vdExpExtract0, 
            (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16, scaleMask1);
        AscendC::MicroAPI::And(vdExpExtract1, 
            (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16, scaleMask1);

        // 逐元素最大指数(将vdExpExtract0 和 vdExpExtract1 对应位置比较取最大值)
        AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
        
        // 归约到每32B块一个最大值(针对16bit的数据就是16个数中取最大值)
        AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

        // 存储到 maxExp_ 缓冲区
        AscendC::MicroAPI::DataCopyUnAlign<uint16_t, 
            AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            maxExpAddr, vdMaxExp, u1, elementAfterReduce);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
}
```

**Step 2: ComputeScale**

```
maxExp_ (uint16_t, 512 个值)
┌─────────────────────────────────────────────────────────────┐
│  max(0) max(1) max(2) max(3)                                │
│    ↓      ↓      ↓      ↓                                   │
│  scale  scale  scale  scale  (E8M0, 每个 Block 1 Byte)      │
│  hScale hScale hScale hScale (BF16 格式 half-scale)         │
├─────────────────────────────────────────────────────────────┤
│              ...          ...          ...                  │
└─────────────────────────────────────────────────────────────┘
         ↓
mxScale_   (E8M0,  128×4  = 512 Byte   = 512B)
halfScale_ (BF16, 128×4 = 512×2B    = 1KB)
```

计算共享 MX scale 和 half-scale。

核心常量：
```cpp
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;      // 254 << 7
constexpr int16_t  SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
```

核心公式：
```
mxScale   = sharedExp >> 7                    (E8M0 块缩放, 0-255)
halfScale = 0x7F00 - sharedExp
          = (254 - mxScale) << 7
```

推理maxExp - fpEmax 去替代maxExp/fp8Emax：

> fp8Emax：表示的是float8_e4m3所能表示的指数位所能表示的最大值为8，即 2^(15-7)，bias为7，15为float8_e4m3所能表示的最大值

maxExp/fp8Emax其中maxExp 是bf16, fpEmax 是float8_e4m3，所以可以表示为$2^{((E-127)-(15-7))}$所以可以表示位$2^{((E-8) - 127))}$所以可以表示位 $(E-8)<<7$ 也就是 maxExp - fpEmax，其中fpEmax 为 $8<<7$ ，maxExp是$E<<7$ 。

| Target Format | `fpEmax_` Value | `fpEmax_ >> 7` | Meaning                             |
| ------------- | --------------- | -------------- | ----------------------------------- |
| FP8 E4M3      | `0x0400`        | `8`            | BF16 exponent = 8 steps above bias  |
| FP8 E5M2      | `0x0780`        | `15`           | BF16 exponent = 15 steps above bias |
| FP4 E2M1      | `0x0100`        | `2`            | BF16 exponent = 2 steps above bias  |
| FP4 E1M2      | `0x0000`        | `0`            | BF16 exponent = 0 steps above bias  |

代码实现：

```cpp
__simd_vf__ inline void ComputeScaleVf(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale, uint16_t fpEmax, uint32_t vlForHalfNumber)
{
    AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
    AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);      // 0x7F80
    AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
    AscendC::MicroAPI::MaskReg cmpResult, zeroMask, invalidDataMask, specialDataMask, preMaskScale;
    AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
    AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax);            // 目标格式最大值
    AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);       // 0x7F00
    AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8); // 0x00FF
    AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
    AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION); // 0x7F81
    AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD); // 0x0040

    for (uint16_t i = 0; i < loopNumScale; i++) {
        preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
        
        // 加载 maxExp
        AscendC::MicroAPI::DataCopy<uint16_t, 
            AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            vdMaxExp, maxExpAddr, vlForHalfNumber);

        // 检测 INF/NAN 和 zero
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
            cmpResult, vdMaxExp, expMask, preMaskScale);  // INF/NAN检测
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
            zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);  // zero检测

        // 限制 maxExp 到目标格式最大值
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(
            invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(
            vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

        // sharedExp = maxExp - fpEmax
        // 只是做一个指数位相减取替代原本的除法，但是因为只有指数位其实是缺少了一个bias即减去127的
        AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);

        // mxScale = sharedExp >> 7 (E8M0格式)
        AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

        // 存储 mxScale (E8M0)
        AscendC::MicroAPI::DataCopy<uint16_t, 
            AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
            mxScaleLocalAddr, scaleValue, vlForHalfNumber >> 1, preMaskScale);

        // 处理特殊情况
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(
            specialDataMask, sharedExp, scaleBias, preMaskScale);

        // halfScale = 0x7F00 - sharedExp = (254 - mxScale) << 7
        AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

        // 存储 halfScale
        AscendC::MicroAPI::DataCopy<uint16_t, 
            AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            halfScaleLocalAddr, halfScale, vlForHalfNumber, preMaskScale);
    }
}
```



**Step 3: ComputeDataForQuantTargetFp8**

![](figure/cast的数据变化逻辑.png)

```
gluRes_ (BF16)        halfScale_ (BF16)
┌──────────────┐      ┌──────────────┐
│ [32个数]     │  ×   │  hScale(0)   │   逐 Block 归一化
│ [32个数]     │  ×   │  hScale(1)   │
│     ...      │  ×   │     ...      │
│ [32个数]     │  ×   │  hScale(n)   │
└──────────────┘      └──────────────┘
         ↓
    BF16 → FP32 → FP8 (castTrait32to8 饱和转换)
         ↓
outputDst (FP8 E4M3, 128×128 Byte = 16KB)
```

缩放数据并转换为 FP8。路径为 BF16 → FP32 → FP8，使用 `castTrait32to8` 进行饱和转换。

**提取的 ComputeDataForQuantTargetFp8 VF 函数**：

```cpp
__simd_vf__ inline void ComputeDataForQuantTargetFp8Vf(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum, uint32_t vlForHalfNumber, uint16_t elementAfterReduce)
{
    uint32_t totalCountInUB2 = totalCountInUB * 2;
    using T = bfloat16_t;
    
    AscendC::MicroAPI::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
    AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
    AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
    AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
    AscendC::MicroAPI::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;

    // 定义转换特征
    static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    for (uint16_t i = 0; i < loopNum; i++) {
        dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
        dataMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
        dataMask3 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
        dataMask4 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);

        // 加载BF16 SwiGLU结果(交错取值)
        // vdExp0：0 2 4 6 8 .....
        // vdExp1: 1 3 5 7 9 .....
        AscendC::MicroAPI::DataCopy<T, 
            AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
            vdExp0, vdExp1, srcAddr, vlForHalfNumber * 2);

        // 加载 half-scale 将数据扩充16倍(倒数乘数)
        AscendC::MicroAPI::DataCopy<uint16_t, 
            AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
            halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);

        // 归一化: data * 2^(127 - mxScale)
        // halfScaleForMul 每个参数同vdExp0和vdExp1的各16个数进行乘法
        AscendC::MicroAPI::Mul(vdExp0, vdExp0, 
            (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
        AscendC::MicroAPI::Mul(vdExp1, vdExp1, 
            (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);

        // 重排数据(将vdExp0和vdExp1的交错取值还原回顺序排布)
        AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);

        // BF16 → FP32(当源操作数和目标操作数的位宽为1：2或2：1时可以使用castTraitZero和castTraitOne)
        // vdExp0: 0 1 0 1 0 1 0 1 
        // vdExp0FP32Zero: 取0索引的位置的数
        // vdExp0FP32One:  取1索引的位置的数
        AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
        AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
        AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);

        // FP32 → FP8 
        AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(
            vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
        AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(
            vdExp0FP8One, vdExp0FP32One, dataMask3);

        // 对 vdExp1 同样处理
        AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
        AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
        AscendC::MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
        AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(
            vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
        AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(
            vdExp1FP8One, vdExp1FP32One, dataMask4);

        // 存储打包的FP8
        AscendC::MicroAPI::DataCopy<int8_t, 
            AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero, 
            OUT_ELE_NUM_ONE_BLK, dataMask3);
        // ... 类似存储 vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One
        AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
        AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
        AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
    }
}
```

### 8.5 关键逻辑 — Scale 布局转置

将 scale 从 `(mSize * 8)` 布局转置为 `(mSize, 32)` 布局，以匹配下游消费格式。

```cpp
__simd_vf__ inline void TransMxScaleLayoutVf(__ubuf__ int8_t *quantScaleOutputInUbAddr,
                                              __ubuf__ int8_t *quantScaleBlockOutputInUbAddr,
                                              uint16_t mSize, uint16_t scaleBlockN)
{
    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        uint32_t elemNum = scaleBlockN;
        AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(elemNum);
        AscendC::MicroAPI::RegTensor<int8_t> vreg0;
        AscendC::MicroAPI::UnalignReg u0, u1;
        
        auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcUb);
        AscendC::MicroAPI::DataCopyUnAlign(vreg0, u0, srcUb);
        
        auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
        AscendC::MicroAPI::DataCopy<int8_t, 
            AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
            dstUb, vreg0, maskScaleN);
    }
}
```

### 8.6 代码片段

见第 8.3 节。完整实现参考：
- `ops-transformer/gmm/common/cgmct/epilogue/block_epilogue_swiglu_mx_quant.h`
- 核心函数：`VFDoSwigluAndQuantForMX`（第 521-585 行）

### 8.7 数据搬运

将量化后的值搬出

```cpp
__aicore__ inline void CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset，AscendC::GlobalTensor<int8_t>&quantOutputGlobal)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.blockLen = singleN_ * sizeof(int8_t);
    // n_:在GM上一行的长度，singleN_此次搬出数据一行的长度
    ub2GmParams.dstStride = (n_ - singleN_) * sizeof(int8_t);
    AscendC::DataCopyPad(quantOutputGlobal[offset], yFp8Buffer_, ub2GmParams);
}

```

将Mx scale 量化参数搬出

```cpp
__aicore__ inline void CopyScaleFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::GlobalTensor<int8_t>&quantScaleGlobal_)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    // sacle 的搬出需要向上64对齐,shap [M,N/64,2],但此次N轴128对齐，经过swiglu后依然保持64对齐，满足此要求
    auto blockScaleN =
        CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(64)) * 2;
    ub2GmParams.blockLen = blockScaleN * sizeof(int8_t);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (scaleN_ - blockScaleN) * sizeof(int8_t);
    AscendC::DataCopyPad(quantScaleGlobal_[offset], scaleE8M0Buffer_, ub2GmParams);
}

```



---

## 9. 修订记录

| 版本 | 日期 | 作者 | 说明 |
| --- | --- | --- | --- |
| v1.0 | 2026-05-09 | 任泽涛 | GMMSQ初稿 |

---

