# GMMSQ算子kernel实现设计说明书

## 总体设计

> 整体的代码实现可以参考[grouped_matmul_finalize_routing](../../../../grouped_matmul_finalize_routing/)算子，核心差异点只在于后处理部分本gmmsq需要将N分两半处理，并进行SwiGLU+Quant操作。
> - Tiling参考代码: `gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.cpp`
> - Kernel参考代码：`gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block`

GMMSQ（GroupedMatmulSwigluQuant）MxA8W4数据流的Kernel实现基于GMMFR（GroupedMatmulFinalizeRouting）的MxA8W4数据流进行扩展。核心架构采用**AIC（Cube）+ AIV（Vector）异构协同**设计：

- **AIC（Cube核）**：负责执行矩阵乘（MMA）计算，包括A矩阵和反量化后的权重矩阵的乘法，以及MX量化的Scale应用
- **AIV（Vector核）**：负责权重的反量化（4bit→8bit）、SwiGLU激活计算、以及最终的MX量化输出

整体数据流：
```
GM输入(float8_e4m3 x, float4_e2m1 weight, float8_e8m0 scale) 
  → AIV反量化(weight: fp4→fp8) 
  → AIC矩阵乘(带MX Scale)
  → AIV SwiGLU激活
  → AIV MX量化输出
  → GM输出(float8_e4m3 output + float8_e8m0 outputScale)
```

## Tiling - 拦截与校验

Tiling层负责在Host侧进行算子参数的校验和场景识别。MxA8W4场景的校验逻辑如下：

### 场景识别条件

```cpp
// 伪代码：MxA8W4场景识别
bool InferScenario() {
    // 必须同时满足以下条件才识别为MxA8W4-WEIGHT-NZ场景：
    // 1. x数据类型为FLOAT8_E4M3FN
    // 2. weight数据类型为FLOAT4_E2M1
    // 3. weight格式为FRACTAL_NZ或FRACTAL_NZ_C0_32
    // 4. scale数据类型为FLOAT8_E8M0
    // 5. perTokenScale数据类型为FLOAT8_E8M0
    if (xDtype == DT_FLOAT8_E4M3FN && 
        wDtype == DT_FLOAT4_E2M1 &&
        (wFormat == FORMAT_FRACTAL_NZ || wFormat == FORMAT_FRACTAL_NZ_C0_32) &&
        scaleDtype == DT_FLOAT8_E8M0 && 
        perTokenScaleDtype == DT_FLOAT8_E8M0) {
        return MX_A8W4_WEIGHT_NZ;
    }
}
```

### 属性校验

```cpp
// 伪代码：属性校验
bool CheckMxA8W4NzAttrPtr() {
    // 1. transpose_x必须为false
    assert(transpose_x == false);
    
    // 2. transpose_w必须为true
    assert(transpose_w == true);
    
    // 3. groupListType必须为0或1
    assert(groupListType == 0 || groupListType == 1);
    
    // 4. dtype必须为0（float32）
    assert(dtype == 0);
    
    // 5. offset必须为nullptr
    assert(offset == nullptr);
}
```

### Shape校验

```cpp
// 伪代码：Shape校验流程
bool CheckMxA8W4InputShape() {
    // 1. 校验x shape: [M, K]
    assert(xDimNum == 2);
    mSize = xShape[0];
    kSize = xShape[1];
    
    // 2. 校验w shape (FRACTAL_NZ格式)
    // 原始shape: [E, N, K]
    // 存储shape: [E, K/32, N/16, 16, 32] (C0_32格式)
    assert(wStorageDimNum == 5);
    nSize = wOriginShape[1];
    kFromW = wOriginShape[2];
    
    // 3. 校验K一致性
    assert(kSize == kFromW);
    
    // 4. 校验K和N对齐
    assert(kSize % 32 == 0);  // K维度32对齐
    assert(nSize % 32 == 0);  // N维度32对齐
    
    // 5. 校验最小值
    assert(kSize >= 32);
    assert(nSize >= 32);
    
    // 6. 校验group_list shape匹配E
    assert(groupListShape[0] == eFromW);
    
    // 7. 校验row_index shape: [M]
    assert(rowIndexShape[0] == mSize);
    
    // 8. 校验logit shape: [M]
    assert(logitShape[0] == mSize);
    
    // 9. 校验bias shape (如果存在): [E, N]
    if (hasBias) {
        assert(biasShape[0] == eFromW);
        assert(biasShape[1] == nSize);
    }
    
    // 10. 校验scale shape: [E, N, CeilDiv(K, 64), 2]
    assert(scaleShape[0] == eFromW);
    assert(scaleShape[1] == nSize);
    assert(scaleShape[2] == CeilDiv(kSize, 64));
    assert(scaleShape[3] == 2);
    
    // 11. 校验pertoken_scale shape: [M, CeilDiv(K, 64), 2]
    assert(pertokenScaleShape[0] == mSize);
    assert(pertokenScaleShape[1] == CeilDiv(kSize, 64));
    assert(pertokenScaleShape[2] == 2);
}
```

## Tiling - tiling data定义与tiling切分

### Tiling Data定义

```cpp
// 伪代码：Tiling Data结构
struct GMMFinalizeRoutingWeightQuantTilingData {
    uint32_t coreNum;        // AIC核心数
    int64_t groupNum;        // 专家数E
    int64_t kSize;           // K维度大小
    int64_t nSize;           // N维度大小
    int64_t outputBs;        // 输出batch size
    int64_t initSize;        // 初始化大小
    int64_t sharedInputOffset; // 共享输入偏移
    int64_t sharedInputLen;    // 共享输入长度
    float sharedInputWeight;   // 共享输入权重
    uint32_t groupListType;  // 分组列表类型
    bool hasBias;            // 是否有bias
    uint32_t groupSize;      // MX分组大小 = 32
};
```

### Tiling切分逻辑

```cpp
// 伪代码：Tiling切分
ge::graphStatus DoOpTiling() {
    // 设置基本tiling数据
    tilingData.coreNum = compileInfo.aicNum;
    tilingData.groupNum = inputParams.groupNum;
    tilingData.kSize = inputParams.kSize;
    tilingData.nSize = inputParams.nSize;
    tilingData.groupSize = MX_BLOCK_SIZE;  // 32
    
    // 计算initSize
    // initSize = (outputBS - sharedInputLen) * nSize
    tilingData.initSize = (inputParams.outputBS - inputParams.sharedInputLen) * inputParams.nSize;
    
    // 设置其他参数
    tilingData.hasBias = inputParams.hasBias;
    tilingData.groupListType = inputParams.groupListType;
    
    return SUCCESS;
}

// Tiling Key生成
uint64_t GetTilingKey() {
    // 当前支持: ATRANS=0, BTRANS=1
    return GET_TPL_TILING_KEY(0, 1, hasBias);
}
```

### 动态K维度切分

```cpp
// 伪代码：动态K维度选择
uint64_t ChooseKbL1Size(uint64_t mL1Size, uint64_t nL1Size) {
    // 小tile → 更大的K buffer（更多K复用）
    // 大tile → 更小的K buffer（更多M/N buffer空间）
    if (mL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD &&
        nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) {
        return 512;  // 512
    } else {
        return 256;  // 256
    }
}
```

## kernel - buffer分配

### L1 buffer分配

> 参考gmmfr分配方案，开DB

``` plain
带bias：
L1 Address Space (0 ~ 512 KB):
├─ [0x00000, 0x10000)  WeightL1_P0        64 KB
├─ [0x10000, 0x11000)  Bias_P0             4 KB
├─ [0x11000, 0x19000)  ScaleAL1_P0        32 KB
├─ [0x19000, 0x21000)  ScaleBL1_P0        32 KB
├─ [0x21000, 0x40000)  AL1_P0            124 KB
├─ [0x40000, 0x5F000)  AL1_P1            124 KB
├─ [0x5F000, 0x67000)  ScaleBL1_P1        32 KB
├─ [0x67000, 0x6F000)  ScaleAL1_P1        32 KB
├─ [0x6F000, 0x70000)  Bias_P1             4 KB
└─ [0x70000, 0x80000)  WeightL1_P1        64 KB
```

``` plain
无bias：
L1 Address Space (0 ~ 512 KB):
├─ [0x00000, 0x10000)  WeightL1_P0        64 KB
├─ [0x10000, 0x18000)  ScaleAL1_P0        32 KB
├─ [0x18000, 0x20000)  ScaleBL1_P0        32 KB
├─ [0x20000, 0x40000)  AL1_P0            128 KB
├─ [0x40000, 0x60000)  AL1_P1            128 KB
├─ [0x60000, 0x68000)  ScaleBL1_P1        32 KB
├─ [0x68000, 0x70000)  ScaleAL1_P1        32 KB
└─ [0x70000, 0x80000)  WeightL1_P1        64 KB
```

### L0 buffer分配

| Buffer | Position | Size | Purpose |
|---|---|---|---|
| L0A	| A2	| 64 KB	| A-matrix (activation) tile |
| L0B	| B2	| 64 KB	| B-matrix (weight) tile |
| L0C	| CO1	| 256×256 floats	| MMA accumulation |
| BiasTable	| C2	| 4 KB	| Bias | lookup table |

### UB buffer分配

> 存在3种划分，GMM划分方法、GMMFR划分方法，第三种是针对GMMSQ适配的划分:

对于GMMSQ来说还需要额外的空间是:

- yF32保存gmm fp32输出结果
- maxB16 32个指数部分最大值
- scaleE8M0 结果的buffer，搬出到GM（存在性能瓶颈点，搬运小包scatter可能搬运性能很差，单独分配个空间尝试优化性能）
- yFp8 结果（单独一个buffer，避免y写出GM阻塞下一轮计算）

| Buffer | Offset | Size | Purpose |
|--------|--------|------|---------|
| `weightLowBit_` | 0 | 64 KB | Packed 4-bit weight input |
| `weightHighBit_` | 64 KB | 128 KB | Anti-quantized fp8 weight |
| `yF32Buffer_` | 64 KB | 128 KB | MMA fixpipe output (reuses weightHighBit_ space) |
| `bias_` | 192 KB | 1 KB | Bias input |
| `biasRescale_` | 193 KB | 1 KB | Rescaled bias |
| `maxB16Buffer_` | 194 KB | 2 KB | Per-block max exponent (ComputeMaxExp) |
| `scaleE8M0Buffer_` | 196 KB | 16 KB | MX E8M0 scales |
| `yFp8Buffer_` | 212 KB | 32 KB | Final quantized FP8/FP4 output |

## kernel 计算流

### 总体架构

GMMSQ的Kernel计算采用**AIC + AIV异构协同**模式，核心流程如下：

```
初始化阶段 (Init)
  ├── AIC侧: 初始化L1 buffer分配，设置AIC→AIV同步信号
  └── AIV侧: 初始化UB buffer，设置MTE2/MTE3同步事件

计算阶段 (ComputeBasicBlock)
  ├── AIV路径: IterateNzNkWithAiv
  │   ├── Pass 1: 预加载前K chunk到UB进行反量化
  │   │   ├── MTE2: GM→UB搬运4bit weight
  │   │   ├── V: 反量化计算(fp4→fp8)
  │   │   ├── MTE3: UB→L1搬运反量化后的weight
  │   │   └── 设置AIC同步信号
  │   ├── Phase B: 处理前一个basic block的SwiGLU+Quant输出（如果存在）
  │   │   ├── 设置AIC fixpipe信号
  │   │   ├── 等待AIC fixpipe完成
  │   │   ├── V: SwiGLU计算（由其他团队实现）
  │   │   ├── V: MX量化计算（由其他团队实现）
  │   │   └── MTE3: 写出量化结果到GM
  │   └── Pass 2: 继续加载剩余K chunk
  │       └── 同Pass 1流程
  └── AIC路径: IterateNzNkWithKAic
      ├── 等待AIV反量化完成信号
      ├── MTE1→MTE2: 搬运Scale到L1
      ├── MTE1: 搬运A矩阵到L1
      ├── MTE2→MTE1: L1→L0加载（带MX Scale）
      ├── M: MMA矩阵乘计算
      └── 设置AIV同步信号

结束阶段 (End)
  ├── AIC侧: 最后一次fixpipe，同步结束
  └── AIV侧: 处理最后一个block，清理同步信号
```

### AIV路径详细流程

```cpp
// 伪代码：AIV路径主循环
void IterateNzNkWithAiv(BasicBlockOffsetParam& curParam, BasicBlockOffsetParam& lastParam) {
    uint64_t kMte2Offset = 0;
    uint64_t curCvLoopIdx = cvLoopIdx_;
    
    // ===== Pass 1: 预加载前K chunk =====
    uint64_t preLoadLimit = Min(curParam.kSize, DOUBLE_BUFFER_NUM * curParam.kbL1Size);
    for (; kMte2Offset < preLoadLimit; kMte2Offset += curParam.kbL1Size, cvLoopIdx_++) {
        
        // 1. 等待前一个V→MTE2完成
        WaitFlag<V_MTE2>(EVENT_ID_V_TO_MTE2);
        
        // 2. MTE2: GM→UB搬运4bit weight
        DataCopyPad2D(weightLowBit_, wGlobal_, ...);
        
        // 3. 等待AIC就绪
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        
        // 4. V: 反量化计算(fp4→fp8)
        WaitFlag<MTE3_V>(EVENT_ID_MTE3_TO_V);
        AntiQuantProcessNzMxA8W4(biasRealSize, kRealSize, kGmOffset, curParam);
        
        // 5. Barrier: 确保V计算完成
        SetFlag<V_MTE3>(EVENT_ID_V_TO_MTE3);
        WaitFlag<V_MTE3>(EVENT_ID_V_TO_MTE3);
        
        // 6. MTE3: UB→L1搬运反量化后的weight
        if (kRealSize > 0) {
            CopyWeightHighBitForAligned(curParam.nL1Size, kRealSize, weightL1_);
        }
        if (hasBias) {
            DataCopy(biasL1_, biasRescale_, biasRealSize);
        }
        
        // 7. 设置MTE3→V完成信号
        SetFlag<MTE3_V>(EVENT_ID_MTE3_TO_V);
        ubComputeLoopIdx_++;
        
        // 8. 设置AIC同步信号
        SetAivToAic<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        
        // 9. 设置V→MTE2完成信号
        SetFlag<V_MTE2>(EVENT_ID_V_TO_MTE2);
        ubMte2LoopIdx_++;
    }
    
    // ===== Phase B: 处理前一个basic block的SwiGLU+Quant =====
    if (curCvLoopIdx > 0) {
        // 10. 设置AIC fixpipe信号
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        
        // 11. 等待AIC fixpipe完成
        WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);
        
        // 12. V: SwiGLU计算（由其他团队实现）
        // VFDoSwiglu(gluRes_, l0cOutUbFirst_, l0cOutUbSecond_, ...);
        
        // 13. V: MX量化计算（由其他团队实现）
        // ComputeMaxExp(gluRes_, maxB16Buffer_, ...);
        // ComputeScale(maxB16Buffer_, scaleE8M0Buffer_, ...);
        // ComputeDataForQuantTargetFp8(gluRes_, halfScaleLocalAddr, yFp8Buffer_, ...);
        
        // 14. MTE3: 写出量化结果到GM
        CopyOutputFromUb2Gm(lastParam.mSize, yOffset, yFp8Buffer_);
        CopyScaleFromUb2Gm(lastParam.mSize, yScaleOffset, scaleE8M0Buffer_);
        
        // 15. 设置MTE3→V完成信号
        SetFlag<MTE3_V>(EVENT_ID_MTE3_TO_V);
        computeLoopIdx_++;
    }
    
    // ===== Pass 2: 继续加载剩余K chunk =====
    for (; kMte2Offset < curParam.kSize; kMte2Offset += curParam.kbL1Size, cvLoopIdx_++) {
        // 同Pass 1的1-9步
        WaitFlag<V_MTE2>(EVENT_ID_V_TO_MTE2);
        DataCopyPad2D(weightLowBit_, wGlobal_, ...);
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        WaitFlag<MTE3_V>(EVENT_ID_MTE3_TO_V);
        AntiQuantProcessNzMxA8W4(biasRealSize, kRealSize, kGmOffset, curParam);
        SetFlag<V_MTE3>(EVENT_ID_V_TO_MTE3);
        WaitFlag<V_MTE3>(EVENT_ID_V_TO_MTE3);
        CopyWeightHighBitForAligned(curParam.nL1Size, kRealSize, weightL1_);
        if (hasBias) {
            DataCopy(biasL1_, biasRescale_, biasRealSize);
        }
        SetFlag<MTE3_V>(EVENT_ID_MTE3_TO_V);
        ubComputeLoopIdx_++;
        SetAivToAic<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        SetFlag<V_MTE2>(EVENT_ID_V_TO_MTE2);
        ubMte2LoopIdx_++;
    }
}
```

### AIC路径详细流程

```cpp
// 伪代码：AIC路径主循环
void IterateNzNkWithKAic(BasicBlockOffsetParam& curParam, BasicBlockOffsetParam& lastParam) {
    
    // Step 1: 如果存在前一个block，先fixpipe输出
    if (cvLoopIdx_ > 0) {
        WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        GetTensorC(ubOutputF32Buffer_, lastParam);
        SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
    }
    
    // Step 2: 等待AIV反量化完成
    WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
    
    // Step 3: K循环
    for (uint64_t kbL1Offset = 0; kbL1Offset < curParam.kSize; 
         kbL1Offset += curParam.kbL1Size, cvLoopIdx_++) {
        
        uint64_t kbL1RealSize = (kbL1Offset + curParam.kbL1Size) >= curParam.kSize ?
                                curParam.kSize - kbL1Offset : curParam.kbL1Size;
        
        // 3.1 搬运Scale到L1 (MTE1→MTE2)
        cubeCompute_.WaitScaleMTE1ToMTE2(kbL1Offset);
        cubeCompute_.CopyMxScaleGmToL1(curParam, kbL1Offset);
        
        // 3.2 搬运A矩阵到L1
        cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curParam);
        cubeCompute_.CopyAAndBiasGmToL1(curParam, kbL1Offset, cvLoopIdx_);
        
        // 3.3 L1 padding（如果K不对齐）
        cubeCompute_.PadL1IfNotAlign(weightL1_, weightL1DbOffset_, cvLoopIdx_, kbL1RealSize, curParam.nL1Size);
        
        // 3.4 等待AIV信号
        WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
        
        // 3.5 启动MMA计算
        cubeCompute_.LaunchMatmul(
            weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_], 
            kbL1Offset, kbL1RealSize, cvLoopIdx_, curParam);
        
        // 3.6 设置同步信号
        cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curParam);
        cubeCompute_.SetScaleMTE1ToMTE2(kbL1Offset, curParam);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }
}
```

### 反量化流程

```cpp
// 伪代码：4bit反量化到8bit
void AntiQuantProcessNzMxA8W4(uint64_t biasRealSize, uint64_t ubMte2KSize, 
                                uint64_t kGmOffset, BasicBlockOffsetParam& offsetParam) {
    // 准备反量化参数
    MxA8W4NzParams params;
    params.nRealSizeAlign = CeilAlign(offsetParam.nL1Size, BLOCK_CUBE);
    params.weightLowBitPhyAddr = weightLowBit_.GetPhyAddr();
    params.weightHighBitPhyAddr = weightHighBit_.GetPhyAddr();
    params.loopKNum = CeilDivide(ubMte2KSize, C0_SIZE);
    params.innerLoopNum = CeilDivide(CeilAlign(offsetParam.nL1Size, BLOCK_CUBE) * C0_SIZE, VECTOR_REG_WIDTH);
    params.innerDstStride = VECTOR_REG_WIDTH * QUADRUPLE_BUFFER_NUM;
    params.loopKDstStride = params.innerLoopNum * params.innerDstStride;
    
    // 执行反量化VF指令
    // 核心逻辑：将4bit数据unpack到8bit
    // DIST_US_B8模式：
    // Vn: 0 1 2 3 4 5 6 7
    // Vd: 0 1 0 1 2 3 2 3 4 5 4 5 6 7 6 7
    FrAntiQuantMxA8W4NzNkVf(params);
}
```

### SwiGLU划分计算

GMM + SwiGLU的gated是通过weight(N, K)划分完成的，N轴划分为两半：

- `W[0:N/2, :]` 前一半用于Swish计算
- `W[N/2:N, :]` 后一半作为gate完成GLU计算

两部分计算共用一个X，可以一起完成：`X[:M, :] @ W[:N, :].T = O[:M, :N]`。对输出结果 `O[:, 0:N/2]` 为Swish分支，`O[:, N/2:N]` 为gate分支。如果基本块选择baseN=256，那么为了完成计算必须同时计算得到Swish和gate结果 `256*256*2*4`，超出UB空间。

为了解决这个问题，可以每个将整个N划分两半，每次取baseN=128构成256的基本块，保证GMM的性能。

> 尾块计算需要按照一半的N计算尾块

## 同步机制

### 核心同步信号

| Flag ID | Direction | Pipe | Purpose |
|---------|-----------|------|---------|
| `SYNC_AIC_AIV_FLAG = 9` | AIC → AIV | MTE1 | Weight ready in L1 |
| `SYNC_AIV_AIC_FLAG = 8` | AIV → AIC | MTE3 | Anti-quantized weight ready |
| `SYNC_AIV_MTE3_AIC_FIX_FLAG = 4` | AIV → AIC | MTE3 | Request fixpipe L0C→UB |
| `SYNC_AIC_FIX_AIV_VF_FLAG = 3` | AIC → AIV | V / FIX | UB output ready for VF |

### Pipe Event定义

```cpp
// Cube侧Event ID
cubeEventIdsMte1ToMte2_[DOUBLE_BUFFER_NUM];    // MTE1→MTE2同步
cubeEventIdsMxScaleMte1ToMte2_[DOUBLE_BUFFER_NUM]; // Scale MTE1→MTE2同步

// Vector侧Event ID
EVENT_ID_V_TO_MTE2 = 0;        // V→MTE2同步
EVENT_ID_MTE2_TO_V = 0;        // MTE2→V同步
EVENT_ID_MTE3_TO_V = 0;        // MTE3→V同步
EVENT_ID_V_TO_MTE3 = 0;        // V→MTE3同步
```

## NOTE

- N是128对齐，是因为N轴切两半，一半为输出另一半为gate，所以实际N是64对齐
- tiling按照256*256切分
- SwiGLU和Quant部分由其他团队成员分工完成
- SwiGLU划分计算部分考虑到quant计算效率，后续可能会更改tiling划分为按singleCore大小切分，每个core所计算的数据在一起，而不是N方向每128个元素一个core负责的分核方式。
    - 当前方案：N方向每128个元素分一个core，每个core搬出quant后结果大小为128/32=4，搬运pattern性能可能较差
    - 后续可能优化方案：N方向singleCore大小都由一个core完成，尽可能增大scatter粒度提升性能
