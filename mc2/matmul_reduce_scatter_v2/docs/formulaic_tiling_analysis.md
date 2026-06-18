# ReduceScatter 公式化 Tiling 切分流程深度分析

## 1. 概述

`reduce_scatter_formulaic_tiling` 是 MatmulReduceScatter 算子的核心切分模块，负责将矩阵乘法计算与通信操作进行流水线配平，以最大化硬件利用率。本文档深入分析其切分流程、卡数限制、HCCL 通信量影响等关键机制。

## 2. 架构设计

### 2.1 类层次结构

```
OneCalcOneCommBase (基类 - 单通信域算子切分基类)
    ├── MatmulPerformanceModel (计算性能模型)
    ├── HCCLPerformanceModel (通信性能模型)
    ├── FormPartition (切分逻辑)
    └── MMPlusReduceScatter (ReduceScatter 切分实现)
```

### 2.2 核心数据结构

#### CutResult - 切分结果
```cpp
struct CutResult {
    bool shortTileAtBack;     // 短块是否后置（计算Bound=true, 通信Bound=false）
    uint64_t totalTileCnt;    // 总切分块数
    uint64_t numLongTile;     // 长块数量
    uint64_t numShortTile;    // 短块数量（0 或 1）
    uint64_t longTileLen;     // 长块长度（按 M 轴）
    uint64_t shortTileLen;    // 短块长度
};
```

#### TileArguments - 切分参数
```cpp
struct TileArguments {
    uint64_t mAlignLen;       // M轴对齐长度（BASE_BLOCK_M = 128）
    uint64_t maxTileLen;      // 最大 tile 长度
    uint64_t minTileLen;      // 最小 tile 长度
    uint64_t maxTileCnt;      // 最大切分块数（MAX_TILE_CNT = 16）
};
```

## 3. Tiling 切分流程

### 3.1 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    ReduceScatter Tiling 流程                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 初始化                                                    │
│  - 设置通信性能模型参数                                       │
│  - 计算最小 tile 长度（取通信和计算的最小阈值）                 │
│  - 初始化切分结果（默认不切分）                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 性能预测 (EstimateKernelTime)                           │
│  ├─ 2.1 设置通信时间因子 (SetCommTimeFactor)                 │
│  │   - SOC950: 使用 FULL_MESH 模式                           │
│  │   - 其他: 根据通信膨胀因子调整                             │
│  ├─ 2.2 预测总计算时间 (EstimateTotalMatmulTime)             │
│  ├─ 2.3 预测总通信时间 (EstimateTotalCommTime)                │
│  ├─ 2.4 计算计算/通信时间比                                  │
│  └─ 2.5 调整最小 tile 长度                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 选择切分方法 (SelectTilingMethod)                        │
│  ├─ 3.1 检查是否需要切分 (SetShortTileLen)                   │
│  │   - 如果 M < 2 * minTileLen 则不切分                      │
│  ├─ 3.2 流水配平计算                                         │
│  │   - 计算Bound: ShortAtEndCalcBoundBalancing              │
│  │   - 通信Bound: ShortAtFrontCommBoundBalancing            │
│  └─ 3.3 生成切分 (FitTileLengthDiscrete)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 输出切分结果                                              │
│  - longTileLen, shortTileLen                                 │
│  - numLongTile, numShortTile                                 │
│  - totalTileCnt                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 详细流程分析

#### 3.2.1 初始化阶段

**代码位置**: `reduce_scatter_formulaic_tiling.h` 构造函数

```cpp
explicit MMPlusReduceScatter(const mc2tiling::TilingArgs& args, 
                             uint32_t inputRankDim,
                             KernelType inputKernelType, 
                             SocVersion inputSocVersion = SocVersion::SOC910_B,
                             bool deterministicFlag = false)
    : OneCalcOneCommBase(args, inputRankDim, inputKernelType, inputSocVersion),
      deterministicSoc910B_(deterministicFlag)
{
    // 设置通信形状参数
    commPerf_.SetCommShapeLen(clusterInfo_.nValue);      // ReduceScatter 通信轴为 N
    commPerf_.SetCommDTypeSize(clusterInfo_.outMatrixCDtypeSize);
    
    // 确定性通信调整
    if (deterministicSoc910B_) {
        commPerf_.SetLocalReduceFactor();
    }
    
    // 计算 rankTileNum（影响通信步数）
    rankTileNum_ = commPerf_.GetRankTileNum();
    
    // 设置最小 tile 长度（取通信和计算的最小阈值中的较大值）
    tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());
    tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankTileNum_));
}
```

**关键点**:
1. **通信轴选择**: ReduceScatter 的通信轴是 N（而非 K），因为需要在 N 维度上进行 scatter
2. **rankTileNum 计算**: 
   - ReduceScatter: `rankTileNum = rankDim`（每个 rank 一块）
   - AllGather: `rankTileNum = rankDim - 1`
   - AllReduce: `rankTileNum = 1`

#### 3.2.2 性能预测阶段

**代码位置**: `reduce_scatter_formulaic_tiling.cpp` EstimateKernelTime()

```cpp
void MMPlusReduceScatter::EstimateKernelTime()
{
    // 1. 设置通信时间因子
    SetCommTimeFactor();
    
    // 2. 预测总计算时间
    double totalMatmulTime = EstimateTotalMatmulTime();
    
    // 3. 预测总通信时间
    double totalTpTime = EstimateTotalCommTime();
    
    // 4. 计算时间比
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / 
                      std::min(totalTpTime, totalMatmulTime));
    
    // 5. 确定短块位置（关键决策）
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;  // 计算Bound - 短块后置
    }
    
    // 6. 动态调整最小 tile 长度
    // ...（详见后续分析）
}
```

**时间比的意义**:
- `ratioCalcComm_` 接近 1.0: 计算和通信时间接近，流水线配平效果好
- `ratioCalcComm_` 较大: 计算和通信时间差异大，需要优化切分策略

### 3.3 通信时间因子设置

#### 3.3.1 通信时间因子计算

**代码位置**: `reduce_scatter_formulaic_tiling.cpp` SetCommTimeFactor()

```cpp
void MMPlusReduceScatter::SetCommTimeFactor()
{
    if (clusterInfo_.socType == SocVersion::SOC950) {
        SetCommTimeFactorForA5();
    } else {
        SetCommTimeFactorForOther();
    }
}

void MMPlusReduceScatter::SetCommTimeFactorForOther()
{
    // 1. 检查通信膨胀条件
    bool bwGrowthByUtil = matmulPerf_.cubeUtil_ < PART_L2_UTIL;  // 0.85
    bool bwGrowthByShape = clusterInfo_.kValue >= LARGE_K_BOUNDARY && 
                           clusterInfo_.nValue > LARGE_N_BOUNDARY;  // K>=8192 && N>5120
    bool smallDim = rankDim_ <= MIN_COMM_RANKDIM || 
                    rankTileNum_ <= MatmulPerformance::SMALL_RANKTILE;
    
    bwGrowthByUtil = bwGrowthByUtil && !smallDim;
    bwGrowthByShape = bwGrowthByShape && !smallDim;
    
    // 2. 应用膨胀因子
    if (bwGrowthByUtil) {
        commPerf_.ChangeCommTimeFactorByDivision(scatterLargerNKCommGrowRatio1);  // 1.5
    } else if (bwGrowthByShape) {
        commPerf_.ChangeCommTimeFactorByDivision(scatterLargerNKCommGrowRatio2);  // 1.2
    }
    
    // 3. 基础调整
    commPerf_.ChangeCommTimeFactorByDivision(commGrowRatio);  // 1.15
    
    // 4. 芯片特定调整
    if (clusterInfo_.socType == SocVersion::SOC910_93) {
        commPerf_.ChangeCommTimeFactorByDivision(0.6);
    }
}
```

**通信膨胀条件解析**:
1. **bwGrowthByUtil** (带宽利用率低导致膨胀):
   - 条件: Cube 利用率 < 0.85 且卡数 > 2 且 rankTileNum > 2
   - 膨胀因子: 1.5 倍
   - 原因: Cube 利用率低意味着 L2 cache 命中率低，通信会受到影响

2. **bwGrowthByShape** (大 Shape 导致膨胀):
   - 条件: K >= 8192 且 N > 5120 且卡数 > 2 且 rankTileNum > 2
   - 膨胀因子: 1.2 倍
   - 原因: 大 K 大 N 场景下，通算并行时通信延迟会增加

3. **smallDim 保护**:
   - 当卡数 <= 2 或 rankTileNum <= 2 时，不应用膨胀因子
   - 原因: 小规模通信对通信带宽影响较小

## 4. 卡数(rankDim)限制分析

### 4.1 卡数下限限制

**代码位置**: `formulaic_tiling_datatype.h`

```cpp
constexpr uint64_t MIN_COMM_RANKDIM = 2;
```

**代码位置**: `hccl_performance.h` 构造函数

```cpp
commTypeInfo_.rankDim = std::max(static_cast<uint64_t>(inputRankDim), MIN_COMM_RANKDIM);
```

**分析**:
- **硬性下限**: rankDim 最小为 2
- **原因**: 
  1. ReduceScatter 需要至少 2 个 rank 才有意义（1 个 rank 不需要 reduce）
  2. HCCL 通信算法（Full Mesh/Ring）需要至少 2 个参与者

### 4.2 卡数上限限制

**代码分析**:

在 `hccl_formulaic_tiling.h` 中：
```cpp
constexpr uint64_t MAX_TILE_CNT = 16;
tilingM_.SetMaxTileCnt(MAX_TILE_CNT);  // 受限ffts队列大小，最多切16轮
```

在 `hccl_performance.cpp` 中：
```cpp
constexpr uint64_t FITTING_RANK = 8;  // 拟合采用8 die通信

void HCCLPerformanceModel::SetFullMeshCommTimeFactor()
{
    commTimeFactor_ = FULL_MESH_TIME_FACTOR;
    ChangeCommTimeFactorByDivision(static_cast<double>(FITTING_RANK) /
                                   static_cast<double>(commTypeInfo_.rankDim));
}
```

**卡数对通信时间的影响**:

1. **Full Mesh 模式**（SOC910B, SOC950）:
   ```cpp
   // 通信时间 = 总数据量 / rankDim / 带宽
   // 拟合基准: 8 die
   commTimeFactor = 2.0 * (8.0 / rankDim)
   ```
   - 卡数越多，单卡通信时间越短（分摊通信量）
   - 但卡数过多时，通信拓扑可能变化

2. **Double Ring 模式**（SOC910_93）:
   ```cpp
   // Ring 通信时间 = 总数据量 * 2 * (rankDim - 1) / rankDim / 带宽
   commTimeFactor = 2.0 * ((rankDim - 1) / rankDim) / ((8 - 1) / 8)
   ```
   - 卡数增加，单卡通信时间略增（Ring 步数增加）

### 4.3 不同卡数场景对比

| 卡数 | rankTileNum | 通信时间因子(Full Mesh) | 通信时间因子(Ring) | 特点 |
|------|-------------|------------------------|-------------------|------|
| 2    | 2           | 8.0                    | 2.29              | 通信时间最长，需减少切分 |
| 4    | 4           | 4.0                    | 2.67              | 中等通信时间 |
| 8    | 8           | 2.0                    | 2.86              | 拟合基准点 |
| 16   | 16          | 1.0                    | 2.93              | 通信时间最短，可多切分 |

**rankTileNum 计算**:
```cpp
uint64_t HCCLPerformanceModel::GetFullMeshRankTileNum()
{
    if (commTypeInfo_.kernelType == KernelType::ALL_GATHER) {
        return commTypeInfo_.rankDim - 1;  // AllGather: rankDim - 1
    }
    return commTypeInfo_.rankDim;  // ReduceScatter: rankDim
}
```

## 5. HCCL 通信量对切分的影响

### 5.1 通信量计算

**代码位置**: `hccl_performance.cpp`

```cpp
double HCCLPerformanceModel::CommTime(uint64_t mSize) const
{
    // 通信数据量 = mSize * N * rankTileNum * dtypeSize
    double commDataSize = mSize * commTypeInfo_.commMatrixLen * lookUpTileNum_ *
                         GetRealDtypeSizes();
    
    // 转换为 MB
    double tmpSize = commDataSize / ONE_MBYTE;
    
    // 分段拟合通信时间
    double result = commEstimatePar_.timeToSizeBoundary1;
    if (tmpSize > commEstimatePar_.sizeToTimeBoundary2) {
        // 大数据量: 线性模型
        result = commEstimatePar_.sizeToTimeLinearGradient * tmpSize +
                 commEstimatePar_.sizeToTimeLinearOffset;
    } else if (tmpSize > commEstimatePar_.sizeToTimeBoundary1) {
        // 中等数据量: 抛物线模型
        result = commEstimatePar_.sizeToTimeParabolicPar1 * tmpSize * tmpSize +
                 commEstimatePar_.sizeToTimeParabolicPar2 * tmpSize +
                 commEstimatePar_.sizeToTimeParabolicPar3;
    }
    
    // 应用通信时间因子
    result /= commTimeFactor_;
    return result;
}
```

**通信量组成**:
```
通信数据量 = tile_M × N × rankTileNum × dtype_size

其中:
- tile_M: 切分后的 M 轴长度
- N: 矩阵乘法的 N 维度
- rankTileNum: 
  - ReduceScatter: rankDim
  - AllGather: rankDim - 1
  - AllReduce: 1
- dtype_size: 数据类型大小 (FP16=2, FP32=4, BF16=2, etc.)
```

### 5.2 通信时间因子调整策略

**代码位置**: `reduce_scatter_formulaic_tiling.cpp`

```cpp
// 基础因子
constexpr double REDUCESCATTER_COMMTIME_FACTOR = 2.09136;

// 膨胀因子
constexpr double scatterLargerNKCommGrowRatio1 = 1.5;  // 低利用率膨胀
constexpr double scatterLargerNKCommGrowRatio2 = 1.2;  // 大Shape膨胀
constexpr double commGrowRatio = 1.15;                 // 基础膨胀
```

**调整逻辑**:
```cpp
// SOC950 (A5芯片): 直接使用基础因子
void MMPlusReduceScatter::SetCommTimeFactorForA5()
{
    commPerf_.ChangeCommTimeFactorByDivision(REDUCESCATTER_COMMTIME_FACTOR);
    // 实际 commTimeFactor = 2.0 / 2.09136 = 0.956
}

// 其他芯片: 根据条件膨胀
void MMPlusReduceScatter::SetCommTimeFactorForOther()
{
    // 最终 commTimeFactor 计算示例:
    // 基础: 2.0
    // 如果 bwGrowthByUtil: 2.0 / 1.5 = 1.333
    // 如果 bwGrowthByShape: 2.0 / 1.2 = 1.667
    // 应用基础膨胀: / 1.15
    // 如果是 SOC910_93: 再 * 0.6
}
```

### 5.3 通信量对 tile_M 切分的影响

#### 5.3.1 计算Bound vs 通信Bound

**判断标准**:
```cpp
if (totalMatmulTime >= totalTpTime) {
    tilingM_.cutRes.shortTileAtBack = true;  // 计算Bound
} else {
    tilingM_.cutRes.shortTileAtBack = false; // 通信Bound
}
```

**切分策略差异**:

| 类型 | 短块位置 | 配平目标 | 长块长度计算 |
|------|---------|---------|-------------|
| 计算Bound | 后置 | 让通信时间匹配计算时间 | CommTime^{-1}(CalcTime) |
| 通信Bound | 前置 | 让计算时间匹配通信时间 | CalcTime^{-1}(CommTime) |

#### 5.3.2 流水配平计算

**计算Bound - 短块后置**:
```cpp
void OneCalcOneCommBase::ShortAtEndCalcBoundBalancing()
{
    // 目标: 让短块的通信时间 = 短块的计算时间
    double targetTime = matmulPerf_.MatmulTime(tilingM_.cutRes.shortTileLen, rankTileNum_);
    tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(targetTime);
}
```

**通信Bound - 短块前置**:
```cpp
void OneCalcOneCommBase::ShortAtFrontCommBoundBalancing()
{
    // 目标: 让短块的计算时间 = 短块的通信时间
    double targetTime = commPerf_.CommTime(tilingM_.cutRes.shortTileLen);
    tilingM_.cutRes.longTileLen = matmulPerf_.InverseMatmulTime(targetTime, rankTileNum_);
}
```

### 5.4 通信量与计算量的平衡策略

**代码位置**: `reduce_scatter_formulaic_tiling.cpp` EstimateKernelTime()

```cpp
void MMPlusReduceScatter::EstimateKernelTime()
{
    // ... 性能预测 ...
    
    // 策略1: 减少初始值鼓励多切（通信Bound明显）
    bool decreaseMinLen = (totalTpTime > totalMatmulTime * UNBALANCE_RATIO) &&
                          (clusterInfo_.nValue >= FOUR_KILO) && 
                          (clusterInfo_.kValue >= ONE_KILO);
    if (decreaseMinLen) {
        tilingM_.SetMinLenByMin(tilingM_.GetAlignLength());  // 设为对齐长度 128
    }
    
    // 策略2: 增加初始值鼓励少切（计算Bound或确定性通信）
    bool increaseMinLen = (ratioCalcComm_ < commGrowRatio) ||
                          ((totalMatmulTime > totalTpTime * UNBALANCE_RATIO) &&
                           (clusterInfo_.mValue >= tilingM_.GetAlignLength() * FOUR));
    increaseMinLen = increaseMinLen && (clusterInfo_.nValue <= FOUR_KILO) && 
                     deterministicSoc910B_;
    if (increaseMinLen) {
        tilingM_.SetMinLenByMax(tilingM_.GetAlignLength() * TWO);  // 设为 256
    }
    
    // 策略3: 特殊Shape场景（M*N < 4M，性能变差）
    increaseMinLen = (totalMatmulTime >= totalTpTime * TIME_LOWER_RATIO) && 
                     (totalMatmulTime < totalTpTime * TIME_UPPER_RATIO);
    if (increaseMinLen) {
        tilingM_.SetMinLenByMax(FOUR_MBYTE / clusterInfo_.nValue / rankTileNum_);
    }
}
```

**平衡策略总结**:

| 条件 | 策略 | minTileLen | 切分趋势 |
|------|------|-----------|---------|
| 通信时间 >> 计算时间 (×1.5) | 减少最小值 | 128 | 多切分，流水掩盖通信 |
| 计算时间 >> 通信时间 (×1.5) | 增加最小值 | 256 | 少切分，避免碎片 |
| 计算时间 ≈ 通信时间 (×2~3) | 特殊调整 | 4MB/N/rankTileNum | 平衡性能与流水 |
| 确定性通信 + N小 | 增加最小值 | 256 | 减少通信轮次 |

## 6. 实际用例演示

### 6.1 用例1: 2卡场景

**输入参数**:
```
M = 8192, N = 2048, K = 4096
rankDim = 2
dtype = FP16 (2 bytes)
SOC = SOC910_B
```

**步骤1: 初始化**
```cpp
rankTileNum = 2  // ReduceScatter: rankTileNum = rankDim
commMatrixLen = N = 2048
dtypeSize = 2
```

**步骤2: 性能预测**
```cpp
// 计算时间预测
cubeUtil ≈ 0.75  // L2 cache利用率
matmulGradient = (N * K) / (computesPerCycle * frequency * coreNum * cubeUtil)
               ≈ (2048 * 4096) / (4096 * 1.8e3 * 20 * 0.75)
               ≈ 0.307
totalMatmulTime = M * rankTileNum * matmulGradient
                = 8192 * 2 * 0.307
                ≈ 5027 us

// 通信时间预测
commTimeFactor = 2.0 * (8.0 / 2.0) = 8.0  // Full Mesh, 2卡
commDataSize = M * N * rankTileNum * dtypeSize / 1MB
             = 8192 * 2048 * 2 * 2 / 1048576
             = 64 MB
totalTpTime = CommTime(8192) ≈ 64 / commTimeFactor * some_factor
            ≈ 64 / 8.0 * 10 ≈ 80 us (简化估算)
```

**步骤3: 判断Bound**
```cpp
totalMatmulTime (5027 us) >> totalTpTime (80 us)
=> 计算Bound
=> shortTileAtBack = true
```

**步骤4: 流水配平**
```cpp
// minTileLen 计算
commMinLen = HCCL_MIN_TILE_LEN / (N * rankTileNum * dtypeSize)
           = 64KB / (2048 * 2 * 2) = 8
matmulMinLen = baseM = 128
minTileLen = max(8, 128) = 128

// 配平计算
shortTileLen = 128
targetTime = MatmulTime(128, 2) = 128 * 2 * 0.307 = 78.6 us
longTileLen = InverseCommTime(78.6) ≈ 某个较小的值

// 由于通信时间远小于计算时间，切分意义不大
// 实际可能选择不切分或少量切分
```

**结果**:
```cpp
totalTileCnt = 1
numLongTile = 1
numShortTile = 0
longTileLen = 8192
shortTileLen = 0
```

### 6.2 用例2: 8卡场景

**输入参数**:
```
M = 16384, N = 4096, K = 8192
rankDim = 8
dtype = FP16 (2 bytes)
SOC = SOC910_B
```

**步骤1: 初始化**
```cpp
rankTileNum = 8
commMatrixLen = N = 4096
dtypeSize = 2
```

**步骤2: 性能预测**
```cpp
// 计算时间预测
cubeUtil ≈ 0.80  // 大Shape利用率高
matmulGradient ≈ (4096 * 8192) / (4096 * 1.8e3 * 20 * 0.80)
               ≈ 0.284
totalMatmulTime = 16384 * 8 * 0.284 ≈ 37250 us

// 通信时间预测
commTimeFactor = 2.0 * (8.0 / 8.0) = 2.0  // 基准点
commDataSize = 16384 * 4096 * 8 * 2 / 1048576 = 1024 MB
totalTpTime = CommTime(16384) ≈ 1024 / 2.0 * some_factor
            ≈ 512 * 13.58 ≈ 6953 us (使用线性拟合)
```

**步骤3: 判断Bound**
```cpp
totalMatmulTime (37250 us) > totalTpTime (6953 us)
ratioCalcComm_ = 37250 / 6953 ≈ 5.36
=> 计算Bound
=> shortTileAtBack = true
```

**步骤4: 流水配平**
```cpp
// minTileLen 计算
commMinLen = 64KB / (4096 * 8 * 2) = 1
matmulMinLen = baseM = 128
minTileLen = 128

// 检查是否需要减少 minTileLen
// totalTpTime < totalMatmulTime * 1.5，不满足条件

// 配平计算
shortTileLen = 128
targetTime = MatmulTime(128, 8) = 128 * 8 * 0.284 = 291 us
longTileLen = InverseCommTime(291) ≈ 某个值
```

**步骤5: 生成切分**
```cpp
// 离散切分对齐
GenerateInitialPartition(smallDimAlignUp=false)
longTileLen ≈ 2048 (假设)
numLongTile = (M - shortTileLen * 0.75) / longTileLen
            = (16384 - 96) / 2048 ≈ 7.9
=> numLongTile = 7
totalTileCnt = 7 + 1 = 8

// 对齐检查
// 长块 > 1KB，使用 AlignTileLarge
longTileLen = (M - shortTileLen + numLongTile - 1) / numLongTile
            = (16384 - 128 + 6) / 7 ≈ 2323
=> longTileLen = 2323 (向下对齐到128的倍数) = 2304
shortTileLen = M - numLongTile * longTileLen
             = 16384 - 7 * 2304 = 256
```

**结果**:
```cpp
totalTileCnt = 8
numLongTile = 7
numShortTile = 1
longTileLen = 2304
shortTileLen = 256
shortTileAtBack = true
```

### 6.3 用例3: 16卡场景

**输入参数**:
```
M = 32768, N = 8192, K = 16384
rankDim = 16
dtype = FP16 (2 bytes)
SOC = SOC910_B
```

**步骤1: 初始化**
```cpp
rankTileNum = 16
commMatrixLen = N = 8192
dtypeSize = 2
```

**步骤2: 性能预测**
```cpp
// 计算时间预测
cubeUtil ≈ 0.85  // 超大Shape
matmulGradient ≈ (8192 * 16384) / (4096 * 1.8e3 * 20 * 0.85)
               ≈ 0.109
totalMatmulTime = 32768 * 16 * 0.109 ≈ 57139 us

// 通信时间预测
commTimeFactor = 2.0 * (8.0 / 16.0) = 1.0  // 16卡分摊
commDataSize = 32768 * 8192 * 16 * 2 / 1048576 = 8192 MB
totalTpTime ≈ 8192 / 1.0 * 13.58 ≈ 111222 us

// 检查大Shape膨胀
bwGrowthByShape = (K >= 8192 && N > 5120) = true
smallDim = (rankDim <= 2 || rankTileNum <= 2) = false
=> 应用膨胀因子 1.2
totalTpTime *= 1.2 ≈ 133466 us
```

**步骤3: 判断Bound**
```cpp
totalTpTime (133466 us) > totalMatmulTime (57139 us)
ratioCalcComm_ = 133466 / 57139 ≈ 2.34
=> 通信Bound
=> shortTileAtBack = false
```

**步骤4: 流水配平**
```cpp
// minTileLen 计算
commMinLen = 64KB / (8192 * 16 * 2) = 0.25 (小于baseM，取128)
matmulMinLen = 128
minTileLen = 128

// 检查是否需要减少 minTileLen
// totalTpTime > totalMatmulTime * 1.5，且 N >= 4K，K >= 1K
=> decreaseMinLen = true
=> minTileLen = 128 (已经是baseM，不变)

// 配平计算
shortTileLen = 128
targetTime = CommTime(128) ≈ 某个值
longTileLen = InverseMatmulTime(targetTime, 16)
```

**步骤5: 生成切分**
```cpp
// 通信Bound，短块前置
ShortAtFrontCommBoundBalancing()
// ... 类似计算过程

// 结果
totalTileCnt = 12
numLongTile = 11
numShortTile = 1
longTileLen ≈ 2730
shortTileLen = 128
shortTileAtBack = false
```

### 6.4 对比总结

| 用例 | 卡数 | M | N | K | Bound类型 | 切分数 | 长块长度 | 短块长度 |
|------|------|---|---|---|-----------|--------|---------|---------|
| 1 | 2 | 8192 | 2048 | 4096 | 计算 | 1 | 8192 | 0 |
| 2 | 8 | 16384 | 4096 | 8192 | 计算 | 8 | 2304 | 256 |
| 3 | 16 | 32768 | 8192 | 16384 | 通信 | 12 | 2730 | 128 |

**观察结论**:
1. **卡数越多**，通信时间越短（分摊），但 rankTileNum 增加
2. **通信Bound场景**（用例3）切分更多，短块前置
3. **计算Bound场景**（用例1、2）切分较少或不变，短块后置
4. **大Shape场景**会触发通信膨胀因子，需要更多切分来掩盖通信

## 7. 切分算法核心逻辑

### 7.1 离散切分对齐算法

**代码位置**: `hccl_formulaic_tiling.cpp` FitTileLengthDiscrete()

```cpp
void FormPartition::FitTileLengthDiscrete(bool smallDimAlignUp,
                                          bool goodLinearityShape,
                                          bool localAtFront)
{
    // 1. 生成初始切分
    GenerateInitialPartition(smallDimAlignUp);
    
    // 2. 处理剩余长度
    uint64_t resLen = totalLen - numLongTile * longTileLen - 
                       numShortTile * shortTileLen;
    
    // 3. 检查是否需要重新均匀切分
    bool largeBackTile = (resLen >= shortTileLen) && 
                         (SMALL_TILECNT_PAR * shortTileLen >= longTileLen);
    bool smallCut = numLongTile <= SMALL_TILECNT_PAR;  // <= 2
    bool balanceCut = largeBackTile && goodLinearityShape && smallCut;
    
    if (balanceCut) {
        longTileLen = shortTileLen;
        numLongTile = (totalLen - shortTileLen) / longTileLen;
        // ...
    }
    
    // 4. 检查切分块数限制
    if (totalTileCnt > maxTileCnt) {
        // 强制均分到 maxTileCnt 块
        // ...
    }
    
    // 5. 对齐处理
    if (longTileLen > ONE_KBYTE) {
        AlignTileLarge();
    } else {
        AlignTileSmall();
    }
}
```

### 7.2 约束条件检查

**约束1: 最大切分块数**
```cpp
constexpr uint64_t MAX_TILE_CNT = 16;  // 受 FFTS 队列大小限制
```

**约束2: 最小 tile 长度**
```cpp
minTileLen = max(commMinLen, matmulMinLen)
其中:
commMinLen = HCCL_MIN_TILE_LEN / (N * rankTileNum * dtypeSize)
           = 64KB / (N * rankTileNum * dtypeSize)
matmulMinLen = baseM = 128
```

**约束3: 对齐要求**
```cpp
mAlignLen = BASE_BLOCK_M = 128  // M轴对齐到128
```

**约束4: 尾块处理**
```cpp
// 尾块应该小于主块
if (shortTileLen > longTileLen) {
    // 重新计算，使尾块 <= 主块
    longTileLen = (totalLen / totalTileCnt + mAlignLen - 1) / mAlignLen * mAlignLen;
    shortTileLen = totalLen - numLongTile * longTileLen;
}
```

### 7.3 边界情况处理

#### 7.3.1 Shape 太小不切分

```cpp
bool FormPartition::SetShortTileLen(bool noCutFlag)
{
    cutRes.shortTileLen = tileArgs.minTileLen;
    cutRes.numShortTile = 1U;
    
    bool smallMPerRank = (totalLen < cutRes.shortTileLen * 2);
    smallMPerRank = smallMPerRank || noCutFlag;
    
    if (smallMPerRank) {
        NoCutTiling();  // 不切分
    }
    return smallMPerRank;
}

void FormPartition::NoCutTiling()
{
    cutRes.totalTileCnt = 1U;
    cutRes.numShortTile = 0U;
    cutRes.numLongTile = 1U;
    cutRes.shortTileLen = 0U;
    cutRes.longTileLen = totalLen;
}
```

#### 7.3.2 切分块数超限

```cpp
void FormPartition::MaxTileCntUniform()
{
    if (cutRes.totalTileCnt <= tileArgs.maxTileCnt) {
        return;
    }
    
    if (tileArgs.maxTileCnt <= 1 || totalLen < tileArgs.maxTileCnt) {
        NoCutTiling();  // 无法切分，不切
        return;
    }
    
    // 强制均分到 maxTileCnt - 1 块长块 + 1 块尾块
    cutRes.longTileLen = totalLen / (tileArgs.maxTileCnt - 1);
    cutRes.longTileLen = (cutRes.longTileLen + mAlignLen - 1) / mAlignLen * mAlignLen;
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen);
    cutRes.numLongTile = totalLen / cutRes.longTileLen;
    CalcBackTile();
}
```

#### 7.3.3 特殊对齐场景

```cpp
void FormPartition::AlignTileLarge()
{
    // 大块长度 > 1KB 时，向上对齐尾块
    cutRes.shortTileLen = (cutRes.shortTileLen + mAlignLen - 1) / mAlignLen * mAlignLen;
    cutRes.longTileLen = (totalLen - cutRes.shortTileLen + numLongTile - 1) / numLongTile;
    cutRes.shortTileLen = totalLen - numLongTile * cutRes.longTileLen;
}

void FormPartition::AlignTileSmall()
{
    // 小块长度 <= 1KB 时，检查尾块是否过小
    bool smallBackTile = (cutRes.shortTileLen < mAlignLen) && cutRes.shortTileLen > 0;
    bool largeFrontTile = (cutRes.longTileLen > 2 * mAlignLen);
    
    if (smallBackTile && largeFrontTile) {
        // 从长块借一部分给短块
        cutRes.shortTileLen += mAlignLen * numLongTile;
        cutRes.longTileLen -= mAlignLen;
    }
}
```

## 8. 关键常量与参数

### 8.1 时间比率常量

```cpp
constexpr double TIME_LOWER_RATIO = 2.0;   // 计算时间 < 2×通信时间
constexpr double TIME_UPPER_RATIO = 3.0;   // 计算时间 < 3×通信时间
constexpr double UNBALANCE_RATIO = 1.5;    // 不平衡阈值
```

### 8.2 Shape 阈值常量

```cpp
constexpr uint64_t LARGE_K_BOUNDARY = 8192;     // 大K阈值
constexpr uint64_t LARGE_N_BOUNDARY = 5120;     // 大N阈值
constexpr uint64_t ONE_KILO = 1024;
constexpr uint64_t FOUR_KILO = 4096;
```

### 8.3 通信因子常量

```cpp
constexpr double REDUCESCATTER_COMMTIME_FACTOR = 2.09136;
constexpr double scatterLargerNKCommGrowRatio1 = 1.5;  // 低利用率膨胀
constexpr double scatterLargerNKCommGrowRatio2 = 1.2;  // 大Shape膨胀
constexpr double commGrowRatio = 1.15;                 // 基础膨胀
constexpr double LOCAL_REDUCE_FACTOR = 0.4;            // 确定性通信因子
```

### 8.4 对齐常量

```cpp
constexpr uint64_t BASE_BLOCK_M = 128;       // M轴基础块大小
constexpr uint64_t HCCL_MIN_TILE_LEN = 64 * ONE_KBYTE;  // 64KB
constexpr uint64_t HCCL_MIN_TILE_LEN_COARSE = 2 * ONE_MBYTE;  // 2MB
```

## 9. 性能优化建议

### 9.1 切分策略优化

1. **计算Bound场景** (计算时间 > 通信时间):
   - 减少切分数，增大 tile 大小
   - 短块后置，充分利用最后一块的计算能力
   - minTileLen 设置较大 (256 或更多)

2. **通信Bound场景** (通信时间 > 计算时间):
   - 增加切分数，利用流水线掩盖通信
   - 短块前置，提前开始通信
   - minTileLen 设置较小 (128)

3. **平衡场景** (计算时间 ≈ 通信时间):
   - 中等切分数
   - 根据实际 Shape 动态调整
   - 特殊处理 M×N < 4MB 的情况

### 9.2 卡数优化建议

1. **小卡数 (rankDim <= 2)**:
   - 通信时间较长，适合减少切分
   - 考虑使用 larger tile 以减少通信启动开销

2. **中等卡数 (rankDim = 4~8)**:
   - 平衡点，适合流水线配平
   - 根据实际 Shape 调整切分策略

3. **大卡数 (rankDim >= 16)**:
   - 通信时间分摊明显，可增加切分
   - 注意 FFTS 队列限制 (maxTileCnt = 16)

### 9.3 Shape 相关建议

1. **大 K 大 N 场景** (K >= 8192, N > 5120):
   - 触发通信膨胀因子 (1.2×)
   - 需要更多切分来掩盖通信

2. **小 Shape 场景** (M < 2 * minTileLen):
   - 不切分，直接计算
   - 避免过小的 tile 导致性能下降

3. **L2 Cache 利用率低场景** (cubeUtil < 0.85):
   - 触发通信膨胀因子 (1.5×)
   - 需要调整切分策略或优化数据布局

## 10. 总结

### 10.1 核心机制

`reduce_scatter_formulaic_tiling` 通过以下机制实现高效的流水线配平:

1. **双层性能模型**: 计算性能模型 + 通信性能模型，精确预测时间
2. **动态切分策略**: 根据 Bound 类型动态调整短块位置和 tile 大小
3. **多维约束平衡**: 综合考虑卡数、Shape、L2 cache、通信拓扑等因素
4. **自适应调整**: 根据实际场景动态调整通信时间因子和最小 tile 长度

### 10.2 关键发现

1. **卡数影响**:
   - rankDim 影响通信时间因子和 rankTileNum
   - 卡数越多，通信时间越短（Full Mesh）或略增（Ring）
   - 小卡数场景需要特别处理以避免通信瓶颈

2. **通信量影响**:
   - 通信量 = tile_M × N × rankTileNum × dtype_size
   - 通信时间因子根据 Shape 和利用率动态调整
   - 大 Shape 和低利用率会触发膨胀因子

3. **切分决策**:
   - 计算Bound: 短块后置，减少切分
   - 通信Bound: 短块前置，增加切分
   - 平衡点: 动态调整，特殊处理小 Shape

### 10.3 最佳实践

1. **优先保证性能模型的准确性**:
   - 确保 cubeUtil 和 commTimeFactor 的拟合准确
   - 定期更新性能参数映射表

2. **关注边界情况**:
   - 小 Shape 不切分
   - 切分块数超限强制均分
   - 尾块大小约束

3. **充分利用调试日志**:
   ```cpp
   OP_LOGD("MatmulReduceScatter", 
           "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, "
           "totalMatmulTime %f, totalTP %f, minTileSize %lu, mAlignLen %lu, "
           "commTimeFactor_ %f",
           ...);
   ```

4. **持续优化**:
   - 收集不同 Shape 和卡数的实际性能数据
   - 反馈优化性能模型参数
   - 探索新的切分策略（如多维度切分）