# reduce_scatter_fit_balance_tiling 深度分析报告

## 1. 概述

`reduce_scatter_fit_balance_tiling` 是 MatmulReduceScatter 算子的一种优化切分方案，用于在 Arch35 (Ascend910B) 平台上实现计算与通信的流水线配平。本报告深入分析其切分逻辑、最大块数限制、与其他方法的对比以及实际用例演示。

---

## 2. 类继承结构与架构

### 2.1 继承关系

```
Mc2FitBasedBalanceTiling (基类 - Fit-Based Balance切分基类)
├── MatmulPerformanceArch35 (计算性能模型 - Arch35专用)
├── HCCLPerformanceArch35 (通信性能模型 - Arch35专用)
├── FormPartition (切分逻辑核心)
└── MMReduceScatterFitBalanceTiling (ReduceScatter专用实现)
```

### 2.2 核心数据结构

#### CutResult - 切分结果（与formulaic相同）
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

#### MatmulParameters - 矩阵参数
```cpp
struct MatmulParameters {
    uint64_t mValue;              // M轴长度
    uint64_t nValue;              // N轴长度
    uint64_t kValue;              // K轴长度
    uint64_t inMatrixADtypeSize;  // 输入矩阵A的数据类型大小
    uint64_t inMatrixBDtypeSize;  // 输入矩阵B的数据类型大小
    uint64_t outMatrixCDtypeSize; // 输出矩阵C的数据类型大小
    SocVersion socType;           // 芯片类型
};
```

---

## 3. 完整的切分流程

### 3.1 主流程图

```
┌─────────────────────────────────────────────────────────────┐
│           MMReduceScatterFitBalanceTiling 切分流程            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 初始化 (构造函数)                                          │
│  ├─ 1.1 设置通信形状参数 (N轴, 输出dtype)                       │
│  ├─ 1.2 设置最小tile长度为BaseM                                │
│  ├─ 1.3 设置对齐长度为BaseM                                    │
│  ├─ 1.4 检测是否为量化矩阵乘法                                  │
│  └─ 1.5 设置最大切分块数 MAX_TILE_CNT=16                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 性能预测 (EstimateMMCommTime)                             │
│  ├─ 2.1 查找Cube利用率 (FindCubeUtil)                         │
│  ├─ 2.2 获取Matmul梯度 (GetMatmulGradient)                    │
│  ├─ 2.3 计算总Matmul时间                                       │
│  ├─ 2.4 计算总通信时间                                         │
│  ├─ 2.5 计算时间比率 ratioCalcComm                            │
│  └─ 2.6 确定短块位置 shortTileAtBack                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 设置短块长度 (SetShortTileLen)                            │
│  ├─ 3.1 计算L2 Cache使用量                                     │
│  ├─ 3.2 判断是否超过L2 Cache                                   │
│  ├─ 3.3 动态调整最小长度                                       │
│  │   - 超过L2: minLen *= 2                                    │
│  │   - 小shape: minLen = mValue                               │
│  └─ 3.4 初始化短块长度                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 检查是否需要切分                                            │
│  ├─ 4.1 如果 totalLen < 2 * shortTileLen                      │
│  └─ 4.2 执行 NoCutTiling (不切分)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 设置长块长度 (SetLongTileLen)                             │
│  ├─ 5.1 流水配平计算                                           │
│  │   - 计算Bound: longTileLen = InverseCommTime(targetTime)   │
│  │   - 通信Bound: longTileLen = InverseMatmulTime(targetTime) │
│  ├─ 5.2 特殊场景调整                                           │
│  │   - L2 Cache + N轴大: longTileLen = shortTileLen           │
│  │   - 量化 + K小 + 不平衡: longTileLen = shortTileLen        │
│  └─ 5.3 输出长块长度                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. 调整长短块 (AdjustLongShortTileLen)                       │
│  ├─ 6.1 判断是否为高线性形状                                    │
│  ├─ 6.2 执行 FitTileLengthDiscrete                            │
│  ├─ 6.3 检查是否需要合并                                       │
│  │   - longTileLen == shortTileLen: 合并为单一块               │
│  └─ 6.4 输出最终切分结果                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  7. 输出切分结果                                                │
│  - shortTileAtBack, longTileLen, numLongTile                  │
│  - shortTileLen, numShortTile, totalTileCnt                   │
│  - 最大切分块数限制检查 (≤16)                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 最大允许切分块数分析

### 4.1 核心限制：MAX_TILE_CNT = 16

#### 定义位置
```cpp
// 文件: mc2/common/op_host/op_tiling/hccl_formulaic_tiling.h
// 第 38 行
constexpr uint64_t MAX_TILE_CNT = 16;
```

#### 设置位置
```cpp
// 文件: mc2/common/op_host/op_tiling/mc2_fit_based_balance_tiling.h
// 第 43 行
explicit Mc2FitBasedBalanceTiling(...) {
    ...
    tilingM_.SetMaxTileCnt(MAX_TILE_CNT); 
    // Due to the limited FFTS queue size, a maximum of 16 rounds can be cut.
}
```

### 4.2 限制的技术原因

#### FFTS队列限制
- **FFTS (Fast Fourier Transform Scheduler)**: 硬件任务调度队列
- **队列容量**: 最大支持16个并发通信任务
- **通信轮次**: 每个切分块对应一个通信轮次，占用一个FFTS队列项

#### 注释原文
```cpp
// Due to the limited FFTS queue size, a maximum of 16 rounds can be cut.
// 受限于FFTS队列大小，最多切16轮
```

### 4.3 限制检查机制

#### MaxTileCntUniform函数（核心检查）

**位置**: `hccl_formulaic_tiling.cpp` 第 317-333 行

```cpp
void FormPartition::MaxTileCntUniform() {
    // 检查是否超过限制
    if (cutRes.totalTileCnt <= tileArgs.maxTileCnt) {
        return;  // 未超限，直接返回
    }
    
    // 超限处理
    if (tileArgs.maxTileCnt <= 1 || totalLen < tileArgs.maxTileCnt) {
        NoCutTiling();  // 无法切分，不切
        return;
    }

    // 强制切分为 maxTileCnt - 1 块长块 + 1 块尾块
    cutRes.longTileLen = totalLen / (tileArgs.maxTileCnt - 1);
    cutRes.longTileLen = (cutRes.longTileLen + tileArgs.mAlignLen - 1) /
                         tileArgs.mAlignLen * tileArgs.mAlignLen;  // 对齐
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen);
    cutRes.numLongTile = totalLen / cutRes.longTileLen;
    CalcBackTile();  // 计算尾块
}
```

#### FitTileLengthDiscrete中的检查

**位置**: `hccl_formulaic_tiling.cpp` 第 148-162 行

```cpp
void FormPartition::FitTileLengthDiscrete(...) {
    // ... 前面的切分计算 ...
    
    // 检查切分轮次总数符合约束 (第 148-162 行)
    tileArgs.maxTileCnt = std::max(tileArgs.maxTileCnt, ONE);  // 确保 > 0
    if (cutRes.totalTileCnt > tileArgs.maxTileCnt) {
        // 强制切分为 maxTileCnt 块，每块长度为 totalLen / maxTileCnt
        uint64_t tileLen = totalLen / tileArgs.maxTileCnt;
        bool isDivisible = (tileLen * tileArgs.maxTileCnt == totalLen) &&
                           (tileLen % tileArgs.mAlignLen == 0U);
        if (isDivisible) {
            cutRes.numLongTile = tileArgs.maxTileCnt;
            cutRes.longTileLen = tileLen;
            cutRes.numShortTile = 0U;
            cutRes.shortTileLen = 0U;
            cutRes.totalTileCnt = tileArgs.maxTileCnt;
        }
    }
    MaxTileCntUniform();  // 再次检查并调整
}
```

### 4.4 数学关系验证

```
总切分块数 = numLongTile + numShortTile
           ≤ MAX_TILE_CNT (16)

其中：
- numLongTile: 长块（主块）数量，通常为 [1, 15]
- numShortTile: 短块（尾块）数量，值为 0 或 1
```

---

## 5. 核心切分算法详解

### 5.1 EstimateMMCommTime - 性能预测

#### 代码实现
```cpp
void MMReduceScatterFitBalanceTiling::EstimateMMCommTime()
{
    // 1. 查找Cube利用率（关键性能指标）
    matmulPerf_.FindCubeUtil(rankTileNum_);
    
    // 2. 获取Matmul梯度（用于反向求解tile长度）
    matmulPerf_.GetMatmulGradient();
    
    // 3. 计算总计算时间
    double totalMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, rankDim_);
    
    // 4. 计算总通信时间
    double totalTpTime = commPerf_.CommTime(mmInfo_.mValue);
    
    // 5. 计算时间比率（衡量流水配平效果）
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / 
                      std::min(totalTpTime, totalMatmulTime));
    
    // 6. 确定短块位置（流水配平方向）
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;  // 计算Bound - 短块后置
    }
}
```

#### 关键点分析
- **FindCubeUtil**: 根据输入shape计算Cube单元利用率
- **GetMatmulGradient**: 获取计算时间与tile长度的梯度关系，用于反向求解
- **rankTileNum**: ReduceScatter的rankTileNum = rankDim（每个rank一块）

### 5.2 SetShortTileLen - 设置短块长度

#### 代码实现
```cpp
void MMReduceScatterFitBalanceTiling::SetShortTileLen()
{
    // 1. 计算L2 Cache使用量
    uint64_t l2UseSize = mmInfo_.mValue * mmInfo_.kValue * mmInfo_.inMatrixADtypeSize +
                         mmInfo_.kValue * mmInfo_.nValue * mmInfo_.inMatrixBDtypeSize;
    
    // 2. 判断是否超过L2 Cache (128MB)
    isLargerThanL2Cache_ = l2UseSize > L2_CACHE_SIZE;
    
    // 3. 动态调整最小长度
    if (isLargerThanL2Cache_ && mmInfo_.mValue > tilingM_.GetMinLen() * FOUR) {
        // L2 Cache压力大，增大最小长度，鼓励大块切分
        tilingM_.SetMinLenByMax(tilingM_.GetMinLen() * TWO);
    }
    
    // 4. 小shape场景特殊处理
    if (mmInfo_.mValue * mmInfo_.nValue <= coreNum_ * matmulPerf_.GetBaseM() * 
                                            matmulPerf_.GetBaseN() * TWO) {
        // shape太小，不切分
        tilingM_.SetMinLenByMax(mmInfo_.mValue);
    }
    
    // 5. 初始化短块长度和数量
    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1U;
}
```

#### L2 Cache优化策略
- **超过L2 Cache**: 增大tile长度，减少L2 Cache访问频率
- **小shape**: 不切分，避免切分开销大于收益

### 5.3 SetLongTileLen - 设置长块长度

#### 代码实现
```cpp
void MMReduceScatterFitBalanceTiling::SetLongTileLen()
{
    // 1. 流水配平计算
    if (tilingM_.cutRes.shortTileAtBack) {
        // 计算Bound: 短块时间 = Matmul(shortTileLen)
        // 长块时间应等于短块通信时间，实现流水掩盖
        double targetTime = matmulPerf_.MatmulTime(tilingM_.cutRes.shortTileLen, 
                                                    rankTileNum_);
        tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(targetTime);
    } else {
        // 通信Bound: 短块时间 = Comm(shortTileLen)
        // 长块时间应等于短块计算时间
        double targetTime = commPerf_.CommTime(tilingM_.cutRes.shortTileLen);
        tilingM_.cutRes.longTileLen = matmulPerf_.InverseMatmulTime(targetTime, 
                                                                    rankTileNum_);
    }
    
    // 2. 特殊场景调整
    // 2.1 L2 Cache + N轴大 + M可整除短块
    if (isLargerThanL2Cache_ && 
        mmInfo_.mValue % tilingM_.cutRes.shortTileLen == 0 &&
        (mmInfo_.nValue >= TWO * mmInfo_.mValue * rankDim_)) {
        // 均匀切分，避免L2 Cache不均衡访问
        tilingM_.cutRes.longTileLen = tilingM_.cutRes.shortTileLen;
    }
    
    // 2.2 量化矩阵乘法 + 不平衡 + K小
    else if (isQuantMatmul_ && 
             ratioCalcComm_ > UNBALANCE_RATIO && 
             mmInfo_.kValue < SMALL_K_BOUND &&
             mmInfo_.mValue % tilingM_.cutRes.shortTileLen == 0) {
        // 量化场景K小时计算快，均匀切分更优
        tilingM_.cutRes.longTileLen = tilingM_.cutRes.shortTileLen;
    }
}
```

#### 流水配平原理
```
计算Bound（短块后置）:
┌─────────┐┌─────────┐┌─────────┐
│ Matmul  ││ Matmul  ││ Matmul  │ (短块)
│  tile1  ││  tile2  ││  tile3  │
└─────────┘└─────────┘└─────────┘
      ↓         ↓         ↓
┌──────────────────────────────┐
│         Comm tile            │ (长块，覆盖短块计算时间)
└──────────────────────────────┘

通信Bound（短块前置）:
┌───┐┌───┐┌───┐
│ C ││ C ││ C │ (短块通信)
└───┘└───┘└───┘
  ↓   ↓   ↓
┌───────────────────────┐
│    Matmul (长块)      │ (覆盖短块通信时间)
└───────────────────────┘
```

### 5.4 AdjustLongShortTileLen - 最终调整

#### 代码实现
```cpp
void MMReduceScatterFitBalanceTiling::AdjustLongShortTileLen()
{
    // 1. 判断是否为高线性形状（影响对齐策略）
    bool goodLinearityShape = (mmInfo_.kValue * mmInfo_.nValue >= 
                               LARGE_NK_BAR_BASE * ONE_MBYTE);  // 32MB
    
    // 2. 执行离散切分对齐（核心算法）
    tilingM_.FitTileLengthDiscrete(false, goodLinearityShape);
    
    // 3. 检查是否需要合并长短块
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        // 长短块相等，合并为单一块
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
}
```

### 5.5 FitTileLengthDiscrete - 离散切分核心算法

#### 核心流程
```cpp
void FormPartition::FitTileLengthDiscrete(bool smallDimAlignUp,
                                          bool goodLinearityShape,
                                          bool localAtFront = false)
{
    // Step 1: 生成初始切分
    GenerateInitialPartition(smallDimAlignUp);
    
    // Step 2: 处理剩余长度
    uint64_t resLen = totalLen - cutRes.numLongTile * cutRes.longTileLen -
                      cutRes.numShortTile * cutRes.shortTileLen;
    
    // Step 3: 检查是否需要重新均匀切分
    bool largeBackTile = (resLen >= cutRes.shortTileLen) &&
                         ((SMALL_TILECNT_PAR * cutRes.shortTileLen) >= cutRes.longTileLen);
    goodLinearityShape = (goodLinearityShape && 
                          (cutRes.shortTileLen >= tileArgs.mAlignLen)) ||
                         (!cutRes.shortTileAtBack);
    bool smallCut = cutRes.numLongTile <= SMALL_TILECNT_PAR;
    bool balanceCut = largeBackTile && goodLinearityShape && smallCut;
    
    if (balanceCut) {
        // 重新均匀切分
        cutRes.longTileLen = cutRes.shortTileLen;
        cutRes.numLongTile = (totalLen - cutRes.shortTileLen) / cutRes.longTileLen;
        cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
    }
    
    // Step 4: 增加长块个数（如果尾块过大）
    bool moreFrontTile = largeBackTile && 
                         (cutRes.longTileLen <= cutRes.shortTileLen + resLen);
    if (moreFrontTile) {
        cutRes.numLongTile++;
        CalcBackTile();
    }
    
    // Step 5: 检查最大切分块数限制（≤16）
    if (cutRes.totalTileCnt > tileArgs.maxTileCnt) {
        // 强制切分为 maxTileCnt 块
        uint64_t tileLen = totalLen / tileArgs.maxTileCnt;
        bool isDivisible = (tileLen * tileArgs.maxTileCnt == totalLen) &&
                           (tileLen % tileArgs.mAlignLen == 0U);
        if (isDivisible) {
            cutRes.numLongTile = tileArgs.maxTileCnt;
            cutRes.longTileLen = tileLen;
            cutRes.numShortTile = 0U;
            cutRes.shortTileLen = 0U;
            cutRes.totalTileCnt = tileArgs.maxTileCnt;
        }
    }
    
    // Step 6: 最终检查
    MaxTileCntUniform();
    
    // Step 7: 对齐处理
    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    if (cutRes.longTileLen > ONE_KBYTE) {
        AlignTileLarge();
    } else {
        AlignTileSmall();
    }
}
```

---

## 6. 与其他Tiling方法的对比

### 6.1 方法对比表

| 方法 | 父类 | 切分算法 | 最大块数 | 适用场景 |
|------|------|---------|---------|---------|
| **MMReduceScatterFitBalanceTiling** | Mc2FitBasedBalanceTiling | Fit-based Balance | 16 | Arch35, 标准卡/8P |
| **MMPlusReduceScatter** | OneCalcOneCommBase | Formulaic | 16 | 其他平台 |
| **MMAllReduceFitBalanceTiling** | Mc2FitBasedBalanceTiling | Fit-based Balance | 16 | Arch35, AllReduce |
| **AllGatherMMFitBalanceTiling** | Mc2FitBasedBalanceTiling | Fit-based Balance | 16 | Arch35, AllGather |

### 6.2 Fit-Balance vs Formulaic 详细对比

#### 核心差异

| 特性 | Fit-Balance | Formulaic |
|------|-------------|-----------|
| **性能模型** | Arch35专用（更精确） | 通用模型 |
| **切分策略** | 动态拟合（反向求解） | 公式化参数 |
| **L2 Cache优化** | 显式考虑L2 Cache压力 | 隐式处理 |
| **量化场景** | 特殊优化（UNBALANCE_RATIO） | 无特殊处理 |
| **最小长度调整** | 多级动态调整 | 单级调整 |
| **适用平台** | Arch35 (Ascend910B) | 所有平台 |

#### 代码对比

**Fit-Balance的SetShortTileLen**（ReduceScatter）:
```cpp
void MMReduceScatterFitBalanceTiling::SetShortTileLen() {
    // L2 Cache优化
    uint64_t l2UseSize = ...;
    isLargerThanL2Cache_ = l2UseSize > L2_CACHE_SIZE;
    if (isLargerThanL2Cache_ && mmInfo_.mValue > minLen * FOUR) {
        SetMinLenByMax(minLen * TWO);  // 动态调整
    }
    
    // 小shape优化
    if (mValue * nValue <= coreNum * baseM * baseN * TWO) {
        SetMinLenByMax(mValue);  // 不切分
    }
}
```

**Formulaic的SetShortTileLen**（ReduceScatter）:
```cpp
void MMPlusReduceScatter::EstimateKernelTime() {
    // 通信时间因子设置
    SetCommTimeFactor();
    
    // 时间预测
    double totalMatmulTime = EstimateTotalMatmulTime();
    double totalTpTime = EstimateTotalCommTime();
    
    // 动态调整
    bool decreaseMinLen = (totalTpTime > totalMatmulTime * UNBALANCE_RATIO) &&
                          (nValue >= FOUR_KILO) && (kValue >= ONE_KILO);
    if (decreaseMinLen) {
        SetMinLenByMin(mAlignLen);  // 减少初始值
    }
}
```

### 6.3 不同算子的Fit-Balance实现对比

#### 算子对比表

| 算子 | SetShortTileLen策略 | SetLongTileLen策略 | 特殊优化 |
|------|-------------------|-------------------|---------|
| **ReduceScatter** | L2 Cache + 小shape | N轴大 + 量化K小 | 均匀切分条件 |
| **AllReduce** | L2 Cache + 时间阈值 | 切分开销补偿 | CommBoundTile |
| **AllGather** | 平衡鼓励 + cutLen调整 | LocalTile覆盖 | 前置Local |

#### ReduceScatter vs AllReduce代码对比

**ReduceScatter的SetLongTileLen**:
```cpp
// 特殊场景1: L2 Cache + N轴大
if (isLargerThanL2Cache_ && mValue % shortTileLen == 0 &&
    (nValue >= TWO * mValue * rankDim_)) {
    longTileLen = shortTileLen;  // 均匀切分
}

// 特殊场景2: 量化 + K小
else if (isQuantMatmul_ && ratioCalcComm_ > UNBALANCE_RATIO && 
         kValue < SMALL_K_BOUND) {
    longTileLen = shortTileLen;
}
```

**AllReduce的SetLongTileLen**:
```cpp
// 切分开销补偿
if ((mValue - shortTileLen) / longTileLen >= COMM_EXPANSION_TILENUM) {
    longTileLen *= TWO;  // 增大长块，减少切分开销
    SetMinLenByMax(minLen * TWO);
}
```

---

## 7. 实际用例演示

### 7.1 用例1: 小shape不切分

**输入参数**:
```
M = 256, N = 512, K = 512
rankDim = 4
dtype = FP16 (size=2)
coreNum = 32
baseM = 128, baseN = 256
```

**切分过程**:

#### Step 1: 初始化
```cpp
minTileLen = baseM = 128
mAlignLen = baseM = 128
maxTileCnt = 16
```

#### Step 2: 性能预测
```cpp
cubeUtil = FindCubeUtil(rankTileNum=4) ≈ 0.75
totalMatmulTime = MatmulTime(256, 4) ≈ 50us
totalCommTime = CommTime(256) ≈ 40us
ratioCalcComm = 50 / 40 = 1.25
shortTileAtBack = true  // 计算Bound
```

#### Step 3: SetShortTileLen
```cpp
l2UseSize = 256*512*2 + 512*512*2 = 1MB
isLargerThanL2Cache = false  // 1MB < 128MB

// 小shape检查
mValue * nValue = 256*512 = 128K
coreNum * baseM * baseN * TWO = 32*128*256*2 = 2M
128K <= 2M  // 小shape，不切分

SetMinLenByMax(256)  // minTileLen = 256
shortTileLen = 256
numShortTile = 1
```

#### Step 4: 检查切分
```cpp
totalLen < shortTileLen * 2
256 < 256 * 2  // 不满足
// 执行 NoCutTiling
```

#### 输出结果
```
totalTileCnt = 1
numLongTile = 1, longTileLen = 256
numShortTile = 0, shortTileLen = 0
shortTileAtBack = true
```

### 7.2 用例2: L2 Cache压力场景

**输入参数**:
```
M = 8192, N = 4096, K = 8192
rankDim = 8
dtype = FP16 (size=2)
coreNum = 32
baseM = 128
```

**切分过程**:

#### Step 1: 初始化
```cpp
minTileLen = 128
mAlignLen = 128
maxTileCnt = 16
```

#### Step 2: 性能预测
```cpp
cubeUtil ≈ 0.85
totalMatmulTime ≈ 1500us
totalCommTime ≈ 1200us
ratioCalcComm = 1500 / 1200 = 1.25
shortTileAtBack = true  // 计算Bound
```

#### Step 3: SetShortTileLen
```cpp
l2UseSize = 8192*8192*2 + 8192*4096*2 = 192MB
isLargerThanL2Cache = true  // 192MB > 128MB

// L2 Cache优化
8192 > 128 * 4  // true
SetMinLenByMax(128 * 2)  // minTileLen = 256

shortTileLen = 256
numShortTile = 1
```

#### Step 4: SetLongTileLen
```cpp
// 流水配平
targetTime = MatmulTime(256, rankTileNum=8) ≈ 80us
longTileLen = InverseCommTime(80us) ≈ 512

// N轴大检查
nValue >= TWO * mValue * rankDim_
4096 >= 2 * 8192 * 8  // false

// 不触发特殊调整
longTileLen = 512
```

#### Step 5: FitTileLengthDiscrete
```cpp
// 生成初始切分
longTileLen = 512（对齐后）
numLongTile = (8192 - 256 * 0.75) / 512 ≈ 15
totalTileCnt = 15 + 1 = 16  // 达到最大限制

// 检查最大限制
totalTileCnt = 16 <= maxTileCnt = 16  // OK

// 对齐处理
shortTileLen = 8192 - 15 * 512 = 512
longTileLen > 1K  // true
AlignTileLarge()
```

#### 输出结果
```
totalTileCnt = 16
numLongTile = 15, longTileLen = 512
numShortTile = 1, shortTileLen = 512
shortTileAtBack = true
```

### 7.3 用例3: 量化场景小K

**输入参数**:
```
M = 4096, N = 1024, K = 512
rankDim = 8
dtype = INT8 (量化矩阵乘法)
coreNum = 32
baseM = 128
```

**切分过程**:

#### Step 1: 初始化
```cpp
isQuantMatmul = (inMatrixADtypeSize == 1) && (inMatrixBDtypeSize == 1)
             = true  // INT8量化
```

#### Step 2: 性能预测
```cpp
totalMatmulTime ≈ 200us  // INT8计算快
totalCommTime ≈ 600us
ratioCalcComm = 600 / 200 = 3.0  // 通信Bound严重
shortTileAtBack = false  // 通信Bound - 短块前置
```

#### Step 3: SetShortTileLen
```cpp
l2UseSize = 4096*512*1 + 512*1024*1 = 2.5MB
isLargerThanL2Cache = false

shortTileLen = 128
numShortTile = 1
```

#### Step 4: SetLongTileLen
```cpp
// 流水配平
targetTime = CommTime(128) ≈ 30us
longTileLen = InverseMatmulTime(30us, rankTileNum=8) ≈ 256

// 量化 + K小 + 不平衡检查
ratioCalcComm = 3.0 > UNBALANCE_RATIO = 2  // true
kValue = 512 < SMALL_K_BOUND = 2048  // true
mValue % shortTileLen = 4096 % 128 = 0  // true

// 触发均匀切分
longTileLen = shortTileLen = 128
```

#### Step 5: FitTileLengthDiscrete
```cpp
// 均匀切分
numLongTile = (4096 - 128) / 128 = 31
totalTileCnt = 31 + 1 = 32  // 超过限制

// 强制切分为16块
tileLen = 4096 / 16 = 256
isDivisible = (256 * 16 == 4096) && (256 % 128 == 0)  // true

numLongTile = 16
longTileLen = 256
numShortTile = 0
shortTileLen = 0
totalTileCnt = 16
```

#### 输出结果
```
totalTileCnt = 16
numLongTile = 16, longTileLen = 256
numShortTile = 0, shortTileLen = 0
shortTileAtBack = false
```

---

## 8. 最大块数的边界情况分析

### 8.1 边界情况1: 刚好达到16块

**条件**: `totalTileCnt = 16`

**处理**: 正常通过，无需调整

**示例**:
```
M = 2048, shortTileLen = 128, longTileLen = 128
numLongTile = 15, numShortTile = 1
totalTileCnt = 16 ≤ maxTileCnt = 16
```

### 8.2 边界情况2: 超过16块但可整除

**条件**: `totalTileCnt > 16` 且 `M % 16 == 0`

**处理**: 强制均匀切分为16块

**示例**:
```
M = 4096, 原切分: numLongTile = 31, totalTileCnt = 32
强制切分: numLongTile = 16, longTileLen = 256
         numShortTile = 0, shortTileLen = 0
```

### 8.3 边界情况3: 超过16块且不可整除

**条件**: `totalTileCnt > 16` 且 `M % 16 != 0`

**处理**: MaxTileCntUniform强制调整

**示例**:
```
M = 4097, 原切分: totalTileCnt = 32
强制切分:
longTileLen = 4097 / (16 - 1) = 273  (向下对齐到128 = 256)
numLongTile = 4097 / 256 = 15
shortTileLen = 4097 - 15 * 256 = 497
totalTileCnt = 15 + 1 = 16
```

### 8.4 边界情况4: M太小无法切分

**条件**: `M < maxTileCnt`

**处理**: NoCutTiling（不切分）

**示例**:
```
M = 10, maxTileCnt = 16
M < maxTileCnt  // true
NoCutTiling()
totalTileCnt = 1
```

---

## 9. 核心算法复杂度分析

### 9.1 时间复杂度

| 函数 | 时间复杂度 | 说明 |
|------|-----------|------|
| EstimateMMCommTime | O(1) | 单次性能计算 |
| SetShortTileLen | O(1) | 条件判断和赋值 |
| SetLongTileLen | O(1) | 反向求解和条件判断 |
| FitTileLengthDiscrete | O(1) | 数学计算，无循环 |
| MaxTileCntUniform | O(1) | 数学计算，无循环 |

**总复杂度**: O(1) - 纯数学计算，无迭代或递归

### 9.2 空间复杂度

| 数据结构 | 空间复杂度 | 说明 |
|----------|-----------|------|
| CutResult | O(1) | 固定大小结构体 |
| TileArguments | O(1) | 固定大小结构体 |
| MatmulParameters | O(1) | 固定大小结构体 |

**总复杂度**: O(1) - 固定大小的数据结构

---

## 10. 关键参数常量汇总

### 10.1 切分参数常量

| 常量名 | 值 | 定义位置 | 用途 |
|--------|---|---------|------|
| `MAX_TILE_CNT` | 16 | hccl_formulaic_tiling.h:38 | 最大切分块数 |
| `BASE_BLOCK_M` | 128 | matmul_formulaic_tiling.h:37 | M轴基础块大小 |
| `BASE_BLOCK_N` | 256 | matmul_formulaic_tiling.h:38 | N轴基础块大小 |
| `LARGE_NK_BAR_BASE` | 32 | hccl_formulaic_tiling.h:24 | 高线性形状阈值 |

### 10.2 优化参数常量（ReduceScatter专用）

| 常量名 | 值 | 定义位置 | 用途 |
|--------|---|---------|------|
| `L2_CACHE_SIZE` | 128MB | reduce_scatter_fit_balance_tiling.cpp:18 | L2 Cache大小阈值 |
| `UNBALANCE_RATIO` | 2 | reduce_scatter_fit_balance_tiling.cpp:19 | 不平衡比率阈值 |
| `SMALL_K_BOUND` | 2048 | reduce_scatter_fit_balance_tiling.cpp:20 | 小K边界 |

### 10.3 对齐参数常量

| 常量名 | 值 | 定义位置 | 用途 |
|--------|---|---------|------|
| `ALIGN_RATIO_CALC_BOUND` | 0.5 | hccl_formulaic_tiling.h:34 | 计算Bound对齐比率 |
| `ALIGN_RATIO_COMM_BOUND` | 0.75 | hccl_formulaic_tiling.h:35 | 通信Bound对齐比率 |
| `BACKTILE_ALIGN_RATIO` | 0.75 | hccl_formulaic_tiling.h:29 | 尾块对齐比率 |

---

## 11. 总结

### 11.1 核心结论

1. **最大块数限制**: `MAX_TILE_CNT = 16`，受FFTS队列大小限制，适用于所有Fit-Balance切分方法

2. **切分流程**: 四阶段流程 - 性能预测、短块设置、长块设置、最终调整

3. **优化策略**: 
   - L2 Cache压力优化
   - 小shape不切分优化
   - 量化场景特殊处理
   - 流水配平动态求解

4. **适用场景**: Arch35平台（Ascend910B），标准卡和8P拓扑

### 11.2 与Formulaic的主要差异

1. **性能模型**: Arch35专用模型，更精确的性能预测
2. **优化策略**: 显式L2 Cache优化，量化场景特殊处理
3. **动态调整**: 多级最小长度调整，反向求解tile长度
4. **适用平台**: 仅支持Arch35，不支持其他平台

### 11.3 实际应用建议

1. **小shape场景**: 不切分，避免开销大于收益
2. **L2 Cache压力大**: 增大tile长度，减少L2访问频率
3. **量化小K**: 均匀切分，利用INT8计算优势
4. **边界情况**: 注意16块限制，避免强制切分降低性能

---

## 附录A: 关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| MAX_TILE_CNT定义 | hccl_formulaic_tiling.h | 38 |
| SetMaxTileCnt调用 | mc2_fit_based_balance_tiling.h | 43 |
| MaxTileCntUniform实现 | hccl_formulaic_tiling.cpp | 317-333 |
| FitTileLengthDiscrete实现 | hccl_formulaic_tiling.cpp | 111-171 |
| EstimateMMCommTime实现 | reduce_scatter_fit_balance_tiling.cpp | 22-42 |
| SetShortTileLen实现 | reduce_scatter_fit_balance_tiling.cpp | 44-58 |
| SetLongTileLen实现 | reduce_scatter_fit_balance_tiling.cpp | 60-82 |
| AdjustLongShortTileLen实现 | reduce_scatter_fit_balance_tiling.cpp | 84-94 |

---

## 附录B: 相关文档参考

1. `formulaic_tiling_analysis.md` - Formulaic切分方法详细分析
2. `max_tile_cnt_analysis.md` - 最大块数限制深度分析
3. `tiling_analysis.md` - MatmulReduceScatter整体切分分析