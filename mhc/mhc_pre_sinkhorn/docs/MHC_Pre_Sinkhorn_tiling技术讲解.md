
# AscendC Tiling技术完整解析

## 一、Tiling技术概述

在AscendC算子开发中，**Tiling（分块/切分）** 是核心优化技术，主要解决以下问题：

1. **硬件资源限制**：UB（Unified Buffer）容量有限，无法一次性加载所有数据
2. **多核并行**：将计算任务分配到多个AI Core并行执行
3. **内存访问优化**：优化数据访问模式，提升缓存命中率

## 二、Tiling策略的主要分支

### 2.1 分支架构图

```
mhc_pre_sinkhorn Tiling策略
│
├─ 分支1: 多核切分策略
│   ├─ M维度切分（按batch维度）
│   └─ K维度切分（按特征维度）
│
├─ 分支2: UB内存规划
│   ├─ Stage1内存规划（矩阵乘法阶段）
│   └─ Stage2内存规划（后处理阶段）
│
├─ 分支3: 数据切分因子计算
│   ├─ rowFactor计算（行切分）
│   └─ dFactor计算（特征维度切分）
│
└─ 分支4: 循环次数计算
    ├─ rowLoop计算（行循环次数）
    └─ dLoop计算（特征维度循环次数）
```

### 2.2 分支1：多核切分策略

#### 功能说明
将大张量计算任务分配到多个AI Core并行执行

#### 代码实现
```cpp
// mhc_pre_sinkhorn_tiling.cpp: CalcOpTiling()
ge::graphStatus MhcPreSinkhornTiling::CalcOpTiling() {
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    
    // M维度切分：按batch维度分配到不同AI Core
    uint64_t mDimNum = std::min(aicCoreNum_, 
                                static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
    
    // K维度切分：当K较大时启用多核切分
    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), 
                                  K_MULIT_CORE_SPLIT_BASE_SIZE);
    
    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));
    tilingData_.set_multCoreSplitMSize(singleCoreM);
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
}
```

#### 应用场景
- **M维度切分**：当batch size较大时，将不同batch分配到不同Core
- **K维度切分**：当特征维度`k = hc_mult * d`较大时，启用多核切K策略

#### 数学原理
```
M维度切分：
├─ mDimNum = min(aicCoreNum, ceil(bs / M_L1_MAX_SIZE))
├─ singleCoreM = ceil(ceil(bs / mDimNum) / BLOCK_CUBE) * BLOCK_CUBE
└─ 每个Core处理 singleCoreM 个batch

K维度切分：
├─ kDimNum = aicCoreNum / mDimNum
├─ splitKSize = ceil(ceil(kSize / kDimNum) / 256) * 256
└─ 每个Core处理 splitKSize 的K维度
```

### 2.3 分支2：UB内存规划

#### 功能说明
根据UB容量动态规划各缓冲区大小，确保不发生内存溢出

#### 代码实现
```cpp
// Stage2内存规划（后处理阶段）
ge::graphStatus MhcPreSinkhornTiling::CalcMembaseOpTiling() {
    // 1. 计算各缓冲区大小
    int64_t mix0Size = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;     
    int64_t mix1Size = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
    int64_t mix2Size = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
    int64_t rsqrtSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t ySize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    // ... 其他缓冲区
    
    // 2. 计算总UB占用
    int64_t totalSize = mix0Size + mix1Size + mix2Size + rsqrtSize + 
                        xSize + ySize + postSize + combFragSize + ...;
    
    // 3. 判断是否需要切分
    if (totalSize <= ubSize_) {
        // 全载模式：无需切分
        dLoop_ = 1;
        dFactor_ = d_;
    } else {
        // 分块模式：需要切分d维度
        // ... 切分逻辑
    }
}
```

#### 应用场景
- **全载模式**：当totalSize ≤ ubSize时，一次性加载所有数据
- **分块模式**：当totalSize > ubSize时，需要切分数据分批处理

#### 内存布局规划
```
UB内存布局：
┌─────────────────────────────────────────────┐
│ mix0Size    │ mix1Size    │ mix2Size       │
├─────────────────────────────────────────────┤
│ rsqrtSize   │ xSize       │ ySize          │
├─────────────────────────────────────────────┤
│ postSize    │ combFragSize│ base0Size      │
├─────────────────────────────────────────────┤
│ base1Size   │ base2Size   │ xCastSize      │
├─────────────────────────────────────────────┤
│ yCastSize   │ rowBrcb0Size│ hcBrcb1Size    │
└─────────────────────────────────────────────┘
```

### 2.4 分支3：数据切分因子计算

#### 功能说明
计算每次迭代处理的数据量，平衡计算效率和内存占用

#### 代码实现
```cpp
// dFactor计算：特征维度切分
if (totalSize <= ubSize_) {
    dLoop_ = 1;
    dFactor_ = d_;
    tailDFactor_ = dFactor_;
} else {
    // 计算已使用的UB大小（不包括与d相关的缓冲区）
    int64_t usedUbSize = mix0Size + mix1Size + mix2Size + rsqrtSize + 
                         postSize + combFragSize + base0Size + base1Size + 
                         base2Size + rowBrcb0Size + hcBrcb1Size + reduceBufSize;
    int64_t ubRemain = ubSize_ - usedUbSize;
    
    // 迭代寻找最优dFactor
    dFactor_ = d_;
    int64_t base = 2;
    while (1) {
        dFactor_ = CeilDiv(d_, base);
        xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
        ySize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
        xCastSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 8) * sizeof(float);
        yCastSize = rowOnceLoop * RoundUp(dFactor_, 8) * sizeof(float);
        
        int64_t targetSize = xSize + ySize + xCastSize + yCastSize;
        if (targetSize <= ubRemain) {
            break;
        }
        base++;
    }
    
    // 对齐优化
    if (dFactor_ > 32) {
        dFactor_ = DownAlign(dFactor_, 32);
    }
    
    dLoop_ = CeilDiv(d_, dFactor_);
    tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
}

// rowFactor计算：行切分
if (dFactor_ == d_) {
    while (rowFactor_ <= rowOfFormerBlock_) {
        // 重新计算各缓冲区大小（使用rowFactor_）
        mix0Size = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
        // ... 其他缓冲区
        
        totalSize = mix0Size + mix1Size + ...;
        if (totalSize > ubSize_) {
            rowFactor_ = rowFactor_ - 1;
            break;
        }
        rowFactor_ = rowFactor_ + 1;
    }
    rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
}
```

#### 应用场景
- **dFactor**：控制特征维度的切分粒度
- **rowFactor**：控制batch维度的切分粒度

#### 数学原理
```
dFactor计算：
├─ 目标：找到最大的dFactor，使得targetSize ≤ ubRemain
├─ targetSize = xSize + ySize + xCastSize + yCastSize
├─ xSize = rowOnceLoop * hcMult * RoundUp(dFactor, 16) * 2 * DOUBLE_BUFFER
└─ 迭代搜索：从base=2开始，逐步增加直到满足条件

rowFactor计算：
├─ 目标：找到最大的rowFactor，使得totalSize ≤ ubSize
├─ totalSize = 所有缓冲区大小之和
└─ 迭代搜索：从1开始，逐步增加直到超过UB容量
```

### 2.5 分支4：循环次数计算

#### 功能说明
计算各维度的循环次数，用于Kernel中的迭代控制

#### 代码实现
```cpp
// 计算循环次数
rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? 
                              rowFactor_ : rowOfFormerBlock_ % rowFactor_;
tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? 
                            rowFactor_ : rowOfTailBlock_ % rowFactor_;

dLoop_ = CeilDiv(d_, dFactor_);
tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
```

#### 应用场景
- **rowLoop**：控制外层循环次数（按batch维度）
- **dLoop**：控制内层循环次数（按特征维度）
- **tailFactor**：处理尾部不完整数据块

## 三、关键变量详解

### 3.1 网格尺寸变量

| 变量名 | 数据类型 | 取值范围 | 作用 |
|--------|---------|---------|------|
| `bs_` | int64_t | ≥1 | batch size，总批次数 |
| `hcMult_` | int64_t | ≥1 | 超连接倍数 |
| `d_` | int64_t | ≥1 | 特征维度 |
| `hcMix_` | int64_t | = 2*hcMult + hcMult² | 混合维度 |
| `k` | int64_t | = hcMult * d | K维度大小 |

### 3.2 坐标偏移量

| 变量名 | 数据类型 | 计算公式 | 作用 |
|--------|---------|---------|------|
| `rowOfFormerBlock_` | int64_t | ceil(bs / aivCoreNum) | 每个Core处理的batch数（非尾部Core） |
| `rowOfTailBlock_` | int64_t | bs - (usedAivCoreNums-1) * rowOfFormerBlock | 尾部Core处理的batch数 |
| `mGmOffset` | uint64_t | mBlkDimIdx * mL1Size | M维度在GM中的偏移 |
| `kGmStartOffset` | uint64_t | multCoreSplitKSize * kBlkDimIdx | K维度起始偏移 |

### 3.3 缩放因子

| 变量名 | 数据类型 | 取值范围 | 作用 |
|--------|---------|---------|------|
| `rowFactor_` | int64_t | 1 ~ rowOfFormerBlock | 每次迭代处理的行数 |
| `dFactor_` | int64_t | 1 ~ d | 每次迭代处理的特征维度 |
| `hcMultAlign_` | int64_t | RoundUp(hcMult, 8) | 对齐后的hcMult |

### 3.4 重叠区域参数

| 变量名 | 数据类型 | 作用 |
|--------|---------|------|
| `DOUBLE_BUFFER` | constexpr int64_t = 2 | 双缓冲机制，提升流水线效率 |
| `BLOCK_SIZE` | constexpr int64_t = 32 | 数据块大小，用于对齐 |
| `REPEAT_SIZE` | constexpr int64_t = 256 | Repeat指令处理的数据量 |

### 3.5 瓦片索引

| 变量名 | 数据类型 | 作用 |
|--------|---------|------|
| `mBlkDimIdx` | uint64_t | M维度的Core索引 |
| `kBlkDimIdx` | uint64_t | K维度的Core索引 |
| `rowOuterIdx` | int64_t | 行外层循环索引 |
| `dLoopIdx` | int64_t | 特征维度循环索引 |

## 四、核心计算方法

### 4.1 瓦片尺寸计算方法

#### 数学原理
```
瓦片尺寸 = min(硬件限制, 数据总大小 / 并行度)

具体公式：
├─ mL1Size = min(M_L1_MAX_SIZE, singleCoreM)
├─ kL1Size = min(A_L1_SIZE / singleCoreM, K_L1_MAX_SIZE) / 128 * 128
├─ splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE)
└─ stage1MFactor = ubSize / ((sizeof(int16_t) + sizeof(int32_t)) * DOUBLE_BUFFER * cvLoopKSize)
```

#### 实现步骤
```cpp
// 步骤1: 计算M维度瓦片尺寸
uint64_t mDimNum = std::min(aicCoreNum_, CeilDiv(bs_, M_L1_MAX_SIZE));
uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
uint64_t mL1Size = std::min(M_L1_MAX_SIZE, singleCoreM);

// 步骤2: 计算K维度瓦片尺寸
uint64_t kDimNum = aicCoreNum_ / mDimNum;
uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);
uint64_t kL1Size = std::min(A_L1_SIZE / singleCoreM, K_L1_MAX_SIZE) / 128 * 128;

// 步骤3: 计算Stage1的M维度因子
int64_t lineByteSize = (sizeof(int16_t) + sizeof(int32_t)) * DOUBLE_BUFFER * cvLoopKSize;
int64_t stage1MFactorValue = ubSize_ / lineByteSize;
```

### 4.2 坐标映射算法

#### 数学原理
```
全局坐标 → 核心坐标 → UB坐标

GM坐标计算：
├─ mGmOffset = mBlkDimIdx * mL1Size + rowOuterIdx * rowFactor
├─ kGmOffset = kBlkDimIdx * splitKSize + kInnerOffset
└─ xGmOffset = mGmOffset * k + kGmOffset

UB坐标计算：
├─ xUbOffset = ubLoopIdx * rowFactor * k
└─ yUbOffset = ubLoopIdx * rowFactor * d
```

#### 实现步骤
```cpp
// Kernel中的坐标映射
uint64_t mBlkDimIdx = curBlockIdx / tilingData->cubeBlockDimK;
uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;

// 计算K维度的GM偏移
uint64_t kGmStartOffset = tilingData->multCoreSplitKSize * kBlkDimIdx;
uint64_t kGmEndOffset = kGmStartOffset + tilingData->multCoreSplitKSize;

// 计算M维度的GM偏移
for (uint64_t mGmOffset = mBlkDimIdx * tilingData->mL1Size; 
     mGmOffset < tilingData->bs;
     mGmOffset += tilingData->cubeBlockDimM * tilingData->mL1Size) {
    // 处理当前M分块
}
```

### 4.3 重叠区域计算逻辑

#### 数学原理
```
双缓冲机制：
├─ Buffer0: 当前迭代使用
├─ Buffer1: 下一迭代预加载
└─ 总缓冲区大小 = 单缓冲大小 * DOUBLE_BUFFER

数据对齐：
├─ RoundUp(x, alignment) = ceil(x / alignment) * alignment
├─ 目的：满足硬件对齐要求，提升访问效率
└─ 示例：RoundUp(100, 32) = 128
```

#### 实现步骤
```cpp
// 计算对齐后的缓冲区大小
int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
int64_t yCastSize = rowOnceLoop * RoundUp(d_, 8) * sizeof(float);

// 双缓冲初始化
pipe->InitBuffer(xQue, NUM_TWO, xQueNum * sizeof(T));  // NUM_TWO = 2
pipe->InitBuffer(yQue, NUM_TWO, yQueNum * sizeof(T));
```

### 4.4 瓦片索引生成规则

#### 数学原理
```
三维索引映射：
├─ Core索引 = blockIdx
├─ M维度索引 = blockIdx / cubeBlockDimK
├─ K维度索引 = blockIdx % cubeBlockDimK
└─ UB循环索引 = rowOuterIdx * dLoop + dLoopIdx

循环次数计算：
├─ rowLoop = ceil(rowOfFormerBlock / rowFactor)
├─ dLoop = ceil(d / dFactor)
└─ 总循环次数 = rowLoop * dLoop
```

#### 实现步骤
```cpp
// Stage2的索引生成
int64_t stage2BlockIdx = GetBlockIdx();
int64_t rowOuterLoop = (stage2BlockIdx == stage2UsedCoreNum - 1) ? 
                       tilingData->rowLoopOfTailBlock : 
                       tilingData->rowLoopOfFormerBlock;

for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
    int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? 
                           tailRowFactor : stage2RowFactor;
    
    for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
        int64_t curDFactor = (dLoopIdx == tilingData->dLoop - 1) ? 
                             tilingData->tailDFactor : tilingData->dFactor;
        // 处理当前瓦片
    }
}
```

## 五、协同工作机制

### 5.1 完整数据流

```
Tiling策略执行流程：
│
├─ Step 1: 多核切分
│   ├─ 计算M维度切分：mDimNum, singleCoreM
│   ├─ 计算K维度切分：kDimNum, splitKSize
│   └─ 确定是否启用多核切K策略
│
├─ Step 2: UB内存规划
│   ├─ 计算各缓冲区大小
│   ├─ 判断是否需要切分
│   └─ 选择合适的Tiling策略
│
├─ Step 3: 数据切分因子计算
│   ├─ 计算dFactor（特征维度切分）
│   ├─ 计算rowFactor（行切分）
│   └─ 确保totalSize ≤ ubSize
│
├─ Step 4: 循环次数计算
│   ├─ 计算rowLoop, dLoop
│   ├─ 计算tailFactor（尾部处理）
│   └─ 生成Tiling参数
│
└─ Step 5: Kernel执行
    ├─ 根据Tiling参数加载数据
    ├─ 执行计算
    └─ 输出结果
```

### 5.2 性能优化策略

#### 1. 内存访问优化
```
优化策略：
├─ 数据对齐：RoundUp(x, alignment)
├─ 双缓冲：DOUBLE_BUFFER = 2
├─ 连续访问：优先沿连续内存维度处理
└─ 缓存友好：减少cache miss
```

#### 2. 计算并行优化
```
优化策略：
├─ 多核并行：M维度和K维度双向切分
├─ 流水线并行：MTE2、MTE1、MM、FIXP各阶段并行
├─ 向量化：充分利用Vector Core的Repeat指令
└─ 负载均衡：动态调整rowFactor和dFactor
```

#### 3. 内存占用优化
```
优化策略：
├─ UB复用：不同阶段复用同一UB缓冲区
├─ Workspace优化：合理规划Workspace大小
├─ 稀疏存储：对于稀疏数据采用压缩存储
└─ 延迟归约：减少中间结果的存储
```

## 六、总结

### Tiling技术的核心思想

1. **分而治之**：将大任务分解为小任务，适应硬件资源限制
2. **并行计算**：充分利用多核并行能力，提升计算效率
3. **内存优化**：合理规划内存使用，避免内存溢出
4. **数据局部性**：优化数据访问模式，提升缓存命中率

### 关键设计原则

1. **硬件感知**：充分考虑硬件特性（UB容量、Core数量、缓存大小）
2. **动态规划**：根据输入形状动态调整切分策略
3. **对齐优化**：满足硬件对齐要求，提升访问效率
4. **负载均衡**：合理分配任务，避免Core间负载不均

### 性能影响

| Tiling策略 | 内存占用 | 计算效率 | 并行度 |
|-----------|---------|---------|--------|
| 全载模式 | 高 | 高 | 低 |
| 细粒度切分 | 低 | 中 | 高 |
| 粗粒度切分 | 中 | 高 | 中 |
| **最优策略** | **中** | **高** | **高** |

通过合理的Tiling策略，mhc_pre_sinkhorn算子实现了高效的内存利用和计算性能，充分发挥了ASCEND910B/A3平台的硬件能力。