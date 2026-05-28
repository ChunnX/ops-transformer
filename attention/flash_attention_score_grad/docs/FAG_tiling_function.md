# FlashAttentionScoreGrad (FAG) 分核函数详解

## 1. 分核总体流程

### 1.1 DoOpTiling - 主Tiling入口函数

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:248`

**功能**: FAG算子的主Tiling入口，协调所有分核子模块。

**调用流程**:
```
DoOpTiling()
  ├── SetSplitAxis()        // 设置切分轴策略
  ├── DoSplit()             // S1/S2基本块切分
  ├── DoSparse()            // 稀疏模式处理与分核
  │     ├── GetSparseType()         // 获取稀疏类型
  │     ├── GetDeterSparseTilingKey() // 确定性模式判断
  │     ├── CalcleDeterParam()      // 计算确定性参数
  │     ├── DoBn2s2Sparse()         // BN2S2分核尝试
  │     ├── DoBn2MultiBlkSparse()   // BN2多块分核
  │     ├── GetSparseUnpadBlockInfo() // TND分核
  │     ├── GetSparseBlockInfo()    // 稀疏分核
  │     └── GetIsDeterArr()         // 确定性数组计算
  ├── CheckExceedL2Cache()  // 检查是否超出L2缓存
  ├── InitTilingData()      // 初始化TilingData结构
  ├── DoPreTiling()         // 预处理Tiling
  ├── DoPostTiling()        // 后处理Tiling
  ├── DetermineMode()       // 设置确定性模式
  └── PostTiling()          // 后置Tiling（设置BlockDim等）
```

**关键代码**:
```cpp
ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::DoOpTiling()
{
    SetSplitAxis(context_, fBaseParams);  // 设置切分轴
    DoSplit();                             // 基本块切分
    
    auto ret = DoSparse();                 // 稀疏分核（核心）
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 分核优化判断
    bool isExceedL2Cache = CheckExceedL2Cache();
    bool isLargeInvalidBlk = CheckIsLargeInvalidBlk(fBaseParams);
    fBaseParams.enableSwizzle = (isExceedL2Cache || isLargeInvalidBlk) &&
        !fBaseParams.isDeterministic &&
        fBaseParams.blockOuter == fBaseParams.aicNum;

    ret = InitTilingData();               // 初始化数据结构
    DoPreTiling();                         // 预处理
    DoPostTiling();                        // 后处理
    DetermineMode(fBaseParams);            // 确定性模式
    
    return ge::GRAPH_SUCCESS;
}
```

---

## 2. 基本块切分函数

### 2.1 DoSplit - S1/S2基本块切分

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:283`

**功能**: 将S1和S2序列长度切分成基本块，设置inner/outer/tail参数。

**核心逻辑**:
```cpp
void FlashAttentionScoreGradTilingNormalRegbase::DoSplit()
{
    // 设置Cube-Vector比例（Regbase架构固定为2）
    fBaseParams.s1CvRatio = S1CV_RATIO_DEFAULT;  // = 2
    fBaseParams.s2CvRatio = S2CV_RATIO_DEFAULT;  // = 2
    
    // 获取最佳切分方案
    std::tuple<uint32_t, uint32_t, uint32_t> bestSplitRes = FuzzyForBestSplit();
    int64_t s1Inner = std::get<0>(bestSplitRes);  // S1基本块
    int64_t s2Inner = std::get<1>(bestSplitRes);  // S2基本块
    uint32_t sfmgdInner = std::get<2>(bestSplitRes); // D维度基本块
    
    // 计算S1切分参数
    int64_t s1CvInner = s1Inner * fBaseParams.s1CvRatio;
    if (s1CvInner > fBaseParams.s1) {
        s1CvInner = fBaseParams.s1;  // 不超过总长度
    }
    int64_t s1Outer = (fBaseParams.s1 + s1CvInner - 1) / s1CvInner;  // 向上取整
    int64_t s1TailTmp = fBaseParams.s1 % s1Inner;
    fBaseParams.s1Tail = s1TailTmp == 0 ? s1Inner : s1TailTmp;
    fBaseParams.s1CvTail = fBaseParams.s1 % s1CvInner;
    
    // 设置参数
    fBaseParams.s1Inner = s1Inner;
    fBaseParams.s1CvInner = s1CvInner;  // Cube处理的基本块 = 2 × s1Inner
    fBaseParams.s1Outer = s1Outer;
    
    // S2切分（类似S1）
    // ...
}
```

**切分示意图**:
```
┌─────────────────────────────────────────────────────────────┐
│                    S1切分示意图                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  S1总长度: 512                                               │
│  s1Inner = 128 (基本块尺寸)                                  │
│  s1CvRatio = 2 (Cube:Vector比例)                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │  s1CvInner = 128 × 2 = 256                         │    │
│  │  (Cube一次处理的S1长度，分给2个Vector核)             │    │
│  │                                                     │    │
│  │  S1切分:                                            │    │
│  │  ┌────────┬────────┬────────┐                      │    │
│  │  │Block 0 │Block 1 │Block 2 │                      │    │
│  │  │ 256    │ 256    │ 0(Tail)│                      │    │
│  │  └────────┴────────┴────────┘                      │    │
│  │                                                     │    │
│  │  s1Outer = ceil(512/256) = 2                        │    │
│  │  s1Tail = 512 % 128 = 0 → 使用s1Inner=128           │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Vector核分配:                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │  Block 0 (256):                                     │    │
│  │  ├─ Vector核0: 处理前128 (s1Inner)                  │    │
│  │  └─ Vector核1: 处理后128 (s1Inner)                  │    │
│  │                                                     │    │
│  │  Block 1 (256):                                     │    │
│  │  ├─ Vector核0: 处理前128                            │    │
│  │  └─ Vector核1: 处理后128                            │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 FuzzyForBestSplit - 最佳切分搜索

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:1195`

**功能**: 根据模板类型选择最优的s1Inner/s2Inner/dInner。

```cpp
std::tuple<uint32_t, uint32_t, uint32_t> FuzzyForBestSplit()
{
    auto s1s2TemplateSize = GetS1S2TemplateType(fBaseParams);
    uint32_t s1Inner = s1s2TemplateSize.first / 2;  // 除2因为CVRatio=2
    uint32_t s2Inner = s1s2TemplateSize.second;
    uint32_t dInner = GetDTemplateType(fBaseParams);
    
    return std::tie(s1Inner, s2Inner, dInner);
}
```

---

## 3. 核心分核函数

### 3.1 DoSparse - 稀疏模式分核入口

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:463`

**功能**: 根据稀疏模式和layout选择合适的分核策略。

**决策流程**:
```
┌─────────────────────────────────────────────────────────────┐
│                   DoSparse 分核决策流程                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: fBaseParams (layout, sparseMode, splitAxis等)        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  Step1: 获取稀疏类型                                  │  │
│  │  ├─ GetSparseType() → DENSE/CASUAL/BAND              │  │
│  │  ├─ GetDeterSparseTilingKey() → 确定性类型           │  │
│  │  └─ CalcleDeterParam() → 计算确定性参数              │  │
│  │                                                       │  │
│  │  Step2: 尝试BN2S2分核                                 │  │
│  │  ├─ DoBn2s2Sparse()                                  │  │
│  │  │  ├─ 成功且blockOuter >= aicNum → 返回            │  │
│  │  │  └─ 失败 → 继续尝试其他策略                       │  │
│  │                                                       │  │
│  │  Step3: 尝试BN2多块分核                               │  │
│  │  ├─ DoBn2MultiBlkSparse()                            │  │
│  │  │  ├─ TND → GetBlockInfoOfTNDForBn2()              │  │
│  │  │  ├─ Sparse → GetSparseBlockInfoBn2()             │  │
│  │  │  └─ Normal → 均匀切分                            │  │
│  │                                                       │  │
│  │  Step4: 根据layout选择分核                            │  │
│  │  ├─ TND → GetSparseUnpadBlockInfo()                  │  │
│  │  ├─ Sparse →                                         │  │
│  │  │  ├─ PREFIX → GetSparsePrefixBlockInfo()          │  │
│  │  │  └─ 其他 → GetSparseBlockInfo()                  │  │
│  │  └─ Normal → 均匀切分                                │  │
│  │                                                       │  │
│  │  Step5: 确定性数组计算                                │  │
│  │  └─ GetIsDeterArr() (如果是旧确定性模式)             │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  输出: blockStarts[64], blockEnds[64], blockOuter等         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键代码**:
```cpp
ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::DoSparse()
{
    // Step1: 获取稀疏类型
    fBaseParams.sparseType = GetSparseType();
    fBaseParams.deterSparseType = GetDeterSparseTilingKey();
    CalcleDeterParam();
    
    // Step2: 尝试BN2S2分核（负载均衡优化）
    if (DoBn2s2Sparse() && fBaseParams.blockOuter >= fBaseParams.aicNum) {
        return ge::GRAPH_SUCCESS;  // 成功，直接返回
    }
    
    // Step3: 尝试BN2多块分核
    if (fBaseParams.splitAxis == SplitAxisEnum::BN2 && fBaseParams.isBn2MultiBlk) {
        bool res = DoBn2MultiBlkSparse();
        if (!(fBaseParams.isInvalidCol || fBaseParams.isInvalidRow)) {
            return res;  // 无无效行列，返回
        }
        // 有无效行列，切换到BN2GS1S2
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
    }
    
    // Step4: 根据layout分核
    fBaseParams.splitAxis = fBaseParams.isBn2 ? SplitAxisEnum::BN2 : SplitAxisEnum::BN2GS1S2;
    
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        GetSparseUnpadBlockInfo();  // TND变长分核
    } else if (fBaseParams.isSparse) {
        if (fBaseParams.sparseMode == PREFIX || PREFIX_COMPRESS) {
            GetSparsePrefixBlockInfo();  // 前缀稀疏分核
        } else {
            GetSparseBlockInfo();  // 普通稀疏分核
        }
    } else {
        // Normal均匀切分
        int64_t fusedOuter = b × n2 × g × s1Outer × s2Outer;
        int64_t blockFactor = ceil(fusedOuter / aicNum);
        // 设置blockStarts/blockEnds
    }
    
    // Step5: 确定性数组计算
    if (fBaseParams.deterSparseType == DETER_OLD) {
        GetIsDeterArr();
    }
    
    return ge::GRAPH_SUCCESS;
}
```

---

## 4. BN2S2分核策略

### 4.1 DoBn2s2Sparse - BN2S2分核尝试

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:316`

**功能**: 按B→N2→S2轴切分，用于负载均衡优化。

**适用场景**:
- TND变长场景（不同batch长度差异大）
- S2维度较大
- 需要更好的负载均衡

**切分逻辑**:
```
┌─────────────────────────────────────────────────────────────┐
│                  BN2S2切分示意图                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  参数: b=4, n2=2, g=1, s2Outer=8                            │
│                                                             │
│  原始索引空间:                                               │
│  fusedOuter = b × n2 × g × s2Outer                         │
│            = 4 × 2 × 1 × 8 = 64                             │
│                                                             │
│  每个block包含完整的s1Outer:                                │
│  blockFactor = ceil(64 / aicNum) × s1Outer                 │
│                                                             │
│  分核示意 (aicNum=4):                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  BN2S2索引空间:                                       │  │
│  │                                                       │  │
│  │  B=0, N2=0: S2=[0,1,2,3,4,5,6,7]                      │  │
│  │  B=0, N2=1: S2=[0,1,2,3,4,5,6,7]                      │  │
│  │  B=1, N2=0: S2=[0,1,2,3,4,5,6,7]                      │  │
│  │  B=1, N2=1: S2=[0,1,2,3,4,5,6,7]                      │  │
│  │  ...                                                  │  │
│  │                                                       │  │
│  │  分核分配:                                            │  │
│  │  Core0: B0N2G0S2[0-15] × s1Outer                     │  │
│  │  Core1: B0N2G0S2[16-31] × s1Outer                    │  │
│  │  Core2: B1N2G0S2[0-15] × s1Outer                     │  │
│  │  Core3: B1N2G0S2[16-31] × s1Outer                    │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  优点:                                                      │
│  ├─ S2维度可以并行，适合长KV序列                            │
│  ├─ TND负载均衡更灵活                                       │
│  └─ DK/DV输出可能有重叠（需要同步）                         │
│                                                             │
│  缺点:                                                      │
│  └─ 需要额外的同步机制处理DK/DV重叠                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码详解**:
```cpp
bool FlashAttentionScoreGradTilingNormalRegbase::DoBn2s2Sparse()
{
    // 判断是否支持BN2S2切分
    if (fBaseParams.splitAxis != SplitAxisEnum::BN2S2 ||
        fBaseParams.deterSparseType == DETER_OLD) {
        return false;
    }
    
    // TND或稀疏场景
    if (fBaseParams.isSparse || fBaseParams.layoutType == INPUT_FORMAT_TND) {
        return GetBlockInfoOfBNS4TND();  // 特殊处理
    }
    
    // Normal场景BN2S2切分
    int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s2Outer;
    int64_t bns2Factor = ceil(fusedOuter / aicNum);
    int64_t blockOuter = ceil(fusedOuter / bns2Factor);
    int64_t totalBlock = fusedOuter * fBaseParams.s1Outer;
    int64_t blockFactor = bns2Factor * fBaseParams.s1Outer;
    
    // 设置blockStarts/blockEnds
    for (int64_t i = 0; i < blockOuter; i++) {
        blockStarts[i] = blockFactor * i;
        blockEnds[i] = min(blockFactor * (i + 1), totalBlock);
    }
    
    std::copy(blockStarts, fBaseParams.blockStarts);
    std::copy(blockEnds, fBaseParams.blockEnds);
    
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;
    
    return true;
}
```

---

## 5. TND变长分核

### 5.1 GetBlockInfoOfTNDForBn2 - TND BN2S2负载均衡分核

**位置**: `flash_attention_score_grad_tiling_varlen_regbase.cpp:782`

**功能**: TND变长场景的负载均衡分核，确保各核负载均匀。

**核心算法**: 二分搜索找最优每核block数。

**流程图**:
```
┌─────────────────────────────────────────────────────────────┐
│              TND负载均衡分核流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: actualSeqQlen[64], actualSeqKvlen[64]                │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  Step1: 计算每个batch的有效block信息                  │  │
│  │  FillBlockInfoLoadBalanceForBn2()                    │  │
│  │                                                       │  │
│  │  ├─ 遍历每个batch                                    │  │
│  │  ├─ 计算actualS1Outer × actualS2Outer                │  │
│  │  ├─ 考虑稀疏token范围                                │  │
│  │  ├─ 记录acturalBlockInfo[b][s2Outer]                 │  │
│  │                                                       │  │
│  │  Step2: 二分搜索最优分配                              │  │
│  │  BinarySearchMaxBlockNumPerCore()                    │  │
│  │                                                       │  │
│  │  ├─ 初始范围: [下界, 上界]                           │  │
│  │  ├─ 检查是否可以均匀分配                             │  │
│  │  ├─ 调整搜索范围                                     │  │
│  │  ├─ 找到最小的maxBlockNumPerCore                    │  │
│  │                                                       │  │
│  │  Step3: 分配block到各core                            │  │
│  │  CaclePerCoreBlockInfoBn2()                          │  │
│  │                                                       │  │
│  │  ├─ 累积block数                                      │  │
│  │  ├─ 达到maxBlockNumPerCore时切换到下一个core        │  │
│  │  ├─ 记录blockStarts[coreIdx]                         │  │
│  │  ├─ 记录blockEnds[coreIdx]                           │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  输出: blockStarts[64], blockEnds[64], blockOuter           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**负载均衡示意**:
```
┌─────────────────────────────────────────────────────────────┐
│               TND负载均衡示例                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入:                                                      │
│  Batch 0: seqQLen=128, seqKvLen=256 → 16 blocks            │
│  Batch 1: seqQLen=512, seqKvLen=1024 → 64 blocks           │
│  Batch 2: seqQLen=256, seqKvLen=512 → 32 blocks            │
│  Batch 3: seqQLen=64,  seqKvLen=128 → 8 blocks             │
│                                                             │
│  总block数: 16+64+32+8 = 120 blocks                        │
│  aicNum = 4                                                 │
│                                                             │
│  简单切分（不均衡）:                                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Core0: 30 blocks                                     │  │
│  │  Core1: 30 blocks                                     │  │
│  │  Core2: 30 blocks                                     │  │
│  │  Core3: 30 blocks                                     │  │
│  │                                                       │  │
│  │  问题: Batch1(64)跨Core0和Core1，负载不均衡          │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  负载均衡切分:                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  目标: 每核约30 blocks，但考虑batch边界              │  │
│  │                                                       │  │
│  │  Core0: Batch0(16) + Batch1(前半32) = 48             │  │
│  │  Core1: Batch1(后半32) = 32                          │  │
│  │  Core2: Batch2(32) = 32                              │  │
│  │  Core3: Batch3(8) = 8                                │  │
│  │                                                       │  │
│  │  负载比: 48 : 32 : 32 : 8                            │  │
│  │                                                       │  │
│  │  进一步优化（调整batch分界点）:                       │  │
│  │  Core0: Batch0 + Batch1前16 = 32                    │  │
│  │  Core1: Batch1后48 = 48                              │  │
│  │  Core2: Batch2(32) = 32                              │  │
│  │  Core3: Batch3(8) + 剩余 = 8                         │  │
│  │                                                       │  │
│  │  最终负载比: 32 : 48 : 32 : 8                        │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 FillBlockInfoLoadBalanceForBn2 - 填充负载信息

**位置**: `flash_attention_score_grad_tiling_varlen_regbase.cpp:815`

**功能**: 计算每个batch的有效block数，用于负载均衡。

```cpp
void FillBlockInfoLoadBalanceForBn2(
    std::vector<std::vector<int64_t>> &totalBlockInfo,
    std::vector<std::vector<float>> &acturalBlockInfo)
{
    // 初始化
    acturalBlockInfo[b][0] = 0;      // 存累积block数
    acturalBlockInfo[b+1][0] = 0;    // 存最大block数
    
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];
        
        // 计算outer
        auto actualS1Outer = ceil(actualS1Len / s1CvInner);
        auto actualS2Outer = ceil(actualS2Len / cvS2Inner);
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;
        
        // 考虑稀疏token范围
        int64_t actualCalcS1Token, actualCalcS2Token;
        CalcleActualToken(fBaseParams, i, actualCalcS1Token, actualCalcS2Token);
        
        // 计算每个s2Outer的有效s1范围
        for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
            if (cvS2Inner * j >= actualS2Len) {
                acturalBlockInfo[i][j] = 0;
            } else {
                // 计算有效s1范围
                int64_t leftIntersection = max(cvS2Inner*j - actualCalcS2Token, 0);
                float acturalS1Begin = min(leftIntersection, actualS1Len);
                float acturalS1End = min(actualS1Len, cvS2Inner*j + cvBlockTail + actualCalcS1Token);
                float acturalS1Num = acturalS1End - acturalS1Begin;
                
                acturalBlockInfo[i][j] = acturalS1Num / s1CvInner;
                acturalBlockInfo[b][0] += acturalBlockInfo[i][j] * n2 * g;
            }
        }
        
        // 累积block数
        totalBlockInfo[i][1] = n2 * g * totalBlockInfo[i][0] + totalBlockInfo[i-1][1];
    }
}
```

---

## 6. 稀疏模式分核

### 6.1 GetSparseBlockInfo - 稀疏分核信息

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:1272`

**功能**: 计算稀疏模式下的有效block范围。

**核心步骤**:
1. 计算每个S2 Outer的有效S1 Outer范围
2. 累积有效block数
3. 按有效block数均匀切分到各core

```cpp
ge::graphStatus GetSparseBlockInfo()
{
    // Step1: 解析每个s2Outer的有效s1Outer范围
    int64_t parseInfo[s2Outer][3];  // [begin, end, length]
    GetParseS1S2OuterInfo(parseInfo);
    
    // 计算总有效block数
    int64_t s1s2oCount = parseInfo[s2Outer-1][LENGTH_IDX];
    
    // Step2: 按有效block切分
    int64_t fusedOuter = b × n2 × g × s1s2oCount;
    int64_t blockFactor = ceil(fusedOuter / aicNum);
    int64_t blockOuter = ceil(fusedOuter / blockFactor);
    
    // Step3: 计算blockStarts/blockEnds
    blockStarts[0] = 0;
    blockEnds[blockOuter-1] = b × n2 × g × s1Outer × s2Outer;
    
    for (int64_t c = 1; c < blockOuter; c++) {
        int64_t currentIdx = min(c × blockFactor, fusedOuter);
        
        // 解码: currentIdx → (bIdx, n2Idx, gIdx, s1oIdx, s2oIdx)
        bIdx = currentIdx / (n2 × g × s1s2oCount);
        bTail = currentIdx % (n2 × g × s1s2oCount);
        n2Idx = bTail / (g × s1s2oCount);
        ...
        
        // 获取s1oIdx和s2oIdx
        GetCommonS1S2OuterIndex(fBaseParams, parseInfo, gTail, s1oIdx, s2oIdx);
        
        // 计算全局block索引
        blockStarts[c] = ((bIdx × n2 + n2Idx) × g + gIdx) × s2Outer + s2oIdx) × s1Outer + s1oIdx + 1;
        blockEnds[c-1] = blockStarts[c];
    }
    
    return ge::GRAPH_SUCCESS;
}
```

### 6.2 GetParseS1S2OuterInfo - 解析稀疏范围

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:1228`

**功能**: 计算每个S2 Outer块对应的S1 Outer有效范围。

```
┌─────────────────────────────────────────────────────────────┐
│               稀疏范围解析示意图                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  参数: s1=1024, s2=2048, preToken=512, nextToken=512        │
│        s1CvInner=256, cvS2Inner=256                         │
│                                                             │
│  S2 Outer块计算:                                            │
│  s2Outer = ceil(2048/256) = 8                               │
│                                                             │
│  每个S2 Outer的有效S1范围:                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  S2Outer  S2范围       有效S1范围      有效S1Outer数 │  │
│  │                                                       │  │
│  │  j=0     [0,256)      [0,768)         3              │  │
│  │          left=0       end=256+512     ceil(768/256)  │  │
│  │                                                       │  │
│  │  j=1     [256,512)    [0,1024)        4              │  │
│  │          left=max(256-512,0)=0                       │  │
│  │          end=min(512+512,1024)=1024                  │  │
│  │                                                       │  │
│  │  j=2     [512,768)    [0,1280)        5              │  │
│  │          left=max(512-512,0)=0                       │  │
│  │          end=min(768+512,1024)=1024                  │  │
│  │                                                       │  │
│  │  j=3     [768,1024)   [256,1024]      3              │  │
│  │          left=max(768-512,0)=256                     │  │
│  │          end=1024                                    │  │
│  │                                                       │  │
│  │  ...                                                  │  │
│  │                                                       │  │
│  │  parseInfo[j][BEGIN_IDX] = left / s1CvInner         │  │
│  │  parseInfo[j][END_IDX]   = ceil(end / s1CvInner)    │  │
│  │  parseInfo[j][LENGTH_IDX] = 累积有效block数         │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Band稀疏示意图:                                            │
│  ┌─────────────────────────────────────┐                   │
│  │  S2=256  S1=[0,768]   有效           │                   │
│  │  S2=512  S1=[0,1024]  有效           │                   │
│  │  S2=768  S1=[256,1024] 有效          │                   │
│  │  S2=1024 S1=[512,1024] 有效          │                   │
│  │  ...                                │                   │
│  │  带状区域：preToken + nextToken范围   │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 确定性计算相关函数

### 7.1 GetIsDeterArr - 确定性数组计算

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:725`

**功能**: 遍历所有block，检测dq/dk/dv输出位置重叠，设置同步标记。

**算法流程**:
```
┌─────────────────────────────────────────────────────────────┐
│                  确定性检测算法                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: blockStarts[64], blockEnds[64]                       │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  初始化:                                              │  │
│  │  ├─ dqOffset[64], dkDvOffset[64]  // 记录各核偏移    │  │
│  │  ├─ loopIdx[64] = 0  // 各核当前loop索引             │  │
│  │  ├─ dqNeedDeterpre, dkDvNeedDeterpre = false        │  │
│  │                                                       │  │
│  │  主循环: while (calcNum < maxValidBBLen)             │  │
│  │                                                       │  │
│  │  ├─ 遍历每个core (cBlockIdx = 0..blockOuter-1)      │  │
│  │  │                                                   │  │
│  │  │  ├─ 跳过无效block                                │  │
│  │  │  │  while (!IsValid(blockIdx) && in_range)       │  │
│  │  │  │    loopIdx[cBlockIdx++];                      │  │
│  │  │                                                   │  │
│  │  │  ├─ 计算offset                                   │  │
│  │  │  │  GetOffset(fBaseParams, dqOffset, dkDvOffset, │  │
│  │  │  │              validBlockIdx);                  │  │
│  │  │                                                   │  │
│  │  │  ├─ 检测重叠                                     │  │
│  │  │  │  JudgeIsNeedDeter(...);                       │  │
│  │  │                                                   │  │
│  │  ├─ calcNum++                                       │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  输出:                                                      │
│  ├─ dqIsNeedDeter[4]  // 256位标记数组                     │
│  ├─ dkDvIsNeedDeter[4]                                     │
│  └─ noNeedDeter // 是否需要任何同步                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心代码**:
```cpp
void FlashAttentionScoreGradTilingNormalRegbase::GetIsDeterArr()
{
    std::array<int64_t, CORE_LIST_NUM> dqOffset;
    std::array<int64_t, CORE_LIST_NUM> dkDvOffset;
    std::array<int64_t, CORE_LIST_NUM> dqOffsetpre;
    std::array<int64_t, CORE_LIST_NUM> dkDvOffsetpre;
    std::array<int64_t, CORE_LIST_NUM> loopIdx;
    
    bool dqNeedDeterpre = false;
    bool dkDvNeedDeterpre = false;
    int64_t calcNum = 0;
    
    std::fill(loopIdx.begin(), loopIdx.end(), 0);
    
    while (calcNum < fBaseParams.maxValidBBLen) {
        for (uint16_t cBlockIdx = 0; cBlockIdx < fBaseParams.blockOuter; cBlockIdx++) {
            // 跳过无效block
            while (!IsValid(blockStarts[cBlockIdx] + loopIdx[cBlockIdx]) &&
                   (blockStarts[cBlockIdx] + loopIdx[cBlockIdx] < blockEnds[cBlockIdx])) {
                loopIdx[cBlockIdx]++;
            }
            
            if (blockStarts[cBlockIdx] + loopIdx[cBlockIdx] >= blockEnds[cBlockIdx]) {
                dqOffset[cBlockIdx] = OUTINDEX;  // 核已结束
                dkDvOffset[cBlockIdx] = OUTINDEX;
                continue;
            }
            
            int64_t validBlockIdx = blockStarts[cBlockIdx] + loopIdx[cBlockIdx];
            
            // 计算offset
            GetOffset(fBaseParams, dqOffset[cBlockIdx], dkDvOffset[cBlockIdx], validBlockIdx);
            
            loopIdx[cBlockIdx]++;
        }
        
        // 检测重叠
        JudgeIsNeedDeter(fBaseParams, dqOffset, dkDvOffset, 
                        dqOffsetpre, dkDvOffsetpre, calcNum,
                        fBaseParams.noNeedDeter, dqNeedDeterpre, dkDvNeedDeterpre);
        
        calcNum++;
    }
}
```

### 7.2 JudgeIsNeedDeter - 重叠检测

**位置**: `flash_attention_score_grad_tiling_common_regbase.cpp:361`

**功能**: 检查各核的offset是否有重叠，设置同步标记。

```cpp
void JudgeIsNeedDeter(FuzzyBaseInfoParamsRegbase& fBaseParams,
                      std::array<int64_t, CORE_LIST_NUM>& dqOffset,
                      std::array<int64_t, CORE_LIST_NUM>& dkDvOffset,
                      std::array<int64_t, CORE_LIST_NUM>& dqOffsetpre,
                      std::array<int64_t, CORE_LIST_NUM>& dkDvOffsetpre,
                      int64_t calcNum, bool &noNeedDeter,
                      bool &dqNeedDeterpre, bool &dkDvNeedDeterpre)
{
    bool dqNeedDeter = false;
    bool dkDvNeedDeter = false;
    
    // 检查当前核之间的重叠
    for (uint16_t i = 0; i < fBaseParams.blockOuter - 1; i++) {
        for (uint16_t j = i + 1; j < fBaseParams.blockOuter; j++) {
            if (!dqNeedDeter && dqOffset[i] == dqOffset[j] && dqOffset[i] != OUTINDEX) {
                dqNeedDeter = true;  // dq有重叠
            }
            if (!dkDvNeedDeter && dkDvOffset[i] == dkDvOffset[j] && dkDvOffset[i] != OUTINDEX) {
                dkDvNeedDeter = true;  // dk/dv有重叠
            }
        }
    }
    
    // 检查与前一轮的重叠
    if (calcNum != 0 && ((!dqNeedDeter && dqNeedDeterpre) || (!dkDvNeedDeter && dkDvNeedDeterpre))) {
        for (uint16_t i = 0; i < fBaseParams.blockOuter; i++) {
            for (uint16_t j = 0; j < fBaseParams.blockOuter; j++) {
                if (!dqNeedDeter && dqNeedDeterpre && 
                    dqOffset[i] == dqOffsetpre[j] && dqOffset[i] != OUTINDEX) {
                    dqNeedDeter = true;
                }
                if (!dkDvNeedDeter && dkDvNeedDeterpre && 
                    dkDvOffset[i] == dkDvOffsetpre[j] && dkDvOffset[i] != OUTINDEX) {
                    dkDvNeedDeter = true;
                }
            }
        }
    }
    
    // 更新前一轮offset
    dqNeedDeterpre = dqNeedDeter;
    dkDvNeedDeterpre = dkDvNeedDeter;
    for (uint16_t i = 0; i < fBaseParams.blockOuter; i++) {
        dqOffsetpre[i] = dqOffset[i];
        dkDvOffsetpre[i] = dkDvOffset[i];
    }
    
    // 更新同步标记数组（256位，每位对应一个calcNum）
    int64_t index = calcNum / 64;
    int64_t bitPosition = calcNum % 64;
    
    if (index >= 0 && index < 4) {
        if (dqNeedDeter) {
            fBaseParams.dqIsNeedDeter[index] |= (1ULL << bitPosition);
        } else {
            fBaseParams.dqIsNeedDeter[index] &= ~(1ULL << bitPosition);
        }
        if (dkDvNeedDeter) {
            fBaseParams.dkDvIsNeedDeter[index] |= (1ULL << bitPosition);
        } else {
            fBaseParams.dkDvIsNeedDeter[index] &= ~(1ULL << bitPosition);
        }
    }
    
    noNeedDeter = noNeedDeter && !dqNeedDeter && !dkDvNeedDeter;
}
```

### 7.3 GetOffset - 计算输出偏移

**位置**: `flash_attention_score_grad_tiling_common_regbase.cpp:416`

**功能**: 根据blockIdx计算dq和dk/dv的输出偏移地址。

**公式推导**:
```
┌─────────────────────────────────────────────────────────────┐
│                   Offset计算公式                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: blockIdx                                             │
│                                                             │
│  解码blockIdx → (boIdx, n2oIdx, goIdx, s1oIdx, s2oIdx)      │
│                                                             │
│  BN2GS2D Layout:                                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  dqOffset计算:                                        │  │
│  │  dq[b][n2][g][s1][d]                                  │  │
│  │                                                       │  │
│  │  bOffset = boIdx × (n2 × g × s1 × d)                 │  │
│  │  n2Offset = n2oIdx × (g × s1 × d)                    │  │
│  │  gOffset = goIdx × (s1 × d)                          │  │
│  │  s1Offset = s1oIdx × (s1Inner × 2 × d)  // CVRatio=2 │  │
│  │                                                       │  │
│  │  dqOffset = bOffset + n2Offset + gOffset + s1Offset │  │
│  │                                                       │  │
│  │  dk/dvOffset计算:                                     │  │
│  │  dk[b][n2][s2][d]                                     │  │
│  │                                                       │  │
│  │  bOffset = boIdx × (n2 × s2 × d)                     │  │
│  │  n2Offset = n2oIdx × (s2 × d)                        │  │
│  │  s2Offset = s2oIdx × (s2Inner × d)                   │  │
│  │                                                       │  │
│  │  dkDvOffset = bOffset + n2Offset + s2Offset         │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  TND Layout:                                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  dqOffset:                                            │  │
│  │  bOffset = sum(actualSeqQlen[0..boIdx-1]) × n2 × g × d│  │
│  │  s1Offset = s1oIdx × s1Inner × 2 × n2 × g × d        │  │
│  │  n2Offset = n2oIdx × g × d                            │  │
│  │  gOffset = goIdx × d                                  │  │
│  │                                                       │  │
│  │  dqOffset = bOffset + s1Offset + n2Offset + gOffset  │  │
│  │                                                       │  │
│  │  dk/dvOffset:                                         │  │
│  │  bOffset = sum(actualSeqKvlen[0..boIdx-1]) × n2 × d  │  │
│  │  s2Offset = s2oIdx × s2Inner × n2 × d                │  │
│  │  n2Offset = n2oIdx × d                                │  │
│  │                                                       │  │
│  │  dkDvOffset = bOffset + s2Offset + n2Offset          │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 确定性参数计算

### 8.1 CalcleTNDDeterParam - TND确定性参数

**位置**: `flash_attention_score_grad_tiling_varlen_regbase.cpp:42`

**功能**: TND场景的确定性计算参数，包括deterMaxRound和同步轮次。

**核心逻辑**:
```cpp
void CalcleTNDDeterParam() override
{
    if (fBaseParams.layoutType != INPUT_FORMAT_TND) {
        return;
    }
    
    // 步长采样（减少内存占用）
    if (fBaseParams.b > fBaseParams.deterPrefixThreshold) {
        fBaseParams.deterPrefixStep = ceil(b / deterPrefixThreshold);
    }
    
    // 初始化前缀和数组
    std::fill(fBaseParams.deterPrefix0, deterPrefix0 + 64, 0);
    std::fill(fBaseParams.deterPrefix1, deterPrefix1 + 64, 0);
    std::fill(fBaseParams.deterPrefix2, deterPrefix2 + 64, 0);
    
    // 根据稀疏类型计算
    CalcleTNDDenseDeterParam();   // DENSE模式
    CalcleTNDCausalDeterParam();  // CAUSAL模式
    CalcleTNDBandDeterParam();    // BAND模式
}
```

### 8.2 CalcleTNDDenseDeterParam - Dense确定性

**功能**: 计算DENSE模式的最大计算轮次deterMaxRound。

```cpp
void CalcleTNDDenseDeterParam()
{
    if (fBaseParams.deterSparseType != DETER_DENSE) {
        return;
    }
    
    DeterPrefixData deterPrefixData;
    int64_t s1Max = 0, s2Max = 0;
    
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Outer = ceil(actualSeqQlen[i] / (s1Inner × s1CvRatio));
        int64_t actualS2Outer = ceil(actualSeqKvlen[i] / (s2Inner × s2CvRatio));
        
        // 前缀和
        deterPrefixData.deterPrefix.push_back(
            deterPrefixData.deterPrefix.back() + actualSeqQlen[i] × actualSeqKvlen[i]);
        deterPrefixData.prefix1.push_back(
            deterPrefixData.prefix1.back() + actualS1Outer × actualS2Outer);
        
        s1Max = max(s1Max, actualS1Outer);
        s2Max = max(s2Max, actualS2Outer);
    }
    
    // 最大轮次
    int64_t totalArea = deterPrefixData.prefix1.back() × n1;
    fBaseParams.deterMaxRound = max({
        ceil(totalArea / aicNum),
        s1Max × g,
        s2Max
    });
    
    // 步长采样
    deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix1, deterPrefixStep);
    deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, deterPrefixStep);
    
    // 拷贝到fBaseParams
    std::copy(deterPrefixData.prefix0, fBaseParams.deterPrefix0);
    std::copy(deterPrefixData.deterPrefix, fBaseParams.deterPrefix);
}
```

---

## 9. 同步轮次计算

### 9.1 CalcleTNDDeterSyncRounds - 同步轮次设置

**位置**: `flash_attention_score_grad_tiling_varlen_regbase.cpp:696`

**功能**: 计算需要同步的轮次范围，用于DK/DV输出重叠同步。

```cpp
void CalcleTNDDeterSyncRounds(
    std::vector<std::pair<uint64_t, uint64_t>> &syncRounds,
    std::vector<std::pair<uint64_t, uint64_t>> &syncRoundRanges)
{
    // 如果同步点过多，全部同步
    if (syncRounds.size() + syncRoundRanges.size() > CORE_LIST_NUM) {
        fBaseParams.startNeedSyncRound[0] = 1;
        fBaseParams.endNeedSyncRound[0] = UINT64_MAX;
        return;
    }
    
    // 计算需要同步的具体轮次
    std::vector<uint64_t> needSyncRoundsTmp = CalculateSyncRound(syncRounds);
    
    // 过滤掉已经在范围内的轮次
    std::vector<uint64_t> needSyncRounds;
    for (uint64_t needSyncRound : needSyncRoundsTmp) {
        bool isValid = true;
        for (auto needSyncRoundRange : syncRoundRanges) {
            if (needSyncRound >= needSyncRoundRange.first && 
                needSyncRound <= needSyncRoundRange.second) {
                isValid = false;
                break;
            }
        }
        if (isValid) {
            needSyncRounds.push_back(needSyncRound);
        }
    }
    
    // 设置同步轮次
    for (uint32_t i = 0; i < needSyncRounds.size(); i++) {
        fBaseParams.startNeedSyncRound[i] = needSyncRounds[i];
        fBaseParams.endNeedSyncRound[i] = needSyncRounds[i];
    }
    
    // 设置同步范围
    for (uint32_t i = 0; i < syncRoundRanges.size(); i++) {
        fBaseParams.startNeedSyncRound[i + needSyncRounds.size()] = syncRoundRanges[i].first;
        fBaseParams.endNeedSyncRound[i + needSyncRounds.size()] = syncRoundRanges[i].second;
    }
    
    // 如果同步开销过大，切换到BN2GS1S2切分
    uint64_t allNeedSyncLoopNums = 0;
    for (uint64_t loopIdx = 0; loopIdx < deterMaxRound; loopIdx++) {
        for (uint32_t i = 0; i < CORE_LIST_NUM; i++) {
            if (loopIdx >= startNeedSyncRound[i] && loopIdx <= endNeedSyncRound[i]) {
                allNeedSyncLoopNums++;
                break;
            }
        }
    }
    
    if (deterMaxRound < allNeedSyncLoopNums × 2) {
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;  // 切换切分策略
    }
}
```

---

## 10. Workspace计算

### 10.1 GetWorkspaceSize - Workspace分配

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:1055`

**功能**: 计算算子所需的Workspace大小，包括dq/dk/dv累加空间。

```cpp
ge::graphStatus GetWorkspaceSize()
{
    size_t workspaceSize = RESERVED_WORKSPACE_SIZE;
    
    // 计算Q/K/V大小
    int64_t qSize = ((b × n1 - 1) × s1 + align(s1, 128)) × d;
    int64_t kSize = ((b × n2 - 1) × s2 + align(s2, 128)) × d;
    int64_t vSize = ((b × n2 - 1) × s2 + align(s2, 128)) × d1;
    
    if (layout == TND) {
        qSize = align(t1 × n1, 128) × d;
        kSize = align(t2 × n2, 128) × d;
        vSize = align(t2 × n2, 128) × d1;
    }
    
    // BN2S2切分需要额外的workspace
    if (splitAxis == BN2S2) {
        postTilingData_->set_dqWorkSpaceOffset(workspaceSize);
        workspaceSize += qSize × FP32_BYTES;  // dq累加
        
        postTilingData_->set_dkWorkSpaceOffset(workspaceSize);
        workspaceSize += s2Inner × sfmgdInner × 64 × FP32_BYTES;  // dk
        
        postTilingData_->set_dvWorkSpaceOffset(workspaceSize);
        workspaceSize += s2Inner × sfmgdInner × 64 × FP32_BYTES;  // dv
    }
    
    // 确定性workspace
    GetWorkspaceSize4Deter(workspaceSize);
    
    // Dropout mask workspace
    if (!dropoutIsDivisibleBy8) {
        postTilingData_->set_dropMaskGmOffset(workspaceSize);
        workspaceSize += dropMaskSize;
    }
    
    // FP8 softmax grad workspace
    if (queryType == HIFLOAT8) {
        postTilingData_->set_sfmgWorkSpaceOffset(workspaceSize);
        uint64_t sfmgSize = ((b × n2 × g - 1) × s1 + align(s1, 128)) × 8;
        workspaceSize += sfmgSize × FP32_BYTES;
    }
    
    workspaceSize += WORKSPACE_BUFFER;
    workspaces[0] = workspaceSize;
    
    return ge::GRAPH_SUCCESS;
}
```

---

## 11. 分核策略总结

### 11.1 分核策略对比

```
┌─────────────────────────────────────────────────────────────┐
│                 分核策略对比表                                │
├───────────────┬───────────────┬───────────────┬─────────────┤
│  策略         │  适用场景     │  优点         │  缺点        │
├───────────────┼───────────────┼───────────────┼─────────────┤
│ BN2GS1S2      │ 一般场景      │ 无输出重叠    │ 负载可能     │
│               │ GQA(g>1)      │ 无需同步      │ 不均衡       │
│               │ 确定性计算    │               │              │
├───────────────┼───────────────┼───────────────┼─────────────┤
│ BN2S2         │ TND变长       │ 负载均衡好    │ DK/DV重叠   │
│               │ S2较大        │ S2并行        │ 需同步      │
│               │ Band稀疏      │               │              │
├───────────────┼───────────────┼───────────────┼─────────────┤
│ BN2多块       │ 大batch       │ 多核并行BN   │ 有无效行列  │
│               │ S1×S2大       │ 减少核数     │ 时性能差    │
├───────────────┼───────────────┼───────────────┼─────────────┤
│ Swizzle       │ 超L2缓存      │ 数据重排     │ 仅非确定性  │
│               │ 大无效块      │ 缓存优化     │              │
└───────────────┴───────────────┴───────────────┴─────────────┘
```

### 11.2 关键函数调用关系

```
┌─────────────────────────────────────────────────────────────┐
│                  函数调用关系图                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DoOpTiling()                                               │
│  ├─ SetSplitAxis()                                          │
│  ├─ DoSplit()                                               │
│  │  └─ FuzzyForBestSplit()                                  │
│  │     ├─ GetS1S2TemplateType()                             │
│  │     └─ GetDTemplateType()                                │
│  │                                                          │
│  ├─ DoSparse() ← 核心分核入口                                │
│  │  ├─ GetSparseType()                                      │
│  │  ├─ GetDeterSparseTilingKey()                            │
│  │  ├─ CalcleDeterParam()                                   │
│  │  │  ├─ CalcleTNDDeterParam()                             │
│  │  │  │  ├─ CalcleTNDDenseDeterParam()                     │
│  │  │  │  ├─ CalcleTNDCausalDeterParam()                    │
│  │  │  │  └─ CalcleTNDBandDeterParam()                      │
│  │  │  ├─ CalcleCausalDeterParam()                          │
│  │  │  └─ CalcleBandDeterParam()                            │
│  │  │                                                       │
│  │  ├─ DoBn2s2Sparse()                                      │
│  │  │  └─ GetBlockInfoOfBNS4TND()                           │
│  │  │                                                       │
│  │  ├─ DoBn2MultiBlkSparse()                                │
│  │  │  ├─ GetBlockInfoOfTNDForBn2()                         │
│  │  │  │  ├─ FillBlockInfoLoadBalanceForBn2()               │
│  │  │  │  ├─ BinarySearchMaxBlockNumPerCore()               │
│  │  │  │  └─ CaclePerCoreBlockInfoBn2()                     │
│  │  │  ├─ GetSparseBlockInfoBn2()                           │
│  │  │  └─ Normal均匀切分                                    │
│  │  │                                                       │
│  │  ├─ GetSparseUnpadBlockInfo()                            │
│  │  │  ├─ FillBlockInfo()                                   │
│  │  │  └─ TND特殊处理                                       │
│  │  │                                                       │
│  │  ├─ GetSparseBlockInfo()                                 │
│  │  │  ├─ GetParseS1S2OuterInfo()                           │
│  │  │  └─ GetCommonS1S2OuterIndex()                         │
│  │  │                                                       │
│  │  ├─ GetSparsePrefixBlockInfo()                           │
│  │  │  ├─ GetCommS1S2OuterInfo()                            │
│  │  │  └─ SetSparsePrefixBlockInterval()                    │
│  │  │                                                       │
│  │  └─ GetIsDeterArr()                                      │
│  │     ├─ IsValid()                                         │
│  │     ├─ GetOffset()                                       │
│  │     └─ JudgeIsNeedDeter()                                │
│  │                                                          │
│  ├─ CheckExceedL2Cache()                                    │
│  ├─ InitTilingData()                                        │
│  ├─ DoPreTiling()                                           │
│  ├─ DoPostTiling()                                          │
│  ├─ GetWorkspaceSize()                                      │
│  │  └─ GetWorkspaceSize4Deter()                             │
│  ├─ GetTilingKey()                                          │
│  └─ PostTiling()                                            │
│     └─ SaveToTilingData()                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 附录: 关键数据结构

### A.1 FuzzyBaseInfoParamsRegbase - 核心参数结构

```cpp
struct FuzzyBaseInfoParamsRegbase {
    // Shape参数
    int64_t b, n1, n2, g, s1, s2, d, d1;
    
    // Tiling切分参数
    int64_t s1Inner, s2Inner;       // 基本块尺寸
    int64_t s1CvInner, cvS2Inner;   // Cube处理尺寸
    int64_t s1Outer, s2Outer;       // 外层循环次数
    int64_t s1Tail, s2Tail;         // 尾部尺寸
    
    // 分核参数
    int64_t blockOuter;             // Block外层循环
    int64_t blockFactor;            // 每核Block数
    int64_t blockStarts[64];        // Block起始
    int64_t blockEnds[64];          // Block结束
    int64_t maxValidBBLen;          // 最大有效Block
    
    // 稀疏参数
    uint32_t sparseMode;            // 稀疏模式
    uint8_t sparseType;             // DENSE/CASUAL/BAND
    int64_t s1Token, s2Token;       // Band范围
    
    // 确定性参数
    uint32_t deterSparseType;       // 确定性类型
    int64_t deterMaxRound;          // 最大轮次
    uint64_t dqIsNeedDeter[4];      // dq同步标记(256位)
    uint64_t dkDvIsNeedDeter[4];    // dk/dv同步标记
    int64_t deterPrefix[64];        // 前缀和
    int64_t deterPrefix0[64];
    int64_t deterPrefix1[64];
    int64_t deterPrefix2[64];
    uint64_t startNeedSyncRound[64];
    uint64_t endNeedSyncRound[64];
    
    // TND参数
    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;
    int64_t t1, t2;
    
    // 架构参数
    uint32_t coreNum;               // Vector核数
    uint32_t aicNum;                // Cube核数
    uint64_t ubSize;                // UB大小
    uint64_t l2CacheSize;           // L2缓存大小
};
```

---

**文档版本**: v1.0  
**生成日期**: 2025-05-19  
**适用架构**: Ascend 950 Regbase (DAV_3510)  
**关键文件**: 
- `flash_attention_score_grad_tiling_normal_regbase.cpp`
- `flash_attention_score_grad_tiling_varlen_regbase.cpp`
- `flash_attention_score_grad_tiling_common_regbase.cpp`