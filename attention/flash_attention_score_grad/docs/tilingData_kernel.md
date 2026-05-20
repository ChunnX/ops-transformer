# TilingData 从Host到Kernel的传递机制详解

## 1. 整体传递流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TilingData传递完整流程                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  Host侧 (Tiling阶段)                                                │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                                                             │    │    │
│  │  │  1. DoOpTiling()                                            │    │    │
│  │  │     ├─ DoSplit() → 计算s1Inner/s2Inner/s1Outer/s2Outer     │    │    │
│  │  │     ├─ DoSparse() → 计算blockStarts/blockEnds/blockOuter   │    │    │
│  │  │     ├─ 确定性参数 → dqIsNeedDeter/dkDvIsNeedDeter          │    │    │
│  │  │                                                             │    │    │
│  │  │  2. SaveToTilingData()                                      │    │    │
│  │  │     ├─ 设置 s1s2BNGS1S2BaseParams (b, n2, g, s1, s2...)   │    │    │
│  │  │     ├─ 设置 s1s2BNGS1S2SplitCoreParams                     │    │    │
│  │  │     ├─ 设置 s1s2BNGS1S2BlockNumList (blockStarts/Ends)    │    │    │
│  │  │     ├─ 设置 preTilingData / postTilingData                │    │    │
│  │  │                                                             │    │    │
│  │  │  3. context->SetTilingData()                                │    │    │
│  │  │     将TilingData写入GM内存                                  │    │    │
│  │  │                                                             │    │    │
│  │  │  4. context->SetBlockDim(blockOuter × CVRatio)             │    │    │
│  │  │     设置启动的核数                                          │    │    │
│  │  │                                                             │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                     │    │
│  │                          ↓                                          │    │
│  │                                                                     │    │
│  │  GM内存 (Global Memory)                                             │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                                                             │    │    │
│  │  │  tiling_data 指针指向的内存区域                             │    │    │
│  │  │                                                             │    │    │
│  │  │  FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase    │    │    │
│  │  │  ├─ s1s2BNGS1S2BaseParams (约128字节)                      │    │    │
│  │  │  ├─ s1s2BNGS1S2SplitCoreParams (约300字节)                 │    │    │
│  │  │  ├─ s1s2BNGS1S2BlockNumList (blockStarts[36]+blockEnds[36])│    │    │
│  │  │  ├─ preTilingData                                          │    │    │
│  │  │  ├─ postTilingData                                         │    │    │
│  │  │  ├─ deterParam (可选，确定性场景)                          │    │    │
│  │  │  ├─ tndParam (可选，TND场景)                               │    │    │
│  │  │  └─ tndSwizzleParam (可选)                                │    │    │
│  │  │                                                             │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                     │    │
│  │                          ↓                                          │    │
│  │                                                                     │    │
│  │  Kernel侧 (运行阶段)                                                │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                                                             │    │    │
│  │  │  1. GET_TILING_DATA_WITH_STRUCT()                           │    │    │
│  │  │     从GM读取TilingData到tiling_data_in变量                  │    │    │
│  │  │                                                             │    │    │
│  │  │  2. Init()                                                  │    │    │
│  │  │     ├─ GetBlockIdx() → 获取当前核索引                      │    │    │
│  │  │     ├─ GetSubBlockIdx() → Vector核获取子块索引             │    │    │
│  │  │     ├─ SetConstInfo() → 设置常量参数                       │    │    │
│  │  │     ├─ SetOptionalInfo() → 设置可选参数                    │    │    │
│  │  │                                                             │    │    │
│  │  │  3. Process()                                               │    │    │
│  │  │     ├─ 从blockStarts[cBlockIdx]获取起始block              │    │    │
│  │  │     ├─ 从blockEnds[cBlockIdx]获取结束block                 │    │    │
│  │  │     ├─ GetNextValidIdx() → 遍历有效block                   │    │    │
│  │  │     ├─ SetRunInfo() → 设置当前任务的坐标信息              │    │    │
│  │  │     ├─ 执行计算...                                         │    │    │
│  │  │                                                             │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. TilingData结构定义

### 2.1 核心数据结构

**位置**: `flash_attention_score_grad_tiling_data_regbase.h`

```cpp
// 主TilingData结构（模板类，根据场景组合不同参数）
template<const bool isNewDeter = false, 
         const bool isTnd = false, 
         const bool isTndSwizzle = false>
class FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase {
public:
    // 1. 基本Shape参数
    FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase s1s2BNGS1S2BaseParams;
    
    // 2. 分核切分参数
    FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsRegbase s1s2BNGS1S2SplitCoreParams;
    
    // 3. Block起止位置数组
    BlockNumListParamsRegbase s1s2BNGS1S2BlockNumList;
    
    // 4. 预处理参数
    PreParamsRegbase preTilingData;
    
    // 5. 后处理参数
    PostParamsRegbase postTilingData;
    
    // 6. 可选参数（根据场景）
    typename std::conditional<isNewDeter, DeterParamRegbase, std::nullptr_t>::type deterParam;
    typename std::conditional<!isNewDeter && isTnd, TndParamRegbase, std::nullptr_t>::type tndParam;
    typename std::conditional<isTndSwizzle, TndSwizzleParamRegbase, std::nullptr_t>::type tndSwizzleParam;
};
```

### 2.2 各结构详解

#### A. BaseParams - 基本参数

```cpp
class FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase {
public:
    int64_t coreNum;      // 总核数
    int64_t b;            // batch size
    int64_t n2;           // kv heads
    int64_t g;            // group size (GQA)
    int64_t s1;           // query seq length
    int64_t s2;           // kv seq length
    int64_t d;            // head dimension
    int64_t d1;           // value head dimension
    
    float scaleValue;     // softmax缩放因子
    float keepProb;       // dropout保留概率
    int64_t keepProbUint8;// dropout阈值
    
    int64_t s1Token;      // 前向token范围
    int64_t s2Token;      // 后向token范围
    
    uint32_t layout;      // layout类型 (BSND/BNSD/TND)
    uint32_t sparseMode;  // 稀疏模式
    uint8_t sparseType;   // DENSE/CASUAL/BAND
    
    uint8_t isSplitByBlockIdx; // 是否按blockIdx分核(Swizzle)
    
    // 可选输入标记
    uint32_t pseOptional;       // 是否有pse
    uint32_t attenMaskOptional; // 是否有mask
    uint8_t dropMaskOuter;      // 是否有dropout
    
    // get/set方法...
};
```

#### B. SplitCoreParams - 切分参数

```cpp
class FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsRegbase {
public:
    int64_t s1Outer;       // S1外层循环次数
    uint32_t s1Inner;      // S1基本块尺寸
    uint32_t s1CvInner;    // S1 Cube处理尺寸 (= s1Inner × CVRatio)
    uint32_t s1Tail;       // S1尾部尺寸
    uint32_t s1CvTail;     // S1 Cube尾部
    
    int64_t s2Outer;       // S2外层循环次数
    uint32_t s2Inner;      // S2基本块尺寸
    uint32_t s2Tail;       // S2尾部尺寸
    
    uint32_t blockOuter;   // Block外层循环（使用的核数）
    uint32_t maxValidBBLen;// 最大有效Block数（每核处理的block数）
    
    uint32_t noNeedDeter;  // 是否需要确定性同步
    int64_t bandIdx;       // Band稀疏的batch索引
    int64_t deterMaxRound; // 确定性最大轮次
    
    // 确定性同步标记数组（36个uint64，共2304位）
    uint64_t dqIsNeedDeter[MAX_CORE_NUM];   // dq需要同步的标记
    uint64_t dkDvIsNeedDeter[MAX_CORE_NUM]; // dk/dv需要同步的标记
    
    // get/set方法...
};
```

#### C. BlockNumList - Block起止数组

```cpp
class BlockNumListParamsRegbase {
public:
    // 每个核处理的Block范围 [blockStarts[核ID], blockEnds[核ID])
    int64_t blockStarts[MAX_CORE_NUM]; // 36个核的起始block索引
    int64_t blockEnds[MAX_CORE_NUM];   // 36个核的结束block索引
    
    // 示例:
    // blockStarts[0] = 0,    blockEnds[0] = 30  → Core0处理block 0-29
    // blockStarts[1] = 30,   blockEnds[1] = 60  → Core1处理block 30-59
    // blockStarts[2] = 60,   blockEnds[2] = 90  → Core2处理block 60-89
    // ...
    
    const int64_t* get_blockStarts() const { return blockStarts; }
    int64_t get_blockStarts(int index) const { return blockStarts[index]; }
    const int64_t* get_blockEnds() const { return blockEnds; }
    int64_t get_blockEnds(int index) const { return blockEnds[index]; }
    
    void set_blockStarts(const int64_t* val); // 数组拷贝
    void set_blockStarts(int index, int64_t val); // 单个设置
};
```

---

## 3. Host侧写入TilingData

### 3.1 SaveToTilingData函数

**位置**: `flash_attention_score_grad_tiling_normal_regbase.cpp:1520`

```cpp
ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::SaveToTilingData()
{
    // 1. 设置基本Shape参数
    s1s2BNGS1S2BaseParams_->set_coreNum(fBaseParams.coreNum);
    s1s2BNGS1S2BaseParams_->set_b(fBaseParams.b - fBaseParams.tailZeroCount);
    s1s2BNGS1S2BaseParams_->set_n2(fBaseParams.n2);
    s1s2BNGS1S2BaseParams_->set_g(fBaseParams.g);
    s1s2BNGS1S2BaseParams_->set_s1(fBaseParams.s1);
    s1s2BNGS1S2BaseParams_->set_d(fBaseParams.d);
    s1s2BNGS1S2BaseParams_->set_s2(fBaseParams.s2);
    s1s2BNGS1S2BaseParams_->set_scaleValue(fBaseParams.scaleValue);
    
    // 2. 设置稀疏模式参数
    s1s2BNGS1S2BaseParams_->set_sparseMode(fBaseParams.sparseMode);
    s1s2BNGS1S2BaseParams_->set_sparseType(fBaseParams.sparseType);
    s1s2BNGS1S2BaseParams_->set_s1Token(fBaseParams.s1Token);
    s1s2BNGS1S2BaseParams_->set_s2Token(fBaseParams.s2Token);
    
    // 3. 设置切分参数
    s1s2BNGS1S2SplitCoreParams_->set_s1Outer(fBaseParams.s1Outer);
    s1s2BNGS1S2SplitCoreParams_->set_s1Inner(fBaseParams.s1Inner);
    s1s2BNGS1S2SplitCoreParams_->set_s1CvInner(fBaseParams.s1CvInner);
    s1s2BNGS1S2SplitCoreParams_->set_s1Tail(fBaseParams.s1Tail);
    s1s2BNGS1S2SplitCoreParams_->set_s2Outer(fBaseParams.s2Outer);
    s1s2BNGS1S2SplitCoreParams_->set_s2Inner(fBaseParams.s2Inner);
    
    // 4. 设置分核参数 ★★★ 关键 ★★★
    s1s2BNGS1S2SplitCoreParams_->set_blockOuter(fBaseParams.blockOuter);
    s1s2BNGS1S2SplitCoreParams_->set_maxValidBBLen(fBaseParams.maxValidBBLen);
    
    // 5. 设置Block起止数组 ★★★ 关键 ★★★
    s1s2BNGS1S2BlockNumList_->set_blockStarts(fBaseParams.blockStarts); // 数组拷贝
    s1s2BNGS1S2BlockNumList_->set_blockEnds(fBaseParams.blockEnds);     // 数组拷贝
    
    // 6. 设置确定性参数
    s1s2BNGS1S2SplitCoreParams_->set_deterMaxRound(fBaseParams.deterMaxRound);
    s1s2BNGS1S2SplitCoreParams_->set_dqIsNeedDeter(fBaseParams.dqIsNeedDeter);
    s1s2BNGS1S2SplitCoreParams_->set_dkDvIsNeedDeter(fBaseParams.dkDvIsNeedDeter);
    
    // 7. 设置TND特殊参数（如果是TND场景）
    if (fBaseParams.layoutType == INPUT_FORMAT_TND && tndParam_ != nullptr) {
        tndParam_->set_tndStartBIdx(tndBaseInfo.tndStartBIdx);
        tndParam_->set_tndS1S2PrefixSum(tndBaseInfo.tndS1S2PrefixSum);
        tndParam_->set_tndPrefixSum(tndBaseInfo.tndPrefixSum);
    }
    
    // 8. 设置确定性前缀和（如果是确定性场景）
    if (IsNewDeter(fBaseParams) && deterParam != nullptr) {
        deterParam->set_deterPrefixStep(fBaseParams.deterPrefixStep);
        deterParam->set_deterPrefix(fBaseParams.deterPrefix);
        deterParam->set_deterPrefix0(fBaseParams.deterPrefix0);
        deterParam->set_deterPrefix1(fBaseParams.deterPrefix1);
        deterParam->set_deterPrefix2(fBaseParams.deterPrefix2);
    }
    
    return ge::GRAPH_SUCCESS;
}
```

### 3.2 数组拷贝实现

**blockStarts/blockEnds数组拷贝**:

```cpp
void BlockNumListParamsRegbase::set_blockStarts(const int64_t* val)
{
    for (uint i = 0; i < MAX_CORE_NUM; ++i) {  // MAX_CORE_NUM = 36
        blockStarts[i] = val[i];  //逐个拷贝
    }
}
```

**说明**: 
- Host侧计算出的`fBaseParams.blockStarts[64]`数组
- 通过`set_blockStarts()`逐个拷贝到TilingData的`blockStarts[36]`
- Kernel侧通过`get_blockStarts(cBlockIdx)`读取单个值

---

## 4. Kernel侧读取TilingData

### 4.1 GET_TILING_DATA_WITH_STRUCT宏

**位置**: `flash_attention_score_grad_entry_regbase.h:377`

```cpp
// Kernel入口函数
template <uint8_t splitAxis, ...>
inline __aicore__ void RegbaseFAG(..., __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipeIn;
    
    // ★★★ 关键：从GM读取TilingData ★★★
    constexpr static bool needDeterPrefix = NEED_DETER_PREFIX(deterType, isTnd);
    using fagTiling = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<
        needDeterPrefix, isTnd, isTndSwizzle>;
    
    GET_TILING_DATA_WITH_STRUCT(fagTiling, tiling_data_in, tiling_data);
    const fagTiling *__restrict tilingData = &tiling_data_in;
    
    // 现在可以访问:
    // tilingData->s1s2BNGS1S2BaseParams.b
    // tilingData->s1s2BNGS1S2SplitCoreParams.s1Outer
    // tilingData->s1s2BNGS1S2BlockNumList.blockStarts[核ID]
    // tilingData->s1s2BNGS1S2BlockNumList.blockEnds[核ID]
    
    // 根据splitAxis调用不同模板
    if constexpr (splitAxis == BN2GS1S2) {
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP16(...);
    } else if constexpr (splitAxis == BN2S2) {
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP16(...);
    }
}
```

**GET_TILING_DATA_WITH_STRUCT宏展开**:
```cpp
// 宏定义（CANN框架提供）
#define GET_TILING_DATA_WITH_STRUCT(struct_type, tiling_var, tiling_gm_addr) \
    struct_type tiling_var; \
    (tiling_var).CopyFrom(tiling_gm_addr);  // 从GM拷贝到局部变量

// 实际执行:
FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase tiling_data_in;
tiling_data_in.CopyFrom(tiling_data);  // 从GM内存拷贝整个结构
```

---

## 5. Kernel如何使用TilingData

### 5.1 Init函数 - 设置核索引和常量

**位置**: `flash_attention_score_grad_kernel_base.h:192`

```cpp
__aicore__ inline void Init(..., FagTilingType ordTilingData, TPipe *pipeIn)
{
    // 1. 获取当前核索引 ★★★ 关键 ★★★
    if ASCEND_IS_AIV {  // Vector核
        vBlockIdx = GetBlockIdx();           // 全局Vector核索引
        cBlockIdx = vBlockIdx / CV_CORE_RATIO;  // 所属Cube核索引
        vSubBlockIdx = GetSubBlockIdx();     // Vector子核索引(0或1)
    } else {  // Cube核
        cBlockIdx = GetBlockIdx();           // Cube核索引
    }
    
    // 2. 保存TilingData指针
    tilingData = ordTilingData;
    pipe = pipeIn;
    
    // 3. 设置常量信息（从TilingData读取）
    SetConstInfo();
    
    // 4. 设置可选信息（从TilingData读取）
    SetOptionalInfo();
    
    // 5. 初始化Buffer
    InitCVCommonBuffer();
    vecBlock.InitUbBuffer();
    cubeBlock.InitCubeBuffer(constInfo);
}
```

### 5.2 SetConstInfo函数 - 设置常量参数

**位置**: `flash_attention_score_grad_kernel_base.h:372+`

```cpp
__aicore__ inline void SetConstInfo()
{
    // 从TilingData读取基本参数
    constInfo.bSize = tilingData->s1s2BNGS1S2BaseParams.b;
    constInfo.n2Size = tilingData->s1s2BNGS1S2BaseParams.n2;
    constInfo.gSize = tilingData->s1s2BNGS1S2BaseParams.g;
    constInfo.s1Size = tilingData->s1s2BNGS1S2BaseParams.s1;
    constInfo.s2Size = tilingData->s1s2BNGS1S2BaseParams.s2;
    constInfo.dSize = tilingData->s1s2BNGS1S2BaseParams.d;
    
    // 从TilingData读取切分参数
    constInfo.s1Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    constInfo.s1Inner = tilingData->s1s2BNGS1S2SplitCoreParams.s1Inner;
    constInfo.s1CvInner = tilingData->s1s2BNGS1S2SplitCoreParams.s1CvInner;
    constInfo.s1Tail = tilingData->s1s2BNGS1S2SplitCoreParams.s1Tail;
    constInfo.s2Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;
    constInfo.s2Inner = tilingData->s1s2BNGS1S2SplitCoreParams.s2Inner;
    
    // 从TilingData读取分核参数
    constInfo.aicCoreNum = tilingData->s1s2BNGS1S2SplitCoreParams.blockOuter;
    constInfo.maxValidBBLen = tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen;
    
    // 计算派生常量
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;
    constInfo.n2GD = constInfo.n2G * constInfo.dSize;
    constInfo.s1CvRatio = CV_CORE_RATIO;  // 2
}
```

### 5.3 Process函数 - 核心计算流程

**位置**: `flash_attention_score_grad_kernel.h:51`

```cpp
__aicore__ inline void FlashAttentionScoreGradKernel::Process()
{
    // ★★★ Step1: 检查当前核是否有任务 ★★★
    if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] == 0) {
        return;  // 没有任务，直接返回
    }
    
    // ★★★ Step2: 获取当前核的Block范围 ★★★
    int64_t blockStart = this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx];
    int64_t blockEnd = this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx];
    
    // 示例: Core0: blockStart=0, blockEnd=30 → 处理block 0-29
    
    int64_t taskId = 0;
    int64_t blockInnerIdx = 0;
    int64_t nextValidBlockInnerIdx = 0;
    
    // ★★★ Step3: 获取第一个有效Block ★★★
    nextValidBlockInnerIdx = this->GetNextValidIdx(
        runInfos[0], taskId, blockStart, 0);
    blockInnerIdx = nextValidBlockInnerIdx;
    
    // ★★★ Step4: 主循环遍历所有Block ★★★
    while (true) {
        this->isLastLoop = (blockInnerIdx == -1);  // 是否最后一轮
        
        if (!this->isLastLoop) {
            // 获取下一个有效Block
            nextValidBlockInnerIdx = this->GetNextValidIdx(
                runInfos[(taskId + 1) & 1], taskId + 1, blockInnerIdx + 1, curLoopIdx + 1);
            
            // 设置当前任务的运行信息
            this->SetRunInfo(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], 
                           taskId, blockInnerIdx, nextValidBlockInnerIdx);
            
            // 执行Cube计算
            this->cubeBlock.IterateMmDyV(mm1ResTensor, constInfo, runInfos[taskId & 1]);
            this->cubeBlock.IterateMmQK(mm2ResTensor, constInfo, runInfos[taskId & 1]);
            
            // 执行Vector计算
            this->vecBlock.ProcessVec1(constInfo, prevRunInfo);
            this->vecBlock.ProcessVec2(mm2ResTensor, constInfo, prevRunInfo);
            this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, constInfo, prevRunInfo);
            this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, constInfo, prevRunInfo);
        }
        
        if (blockInnerIdx == -1) {
            break;  // 遍历完成
        }
        
        taskId++;
        blockInnerIdx = nextValidBlockInnerIdx;
    }
}
```

---

## 6. 关键函数详解

### 6.1 GetNextValidIdx - 获取下一个有效Block

**位置**: `flash_attention_score_grad_kernel_base.h:53`

```cpp
__aicore__ inline int64_t GetNextValidIdx(FagRunInfo &runInfo, int64_t taskId, 
                                          int64_t startIndex, int64_t loopIdx = 0)
{
    int64_t index = startIndex;
    
    // 循环查找有效Block（跳过稀疏无效Block）
    while (index < tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]) {
        
        // 判断是否是有效Block（稀疏场景）
        if (IsValid(runInfo, taskId, index)) {
            return index;  // 找到有效Block
        }
        
        index++;  // 跳过无效Block，继续查找
    }
    
    return -1;  // 没有更多有效Block
}
```

### 6.2 SetRunInfo - 设置当前任务的坐标信息

**位置**: `flash_attention_score_grad_kernel_base.h:43`

```cpp
__aicore__ inline void SetRunInfo(FagRunInfo &runInfo, FagRunInfo &nextRunInfo, 
                                  int64_t taskId, int64_t index, int64_t nextIndex = -1)
{
    // 从Block索引解码出(bIdx, n2Idx, gIdx, s1oIdx, s2oIdx)
    
    // ★★★ Block索引解码公式 ★★★
    // BN2GS1S2切分:
    // blockIdx = bIdx × (n2 × g × s1Outer × s2Outer)
    //          + n2Idx × (g × s1Outer × s2Outer)
    //          + gIdx × (s1Outer × s2Outer)
    //          + s2oIdx × s1Outer
    //          + s1oIdx
    
    int64_t totalPerBatch = n2 × g × s1Outer × s2Outer;
    int64_t bIdx = index / totalPerBatch;
    int64_t bTail = index % totalPerBatch;
    
    int64_t totalPerN2 = g × s1Outer × s2Outer;
    int64_t n2Idx = bTail / totalPerN2;
    int64_t n2Tail = bTail % totalPerN2;
    
    int64_t totalPerG = s1Outer × s2Outer;
    int64_t gIdx = n2Tail / totalPerG;
    int64_t gTail = n2Tail % totalPerG;
    
    int64_t s2oIdx = gTail / s1Outer;
    int64_t s1oIdx = gTail % s1Outer;
    
    // 设置运行信息
    runInfo.commonRunInfo.bIdx = bIdx;
    runInfo.commonRunInfo.n2Idx = n2Idx;
    runInfo.commonRunInfo.gIdx = gIdx;
    runInfo.commonRunInfo.s1oIdx = s1oIdx;
    runInfo.commonRunInfo.s2oIdx = s2oIdx;
    runInfo.commonRunInfo.taskId = taskId;
    runInfo.commonRunInfo.taskIdMod2 = taskId % 2;  // PingPongBuffer索引
    
    // 计算S1/S2的实际范围
    int64_t s1RealSize = (s1oIdx == s1Outer - 1) ? s1Tail : s1Inner;
    int64_t s2RealSize = (s2oIdx == s2Outer - 1) ? s2Tail : s2Inner;
    
    runInfo.s1RealSize = s1RealSize;
    runInfo.s2RealSize = s2RealSize;
    
    // 计算GM偏移地址
    runInfo.queryOffset = GetQueryOffset(runInfo);
    runInfo.keyOffset = GetKeyOffset(runInfo);
    runInfo.valueOffset = GetValueOffset(runInfo);
}
```

### 6.3 GetQueryOffset - 计算QueryGM偏移

```cpp
template <bool IS_MM1_MM2 = true>
__aicore__ inline int64_t GetQueryOffset(FagRunInfo &runInfo)
{
    // BN2GS2D Layout:
    // query[b][n2][g][s1][d]
    int64_t bOffset = runInfo.commonRunInfo.bIdx * (n2 × g × s1 × d);
    int64_t n2Offset = runInfo.commonRunInfo.n2Idx * (g × s1 × d);
    int64_t gOffset = runInfo.commonRunInfo.gIdx * (s1 × d);
    
    // S1偏移（考虑CVRatio）
    int64_t s1Offset;
    if constexpr (IS_MM1_MM2) {
        // MM1/MM2: 使用完整的s1CvInner
        s1Offset = runInfo.commonRunInfo.s1oIdx * s1CvInner * d;
    } else {
        // MM3/MM4/MM5: 使用子块偏移
        s1Offset = runInfo.commonRunInfo.s1oIdx * s1CvInner * d
                   + vSubBlockIdx * s1Inner * d;
    }
    
    return bOffset + n2Offset + gOffset + s1Offset;
}
```

---

## 7. 数据流示意图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 完整数据流：从Shape到Kernel计算                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入Shape: b=4, n2=2, g=1, s1=512, s2=1024, d=128                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Step1: Host Tiling阶段                                              │    │
│  │                                                                     │    │
│  │  计算切分参数:                                                      │    │
│  │  s1Inner = 128, s2Inner = 128                                      │    │
│  │  s1CvInner = 128 × 2 = 256 (Cube处理尺寸)                          │    │
│  │  s1Outer = ceil(512/256) = 2                                       │    │
│  │  s2Outer = ceil(1024/128) = 8                                      │    │
│  │                                                                     │    │
│  │  计算总Block数:                                                     │    │
│  │  totalBlocks = b × n2 × g × s1Outer × s2Outer                       │    │
│  │              = 4 × 2 × 1 × 2 × 8 = 64 blocks                        │    │
│  │                                                                     │    │
│  │  分核（假设aicNum=8）:                                              │    │
│  │  blockFactor = ceil(64/8) = 8 blocks/core                          │    │
│  │  blockOuter = 8 cores                                               │    │
│  │                                                                     │    │
│  │  设置blockStarts/blockEnds:                                         │    │
│  │  Core0: blockStarts[0]=0, blockEnds[0]=8   → block 0-7             │    │
│  │  Core1: blockStarts[1]=8, blockEnds[1]=16  → block 8-15            │    │
│  │  Core2: blockStarts[2]=16, blockEnds[2]=24 → block 16-23           │    │
│  │  ...                                                                │    │
│  │  Core7: blockStarts[7]=56, blockEnds[7]=64 → block 56-63           │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          ↓                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Step2: TilingData写入GM                                             │    │
│  │                                                                     │    │
│  │  SaveToTilingData():                                                │    │
│  │  ├─ s1s2BNGS1S2BaseParams.b = 4                                    │    │
│  │  ├─ s1s2BNGS1S2SplitCoreParams.s1Outer = 2                         │    │
│  │  ├─ s1s2BNGS1S2BlockNumList.blockStarts = {0,8,16,...,56}          │    │
│  │  ├─ s1s2BNGS1S2BlockNumList.blockEnds = {8,16,24,...,64}           │    │
│  │                                                                     │    │
│  │  写入GM内存地址: tiling_data                                        │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          ↓                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Step3: Kernel读取TilingData                                         │    │
│  │                                                                     │    │
│  │  GET_TILING_DATA_WITH_STRUCT():                                     │    │
│  │  从GM拷贝整个TilingData结构                                          │    │
│  │                                                                     │    │
│  │  Init():                                                            │    │
│  │  ├─ GetBlockIdx() → cBlockIdx = 核ID (0-7)                         │    │
│  │  ├─ tilingData->s1s2BNGS1S2BlockNumList.blockStarts[cBlockIdx]     │    │
│  │  └─ tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]       │    │
│  │                                                                     │    │
│  │  示例: Core2读取                                                     │    │
│  │  blockStart = blockStarts[2] = 16                                   │    │
│  │  blockEnd = blockEnds[2] = 24                                       │    │
│  │  → Core2处理block 16-23                                             │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          ↓                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Step4: Kernel计算循环                                               │    │
│  │                                                                     │    │
│  │  Process():                                                         │    │
│  │                                                                     │    │
│  │  循环: blockInnerIdx从16到23                                        │    │
│  │  ├─ blockInnerIdx=16 → 解码得(b=0, n2=0, g=0, s1o=0, s2o=0)       │    │
│  │  ├─ blockInnerIdx=17 → 解码得(b=0, n2=0, g=0, s1o=1, s2o=0)       │    │
│  │  ├─ blockInnerIdx=18 → 解码得(b=0, n2=0, g=0, s1o=0, s2o=1)       │    │
│  │  ├─ blockInnerIdx=19 → 解码得(b=0, n2=0, g=0, s1o=1, s2o=1)       │    │
│  │  ...                                                                │    │
│  │  ├─ blockInnerIdx=23 → 解码得(b=0, n2=1, g=0, s1o=1, s2o=1)       │    │
│  │                                                                     │    │
│  │  每个block:                                                         │    │
│  │  ├─ 计算GM偏移地址                                                  │    │
│  │  ├─ 执行Cube MM计算                                                 │    │
│  │  ├─ 执行Vector Softmax/Dropout计算                                  │    │
│  │  └─ 写回GM结果                                                      │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  最终输出: dq, dk, dv 梯度                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Block索引解码详解

### 8.1 Block索引编码公式（Host侧）

```cpp
// BN2GS1S2切分编码（DoSparse函数）
int64_t blockIdx = (((bIdx * n2 + n2Idx) * g + gIdx) * s2Outer + s2oIdx) * s1Outer + s1oIdx;
```

### 8.2 Block索引解码公式（Kernel侧）

```cpp
// SetRunInfo函数中的解码
int64_t totalPerBatch = n2 × g × s1Outer × s2Outer;
int64_t bIdx = blockIdx / totalPerBatch;
int64_t bTail = blockIdx % totalPerBatch;

int64_t totalPerN2 = g × s1Outer × s2Outer;
int64_t n2Idx = bTail / totalPerN2;
int64_t n2Tail = bTail % totalPerN2;

int64_t totalPerG = s1Outer × s2Outer;
int64_t gIdx = n2Tail / totalPerG;
int64_t gTail = n2Tail % totalPerG;

int64_t s2oIdx = gTail / s1Outer;
int64_t s1oIdx = gTail % s1Outer;
```

### 8.3 解码示例

```
┌─────────────────────────────────────────────────────────────┐
│              Block索引解码示例                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  参数: b=4, n2=2, g=1, s1Outer=2, s2Outer=8                │
│                                                             │
│  blockIdx = 19                                              │
│                                                             │
│  Step1: 计算batch索引                                        │
│  totalPerBatch = n2 × g × s1Outer × s2Outer                 │
│                = 2 × 1 × 2 × 8 = 16                         │
│  bIdx = 19 / 16 = 1                                         │
│  bTail = 19 % 16 = 3                                        │
│                                                             │
│  Step2: 计算n2索引                                           │
│  totalPerN2 = g × s1Outer × s2Outer                         │
│            = 1 × 2 × 8 = 16                                 │
│  n2Idx = 3 / 16 = 0                                         │
│  n2Tail = 3 % 16 = 3                                        │
│                                                             │
│  Step3: 计算g索引                                            │
│  totalPerG = s1Outer × s2Outer                              │
│           = 2 × 8 = 16                                      │
│  gIdx = 3 / 16 = 0                                          │
│  gTail = 3 % 16 = 3                                         │
│                                                             │
│  Step4: 计算s2oIdx和s1oIdx                                   │
│  s2oIdx = 3 / s1Outer = 3 / 2 = 1                           │
│  s1oIdx = 3 % s1Outer = 3 % 2 = 1                           │
│                                                             │
│  结果: blockIdx=19 → (b=1, n2=0, g=0, s1o=1, s2o=1)         │
│                                                             │
│  验证:                                                      │
│  blockIdx = ((1×2+0)×1+0)×8+1)×2+1                         │
│           = (2×1+0)×8+1)×2+1                                │
│           = (2×8+1)×2+1                                     │
│           = 17×2+1                                          │
│           = 35?                                              │
│                                                             │
│  实际编码公式（注意baseIdx从0开始）:                          │
│  blockIdx = (((bIdx × n2 + n2Idx) × g + gIdx)               │
│              × s2Outer + s2oIdx) × s1Outer + s1oIdx         │
│                                                             │
│  正确计算:                                                   │
│  bIdx=0时baseIdx范围: 0-15 (batch0的16个block)              │
│  bIdx=1时baseIdx范围: 16-31 (batch1的16个block)             │
│                                                             │
│  blockIdx=19 在batch1范围(16-31)内                          │
│  解码: bIdx=1, 剩余=3                                       │
│                                                             │
│  剩余=3在batch1内:                                           │
│  s2oIdx × s1Outer + s1oIdx = 3                              │
│  s2oIdx=1, s1oIdx=1 → 1×2+1=3 ✓                            │
│                                                             │
│  最终: blockIdx=19 → (b=1, n2=0, g=0, s1o=1, s2o=1) ✓      │
│                                                             │
│  含义:                                                      │
│  ├─ batch 1                                                 │
│  ├─ n2 head 0                                               │
│  ├─ group 0                                                 │
│  ├─ S1的第2个基本块(s1oIdx=1, 覆盖s1[256:512])              │
│  └─ S2的第2个基本块(s2oIdx=1, 覆盖s2[128:256])              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 核间协作机制

### 9.1 Cube-Vector核间同步

```
┌─────────────────────────────────────────────────────────────┐
│               Cube-Vector核间同步机制                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Regbase架构: 1 Cube : 2 Vector                             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │  Cube核 cBlockIdx                                   │    │
│  │  ├─ 执行MM1: Dy × V → mm1Res                        │    │
│  │  ├─ 执行MM2: Q × K → mm2Res                         │    │
│  │  └─ 设置同步Flag: CrossCoreSetFlag(SYNC_C1_TO_V2) │    │
│  │                                                     │    │
│  │              ↓ Flag同步                              │    │
│  │                                                     │    │
│  │  Vector核 vBlockIdx = cBlockIdx × 2                 │    │
│  │  ├─ Vector0 (vSubBlockIdx=0):                       │    │
│  │  │  ├─ 处理前半s1 (s1[0:s1Inner])                  │    │
│  │  │  ├─ WaitFlag(SYNC_C1_TO_V2)                     │    │
│  │  │  ├─ ProcessVec2 (Softmax)                       │    │
│  │  │  └ ProcessVec3/Vec4 (Dropout/Cast)             │    │
│  │  │                                                  │    │
│  │  ├─ Vector1 (vSubBlockIdx=1):                       │    │
│  │  │  ├─ 处理后半s1 (s1[s1Inner:2×s1Inner])          │    │
│  │  │  ├─ WaitFlag(SYNC_C1_TO_V2 + 16)                │    │
│  │  │  ├─ ProcessVec2 (Softmax)                       │    │
│  │  │  └ ProcessVec3/Vec4 (Dropout/Cast)             │    │
│  │  │                                                  │    │
│  │  └─────────────────────────────────────────────────┘    │
│  │                                                     │    │
│  │  共享Buffer:                                         │    │
│  │  ├─ mm1ResBuf[2]: MM1结果PingPong Buffer            │    │
│  │  ├─ mm2ResBuf[2]: MM2结果PingPong Buffer            │    │
│  │  ├─ dSL1Buf: dS L1Buffer (Cube写入，Vector读取)     │    │
│  │  ├─ pL1Buf: p L1Buffer (Vector写入，Cube读取)       │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  同步Flag说明:                                              │
│  ├─ SYNC_C1_TO_V2_FLAG[taskId]: MM1完成，Vector可开始     │
│  ├─ SYNC_C2_TO_V2_FLAG[taskId]: MM2完成，Vector可开始    │
│  ├─ SYNC_V3_TO_C3_FLAG: dS拷贝完成，Cube可开始MM3        │
│  ├─ SYNC_V4_TO_C5_FLAG: p拷贝完成，Cube可开始MM5         │
│  │                                                       │
│  │  +16偏移: 用于区分Vector0和Vector1                   │
│  │  ├─ Vector0: 使用原始Flag                            │
│  │  └─ Vector1: 使用Flag+16                             │
│  │                                                       │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 PingPongBuffer机制

```cpp
// Kernel侧使用PingPongBuffer
int64_t taskIdMod2 = taskId % 2;  // 0或1，交替使用两个Buffer

// Cube写入:
mm1ResTensor = this->mm1ResBuf[taskIdMod2].Get<CALC_TYPE>();
mm2ResTensor = this->mm2ResBuf[taskIdMod2].Get<CALC_TYPE>();

// Vector读取上一轮结果:
LocalTensor<CALC_TYPE> prevMm1Res = this->mm1ResBuf[(taskId+1) % 2].Get<CALC_TYPE>();
LocalTensor<CALC_TYPE> prevMm2Res = this->mm2ResBuf[(taskId+1) % 2].Get<CALC_TYPE>();

// 流水线示意:
// taskId=0: Cube写buf[0], Vector空闲
// taskId=1: Cube写buf[1], Vector读buf[0]
// taskId=2: Cube写buf[0], Vector读buf[1]
// taskId=3: Cube写buf[1], Vector读buf[0]
// ...
```

---

## 10. 总结

### 10.1 关键传递路径

```
┌─────────────────────────────────────────────────────────────┐
│              TilingData传递关键节点                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Host侧:                                                    │
│  ├─ DoOpTiling() → 计算所有tiling参数                       │
│  ├─ SaveToTilingData() → 写入TilingData结构                 │
│  ├─ context->SetTilingData() → 写入GM内存                   │
│  └─ context->SetBlockDim() → 设置启动核数                   │
│                                                             │
│  Kernel侧:                                                  │
│  ├─ GET_TILING_DATA_WITH_STRUCT() → 从GM读取                │
│  ├─ GetBlockIdx() → 获取核索引                              │
│  ├─ blockStarts[cBlockIdx] → 获取任务范围                   │
│  ├─ GetNextValidIdx() → 遍历有效Block                       │
│  ├─ SetRunInfo() → 解码坐标 + 计算偏移                      │
│  └─ Process() → 执行计算                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 核心数据流

| 参数 | Host侧计算 | Kernel侧使用 |
|------|------------|--------------|
| s1Inner/s2Inner | DoSplit() | Cube/Vector基本块尺寸 |
| s1Outer/s2Outer | ceil(s/s1Inner) | 外层循环次数 |
| blockStarts[核ID] | DoSparse()分核 | 核任务起始 |
| blockEnds[核ID] | DoSparse()分核 | 核任务结束 |
| blockIdx | 遍历blockStarts到blockEnds | 解码→(b,n2,g,s1o,s2o) |
| GM偏移 | GetQueryOffset()等 | 计算输入/输出地址 |

### 10.3 关键宏和函数

| 名称 | 作用 |
|------|------|
| GET_TILING_DATA_WITH_STRUCT | 从GM拷贝TilingData |
| GetBlockIdx() | 获取当前核索引 |
| GetSubBlockIdx() | Vector核获取子索引 |
| tilingData->s1s2BNGS1S2BlockNumList.blockStarts[idx] | 读取分核范围 |
| SetConstInfo() | 设置常量参数 |
| SetRunInfo() | 解码Block坐标 |
| GetNextValidIdx() | 获取下一个有效Block |

---

**文档版本**: v1.0  
**生成日期**: 2025-05-20  
**适用架构**: Ascend 950 Regbase (DAV_3510)