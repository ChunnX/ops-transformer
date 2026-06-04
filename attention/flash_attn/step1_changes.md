# Step 1: ConstInfo_t 超大结构体拆分 [P7]

## 严重度
🔴 最高优先级

## 涉及文件
- `op_kernel/utils/flash_attn_common_def.h:371-383`

## 现状问题
`ConstInfo_t<NO_QUANT>` 多重继承 8 个子结构体（CommonConstInfo + PAConstInfo + LseConstInfo + SinkConstInfo + PseConstInfo + TensorListConstInfo + PostQuantConstInfo + LeftPaddingConstInfo + SysPrefixConstInfo），估算 >300B，远超 64B cacheline。热路径每次访问成员都跨 cacheline Load，且编译器无法将整个结构体放入寄存器。

## 修改方案

### 1. 创建 HotConstInfo 小结构体（≤48B）
在 `flash_attn_common_def.h` 中添加：

```cpp
struct HotConstInfo {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dBasicBlock;
    uint64_t s1Size;
    uint64_t s2Size;
    float scaleValue;
    uint32_t coreNum;
    uint8_t subBlockIdx;
    uint32_t aicIdx;
};
```

### 2. 创建 ColdConstInfo 结构体
包含冷字段并继承其他冷结构体：

```cpp
struct ColdConstInfo : PAConstInfo,
                       LseConstInfo,
                       SinkConstInfo,
                       PseConstInfo,
                       TensorListConstInfo,
                       PostQuantConstInfo,
                       LeftPaddingConstInfo,
                       SysPrefixConstInfo {
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t dSizeRope;
    uint64_t cuSeqLensQSize;
    uint64_t cuSeqLensKVSize;
    uint64_t seqUsedQSize;
    uint64_t seqUsedKvSize;
    uint32_t bN2Start;
    uint32_t bN2End;
    uint32_t gS1OStart;
    uint32_t gS1OEnd;
    uint32_t s2OStart;
    uint32_t s2OEnd;
    uint32_t coreFirstTmpOutWsPos;
    uint32_t sparseMode;
    uint32_t attenMaskBatch;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    int64_t preTokens;
    int64_t nextTokens;
    bool isExistRowInvalid;
    uint32_t aivIdx;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
    FA_LAYOUT outputLayout;
    bool needInitOutput;
};
```

### 3. 在 FlashAttentionNoQuantGqaKernel 中添加成员
在 `flash_attn_kernel_base_gqa_template.h` 中：

```cpp
ConstInfoX constInfo;
HotConstInfo hotConst;
ColdConstInfo coldConst;
```

### 4. 在 InitConstInfo() 中填充 hotConst 和 coldConst
在 `flash_attn_kernel_base_gqa_template.h` 的 `InitConstInfo()` 函数末尾添加：

```cpp
hotConst.bSize = constInfo.bSize;
hotConst.n2Size = constInfo.n2Size;
hotConst.gSize = constInfo.gSize;
hotConst.dSize = constInfo.dSize;
hotConst.dSizeV = constInfo.dSizeV;
hotConst.dBasicBlock = constInfo.dBasicBlock;
hotConst.s1Size = constInfo.s1Size;
hotConst.s2Size = constInfo.s2Size;
hotConst.scaleValue = constInfo.scaleValue;
hotConst.coreNum = constInfo.coreNum;
hotConst.subBlockIdx = constInfo.subBlockIdx;
hotConst.aicIdx = constInfo.aicIdx;

coldConst.t1Size = constInfo.t1Size;
coldConst.t2Size = constInfo.t2Size;
coldConst.cuSeqLensQSize = constInfo.cuSeqLensQSize;
coldConst.cuSeqLensKVSize = constInfo.cuSeqLensKVSize;
coldConst.seqUsedQSize = constInfo.seqUsedQSize;
coldConst.seqUsedKvSize = constInfo.seqUsedKvSize;
coldConst.sparseMode = constInfo.sparseMode;
coldConst.attenMaskBatch = constInfo.attenMaskBatch;
coldConst.attenMaskS1Size = constInfo.attenMaskS1Size;
coldConst.attenMaskS2Size = constInfo.attenMaskS2Size;
coldConst.preTokens = constInfo.preTokens;
coldConst.nextTokens = constInfo.nextTokens;
coldConst.isExistRowInvalid = constInfo.isExistRowInvalid;
coldConst.aivIdx = constInfo.aivIdx;
coldConst.accumOutSize = constInfo.accumOutSize;
coldConst.logSumExpSize = constInfo.logSumExpSize;
coldConst.outputLayout = constInfo.outputLayout;
coldConst.needInitOutput = constInfo.needInitOutput;
coldConst.isKvContinuous = constInfo.isKvContinuous;
if constexpr (PAGE_ATTENTION) {
    coldConst.maxBlockNumPerBatch = constInfo.maxBlockNumPerBatch;
    coldConst.blockSize = constInfo.blockSize;
    coldConst.paLayoutType = constInfo.paLayoutType;
}
coldConst.isSoftmaxLseEnable = constInfo.isSoftmaxLseEnable;
coldConst.learnableSinkFlag = constInfo.learnableSinkFlag;
```

### 5. 热路径中使用 hotConst 替代 constInfo
在热路径函数中将 `constInfo.xxx` 替换为 `hotConst.xxx`：
- `FlashAttention()` - aicIdx, coreNum
- `GetTaskDealMode()` - n2Size, gSize
- `CalcParams()` - n2Size, gSize, subBlockIdx
- `UpdateAxisInfo()` - gSize
- `GetFASectionInfo()` - aicIdx
- `Process()` - aicIdx, coreNum

## 预期收益
热路径 cacheline miss 降低 60%+，编译器可将 HotConstInfo 关键字段驻留寄存器。

## 实施状态
✅ 已完成（commit 455f56dd7）
