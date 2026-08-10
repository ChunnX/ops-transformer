# MXFP8 全量化场景 FlashDecode 流程梳理

本文档基于 arch35 平台代码，聚焦 MXFP8（microscaling FP8）全量化场景下 **FlashDecode（FD）归约阶段** 的完整流程，涵盖参数介绍、Host 侧 FD 任务划分、Kernel 侧 FD 计算流程与代码走读。

涉及文件：

- [fia_block_vec_flashdecode_fullquant.h](./fia_block_vec_flashdecode_fullquant.h) — FD 归约阶段 Vector 计算
- [fia_block_vec_fullquant_mx.h](./fia_block_vec_fullquant_mx.h) — FA 阶段 Vector（仅涉及 FD 数据产生侧）
- [split_core_v2.cpp](../../common/op_host/split_core_v2.cpp) — Host 侧分核与 FD 任务划分
- [fia_kernel_fullquant_mx.h](./fia_kernel_fullquant_mx.h) — Kernel 主入口，串联 FA 与 FD
- [fia_public_define_arch35.h](./fia_public_define_arch35.h) — 公共数据结构定义

---

## 一、FlashDecode 背景与触发条件

### 1.1 为什么需要 FlashDecode

在 Attention 计算中，当某一行（某个 BN2 + GS1）的 S2 维度被切分到多个核并行计算时，每个核只能得到该行 S2 一段的局部 softmax 结果（P*V 累积输出、局部 max、局部 sum）。FlashDecode 阶段负责把这些跨核的局部结果归约为该行的最终 Attention 输出。

### 1.2 FD 触发条件

FD 在 Host 侧 [split_core_v2.cpp](../../common/op_host/split_core_v2.cpp) 的分核过程中被判定，在 Kernel 侧 [fia_kernel_fullquant_mx.h](./fia_kernel_fullquant_mx.h) 的 `CalcParams` 中通过 `isS2SplitCore` 标记。

**Host 侧判定**：当 `streamK` 为 true 且某行 M 被分配到多个核（`curKvSplitPart > 1`）时，`RecordFDInfo` 记录一条 FD 任务。

**Kernel 侧标记**：`isS2SplitCore` 在以下情况为 true：

- 当前任务是首行（`bN2Cur==bN2Start && gS1Cur==gS1OStart`）且 `headS2Split`（S2 起始不在块边界）。
- 当前任务是末行（`bN2Cur==bN2End && gS1Cur==gS1OEnd`）且 `tailS2Split`（S2 末尾不在块边界）。
- 单行任务（`bN2Start==bN2End && gS1OStart==gS1OEnd`）。

### 1.3 FD 数据来源

FD 阶段消费 FA 阶段写到 workspace 的三类数据：

| GM Buffer          | 含义                                                 | 产生位置                               |
| ------------------ | ---------------------------------------------------- | -------------------------------------- |
| `accumOutGm`     | 各份的 P*V 累积输出（fp32，shape =`[份数, M, D]`） | FA Vec2`Bmm2ResForFDCopyOut`         |
| `softmaxFDSumGm` | 各份的 softmax sum（fp32，shape =`[份数, M, 8]`）  | FA Vec1`ComputeLogSumExpAndCopyToGm` |
| `softmaxFDMaxGm` | 各份的 softmax max（fp32，shape =`[份数, M, 8]`）  | FA Vec1`ComputeLogSumExpAndCopyToGm` |

workspace 布局由 `constInfo.accumOutSize` / `constInfo.logSumExpSize` 描述，三者顺序为 accumOut → lseSum → lseMax。

### 1.3.1 Workspace 地址分配

#### 总体布局

由 Host 侧 [fia_tiling_fullquant_mx.cpp](../../op_host/arch35/fia_tiling_fullquant_mx.cpp) 的 `CalcWorkspaceSize()` 计算，workspace 自低地址起分为两段：

```
workspace起点
    │
    ▼
┌──────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐
│  sysWorkspace    │  accumOutGm             │  softmaxFDSumGm         │  softmaxFDMaxGm         │
│  (系统空间)       │  (P*V累加, fp32)         │  (softmax_sum, fp32)     │  (softmax_max, fp32)    │
└──────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘
                    ▲                         ▲                         ▲
                    │                         │                         │
            workspace+sysSize         +accumOutSize              +logSumExpSize
```

后三块为 FD 专用空间，仅在 `flashDecodeFlag_=true` 时分配。

> **注**：代码中存在 `bmm2Bytes=0; vec2Bytes=0; workspaceSize_ += (bmm2Bytes + vec2Bytes) * 3 * numBlocks_;` 一段历史预留逻辑，因两者均硬编码为 0，实际不占用 workspace。bmm2/vec2 的 double buffering / preload 完全在 UB 上通过 TQue 管理，不需要 GM 空间。

三块 GM 的 Kernel 侧基地址设置（[fia_block_vec_fullquant_mx.h:208-213](./fia_block_vec_fullquant_mx.h)）：

```cpp
accumOutGm     .SetGlobalBuffer((__gm__ float *)workspace);
softmaxFDSumGm .SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
softmaxFDMaxGm .SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize + constInfo.logSumExpSize);
```

#### Size 计算（Host 侧 `CalcWorkspaceSize`）

每核最多写 **2 份**（头切分 + 尾切分场景），故 workspace 预留 `numBlocks_ * 2` 份空间：

| 字段                                      | 计算公式                                               | 说明                                                  |
| ----------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------- |
| `accumOutSize` (`faTmpAttenGmSize`)   | `numBlocks_ * 2 * mSize * dSize`                     | P*V累加结果，每份 M×D 个 fp32                        |
| `logSumExpSize` (`fatmpResLseGmSize`) | `numBlocks_ * 2 * mSize * 8`                         | LSE中间结果，每份 M×8 个 fp32（8=BLOCK_ELEMENT_NUM） |
| FD 总字节数                               | `(accumOutSize + 2 * logSumExpSize) * sizeof(float)` | LSE 有 sum 和 max 两份                                |

其中：

- `mSize = sOuterFactor_ * platformInfo_.cvRatio`
- `dSize = fiaInfo_->vHeadDim`
- `numBlocks_` 由 `CalcNumBlocks(aicNum)` 根据核数与 vec/cube ratio 计算

#### 份数索引 `faTmpOutWsPos`

三块 GM 都按 "**核 × 份数 × M × D**" 逻辑组织，"份数"维度由 `faTmpOutWsPos` 索引。

**Host 侧赋值链**（[split_core_v2.cpp](../../common/op_host/split_core_v2.cpp)）：

```cpp
// split_core_v2.cpp:579-580  在 CalcSplitPlan 主循环中为每个核分配首份位置
result.firstFdDataWorkspaceIdx[curCoreIdx] = preFdDataNUM + curKvSplitPart - 1U;

// split_core_v2.cpp:611  完成一个 FD 任务后累加
preFdDataNUM += curKvSplitPart;
```

- `preFdDataNUM`：之前所有 FD 归约任务累积的份数（前缀和）
- `curKvSplitPart`：当前 FD 任务的 KV 切分数
- `firstFdDataWorkspaceIdx[curCoreIdx]` 记录到 `fiaMetaDataGm`，Kernel 侧通过 `FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX` 读取

**Kernel 侧赋值**（[fia_kernel_fullquant_mx.h:641-668](./fia_kernel_fullquant_mx.h)）：

```cpp
info.faTmpOutWsPos = constInfo.coreFirstTmpOutWsPos;  // 从 GM metadata 读取
// 尾切分核(headS2Split && tailS2Split): 份位置 +1
info.faTmpOutWsPos = headS2Split ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
```

| 场景                           | `faTmpOutWsPos` 值         | 含义        |
| ------------------------------ | ---------------------------- | ----------- |
| 普通 FD 核                     | `coreFirstTmpOutWsPos`     | 写到第 0 份 |
| 尾切分核（head+tail 同时存在） | `coreFirstTmpOutWsPos + 1` | 写到第 1 份 |

#### 写入偏移（FA 阶段）

**accumOutGm 写入**（Vec2 阶段 [fia_block_vec_fullquant_mx.h:983](./fia_block_vec_fullquant_mx.h)）：

```cpp
gmOffset = faTmpOutWsPos * mBaseSize * dSizeV + (vecMbaseIdx + mStartVec) * dSizeV
           └── 份数偏移(以M*D为单位) ──┘   └── 核内M行偏移 ──────────────┘
```

**softmaxFDSumGm/MaxGm 写入**（Vec1 阶段 [fia_block_vec_fullquant_mx.h:967](./fia_block_vec_fullquant_mx.h)）：

```cpp
gmOffset = faTmpOutWsPos * mBaseSize * fp32BaseSize + vecMbaseIdx * fp32BaseSize
           └── 份数偏移(以M*8为单位) ────────┘   └── 核内M行偏移 ──┘
// fp32BaseSize = 8 (BLOCK_ELEMENT_NUM)
```

#### 读取偏移（FD 规约阶段）

FD 规约核按 `splitKVIndex` 顺序遍历所有份（[fia_block_vec_flashdecode_fullquant.h:231](./fia_block_vec_flashdecode_fullquant.h)）：

**accumOutGm 读取**：

```cpp
combineAccumOutOffset = startRow * dSizeV +                       // M轴偏移
                        splitKVIndex * s1BaseSize * dSizeV;       // 份数偏移
```

**lse GM 读取**（`CopyLseIn`）：

```cpp
combineLseOffset  = (baseOffset + startRow) * BLOCK_ELEMENT_NUM;  // 起始M偏移(8字节对齐)
combineLoopOffset = s1BaseSize * BLOCK_ELEMENT_NUM;               // 每份步长
for (i = 0..actualCombineLoopSize-1):
    DataCopy(lseSum[...], lseSumFdGm[combineLseOffset + i * combineLoopOffset], ...)
    DataCopy(lseMax[...], lseMaxFdGm[combineLseOffset + i * combineLoopOffset], ...)
```

其中 `actualCombineLoopSize = fd.fdS2SplitNum`（参与规约的份数），`baseOffset` 来自 FD 任务的起始 M 偏移，`taskOffset = fd.fdWorkspaceIdx * s1BaseSize` 提供 workspace 份数前缀和偏移。

#### 数据流示意

```
┌──────────────── FA 阶段（各 split core 并行写）────────────────┐
│                                                                │
│  Vec1: ComputeLogSumExpAndCopyToGm                             │
│    softmaxFDSumGm[faTmpOutWsPos*M*8 + vecMbaseIdx*8]  ← sum   │
│    softmaxFDMaxGm[faTmpOutWsPos*M*8 + vecMbaseIdx*8]  ← max   │
│                                                                │
│  Vec2: Bmm2ResForFDCopyOut                                     │
│    accumOutGm[faTmpOutWsPos*M*D + (vecMbaseIdx+mStartVec)*D]  │
│                                          ← vec2ResUb (fp32)    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────── FD 阶段（规约核串行读所有份）──────────────────┐
│                                                                │
│  for splitKVIndex in [0, actualCombineLoopSize):               │
│    accumOutGm[startRow*D + splitKVIndex*M*D]    → mm2Res       │
│    lseSumFdGm[(baseOffset+startRow)*8 + splitKVIndex*M*8]      │
│                                                  → lseSum      │
│    lseMaxFdGm[(baseOffset+startRow)*8 + splitKVIndex*M*8]      │
│                                                  → lseMax      │
│                                                                │
│  ComputeScaleValue_8_VF_FD: 计算 w_i / global_sum              │
│  ReduceFinalRes: dst += scale_i * accumOut_i                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.4 FD 触发后 FA 侧的行为

当 `isS2SplitCore=true` 时，FA 阶段对该行不写最终输出，而是：

- Vec1 末块（`isLastS2Loop`）调用 `ComputeLogSumExpAndCopyToGm`，将 `sumUb/maxUb` broadcast 后写到 `softmaxFDSumGm/softmaxFDMaxGm`。
- Vec2 末块调用 `Bmm2ResForFDCopyOut`，将 `vec2ResUb`（fp32）按 `faTmpOutWsPos` 偏移写到 `accumOutGm`。

非切分核（`isS2SplitCore=false`）即使开 FD，也直接走 `Bmm2ResCastAndCopyOut` 写最终 `attentionOutGm`，不进入 FD 归约。

---

## 二、关键数据结构

### 2.1 ConstInfoX（运行期常量）

定义于 [fia_public_define_arch35.h](./fia_public_define_arch35.h)，FD 阶段使用的字段：

| 字段                                                    | 含义                                 |
| ------------------------------------------------------- | ------------------------------------ |
| `bSize / n2Size / gSize / s1Size / s2Size`            | 各轴大小                             |
| `realN2Size / realGSize`                              | 合轴后实际 N2、G                     |
| `dSizeV`                                              | V 输出 D 维大小                      |
| `bN2Start/bN2End, gS1OStart/gS1OEnd, s2OStart/s2OEnd` | 当前核 FA 区间（FD 阶段不用）        |
| `accumOutSize / logSumExpSize`                        | workspace 中 accumOut、lse 段大小    |
| `sparseMode / preTokens / nextTokens`                 | mask 稀疏模式与窗口（FD 用于无效行） |
| `scaleValue`                                          | softmax 缩放系数                     |
| `aivIdx / coreNum`                                    | 核索引与总核数                       |
| `isSoftmaxLseEnable`                                  | 是否输出 softmaxLse                  |
| `isRowInvalidOpen / isExistRowInvalid`                | 行无效开关                           |
| `outputLayout`                                        | 输出布局（BSH/BNSD/TND/NTD）         |

### 2.2 FDparamsX（FD 阶段单核任务参数）

定义于 [fia_public_define_arch35.h:47](./fia_public_define_arch35.h)，从 `fiaMetaDataGm` 的 FD 段读取，由 [fia_kernel_fullquant_mx.h](./fia_kernel_fullquant_mx.h) 的 `FlashDecode()` 解析：

```cpp
struct FDparamsX {
    uint32_t fdCoreEnable;    // 当前 AIV 是否参与 FD
    uint32_t fdBN2Idx;        // 归约任务对应的 BN2 索引
    uint32_t fdMIdx;          // 归约任务对应的 GS1 块索引
    uint32_t fdS2SplitNum;    // 当前归约任务在 S2 方向被切成几份
    uint32_t mStart;          // 当前核处理的 M 轴起始行
    uint32_t mLen;            // 当前核处理的 M 轴行数
    uint32_t fdWorkspaceIdx;  // 归约数据在 workspace 的前缀和偏移
};
```

| 字段               | 含义                                                          |
| ------------------ | ------------------------------------------------------------- |
| `fdCoreEnable`   | 当前 AIV 是否参与 FD，0 则直接 return                         |
| `fdBN2Idx`       | 归约任务对应的 BN2 索引（用于反推 bIdx/n2Idx）                |
| `fdMIdx`         | 归约任务对应的 GS1 块索引（`gS1Idx = fdMIdx * s1BaseSize`） |
| `fdS2SplitNum`   | 该行 S2 被切了几份，即`actualCombineLoopSize`               |
| `mStart / mLen`  | 当前核在 M 轴上处理的行范围（FD 二次切分）                    |
| `fdWorkspaceIdx` | 该归约任务数据在 workspace 的份数前缀和偏移                   |

### 2.3 TaskInfo（FD 阶段内部任务状态）

定义于 [fia_block_vec_flashdecode_fullquant.h](./fia_block_vec_flashdecode_fullquant.h)：

```cpp
struct TaskInfo {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;   // = fd.fdS2SplitNum
};
```

由 `fdBN2Idx/fdMIdx/fdS2SplitNum` 填充，贯穿整个 FD 计算流程。

### 2.4 Host 侧 FD 相关结构

定义于 [split_core_v2.h](../../common/op_host/split_core_v2.h)：

```cpp
struct FlashDecodeResult {
    uint32_t fdNum;                          // 归约任务数量
    std::vector<uint32_t> fdBN2Idx;          // 每个归约任务的 BN2 索引
    std::vector<uint32_t> fdMIdx;            // 每个归约任务的 GS1 索引
    std::vector<uint32_t> fdS2SplitNum;      // 每个归约任务的 S2 切分份数
    std::vector<uint32_t> fdWorkspaceIdx;    // 每个归约任务在 workspace 的存放位置
    uint32_t fdUsedVecNum;                   // FD 阶段使用的 vector 数量
    std::vector<uint32_t> mSize;             // 每个归约任务的 M 轴大小
    std::vector<uint32_t> taskIdx;           // 每个 vector 处理的归约任务 ID
    std::vector<uint32_t> mStart;            // 每个 vector 处理的 M 轴起点
    std::vector<uint32_t> mLen;              // 每个 vector 处理的 M 轴行数
};
```

---

## 三、Host 侧 FD 任务划分

入口：[SplitCore](../../common/op_host/split_core_v2.cpp)，FD 相关流程在 FA 分核完成后执行。

### 3.1 FD 任务记录（RecordFDInfo）

在 `CalcSplitPlan` 的分配循环中，当某行 M 被切到多个核（`curKvSplitPart > 1`）且游标已越过切分点时，调用 `RecordFDInfo` 记录一条 FD 任务：

```cpp
void RecordFDInfo(const SplitContext &splitContext,
                  const AssignContext &assignContext,
                  FAMetaData &result) {
    // 归约数据对应上一核的切分点
    uint32_t splitBIdx = result.bN2End[curCoreIdx - 1] / n2Size;
    uint32_t splitMIdx = result.mEnd[curCoreIdx - 1];
    uint32_t s1Size = GetS1SeqSize(splitBIdx, baseInfo);

    // 计算归约数据的 M 轴大小（尾块可能小于 mBaseSize）
    uint32_t curFdS1gSize = (splitMIdx == mBaseNum[splitBIdx] - 1) ?
        (s1Size * gSize - splitMIdx * mBaseSize) : mBaseSize;

    // FD 阶段 M 轴二次切分（基本块 gS1BaseSizeOfFd，默认 8）
    uint32_t curFdS1gSplitPart = ceil(curFdS1gSize, gS1BaseSizeOfFd);

    // 记录到 fdRes
    result.fdRes.fdBN2Idx[fdNum]     = splitBIdx * n2Size + ...;
    result.fdRes.fdMIdx[fdNum]       = splitMIdx;
    result.fdRes.fdS2SplitNum[fdNum] = curKvSplitPart;
    result.fdRes.fdWorkspaceIdx[fdNum] = preFdDataNUM;
    result.fdRes.mSize[fdNum]        = curFdS1gSize;
    result.fdRes.fdNum++;
}
```

`IsNeedRecordFDInfo` 判断条件：

- 不是核 0
- 存在跨核行（`curKvSplitPart > 1`）
- 该行已处理完（游标已到下一行或下一核）

`preFdDataNUM` 累加 `curKvSplitPart`，作为下一个 FD 任务在 workspace 的份数偏移。

### 3.2 FD 选择策略（CheckChooseWithFd）

开 FD 会引入 workspace 读写与归约开销，并非总是划算。`CheckChooseWithFd` 比较开/不开 FD 的代价：

```cpp
bool CheckChooseWithFd(int64_t noFdCost, int64_t withFdCost,
                       uint32_t actualMFdsize, const SplitParam &param) {
    const int64_t fullBlockCost = CalcCost(actualMFdsize, param.s2BaseSize);
    // 不开 FD 代价很小则不开
    if (noFdCost <= param.fdLeastBlock * fullBlockCost) {
        return false;
    }
    // 开 FD 收益超过容差才开
    int64_t fdTolerance = param.fdTolerance * fullBlockCost;
    return noFdCost - fdTolerance > withFdCost;
}
```

其中代价函数 `CalcCost`：`cost = 6 * ceil(M, 16) + 10 * ceil(S2, 64)`。

调用点（`SplitCore` 末尾）：

```cpp
if (param.streamK && result.fdRes.fdNum > 0) {
    splitContext.splitParam.streamK = false;          // 关闭 streamK 重新算一次
    CalcSplitPlan(result.usedCoreNum, INT64_MAX, splitContext, tmpResult);
    if (!CheckChooseWithFd(tmpResult.maxCost, result.maxCost, ...)) {
        CopyTmpResult(tmpResult, result);             // 不开 FD，用无 FD 方案
    }
}
```

### 3.3 FD 负载均衡（SplitFD）

若保留 FD 方案，调用 `SplitFD` 将 FD 任务分配给 vec 核。vec 核总数 `maxVectorNum = usedCoreNum * vecCubeRatio`。

```cpp
void SplitFD(FAMetaData &result) {
    // 1. 计算总 FD 负载
    uint64_t totalFDLoad = 0;
    for (i in [0, fdNum)) {
        totalFDLoad += fdS2SplitNum[i] * mSize[i];
    }

    // 2. 每核平均负载（向上取整避免空核）
    uint64_t averageLoad = ceil(totalFDLoad, maxVectorNum);
    uint32_t emptyVectorNum = maxVectorNum - fdNum;   // FD 任务自身各占 1 核

    // 3. 逐任务分配 vec 核
    for (i in [0, fdNum)) {
        uint32_t curFDVectorNum = max(fdS2SplitNum[i] * mSize[i] / averageLoad, 1);
        uint32_t curAvgMSize = ceil(mSize[i], curFDVectorNum);
        curFDVectorNum = min(curFDVectorNum, emptyVectorNum + 1);

        for (vid in [0, curFDVectorNum)) {
            fdRes.taskIdx[curCoreIndex] = i;
            fdRes.mStart[curCoreIndex]  = vid * curAvgMSize;
            fdRes.mLen[curCoreIndex]    = (vid == curFDVectorNum-1) ?
                                          (mSize[i] - vid * curAvgMSize) : curAvgMSize;
            curCoreIndex++;
        }
        emptyVectorNum -= (curFDVectorNum - 1);
    }
    fdRes.fdUsedVecNum = curCoreIndex;
}
```

分配结果写入 `fiaMetaDataGm` 的 FD 段，Kernel 侧通过 `GetFDMetaDataIndex` 读取。

---

## 四、Kernel 侧 FD 计算流程

### 4.1 FlashDecode 入口

定义于 [fia_kernel_fullquant_mx.h](./fia_kernel_fullquant_mx.h)：

```cpp
__aicore__ inline void FlashDecode()
{
    vecFdBlock.InitBuffers(this->pipe);
    AscendC::ICachePreLoad(2);

    // 从 fiaMetaDataGm FD 段读取本核任务
    uint32_t fdCoreEnable = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(aivIdx, FD_CORE_ENABLE_INDEX));
    uint32_t fdBN2Idx     = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(aivIdx, FD_BN2_IDX_INDEX));
    uint32_t fdMIdx       = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(aivIdx, FD_M_IDX_INDEX));
    uint32_t fdS2SplitNum = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(aivIdx, FD_S2_SPLIT_NUM_INDEX));
    uint32_t mStart       = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(aivIdx, FD_M_START_INDEX));
    uint32_t mLen         = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(aivIdx, FD_M_NUM_INDEX));
    uint32_t fdWorkspaceIdx = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(aivIdx, FD_WORKSPACE_IDX_INDEX));

    FDparamsX fdParams = {fdCoreEnable, fdBN2Idx, fdMIdx, fdS2SplitNum, mStart, mLen, fdWorkspaceIdx};
    vecFdBlock.AllocEventID();
    SyncAll();                          // 等 FA 阶段全部完成
    vecFdBlock.FlashDecode(fdParams);
    vecFdBlock.FreeEventID();
}
```

`GetFDMetaDataIndex` 偏移：`FA_METADATA_SIZE * NPU_AIC_CORE_NUM + FD_METADATA_SIZE * coreIdx + metaIdx`，即 FD 元数据紧跟在 FA 元数据之后。

### 4.2 FlashDecode 主循环

定义于 [fia_block_vec_flashdecode_fullquant.h:468](./fia_block_vec_flashdecode_fullquant.h)：

```cpp
__aicore__ inline void FlashDecode(FDparamsX &fd)
{
    if (!fd.fdCoreEnable) return;

    // FD 阶段 M 轴二次切分，每块 8 行（fdBalanceMBaseSize）
    uint32_t fdBalanceMBaseSize = 8U;
    uint32_t fdBalanceMSplitNum = ceil(fd.mLen, 8);
    uint32_t fdBalanceMTailSize = (fd.mLen % 8 == 0) ? 8 : fd.mLen % 8;

    // 填充 TaskInfo
    taskInfo.bIdx = fd.fdBN2Idx / constInfo.realN2Size;
    taskInfo.n2Idx = fd.fdBN2Idx % constInfo.realN2Size;
    taskInfo.gS1Idx = fd.fdMIdx * s1BaseSize;
    taskInfo.actualCombineLoopSize = fd.fdS2SplitNum;       // kv 方向份数
    uint64_t taskOffset = fd.fdWorkspaceIdx * s1BaseSize;   // workspace 份数偏移

    for (uint32_t fdS1gOuterMIdx = 0; fdS1gOuterMIdx < fdBalanceMSplitNum; ++fdS1gOuterMIdx) {
        uint32_t actualGSplitSize = (尾块) ? fdBalanceMTailSize : 8;
        uint32_t startRow = fd.mStart + fdS1gOuterMIdx * 8;

        // ① 搬入 lseSum/lseMax + 预取 accumOut
        LoadLseAndPreloadData(startRow, actualGSplitSize, taskOffset, reduceMLoop, reduceGlobaLoop);

        // ② 计算各份的 exp 权重
        ComputeScaleValue(lseExp, actualGSplitSize, taskInfo.actualCombineLoopSize, reduceMLoop, startRow);

        // ③ 计算 mask 窗口（用于无效行判断）
        CalcPreNextTokens();

        // ④ 输出最终 lse
        CopySoftmaxLseOutput(actualGSplitSize, startRow);

        // ⑤ 加权累加各份 accumOut
        for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; ++i) {
            if (i >= preLoadNum) {                    // 滚动预取后续份
                CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
            }
            WaitFlag<MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
            ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
            SetFlag<V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
            reduceGlobaLoop++;
        }

        // ⑥ Cast + 搬出最终结果
        CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
        reduceMLoop++;
    }
}
```

外层循环按 8 行一块遍历本核负责的 M 轴范围，每块依次执行 6 个步骤。

---

## 五、FD 关键子函数

### 5.1 LoadLseAndPreloadData

```cpp
__aicore__ inline void LoadLseAndPreloadData(uint32_t startRow, uint32_t actualGSplitSize,
                                             uint64_t taskOffset, uint32_t reduceMLoop,
                                             uint32_t reduceGlobaLoop)
{
    // 1. 搬入 lseSum/lseMax（双 buffer，按 reduceMLoop&1 选择）
    WaitFlag<V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
    CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
    SetFlag<MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
    WaitFlag<MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));

    // 2. 预取 preLoadNum=2 份 accumOut（双 buffer，按 reduceGlobaLoop&1 选择）
    for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; ++preLoadIdx) {
        LocalTensor<T> mm2Res = ((reduceGlobaLoop + preLoadIdx) & 1) == 0 ?
                                fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
        WaitFlag<V_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
        CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
        SetFlag<MTE2_V>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
    }
}
```

#### CopyLseIn

```cpp
__aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount,
                                  uint64_t baseOffset, uint32_t cntM)
{
    LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    uint64_t combineLseOffset = (baseOffset + startRow) * BLOCK_ELEMENT_NUM;   // 8 元素/行（32B 对齐）
    uint64_t combineLoopOffset = s1BaseSize * BLOCK_ELEMENT_NUM;
    uint64_t dealRowCountAlign = dealRowCount * BLOCK_ELEMENT_NUM;

    for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; ++i) {
        DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign);
        DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign);
    }
}
```

每份 lseSum/lseMax 按 `actualCombineLoopSize` 份连续搬入，每行 8 元素（32B 对齐），份数间 stride = `s1BaseSize * 8`。

#### CopyAccumOutIn

```cpp
__aicore__ inline void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex,
                                      uint32_t startRow, uint32_t dealRowCount)
{
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (dSizeV_Align - constInfo.dSizeV) / BLOCK_ELEMENT_NUM;

    DataCopyPadExtParams<T> copyInPadParams;
    copyInPadParams.isPad = true;
    copyInPadParams.rightPadding = (dSizeV_Align - constInfo.dSizeV) % BLOCK_ELEMENT_NUM;

    uint64_t combineAccumOutOffset = startRow * constInfo.dSizeV +
                                     splitKVIndex * s1BaseSize * constInfo.dSizeV;
    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
}
```

accumOut 每份按 `splitKVIndex * s1BaseSize * dSizeV` 偏移，`dSizeV` 非 32 倍数时用 `DataCopyPad` 右侧补零到 `dSizeV_Align`。

### 5.2 ComputeScaleValue

```cpp
__aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t dealRowCount,
                                         uint32_t actualCombineLoopSize, uint32_t cntM,
                                         uint32_t startRow)
{
    LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();
    LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1.Get<T>() : fdLseMaxUbBuf2.Get<T>();

    LocalTensor<T> sinkExpBuf = fdSinkExpBuf.Get<T>();
    LocalTensor<T> maxLseUb = fdLseUbBuf.Get<T>();
    ComputeScaleValue_VF_FD(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb,
                            dealRowCount, actualCombineLoopSize,
                            constInfo.isSoftmaxLseEnable, learnableSinkFlag);
}
```

调用 `ComputeScaleValue_VF_FD`（定义于 [vf_flash_decode.h](../../common/op_kernel/arch35/vf/vf_flash_decode.h)），对 `actualCombineLoopSize` 份 lseMax/lseSum 计算：

**数学公式**：设每份 max 为 `m_i`、sum 为 `s_i`：

- `M = max(m_i)`（全局最大）
- `w_i = exp(m_i - M) * s_i`（每份的未归一权重）
- `W = Σ(w_i)`（全局 sum）
- `lseExp_i = w_i / W`（每份的归一化权重，供 ReduceFinalRes 使用）
- `maxLseUb = log(W) + M`（最终 lse，供 CopySoftmaxLseOutput 输出）

`maxLseUb` 同时用于后续 `DealInvalidMaskRows` 判断行是否无效（值为 -inf 表示无效行）。

### 5.3 CalcPreNextTokens

```cpp
__aicore__ inline void CalcPreNextTokens()
{
    actSeqLensQ = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
    actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);

    int64_t safePreToken = constInfo.preTokens;
    int64_t safeNextToken = constInfo.nextTokens;
    fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, safePreToken, safeNextToken,
                                    constInfo.sparseMode);

    if (sparseMode == BAND) {
        preTokensPerBatch = safePreToken;
        nextTokensPerBatch = actSeqLensKv - actSeqLensQ + safeNextToken;
    } else if (sparseMode == DEFAULT_MASK && attenMaskFlag) {
        nextTokensPerBatch = safeNextToken;
        preTokensPerBatch = actSeqLensKv - actSeqLensQ + safePreToken;
    } else {
        nextTokensPerBatch = actSeqLensKv - actSeqLensQ;
        preTokensPerBatch = 0;
    }
}
```

为 `DealInvalidRows`/`DealInvalidMaskRows` 准备 mask 窗口上下界。

### 5.4 CopySoftmaxLseOutput

```cpp
__aicore__ inline void CopySoftmaxLseOutput(uint32_t actualGSplitSize, uint32_t startRow)
{
    if (constInfo.isSoftmaxLseEnable) {
        LocalTensor<T> maxLseUb = fdLseUbBuf.Get<T>();
        SetFlag<V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
        WaitFlag<V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);

        uint32_t mOffset = taskInfo.gS1Idx + startRow;
        // 按 outputLayout 分派不同的 DataCopySoftmaxLse*Arch35
        if constexpr (layout == LAYOUT_TND) {
            uint32_t prefixBS1 = qActSeqLensParser.GetTBase(taskInfo.bIdx);
            uint64_t bN2Offset = prefixBS1 * realGSize * realN2Size + n2Idx * realGSize;
            DataCopySoftmaxLseTNDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset,
                                                       mOffset, actualGSplitSize, constInfo);
        } else if constexpr (layout == LAYOUT_NTD) { ... }
        else if constexpr (layout == LAYOUT_BSH)  { ... }
        else { /* BNSD */ }
    }
}
```

将最终 lse（`maxLseUb`）按输出布局写到 `softmaxLseGm`。TND/NTD 用 `qActSeqLensParser.GetTBase(bIdx)` 计算累积偏移；BSH/BNSD 用 `bIdx * n2 * g * s1` 偏移。

### 5.5 ReduceFinalRes

```cpp
__aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res,
                                      LocalTensor<T> &lseLocal, uint32_t cntKV,
                                      uint32_t dealRowCount)
{
    uint64_t dSizeV_Align = (uint64_t)this->dSizeV_Align;
    ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align, cntKV);
}
```

调用 `ReduceFinalRes_VF<T>`（定义于 [vf_flash_decode.h](../../common/op_kernel/arch35/vf/vf_flash_decode.h)），对第 `cntKV` 份 accumOut 做：

```
reduceOut += lseExp[cntKV] * mm2Res
```

循环 `actualCombineLoopSize` 次后，`reduceOut` 即为加权累加结果 `O = Σ(lseExp_i * o_i)`。

### 5.6 CopyFinalResOut

```cpp
__aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow,
                                       uint32_t dealRowCount, uint32_t cntM)
{
    LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUTPUT_T>();
    AscendC::PipeBarrier<PIPE_V>();

    // ① 处理无效行
    DealInvalidRows(accumOutLocal, startRow, dealRowCount, dSizeV_Align);
    DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, dSizeV_Align, cntM);

    // ② Cast fp32 → OUTPUT_T
    WaitFlag<MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT,
             dealRowCount * dSizeV_Align);        // bf16 四舍六入五成双
    } else {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
             dealRowCount * dSizeV_Align);
    }
    SetFlag<V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);

    // ③ 按输出布局搬到 attentionOutGm
    Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, dSizeV_Align);
    SetFlag<MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}
```

#### DealInvalidRows

```cpp
template <typename UBOUT_T>
__aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow,
                                       uint32_t dealRowCount, uint32_t columnCount)
{
    if (!attenMaskFlag) return;
    if (sparseMode == ALL_MASK || sparseMode == LEFT_UP_CAUSAL) return;

    fa_base_vector::InvalidRowParams params{
        .actS1Size = actSeqLensQ,
        .gSize = realGSize,
        .gS1Idx = taskInfo.gS1Idx + startRow,
        .dealRowCount = dealRowCount,
        .columnCount = columnCount,
        .preTokensPerBatch = preTokensPerBatch,
        .nextTokensPerBatch = nextTokensPerBatch,
    };
    fa_base_vector::InvalidRows<UBOUT_T, GeInputUbFormat<layout>()> invalidRows;
    invalidRows(attenOutUb, params);
}
```

按 mask 窗口将超出 `preTokensPerBatch/nextTokensPerBatch` 的行清零。

#### DealInvalidMaskRows

```cpp
template <typename UBOUT_T>
__aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
{
    if (!constInfo.isRowInvalidOpen || !attenMaskFlag) return;
    if (sparseMode != DEFAULT_MASK && sparseMode != ALL_MASK) return;

    LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1.Get<T>() : fdLseMaxUbBuf2.Get<T>();
    // lseMaxUb 中 max 为 -inf 的行对应 attenOutUb 的无效行，清零
    ...
}
```

根据 `lseMaxUb` 是否为 -inf 判断行是否完全无效（无任何 S2 块参与），将对应行清零。

#### Bmm2DataCopyOutTrans

```cpp
__aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
                                            uint32_t dealRowCount, uint32_t columnCount)
{
    FaUbTensor<OUTPUT_T> ubTensor{
        .tensor = attenOutUb,
        .rowCount = dealRowCount,
        .colCount = columnCount,
    };
    GmCoord gmCoord{.bIdx = taskInfo.bIdx,
                    .n2Idx = taskInfo.n2Idx,
                    .gS1Idx = taskInfo.gS1Idx + startRow,
                    .dIdx = 0,
                    .gS1DealSize = dealRowCount,
                    .dDealSize = (uint32_t)constInfo.dSizeV};

    // 按 outputLayout 分派
    if (outputLayout == BSH) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
        FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(bSize, realN2Size, realGSize, s1Size, dSizeV,
                                          actualSeqLengthsGmQ, actualSeqLenSize);
        CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (outputLayout == BNSD) { ... GmFormat::BNGSD ... }
    else if (outputLayout == TND)    { ... GmFormat::TNGD ... }
    else if (outputLayout == NTD)    { ... GmFormat::NGTD ... }
}
```

通过 `FaGmTensor` + `CopyAttenOutUbToGm` 适配 4 种输出布局，将 UB 中的结果按 `gmCoord` 搬到 `attentionOutGm`。

---

## 六、Buffer 与事件同步

### 6.1 Buffer 分配

`InitBuffers`（AIV）分配的 UB：

| Buffer                              | 大小  | 用途                                                     |
| ----------------------------------- | ----- | -------------------------------------------------------- |
| `fdSumBuf1 / fdSumBuf2`           | 4K+2K | lseSum 双 buffer（按`reduceMLoop&1` 选择）             |
| `fdMaxBuf1 / fdMaxBuf2`           | 4K+2K | lseMax 双 buffer                                         |
| `fdLseExpBuf`                     | 4K+2K | 各份 exp 归一化权重`lseExp`                            |
| `fdMm2ResBuf1 / fdMm2ResBuf2`     | 16K   | accumOut 双 buffer（按`reduceGlobaLoop&1` 选择）       |
| `fdReduceBuf`                     | 16K   | 归约累加结果`reduceOut`                                |
| `fdOutputBuf`                     | 16K   | Cast 输出（OUTPUT_T）                                    |
| `fdLseMaxUbBuf1 / fdLseMaxUbBuf2` | 256B  | 单行 lseMax（DealInvalidMaskRows 用）                    |
| `fdLseUbBuf`                      | 256B  | 最终 lse 输出                                            |
| `fdSinkExpBuf`                    | 256B  | learnableSink（本场景`learnableSinkFlag=false`，预留） |

### 6.2 事件 ID

`AllocEventID` 预设 8 个事件：

| 事件                               | 类型   | 用途                       |
| ---------------------------------- | ------ | -------------------------- |
| `SYNC_LSE_MAX_SUM_BUF1_FLAG` (0) | V_MTE2 | lseSum/lseMax buffer1 同步 |
| `SYNC_LSE_MAX_SUM_BUF2_FLAG` (1) | V_MTE2 | lseSum/lseMax buffer2 同步 |
| `SYNC_MM2RES_BUF1_FLAG` (2)      | V_MTE2 | accumOut buffer1 同步      |
| `SYNC_MM2RES_BUF2_FLAG` (3)      | V_MTE2 | accumOut buffer2 同步      |
| `SYNC_FDOUTPUT_BUF_FLAG` (4)     | MTE3_V | 输出 buffer 同步           |
| `SYNC_LSEOUTPUT_BUF_FLAG` (5)    | MTE3_V | lse 输出同步               |
| `SYNC_SINK_BUF1_FLAG` (6)        | V_MTE2 | sink buffer1（预留）       |
| `SYNC_SINK_BUF2_FLAG` (7)        | V_MTE2 | sink buffer2（预留）       |

`FreeEventID` 在 FD 结束时 Wait 所有事件并释放。

### 6.3 双 buffer 流水

FD 阶段使用两组双 buffer 实现 M 轴块间流水：

- **lseSum/lseMax 双 buffer**（`fdSumBuf1/2`, `fdMaxBuf1/2`）：按 `reduceMLoop&1` 切换，当前块计算时下一块 lse 可搬入。
- **accumOut 双 buffer**（`fdMm2ResBuf1/2`）：按 `reduceGlobaLoop&1` 切换，配合 `preLoadNum=2` 实现 S2 份间滚动预取。

同步模式：MTE2 搬入后 `SetFlag<MTE2_V>`，V 计算前 `WaitFlag<MTE2_V>`；V 计算后 `SetFlag<V_MTE2>`，下一次搬入前 `WaitFlag<V_MTE2>`。

---

## 七、代码走读：FD 完整调用链

```
[Kernel AIV] FlashAttentionFullQuantMxKernel::Process
  └─ if (FLASH_DECODE && AIV) FlashDecode()
      ├─ vecFdBlock.InitBuffers(pipe)        (分配 UB)
      ├─ ICachePreLoad(2)
      ├─ 读 fiaMetaDataGm FD 段 → FDparamsX
      ├─ vecFdBlock.AllocEventID()           (预设 8 个事件)
      ├─ SyncAll()                           (等 FA 全部完成)
      │
      └─ vecFdBlock.FlashDecode(fd)
          ├─ if (!fd.fdCoreEnable) return
          ├─ 计算 M 轴二次切分 (fdBalanceMSplitNum, 每 8 行一块)
          ├─ 填充 taskInfo (bIdx, n2Idx, gS1Idx, actualCombineLoopSize)
          │
          └─ for fdS1gOuterMIdx in [0, fdBalanceMSplitNum-1]:
              ├─ actualGSplitSize = (尾块) ? fdBalanceMTailSize : 8
              ├─ startRow = fd.mStart + fdS1gOuterMIdx * 8
              │
              ├─ ① LoadLseAndPreloadData
              │   ├─ CopyLseIn
              │   │   (lseSumFdGm/lseMaxFdGm → fdSumBuf1/2, fdMaxBuf1/2)
              │   │   (按 actualCombineLoopSize 份循环搬入)
              │   └─ CopyAccumOutIn × preLoadNum=2
              │       (accumOutGm → fdMm2ResBuf1/2, DataCopyPad 补齐 dSizeV)
              │
              ├─ ② ComputeScaleValue
              │   └─ ComputeScaleValue_VF_FD
              │       (globalMax = max(m_i)
              │        w_i = exp(m_i - globalMax) * s_i
              │        globalSum = Σ(w_i)
              │        lseExp_i = w_i / globalSum
              │        maxLseUb = log(globalSum) + globalMax)
              │
              ├─ ③ CalcPreNextTokens
              │   (根据 sparseMode 计算 preTokensPerBatch/nextTokensPerBatch)
              │
              ├─ ④ CopySoftmaxLseOutput
              │   (maxLseUb → softmaxLseGm, 按 layout 分派)
              │
              ├─ ⑤ for i in [0, actualCombineLoopSize):
              │   ├─ if (i >= preLoadNum) CopyAccumOutIn  (滚动预取)
              │   ├─ WaitFlag<MTE2_V>(MM2RES_BUF)
              │   ├─ ReduceFinalRes
              │   │   └─ ReduceFinalRes_VF
              │   │       (reduceOut += lseExp[i] * mm2Res)
              │   └─ SetFlag<V_MTE2>(MM2RES_BUF)
              │
              └─ ⑥ CopyFinalResOut
                  ├─ DealInvalidRows        (按 mask 窗口清零)
                  ├─ DealInvalidMaskRows    (lseMaxUb=-inf 的行清零)
                  ├─ Cast (fp32 → OUTPUT_T)
                  │   (bf16: CAST_RINT, 其他: CAST_ROUND)
                  └─ Bmm2DataCopyOutTrans
                      └─ CopyAttenOutUbToGm (按 outputLayout: BSH/BNSD/TND/NTD)
                          (fdOutputBuf → attentionOutGm)
```

---

## 八、FD 归约数学总结

设某行 S2 被切成 `K = fdS2SplitNum` 份，每份的局部 max 为 `m_k`、局部 sum 为 `s_k`、局部 P*V 累积输出为 `o_k`（形状 `[M, D]`）。FD 归约目标：

$$
O = \sum_{k=0}^{K-1} w_k \cdot o_k
$$

其中权重 `w_k` 由各份的 `(m_k, s_k)` 决定：

$$
M = \max_k(m_k), \quad
\tilde{w}_k = e^{m_k - M} \cdot s_k, \quad
W = \sum_k \tilde{w}_k, \quad
w_k = \frac{\tilde{w}_k}{W}
$$

最终 logSumExp：

$$
\text{lse} = \log W + M
$$

这与标准 FlashAttention 跨块归约公式一致，保证 FD 归约结果与单核计算等价。

对应代码实现：

- `M` → `ComputeScaleValue_VF_FD` 内的 `globalMax`
- `w_k` → `lseExp[k]`（已归一化）
- `O` → `reduceOut`（`ReduceFinalRes` 累加）
- `lse` → `maxLseUb`（`CopySoftmaxLseOutput` 输出）
