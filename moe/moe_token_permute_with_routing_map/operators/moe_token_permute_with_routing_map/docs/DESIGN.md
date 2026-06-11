# moe_token_permute_with_routing_map 详细设计文档

## 1. 修订记录

| 版本 | 日期 | 作者 | 修订内容 |
|------|------|------|----------|
| v1.0 | 2026-05-25 | - | 初始版本 |

## 2. 设计概述

### 2.1 设计目标

增强 `moe_token_permute_with_routing_map` 算子，在 `dropAndPad=false` 时支持每个 token 选中的专家数少于 topK 的场景。

### 2.2 核心设计思路

1. **maskselect**：汇总各核算出的非零元素 offset，得出实际 `actualOutTokens`
2. **sort**：新增 tiling key 分支，当 `numOutTokens != topK * numTokens` 时强制多核排序；归并输出阶段用 -1 补齐每个 token 到 topK 个条目
3. **tokencopy**：遍历 sorted_indices 时，跳过值为 -1 的位置

### 2.3 整体数据流

```
routing_map [num_experts, num_tokens]
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ maskselect (KernelMaskedSelectV3)                           │
│   输出: sorted_indices (actualOutTokens 个 int32)           │
│   新增: 汇总 actualOutTokens 写入 workspace                  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ sort (MoeSortMultiCore)                                     │
│   新增分支: 判断 actualOutTokens != topK * numTokens        │
│   → 强制多核排序, 基于 actualOutTokens 计算 tiling          │
│   → MoeMrgsortOut::Extract 后用 -1 补齐到 topK*numTokens   │
│   输出: sorted_indices [topK * numTokens] (含 -1 补齐)      │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ tokencopy (MoeindexCopyOp / MoeindexCopySplitDOp)           │
│   新增: 遍历 indices 时跳过值为 -1 的位置                   │
│   输出: permuted_tokens [topK * numTokens, hidden]          │
└─────────────────────────────────────────────────────────────┘
```

## 3. 详细修改方案

> **设计原则**：Tiling 侧不修改。因为只有在 kernel 侧 maskselect 计算完后才能知道总 token 个数和 `topK * numTokens` 是否相同，tiling 侧编译时无法判断。所有逻辑判断和参数重算都在 kernel 侧运行时完成。

### 3.1 Tiling 层（不修改）

**Tiling 侧完全不做修改**。`tilingKey` 仍按原有的 `totalLength = topK * numTokens` 计算（SORT_ONE_CORE_MODE 或 SORT_MULTI_CORE_MODE），workspace 大小也不变。

原因：tiling 侧只拿到 `numOutTokens` 参数，但 `numOutTokens` 是用户传入的上限值，实际 `actualOutTokens` 要等到 kernel 侧 maskselect 执行完才知道（因为 routing_map 是运行时数据，tiling 阶段不可见）。

### 3.2 Kernel 层修改

#### 3.2.1 整体流程变化

```
原有流程:
  maskselect → sort → tokencopy

修改后流程:
  maskselect → [计算 actualOutTokens] → sort → [补齐 -1] → tokencopy
                        │                              │
                        ▼                              ▼
              actualOutTokens !=        sorted_indices 中 -1 位置
              topK*numTokens 时重算     tokencopy 跳过不搬出
              sort tiling 参数
```

#### 3.2.2 maskselect — 输出 actualOutTokens

**文件**: `op_kernel/masked_select_v3.h`

在 `KernelMaskedSelectV3::Process()` 末尾，所有核 SyncAll 之后，由 blockIdx=0 的核汇总所有核的 offset，计算出总 `actualOutTokens`。

当前已有关键逻辑：各核的 offset 通过 `offsetLocal` 做了 AllGather 前缀和，所以 `offsetLocal.GetValue(needCoreNum - 1)` 已经是最末尾核的前缀和 = actualOutTokens。

**修改**：`Process()` 返回 `actualOutTokens`（通过输出参数），不写 GM，在主 kernel 中直接作为局部变量传递给后续 sort 阶段。

```cpp
// 在 KernelMaskedSelectV3::Process() 末尾，SyncAll() 之后新增：
// 通过输出参数 actualOutTokens 返回，blockIdx=0 核汇总
if (this->blockIdx == 0) {
    actualOutTokens = offsetLocal.GetValue(needCoreNum - 1);
}
// sync 确保所有核读到一致的 actualOutTokens
```

#### 3.2.3 sort — Kernel 侧重算排序 tiling 参数

**文件**: `op_kernel/moe_token_permute_with_routing_map.cpp`（主 kernel 文件）

maskselect 完成后，在进入 sort 前读取 `actualOutTokens`。若 `actualOutTokens != topK * numTokens`，则需要重算 sort 的 tiling 参数（在 kernel 侧完成），强制使用多核排序。

```cpp
// 主 kernel 流程中，maskselect 之后，sort 之前新增：
// actualOutTokens 是 maskselect 返回的局部变量，不需要从 GM 读取

int64_t totalLength = topK * numTokens;
bool needPad = (actualOutTokens < totalLength);

// 重算 sort tiling 参数
PermuteVBSComputeRMTilingData localSortTiling;
if (needPad) {
    // 基于 actualOutTokens 重新计算多核排序的切分参数
    // 复用与 tiling 侧 Tinlig4VBSMultiCoreCompute 相同的计算逻辑
    int64_t oneLoopMaxElements = vbsTilingData->get_oneLoopMaxElements();
    int64_t needCoreNum = GetCeilInt(actualOutTokens, oneLoopMaxElements);
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
    needCoreNum = std::min(needCoreNum, realCoreNumAiv);
    
    int64_t perCoreElements = GetDiv(actualOutTokens, needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = actualOutTokens - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = GetCeilInt(actualOutTokens, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }
    
    localSortTiling.set_needCoreNum(needCoreNum);
    localSortTiling.set_perCoreElements(perCoreElements);
    localSortTiling.set_lastCoreElements(actualOutTokens - (needCoreNum - 1) * perCoreElements);
    // ... 设置其他必要字段
}
```

**关键**：sort 阶段需要使用 `localSortTiling`（基于 actualOutTokens 重算的）而不是原始的 vbsTilingData。

#### 3.2.4 sort — MoeSortMultiCore 适配

**文件**: `op_kernel/moe_sort_multi_core.h`

`MoeSortMultiCore` 需要适配：
1. `Init()` 中读取 `needPad` 标志，设置 `mrgsortParam.needPad`
2. 将 `needPad`、`topK`、`numTokens` 传递给 `MoeMrgsortOut`

#### 3.2.5 sort — MoeMrgsortOut 补齐 -1（策略1：Scalar 操作）

**文件**: `op_kernel/moe_mrgsort_out.h`

在 `Extract()` 之后、`CopyOut()` 之前，插入补齐逻辑。采用策略1（Scalar 操作）：遍历排序结果，逐 token 统计条目数，不足 topK 的位置写入 -1。

**逻辑说明**：

Extract 后的数据格式为 `(key, value)` 对，其中 key 是 token index（排序用的键），value 是排序后该 key 对应的位置。排序结果按 key 分组排列。

举例（token 数=3，topK=3，actualOutTokens=6）：
- Extract 前排序结果：`(0,2), (0,4), (1,5), (1,6), (1,7)` — 共 5 对，token0 有 2 个，token1 有 3 个
- 补齐后应输出：`2, 4, -1, 5, 6, 7, -1, -1, -1` — 共 9 个（topK × numTokens）

**实现方案**：

```cpp
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::PadOutput()
{
    if (!this->param->needPad) {
        return;
    }

    int64_t topK = this->param->topK;
    int64_t numTokens = this->param->numTokens;

    // ubOutput1 中是排序后的 key (float, 即 token index)
    // ubOutput2 中是排序后的 value (int32, 即排列位置)
    // tempBuffer 用作补齐后的输出缓冲区
    LocalTensor<int32_t> keysLocal = this->ubOutput1.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> valuesLocal = this->ubOutput2.ReinterpretCast<int32_t>();

    int64_t sortedCount = curLoopSortedNum;  // Extract 后的有效元素数
    int64_t outPos = 0;      // 补齐后输出位置
    int64_t sortedIdx = 0;   // 排序结果的读取位置

    // 按 token 分组补齐
    for (int64_t tokenId = 0; tokenId < numTokens && sortedIdx < sortedCount; tokenId++) {
        int64_t countForToken = 0;
        // 统计当前 token 在排序结果中的条目数
        int64_t startIdx = sortedIdx;
        while (sortedIdx < sortedCount && keysLocal.GetValue(sortedIdx) == tokenId) {
            sortedIdx++;
            countForToken++;
        }
        // 输出该 token 的有效 value
        for (int64_t j = 0; j < countForToken; j++) {
            tempBuffer.SetValue(outPos++, valuesLocal.GetValue(startIdx + j));
        }
        // 用 -1 补齐到 topK
        for (int64_t j = countForToken; j < topK; j++) {
            tempBuffer.SetValue(outPos++, -1);
        }
    }
    // 处理末尾 token（排序结果中可能没有最后一两个 token 的数据）
    for (int64_t tokenId = sortedIdx > 0 ? keysLocal.GetValue(sortedIdx - 1) + 1 : 0; 
         tokenId < numTokens; tokenId++) {
        for (int64_t j = 0; j < topK; j++) {
            tempBuffer.SetValue(outPos++, -1);
        }
    }

    // 将补齐后的结果从 tempBuffer 拷贝回 ubOutput2，后续 CopyOut 使用
    // ubOutput2 = tempBuffer[0:outPos]
    for (int64_t i = 0; i < outPos; i++) {
        this->ubOutput2.SetValue(i, tempBuffer.GetValue(i));
    }

    // 更新 curLoopSortedNum 为补齐后的长度
    curLoopSortedNum = outPos;
    this->allRemainElements = outPos;  // 已补齐，剩余需要输出的元素数
}
```

**Process 流程修改**：

```cpp
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::Process()
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        Extract();
        PadOutput();      // 新增：补齐 -1
        CopyOut();
    }
    ClearCache();
}
```

**MoeMrgsortParam 结构体扩展**（`moe_mrgsort.h`）：

```cpp
struct MoeMrgsortParam {
    int64_t perListElements;
    int64_t lastListElements;
    int64_t oneLoopMaxElements;
    bool needPad;        // 新增：是否需要 -1 补齐
    int64_t topK;        // 新增：topK 值
    int64_t numTokens;   // 新增：token 总数
};
```

#### 3.2.6 tokencopy — 跳过 -1 位置不搬出

**文件**: `op_kernel/moe_index_copy.h` 和 `op_kernel/moe_index_copy_split_d.h`

遍历 sorted_indices 时增加 -1 判断，遇到 -1 则跳过不搬出。

**MoeindexCopyOp 修改**：

```cpp
// 在 CopyOut 中的 topKId 循环内增加判断：
auto indicesValue = indicesLocal.GetValue(offset);
if (indicesValue == -1) {  // 新增：跳过 -1
    offset++;
    continue;
}
```

**MoeindexCopySplitDOp 修改**：

```cpp
// 在 CopyInAndOut 的两处 topKId 循环中增加判断：
for (int64_t topKId = 0; topKId < topK; topKId++) {
    auto indicesValue = indicesLocal.GetValue(indicesOffset + topKId);
    if (indicesValue == -1) {  // 新增：跳过 -1
        continue;
    }
    // ... 原有搬移逻辑
}
```

### 3.3 数据结构修改汇总

#### 3.3.1 MoeMrgsortParam（moe_mrgsort.h）

```cpp
struct MoeMrgsortParam {
    int64_t perListElements;
    int64_t lastListElements;
    int64_t oneLoopMaxElements;
    bool needPad;       // 新增：是否需要 -1 补齐
    int64_t topK;       // 新增：topK 值
    int64_t numTokens;  // 新增：token 总数
};
```

#### 3.3.2 Tiling 结构体（不修改）

Tiling 侧所有结构体保持不变。`needPad`、`topK`、`numTokens` 等运行时判断参数在 kernel 侧通过局部变量传递，不写入 tiling data。

### 3.4 数据传递

maskselect → sort 之间通过局部变量传递 `actualOutTokens`，不经过 GM/workspace：

```cpp
// 主 kernel 中：
int32_t actualOutTokens = 0;
KernelMaskedSelectV3::Process(..., actualOutTokens);  // 输出参数

// 直接传递给 sort 阶段
if (actualOutTokens < topK * numTokens) {
    // 重算 tiling，强制多核排序
}
```

不需要额外 GM 空间或 workspace。

### 3.5 向后兼容性

| 场景 | 条件 | 行为 |
|------|------|------|
| 原有对齐场景 | `actualOutTokens == topK * numTokens` | `needPad=false`，不触发任何新增逻辑，与原完全一致 |
| 新增非对齐场景 | `actualOutTokens < topK * numTokens` | kernel 侧重算 tiling + -1 补齐 + 跳过 -1 搬出 |

## 4. 向后兼容性

| 场景 | 条件 | 行为 |
|------|------|------|
| 原有对齐场景 | `numOutTokens == topK * numTokens` | tiling key 不变，与原逻辑一致 |
| 新增非对齐场景 | `numOutTokens < topK * numTokens` | 新 tiling key + -1 补齐 + 跳过 -1 |
| padded 模式 | `paddedMode == true` | 不受影响，tiling key 9/19 不变 |

## 5. 性能评估

| 阶段 | 额外开销 | 说明 |
|------|---------|------|
| maskselect | +O(1) | 仅增加一次汇总写入 |
| sort | +O(N) scalar 遍历 | 补齐逻辑为线性扫描 |
| tokencopy | +O(K) 分支判断 | 每个 topKId 增加一次比较 |

对于典型场景（numTokens=1024, topK=8），补齐逻辑处理 ~8192 个元素，开销可忽略。

## 6. 测试策略

### 6.1 单元测试

- op_host: Tiling key 分支覆盖（原有 key 1-8 + 新增 key 8-14）
- op_kernel: 各 tiling key 分支的 kernel 编译验证

### 6.2 功能测试

| 用例 | 描述 |
|------|------|
| 对齐场景 | numOutTokens == topK * numTokens，验证与原有行为一致 |
| 少一个专家 | 某个 token 少选 1 个专家 |
| 极端稀疏 | 每个 token 只选 1 个专家 |
| 混合场景 | 部分 token 满 topK，部分不满 |

### 6.3 精度测试

- sorted_indices 中 -1 的位置应与参考实现一致
- permuted_tokens 中有效位置的数值应与参考实现一致

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| -1 补齐逻辑 UB 溢出 | 数据错误 | 计算补齐后 curLoopSortedNum，确保不超过 UB 容量 |
| tiling key 冲突 | 分支错误 | 新增 key 使用未被占用的值（8/10/12/14） |
| workspace 越界 | 内存错误 | 在 GetWorkspaceSize 中预留 actualOutTokens 存储空间 |