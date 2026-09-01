# aclnnGenericSparseAttentionGradMetadata

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **算子功能**：GenericSparseAttentionGradMetadata 是 AICPU 算子，为 `aclnnGenericSparseAttentionGrad` 的前置算子。根据 KV2Q 稀疏块索引表 `rsvdBlockIdx` / `rsvdBlockCount` 以及 layout、head、block 等配置，按 **B → N2 → J → G** 顺序展开 `(b, n2, j, g)` 任务列表，并在 AIC 核间做 `[baseM, baseN] = [128, 128]` 基本块粒度的贪心负载均衡。输出 `metadata` 供主 Grad 算子消费，减少主算子 tiling 阶段对 host 侧稀疏表的访问。

  **该算子不建议单独使用，建议与 `aclnnGenericSparseAttentionGrad` 配合使用，形成完整工作流。**

  1. 接收主算子的稀疏表与 shape 信息（`maxQSeqlen`、`maxKvSeqlen`、`numQHeads`、`numKvHeads`、`headDim`、`blockShape`、layout 等）。
  2. 以 **(b, n2, j) KV 块** 为调度原子单元，同块内 `g = 0..G-1` 连续展开，保证 KV L1 复用。
  3. 单 KV 块开销 = `G × ⌈count/baseM⌉ × ⌈kvBlockLen/baseN⌉`，其中 `kvBlockLen = min(By, actS2 - j×By)`。
  4. 输出 `metadata`（int64 一维张量）包含分核边界与任务列表，作为 `aclnnGenericSparseAttentionGrad` 的 `metadata` 输入。

## Metadata 布局

| 偏移（int64 元素下标） | 字段 | 含义 |
| :--- | :--- | :--- |
| 0 | total_num | 有效任务总数（= KV 块组数 × G） |
| 1 | total_block_cost | 全部 `[128,128]` 基本块总数 |
| 2 | max_core_block_cost | 单核最大基本块数 |
| 3 | base_m | 128 |
| 4 | base_n | = blockShapeY（默认 128） |
| 5 | used_core_num | 实际启用核数 |
| 6 | group_size | G = N1 / N2 |
| 7 | max_task_count | 单 KV 块最大 count |
| 8..43 | core_task_start | 每核任务起始下标（最多 36 个 AIC 核） |
| 44..79 | core_task_end | 每核任务结束下标（开区间） |
| 80+ | task_list | 每任务 4 元组 `(b, n2, j, g)` |

**metadata 长度（int64 元素个数）**：

$$
\text{metaSize} = 80 + B \times N1 \times J \times 4
$$

其中 `J = ceilDiv(maxKvSeqlen, blockShapeY)`，`TASK_LIST_OFFSET = 80`，`TASK_ENTRY_SIZE = 4`。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGenericSparseAttentionGradMetadataGetWorkspaceSize”获取 workspace 大小，再调用“aclnnGenericSparseAttentionGradMetadata”执行计算。

```c++
aclnnStatus aclnnGenericSparseAttentionGradMetadataGetWorkspaceSize(
    const aclTensor *rsvdBlockIdx,
    const aclTensor *rsvdBlockCount,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen,
    int64_t maxKvSeqlen,
    int64_t numQHeads,
    int64_t numKvHeads,
    int64_t headDim,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    char *layoutQ,
    char *layoutKv,
    int64_t maskType,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnGenericSparseAttentionGradMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnGenericSparseAttentionGradMetadataGetWorkspaceSize

### 参数说明

<table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
<col style="width: 220px">
<col style="width: 100px">
<col style="width: 320px">
<col style="width: 400px">
<col style="width: 150px">
<col style="width: 80px">
<col style="width: 190px">
<col style="width: 80px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>使用说明</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>rsvdBlockIdx</td>
    <td>输入</td>
    <td>KV2Q 稀疏索引表。</td>
    <td>不支持空 Tensor。有效位置填 Q token 下标，无效填 -1。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[B, N2, J, maxS1]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>rsvdBlockCount</td>
    <td>输入</td>
    <td>每个 KV 块选中的 Q 数量。</td>
    <td>不支持空 Tensor。须与 rsvdBlockIdx 的 B/N2/J 一致。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[B, N2, J]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>cuSeqLengthsQOptional</td>
    <td>输入</td>
    <td>Query 累积序列长度（packed）。</td>
    <td>layoutQ 为 "TND" 时必选。shape [B+1]，dtype INT64。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
  </tr>
  <tr>
    <td>cuSeqLengthsKvOptional</td>
    <td>输入</td>
    <td>Key/Value 累积序列长度（packed）。</td>
    <td>layoutKv 为 "TND" 时必选。shape [B+1]，dtype INT64。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
  </tr>
  <tr>
    <td>sequsedQOptional</td>
    <td>输入</td>
    <td>每个 batch 实际 Q 长度。</td>
    <td>可选。shape [B]，dtype INT32。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
  </tr>
  <tr>
    <td>sequsedKvOptional</td>
    <td>输入</td>
    <td>每个 batch 实际 KV 长度。</td>
    <td>可选。shape [B]，dtype INT32。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
  </tr>
  <tr>
    <td>maxQSeqlen</td>
    <td>输入</td>
    <td>最大 Q 序列长度。</td>
    <td>须 ≥ 0。BNSD/BSND 场景须与 query 的 S 维一致；TND 场景用于校验 rsvdBlockIdx 最后一维。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>maxKvSeqlen</td>
    <td>输入</td>
    <td>最大 KV 序列长度。</td>
    <td>须 ≥ 0。用于计算 J = ceilDiv(maxKvSeqlen, blockShapeY)。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numQHeads</td>
    <td>输入</td>
    <td>Q head 数 N1。</td>
    <td>须 > 0，且能被 numKvHeads 整除。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numKvHeads</td>
    <td>输入</td>
    <td>KV head 数 N2。</td>
    <td>须 > 0，且与 rsvdBlockIdx 的 N2 维一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>headDim</td>
    <td>输入</td>
    <td>Head 维度 D。</td>
    <td>当前固定 128。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>blockShape</td>
    <td>输入</td>
    <td>稀疏块尺寸 [blockShapeX, blockShapeY]。</td>
    <td>blockShapeX=1；blockShapeY≥128 且 64 对齐。默认 [1, 128]。</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>isPackedGQA</td>
    <td>输入</td>
    <td>是否 packed GQA。</td>
    <td>当前仅支持 1。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layoutQ</td>
    <td>输入</td>
    <td>Q 侧 layout。</td>
    <td>支持 "TND"/"BNSD"/"BSND"，须与 layoutKv 一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layoutKv</td>
    <td>输入</td>
    <td>KV 侧 layout。</td>
    <td>支持 "TND"/"BNSD"/"BSND"，须与 layoutQ 一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>maskType</td>
    <td>输入</td>
    <td>内置 mask 类型。</td>
    <td>当前仅支持 1：RIGHT_DOWN_CAUSAL。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>softmaxPrecision</td>
    <td>输入</td>
    <td>Softmax 精度模式。</td>
    <td>传递给主 Grad 算子。当前传 0。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>winLeft</td>
    <td>输入</td>
    <td>滑窗左边界。</td>
    <td>当前须传 -1。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>winRight</td>
    <td>输入</td>
    <td>滑窗右边界。</td>
    <td>当前须传 -1。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>metadata</td>
    <td>输出</td>
    <td>分核与任务列表元数据。</td>
    <td>不支持空 Tensor。shape [metaSize]，dtype INT64，metaSize ≥ 80 + B×N1×J×4。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>Device 侧 workspace 字节数。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>输出</td>
    <td>算子执行器。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</tbody>
</table>

### 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
<col style="width: 319px">
<col style="width: 144px">
<col style="width: 671px">
</colgroup>
<thead>
  <tr>
    <th>返回码</th>
    <th>错误码</th>
    <th>描述</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
    <td>561101</td>
    <td>创建 aclOpExecutor 失败。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_INNER_NULLPTR</td>
    <td>561103</td>
    <td>workspaceSize 或 executor 为空指针；Contiguous 处理后中间 Tensor 为空。</td>
  </tr>
  <tr>
    <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3">161002</td>
    <td>rsvdBlockIdx/rsvdBlockCount/metadata 为空或 shape/dtype 非法。</td>
  </tr>
  <tr>
    <td>layout 为 TND 时未提供 cuSeqLengths；headDim、blockShape、isPackedGQA、window 参数不在支持范围。</td>
  </tr>
  <tr>
    <td>rsvdBlockIdx 的 J 维与 ceilDiv(maxKvSeqlen, blockShapeY) 不一致；metadata 长度不足；B×N1×J 超过任务数上限（1M）。</td>
  </tr>
</tbody>
</table>

## aclnnGenericSparseAttentionGradMetadata

### 参数说明

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
<col style="width: 168px">
<col style="width: 128px">
<col style="width: 854px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>workspace</td>
    <td>输入</td>
    <td>Device 侧 workspace 地址。</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>由 GetWorkspaceSize 返回的 workspace 大小。</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>输入</td>
    <td>算子执行器。</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>输入</td>
    <td>AscendCL stream。</td>
  </tr>
</tbody>
</table>

### 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

* 该算子为 AICPU 算子，输出为确定性结果。
* 须与 [`aclnnGenericSparseAttentionGrad`](../generic_sparse_attention_grad/docs/aclnnGenericSparseAttentionGrad.md) 配合使用；主算子调用前必须先成功执行本算子。
* HeadDim 固定为 **128**。
* `numQHeads % numKvHeads == 0`，`G = numQHeads / numKvHeads`。
* `blockShape` 当前仅支持 `[1, 128]`。
* `isPackedGQA` 当前仅支持 **1**。
* `winLeft` / `winRight` 当前仅支持 **-1**。
* `layoutQ` 与 `layoutKv` 须相同，取值 `"TND"` / `"BNSD"` / `"BSND"`。
* TND 布局下 `cuSeqLengthsQOptional` / `cuSeqLengthsKvOptional` 必选。
* `rsvdBlockIdx` 最后一维 `maxS1` 须 ≥ `maxQSeqlen`；`J = ceilDiv(maxKvSeqlen, blockShapeY)` 须与 rsvdBlockIdx 第 3 维一致。
* metadata 长度须满足 `shape[0] ≥ 80 + B × numQHeads × J × 4`。
* 任务数上界 `B × numQHeads × J ≤ 1048576`。

### maskType

| maskType | 含义 | 备注 |
| :--- | :--- | :--- |
| 1 | RIGHT_DOWN_CAUSAL（右下角原点因果 mask） | 当前仅支持 |

* `numQHeads` / `numKvHeads` 须落在 **[1, 128]**。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。完整可运行样例见 `examples/test_aclnn_generic_sparse_attention_grad_metadata.cpp`；与主算子联调见 `../generic_sparse_attention_grad/examples/test_aclnn_generic_sparse_attention_grad.cpp`。

```Cpp
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_sparse_attention_grad_metadata.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t n = 1;
    for (auto v : shape) {
        n *= v;
    }
    return n;
}

int main()
{
    int32_t deviceId = 0;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    const int64_t B = 1, N1 = 1, N2 = 1, S1 = 128, S2 = 128, D = 128;
    const int64_t blockY = 128;
    const int64_t J = (S2 + blockY - 1) / blockY;
    const int64_t metaSize = 80 + B * N1 * J * 4;

    std::vector<int64_t> idxShape = {B, N2, J, S1};
    std::vector<int64_t> cntShape = {B, N2, J};
    std::vector<int64_t> metaShape = {metaSize};

    std::vector<int32_t> idxHost(static_cast<size_t>(GetShapeSize(idxShape)), -1);
    std::vector<int32_t> cntHost(static_cast<size_t>(GetShapeSize(cntShape)), 0);
    std::vector<int64_t> metaHost(static_cast<size_t>(metaSize), 0);
    for (int64_t q = 0; q < S1; ++q) {
        idxHost[static_cast<size_t>(q)] = static_cast<int32_t>(q);
    }
    cntHost[0] = static_cast<int32_t>(S1);

    void *idxAddr = nullptr, *cntAddr = nullptr, *metaAddr = nullptr;
    aclTensor *idx = nullptr, *cnt = nullptr, *metadata = nullptr;

    aclrtMalloc(&idxAddr, idxHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(idxAddr, idxHost.size() * sizeof(int32_t), idxHost.data(), idxHost.size() * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMalloc(&cntAddr, cntHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(cntAddr, cntHost.size() * sizeof(int32_t), cntHost.data(), cntHost.size() * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMalloc(&metaAddr, metaHost.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(metaAddr, metaHost.size() * sizeof(int64_t), metaHost.data(), metaHost.size() * sizeof(int64_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    idx = aclCreateTensor(idxShape.data(), idxShape.size(), aclDataType::ACL_INT32, nullptr, 0,
                          aclFormat::ACL_FORMAT_ND, idxShape.data(), idxShape.size(), idxAddr);
    cnt = aclCreateTensor(cntShape.data(), cntShape.size(), aclDataType::ACL_INT32, nullptr, 0,
                          aclFormat::ACL_FORMAT_ND, cntShape.data(), cntShape.size(), cntAddr);
    metadata = aclCreateTensor(metaShape.data(), metaShape.size(), aclDataType::ACL_INT64, nullptr, 0,
                               aclFormat::ACL_FORMAT_ND, metaShape.data(), metaShape.size(), metaAddr);

    const int64_t blockShapeData[] = {1, blockY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    char layout[] = "BNSD";

    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnGenericSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, 1, layout, layout, 1, 0, -1, -1,
        metadata, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *ws = nullptr;
    if (wsSize > 0) {
        aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnGenericSparseAttentionGradMetadata(ws, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclrtSynchronizeStream(stream);

    std::cout << "GenericSparseAttentionGradMetadata finished successfully." << std::endl;

    aclDestroyIntArray(blockShape);
    aclDestroyTensor(idx);
    aclDestroyTensor(cnt);
    aclDestroyTensor(metadata);
    aclrtFree(idxAddr);
    aclrtFree(cntAddr);
    aclrtFree(metaAddr);
    if (wsSize > 0) {
        aclrtFree(ws);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
