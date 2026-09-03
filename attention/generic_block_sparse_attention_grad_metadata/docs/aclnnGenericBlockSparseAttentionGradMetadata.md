# aclnnGenericBlockSparseAttentionGradMetadata

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

- **算子功能**：`aclnnGenericBlockSparseAttentionGradMetadata` 根据 `rsvdBlockIdx`、`rsvdBlockCount`、seqlen 等信息进行稀疏 attention 的分核与负载均衡，为 `aclnnGenericBlockSparseAttentionGrad` 的前置 AICPU 算子。按 **B → N2 → J → G** 顺序展开 `(b, n2, j, g)` 任务列表，并在 AIC 核间做 `[baseM, baseN] = [128, 128]` 基本块粒度的贪心负载均衡。输出 `metadata` 供主 Grad 算子消费。

  **该算子不建议单独使用，建议与 `aclnnGenericBlockSparseAttentionGrad` 配合使用，形成完整工作流。**

  1. 接收主算子的稀疏表与 shape 信息（`maxQSeqlen`、`maxKvSeqlen`、`numQHeads`、`numKvHeads`、`headDim`、`blockShape`、layout 等）。
  2. 以 **(b, n2, j) KV 块** 为调度原子单元，同块内 `g = 0..G-1` 连续展开，保证 KV L1 复用。
  3. 单 KV 块开销 = `G × ⌈count/baseM⌉ × ⌈kvBlockLen/baseN⌉`，其中 `kvBlockLen = min(By, actS2 - j×By)`。
  4. 输出 `metadata`（int64 一维张量）包含分核边界与任务列表，作为 `aclnnGenericBlockSparseAttentionGrad` 的 `metadata` 输入。

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

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize”获取 workspace 大小，再调用“aclnnGenericBlockSparseAttentionGradMetadata”执行计算。

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
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
aclnnStatus aclnnGenericBlockSparseAttentionGradMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize

### 参数说明


| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rsvdBlockIdx | 输入，必选 | Device侧的aclTensor，稀疏块索引数组，指定每个KV块选择的Q块/token索引。 | 存储每个KV块选择的Q索引。<br>TND：[B, N2, ceilDiv(maxS2, blockShapeY), maxS1]。<br>BNSD/BSND：[B, N2, ceilDiv(S2, blockShapeY), S1]。<br>同group每个KVHead对应的Q稀疏pattern一致（isPackedGQA=1）。 | INT32 | ND | 4 | √ |
| rsvdBlockCount | 输入，必选 | Device侧的aclTensor，指定每个KV块实际选择的Q数量。 | 存储每个KV块实际选择的Q数量。<br>TND：[B, N2, ceilDiv(maxS2, blockShapeY)]。<br>BNSD/BSND：[B, N2, ceilDiv(S2, blockShapeY)]。 | INT32 | ND | (B, N2, ceilDiv(maxS2, blockShapeY))或(B, N2, ceilDiv(S2, blockShapeY)) | √ |
| cuSeqLengthsQOptional | 输入，可选 | Device侧的aclTensor，描述每个Batch对应的query序列长度，以前缀和形式存储。 | 用于变长序列场景。<br>layoutQ为"TND"时必须配置；为"BNSD"或"BSND"时传nullptr。 | INT64 | ND | 1 | - |
| cuSeqLengthsKvOptional | 输入，可选 | Device侧的aclTensor，描述每个Batch对应的key/value序列长度，以前缀和形式存储。 | 用于变长序列场景。<br>layoutKv为"TND"时必须配置；为"BNSD"或"BSND"时传nullptr。 | INT64 | ND | 1 | - |
| sequsedQOptional | 输入，可选 | Device侧的aclTensor，各batch中query的实际序列长度。 | - | INT32 | ND | 1 | √ |
| sequsedKvOptional | 输入，可选 | Device侧的aclTensor，各batch中kv的实际序列长度。 | - | INT32 | ND | 1 | √ |
| maxQSeqlen | 输入，Attr | 所有batch中的qSeqlen的最大值。 | 须 ≥ 0。BNSD/BSND场景须与query的S维一致；TND场景用于校验rsvdBlockIdx最后一维。 | INT64 | - | - | - |
| maxKvSeqlen | 输入，Attr | 所有batch中的kvSeqlen的最大值。 | 须 ≥ 0。用于计算 J = ceilDiv(maxKvSeqlen, blockShapeY)。 | INT64 | - | - | - |
| numQHeads | 输入，Attr | query的head数（N1）。 | 须落在[1, 128]，且能被numKvHeads整除。 | INT64 | - | - | - |
| numKvHeads | 输入，Attr | key/value的head数（N2）。 | 须落在[1, 128]，且与rsvdBlockIdx的N2维一致。 | INT64 | - | - | - |
| headDim | 输入，Attr | query/key/value的embed维度。 | 当前固定128。 | INT64 | - | - | - |
| blockShape | 输入，Attr | 代表稀疏块形状数组。 | 含两个元素[blockShapeX, blockShapeY]。<br>blockShapeX当前仅支持1。<br>blockShapeY须≥128且按64对齐；当前实现仅支持128。 | INT64 | - | - | - |
| isPackedGQA | 输入，Attr | 代表进行块状稀疏时，同一个group内的qHead是否共享同样的稀疏pattern（当前仅支持1）。 | 0表示不共享；1表示共享。不同batch之间不共享。 | INT64 | N/A | - | - |
| layoutQ | 输入，Attr | Host侧的string，代表输入query的数据排布格式。 | 当前支持"TND"、"BNSD"、"BSND"。 | String | - | - | - |
| layoutKv | 输入，Attr | Host侧的string，代表输入key、value的数据排布格式。 | 当前支持"TND"、"BNSD"、"BSND"。 | String | - | - | - |
| maskType | 输入，Attr | Host侧的int64_t，表示attention计算中的掩码类型。当前仅支持1。 | 具体见“掩码说明”。 | INT64 | - | - | - |
| softmaxPrecision | 输入，Attr | Softmax计算采取的精度级别。 | 仅支持0或1。<br>0：online softmax和rescale均使用fp32。<br>1：online softmax使用fp16/bf16，rescale使用fp32，可能发生数值溢出。当前实现传0。 | INT64 | - | - | - |
| winLeft | 输入，Attr | Host侧的int64_t，滑窗attention场景下，滑窗需要向前包含多少个token。 | 不使能时必须为-1，需要与maskType、mask配合使用。 | INT64 | - | - | - |
| winRight | 输入，Attr | Host侧的int64_t，滑窗attention场景下，滑窗需要向后包含多少个token。 | 不使能时必须为-1，需要与maskType、mask配合使用。 | INT64 | - | - | - |
| metadata | 输出 | Device侧的aclTensor，稀疏attention的分核信息。 | 不支持空Tensor。shape [metaSize]，dtype INT64，metaSize ≥ 80 + B×N1×J×4。 | INT64 | ND | 1 | × |
| workspaceSize | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
| executor | 输出 | 返回op执行器，包含算子计算流程。 | - | - | - | - | - |

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

## aclnnGenericBlockSparseAttentionGradMetadata

### 参数说明


| 参数名           | 输入/输出 | 描述                                                                                           |
| ------------- | ----- | -------------------------------------------------------------------------------------------- |
| workspace     | 输入    | 在Device侧申请的workspace内存地址。                                                                    |
| workspaceSize | 输入    | 在Device侧申请的workspace大小，由第一段接口aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize获取。 |
| executor      | 输入    | op执行器，包含了算子计算流程。                                                                             |
| stream        | 输入    | 指定执行任务的Stream。                                                                               |


### 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 掩码说明


| maskType | 含义 | 备注 |
| :--- | :--- | :--- |
| 1 | causal mask（RIGHT_DOWN_CAUSAL） | 当前仅支持 |

## 约束说明

- 确定性计算：当前不支持（本算子输出为确定性结果，但主 Grad 算子不支持确定性计算）。
- 须与 [`aclnnGenericBlockSparseAttentionGrad`](../../generic_block_sparse_attention_grad/docs/aclnnGenericBlockSparseAttentionGrad.md) 配合使用；主算子调用前必须先成功执行本算子。
- `maskType` 当前仅支持 **1**。
- `winLeft` 和 `winRight` 不使能时必须为 **-1**。
- HeadDim 固定为 **128**。
- `numQHeads` / `numKvHeads` 须落在 **[1, 128]**，且 `numQHeads % numKvHeads == 0`，`G = numQHeads / numKvHeads`。
- `blockShape` 当前仅支持 `[1, 128]`。
- `isPackedGQA` 当前仅支持 **1**。
- `layoutQ` 与 `layoutKv` 须相同，取值 `"TND"` / `"BNSD"` / `"BSND"`。
- TND 布局下 `cuSeqLengthsQOptional` / `cuSeqLengthsKvOptional` 必选。
- `rsvdBlockIdx` 最后一维 `maxS1` 须 ≥ `maxQSeqlen`；`J = ceilDiv(maxKvSeqlen, blockShapeY)` 须与 rsvdBlockIdx 第 3 维一致。
- metadata 长度须满足 `shape[0] ≥ 80 + B × numQHeads × J × 4`。
- 任务数上界 `B × numQHeads × J ≤ 1048576`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。完整可运行样例见 `examples/test_aclnn_generic_block_sparse_attention_grad_metadata.cpp`；与主算子联调见 `../generic_block_sparse_attention_grad/examples/test_aclnn_generic_block_sparse_attention_grad.cpp`。

```Cpp
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad_metadata.h"

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
    auto ret = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, 1, layout, layout, 1, 0, -1, -1,
        metadata, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *ws = nullptr;
    if (wsSize > 0) {
        aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnGenericBlockSparseAttentionGradMetadata(ws, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclrtSynchronizeStream(stream);

    std::cout << "GenericBlockSparseAttentionGradMetadata finished successfully." << std::endl;

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
