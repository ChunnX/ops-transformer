# aclnnGenericBlockSparseAttentionMetadata

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

- **接口功能**：生成`aclnnGenericBlockSparseAttention`计算所需的`metadataOptional`。
- 输出`metadataOptional`必须作为主算子`aclnnGenericBlockSparseAttention`的`metadataOptional`输入使用，每次调用主算子前均须重新生成。
- `metadataOptional`为不透明数据，调用者不应解析或修改其中的内容。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize`接口获取workspace大小以及包含算子计算流程的执行器，再调用`aclnnGenericBlockSparseAttentionMetadata`接口执行计算。

```c++
aclnnStatus aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
  const aclTensor   *sparseBlockIdx,
  const aclTensor   *sparseBlockCount,
  const aclTensor   *cuSeqLengthsQOptional,
  const aclTensor   *cuSeqLengthsKvOptional,
  const aclTensor   *sequsedQOptional,
  const aclTensor   *sequsedKvOptional,
  int64_t            maxQSeqLen,
  int64_t            maxKvSeqLen,
  int64_t            numQHeads,
  int64_t            numKvHeads,
  int64_t            headDim,
  const aclIntArray *blockShape,
  int64_t            isPackedGQA,
  const char         *layoutQ,
  const char         *layoutKv,
  int64_t            maskType,
  int64_t            quantType,
  int64_t            softmaxPrecision,
  int64_t            winLeft,
  int64_t            winRight,
  const aclTensor   *metadataOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttentionMetadata(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize

- **参数说明**

  表格中shape变量含义见<a href="#基准说明">基准说明</a>。

  <table style="table-layout: fixed; width: 100%; word-break: break-word; overflow-wrap: anywhere;">
  <colgroup>
  <col style="width: 13%">
  <col style="width: 7%">
  <col style="width: 17%">
  <col style="width: 28%">
  <col style="width: 10%">
  <col style="width: 6%">
  <col style="width: 13%">
  <col style="width: 6%">
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
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引数组，指定每个Q块选择的KV块索引。</td>
      <td>当前仅支持TND + isPackedGQA=1。无效位置可用-1填充，有效值须位于前sparseBlockCount个位置。完整数据约束见<a href="#其他约束">其他约束</a>。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量。</td>
      <td>当前仅支持TND + isPackedGQA=1。每个元素须位于[0, maxSparseBlockCount]范围内。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的query序列长度，以前缀和形式存储。</td>
      <td>当前layoutQ为TND，必须传入。第0个元素为0，后一个元素须大于等于前一个元素；相邻差分得到各Batch的Query存储长度。详见<a href="#长度类输入约束">长度类输入约束</a>。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的key/value序列长度，以前缀和形式存储。</td>
      <td>当前layoutKv为PAGED_BBND，必须传入。第0个元素为0，后一个元素须大于等于前一个元素；相邻差分得到各Batch的KV存储长度。详见<a href="#长度类输入约束">长度类输入约束</a>。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入</td>
      <td>各Batch中query的实际序列长度。</td>
      <td>不指定实际长度可传入nullptr，表示与cuSeqLengthsQOptional差分得到的存储长度相同。传入时每个元素须大于等于0且不超过对应Batch的存储长度。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各Batch中key/value的实际序列长度。</td>
      <td>不指定实际长度可传入nullptr，表示与cuSeqLengthsKvOptional差分得到的存储长度相同。传入时每个元素须大于等于0且不超过对应Batch的存储长度。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>maxQSeqLen</td>
      <td>输入</td>
      <td>主算子Query的最大Sequence Length。</td>
      <td>必须大于0，并与主算子及长度类输入保持一致，详见<a href="#其他约束">其他约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxKvSeqLen</td>
      <td>输入</td>
      <td>主算子Key/Value的最大Sequence Length。</td>
      <td>必须大于0，并与主算子及长度类输入保持一致，详见<a href="#其他约束">其他约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQHeads</td>
      <td>输入</td>
      <td>主算子Query的head数。</td>
      <td>必须大于0，并与主算子保持一致。Q/KV头数关系见<a href="#其他约束">其他约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKvHeads</td>
      <td>输入</td>
      <td>主算子Key/Value的head数。</td>
      <td>必须大于0，并与主算子保持一致。Q/KV头数关系见<a href="#其他约束">其他约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>输入</td>
      <td>主算子Query、Key和Value每个head的特征维度。</td>
      <td>当前仅支持128，并须与主算子保持一致。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>输入</td>
      <td>代表稀疏块形状数组。</td>
      <td>含两个元素[blockShapeX, blockShapeY]，当前仅支持[1, 128]。量化场景的关联约束见主算子<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#量化相关说明">量化相关说明</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isPackedGQA</td>
      <td>输入</td>
      <td>代表进行块状稀疏时，同一个group内的qHead是否共享相同的稀疏pattern。</td>
      <td>0表示同group内的qHead不共享稀疏pattern，1表示共享；当前仅支持1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>输入</td>
      <td>代表主算子输入query的数据排布格式。</td>
      <td>当前仅支持"TND"。布局含义见主算子<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#layout对应关系说明">layout对应关系说明</a>。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>输入</td>
      <td>代表主算子输入key、value的数据排布格式。</td>
      <td>当前仅支持"PAGED_BBND"。布局含义见主算子<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#layout对应关系说明">layout对应关系说明</a>。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>输入</td>
      <td>表示Attention计算中的掩码类型。</td>
      <td>当前仅支持1（causal mask）。完整含义见主算子<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#掩码说明">掩码说明</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>输入</td>
      <td>代表主算子采用的量化方式。</td>
      <td>当前支持0；Ascend 950上可选5。完整配置见主算子<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#量化相关说明">量化相关说明</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>输入</td>
      <td>Softmax计算采用的精度级别。</td>
      <td>取值为0或1；芯片和数据类型约束见主算子<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#量化相关说明">量化相关说明</a>及本文<a href="#其他约束">其他约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>输入</td>
      <td>滑窗Attention场景下，滑窗向前包含的token数。</td>
      <td>当前仅支持-1，须与maskType配合。完整含义见主算子<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#掩码说明">掩码说明</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>输入</td>
      <td>滑窗Attention场景下，滑窗向后包含的token数。</td>
      <td>当前仅支持-1，须与maskType配合。完整含义见主算子<a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#掩码说明">掩码说明</a>。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输出</td>
      <td>AICPU算子生成的分核结果。</td>
      <td>必须传入。调用者须预先申请INT32、shape为(1024,)的Device侧Tensor；其内容为不透明格式，不应解析、修改或跨不同输入和属性复用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>当前返回0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 280px">
  <col style="width: 90px">
  <col style="width: 785px">
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
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>必选Tensor未传入，或Tensor类型、shape、layout、blockShape、头数关系、maskType、quantType、softmaxPrecision、窗口属性等不满足当前约束。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>API内部创建aclOpExecutor失败。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>workspaceSize或executor为空，或API内部执行Contiguous、注册AICPU任务时出现空指针异常。</td>
    </tr>
  </tbody>
  </table>

## aclnnGenericBlockSparseAttentionMetadata

- **参数说明**

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
      <td>在Device侧申请的workspace内存地址。当前workspaceSize为0，可传入nullptr。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 基准说明

| 命名 | 含义 |
| :---: | :--- |
| B | Batch Size，输入样本批量大小 |
| numQHeads | 主算子query的Head Num |
| numKvHeads | 主算子key/value的Head Num |
| headDim | 主算子query、key和value的Head Dim |
| qStorageLen_i | cuSeqLengthsQOptional相邻元素差分得到的第i个Batch的Query存储长度 |
| totalQBlocks | 按存储长度分块后的Q块总数，即$\sum_i \mathrm{ceilDiv}(qStorageLen_i, blockShapeX)$ |
| maxSparseBlockCount | sparseBlockIdx最后一维，须不小于sparseBlockCount中所有元素的最大值 |

### 布局与稀疏输入约束

- 当前仅支持`layoutQ="TND"`和`layoutKv="PAGED_BBND"`。布局含义见主算子[layout对应关系说明](../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#layout对应关系说明)。
- 当前仅支持`isPackedGQA=1`。此时同group内的qHead共享相同的稀疏pattern：
  - `sparseBlockIdx`的shape为`[numKvHeads, totalQBlocks, maxSparseBlockCount]`；
  - `sparseBlockCount`的shape为`[numKvHeads, totalQBlocks]`。
- `maxSparseBlockCount`须大于0、不超过256，且须不小于`sparseBlockCount`中所有元素的最大值。
- `sparseBlockCount`每个元素须位于`[0, maxSparseBlockCount]`范围内；`sparseBlockIdx`前`sparseBlockCount`个位置存放当前Q块的有效KV块索引。

### 长度类输入约束

- `cuSeqLengthsQOptional`和`cuSeqLengthsKvOptional`当前必须传入，数据类型为INT64，shape均为`(B+1,)`。
- 两个前缀和Tensor的第0个元素均须为0，后一个元素须大于等于前一个元素；各Batch的差分长度不得超过对应的`maxQSeqLen`或`maxKvSeqLen`。
- `sequsedQOptional`和`sequsedKvOptional`的数据类型为INT32、shape为`(B,)`，可传入nullptr。
- 未传入seqused时，实际长度等于对应前缀和差分得到的存储长度；传入时，每个元素须位于`[0, 对应Batch存储长度]`范围内。
- 分核和任务空间按各Batch实际有效长度累加；稀疏块分块与主算子QKV寻址仍按cu前缀和描述的存储长度计算。

### 属性与平台约束

- `maxQSeqLen`、`maxKvSeqLen`、`numQHeads`、`numKvHeads`和`headDim`均须大于0。
- `numQHeads >= numKvHeads`且`numQHeads % numKvHeads == 0`；`groupSize = numQHeads / numKvHeads`不超过128。
- 当前仅支持`headDim=128`、`blockShape=[1, 128]`。
- `maskType`当前仅支持1，`winLeft`和`winRight`当前仅支持-1。完整含义见主算子[掩码说明](../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#掩码说明)。
- `quantType`当前支持0；Ascend 950上可选5，其他取值不支持。量化配置须与随后调用的主算子完全一致，详见主算子[量化相关说明](../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#量化相关说明)。
- `softmaxPrecision`取值为0或1；Ascend 950仅支持1。其取值须与随后调用的主算子完全一致。

### 其他约束

- 本接口必须与`aclnnGenericBlockSparseAttention`配套使用。共同Tensor和属性须与随后调用的主算子完全一致，每次调用主算子前均须重新生成`metadataOptional`。
- 主算子的完整输入组合、数据类型、Paged Attention和量化约束见[aclnnGenericBlockSparseAttention](../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md#约束说明)。
- 输出`metadataOptional`固定为INT32、shape为`(1024,)`，对应主算子的`metadataOptional`输入，不得解析、修改或跨不同输入和属性复用。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

aclnnStatus RunMetadata(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                        const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional,
                        aclTensor *metadataOptional, aclrtStream stream)
{
    const int64_t blockShapeData[] = {1, 128};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    if (blockShape == nullptr) {
        return ACL_ERROR_BAD_ALLOC;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        sparseBlockIdx, sparseBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, nullptr, nullptr, 16, 2048,
        32, 8, 128, blockShape, 1, "TND", "PAGED_BBND", 1, 0, 0, -1, -1, metadataOptional, &workspaceSize,
        &executor);
    aclDestroyIntArray(blockShape);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    ret = aclnnGenericBlockSparseAttentionMetadata(workspace, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
