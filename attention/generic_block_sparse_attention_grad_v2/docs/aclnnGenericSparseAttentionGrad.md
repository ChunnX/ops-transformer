# aclnnGenericSparseAttentionGrad

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

* **算子功能**：GenericSparseAttentionGrad 是通用块稀疏注意力的反向计算算子。算子以 **KV2Q** 稀疏表（`rsvdBlockIdx` / `rsvdBlockCount`）描述每个 KV 块选中的 Q token 集合，结合正向输出 `out`、`lse` 与上游梯度 `dout`，计算 `dQuery`、`dKey`、`dValue`。调用前须先通过 `aclnnGenericSparseAttentionGradMetadata` 生成分核 `metadata`。
* **稀疏表语义**：

  * `rsvdBlockIdx` shape 为 `[B, N2, J, maxS1]`：第 `j` 个 KV 块（长度 `blockShapeY`）对应的选中 Q token 下标列表，无效位置填 `-1`。
  * `rsvdBlockCount` shape 为 `[B, N2, J]`：每个 KV 块实际有效 Q 数量。
  * `J = ceilDiv(maxKvSeqLength, blockShapeY)`。
* **计算公式**：稀疏块大小为 $blockShapeX \times blockShapeY$（当前 $blockShapeX=1$）。对每个有效 KV 块，按稀疏表 Gather 对应 Q 行后执行与 FlashAttention 同类的反向：

  已知正向在稀疏支持集上满足：

  $$
  S = Mask(scale \cdot Q_{sel} \cdot K_{blk}^{T}, atten\_mask)
  $$

  $$
  P = Softmax(S)
  $$

  $$
  Out_{sel} = P \cdot V_{blk}
  $$

  记正向写出的全序列结果为 $Out$、$LSE$。反向计算：

  $$
  softmax\_grad = SoftmaxGrad(dOut, Out)
  $$

  $$
  dP = dOut_{sel} \cdot V_{blk}^{T}
  $$

  $$
  dS = P \odot (dP - softmax\_grad_{sel})
  $$

  $$
  dV_{blk} = P^{T} \cdot dOut_{sel}
  $$

  $$
  dQ_{sel} = (dS \cdot K_{blk}) \cdot scale
  $$

  $$
  dK_{blk} = (dS^{T} \cdot Q_{sel}) \cdot scale
  $$

  其中 $dQ$ 按稀疏表 ScatterAdd 回原 Q 位置；$dK$、$dV$ 按 KV 块写回。

输入 `query`/`key`/`value`/`dout`/`out` 的排布由 `qInputLayout`、`kvInputLayout` 指定。布局字母含义：

* B：Batch
* T：B 与 S 合轴后的 Total tokens
* S：Seq-Length
* N：Head-Num
* D：Head-Dim

当前支持：

* `qInputLayout` / `kvInputLayout`：`"TND"`、`"BNSD"`、`"BSND"`（二者须一致）

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGenericSparseAttentionGradGetWorkspaceSize”获取 workspace 与执行器，再调用“aclnnGenericSparseAttentionGrad”执行计算。

```c++
aclnnStatus aclnnGenericSparseAttentionGradGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *dout,
    const aclTensor *out,
    const aclTensor *lse,
    const aclTensor *rsvdBlockIdx,
    const aclTensor *rsvdBlockCount,
    const aclTensor *metadata,
    const aclTensor *attenMaskOptional,
    const aclTensor *cuSeqLengthsOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclIntArray *blockShape,
    int64_t isPackedGqa,
    char *qInputLayout,
    char *kvInputLayout,
    double scaleValue,
    int64_t maskType,
    int64_t softmaxPrecision,
    int64_t windowSizeLeft,
    int64_t windowSizeRight,
    aclTensor *dQuery,
    aclTensor *dKey,
    aclTensor *dValue,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnGenericSparseAttentionGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
```

## aclnnGenericSparseAttentionGradGetWorkspaceSize

### 参数说明

<table style="undefined;table-layout: fixed; width: 1550px">
<colgroup>
    <col style="width: 220px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 400px">
    <col style="width: 180px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 100px">
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
    <td>query</td>
    <td>输入</td>
    <td>注意力 Query。</td>
    <td>不支持空 Tensor。<br>支持的 shape：<ul><li>TND: [T1, N1, D]</li><li>BNSD: [B, N1, S1, D]</li><li>BSND: [B, S1, N1, D]</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key</td>
    <td>输入</td>
    <td>注意力 Key。</td>
    <td>不支持空 Tensor。<br>支持的 shape：<ul><li>TND: [T2, N2, D]</li><li>BNSD: [B, N2, S2, D]</li><li>BSND: [B, S2, N2, D]</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>value</td>
    <td>输入</td>
    <td>注意力 Value。</td>
    <td>不支持空 Tensor。shape 与 key 一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>dout</td>
    <td>输入</td>
    <td>正向输出梯度。</td>
    <td>不支持空 Tensor。shape/dtype 与 query 一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>out</td>
    <td>输入</td>
    <td>正向注意力输出。</td>
    <td>不支持空 Tensor。shape/dtype 与 query 一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>lse</td>
    <td>输入</td>
    <td>正向 Softmax log-sum-exp。</td>
    <td>不支持空 Tensor。<br>推荐 shape：<ul><li>TND: [T1, N1] 或 [T1, N1, 1]</li><li>BNSD/BSND: [B, N1, S1] 或 [B, N1, S1, 1]</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>2-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>rsvdBlockIdx</td>
    <td>输入</td>
    <td>KV2Q 稀疏索引表。</td>
    <td>不支持空 Tensor。有效位置填 Q token 下标，无效填 -1。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[B, N2, J, maxS1]</td>
    <td>×</td>
    </tr>
    <tr>
    <td>rsvdBlockCount</td>
    <td>输入</td>
    <td>每个 KV 块选中的 Q 数量。</td>
    <td>不支持空 Tensor。须与 rsvdBlockIdx 的 B/N2/J 一致。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[B, N2, J]</td>
    <td>×</td>
    </tr>
    <tr>
    <td>metadata</td>
    <td>输入</td>
    <td>分核与任务列表元数据。</td>
    <td>不支持空 Tensor。须由 aclnnGenericSparseAttentionGradMetadata 生成。<br>长度 ≥ <code>80 + B × N1 × J × 4</code>（int64 元素个数）。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attenMaskOptional</td>
    <td>输入</td>
    <td>显式注意力 mask。</td>
    <td>可选。当前版本建议传 nullptr，由 maskType 控制内置 mask。</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>-</td>
    <td>×</td>
    </tr>
    <tr>
    <td>cuSeqLengthsOptional</td>
    <td>输入</td>
    <td>Query 累积序列长度（packed）。</td>
    <td>qInputLayout 为 "TND" 时必选。shape [B+1]，dtype INT64。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>cuSeqLengthsKvOptional</td>
    <td>输入</td>
    <td>Key/Value 累积序列长度（packed）。</td>
    <td>kvInputLayout 为 "TND" 时必选。shape [B+1]，dtype INT64。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>sequsedQOptional</td>
    <td>输入</td>
    <td>每个 batch 实际 Q 长度。</td>
    <td>可选。TND 场景可与 cuSeqLengths 配合使用。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[B]</td>
    <td>×</td>
    </tr>
    <tr>
    <td>sequsedKvOptional</td>
    <td>输入</td>
    <td>每个 batch 实际 KV 长度。</td>
    <td>可选。TND 场景可与 cuSeqLengthsKv 配合使用。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[B]</td>
    <td>×</td>
    </tr>
    <tr>
    <td>blockShape</td>
    <td>输入</td>
    <td>稀疏块尺寸 [blockShapeX, blockShapeY]。</td>
    <td>当前要求 blockShapeX=1；blockShapeY≥128 且为 64 的倍数，且 blockShapeY≤128（当前 cube baseN 预算）。默认 [1, 128]。</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>isPackedGqa</td>
    <td>输入</td>
    <td>是否按 packed GQA 解释 head。</td>
    <td>当前仅支持传 1。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>qInputLayout</td>
    <td>输入</td>
    <td>query/dout/out 布局。</td>
    <td>支持 "TND"/"BNSD"/"BSND"，须与 kvInputLayout 一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>kvInputLayout</td>
    <td>输入</td>
    <td>key/value 布局。</td>
    <td>支持 "TND"/"BNSD"/"BSND"，须与 qInputLayout 一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue</td>
    <td>输入</td>
    <td>注意力缩放系数。</td>
    <td>一般取 D^-0.5。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskType</td>
    <td>输入</td>
    <td>内置 mask 类型。</td>
    <td>当前仅支持 1：RIGHT_DOWN_CAUSAL（右下角原点因果 mask）。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>softmaxPrecision</td>
    <td>输入</td>
    <td>Softmax 精度模式。</td>
    <td>当前传 0。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>windowSizeLeft</td>
    <td>输入</td>
    <td>滑窗左边界。</td>
    <td>当前须传 -1（不支持滑窗）。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>windowSizeRight</td>
    <td>输入</td>
    <td>滑窗右边界。</td>
    <td>当前须传 -1（不支持滑窗）。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dQuery</td>
    <td>输出</td>
    <td>query 梯度。</td>
    <td>不支持空 Tensor。shape/dtype 与 query 一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dKey</td>
    <td>输出</td>
    <td>key 梯度。</td>
    <td>不支持空 Tensor。shape/dtype 与 key 一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dValue</td>
    <td>输出</td>
    <td>value 梯度。</td>
    <td>不支持空 Tensor。shape/dtype 与 value 一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
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

<table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
<col style="width: 268px">
<col style="width: 140px">
<col style="width: 762px">
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
    <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
    <td rowspan="3">161001</td>
    <td>query/key/value/dout/out/lse/rsvdBlockIdx/rsvdBlockCount/metadata/dQuery/dKey/dValue 或 layout 字符串为空指针。</td>
  </tr>
  <tr>
    <td>qInputLayout 为 "TND" 时未提供 cuSeqLengthsOptional。</td>
  </tr>
  <tr>
    <td>kvInputLayout 为 "TND" 时未提供 cuSeqLengthsKvOptional。</td>
  </tr>
  <tr>
    <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="4">161002</td>
    <td>输入 dtype 不在支持范围，或 query/key/value/dout/out dtype 不一致；lse 非 FLOAT；稀疏表非 INT32；metadata 非 INT64。</td>
  </tr>
  <tr>
    <td>qInputLayout 与 kvInputLayout 不一致，或取值非法。</td>
  </tr>
  <tr>
    <td>isPackedGqa != 1，或 windowSizeLeft/Right != -1。</td>
  </tr>
  <tr>
    <td>blockShape 不满足 [1, By]，By≥128 且 64 对齐。</td>
  </tr>
</tbody></table>

## aclnnGenericSparseAttentionGrad

### 参数说明

<table style="undefined;table-layout: fixed; width: 953px"><colgroup>
<col style="width: 173px">
<col style="width: 112px">
<col style="width: 668px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
  </tr></thead>
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

* 该接口与 PyTorch 配合使用时，需保证 CANN 与 PyTorch 相关包版本匹配。
* 须先调用 [`aclnnGenericSparseAttentionGradMetadata`](../generic_sparse_attention_grad_metadata/docs/aclnnGenericSparseAttentionGradMetadata.md) 生成 `metadata`，再调用本接口。
* HeadDim 固定为 **128**。
* `N1` / `N2` 须落在 **[1, 128]**，且 `N1 % N2 == 0`，支持 MHA / GQA / MQA（packed GQA，`isPackedGqa=1`）。
* `blockShape` 当前仅支持 `[1, 128]`（By 须 ≥128、64 对齐且 ≤128）。
* `windowSizeLeft` / `windowSizeRight` 当前仅支持 `-1`。
* 不支持确定性计算（Deterministic）。
* `maskType` 当前仅支持 **1**（RIGHT_DOWN_CAUSAL，有效条件 $k \le q + (S2 - S1)$）。
* TND 布局下 `cuSeqLengths` / `cuSeqLengthsKv` 必选；BNSD/BSND 下可省略。
* Softmax LSE 的 head/seq 轴语义须与 query 布局一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。完整可运行样例见 `examples/test_aclnn_generic_sparse_attention_grad.cpp`。

```Cpp
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_sparse_attention_grad.h"
#include "aclnnop/aclnn_generic_sparse_attention_grad_metadata.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                            \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // BNSD smoke case
    const int64_t B = 1;
    const int64_t N1 = 1;
    const int64_t N2 = 1;
    const int64_t S1 = 128;
    const int64_t S2 = 128;
    const int64_t D = 128;
    const int64_t blockX = 1;
    const int64_t blockY = 128;
    const int64_t J = (S2 + blockY - 1) / blockY;
    const int64_t maskType = 1;
    const double scaleValue = 1.0 / std::sqrt(static_cast<double>(D));
    // metadata size = 80 + B * N1 * J * 4
    const int64_t metaSize = 80 + B * N1 * J * 4;

    std::vector<int64_t> qShape = {B, N1, S1, D};
    std::vector<int64_t> kvShape = {B, N2, S2, D};
    std::vector<int64_t> lseShape = {B, N1, S1};
    std::vector<int64_t> idxShape = {B, N2, J, S1};
    std::vector<int64_t> cntShape = {B, N2, J};
    std::vector<int64_t> metaShape = {metaSize};

    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0x2E66); // ~0.1 fp16
    std::vector<uint16_t> kHost(GetShapeSize(kvShape), 0x2E66);
    std::vector<uint16_t> vHost(GetShapeSize(kvShape), 0x2E66);
    std::vector<uint16_t> doutHost(GetShapeSize(qShape), 0x211E); // ~0.01 fp16
    std::vector<uint16_t> outHost(GetShapeSize(qShape), 0x2E66);
    std::vector<float> lseHost(GetShapeSize(lseShape), 5.0f);
    std::vector<int32_t> idxHost(GetShapeSize(idxShape), -1);
    std::vector<int32_t> cntHost(GetShapeSize(cntShape), 0);
    std::vector<int64_t> metaHost(metaSize, 0);
    std::vector<uint16_t> dqHost(GetShapeSize(qShape), 0);
    std::vector<uint16_t> dkHost(GetShapeSize(kvShape), 0);
    std::vector<uint16_t> dvHost(GetShapeSize(kvShape), 0);

    // 单个 KV 块选中全部 Q token
    for (int64_t q = 0; q < S1; ++q) {
        idxHost[q] = static_cast<int32_t>(q);
    }
    cntHost[0] = static_cast<int32_t>(S1);

    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
    void *doutAddr = nullptr, *outAddr = nullptr, *lseAddr = nullptr;
    void *idxAddr = nullptr, *cntAddr = nullptr, *metaAddr = nullptr;
    void *dqAddr = nullptr, *dkAddr = nullptr, *dvAddr = nullptr;

    aclTensor *q = nullptr, *k = nullptr, *v = nullptr;
    aclTensor *dout = nullptr, *out = nullptr, *lse = nullptr;
    aclTensor *idx = nullptr, *cnt = nullptr, *metadata = nullptr;
    aclTensor *dq = nullptr, *dk = nullptr, *dv = nullptr;

    ret = CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHost, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHost, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doutHost, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &dout);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, qShape, &outAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lseHost, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cntHost, cntShape, &cntAddr, aclDataType::ACL_INT32, &cnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metaHost, metaShape, &metaAddr, aclDataType::ACL_INT64, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dqHost, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dkHost, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dvHost, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t blockShapeData[] = {blockX, blockY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    char qLayout[] = "BNSD";
    char kvLayout[] = "BNSD";

    // 1) Metadata
    uint64_t metaWsSize = 0;
    aclOpExecutor *metaExecutor = nullptr;
    ret = aclnnGenericSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, 1, qLayout, kvLayout, maskType, 0,
        -1, -1, metadata, &metaWsSize, &metaExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericSparseAttentionGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void *metaWs = nullptr;
    if (metaWsSize > 0) {
        ret = aclrtMalloc(&metaWs, metaWsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGenericSparseAttentionGradMetadata(metaWs, metaWsSize, metaExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Metadata failed. ERROR: %d\n", ret); return ret);

    // 2) Grad
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGenericSparseAttentionGradGetWorkspaceSize(
        q, k, v, dout, out, lse, idx, cnt, metadata, nullptr, nullptr, nullptr, nullptr, nullptr, blockShape, 1,
        qLayout, kvLayout, scaleValue, maskType, 0, -1, -1, dq, dk, dv, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericSparseAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGenericSparseAttentionGrad(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericSparseAttentionGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("GenericSparseAttentionGrad finished successfully.\n");

    aclDestroyIntArray(blockShape);
    aclDestroyTensor(q); aclDestroyTensor(k); aclDestroyTensor(v);
    aclDestroyTensor(dout); aclDestroyTensor(out); aclDestroyTensor(lse);
    aclDestroyTensor(idx); aclDestroyTensor(cnt); aclDestroyTensor(metadata);
    aclDestroyTensor(dq); aclDestroyTensor(dk); aclDestroyTensor(dv);
    aclrtFree(qAddr); aclrtFree(kAddr); aclrtFree(vAddr);
    aclrtFree(doutAddr); aclrtFree(outAddr); aclrtFree(lseAddr);
    aclrtFree(idxAddr); aclrtFree(cntAddr); aclrtFree(metaAddr);
    aclrtFree(dqAddr); aclrtFree(dkAddr); aclrtFree(dvAddr);
    if (metaWsSize > 0) {
        aclrtFree(metaWs);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
