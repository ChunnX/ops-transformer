# aclnnGenericBlockSparseAttentionGrad

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品：不支持
- Atlas 训练系列产品：不支持

## 功能说明

- **算子功能**：GenericBlockSparseAttentionGrad 是通用块稀疏注意力的反向计算算子。依据 `rsvdBlockIdx` / `rsvdBlockCount`（稀疏块索引表）定义的索引，仅在被选中的 KV 块上计算和传播梯度，在反向传播时精确路由梯度，支持动态、可变长的分块稀疏模式。调用前须先通过 `aclnnGenericBlockSparseAttentionGradMetadata` 生成分核 `metadata`。
- **计算公式**：
  1. 计算 Softmax 后的注意力权重矩阵
  $$
  P = SimpleSoftmax(Mask(Q @ selectedK^{T} \cdot scale), lse)
  $$
  1. 计算 $dP$（Softmax 梯度的直接部分）
  $$
  dP = dO @ selectedV^{T}
  $$
  1. 计算 $dS$（注意力分数梯度）
  $$
  dS = P \odot (dP - SoftmaxGrad(dO, O))
  $$
  1. 计算 $dQ$、$dK$、$dV$
  $$
  dQ = dS @ selectedK \cdot scale
  $$
  $$
  dK = dS^{T} @ Q \cdot scale
  $$
  $$
  dV = P^{T} @ dO
  $$

  | 符号                   | 含义                       |
  | -------------------- | ------------------------ |
  | $Q$                  | 查询矩阵                     |
  | $K, V$               | 键、值矩阵                    |
  | $P$                  | Softmax 后的注意力权重矩阵        |
  | $S$                  | 注意力分数（缩放掩码后、Softmax 前）   |
  | $O$                  | 注意力输出矩阵（$O = P \cdot V$） |
  | $\text{lse}$         | log-sum-exp 值，前向保存供反向使用  |
  | $dO$                 | 输出梯度（从上层传入）              |
  | $\text{scale}$       | 缩放因子，通常为 $1/\sqrt{d_k}$  |
  | $dP, dS, dQ, dK, dV$ | 各中间变量和参数的梯度              |


输入 `query`/`key`/`value`/`dout`/`out` 的排布由 `qInputLayout`、`kvInputLayout` 指定。布局字母含义：

- B：Batch
- T：B 与 S 合轴后的 Total tokens
- S：Seq-Length
- N：Head-Num
- D：Head-Dim

当前支持的布局：

- `qInputLayout`：`"TND"`、`"BNSD"`、`"BSND"`
- `kvInputLayout`：`"TND"`、`"BNSD"`、`"BSND"`（须与 `qInputLayout` 一致）



## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGenericBlockSparseAttentionGradGetWorkspaceSize”获取 workspace 与执行器，再调用“aclnnGenericBlockSparseAttentionGrad”执行计算。

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
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
aclnnStatus aclnnGenericBlockSparseAttentionGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
```



## aclnnGenericBlockSparseAttentionGradGetWorkspaceSize



### 参数说明


| 参数名                    | 输入/输出   | 描述                                                    | 使用说明                                                                                                                                                           | 数据类型             | 数据格式 | 维度(shape)                                                                                                                                 | 非连续Tensor |
| ---------------------- | ------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| query                  | 输入，必选   | Device侧的aclTensor，attention结构的输入Q。                    | query、key、value、dout、out、lse 的 Shape 维度语义保持一致。                                                                                                                 | BFLOAT16、FLOAT16 | ND   | (B,S1,N1,D)、(T1,N1,D)、(B,N1,S1,D) B：支持泛化；S1：支持泛化；N1：支持1~128，且能被N2整除；D：128；T1：B × S1                                                       | √         |
| key                    | 输入，必选   | Device侧的aclTensor，attention结构的输入K。                    | -                                                                                                                                                              | BFLOAT16、FLOAT16 | ND   | (B,S2,N2,D)、(T2,N2,D)、(B,N2,S2,D) B：支持泛化；S2：支持泛化；N2：支持1~128；D：128；T2：B × S2                                                               | √         |
| value                  | 输入，必选   | Device侧的aclTensor，attention结构的输入V。                    | -                                                                                                                                                              | BFLOAT16、FLOAT16 | ND   | (B,S2,N2,D)、(T2,N2,D)、(B,N2,S2,D) Shape与key相同。                                                                                            | √         |
| dout                   | 输入，必选   | Device侧的aclTensor，注意力输出矩阵的梯度。                         | -                                                                                                                                                              | BFLOAT16、FLOAT16 | ND   | (B,S1,N1,D)、(T1,N1,D)、(B,N1,S1,D) B、S1、N1与query保持一致；D：128；T1：B × S1                                                                       | √         |
| out                    | 输入，必选   | Device侧的aclTensor，注意力输出矩阵。                            | -                                                                                                                                                              | BFLOAT16、FLOAT16 | ND   | (B,S1,N1,D)、(T1,N1,D)、(B,N1,S1,D) Shape与dout保持一致。                                                                                         | √         |
| lse                    | 输入，必选   | Device侧的aclTensor，注意力正向计算的输出lse，layout与正向输出保持一致。      | -                                                                                                                                                              | FLOAT32          | ND   | query为"TND"：[totalQTokens, headNum, 1]。 query为"BNSD"：[batch, headNum, maxQSeqLength, 1]。 query为"BSND"：[batch, maxQSeqLength, headNum, 1]。 | √         |
| rsvdBlockIdx           | 输入，必选   | Device侧的aclTensor，稀疏块索引数组，指定每个KV块选择的Q块/token索引。       | 存储每个KV块选择的Q索引。 TND：[B, N2, ceilDiv(maxS2, blockShapeY), maxS1]。 BNSD/BSND：[B, N2, ceilDiv(S2, blockShapeY), S1]。 同group每个KVHead对应的Q稀疏pattern一致（isPackedGqa=1）。 | INT32            | ND   | 4                                                                                                                                         | √         |
| rsvdBlockCount         | 输入，必选   | Device侧的aclTensor，指定每个KV块实际选择的Q数量。                    | 存储每个KV块实际选择的Q数量。 TND：[B, N2, ceilDiv(maxS2, blockShapeY)]。 BNSD/BSND：[B, N2, ceilDiv(S2, blockShapeY)]。                                                        | INT32            | ND   | (B, N2, ceilDiv(maxS2, blockShapeY))或(B, N2, ceilDiv(S2, blockShapeY))                                                                    | √         |
| metadata               | 输入，必选   | Device侧的aclTensor，表示tiling下沉的AICPU算子输出结果。             | 须由 aclnnGenericBlockSparseAttentionGradMetadata 生成。长度 ≥ `80 + B × N1 × J × 4`（int64 元素个数）。                                                                          | INT64            | ND   | (x)                                                                                                                                       | √         |
| attenMaskOptional      | 输入，可选   | Device侧的aclTensor，公式中的atten_mask，与稀疏pattern叠加产生作用。    | 当前暂不支持，应传nullptr。                                                                                                                                              | BOOL             | ND   | -                                                                                                                                         | √         |
| cuSeqLengthsOptional   | 输入，可选   | Device侧的aclTensor，描述每个Batch对应的query序列长度，以前缀和形式存储。     | 用于变长序列场景。 qInputLayout为"TND"时必须配置；为"BNSD"或"BSND"时传nullptr。                                                                                                     | INT64            | ND   | 1                                                                                                                                         | -         |
| cuSeqLengthsKvOptional | 输入，可选   | Device侧的aclTensor，描述每个Batch对应的key/value序列长度，以前缀和形式存储。 | 用于变长序列场景。 kvInputLayout为"TND"时必须配置；为"BNSD"或"BSND"时传nullptr。                                                                                                    | INT64            | ND   | 1                                                                                                                                         | -         |
| sequsedQOptional       | 输入，可选   | Device侧的aclTensor，各batch中query的实际序列长度。                | -                                                                                                                                                              | INT32            | ND   | 1                                                                                                                                         | √         |
| sequsedKvOptional      | 输入，可选   | Device侧的aclTensor，各batch中kv的实际序列长度。                   | -                                                                                                                                                              | INT32            | ND   | 1                                                                                                                                         | √         |
| blockShape             | 输入，Attr | 代表稀疏块形状数组。                                            | 含两个元素[blockShapeX, blockShapeY]。 blockShapeX当前仅支持1。 blockShapeY须≥128且按64对齐；当前实现仅支持128。                                                                         | INT64            | -    | -                                                                                                                                         | -         |
| isPackedGqa            | 输入，Attr | 代表进行块状稀疏时，同一个group内的qHead是否共享同样的稀疏pattern（当前仅支持1）。    | 0表示不共享；1表示共享。不同batch之间不共享。                                                                                                                                     | INT64            | N/A  | -                                                                                                                                         | -         |
| qInputLayout           | 输入，Attr | Host侧的string，代表输入query的数据排布格式。                        | 当前支持"TND"、"BNSD"、"BSND"。                                                                                                                                       | String           | -    | -                                                                                                                                         | -         |
| kvInputLayout          | 输入，Attr | Host侧的string，代表输入key、value的数据排布格式。                    | 当前支持"TND"、"BNSD"、"BSND"。                                                                                                                                       | String           | -    | -                                                                                                                                         | -         |
| scaleValue             | 输入，Attr | Host侧的double，公式中的scale，代表缩放系数。                        | 建议值：公式中d开根号的倒数。                                                                                                                                                | FLOAT            | N/A  | -                                                                                                                                         | -         |
| maskType               | 输入，Attr | Host侧的int64_t，表示attention计算中的掩码类型。当前仅支持1。             | 具体见“掩码说明”。                                                                                                                                                     | INT64            | -    | -                                                                                                                                         | -         |
| softmaxPrecision       | 输入，Attr | Softmax计算采取的精度级别。                                     | 仅支持0或1。 0：online softmax和rescale均使用fp32。 1：online softmax使用fp16/bf16，rescale使用fp32，可能发生数值溢出。当前实现传0。                                                            | INT64            | -    | -                                                                                                                                         | -         |
| windowSizeLeft         | 输入，Attr | Host侧的int64_t，滑窗attention场景下，滑窗需要向前包含多少个token。        | 不使能时必须为-1，需要与maskType、mask配合使用。                                                                                                                                | INT64            | -    | -                                                                                                                                         | -         |
| windowSizeRight        | 输入，Attr | Host侧的int64_t，滑窗attention场景下，滑窗需要向后包含多少个token。        | 不使能时必须为-1，需要与maskType、mask配合使用。                                                                                                                                | INT64            | -    | -                                                                                                                                         | -         |
| dQuery                 | 输出      | Device侧的aclTensor，表示query的梯度。                         | 与输入query的Shape维度保持一致。                                                                                                                                          | BFLOAT16、FLOAT16 | ND   | -                                                                                                                                         | √         |
| dKey                   | 输出      | Device侧的aclTensor，表示key的梯度。                           | 与输入key的Shape维度保持一致。                                                                                                                                            | BFLOAT16、FLOAT16 | ND   | -                                                                                                                                         | √         |
| dValue                 | 输出      | Device侧的aclTensor，表示value的梯度。                         | 与输入value的Shape维度保持一致。                                                                                                                                          | BFLOAT16、FLOAT16 | ND   | -                                                                                                                                         | √         |
| workspaceSize          | 输出      | 返回需要在Device侧申请的workspace大小。                           | -                                                                                                                                                              | -                | -    | -                                                                                                                                         | -         |
| executor               | 输出      | 返回op执行器，包含算子计算流程。                                     | -                                                                                                                                                              | -                | -    | -                                                                                                                                         | -         |




### 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

第一段接口完成入参校验，出现以下场景时报错：


| 返回码                                                | 错误码    | 描述                                                                                                     |
| -------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------ |
| ACLNN_ERR_PARAM_NULLPTR                            | 161001 | query/key/value/dout/out/lse/rsvdBlockIdx/rsvdBlockCount/metadata/dQuery/dKey/dValue 或 layout 字符串为空指针。 |
| qInputLayout 为 "TND" 时未提供 cuSeqLengthsOptional。    |        |                                                                                                        |
| kvInputLayout 为 "TND" 时未提供 cuSeqLengthsKvOptional。 |        |                                                                                                        |
| ACLNN_ERR_PARAM_INVALID                            | 161002 | 输入 dtype 不在支持范围，或 query/key/value/dout/out dtype 不一致；lse 非 FLOAT；稀疏表非 INT32；metadata 非 INT64。          |
| qInputLayout 与 kvInputLayout 不一致，或取值非法。            |        |                                                                                                        |
| isPackedGqa != 1，或 windowSizeLeft/Right != -1。     |        |                                                                                                        |
| blockShape 不满足 [1, By]，By≥128 且 64 对齐。             |        |                                                                                                        |




## aclnnGenericBlockSparseAttentionGrad



### 参数说明


| 参数名           | 输入/输出 | 描述                                                                              |
| ------------- | ----- | ------------------------------------------------------------------------------- |
| workspace     | 输入    | 在Device侧申请的workspace内存地址。                                                       |
| workspaceSize | 输入    | 在Device侧申请的workspace大小，由第一段接口aclnnGenericBlockSparseAttentionGradGetWorkspaceSize获取。 |
| executor      | 输入    | op执行器，包含了算子计算流程。                                                                |
| stream        | 输入    | 指定执行任务的Stream。                                                                  |




### 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 掩码说明


| maskType | 含义                             | attenMaskOptional | windowSizeLeft/Right |
| -------- | ------------------------------ | ----------------- | -------------------- |
| 1        | causal mask（RIGHT_DOWN_CAUSAL） | 当前应传nullptr       | -1/-1                |




## 约束说明

- 确定性计算：默认为非确定性实现，暂不支持确定性实现，确定性计算配置后不会生效。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 须先调用 `[aclnnGenericBlockSparseAttentionGradMetadata](../generic_block_sparse_attention_grad_metadata/docs/aclnnGenericBlockSparseAttentionGradMetadata.md)` 生成 `metadata`，再调用本接口。
- layoutQ（`qInputLayout`）当前仅支持"TND"、"BNSD"、"BSND"。
- layoutKv（`kvInputLayout`）当前仅支持"TND"、"BNSD"、"BSND"。
- query、key、value的InputLayout必须保持一致。
- 输入query、key、value的数据类型必须一致，支持FLOAT16、BFLOAT16。
- HeadDim固定为**128**；`N1`/`N2`须落在**[1, 128]**，且`N1 % N2 == 0`。
- `cuSeqLengthsOptional`在`qInputLayout`为"TND"时必选；`cuSeqLengthsKvOptional`在`kvInputLayout`为"TND"时必选。
- `rsvdBlockIdx`第4维maxS1应 ≥ `rsvdBlockCount` tensor中所有元素的最大值，不限制上限。
- `windowSizeLeft`和`windowSizeRight`不使能时必须为-1。
- `maskType`当前仅支持**1**。
- `isPackedGqa`当前仅支持**1**。
- Softmax LSE的head/seq轴语义须与query布局一致。



## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。完整可运行样例见 `examples/test_aclnn_generic_block_sparse_attention_grad.cpp`。

```Cpp
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad_metadata.h"

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
    ret = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, 1, qLayout, kvLayout, maskType, 0,
        -1, -1, metadata, &metaWsSize, &metaExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void *metaWs = nullptr;
    if (metaWsSize > 0) {
        ret = aclrtMalloc(&metaWs, metaWsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGenericBlockSparseAttentionGradMetadata(metaWs, metaWsSize, metaExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Metadata failed. ERROR: %d\n", ret); return ret);

    // 2) Grad
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
        q, k, v, dout, out, lse, idx, cnt, metadata, nullptr, nullptr, nullptr, nullptr, nullptr, blockShape, 1,
        qLayout, kvLayout, scaleValue, maskType, 0, -1, -1, dq, dk, dv, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGenericBlockSparseAttentionGrad(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("GenericBlockSparseAttentionGrad finished successfully.\n");

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
