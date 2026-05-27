# aclnnSparseLightningIndexerKLLossGrad

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

`aclnnSparseLightningIndexerKLLossGrad` 是 Sparse Lightning Indexer KL Loss 的反向计算接口。
该接口接收 Lightning Indexer 分支的 `q/k/w/sparse_indices`，以及由主 Attention 分支预先计算得到的
`attn_softmax_l1_norm`，计算并输出：

- `dq`：`q` 的梯度。
- `dk`：`k` 的梯度。
- `dw`：`w` 的梯度。
- `softmax_out`：Indexer 分支 logits 经过 ReLU、加权、按 head group reduce 后的 softmax 结果。

与旧版 `SparseLightningIndexerGradKLLoss` 相比，本接口不再在 kernel 内部重算主 Attention 的
`Q @ K -> softmax -> reduce/l1 norm`，也不再输出 loss。主 Attention 的目标分布由
`attn_softmax_l1_norm` 输入提供，loss 可由后续算子继续计算。

- 计算公式：
   用于取 Top-k 的 value 的 Indexer logits 可表示为：

   $$
   S_{t,:}=q_{t,:}@K_{\operatorname{topk}(t),:}^{T}
   $$

   $$
   I_{t,:}=W_{t,:}@\operatorname{ReLU}(S_{t,:})
   $$

   其中，$q$ 和 $K$ 分别对应本接口的 `q`、`k`，$W$ 对应本接口的 `w`，
   $\operatorname{topk}(t)$ 由 `sparse_indices` 给出。

   Indexer 分支的 softmax 输出为：

   $$
   y_{t,:}=\operatorname{Softmax}(I_{t,:})
   $$

   本接口将 $y$ 写出到 `softmax_out`。目标分布 $p$ 由 `attn_softmax_l1_norm` 输入提供，
   它等价于旧版 kernel 内部由 main attention score 经 head 求和和 L1 归一化得到的结果。若后续继续计算
   KL Loss，其形式与旧版保持一致：

   $$
   L(I)=\sum_tD_{KL}(p_{t,:}||\operatorname{Softmax}(I_{t,:}))
   $$

   $$
   D_{KL}(a||b)=\sum_ia_i\log{\left(\frac{a_i}{b_i}\right)}
   $$

   对应的 softmax-KL 反向梯度为：

   $$
   dI_{t,:}=y_{t,:}-p_{t,:}
   $$

   利用链式法则可以计算 `w`、`q` 和 `k` 的梯度：

   $$
   dW_{t,:}=dI_{t,:}@\operatorname{ReLU}(S_{t,:})^{T}
   $$

   $$
   dq_{t,:}=dS_{t,:}@K_{\operatorname{topk}(t),:}
   $$

   $$
   dK_{\operatorname{topk}(t),:}=dS_{t,:}^{T}@q_{t,:}
   $$

   `dK` 写回时会按照 `sparse_indices` 指向的 key 位置做 scatter-add，无效 top-k 位置不参与计算。

## 函数原型

每个 aclnn 算子分为两段式接口。必须先调用
`aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize` 获取 workspace 大小和 executor，再调用
`aclnnSparseLightningIndexerKLLossGrad` 执行计算。

```c++
aclnnStatus aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *sparseIndices,
    const aclTensor *attnSoftmaxL1Norm,
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *metadataOptional,
    char *layoutQ,
    char *layoutK,
    int64_t maskMode,
    int64_t cmpRatio,
    int64_t deterministic,
    const aclTensor *dq,
    const aclTensor *dk,
    const aclTensor *dw,
    const aclTensor *softmaxOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```c++
aclnnStatus aclnnSparseLightningIndexerKLLossGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | Shape | 是否支持非连续 |
| --- | --- | --- | --- | --- | --- | :---: |
| q | 输入 | Lightning Indexer 分支的 query。 | FLOAT16 / BFLOAT16 | ND | BSND: `(B,S1,N1,D)`；TND: `(T1,N1,D)` | × |
| k | 输入 | Lightning Indexer 分支的 key。当前仅支持 `N2=1`。 | FLOAT16 / BFLOAT16 | ND | BSND: `(B,S2,N2,D)`；TND: `(T2,N2,D)` | × |
| w | 输入 | Indexer logits 的 head 权重。 | FLOAT32 | ND | BSND: `(B,S1,N1)`；TND: `(T1,N1)` | × |
| sparseIndices | 输入 | 每个 query 对应的 top-k key 下标。有效部分填入 key 下标，无效部分填 `-1`。 | INT32 | ND | BSND: `(B,S1,N2,K)`；TND: `(T1,N2,K)` | × |
| attnSoftmaxL1Norm | 输入 | 主 Attention 分支预先计算出的目标分布，用于替代旧 kernel 内部的主 Attention 重算结果。 | FLOAT32 | ND | BSND: `(B,S1,N2,K)`；TND: `(T1,N2,K)` | × |
| cuSeqLensQOptional | 可选输入 | TND layout 下 q 的累积序列长度。 | INT32 | ND | `(B+1,)` | × |
| cuSeqLensKOptional | 可选输入 | TND layout 下 k 的累积序列长度。 | INT32 | ND | `(B+1,)` | × |
| seqUsedQOptional | 可选输入 | 预留字段，表示每个 batch 实际使用的 q 长度。当前 kernel 路径暂不使用。 | INT32 | ND | `(B,)` | × |
| seqUsedKOptional | 可选输入 | 预留字段，表示每个 batch 实际使用的 k 长度。当前 kernel 路径暂不使用。 | INT32 | ND | `(B,)` | × |
| cmpResidualKOptional | 可选输入 | `maskMode=3` 且 `cmpRatio!=1` 时的预留字段。 | INT32 | ND | `(B,)` | × |
| metadataOptional | 可选输入 | 由 `aclnnSparseLightningIndexerKLLossGradMetadata` 生成的分核信息。传入后 kernel 将使用 metadata 中的分核切分。 | INT32 | ND | `(x,)` | × |
| layoutQ | 输入属性 | q/w/sparseIndices/attnSoftmaxL1Norm/dq/dw/softmaxOut 的 layout。 | STRING | - | `BSND` / `TND` | - |
| layoutK | 输入属性 | k/dk 的 layout。 | STRING | - | `BSND` / `TND` | - |
| maskMode | 输入属性 | mask 模式。`0` 表示无 mask，`3` 表示右下角 causal mask。 | INT64 | - | `0` / `3` | - |
| cmpRatio | 输入属性 | 压缩比。 | INT64 | - | `[1,128]` | - |
| deterministic | 输入属性 | 确定性计算开关。`0` 表示跟随 torch 全局配置；`1` 表示不启用确定性计算；`2` 表示启用确定性计算。 | INT64 | - | `0` / `1` / `2` | - |
| dq | 输出 | `q` 的梯度。 | FLOAT16 / BFLOAT16 | ND | 与 `q` 相同 | - |
| dk | 输出 | `k` 的梯度。 | FLOAT16 / BFLOAT16 | ND | 与 `k` 相同 | - |
| dw | 输出 | `w` 的梯度。 | FLOAT32 | ND | 与 `w` 相同 | - |
| softmaxOut | 输出 | Indexer 分支 softmax 输出。 | FLOAT32 | ND | 与 `attnSoftmaxL1Norm` 相同 | - |
| workspaceSize | 输出 | 返回用户需要在 device 侧申请的 workspace 大小。 | UINT64 | - | 标量 | - |
| executor | 输出 | 返回 op 执行器，包含算子计算流程。 | aclOpExecutor* | - | - | - |

## 约束与限制

- `q` 和 `k` 数据类型必须一致，支持 `FLOAT16` 和 `BFLOAT16`。
- `dq` 数据类型与 `q` 一致，`dk` 数据类型与 `k` 一致。
- `w/dw/attnSoftmaxL1Norm/softmaxOut` 数据类型为 `FLOAT32`。
- `sparseIndices/cuSeqLensQOptional/cuSeqLensKOptional/seqUsedQOptional/seqUsedKOptional/cmpResidualKOptional/metadataOptional`
  数据类型为 `INT32`。
- `layoutQ` 和 `layoutK` 当前支持 `BSND` 和 `TND`。
- 当 `layoutQ=TND` 时，必须传入 `cuSeqLensQOptional` 和 `cuSeqLensKOptional`。
- 当前 kernel 路径仅支持 `N2=1`。
- `N1` 必须能被 `N2` 整除，且当前实现支持的 `N1/N2` 取值为 `{8,16,32,64}`。
- `D` 需要满足 `q` 和 `k` 最后一维一致。
- `K` 当前要求小于等于 `8192`，并且为 `512` 或 `1024` 的整数倍。
- `sparseIndices` 中有效位置必须位于当前 batch 的 key 序列范围内；无效位置使用 `-1` 填充。
- `attnSoftmaxL1Norm` 的无效 top-k 位置建议置零，并与 `sparseIndices` 的有效位置保持一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_lightning_indexer_kl_loss_grad.h"
#include "aclnnop/aclnn_sparse_lightning_indexer_kl_loss_grad_metadata.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
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
    return ACL_SUCCESS;
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

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

void PrintFloatResult(const std::vector<int64_t> &shape, void *deviceAddr)
{
    auto elemCount = GetShapeSize(shape);
    std::vector<float> resultData(elemCount, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), deviceAddr,
                           resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < std::min<int64_t>(elemCount, 5); ++i) {
        LOG_PRINT("softmaxOut[%ld] is: %f\n", i, resultData[i]);
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t batchSize = 1;
    const int64_t t1 = 16;
    const int64_t t2 = 4;
    const int64_t numHeadsQ = 8;
    const int64_t numHeadsK = 1;
    const int64_t headDim = 128;
    const int64_t topk = 512;
    const int64_t maskMode = 3;
    const int64_t cmpRatio = 4;
    const int64_t deterministic = 0;

    std::vector<int64_t> qShape = {t1, numHeadsQ, headDim};
    std::vector<int64_t> kShape = {t2, numHeadsK, headDim};
    std::vector<int64_t> wShape = {t1, numHeadsQ};
    std::vector<int64_t> sparseShape = {t1, numHeadsK, topk};
    std::vector<int64_t> softmaxShape = {t1, numHeadsK, topk};
    std::vector<int64_t> cuSeqLensShape = {batchSize + 1};
    std::vector<int64_t> metadataShape = {64};

    std::vector<uint16_t> qHostData(GetShapeSize(qShape), 0x3C00); // fp16 1.0
    std::vector<uint16_t> kHostData(GetShapeSize(kShape), 0x3C00);
    std::vector<float> wHostData(GetShapeSize(wShape), 1.0f);
    std::vector<int32_t> sparseHostData(GetShapeSize(sparseShape), -1);
    std::vector<float> attnSoftmaxL1NormHostData(GetShapeSize(softmaxShape), 0.0f);
    for (int64_t t = 0; t < t1; ++t) {
        auto validK = std::min<int64_t>(std::max<int64_t>(1, (t + 1) / cmpRatio), t2);
        for (int64_t kIdx = 0; kIdx < validK; ++kIdx) {
            sparseHostData[t * numHeadsK * topk + kIdx] = static_cast<int32_t>(kIdx);
            attnSoftmaxL1NormHostData[t * numHeadsK * topk + kIdx] = 1.0f / static_cast<float>(validK);
        }
    }
    std::vector<int32_t> cuSeqLensQHostData = {0, static_cast<int32_t>(t1)};
    std::vector<int32_t> cuSeqLensKHostData = {0, static_cast<int32_t>(t2)};
    std::vector<int32_t> metadataHostData(GetShapeSize(metadataShape), 0);
    std::vector<uint16_t> dqHostData(GetShapeSize(qShape), 0);
    std::vector<uint16_t> dkHostData(GetShapeSize(kShape), 0);
    std::vector<float> dwHostData(GetShapeSize(wShape), 0.0f);
    std::vector<float> softmaxOutHostData(GetShapeSize(softmaxShape), 0.0f);

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *wDeviceAddr = nullptr;
    void *sparseDeviceAddr = nullptr;
    void *attnSoftmaxL1NormDeviceAddr = nullptr;
    void *cuSeqLensQDeviceAddr = nullptr;
    void *cuSeqLensKDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    void *dqDeviceAddr = nullptr;
    void *dkDeviceAddr = nullptr;
    void *dwDeviceAddr = nullptr;
    void *softmaxOutDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *w = nullptr;
    aclTensor *sparseIndices = nullptr;
    aclTensor *attnSoftmaxL1Norm = nullptr;
    aclTensor *cuSeqLensQ = nullptr;
    aclTensor *cuSeqLensK = nullptr;
    aclTensor *metadata = nullptr;
    aclTensor *dq = nullptr;
    aclTensor *dk = nullptr;
    aclTensor *dw = nullptr;
    aclTensor *softmaxOut = nullptr;

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT, &w);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sparseHostData, sparseShape, &sparseDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnSoftmaxL1NormHostData, softmaxShape, &attnSoftmaxL1NormDeviceAddr,
                          aclDataType::ACL_FLOAT, &attnSoftmaxL1Norm);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32,
                          &cuSeqLensQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqLensKHostData, cuSeqLensShape, &cuSeqLensKDeviceAddr, aclDataType::ACL_INT32,
                          &cuSeqLensK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dkHostData, kShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dwHostData, wShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxOutHostData, softmaxShape, &softmaxOutDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;
    ret = aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
        cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, batchSize, t1, t2, numHeadsQ, numHeadsK, headDim, topk,
        const_cast<char *>("TND"), const_cast<char *>("TND"), maskMode, cmpRatio, deterministic, metadata,
        &metadataWorkspaceSize, &metadataExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *metadataWorkspaceAddr = nullptr;
    if (metadataWorkspaceSize > 0) {
        ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSparseLightningIndexerKLLossGradMetadata(metadataWorkspaceAddr, metadataWorkspaceSize,
                                                       metadataExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSparseLightningIndexerKLLossGradMetadata failed. ERROR: %d\n", ret); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize(
        q, k, w, sparseIndices, attnSoftmaxL1Norm, cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, metadata,
        const_cast<char *>("TND"), const_cast<char *>("TND"), maskMode, cmpRatio, deterministic, dq, dk, dw,
        softmaxOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSparseLightningIndexerKLLossGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSparseLightningIndexerKLLossGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    PrintFloatResult(softmaxShape, softmaxOutDeviceAddr);

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(w);
    aclDestroyTensor(sparseIndices);
    aclDestroyTensor(attnSoftmaxL1Norm);
    aclDestroyTensor(cuSeqLensQ);
    aclDestroyTensor(cuSeqLensK);
    aclDestroyTensor(metadata);
    aclDestroyTensor(dq);
    aclDestroyTensor(dk);
    aclDestroyTensor(dw);
    aclDestroyTensor(softmaxOut);

    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(wDeviceAddr);
    aclrtFree(sparseDeviceAddr);
    aclrtFree(attnSoftmaxL1NormDeviceAddr);
    aclrtFree(cuSeqLensQDeviceAddr);
    aclrtFree(cuSeqLensKDeviceAddr);
    aclrtFree(metadataDeviceAddr);
    aclrtFree(dqDeviceAddr);
    aclrtFree(dkDeviceAddr);
    aclrtFree(dwDeviceAddr);
    aclrtFree(softmaxOutDeviceAddr);
    if (metadataWorkspaceSize > 0) {
        aclrtFree(metadataWorkspaceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
