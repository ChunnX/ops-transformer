# aclnnAttentionResidualsGrad

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：AttentionResidualsGrad是attention residual正向计算的反向算子，使用场景为attention residual分支的梯度回传。该接口根据前向保存的inv_rms、probs以及正向参数proj_weight、norm_weight，结合grad_output，计算prefix_sum、block_residual、proj_weight和norm_weight的梯度。
- V版本说明：当前无V版本差异。
- 芯片差异：<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>计算语义一致。

- 计算公式：

  正向计算可表示为：

  $$
  v_{b,i,h} =
  \begin{cases}
  block\_residual_{b,i,h}, & i < N \\
  prefix\_sum_{b,h}, & i = N
  \end{cases}
  $$

  $$
  k_{b,i,h} = v_{b,i,h} \cdot inv\_rms_{b,i}
  $$

  $$
  score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
  $$

  $$
  s_{b,i} = \sum_{h=0}^{H-1} k_{b,i,h} \cdot score\_weight_{h}
  $$

  $$
  probs_{b,i} = \frac{e^{s_{b,i}}}{\sum_{j=0}^{N} e^{s_{b,j}}}
  $$

  $$
  out_{b,h} = \sum_{i=0}^{N} probs_{b,i} \cdot v_{b,i,h}
  $$

  反向计算：

  $$
  g_{b,i} = \sum_{h=0}^{H-1} grad\_output_{b,h} \cdot v_{b,i,h}
  $$

  $$
  grad\_score_{b,i} = probs_{b,i} \cdot \left(g_{b,i} - \sum_{j=0}^{N} probs_{b,j} \cdot g_{b,j}\right)
  $$

  $$
  grad\_k_{b,i,h} = grad\_score_{b,i} \cdot score\_weight_{h}
  $$

  $$
  grad\_score\_weight_{h} = \sum_{b=0}^{B-1} \sum_{i=0}^{N} grad\_score_{b,i} \cdot k_{b,i,h}
  $$

  $$
  grad\_inv\_rms_{b,i} = \sum_{h=0}^{H-1} grad\_k_{b,i,h} \cdot v_{b,i,h}
  $$

  $$
  grad\_v_{b,i,h} = grad\_output_{b,h} \cdot probs_{b,i} + grad\_k_{b,i,h} \cdot inv\_rms_{b,i} - \frac{grad\_inv\_rms_{b,i} \cdot inv\_rms_{b,i}^{3}}{H} \cdot v_{b,i,h}
  $$

  其中：

  $$
  grad\_block\_residual_{b,i,h} = grad\_v_{b,i,h}, \quad i < N
  $$

  $$
  grad\_prefix\_sum_{b,h} = grad\_v_{b,N,h}
  $$

  $$
  grad\_norm\_weight_{h} = grad\_score\_weight_{h} \cdot proj\_weight_{0,h}
  $$

  $$
  grad\_proj\_weight_{0,h} = grad\_score\_weight_{h} \cdot norm\_weight_{h}
  $$

### 符号说明

| 符号 | 含义 |
|------|------|
| $B$ | batch size |
| $N$ | block_residual的block数量 |
| $H$ | hidden size |
| $v_{b,i,h}$ | block_residual与prefix_sum拼接后的value，维度为 $(B,N+1,H)$ |
| $inv\_rms_{b,i}$ | 前向保存的逐行归一化系数 |
| $score\_weight_{h}$ | norm_weight与proj_weight逐元素相乘的结果 |
| $probs_{b,i}$ | 前向softmax输出概率 |
| $grad\_output_{b,h}$ | 前向输出out的梯度 |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnAttentionResidualsGradGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnAttentionResidualsGrad”接口执行计算。

```cpp
aclnnStatus aclnnAttentionResidualsGradGetWorkspaceSize(
    const aclTensor *prefix_sum,
    const aclTensor *block_residual,
    const aclTensor *proj_weight,
    const aclTensor *norm_weight,
    const aclTensor *grad_output,
    const aclTensor *inv_rms,
    const aclTensor *probs,
    const aclTensor *grad_prefix_sum,
    const aclTensor *grad_block_residual,
    const aclTensor *grad_proj_weight,
    const aclTensor *grad_norm_weight,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnAttentionResidualsGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnAttentionResidualsGradGetWorkspaceSize

### 参数说明

<table style="table-layout: fixed; width: 1500px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 300px">
            <col style="width: 350px">
            <col style="width: 210px">
            <col style="width: 100px">
            <col style="width: 100px">
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
        </tr></thead>
        <tbody>
        <tr>
            <td>prefix_sum（aclTensor*）</td>
            <td>输入</td>
            <td>前向输入前缀和，对应公式中第N+1个value。</td>
            <td>不支持空Tensor；数据类型与其余输入保持一致。</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(B,H)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>block_residual（aclTensor*）</td>
            <td>输入</td>
            <td>前向输入分块残差，对应公式中前N个value。</td>
            <td>不支持空Tensor；数据类型与prefix_sum保持一致。</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(B,N,H)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>proj_weight（aclTensor*）</td>
            <td>输入</td>
            <td>前向线性投影权重，与norm_weight共同构成score_weight。</td>
            <td>不支持空Tensor；数据类型与prefix_sum保持一致。</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(1,H)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>norm_weight（aclTensor*）</td>
            <td>输入</td>
            <td>前向归一化权重，与proj_weight共同构成score_weight。</td>
            <td>不支持空Tensor；数据类型与prefix_sum保持一致。</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(H)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>grad_output（aclTensor*）</td>
            <td>输入</td>
            <td>前向输出out的梯度。</td>
            <td>不支持空Tensor；数据类型与prefix_sum保持一致。</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(B,H)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>inv_rms（aclTensor*）</td>
            <td>输入</td>
            <td>前向保存的逐行归一化系数。</td>
            <td>不支持空Tensor；仅支持FLOAT32。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N+1)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>probs（aclTensor*）</td>
            <td>输入</td>
            <td>前向softmax输出概率。</td>
            <td>不支持空Tensor；仅支持FLOAT32。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N+1)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>grad_prefix_sum（aclTensor*）</td>
            <td>输出</td>
            <td>prefix_sum的梯度。</td>
            <td>不支持空Tensor；数据类型和shape与prefix_sum保持一致。</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(B,H)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>grad_block_residual（aclTensor*）</td>
            <td>输出</td>
            <td>block_residual的梯度。</td>
            <td>不支持空Tensor；数据类型和shape与block_residual保持一致。</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(B,N,H)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>grad_proj_weight（aclTensor*）</td>
            <td>输出</td>
            <td>proj_weight的梯度。</td>
            <td>不支持空Tensor；数据类型和shape与proj_weight保持一致。</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(1,H)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>grad_norm_weight（aclTensor*）</td>
            <td>输出</td>
            <td>norm_weight的梯度。</td>
            <td>不支持空Tensor；数据类型和shape与norm_weight保持一致。</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(H)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>workspaceSize（uint64_t*）</td>
            <td>输出</td>
            <td>返回需要在Device侧申请的workspace大小。</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>executor（aclOpExecutor**）</td>
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

### 返回值

返回aclnnStatus状态码。

第一段接口完成入参校验，出现以下场景时报错：

<table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>prefix_sum、block_residual、proj_weight、norm_weight、grad_output、inv_rms、probs及输出张量存在空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入张量的数据类型、数据格式或shape不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>

## aclnnAttentionResidualsGrad

### 参数说明

<table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
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
        <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnAttentionResidualsGradGetWorkspaceSize获取。</td>
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

### 返回值

返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 确定性计算

- aclnnAttentionResidualsGrad默认确定性实现。

### 规格约束

- prefix_sum、block_residual、proj_weight、norm_weight、grad_output及其对应输出的数据类型保持一致，支持FLOAT16、BFLOAT16、FLOAT32；inv_rms和probs仅支持FLOAT32。
- prefix_sum、block_residual、proj_weight、norm_weight、grad_output、inv_rms和probs均只支持ND格式。
- shape需满足：prefix_sum为(B,H)，block_residual为(B,N,H)，proj_weight为(1,H)，norm_weight为(H)，grad_output为(B,H)，inv_rms为(B,N+1)，probs为(B,N+1)，其中B、N、H均为正整数，且各张量中的B、H以及inv_rms和probs的N+1保持一致。
- 输入张量支持非连续Tensor，接口内部统一转为Contiguous后计算。
- probs应为前向softmax输出；inv_rms应为前向逐行归一化系数。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_residuals_grad.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t B  = 2;
    const int64_t N  = 4;
    const int64_t H  = 64;
    const int64_t N1 = N + 1;

    std::vector<int64_t> prefixSumShape      = {B, H};
    std::vector<int64_t> blockResidualShape  = {B, N, H};
    std::vector<int64_t> projWeightShape     = {1, H};
    std::vector<int64_t> normWeightShape     = {H};
    std::vector<int64_t> gradOutputShape     = {B, H};
    std::vector<int64_t> invRmsShape         = {B, N1};
    std::vector<int64_t> probsShape          = {B, N1};

    std::vector<uint16_t> prefixSumData     (GetShapeSize(prefixSumShape),     0x3C00);
    std::vector<uint16_t> blockResidualData (GetShapeSize(blockResidualShape),  0x3C00);
    std::vector<uint16_t> projWeightData    (GetShapeSize(projWeightShape),     0x3C00);
    std::vector<uint16_t> normWeightData    (GetShapeSize(normWeightShape),     0x3C00);
    std::vector<uint16_t> gradOutputData    (GetShapeSize(gradOutputShape),     0x3C00);
    std::vector<float> invRmsData           (GetShapeSize(invRmsShape),         1.0f);
    std::vector<float> probsData            (GetShapeSize(probsShape),          1.0f);

    aclTensor *prefix_sum = nullptr, *block_residual = nullptr, *proj_weight = nullptr, *norm_weight = nullptr;
    aclTensor *grad_output = nullptr, *inv_rms = nullptr, *probs = nullptr;
    void *d_prefix_sum = nullptr, *d_block_residual = nullptr, *d_proj_weight = nullptr, *d_norm_weight = nullptr;
    void *d_grad_output = nullptr, *d_inv_rms = nullptr, *d_probs = nullptr;

    ret = CreateAclTensor(prefixSumData, prefixSumShape, &d_prefix_sum, aclDataType::ACL_FLOAT16, &prefix_sum); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockResidualData, blockResidualShape, &d_block_residual, aclDataType::ACL_FLOAT16, &block_residual); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(projWeightData, projWeightShape, &d_proj_weight, aclDataType::ACL_FLOAT16, &proj_weight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normWeightData, normWeightShape, &d_norm_weight, aclDataType::ACL_FLOAT16, &norm_weight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradOutputData, gradOutputShape, &d_grad_output, aclDataType::ACL_FLOAT16, &grad_output); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invRmsData, invRmsShape, &d_inv_rms, aclDataType::ACL_FLOAT, &inv_rms); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsData, probsShape, &d_probs, aclDataType::ACL_FLOAT, &probs); CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *grad_prefix_sum = nullptr, *grad_block_residual = nullptr, *grad_proj_weight = nullptr, *grad_norm_weight = nullptr;
    void *d_grad_prefix_sum = nullptr, *d_grad_block_residual = nullptr, *d_grad_proj_weight = nullptr, *d_grad_norm_weight = nullptr;

    std::vector<uint16_t> gradPrefixSumData    (GetShapeSize(prefixSumShape),     0);
    std::vector<uint16_t> gradBlockResidualData(GetShapeSize(blockResidualShape), 0);
    std::vector<uint16_t> gradProjWeightData   (GetShapeSize(projWeightShape),    0);
    std::vector<uint16_t> gradNormWeightData   (GetShapeSize(normWeightShape),    0);

    ret = CreateAclTensor(gradPrefixSumData, prefixSumShape, &d_grad_prefix_sum, aclDataType::ACL_FLOAT16, &grad_prefix_sum); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradBlockResidualData, blockResidualShape, &d_grad_block_residual, aclDataType::ACL_FLOAT16, &grad_block_residual); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradProjWeightData, projWeightShape, &d_grad_proj_weight, aclDataType::ACL_FLOAT16, &grad_proj_weight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormWeightData, normWeightShape, &d_grad_norm_weight, aclDataType::ACL_FLOAT16, &grad_norm_weight); CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnAttentionResidualsGradGetWorkspaceSize(
        prefix_sum, block_residual, proj_weight, norm_weight,
        grad_output, inv_rms, probs,
        grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionResidualsGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnAttentionResidualsGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionResidualsGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("AttentionResidualsGrad example ran successfully!\n");

    aclDestroyTensor(prefix_sum); aclDestroyTensor(block_residual); aclDestroyTensor(proj_weight);
    aclDestroyTensor(norm_weight); aclDestroyTensor(grad_output); aclDestroyTensor(inv_rms);
    aclDestroyTensor(probs);
    aclrtFree(d_prefix_sum); aclrtFree(d_block_residual); aclrtFree(d_proj_weight);
    aclrtFree(d_norm_weight); aclrtFree(d_grad_output); aclrtFree(d_inv_rms);
    aclrtFree(d_probs);

    aclDestroyTensor(grad_prefix_sum); aclDestroyTensor(grad_block_residual); aclDestroyTensor(grad_proj_weight);
    aclDestroyTensor(grad_norm_weight);
    aclrtFree(d_grad_prefix_sum); aclrtFree(d_grad_block_residual); aclrtFree(d_grad_proj_weight);
    aclrtFree(d_grad_norm_weight);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
