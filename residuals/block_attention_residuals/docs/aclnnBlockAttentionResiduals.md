# aclnnBlockAttentionResiduals

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

- 接口功能：BlockAttentionResiduals是attention residual正向计算算子。该接口将prefix_sum与block_res拼接后，经RMS归一化、线性投影和softmax加权融合，输出hidden_states；当needBackward为true时，同时输出反向计算所需的invNorm和probs。
- 使用场景：attention residual分支的前向融合计算，needBackward为true时同时保存反向计算所需的invNorm和probs。
- V版本说明：当前无V版本差异。
- 芯片差异：<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>计算语义一致；A2/A3使用arch22实现，Ascend950使用arch35 regbase实现。

- 计算公式：

  正向计算可表示为：

  $$
  v_{t,i,h} =
  \begin{cases}
  block\_residual_{t,i,h}, & i < N \\
  prefix\_sum_{t,h}, & i = N
  \end{cases}
  $$

  $$
  variance_{t,i} = \frac{1}{H}\sum_{h=0}^{H-1} v_{t,i,h}^{2}
  $$

  $$
  inv\_rms_{t,i} = \frac{1}{\sqrt{variance_{t,i} + norm\_eps}}
  $$

  $$
  k_{t,i,h} = v_{t,i,h} \cdot inv\_rms_{t,i}
  $$

  $$
  score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
  $$

  $$
  s_{t,i} = \sum_{h=0}^{H-1} k_{t,i,h} \cdot score\_weight_{h}
  $$

  $$
  probs_{t,i} = \frac{e^{s_{t,i}}}{\sum_{j=0}^{N} e^{s_{t,j}}}
  $$

  $$
  hidden\_states_{t,h} = \sum_{i=0}^{N} probs_{t,i} \cdot v_{t,i,h}
  $$

### 符号说明

| 符号 | 含义 |
|------|------|
| $T$ | token数量 |
| $N$ | block_res的block数量 |
| $H$ | hidden size |
| $v_{t,i,h}$ | block_res与prefix_sum拼接后的value，维度为 $(T,N+1,H)$ |
| $inv\_rms_{t,i}$ | 前向逐行归一化系数 |
| $score\_weight_{h}$ | norm_weight与proj_weight逐元素相乘的结果 |
| $probs_{t,i}$ | 前向softmax输出概率 |
| $hidden\_states_{t,h}$ | 前向加权融合输出 |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBlockAttentionResidualsGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnBlockAttentionResiduals”接口执行计算。

```cpp
aclnnStatus aclnnBlockAttentionResidualsGetWorkspaceSize(
    const aclTensor *prefixSum,
    const aclTensor *blockRes,
    const aclTensor *projWeight,
    const aclTensor *normWeight,
    int64_t          validBlockNum,
    double           normEps,
    bool             needBackward,
    aclTensor       *hiddenStates,
    aclTensor       *invNorm,
    aclTensor       *probs,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnBlockAttentionResiduals(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnBlockAttentionResidualsGetWorkspaceSize

### 参数说明

<table style="table-layout: fixed; width: 1200px; max-width: 100%"><colgroup>
  <col style="width: 180px">
  <col style="width: 90px">
  <col style="width: 220px">
  <col style="width: 290px">
  <col style="width: 160px">
  <col style="width: 70px">
  <col style="width: 100px">
  <col style="width: 90px">
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
    <td>prefixSum（aclTensor*）</td>
    <td>输入</td>
    <td>前向输入前缀和，对应公式中拼接后的第N+1个value。</td>
    <td>支持T=0的空Tensor；T、H与blockRes保持一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>(T,H)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>blockRes（aclTensor*）</td>
    <td>输入</td>
    <td>前向输入分块残差，对应公式中前N个value。</td>
    <td>支持T=0的空Tensor；dtype与prefixSum保持一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>(T,N,H)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>projWeight（aclTensor*）</td>
    <td>输入</td>
    <td>前向线性投影权重，与normWeight共同构成score_weight。</td>
    <td>shape支持(1,H)或(H)；dtype与prefixSum保持一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>(1,H)或(H)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>normWeight（aclTensor*）</td>
    <td>输入</td>
    <td>前向归一化权重，与projWeight共同构成score_weight。</td>
    <td>shape为(H)；dtype与prefixSum保持一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>(H)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>validBlockNum（int64_t）</td>
    <td>输入</td>
    <td>当前有效的block数量。</td>
    <td>当前仅支持默认场景，需等于blockRes的N；缺省语义为N。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>normEps（double）</td>
    <td>输入</td>
    <td>RMS归一化数值稳定性参数。</td>
    <td>需大于0，默认1e-5。</td>
    <td>DOUBLE</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>needBackward（bool）</td>
    <td>输入</td>
    <td>是否需要保存invNorm和probs供反向计算使用。</td>
    <td>默认false；true时invNorm/probs不能为空。</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>hiddenStates（aclTensor*）</td>
    <td>输出</td>
    <td>前向加权融合输出，对应公式中hidden_states。</td>
    <td>不支持空Tensor；dtype与prefixSum保持一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>(T,H)</td>
    <td>-</td>
  </tr>
  <tr>
    <td>invNorm（aclTensor*）</td>
    <td>输出</td>
    <td>前向保存的逐行归一化系数。</td>
    <td>needBackward=false时传空指针；仅支持FLOAT32。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(T,N+1)</td>
    <td>-</td>
  </tr>
  <tr>
    <td>probs（aclTensor*）</td>
    <td>输出</td>
    <td>前向softmax输出概率。</td>
    <td>needBackward=false时传空指针；仅支持FLOAT32。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(T,N+1)</td>
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
  <tr>
    <th>返回值</th>
    <th>错误码</th>
    <th>描述</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3">161002</td>
    <td>输入输出张量存在空指针（含needBackward为true时invNorm/probs为空），返回实际日志包含空指针校验错误。</td>
  </tr>
  <tr>
    <td>输入张量的数据类型不在支持的范围内。</td>
  </tr>
  <tr>
    <td>实际错误文本包括：<code>invalid tensor rank</code>、<code>only default valid_block_num=N is supported, got %ld</code>、<code>shape range requires T&gt;=0, H&gt;=1 and 1&lt;=N&lt;=%ld</code>、<code>block_res shape must be [T,N,H]</code>、<code>proj_weight must be [H] or [1,H], and norm_weight must be [H]</code>、<code>hidden_states shape must be [T,H]</code>、<code>inv_norm/probs must be 2D [T,B]</code>、<code>inv_norm/probs shape must be [%ld,%ld]</code>、<code>normEps must be &gt; 0</code>。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>API调用npu runtime的接口异常。</td>
  </tr>
</tbody>
</table>

## aclnnBlockAttentionResiduals

### 参数说明

<table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
</colgroup>
<thead>
  <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
</thead>
<tbody>
  <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnBlockAttentionResidualsGetWorkspaceSize获取。</td></tr>
  <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
  <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
</tbody>
</table>

### 返回值

返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 确定性计算

- aclnnBlockAttentionResiduals默认确定性实现。

### 规格约束

- prefixSum、blockRes、projWeight、normWeight和hiddenStates支持FLOAT16、BFLOAT16、FLOAT32；invNorm和probs仅支持FLOAT32。
- shape需满足：prefixSum为(T,H)，blockRes为(T,N,H)，projWeight为(1,H)或(H)，normWeight为(H)，hiddenStates为(T,H)，invNorm为(T,N+1)，probs为(T,N+1)，其中T可为0，H大于等于1，N在1到100之间，且各张量中的T、H保持一致。
- validBlockNum为INT64，当前仅支持validBlockNum=N。
- normEps为DOUBLE，需大于0，默认值为1e-5；传入0或负值时返回ACLNN_ERR_PARAM_INVALID，实际错误文本为“normEps must be > 0”。
- needBackward为true时，invNorm和probs必须为非空Tensor且shape为(T,N+1)。
- 输入张量支持非连续Tensor，接口内部统一转为Contiguous后计算；输出通过ViewCopy写回。
- workspace为系统预留16MB。

### 预期异常

以下输入校验异常统一返回`ACLNN_ERR_PARAM_INVALID`（161002），ATK异常用例按实际错误文本匹配：

- `validBlockNum != blockRes.shape[1]`：`only default valid_block_num=N is supported, got %ld`。
- `normEps <= 0`：`normEps must be > 0`。
- rank、跨输入shape或反向输出shape不匹配：分别匹配`invalid tensor rank`、`block_res shape must be [T,N,H]`、`proj_weight must be [H] or [1,H], and norm_weight must be [H]`、`hidden_states shape must be [T,H]`、`inv_norm/probs must be 2D [T,B]`或`inv_norm/probs shape must be [%ld,%ld]`。
- T/H/N越界：`shape range requires T>=0, H>=1 and 1<=N<=100`。
- `needBackward=false`时`invNorm`和`probs`按接口约定传空指针；`needBackward=true`时两者必须非空且为`FLOAT32`、shape为(T,N+1)。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_block_attention_residuals.h"

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
    for (auto dim : shape) {
        shapeSize *= dim;
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

int CreateAclTensor(
    const std::vector<uint16_t>& hostData, const std::vector<int64_t>& shape,
    void** deviceAddr, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(uint16_t);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), aclDataType::ACL_BF16, nullptr, 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t T = 2;
    const int64_t N = 4;
    const int64_t H = 64;

    std::vector<int64_t> prefixShape     = {T, H};
    std::vector<int64_t> blockShape      = {T, N, H};
    std::vector<int64_t> projShape       = {1, H};
    std::vector<int64_t> normShape       = {H};
    std::vector<int64_t> hiddenShape     = {T, H};

    std::vector<uint16_t> prefixData(GetShapeSize(prefixShape), 0x3C00);
    std::vector<uint16_t> blockData(GetShapeSize(blockShape), 0x3C00);
    std::vector<uint16_t> projData(GetShapeSize(projShape), 0x3C00);
    std::vector<uint16_t> normData(GetShapeSize(normShape), 0x3C00);
    std::vector<uint16_t> hiddenData(GetShapeSize(hiddenShape), 0);

    void* prefixDev = nullptr;
    void* blockDev = nullptr;
    void* projDev = nullptr;
    void* normDev = nullptr;
    void* hiddenDev = nullptr;

    aclTensor* prefixTensor = nullptr;
    aclTensor* blockTensor = nullptr;
    aclTensor* projTensor = nullptr;
    aclTensor* normTensor = nullptr;
    aclTensor* hiddenTensor = nullptr;

    ret = CreateAclTensor(prefixData, prefixShape, &prefixDev, &prefixTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockData, blockShape, &blockDev, &blockTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(projData, projShape, &projDev, &projTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normData, normShape, &normDev, &normTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hiddenData, hiddenShape, &hiddenDev, &hiddenTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnBlockAttentionResidualsGetWorkspaceSize(
        prefixTensor, blockTensor, projTensor, normTensor, N, 1e-5, false,
        hiddenTensor, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnBlockAttentionResiduals(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResiduals failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("BlockAttentionResiduals example ran successfully!\n");

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyTensor(prefixTensor);
    aclDestroyTensor(blockTensor);
    aclDestroyTensor(projTensor);
    aclDestroyTensor(normTensor);
    aclDestroyTensor(hiddenTensor);
    aclrtFree(prefixDev);
    aclrtFree(blockDev);
    aclrtFree(projDev);
    aclrtFree(normDev);
    aclrtFree(hiddenDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
