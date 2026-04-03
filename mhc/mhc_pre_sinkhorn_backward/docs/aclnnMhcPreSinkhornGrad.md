# aclnnMhcPreSinkhornBackward

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_pre_sinkhorn_grad)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term> |      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

- **接口功能**：MhcPreSinkhornBackward是MhcPreSinkhorn的反向算子,实现对应的梯度方向传播。

- **计算公式**：

  MhcPreSinkhornBackward算子是MhcPreSinkhorn前向算子的反向传播，根据输出梯度和前向保存的中间结果计算输入张量和参数的梯度。



## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnMhcPreSinkhornBackwardGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMhcPreSinkhornBackward"接口执行计算。

```cpp
aclnnStatus aclnnMhcPreSinkhornBackwardGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *gradPost,
    const aclTensor *gradCombFrag,
    const aclTensor *x,
    const aclTensor *hcFn,
    const aclTensor *hcScale,
    const aclTensor *hcBase,
    const aclTensor *pre,
    const aclTensor *hcBeforeNorm,
    const aclTensor *rsqrt,
    const aclTensor *sumOut,
    const aclTensor *normOut,
    double hcEps,
    aclTensor *gradX,
    aclTensor *gradHcFn,
    aclTensor *gradHcScale,
    aclTensor *gradHcBase,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnMhcPreSinkhornBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnMhcPreSinkhornBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 265px">
    <col style="width: 120px">
    <col style="width: 223px">  
    <col style="width: 391px">  
    <col style="width: 181px">  
    <col style="width: 111px"> 
    <col style="width: 126px">
    <col style="width: 145px">
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
      <td>gradY</td>
      <td>输入</td>
      <td>输出y的梯度。</td>
      <td><ul><li>不支持空tensor。</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(B, S, C)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradPost</td>
      <td>输入</td>
      <td>输出post的梯度。</td>
      <td><ul><li>不支持空tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B, S, N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradCombFrag</td>
      <td>输入</td>
      <td>输出comb_frag的梯度。</td>
      <td><ul><li>不支持空tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B, S, N*N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>前向输入x。</td>
      <td><ul><li>不支持空tensor。</li><li>前向计算时保存的中间结果。</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(B, S, N, C)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hcFn</td>
      <td>输入</td>
      <td>前向参数hc_fn。</td>
      <td><ul><li>不支持空tensor。</li><li>前向计算时保存的中间结果。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(N*N+2*N, N*C)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hcScale</td>
      <td>输入</td>
      <td>前向参数hc_scale。</td>
      <td><ul><li>不支持空tensor。</li><li>前向计算时保存的中间结果。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(3)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hcBase</td>
      <td>输入</td>
      <td>前向参数hc_base。</td>
      <td><ul><li>不支持空tensor。</li><li>前向计算时保存的中间结果。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(N*N+2*N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pre</td>
      <td>输入</td>
      <td>前向保存的中间结果pre。</td>
      <td><ul><li>不支持空tensor。</li><li>前向计算时保存的中间结果。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B, S, N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hcBeforeNorm</td>
      <td>输入</td>
      <td>前向保存的中间结果hc_before_norm。</td>
      <td><ul><li>不支持空tensor。</li><li>前向计算时保存的中间结果。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B, S, N*N+2*N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rsqrt</td>
      <td>输入</td>
      <td>前向保存的中间结果rsqrt(1/r)。</td>
      <td><ul><li>不支持空tensor。</li><li>前向计算时保存的中间结果。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B, S, 1)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumOut</td>
      <td>输入</td>
      <td>Sinkhorn变换正向计算保存的中间sum结果。</td>
      <td><ul><li>不支持空tensor。</li><li>前向计算时保存的中间结果。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(2*sk_iter_count, B, S, N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normOut</td>
      <td>输入</td>
      <td>Sinkhorn变换正向计算保存的中间norm结果。</td>
      <td><ul><li>不支持空tensor。</li><li>前向计算时保存的中间结果。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(2*sk_iter_count, B, S, N, N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hcEps</td>
      <td>输入</td>
      <td>数值稳定性参数。</td>
      <td>默认值1e-6。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradX</td>
      <td>输出</td>
      <td>输入x的梯度。</td>
      <td><ul><li>数据类型同x。</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(B, S, N, C)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradHcFn</td>
      <td>输出</td>
      <td>参数hc_fn的梯度。</td>
      <td><ul><li>数据类型同hcFn。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(N*N+2*N, N*C)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradHcScale</td>
      <td>输出</td>
      <td>参数hc_scale的梯度。</td>
      <td><ul><li>数据类型同hcScale。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(3)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradHcBase</td>
      <td>输出</td>
      <td>参数hc_base的梯度。</td>
      <td><ul><li>数据类型同hcBase。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(N*N+2*N)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1180px"> 
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
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
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>输入和输出的Tensor是空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>输入和输出的数据类型不在支持的范围内。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>API内存调用npu runtime的接口异常。</td>
      </tr>
    </tbody>
  </table>

## aclnnMhcPreSinkhornBackward

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1180px"> 
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
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
        <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMhcPreSinkhornBackwardGetWorkspaceSize</code>获取。</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream流。</td>
      </tr>
    </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
    
## 约束说明

- 确定性计算：
  - aclnnMhcPreSinkhornBackward默认确定性实现。

- 规格约束：

  | 规格项 | 规格 | 规格说明 |
  |:--- |:--- |:--- |
  | sk_iter_count | 1~100 | Sinkhorn迭代次数超出该范围会返回参数无效错误。 |
  | N | 4、6、8 | 输入Tensor中N维度的大小仅支持这三个值。 |
  | 维度数 | 3/4 | 输入gradY支持3维(B,S,C)，输入x支持4维(B,S,N,C)。 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_sinkhorn_grad.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)
#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    
    const int64_t B = 2;
    const int64_t S = 512;
    const int64_t N = 8;
    const int64_t C = 4096;
    const int64_t sk_iter_count = 4;

    std::vector<int64_t> gradYShape = {B, S, C};
    std::vector<int64_t> gradPostShape = {B, S, N};
    std::vector<int64_t> gradCombFragShape = {B, S, N * N};
    std::vector<int64_t> xShape = {B, S, N, C};
    std::vector<int64_t> hcFnShape = {N * N + 2 * N, N * C};
    std::vector<int64_t> hcScaleShape = {3};
    std::vector<int64_t> hcBaseShape = {N * N + 2 * N};
    std::vector<int64_t> preShape = {B, S, N};
    std::vector<int64_t> hcBeforeNormShape = {B, S, N * N + 2 * N};
    std::vector<int64_t> rsqrtShape = {B, S, 1};
    std::vector<int64_t> sumOutShape = {2 * sk_iter_count, B, S, N};
    std::vector<int64_t> normOutShape = {2 * sk_iter_count, B, S, N, N};

    void* gradYDeviceAddr = nullptr;
    void* gradPostDeviceAddr = nullptr;
    void* gradCombFragDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* hcFnDeviceAddr = nullptr;
    void* hcScaleDeviceAddr = nullptr;
    void* hcBaseDeviceAddr = nullptr;
    void* preDeviceAddr = nullptr;
    void* hcBeforeNormDeviceAddr = nullptr;
    void* rsqrtDeviceAddr = nullptr;
    void* sumOutDeviceAddr = nullptr;
    void* normOutDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    void* gradHcFnDeviceAddr = nullptr;
    void* gradHcScaleDeviceAddr = nullptr;
    void* gradHcBaseDeviceAddr = nullptr;

    aclTensor* gradY = nullptr;
    aclTensor* gradPost = nullptr;
    aclTensor* gradCombFrag = nullptr;
    aclTensor* x = nullptr;
    aclTensor* hcFn = nullptr;
    aclTensor* hcScale = nullptr;
    aclTensor* hcBase = nullptr;
    aclTensor* pre = nullptr;
    aclTensor* hcBeforeNorm = nullptr;
    aclTensor* rsqrt = nullptr;
    aclTensor* sumOut = nullptr;
    aclTensor* normOut = nullptr;
    aclTensor* gradX = nullptr;
    aclTensor* gradHcFn = nullptr;
    aclTensor* gradHcScale = nullptr;
    aclTensor* gradHcBase = nullptr;

    int64_t gradYSize = GetShapeSize(gradYShape);
    int64_t gradPostSize = GetShapeSize(gradPostShape);
    int64_t gradCombFragSize = GetShapeSize(gradCombFragShape);
    int64_t xSize = GetShapeSize(xShape);
    int64_t hcFnSize = GetShapeSize(hcFnShape);
    int64_t hcScaleSize = GetShapeSize(hcScaleShape);
    int64_t hcBaseSize = GetShapeSize(hcBaseShape);
    int64_t preSize = GetShapeSize(preShape);
    int64_t hcBeforeNormSize = GetShapeSize(hcBeforeNormShape);
    int64_t rsqrtSize = GetShapeSize(rsqrtShape);
    int64_t sumOutSize = GetShapeSize(sumOutShape);
    int64_t normOutSize = GetShapeSize(normOutShape);

    std::vector<float> gradYHostData(gradYSize, 1.0f);
    std::vector<float> gradPostHostData(gradPostSize, 1.0f);
    std::vector<float> gradCombFragHostData(gradCombFragSize, 1.0f);
    std::vector<float> xHostData(xSize, 1.0f);
    std::vector<float> hcFnHostData(hcFnSize, 1.0f);
    std::vector<float> hcScaleHostData(hcScaleSize, 1.0f);
    std::vector<float> hcBaseHostData(hcBaseSize, 1.0f);
    std::vector<float> preHostData(preSize, 1.0f);
    std::vector<float> hcBeforeNormHostData(hcBeforeNormSize, 1.0f);
    std::vector<float> rsqrtHostData(rsqrtSize, 1.0f);
    std::vector<float> sumOutHostData(sumOutSize, 1.0f);
    std::vector<float> normOutHostData(normOutSize, 1.0f);
    std::vector<float> gradXHostData(xSize, 0.0f);
    std::vector<float> gradHcFnHostData(hcFnSize, 0.0f);
    std::vector<float> gradHcScaleHostData(hcScaleSize, 0.0f);
    std::vector<float> gradHcBaseHostData(hcBaseSize, 0.0f);

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_BF16, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradPostHostData, gradPostShape, &gradPostDeviceAddr, aclDataType::ACL_FLOAT, &gradPost);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradCombFragHostData, gradCombFragShape, &gradCombFragDeviceAddr, aclDataType::ACL_FLOAT, &gradCombFrag);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hcFnHostData, hcFnShape, &hcFnDeviceAddr, aclDataType::ACL_FLOAT, &hcFn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hcScaleHostData, hcScaleShape, &hcScaleDeviceAddr, aclDataType::ACL_FLOAT, &hcScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hcBaseHostData, hcBaseShape, &hcBaseDeviceAddr, aclDataType::ACL_FLOAT, &hcBase);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(preHostData, preShape, &preDeviceAddr, aclDataType::ACL_FLOAT, &pre);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hcBeforeNormHostData, hcBeforeNormShape, &hcBeforeNormDeviceAddr, aclDataType::ACL_FLOAT, &hcBeforeNorm);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rsqrtHostData, rsqrtShape, &rsqrtDeviceAddr, aclDataType::ACL_FLOAT, &rsqrt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sumOutHostData, sumOutShape, &sumOutDeviceAddr, aclDataType::ACL_FLOAT, &sumOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normOutHostData, normOutShape, &normOutDeviceAddr, aclDataType::ACL_FLOAT, &normOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(gradXHostData, xShape, &gradXDeviceAddr, aclDataType::ACL_BF16, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHcFnHostData, hcFnShape, &gradHcFnDeviceAddr, aclDataType::ACL_FLOAT, &gradHcFn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHcScaleHostData, hcScaleShape, &gradHcScaleDeviceAddr, aclDataType::ACL_FLOAT, &gradHcScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHcBaseHostData, hcBaseShape, &gradHcBaseDeviceAddr, aclDataType::ACL_FLOAT, &gradHcBase);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double hcEps = 1e-6;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnMhcPreSinkhornBackwardGetWorkspaceSize(
        gradY, gradPost, gradCombFrag,
        x, hcFn, hcScale, hcBase,
        pre, hcBeforeNorm, rsqrt,
        sumOut, normOut, hcEps,
        gradX, gradHcFn, gradHcScale, gradHcBase,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhornBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnMhcPreSinkhornBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhornBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("MhcPreSinkhornBackward execution succeeded\n");

    aclDestroyTensor(gradY);
    aclDestroyTensor(gradPost);
    aclDestroyTensor(gradCombFrag);
    aclDestroyTensor(x);
    aclDestroyTensor(hcFn);
    aclDestroyTensor(hcScale);
    aclDestroyTensor(hcBase);
    aclDestroyTensor(pre);
    aclDestroyTensor(hcBeforeNorm);
    aclDestroyTensor(rsqrt);
    aclDestroyTensor(sumOut);
    aclDestroyTensor(normOut);
    aclDestroyTensor(gradX);
    aclDestroyTensor(gradHcFn);
    aclDestroyTensor(gradHcScale);
    aclDestroyTensor(gradHcBase);

    aclrtFree(gradYDeviceAddr);
    aclrtFree(gradPostDeviceAddr);
    aclrtFree(gradCombFragDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(hcFnDeviceAddr);
    aclrtFree(hcScaleDeviceAddr);
    aclrtFree(hcBaseDeviceAddr);
    aclrtFree(preDeviceAddr);
    aclrtFree(hcBeforeNormDeviceAddr);
    aclrtFree(rsqrtDeviceAddr);
    aclrtFree(sumOutDeviceAddr);
    aclrtFree(normOutDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradHcFnDeviceAddr);
    aclrtFree(gradHcScaleDeviceAddr);
    aclrtFree(gradHcBaseDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
