# aclnnAiInfraMhcPostGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |

## 功能说明

- 算子功能：mhc_post基于一系列计算对mHC架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_j$进行ResMapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。该算子实现前述过程的反向功能。
  
- 计算公式：
  $$
  grad\_x = H_{l}^{res} \times grad\_output\\
  grad\_h\_res = x_{l} \times {grad\_output}^{T}
  $$
  $$
  grad\_h\_out=({grad\_output} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
  grad\_h\_post=({grad\_output} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
  $$

## 函数原型

算子执行接口为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnAiInfraMhcPostGradGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnAiInfraMhcPostGrad”接口执行计算。

```c++
aclnnStatus aclnnAiInfraMhcPostGradGetWorkspaceSize(
    const aclTensor     *grad_output,
    const aclTensor     *x, 
    const aclTensor     *h_res, 
    const aclTensor     *h_out, 
    const aclTensor     *h_post, 
    const aclTensor     *grad_x, 
    const aclTensor     *grad_h_res, 
    const aclTensor     *grad_h_out, 
    const aclTensor     *grad_h_post,
    const aclTensor     *output, 

    uint64_t            *workspaceSize, 
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnAiInfraMhcPostGrad(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    const aclrtStream stream)
```

## aclnnAiInfraMhcPostGradGetWorkspaceSize

- **参数说明：**
  
  > 列表中的列项详细信息可以参考：<br>
  > [非连续的Tensor](../../../docs/context/非连续的Tensor.md)<br>
  > [数据格式](../../../docs/context/数据格式.md)<br>
  > [约束说明](#约束说明)
  
  <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 500px">  
            <col style="width: 212px">  
            <col style="width: 100px">
            <col style="width: 190px">
            <col style="width: 145px">
            </colgroup>
        <thead>
        <tr>
            <th>参数名</th>
            <th>输入/输出</th>
            <th>描述说明</th>
            <th>数据类型</th>
            <th>数据格式</th>
            <th>非连续Tensor</th>
        </tr></thead>
        <tbody>
        <tr>
            <td>grad_output</td>
            <td>输入</td>
            <td>待计算的数据，表示网络中mHC层的输入数据。<br>维度shape支持[B,S,n,D]，或者为[T,n,D]。</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>√</td>
        </tr>
        <tr>
            <td>x</td>
            <td>输入</td>
            <td>待计算的数据，表示网络中mHC层的输入数据。<br>维度shape支持[B,S,n,D]，或者为[T,n,D]。</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>√</td>
        </tr>
        <tr>
            <td>h_res</td>
            <td>输入</td>
            <td>mHC的h_res变换矩阵，是做完sinkhorn变换后的双随机矩阵。<br>维度shape支持[B,S,n,n]，或者为[T,n,n]。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>√</td>
        </tr>
        <tr>
            <td>h_out</td>
            <td>输入</td>
            <td>Atten/MLP层的输出。<br>维度shape支持[B,S,D]，或者为[T,D]。</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>√</td>
        </tr>
        <tr>
            <td>h_post</td>
            <td>输入</td>
            <td>mHC的h_post变换矩阵。<br>维度shape支持[B,S,n]，或者为[T,n]。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>√</td>
        </tr>
        <tr>
            <td>grad_x</td>
            <td>输出</td>
            <td>表示网络中mHC层输入数据x的梯度。<br>维度shape支持[B,S,n,D]，或者为[T,n,D]。</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>×</td>
        </tr>
        <tr>
            <td>grad_h_res</td>
            <td>输出</td>
            <td>mHC的h_res变换矩阵。<br>维度shape支持[B,S,n,n]，或者为[T,n,n]。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>×</td>
        </tr>
        <tr>
            <td>grad_h_out</td>
            <td>输出</td>
            <td>Atten/MLP层的输出。<br>维度shape支持[B,S,D]，或者为[T,D]。</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>×</td>
        </tr>
        <tr>
            <td>grad_h_post</td>
            <td>输出</td>
            <td>mHC的h_post变换矩阵。<br>维度shape支持[B,S,n]，或者为[T,n]。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>×</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>输出</td>
            <td>返回用户需要在Device侧申请的workspace大小</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>输出</td>
            <td>返回op执行器，包含了算子计算流程</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
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
                <td>传入的query、key、value、attentionOut是空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>query、key、value、pseShift、attenMaskOptional、attentionOut的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>

## aclnnAiInfraMhcPostGrad

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnAiInfraMhcPostGradGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的AscendCL stream流。</td>
        </tr>
    </tbody>
    </table>
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

### 确定性计算

- AiInfraMhcPostGrad 算子默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束
- 输入约束：
   - 所有输入参数不能为空，否则返回失败。
   - h_res 分布满足双随机矩阵

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 1~512k | B*S 或T支持1~512k范围以内。|
| n | 4、6、8 | n值目前支持4, 6, 8。|
| D | 384~24k | D支持384~24k范围以内。|

### 典型值

| 规格项 | 典型值 |
|:--- |:--- |
| T或 B*S | 1024/2048/4096 |
| n | 4（推荐） |
| D | 2560/5120（推荐） |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_ai_infra_mhc_post_grad.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    // Fixed writing method, AscendCL initialization.
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    // Shape: TND (T=1024, n=4, D=2560)
    const int64_t T = 1024;
    const int64_t n = 4;
    const int64_t D = 2560;

    std::vector<int64_t> grad_outputShape = {T, n, D};
    std::vector<int64_t> xShape = {T, n, D};
    std::vector<int64_t> h_resShape = {T, n, n};
    std::vector<int64_t> h_outShape = {T, D};
    std::vector<int64_t> h_postShape = {T, n};
    std::vector<int64_t> grad_xShape = {T, n, D};
    std::vector<int64_t> grad_h_resShape = {T, n, n};
    std::vector<int64_t> grad_h_outShape = {T, D};
    std::vector<int64_t> grad_h_postShape = {T, n};

    void *grad_outputDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;
    void *h_resDeviceAddr = nullptr;
    void *h_outDeviceAddr = nullptr;
    void *h_postDeviceAddr = nullptr;
    void *grad_xDeviceAddr = nullptr;
    void *grad_h_resDeviceAddr = nullptr;
    void *grad_h_outDeviceAddr = nullptr;
    void *grad_h_postDeviceAddr = nullptr;

    aclTensor *grad_outputTensor = nullptr;
    aclTensor *xTensor = nullptr;
    aclTensor *h_resTensor = nullptr;
    aclTensor *h_outTensor = nullptr;
    aclTensor *h_postTensor = nullptr;
    aclTensor *grad_xTensor = nullptr;
    aclTensor *grad_h_resTensor = nullptr;
    aclTensor *grad_h_outTensor = nullptr;
    aclTensor *grad_h_postTensor = nullptr;

    int64_t grad_outputShapeSize = GetShapeSize(grad_outputShape);
    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t h_resShapeSize = GetShapeSize(h_resShape);
    int64_t h_outShapeSize = GetShapeSize(h_outShape);
    int64_t h_postShapeSize = GetShapeSize(h_postShape);
    int64_t grad_xShapeSize = GetShapeSize(grad_xShape);
    int64_t grad_h_resShapeSize = GetShapeSize(grad_h_resShape);
    int64_t grad_h_outShapeSize = GetShapeSize(grad_h_outShape);
    int64_t grad_h_postShapeSize = GetShapeSize(grad_h_postShape);

    // Create inpit data with default values
    std::vector<uinit16_t> grad_outputHostData(grad_outputShapeSize, 1);    // FP16/BF16 data
    std::vector<uinit16_t> xHostData(xShapeSize, 1);                        // FP16/BF16 data
    std::vector<float> h_resHostData(h_resShapeSize, 0.25f);                // FP32 data double-stochastistic constrained
    std::vector<uinit16_t> h_outHostData(h_outShapeSize, 1);                // FP16/BF16 data
    std::vector<float> h_postHostData(h_postShapeSize, 1.0f);               // FP32 data
    
    std::vector<uinit16_t> grad_xHostData(grad_xShapeSize, 0);              // FP16/BF16 output
    std::vector<float> grad_h_resHostData(grad_h_resShapeSize, 0.25f);      // FP32 output
    std::vector<uinit16_t> grad_h_outHostData(grad_h_outShapeSize, 1);      // FP16/BF16 output
    std::vector<float> grad_h_postHostData(grad_h_postShapeSize, 1.0f);     // FP32 output

    // Create grad_output aclTensor.
    ret = CreateAclTensor(grad_outputHostData, grad_outputShape, &grad_outputDeviceAddr, aclDataType::ACL_FLOAT16, &grad_outputTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create h_res aclTensor.
    ret = CreateAclTensor(h_resHostData, h_resShape, &h_resDeviceAddr, aclDataType::ACL_FLOAT, &h_resTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create h_out aclTensor.
    ret = CreateAclTensor(h_outHostData, h_outShape, &h_outDeviceAddr, aclDataType::ACL_FLOAT16, &h_outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create h_post aclTensor.
    ret = CreateAclTensor(h_postHostData, h_postShape, &h_postDeviceAddr, aclDataType::ACL_FLOAT, &h_postTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create grad_x aclTensor.
    ret = CreateAclTensor(grad_xHostData, grad_xShape, &grad_xDeviceAddr, aclDataType::ACL_FLOAT16, &grad_xTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create grad_h_res aclTensor.
    ret = CreateAclTensor(grad_h_resHostData, grad_h_resShape, &grad_h_resDeviceAddr, aclDataType::ACL_FLOAT, &grad_h_resTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create grad_h_out aclTensor.
    ret = CreateAclTensor(grad_h_outHostData, grad_h_outShape, &grad_h_outDeviceAddr, aclDataType::ACL_FLOAT16, &grad_h_outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create grad_h_post aclTensor.
    ret = CreateAclTensor(grad_h_postHostData, grad_h_postShape, &grad_h_postDeviceAddr, aclDataType::ACL_FLOAT, &grad_h_postTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnAiInfraMhcPostGradGetWorkspaceSize(
        grad_outputTensor, xTensor, h_resTensor, h_outTensor, h_postTensor,
        grad_xTensor, grad_h_resTensor, grad_h_outTensor, grad_h_postTensor,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnAiInfraMhcPostGradGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnAiInfraMhcPostGrad(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnAiInfraMhcPostGrad failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. Retrieve the output value, copy the result from the device side memory to the host side, and modify it
    // Retrive grad_x
    std::vector<uint16_t> grad_xResultData(grad_xShapeSize, 0);
    ret = aclrtMemcpy(grad_xResultData.data(), grad_xResultData.size() * sizeof(grad_xResultData[0]), grad_xDeviceAddr,
                      grad_xShapeSize * sizeof(grad_xResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("copy grad_x result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < grad_xShapeSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // Retrive grad_h_res
    std::vector<uint16_t> grad_h_resResultData(grad_h_resShapeSize, 0);
    ret = aclrtMemcpy(grad_h_resResultData.data(), grad_h_resResultData.size() * sizeof(grad_h_resResultData[0]), grad_h_resDeviceAddr,
                      grad_h_resShapeSize * sizeof(grad_h_resResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("copy grad_h_res result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < grad_h_resShapeSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // Retrive grad_h_out
    std::vector<uint16_t> grad_h_outResultData(grad_h_outShapeSize, 0);
    ret = aclrtMemcpy(grad_h_outResultData.data(), grad_h_outResultData.size() * sizeof(grad_h_outResultData[0]), grad_h_outDeviceAddr,
                      grad_h_outShapeSize * sizeof(grad_h_outResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("copy grad_h_out result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < grad_h_outShapeSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // Retrive grad_h_post
    std::vector<uint16_t> grad_h_postResultData(grad_h_postShapeSize, 0);
    ret = aclrtMemcpy(grad_h_postResultData.data(), grad_h_postResultData.size() * sizeof(grad_h_postResultData[0]), grad_h_postDeviceAddr,
                      grad_h_postShapeSize * sizeof(grad_h_postResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("copy grad_h_post result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < grad_h_postShapeSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release resources.
    aclDestroyTensor(grad_outputTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(h_resTensor);
    aclDestroyTensor(h_outTensor);
    aclDestroyTensor(h_postTensor);
    aclDestroyTensor(grad_xTensor);
    aclDestroyTensor(grad_h_resTensor);
    aclDestroyTensor(grad_h_outTensor);
    aclDestroyTensor(grad_h_postTensor);

    aclrtFree(grad_outputDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(h_resDeviceAddr);
    aclrtFree(h_outDeviceAddr);
    aclrtFree(h_postDeviceAddr);
    aclrtFree(grad_xDeviceAddr);
    aclrtFree(grad_h_resDeviceAddr);
    aclrtFree(grad_h_outDeviceAddr);
    aclrtFree(grad_h_postDeviceAddr);
    
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    return 0;
}
```
