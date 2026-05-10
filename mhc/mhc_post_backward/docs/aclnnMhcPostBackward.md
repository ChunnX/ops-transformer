# aclnnMhcPostBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：mhc_post基于一系列计算对MHC（Manifold-Constrained Hyper-Connection）架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_j$进行ResMapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。该算子实现前述过程的反向功能。
  
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

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMhcPostBackwardGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnMhcPostBackward”接口执行计算。

```c++
aclnnStatus aclnnMhcPostBackwardGetWorkspaceSize(
    const aclTensor     *gradOutput,
    const aclTensor     *x, 
    const aclTensor     *hRes, 
    const aclTensor     *hOut, 
    const aclTensor     *hPost, 
    aclTensor           *gradX, 
    aclTensor           *gradHres, 
    aclTensor           *gradHout, 
    aclTensor           *gradHpost,
    uint64_t            *workspaceSize, 
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnMhcPostBackward(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    const aclrtStream stream)
```

## aclnnMhcPostBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width:  1400px"><colgroup>
  <col style="width: 145px">
  <col style="width: 90px">
  <col style="width: 441px">
  <col style="width: 158px">
  <col style="width: 186px">
  <col style="width: 80px">
  <col style="width: 155px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>待计算的数据，表示网络中MHC层的输入数据</td>
      <td>
        <ul>      
          <li>不支持空Tensor</li>
        <ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,N,D]、[T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待计算的数据，表示网络中MHC层的输入数据</td>
      <td>
        <ul>      
          <li>不支持空Tensor</li>
        <ul>
      </td>
      <td>数据类型与gradOutput一致</td>
      <td>ND</td>
      <td>[B,S,N,D]、[T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hRes</td>
      <td>输入</td>
      <td>MHC的hRes变换矩阵</td>
      <td>
        <ul>      
          <li>不支持空Tensor</li>
        <ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,N,N]、[T,N,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hOut</td>
      <td>输入</td>
      <td>Atten/MLP层的输出</td>
      <td>
        <ul>      
          <li>不支持空Tensor</li>
        <ul>
      </td>
      <td>数据类型与gradOutput一致</td>
      <td>ND</td>
      <td>[B,S,D]、[T,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hPost</td>
      <td>输入</td>
      <td>MHC的hPost变换矩阵</td>
      <td>
        <ul>      
          <li>不支持空Tensor</li>
        <ul>
      </td>
      <td>数据类型与hRes一致</td>
      <td>ND</td>
      <td>[B,S,N]、[T,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradX</td>
      <td>输出</td>
      <td>网络中MHC层的输入数据x的梯度</td>
      <td>-</td>
      <td>数据类型与gradOutput一致</td>
      <td>ND</td>
      <td>[B,S,N,D]、[T,N,D]</td>
      <td>√</td>
      </tr>
    </tr>
    <tr>
      <td>gradHRes</td>
      <td>输出</td>
      <td>网络中MHC层的输入数据hRes的梯度</td>
      <td>-</td>
      <td>数据类型与hRes一致</td>
      <td>ND</td>
      <td>[B,S,N,N]、[T,N,N]</td>
      <td>√</td>
      </tr>
    </tr>
    <tr>
      <td>gradHout</td>
      <td>输出</td>
      <td>网络中MHC层的输入数据hOut的梯度</td>
      <td>-</td>
      <td>数据类型与hOut一致</td>
      <td>ND</td>
      <td>[B,S,D]、[T,D]</td>
      <td>√</td>
      </tr>
    </tr>
    <tr>
      <td>gradHpost</td>
      <td>输出</td>
      <td>网络中MHC层的输入数据h_post的梯度</td>
      <td>-</td>
      <td>数据类型与hPost一致</td>
      <td>ND</td>
      <td>[B,S,N]、[T,N]</td>
      <td>√</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1000px"><colgroup>
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
          <td>gradOutput、x、hRes、hOut、hPost存在空指针。</td>
        </tr>
        <tr>
          <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
          <td rowspan="3">161002</td>
          <td>gradOutput、x、hRes、hOut、hPost的数据类型不在支持的范围内。</td>
        </tr>
          <tr>
          <td>gradOutput、x、hRes、hOut、hPost的shape维度不在支持的范围内。</td>
        </tr>
        <tr>
          <td>gradOutput、x、hRes、hOut、hPost的数据类型或shape不匹配。</td>
        </tr>
  </tbody></table>


## aclnnMhcPostBackwardGrad

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPostBacwardGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
参数说明中维度N的取值目前仅支持4、6和8。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_post_backward.h"
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
    ret = aclnnMhcPostBackwardGetWorkspaceSize(
        grad_outputTensor, xTensor, h_resTensor, h_outTensor, h_postTensor,
        grad_xTensor, grad_h_resTensor, grad_h_outTensor, grad_h_postTensor,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
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
    ret = aclnnMhcPostBackward(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackward failed. ERROR: %d\n", ret);
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
