# aclnnMhcPreSplitPostRes

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_pre_split_post_res)

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

- 接口功能：MhcPreSplitPostRes是MhcPre算子的一部分，负责计算mHC（Manifold-Constrained Hyper-Connections）架构中x2输入对应的post分支和comb/Res分支。具体流程为：将输入x展平并转FP32后计算RMSNorm的InvRMS因子，执行线性投影得到x_hat，然后切分为post分支和comb分支分别计算h_post和h_res。**h_res为可选输出，由phi形态决定**：phi行数为n^2+n（完整形态）时输出h_post与h_res；phi行数为n（仅post形态）时仅输出h_post（h_res传空）


- 计算公式：

  输入展平（将输入x按token展平为行向量）：

  $$
  X_{flat} = \mathrm{reshape}(x, [B,S,N \cdot D])
  $$

  RMSNorm因子计算（其中 $n = N \cdot D$）：

  $$
  inv\_rms = \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}X_{flat,i}^2 + norm\_eps}}
  $$

  线性投影与归一化融合：

  $$
  x\_hat = \left(X_{flat} \cdot phi^{T}\right) \odot inv\_rms
  $$

  post/comb分支切分：

  $$
  [x\_post, x\_comb] = \mathrm{split}(x\_hat, [N, N^2])
  $$

  post分支（取x_hat前N维）：

  $$
  h\_post = 2 \cdot \sigma(\alpha_0 \cdot x\_post + b_{post})
  $$

  comb/Res分支（取x_hat后N·N维）：

  $$
  h\_res = \alpha_1 \cdot \mathrm{reshape}(x\_comb, [B,S,N,N]) + \mathrm{reshape}(b_{comb}, [N,N])
  $$

  其中 $\sigma$ 为Sigmoid函数，$b_{post} = bias[0:N]$，$b_{comb} = bias[N:N^2+N]$。

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnMhcPreSplitPostResGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMhcPreSplitPostRes`接口执行计算。

```c++
aclnnStatus aclnnMhcPreSplitPostResGetWorkspaceSize(
    const aclTensor  *x,
    const aclTensor  *phi,
    const aclTensor  *alpha,
    const aclTensor  *bias,
    float             normEps,
    const aclTensor  *hPostOut,
    const aclTensor  *hResOut,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnMhcPreSplitPostRes(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnMhcPreSplitPostResGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1470px"><colgroup>
  <col style="width: 145px">
  <col style="width: 120px">
  <col style="width: 441px">
  <col style="width: 158px">
  <col style="width: 186px">
  <col style="width: 120px">
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
      <td>x</td>
      <td>输入</td>
      <td>待计算的数据，表示网络中mHC层的输入数据x2。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,n,D]、[T,n,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>phi</td>
      <td>输入</td>
      <td>mHC的变换矩阵phi（线性投影矩阵）。两种形态：[n^2+n, n·D]（完整，输出h_res）或[n, n·D]（仅post，无h_res）。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[n²+n, n·D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>mHC的缩放参数，对应post/comb两分支的缩放系数。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>mHC的bias参数，对应post/comb两分支的偏置。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[n²+n]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normEps</td>
      <td>输入</td>
      <td>RMSNorm防除0参数（数值稳定项）。</td>
      <td>必须大于0，默认值为1e-6。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hPostOut</td>
      <td>输出</td>
      <td>mHC的h_post变换矩阵。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n]、[T,n]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hResOut</td>
      <td>输出</td>
      <td>mHC的comb/Res分支输出矩阵。完整形态（phi行数为n²+n）时必须提供；仅post形态（phi行数为n）时可传nullptr或0元素Tensor。</td>
      <td>完整形态不支持空Tensor；仅post形态允许0元素Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n,n]、[T,n,n]</td>
      <td>-</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
              <td>必选参数或者输出是空指针。</td>
          </tr>
          <tr>
              <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
              <td rowspan="3">161002</td>
              <td>x、phi、alpha、bias、hPostOut、hResOut的数据类型不在支持的范围内。</td>
          </tr>
          <tr>
              <td>x、phi、alpha、bias、hPostOut、hResOut的shape维度不在支持的范围内（如n不等于4、D不在白名单{2560,5120,6144,7168}、phi行数非n或n^2+n、完整形态下hResOut为空、仅post形态下hResOut传入非0元素Tensor）。</td>
          </tr>
          <tr>
              <td>normEps不大于0。</td>
          </tr>
          <tr>
              <td>ACLNN_ERR_INNER_TILING_ERROR</td>
              <td>561002</td>
              <td>Tiling计算或内部执行流程构建失败。</td>
          </tr>
      </tbody>
  </table>

## aclnnMhcPreSplitPostRes

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 969px"><colgroup>
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
      <td>在Device侧申请的workspace内存地址，大小为0时允许为空。</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPreSplitPostResGetWorkspaceSize获取。</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的stream流。</td>
      </tr>
  </tbody>
  </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- aclnnMhcPreSplitPostRes默认确定性实现。
- 所有输入/输出Tensor的数据格式仅支持ND。
- 输入Tensor不支持空Tensor；输出h_res在仅post形态（phi行数为n）时为0元素Tensor（允许）。

  | 规格项   | 规格               | 规格说明                                |
  | :------- | :----------------- | :------------------------------------- |
  | T或B*S   | 1及以上             | B*S或T至少为1，当前实现不设置上限               |
  | n        | 4                  | 固定为4                                |
  | D        | {2560,5120,6144,7168} | 仅支持4种典型值，其余拦截              |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_split_post_res.h"
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
    // Fixed writing method, resource initialization.
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

void PrintOutResult(const std::vector<int64_t> &shape, void *deviceAddr, const char *name) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0.0f);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return;
    }
    LOG_PRINT("%s result (first 10 elements):\n", name);
    for (int64_t i = 0; i < min(size, static_cast<int64_t>(10)); i++) {
        LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
    }
}

} // namespace

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to the API's external interface list.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    // Example: TND format (T, n, D)
    std::vector<int64_t> xShape = {1024, 4, 2560};      // TND
    std::vector<int64_t> phiShape = {20, 10240};        // 完整形态n^2+n、n*D（仅post形态取{4, 10240}）
    std::vector<int64_t> alphaShape = {2};              // [alpha0, alpha1]
    std::vector<int64_t> biasShape = {20};              // n^2+n
    std::vector<int64_t> hPostOutShape = {1024, 4};     // Tn
    std::vector<int64_t> hResOutShape = {1024, 4, 4};   // Tnn

    void *xDeviceAddr = nullptr;
    void *phiDeviceAddr = nullptr;
    void *alphaDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *hPostOutDeviceAddr = nullptr;
    void *hResOutDeviceAddr = nullptr;

    aclTensor *xTensor = nullptr;
    aclTensor *phiTensor = nullptr;
    aclTensor *alphaTensor = nullptr;
    aclTensor *biasTensor = nullptr;
    aclTensor *hPostOutTensor = nullptr;
    aclTensor *hResOutTensor = nullptr;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t phiShapeSize = GetShapeSize(phiShape);
    int64_t alphaShapeSize = GetShapeSize(alphaShape);
    int64_t biasShapeSize = GetShapeSize(biasShape);
    int64_t hPostOutShapeSize = GetShapeSize(hPostOutShape);
    int64_t hResOutShapeSize = GetShapeSize(hResOutShape);

    std::vector<short> xHostData(xShapeSize, 1);
    std::vector<float> phiHostData(phiShapeSize, 1.0f);
    std::vector<float> alphaHostData(alphaShapeSize, 1.0f);
    std::vector<float> biasHostData(biasShapeSize, 1.0f);
    std::vector<float> hPostOutHostData(hPostOutShapeSize, 0.0f);
    std::vector<float> hResOutHostData(hResOutShapeSize, 0.0f);

    // Create x aclTensor (BF16).
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &xTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create phi aclTensor.
    ret = CreateAclTensor(phiHostData, phiShape, &phiDeviceAddr, aclDataType::ACL_FLOAT, &phiTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create alpha aclTensor.
    ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alphaTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create bias aclTensor.
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &biasTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hPostOut aclTensor.
    ret = CreateAclTensor(hPostOutHostData, hPostOutShape, &hPostOutDeviceAddr, aclDataType::ACL_FLOAT,
                          &hPostOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hResOut aclTensor.
    ret = CreateAclTensor(hResOutHostData, hResOutShape, &hResOutDeviceAddr, aclDataType::ACL_FLOAT,
                          &hResOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    float normEps = 1e-6f;

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnMhcPreSplitPostResGetWorkspaceSize(
        xTensor, phiTensor, alphaTensor, biasTensor, normEps, hPostOutTensor, hResOutTensor,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPreSplitPostResGetWorkspaceSize failed. ERROR: %d\n", ret);
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
    ret = aclnnMhcPreSplitPostRes(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPreSplitPostRes failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. Retrieve the output value, copy the result from the device side memory to the host side.
    PrintOutResult(hPostOutShape, hPostOutDeviceAddr, "hPostOut");
    PrintOutResult(hResOutShape, hResOutDeviceAddr, "hResOut");

    // 6. Release resources.
    aclDestroyTensor(xTensor);
    aclDestroyTensor(phiTensor);
    aclDestroyTensor(alphaTensor);
    aclDestroyTensor(biasTensor);
    aclDestroyTensor(hPostOutTensor);
    aclDestroyTensor(hResOutTensor);
    aclrtFree(xDeviceAddr);
    aclrtFree(phiDeviceAddr);
    aclrtFree(alphaDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(hPostOutDeviceAddr);
    aclrtFree(hResOutDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
