# aclnnMoeTokenPermuteWithEpGrad


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- **算子功能**：aclnnMoeTokenPermuteWithEp的反向传播计算。
- **计算公式**：
    
  - 首先计算tokenGradOut：
    - 当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时：

      $$
      tokenGradOut[i] = permutedTokensOutputGrad[sortedIndices[i]-rangeOptional[0]]
      $$

    - 否则：
      
      $$
      tokenGradOut[i] = 0
      $$

  - 接着计算：

    $$
    tokenGradOut = tokenGradOut.reshape(-1, topK, hiddenSize)
    $$

    $$
    tokenGradOut = tokenGradOut.sum(dim = 1)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize” 接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用 “aclnnMoeTokenPermuteWithEpGrad” 接口执行计算。

* `aclnnStatus aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize(const aclTensor *permutedTokensOutputGrad, const aclTensor *sortedIndices, const aclTensor *permutedProbsOutputGradOptional, int64_t numTopk, const aclIntArray *rangeOptional, bool paddedMode, const aclTensor *tokenGradOut, const aclTensor *probsGradOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermuteWithEpGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize

- **参数说明：**：
  - permutedTokensOutputGrad（aclTensor \*，计算输入）：表示正向输出permutedTokens的梯度，公式中的`permutedTokensOutputGrad`，Device侧的aclTensor。shape支持2D维度，shape为（（rangeOptional[1] - rangeOptional[0]）* topK\_num，hidden\_size），topK\_num为numTopk的值。数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
  - sortedIndices（aclTensor \*，计算输入）：表示正向输出的permuteTokensOut和正向输入的tokens的映射关系，公式中的`sortedIndices`，Device侧的aclTensor。shape支持1D维度，shape为（num\_tokens * topK\_num），num\_tokens为token数目。数据类型支持INT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持非连续输入，不支持空Tensor。
  - permutedProbsOutputGradOptional（aclTensor\*，计算输入）：表示正向输出permutedProbs的梯度Device侧的aclTensor。可选计算输入，与计算输出probsGradOut对应，传入空则不输出probsGradOut。shape支持1D维度，shape为（（rangeOptional[1] - rangeOptional[0]） * topK\_num），topK\_num为numTopk的值。数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持非连续输入。
  - numTopk（int64\_t，计算输入）：被选中的专家个数。
  - rangeOptional（aclIntArray \*，计算输入）：ep切分的有效范围，size为2。为空时，忽略permutedProbsOutputGradOptional和probsGradOut，执行逻辑回退到[aclnnMoeTokenPermuteGrad](aclnnMoeTokenPermuteGrad.md)。
  - paddedMode（bool，计算输入）：true表示开启paddedMode，false表示关闭paddedMode，目前仅支持false。
  - tokenGradOut（aclTensor \*，计算输出）：输入token的梯度，要求为一个维度为2D的Tensor，shape为（num\_tokens，hidden_size），数据类型支持FLOAT、FLOAT16、BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - probsGradOut（aclTensor \*，计算输出）：输入probs的梯度，要求为一个维度为2D的Tensor，shape为（num\_tokens，topK\_num），数据类型支持FLOAT、FLOAT16、BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - workspaceSize（uint64\_t \*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 输入和输出的Tensor是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. 输入和输出的数据类型和数据格式不在支持的范围之内。
  ```
## aclnnMoeTokenPermuteWithEpGrad

- **参数说明：**
  - workspace（void \*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize获取。
  - executor（aclOpExecutor \*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeTokenPermuteWithEpGrad默认确定性实现。

 - top_k <= 512。
 - 不支持paddedMode为`True`。
 - 当rangeOptional为空时，忽略permutedProbsOutputGradOptional和probsGradOut，执行逻辑回退到[aclnnMoeTokenPermuteGrad](aclnnMoeTokenPermuteGrad.md)。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_ep_grad.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Call aclCreateIntArray to create an aclIntArray.
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造

  int64_t num_topk = 2;
  std::vector<float> permuted_token_output_grad_Data = {2, 2, 1, 1, 3, 3, 2, 2};
  std::vector<float> permuted_prob_output_grad_Data = {0.2, 0.5, 0.4, 0.4};
  std::vector<int64_t> permuted_token_output_grad_Shape = {4, 2};
  std::vector<int64_t> permuted_prob_output_grad_Shape = {4};
  void *permuted_token_output_grad_Addr = nullptr;
  void *permuted_prob_output_grad_Addr = nullptr;
  aclTensor *permuted_token_output_grad = nullptr;
  aclTensor *permuted_prob_output_grad = nullptr;

  ret = CreateAclTensor(permuted_token_output_grad_Data, permuted_token_output_grad_Shape,
                        &permuted_token_output_grad_Addr, aclDataType::ACL_BF16,
                        &permuted_token_output_grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(permuted_prob_output_grad_Data, permuted_prob_output_grad_Shape,
                        &permuted_prob_output_grad_Addr, aclDataType::ACL_BF16,
                        &permuted_prob_output_grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int> sortedIndicesData = {2, 0, 4, 1, 5, 3};
  std::vector<int64_t> sortedIndicesShape = {6};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                        aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* rangeDeviceAddr = nullptr;
  aclIntArray* range = nullptr;
  std::vector<int64_t> rangeHostData = {1, 5};
  ret = CreateAclIntArray(rangeHostData, &rangeDeviceAddr, &range);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> tokenOutData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> tokenOutShape = {3, 2};
  void *tokenOutAddr = nullptr;
  aclTensor *tokenOut = nullptr;

  ret = CreateAclTensor(tokenOutData, tokenOutShape, &tokenOutAddr, aclDataType::ACL_BF16,
                        &tokenOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probOutData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> probOutShape = {3, 2};
  void *probOutAddr = nullptr;
  aclTensor *probOut = nullptr;

  ret = CreateAclTensor(probOutData, probOutShape, &probOutAddr, aclDataType::ACL_BF16,
                        &probOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenPermuteWithEpGrad第一段接口
  ret = aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize(permuted_token_output_grad, sortedIndices, permuted_prob_output_grad,
                                                       num_topk, range, false,
                                                       tokenOut, probOut, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnMoeTokenPermuteWithEpGrad第二段接口
  ret = aclnnMoeTokenPermuteWithEpGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenPermuteWithEpGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(tokenOutShape, &tokenOutAddr);
  PrintOutResult(probOutShape, &probOutAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permuted_token_output_grad);
  aclDestroyTensor(permuted_prob_output_grad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(tokenOut);
  aclDestroyTensor(probOut);

  // 7. 释放device资源
  aclrtFree(permuted_token_output_grad_Addr);
  aclrtFree(permuted_prob_output_grad_Addr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(tokenOutAddr);
  aclrtFree(probOutAddr);
  aclrtFree(rangeDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
