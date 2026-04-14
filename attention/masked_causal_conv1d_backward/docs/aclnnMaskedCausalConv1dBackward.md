# aclnnMaskedCausalConv1dBackward

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：对hidden层的token之间进行一维分组卷积操作的反向梯度计算。

- 计算公式：
  假定卷积输入input、卷积输出的梯度grad_output和卷积输入的梯度grad_input的shape是[S, B, H]，weight的shape是[W, H]，i和j分别表示S/B轴的索引，k为卷积窗口W内的索引，那么计算将被表示为：

  $$
  grad\_output\_masked[i,j] = mask[j,i] * grad\_output[i,j]
  $$

  $$
  grad\_input[i,j] = \sum_{k=0}^{W-1} grad\_output\_masked[i+k,j] * weight[W-1-k]
  $$

  $$
  grad\_weight[k] = \sum_{j=0}^{B-1}\sum_{i=0}^{S-1} grad\_output\_masked[i+W-1-k,j] * input[i,j]
  $$

  其中，无效位置的padding为0填充；当前W仅支持3。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMaskedCausalConv1dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaskedCausalConv1dBackward”接口执行计算。

```c++
aclnnStatus aclnnMaskedCausalConv1dBackwardGetWorkspaceSize(
  const aclTensor   *grad_output,
  const aclTensor   *input,
  const aclTensor   *weight,
  const aclTensor   *maskOptional,
  const aclTensor   *grad_input,
  const aclTensor   *grad_weight,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```
```c++
aclnnStatus aclnnMaskedCausalConv1dBackward(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```
### aclnnMaskedCausalConv1dBackwardGetWorkspaceSize

- **参数说明：**

<table style="undefined;table-layout: fixed; width: 1565px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
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
      <td>grad_output</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示分组卷积输出output的梯度，对应公式中的grad_output。</td>
      <td><ul><li>不支持空Tensor。<li>shape为[S, B, H]。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示分组卷积输入，对应公式中的input。</td>
      <td><ul><li>不支持空Tensor。<li>shape和数据类型与grad_output一致。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示卷积权重，对应公式中的weight。</td>
      <td><ul><li>不支持空Tensor。<li>shape为[W, H]。<li>W目前只支持3。<li>数据类型需与grad_output一致。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>maskOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示卷积操作的输出掩码，对应公式中的mask。</td>
      <td><ul><li>shape为[B, S]。<li>可选输入，默认值是None。</ul></td>
      <td>BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>Device侧的aclTensor，表示分组卷积输入input的梯度，对应公式中的grad_input。</td>
      <td><ul><li>不支持空Tensor。<li>shape和数据类型与grad_output一致。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>grad_weight</td>
      <td>输出</td>
      <td>Device侧的aclTensor，表示分组卷积输入weight的梯度，对应公式中的grad_weight。</td>
      <td><ul><li>不支持空Tensor。<li>shape为[W, H]。<li>W目前只支持3。<li>数据类型需与grad_output一致。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>x</td>
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
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>


- **返回值：**

返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
<table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
<col style="width: 319px">
<col style="width: 144px">
<col style="width: 671px">
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
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
  </tr>
  <tr>
    <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3">161002</td>
    <td>grad_output、input、weight、mask、grad_input和grad_weight的数据类型不在支持的范围内。</td>
  </tr>
  <tr>
    <td>grad_output、input、weight、mask、grad_input和grad_weight的数据格式不在支持的范围内。</td>
  </tr>
</tbody>
</table>


### aclnnMaskedCausalConv1dBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMaskedCausalConv1dBackwardGetWorkspaceSize获取。</td>
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

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 确定性计算：
  - aclnnMaskedCausalConv1dBackward默认确定性实现。
- 输入限制：
  - grad_output、input、weight、grad_input和grad_weight的数据类型必须一致。
  - 输入输出的shape数据范围约束如下：
    - B * S（Batchsize * SeqLength）：取值范围为1\~512K。
    - H（hiddenSize）：取值范围192 * 2\~192 * 128，且是64的倍数。
    - W：W当前只支持3。


## 调用示例

通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_masked_causal_conv1d_backward.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. 固定写法，device/stream初始化, 参考acl API
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    int64_t S = 1;
    int64_t B = 2;
    int64_t H = 192 * 2;
    int64_t W = 3;

    std::vector<int64_t> gradOutputShape = {S, B, H};
    std::vector<int64_t> inputShape = {S, B, H};
    std::vector<int64_t> weightShape = {W, H};
    std::vector<int64_t> maskShape = {B, S};
    std::vector<int64_t> gradInputShape = {S, B, H};
    std::vector<int64_t> gradWeightShape = {B, S};

    std::vector<int16_t> gradOutputHostData(S * B * H, 0);
    std::vector<int16_t> inputHostData(S * B * H, 0);
    std::vector<int16_t> weightHostData(W * H, 0);
    std::vector<int16_t> maskHostData(B * S, 0);
    std::vector<int16_t> gradInputHostData(S * B * H, 0);
    std::vector<int16_t> gradWeightHostData(W * H, 0);

    void* gradOutputDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* maskDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;

    aclTensor* gradOutput = nullptr;
    aclTensor* input = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* mask = nullptr;
    aclTensor* gradInput = nullptr;
    aclTensor* gradWeight = nullptr;

    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT16, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_BOOL, &mask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT16, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT16, &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnMaskedCausalConv1dBackward第一段接口
    ret = aclnnMaskedCausalConv1dBackwardGetWorkspaceSize(
        gradOutput, input, weight, mask, gradInput, gradWeight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedCausalConv1dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnMaskedCausalConv1dBackward第二段接口
    ret = aclnnMaskedCausalConv1dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedCausalConv1dBackward failed. ERROR: %d\n", ret); return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(gradInputShape, &gradInputDeviceAddr);
    PrintOutResult(gradWeightShape, &gradWeightDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(mask);
    aclDestroyTensor(gradInput);
    aclDestroyTensor(gradWeight);

    // 7. 释放device 资源
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(maskDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```