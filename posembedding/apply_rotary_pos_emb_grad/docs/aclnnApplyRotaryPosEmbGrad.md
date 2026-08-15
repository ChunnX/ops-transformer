# aclnnApplyRotaryPosEmbGrad

[查看源码](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/apply_rotary_pos_emb_grad)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：执行双路旋转位置编码[aclnnApplyRotaryPosEmb](../../apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmb.md)的反向计算。同时计算 query 和 key 的 rope 反向梯度，融合为一次 kernel 调用。

- 计算公式：

    - <term>Ascend 950PR/Ascend 950DT</term>：

    取旋转位置编码的正向计算中，broadcast的轴列表为`dims`，half模式计算公式如下：

    $$
    grad\_q_1, grad\_q_2 = chunk(grad\_query\_embed, chunks=2, dim=-1)
    $$

    $$
    grad\_k_1, grad\_k_2 = chunk(grad\_key\_embed, chunks=2, dim=-1)
    $$

    $$
    cos_1, cos_2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin_1, sin_2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    query\_rotate = cat((-query_2, query_1), dim=-1)
    $$

    $$
    key\_rotate = cat((-key_2, key_1), dim=-1)
    $$

    $$
    grad\_query = cat(cos_1 * grad\_q_1 + sin_2 * grad\_q_2, cos_2 * grad\_q_2 - sin_1 * grad\_q_1, dim=-1)
    $$

    $$
    grad\_key = cat(cos_1 * grad\_k_1 + sin_2 * grad\_k_2, cos_2 * grad\_k_2 - sin_1 * grad\_k_1, dim=-1)
    $$

    $$
    grad\_cos = sum(grad\_query\_embed * query + grad\_key\_embed * key, dims)
    $$

    $$
    grad\_sin = sum(grad\_query\_embed * query\_rotate + grad\_key\_embed * key\_rotate, dims)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnApplyRotaryPosEmbGradGetWorkspaceSize"接口获取入参并根据流程计算所需workspace大小，再调用"aclnnApplyRotaryPosEmbGrad"接口执行计算。

```c++
aclnnStatus aclnnApplyRotaryPosEmbGradGetWorkspaceSize(
    const aclTensor *gradQueryEmbed,
    const aclTensor *gradKeyEmbed,
    const aclTensor *cos,
    const aclTensor *sin,
    const aclTensor *queryOptional,
    const aclTensor *keyOptional,
    char            *rotaryModeOptional,
    int64_t          layout,
    const aclTensor *gradQueryOut,
    const aclTensor *gradKeyOut,
    const aclTensor *gradCosOutOptional,
    const aclTensor *gradSinOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnApplyRotaryPosEmbGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnApplyRotaryPosEmbGradGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 403px">
  <col style="width: 169px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
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
      <td>gradQueryEmbed</td>
      <td>输入</td>
      <td>正向输出query的导数，对应公式中grad_q_embed。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradKeyEmbed</td>
      <td>输入</td>
      <td>正向输出key的导数，对应公式中grad_k_embed。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>正向计算输入cos。</td>
      <td>与gradQueryEmbed数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>正向计算输入sin。</td>
      <td>与gradQueryEmbed数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>queryOptional</td>
      <td>可选输入</td>
      <td>正向计算输入query，空指针时不计算gradCosOutOptional和gradSinOutOptional。</td>
      <td>与gradQueryEmbed数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyOptional</td>
      <td>可选输入</td>
      <td>正向计算输入key，空指针时不计算gradCosOutOptional和gradSinOutOptional。</td>
      <td>与gradQueryEmbed数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rotaryModeOptional</td>
      <td>输入</td>
      <td>旋转模式。</td>
      <td>仅支持"half"，默认"half"。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>输入</td>
      <td>输入Tensor的布局格式。1=BSND，2=SBND，4=TND，3=BNSD（预留）。</td>
      <td>默认值为1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradQueryOut</td>
      <td>输出</td>
      <td>正向计算输入query的导数。</td>
      <td>与gradQueryEmbed数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradKeyOut</td>
      <td>输出</td>
      <td>正向计算输入key的导数。</td>
      <td>与gradQueryEmbed数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradCosOutOptional</td>
      <td>可选输出</td>
      <td>正向计算输入cos的导数，queryOptional和keyOptional非空时有效。</td>
      <td>与gradQueryEmbed数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradSinOutOptional</td>
      <td>可选输出</td>
      <td>正向计算输入sin的导数，queryOptional和keyOptional非空时有效。</td>
      <td>与gradQueryEmbed数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
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
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 288px">
  <col style="width: 125px">
  <col style="width: 742px">
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
      <td>传入的必选输入gradQueryEmbed、gradKeyEmbed、cos、sin和必选输出gradQueryOut、gradKeyOut是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>传入的输入输出数据类型和格式不在支持的范围内、shape不满足校验条件、维度不在支持的范围、或rotaryMode/layout不符合支持的值。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyRotaryPosEmbGrad

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnApplyRotaryPosEmbGradGetWorkspaceSize获取。</td>
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

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnApplyRotaryPosEmbGrad默认确定性实现。

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 输入只支持3维或4维。
  - 输入输出Tensor的dtype必须相同。
  - 输入输出Tensor的D轴必须相同，在half模式下必须≤1024且能被2整除。
  - `gradQueryEmbed`、`gradQueryOut`的shape必须相同，`gradKeyEmbed`、`gradKeyOut`的shape必须相同。
  - 对于任意`layout`，`gradQueryEmbed`和`gradKeyEmbed`除N维度外其它维度必须相同。
  - `cos`、`sin`的N维度必须等于1，`cos`、`sin`的B维度可以等于1，也可以和`gradQueryEmbed`的B维度一致，其余维度需与`gradQueryEmbed`一致。
  - `cos`、`sin`、`gradCosOutOptional`、`gradSinOutOptional`的shape必须相同。
  - `query`维度需与`gradQueryEmbed`一致，`key`维度需与`gradKeyEmbed`一致。
  - `rotaryModeOptional`仅支持"half"。
  - `layout`仅支持{1, 2, 4}，对应{BSND, SBND, TND}。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb_grad.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. 固定写法，device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> gradQEmbedShape = {1, 1, 1, 128};
    std::vector<int64_t> gradKEmbedShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    std::vector<int64_t> gradQueryOutShape = {1, 1, 1, 128};
    std::vector<int64_t> gradKeyOutShape = {1, 1, 1, 128};
    int64_t layout = 1;
    const char* rotaryModeOptional = "half";

    void* gradQEmbedDeviceAddr = nullptr;
    void* gradKEmbedDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* gradQueryOutDeviceAddr = nullptr;
    void* gradKeyOutDeviceAddr = nullptr;
    aclTensor* gradQueryEmbed = nullptr;
    aclTensor* gradKeyEmbed = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* gradQueryOut = nullptr;
    aclTensor* gradKeyOut = nullptr;
    aclTensor* gradCosOutOptional = nullptr;
    aclTensor* gradSinOutOptional = nullptr;

    std::vector<float> gradQEmbedHostData(128, 0.0f);
    std::vector<float> gradKEmbedHostData(128, 0.0f);
    std::vector<float> cosHostData(128, 0.0f);
    std::vector<float> sinHostData(128, 0.0f);
    std::vector<float> gradQueryOutHostData(128, 0.0f);
    std::vector<float> gradKeyOutHostData(128, 0.0f);

    // 创建gradQueryEmbed aclTensor
    ret = CreateAclTensor(gradQEmbedHostData, gradQEmbedShape, &gradQEmbedDeviceAddr, aclDataType::ACL_FLOAT, &gradQueryEmbed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradKeyEmbed aclTensor
    ret = CreateAclTensor(gradKEmbedHostData, gradKEmbedShape, &gradKEmbedDeviceAddr, aclDataType::ACL_FLOAT, &gradKeyEmbed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cos aclTensor
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sin aclTensor
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradQueryOut aclTensor
    ret = CreateAclTensor(gradQueryOutHostData, gradQueryOutShape, &gradQueryOutDeviceAddr, aclDataType::ACL_FLOAT, &gradQueryOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradKeyOut aclTensor
    ret = CreateAclTensor(gradKeyOutHostData, gradKeyOutShape, &gradKeyOutDeviceAddr, aclDataType::ACL_FLOAT, &gradKeyOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradCosOutOptional和gradSinOutOptional（不计算cos/sin梯度，传入空指针）
    std::vector<int64_t> emptyTensorOutShape = {1, 1, 1, 0};
    std::vector<int64_t> emptyTensorStride(emptyTensorOutShape.size(), 0);
    gradCosOutOptional = aclCreateTensor(emptyTensorOutShape.data(), emptyTensorOutShape.size(), aclDataType::ACL_FLOAT,
                                         emptyTensorStride.data(), 0, aclFormat::ACL_FORMAT_ND, emptyTensorOutShape.data(),
                                         emptyTensorOutShape.size(), nullptr);
    gradSinOutOptional = aclCreateTensor(emptyTensorOutShape.data(), emptyTensorOutShape.size(), aclDataType::ACL_FLOAT,
                                         emptyTensorStride.data(), 0, aclFormat::ACL_FORMAT_ND, emptyTensorOutShape.data(),
                                         emptyTensorOutShape.size(), nullptr);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnApplyRotaryPosEmbGrad第一段接口
    ret = aclnnApplyRotaryPosEmbGradGetWorkspaceSize(gradQueryEmbed, gradKeyEmbed, cos, sin, nullptr, nullptr,
                                                      const_cast<char*>(rotaryModeOptional), layout,
                                                      gradQueryOut, gradKeyOut, gradCosOutOptional, gradSinOutOptional,
                                                      &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnApplyRotaryPosEmbGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnApplyRotaryPosEmbGrad第二段接口
    ret = aclnnApplyRotaryPosEmbGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyRotaryPosEmbGrad failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(gradQueryOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradQueryOutDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar
    aclDestroyTensor(gradQueryEmbed);
    aclDestroyTensor(gradKeyEmbed);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(gradQueryOut);
    aclDestroyTensor(gradKeyOut);

    // 7. 释放device资源
    aclrtFree(gradQEmbedDeviceAddr);
    aclrtFree(gradKEmbedDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(gradQueryOutDeviceAddr);
    aclrtFree(gradKeyOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
