# aclnnMoeTutelCombineX

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

- 接口功能：完成 Mixture-of-Experts（MoE）模型反向传播中的 Combine 散射计算。算子按路由元数据（专家索引 `indices` 与专家 buffer 内位置 `locations`）从专家输出梯度 `y_grad` 中取出对应行，乘以门控权重 `gates` 后作为原始 token 位置的输入梯度 `x_grad`（即 `out`）输出。支持无 batch（`y_grad` 为 2 维）与带 batch（`y_grad` 为 3 维）两种张量形态。
- 计算公式：
  - 无 batch 场景：`y_grad` 形状为 `[E*C, H]`，`gates`/`indices`/`locations` 形状为 `[S]`，`out`（`x_grad`）形状为 `[S, H]`。对每个 token `t`，设 `idx0 = indices[t]`、`idx1 = locations[t]`、`srcRow = idx0 * C + idx1`：

    $$
    x\_grad[t, h] =
    \begin{cases}
    gates[t] \times y\_grad[srcRow, h], & \text{if } idx0 \ge 0 \text{ and } idx1 < C \text{ and } srcRow < E \cdot C \\
    0, & \text{otherwise}
    \end{cases}
    $$

  - 带 batch 场景：`y_grad` 形状为 `[B, E*C, H]`，`gates`/`indices`/`locations` 形状为 `[B, S]`，`out`（`x_grad`）形状为 `[B, S, H]`。对每个 batch `b`、每个 token `t` 按同一规则计算。
  - 其中 `E` 为路由覆盖的专家数（`E = max(indices) + 1`），`C` 为每个专家 buffer 的容量（`C = eC / E`，`eC` 为 `y_grad` 在无 batch 时的行数或带 batch 时的第二维长度），`H` 为隐藏维度（`y_grad` 最后一维），`S` 为 token 数（`gates` 最后一维），`B` 为 batch 数。
- 计算规则补充说明：
  - `out` 的每个输出行只由对应 token 的一条路由条目决定：有效条目写 `gates[t] * y_grad[srcRow, :]`，无效条目（`indices[t] < 0`、`locations[t] >= C` 或 `srcRow` 越出 `y_grad` 行范围）输出全零行，因此调用前无需对 `out` 预置初值。
  - 空 Tensor（任一允许维度为 0，如 `S=0`、`eC=0` 或 `H=0`）走空 Tensor 快速路径，`out` 被清零或保持为空，调用正常返回。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnMoeTutelCombineXGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTutelCombineX”接口执行计算。

```cpp
aclnnStatus aclnnMoeTutelCombineXGetWorkspaceSize(
  const aclTensor *yGrad,
  const aclTensor *gates,
  const aclTensor *indices,
  const aclTensor *locations,
  const aclTensor *out,
  uint64_t *workspaceSize,
  aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnMoeTutelCombineX(
  void *workspace,
  uint64_t workspaceSize,
  aclOpExecutor *executor,
  aclrtStream stream)
```

## aclnnMoeTutelCombineXGetWorkspaceSize

- **参数说明：**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
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
      <td>yGrad（const aclTensor *）</td>
      <td>输入</td>
      <td>表示专家 buffer 输出的梯度，即公式中的 y_grad。</td>
      <td><li>支持空Tensor。</li><li>数据类型与 gates、out 保持一致。</li><li>要求数据连续存储（strides 与 shape 一致）。</li></td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>2维 [E*C, H] 或 3维 [B, E*C, H]，其中最后一维 H 与 out 的最后一维一致。</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gates（const aclTensor *）</td>
      <td>输入</td>
      <td>表示每个 token 副本的路由权重（门控值），即公式中的 gates。</td>
      <td><li>支持空Tensor。</li><li>数据类型与 yGrad、out 保持一致。</li><li>shape 需与 indices、locations 一致，并满足 out 的 shape 等于 gates 的 shape 拼接 yGrad 最后一维。</li><li>要求数据连续存储（strides 与 shape 一致）。</li></td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>1维 [S] 或 2维 [B, S]，其中 S 为 token 数。</td>
      <td>×</td>
    </tr>
    <tr>
      <td>indices（const aclTensor *）</td>
      <td>输入</td>
      <td>表示每个 token 副本被分配到的专家 ID，即公式中的 indices。</td>
      <td><li>支持空Tensor。</li><li>shape 需与 gates 一致。</li><li>取值小于 0 的条目视为无效，对应输出行置零。</li><li>要求数据连续存储（strides 与 shape 一致）。</li></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维 [S] 或 2维 [B, S]，shape 与 gates 一致。</td>
      <td>×</td>
    </tr>
    <tr>
      <td>locations（const aclTensor *）</td>
      <td>输入</td>
      <td>表示每个 token 副本在其分配专家 buffer 内的位置偏移，即公式中的 locations。</td>
      <td><li>支持空Tensor。</li><li>shape 需与 gates 一致。</li><li>取值大于等于 C 的条目视为无效，对应输出行置零。</li><li>要求数据连续存储（strides 与 shape 一致）。</li></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维 [S] 或 2维 [B, S]，shape 与 gates 一致。</td>
      <td>×</td>
    </tr>
    <tr>
      <td>out（const aclTensor *）</td>
      <td>输出</td>
      <td>表示原始 token 位置的输入梯度 x_grad，即公式中的 x_grad。</td>
      <td><li>支持空Tensor。</li><li>数据类型与 yGrad、gates 保持一致。</li><li>输出每一行由算子按公式覆盖写（有效条目写乘积结果、无效条目写全零），调用前无需预置初值。</li><li>要求数据连续存储（strides 与 shape 一致）。</li></td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>2维 [S, H] 或 3维 [B, S, H]，等于 gates 的 shape 拼接 yGrad 的最后一维。</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t *）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor **）</td>
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

  aclnnStatus：返回状态码，具体参见aclnn返回码。

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>yGrad、gates、indices、locations、out 任一为 NULL，或 executor 输出指针为 NULL。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入 Tensor 的数据类型组合不在支持范围内（例如 yGrad/gates 混用不同浮点类型，或 indices/locations 数据类型不是 INT32）。</td>
    </tr>
  </tbody></table>

## aclnnMoeTutelCombineX

- **参数说明：**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。本算子不依赖额外workspace，第一段接口返回的workspaceSize为0，传NULL即可。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMoeTutelCombineXGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见aclnn返回码。

## 约束说明

- 确定性说明：aclnnMoeTutelCombineX默认确定性实现，同一输入下多次调用结果一致。

- shape 与 rank 约束（满足以下约定时语义如“功能说明”所述；用户需保证传入的 Tensor 满足这些关系）：
  - `y_grad` 的 rank 为 2 或 3，`gates`/`indices`/`locations` 的 rank 为 1 或 2，且 `rank(y_grad) = rank(gates) + 1`。
  - `gates`、`indices`、`locations` 三者 shape 一致；`out` 的 shape 等于 `gates` 的 shape 拼接 `y_grad` 最后一维。
  - 隐藏维度 `H` 取值范围为 [1, 65535]，token 数 `S` 取值范围为 [1, 9223372036854775807]。
- 数据类型约束：仅支持下表 3 种组合，`y_grad`、`gates`、`out` 三者数据类型必须一致，`indices`、`locations` 固定为 INT32。

  | y_grad | gates | indices | locations | out |
  |--------|-------|---------|-----------|-----|
  | FLOAT16 | FLOAT16 | INT32 | INT32 | FLOAT16 |
  | FLOAT | FLOAT | INT32 | INT32 | FLOAT |
  | BFLOAT16 | BFLOAT16 | INT32 | INT32 | BFLOAT16 |

- 数据格式约束：仅支持 ND 格式。
- 存储约束：传入的 Tensor 需为连续存储（strides 与 shape 一致），不支持非连续 Tensor。
- 空 Tensor 说明：shape 的任一允许维度为 0（如 `S=0`、`eC=0`、`H=0`）即视为空 Tensor，接口正常返回；空 Tensor 时算子不执行散射写，`out` 被置零或保持为空。
- 索引语义：`indices[t] < 0`、`locations[t] >= C` 或计算得到的源行越出 `y_grad` 行范围的条目为无效条目，对应输出行置零，不访问越界内存。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```cpp
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_moe_tutel_combine_x.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
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
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册，deviceId按实际填写
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出：E=2 个专家、C=4 容量、eC=8、S=4 个token、H=8
    std::vector<int64_t> yGradShape = {8, 8};   // [E*C, H]
    std::vector<int64_t> metaShape = {4};       // [S]
    std::vector<int64_t> outShape = {4, 8};     // [S, H]

    void* yGradDeviceAddr = nullptr;
    void* gatesDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* locationsDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* yGrad = nullptr;
    aclTensor* gates = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* locations = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> yGradHostData(8 * 8, 1.0f);
    for (int64_t i = 0; i < 8 * 8; i++) {
        yGradHostData[i] = static_cast<float>(i % 16);
    }
    std::vector<float> gatesHostData = {1.0f, 0.5f, 2.0f, 0.25f};
    std::vector<int32_t> indicesHostData = {0, 1, 0, 1};
    std::vector<int32_t> locationsHostData = {0, 1, 2, 3};
    std::vector<float> outHostData(4 * 8, 0.0f);

    ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT, &yGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gatesHostData, metaShape, &gatesDeviceAddr, aclDataType::ACL_FLOAT, &gates);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, metaShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(locationsHostData, metaShape, &locationsDeviceAddr, aclDataType::ACL_INT32, &locations);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用算子两段式接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 调用第一段接口
    ret = aclnnMoeTutelCombineXGetWorkspaceSize(yGrad, gates, indices, locations, out,
                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclnnMoeTutelCombineXGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, printf("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用第二段接口
    ret = aclnnMoeTutelCombineX(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclnnMoeTutelCombineX failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 将device侧结果拷贝到host侧并打印
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, printf("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        printf("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor
    aclDestroyTensor(yGrad);
    aclDestroyTensor(gates);
    aclDestroyTensor(indices);
    aclDestroyTensor(locations);
    aclDestroyTensor(out);

    // 7. 释放device资源
    aclrtFree(yGradDeviceAddr);
    aclrtFree(gatesDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(locationsDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
