# aclnnAiInfraApplyRotaryPosEmbGrad

## 产品支持情况

<!-- npu="950" id1 -->

- <term>Ascend 950PR/Ascend 950DT</term>：不支持

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持

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

- 接口功能：aclnnAiInfraApplyRotaryPosEmbGrad 是 aclnnAiInfraApplyRotaryPosEmb 的反向算子，用于计算旋转位置编码（RoPE）反向传播时对输入 query、key 以及 cos、sin 的梯度。
- 计算公式：当 rotary_mode="half"，将输入沿最后一个维度均分为两部分，相关符号说明如下。

  - $grad\_q\_embed$ / $grad\_k\_embed$：来自 $q\_embed$ / $k\_embed$ 输出端的上游梯度。
  - $grad_{q_1}, grad_{q_2}$ / $grad_{k_1}, grad_{k_2}$：$grad\_q\_embed$ / $grad\_k\_embed$ 沿最后维度均分的两半梯度。
  - $c_1, c_2$ / $s_1, s_2$：$cos$ / $sin$ 沿最后维度均分的两半值。
  - $q_1, q_2$ / $k_1, k_2$：$query$ / $key$ 沿最后维度均分的两半输入。
  - $query\_rotate$ / $key\_rotate$：正向旋转构造的中间张量。
  - $grad\_query$ / $grad\_key$：反向计算得到的完整 Query / Key 梯度。
  - $grad\_cos$ / $grad\_sin$：反向计算得到的完整 Cos / Sin 梯度。
  - $*$：逐元素乘法。
  - $\mathrm{chunk}(\cdot, \mathrm{chunks}=2, \mathrm{dim}=-1)$：沿最后一个维度将张量均分为两部分。
  - $\mathrm{cat}(\cdot, \mathrm{dim}=-1)$：沿最后一个维度拼接张量。
  - $\mathrm{sum}(\cdot, \mathrm{dims})$：对指定的广播维度集合执行求和，将梯度形状还原至正向输入 $cos$ / $sin$ 的原始形状；$\mathrm{dims}$ 为正向计算中 $cos$ / $sin$ 相对于 $query$ / $key$ 的广播维度集合。

  中间量计算：

  $$
  \begin{aligned}
  grad_{q_1}, grad_{q_2} &= \mathrm{chunk}(grad\_q\_embed, \mathrm{chunks}=2, \mathrm{dim}=-1) \\
  grad_{k_1}, grad_{k_2} &= \mathrm{chunk}(grad\_k\_embed, \mathrm{chunks}=2, \mathrm{dim}=-1) \\
  c_1, c_2 &= \mathrm{chunk}(\cos, \mathrm{chunks}=2, \mathrm{dim}=-1) \\
  s_1, s_2 &= \mathrm{chunk}(\sin, \mathrm{chunks}=2, \mathrm{dim}=-1) \\
  \text{query\_rotate} &= \mathrm{cat}(-q_2,\ q_1,\ \mathrm{dim}=-1) \\
  \text{key\_rotate} &= \mathrm{cat}(-k_2,\ k_1,\ \mathrm{dim}=-1)
  \end{aligned}
  $$

  输出计算：

  $$
  \begin{aligned}
  grad\_{query} &= \mathrm{cat}\big(c_1 * grad_{q_1} + s_2 * grad_{q_2},\ c_2 * grad_{q_2} - s_1 * grad_{q_1},\ \mathrm{dim}=-1\big) \\
  grad\_{key} &= \mathrm{cat}\big(c_1 * grad_{k_1} + s_2 * grad_{k_2},\ c_2 * grad_{k_2} - s_1 * grad_{k_1},\ \mathrm{dim}=-1\big) \\
  grad\_cos &= \mathrm{sum}(grad\_q\_embed * \text{query} + grad\_k\_embed * \text{key},\ \mathrm{dims}) \\
  grad\_sin &= \mathrm{sum}(grad\_q\_embed * \text{query\_rotate} + grad\_k\_embed * \text{key\_rotate},\ \mathrm{dims})
  \end{aligned}
  $$

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnAiInfraApplyRotaryPosEmbGradGetWorkspaceSize”接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用“aclnnAiInfraApplyRotaryPosEmbGrad”接口执行计算。

```cpp
aclnnStatus aclnnAiInfraApplyRotaryPosEmbGradGetWorkspaceSize(
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

```cpp
aclnnStatus aclnnAiInfraApplyRotaryPosEmbGrad(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnAiInfraApplyRotaryPosEmbGradGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
   <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>gradQueryEmbed（aclTensor*）</td>
      <td>输入</td>
      <td>正向算子输出 query_embed 的梯度，对应公式中 grad_q_embed。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape 需为 4D 或 3D，layout 为 BSND/SBND 时形状为 [B, S, N, D] / [S, B, N, D]，layout 为 TND 时形状为 [T, N, D]。</li><li>最后一维 D 需满足 16 ≤ D ≤ 128 且 D 为 16 的倍数。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradKeyEmbed（aclTensor*）</td>
      <td>输入</td>
      <td>正向算子输出 key_embed 的梯度，对应公式中 grad_k_embed。</td>
      <td><ul><li>不支持空Tensor。</li><li>维度数需与 gradQueryEmbed 一致；除 N 维度可不同外，其余维度需与 gradQueryEmbed 一致。</li><li>数据类型需与 gradQueryEmbed 一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos（aclTensor*）</td>
      <td>输入</td>
      <td>正向算子输入 cos。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape 需与 gradQueryEmbed 一致，但 N 维度必须为 1。</li><li>数据类型需与 gradQueryEmbed 一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin（aclTensor*）</td>
      <td>输入</td>
      <td>正向算子输入 sin。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape 需与 gradQueryEmbed 一致，但 N 维度必须为 1。</li><li>数据类型需与 gradQueryEmbed 一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>queryOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>正向算子输入 query，用于计算 grad_cos 与 grad_sin。</td>
      <td><ul><li>传空指针时不计算 gradCosOutOptional 与 gradSinOutOptional</li><li>非空时 shape 需与 gradQueryEmbed 一致，数据类型需与 gradQueryEmbed 一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>正向算子输入 key，用于计算 grad_cos 与 grad_sin。</td>
      <td><ul><li>传空指针时不计算 gradCosOutOptional 与 gradSinOutOptional</li><li>非空时 shape 需与 gradKeyEmbed 一致，数据类型需与 gradQueryEmbed 一致。</li><li>queryOptional 与 keyOptional 必须同时为空或同时非空。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rotaryModeOptional（char*）</td>
      <td>输入</td>
      <td>旋转模式。</td>
      <td>当前仅支持 "half"，传入其他值会报错。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout（int64_t）</td>
      <td>输入</td>
      <td>输入 Tensor 的布局格式。</td>
      <td>1 表示 BSND，2 表示 SBND，4 表示 TND，其他取值会报错。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradQueryOut（aclTensor*）</td>
      <td>输出</td>
      <td>正向输入 query 的梯度，对应公式中的 grad_query。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape 需与 gradQueryEmbed 完全一致。</li><li>数据类型需与 gradQueryEmbed 一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradKeyOut（aclTensor*）</td>
      <td>输出</td>
      <td>正向输入 key 的梯度，对应公式中 grad_key。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape 需与 gradKeyEmbed 完全一致。</li><li>数据类型需与 gradQueryEmbed 一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradCosOutOptional（aclTensor*）</td>
      <td>可选输出</td>
      <td>正向输入 cos 的梯度。</td>
      <td><ul><li>queryOptional 与 keyOptional 同时非空时有效。</li><li>shape 需与 cos 完全一致。</li><li>数据类型需与 gradQueryEmbed 一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradSinOutOptional（aclTensor*）</td>
      <td>可选输出</td>
      <td>正向输入 sin 的梯度。</td>
      <td><ul><li>queryOptional 与 keyOptional 同时非空时有效。</li><li>shape 需与 sin 完全一致。</li><li>数据类型需与 gradQueryEmbed 一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4/3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    </tbody></table>

- **返回值**

  aclnnStatus：返回状态码。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
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
      <td>gradQueryEmbed、gradKeyEmbed、cos、sin 存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>gradQueryEmbed、gradKeyEmbed、cos、sin、queryOptional、keyOptional 的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradQueryEmbed、gradKeyEmbed、cos、sin、queryOptional、keyOptional 的数据类型不匹配。</td>
    </tr>
    <tr>
      <td>gradQueryEmbed、gradKeyEmbed、cos、sin 的 shape 维度不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradQueryEmbed、gradKeyEmbed、cos、sin 的 shape 维度不匹配。</td>
    </tr>
    <tr>
      <td>queryOptional 与 keyOptional 仅其中一个为空指针。</td>
    </tr>
    <tr>
      <td>rotaryModeOptional 取值非法或 layout 取值非法。</td>
    </tr>
    <tr>
      <td>输出 shape 与对应输入 shape 不一致或输出数据类型与输入数据类型不一致。</td>
    </tr>
  </tbody></table>

## aclnnAiInfraApplyRotaryPosEmbGrad

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnAiInfraApplyRotaryPosEmbGradGetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码。

## 约束说明

- 确定性说明：

  aclnnAiInfraApplyRotaryPosEmbGrad 默认确定性实现。

- 输入 shape 限制：

  - queryOptional 与 keyOptional 必须同时为空或同时非空。


## 调用示例

示例代码如下，仅供参考。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_apply_rotary_pos_emb_grad.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用 aclCreateTensor 接口创建 aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1.（固定写法）device/stream 初始化，参考 acl API 手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据 API 的接口自定义构造
    std::vector<int64_t> gradQueryShape = {2, 4, 8, 32};  // BSND layout
    std::vector<int64_t> gradKeyShape = {2, 4, 1, 32};
    std::vector<int64_t> cosSinShape = {2, 4, 1, 32};
    std::vector<int64_t> queryShape = {2, 4, 8, 32};
    std::vector<int64_t> keyShape = {2, 4, 1, 32};

    void* gradQueryDeviceAddr = nullptr;
    void* gradKeyDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* gradQueryOutDeviceAddr = nullptr;
    void* gradKeyOutDeviceAddr = nullptr;
    void* gradcosOutDeviceAddr = nullptr;
    void* gradsinOutDeviceAddr = nullptr;

    aclTensor* gradQuery = nullptr;
    aclTensor* gradKey = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* gradQueryOut = nullptr;
    aclTensor* gradKeyOut = nullptr;
    aclTensor* gradcosOut = nullptr;
    aclTensor* gradsinOut = nullptr;

    int64_t gradQuerySize = GetShapeSize(gradQueryShape);
    int64_t gradKeySize = GetShapeSize(gradKeyShape);
    int64_t cosSinSize = GetShapeSize(cosSinShape);
    int64_t querySize = GetShapeSize(queryShape);
    int64_t keySize = GetShapeSize(keyShape);

    std::vector<float> gradQueryHostData(gradQuerySize, 1.0f);
    std::vector<float> gradKeyHostData(gradKeySize, 1.0f);
    std::vector<float> cosHostData(cosSinSize, 1.0f);
    std::vector<float> sinHostData(cosSinSize, 0.0f);
    std::vector<float> queryHostData(querySize, 1.0f);
    std::vector<float> keyHostData(keySize, 1.0f);
    std::vector<float> gradQueryOutHostData(gradQuerySize, 0.0f);
    std::vector<float> gradKeyOutHostData(gradKeySize, 0.0f);
    std::vector<float> gradcosOutHostData(cosSinSize, 0.0f);
    std::vector<float> gradsinOutHostData(cosSinSize, 0.0f);

    ret = CreateAclTensor(gradQueryHostData, gradQueryShape, &gradQueryDeviceAddr, aclDataType::ACL_FLOAT, &gradQuery);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradKeyHostData, gradKeyShape, &gradKeyDeviceAddr, aclDataType::ACL_FLOAT, &gradKey);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosHostData, cosSinShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinHostData, cosSinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradQueryOutHostData, gradQueryShape, &gradQueryOutDeviceAddr, aclDataType::ACL_FLOAT, &gradQueryOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradKeyOutHostData, gradKeyShape, &gradKeyOutDeviceAddr, aclDataType::ACL_FLOAT, &gradKeyOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradcosOutHostData, cosSinShape, &gradcosOutDeviceAddr, aclDataType::ACL_FLOAT, &gradcosOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradsinOutHostData, cosSinShape, &gradsinOutDeviceAddr, aclDataType::ACL_FLOAT, &gradsinOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 CANN 算子库 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    char* rotaryMode = "half";
    int64_t layout = 1;  // BSND

    ret = aclnnAiInfraApplyRotaryPosEmbGradGetWorkspaceSize(
        gradQuery, gradKey, cos, sin, query, key, rotaryMode, layout,
        gradQueryOut, gradKeyOut, gradcosOut, gradsinOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAiInfraApplyRotaryPosEmbGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的 workspaceSize 申请 device 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用第二段接口
    ret = aclnnAiInfraApplyRotaryPosEmbGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAiInfraApplyRotaryPosEmbGrad failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将 device 侧内存上的结果拷贝至 host 侧
    std::vector<float> resultData(gradQuerySize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradQueryOutDeviceAddr,
                      gradQuerySize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < gradQuerySize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放 aclTensor
    aclDestroyTensor(gradQuery);
    aclDestroyTensor(gradKey);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(gradQueryOut);
    aclDestroyTensor(gradKeyOut);
    aclDestroyTensor(gradcosOut);
    aclDestroyTensor(gradsinOut);

    // 7. 释放 device 资源
    aclrtFree(gradQueryDeviceAddr);
    aclrtFree(gradKeyDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(gradQueryOutDeviceAddr);
    aclrtFree(gradKeyOutDeviceAddr);
    aclrtFree(gradcosOutDeviceAddr);
    aclrtFree(gradsinOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
