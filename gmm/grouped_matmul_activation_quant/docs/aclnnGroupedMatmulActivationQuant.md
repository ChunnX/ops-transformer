# aclnnGroupedMatmulActivationQuant

[查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_activation_quant)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | 支持 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | 不支持 |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | 不支持 |
| <term>Atlas 200I/500 A2 推理产品</term> | 不支持 |
| <term>Atlas 推理系列产品</term> | 不支持 |
| <term>Atlas 训练系列产品</term> | 不支持 |

## 功能说明

- 接口功能：融合 GroupedMatmul、activation 和 MX quant。当前版本仅支持 `activationType` 为 `"gelu_tanh"`，仅支持 MX 量化场景。

- 计算公式：

  - 定义：

    - `*` 表示矩阵乘法。
    - `GeluTanh(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`。
    - `blocksize` 表示 MX 量化共享指数的分组大小，当前仅支持 32。
    - `E` 表示专家数，`M` 表示 token 数，`K` 表示输入特征维度，`N` 表示输出特征维度。

  - 输入：

    - `X`：左矩阵，shape 为 `[M, K]`，数据类型支持 `FLOAT8_E4M3FN`、`FLOAT8_E5M2`。
    - `W`：分组权重矩阵，shape 为 `[E, K, N]` 或等价 tensorlist 表达，数据类型支持 `FLOAT8_E4M3FN`、`FLOAT8_E5M2`。
    - `x_scale`：左矩阵 MX scale，shape 为 `[M, ceil(K / 32), 2]`，数据类型支持 `FLOAT8_E8M0`。
    - `weight_scale`：权重矩阵 MX scale，shape 为 `[E, ceil(K / 32), N, 2]` 或等价 tensorlist 表达，数据类型支持 `FLOAT8_E8M0`。
    - `groupList`：分组信息，shape 为 `[E]`，数据类型支持 `INT64`。
    - `bias`：可选偏置，支持为空；非空时与专家维度对应。

  - 输出：

    - `Y`：激活并量化后的输出，shape 为 `[M, N]`，数据类型由 `quantDtype` 指定，当前支持 `FLOAT8_E4M3FN`、`FLOAT8_E5M2`。
    - `Y_scale`：输出 MX scale，shape 为 `[M, ceil(N / 32), 2]`，数据类型支持 `FLOAT8_E8M0`。

  - 计算过程：

    1. 根据 `groupList` 和 `groupListType` 确定每个专家对应的 token 范围。

    2. 对每个专家分组执行 grouped matmul 和 MX 反量化：

       $$
       C_i = (X_i * W_i) \odot x\_scale_i \odot weight\_scale_i
       $$

    3. 执行激活函数：

       $$
       S_i = GeluTanh(C_i)
       $$

    4. 对激活结果执行 MX 量化：

       $$
       shared\_exp = floor(log2(max(abs(S_i)))) - emax
       $$

       $$
       Y\_scale = 2^{shared\_exp}
       $$

       $$
       Y_i = quantize\_to\_element\_format(S_i / Y\_scale)
       $$

       其中 `emax` 由 `quantDtype` 决定：`FLOAT8_E4M3FN` 对应 8，`FLOAT8_E5M2` 对应 15。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnGroupedMatmulActivationQuantGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnGroupedMatmulActivationQuant` 接口执行计算。

```Cpp
aclnnStatus aclnnGroupedMatmulActivationQuantGetWorkspaceSize(
    const aclTensor     *x,
    const aclTensor     *groupList,
    const aclTensor     *xScaleOptional,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *biasOptional,
    const char          *activationType,
    int64_t              dequantDtype,
    bool                 transposeWeight,
    int64_t              groupListType,
    const aclIntArray   *tuningConfigOptional,
    int64_t              quantDtype,
    const char          *roundMode,
    int64_t              scaleAlg,
    double               dstTypeMax,
    aclTensor           *y,
    aclTensor           *yScale,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnGroupedMatmulActivationQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulActivationQuantGetWorkspaceSize

- **参数说明：**

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `x` | 输入 | 左矩阵。 | 必选参数。 | `FLOAT8_E4M3FN`、`FLOAT8_E5M2` | ND | 2维，`[M, K]` |
| `groupList` | 输入 | 分组信息。 | 必选参数。根据 `groupListType` 表示 cumsum 或 count。 | `INT64` | ND | 1维，`[E]` |
| `xScaleOptional` | 输入 | `x` 的 MX scale。 | 可选参数。当前 MX 量化场景下需要传入。 | `FLOAT8_E8M0` | ND | 3维，`[M, ceil(K / 32), 2]` |
| `weight` | 输入 | 分组权重矩阵。 | 动态输入，可以为一个 tensor，也可以为 tensorlist。当前仅支持 ND。 | `FLOAT8_E4M3FN`、`FLOAT8_E5M2` | ND | 单 tensor 为3维，`[E, K, N]`；tensorlist 时每个 tensor 表示一个或多个专家权重 |
| `weightScale` | 输入 | `weight` 的 MX scale。 | 动态输入，可以为一个 tensor，也可以为 tensorlist。 | `FLOAT8_E8M0` | ND | 单 tensor 为4维，`[E, ceil(K / 32), N, 2]` |
| `biasOptional` | 输入 | 偏置。 | 动态可选输入，可以为一个 tensor、tensorlist 或空。 | `FLOAT32` | ND | 与专家维度和输出维度对应 |
| `activationType` | 输入 | 激活函数类型。 | 必选属性。当前仅支持 `"gelu_tanh"`。 | - | - | - |
| `dequantDtype` | 输入 | GroupedMatmul 中间结果反量化后的计算类型。 | 可选属性，默认值为 `0`。当前仅支持实现定义的默认值。 | - | - | - |
| `transposeWeight` | 输入 | 权重是否按转置语义解释。 | 可选属性，默认值为 `false`。 | - | - | - |
| `groupListType` | 输入 | `groupList` 解释方式。 | 可选属性，默认值为 `0`。`0` 表示 cumsum，`1` 表示 count。 | - | - | - |
| `tuningConfigOptional` | 输入 | 调优参数。 | 可选属性，默认值为 `[0]`。用于选择或辅助选择实现模板。 | - | - | - |
| `quantDtype` | 输入 | 输出 `y` 的数据类型枚举值。 | 可选属性，默认值为 `36`，表示 `FLOAT8_E4M3FN`。当前支持 `FLOAT8_E4M3FN`、`FLOAT8_E5M2`。 | - | - | - |
| `roundMode` | 输入 | 量化舍入模式。 | 可选属性，默认值为 `"rint"`。当前仅支持 `"rint"`。 | - | - | - |
| `scaleAlg` | 输入 | scale 计算算法。 | 可选属性，默认值为 `0`。用于控制量化 scale 的计算方式，当前仅支持 `0`。 | - | - | - |
| `dstTypeMax` | 输入 | 目标量化类型的最大值配置。 | 可选属性，默认值为 `0`。当为 `0` 时由 `quantDtype` 使用默认最大值。 | - | - | - |
| `y` | 输出 | 激活并量化后的输出。 | 必选参数。 | `FLOAT8_E4M3FN`、`FLOAT8_E5M2` | ND | 2维，`[M, N]` |
| `yScale` | 输出 | 输出 MX scale。 | 必选参数。 | `FLOAT8_E8M0` | ND | 3维，`[M, ceil(N / 32), 2]` |
| `workspaceSize` | 输出 | 返回需要在 Device 侧申请的 workspace 大小。 | 不可为空。 | - | - | - |
| `executor` | 输出 | 返回 op 执行器，包含算子计算流程。 | 不可为空。 | - | - | - |

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

| 返回码 | 错误码 | 描述 |
| :--- | :--- | :--- |
| `ACLNN_ERR_PARAM_NULLPTR` | 161001 | 必选参数为空指针，例如 `x`、`groupList`、`weight`、`weightScale`、`y`、`yScale`、`workspaceSize` 或 `executor` 为空。 |
| `ACLNN_ERR_PARAM_INVALID` | 161002 | 参数不满足约束，例如 `activationType` 不是 `"gelu_tanh"`，输入 dtype/format 不支持，shape 维度不匹配，或属性值不在支持范围内。 |

## aclnnGroupedMatmulActivationQuant

- **参数说明：**

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| `workspace` | 输入 | 在 Device 侧申请的 workspace 地址。 |
| `workspaceSize` | 输入 | workspace 大小，由第一段接口 `aclnnGroupedMatmulActivationQuantGetWorkspaceSize` 获取。 |
| `executor` | 输入 | op 执行器，包含算子计算流程。 |
| `stream` | 输入 | acl stream 流。 |

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 当前仅支持 <term>Ascend 950PR/Ascend 950DT</term>。
- 当前仅支持 `activationType` 为 `"gelu_tanh"`。
- 当前仅支持 MX 量化。
- `x`、`weight`、`y` 仅支持 `FLOAT8_E4M3FN`、`FLOAT8_E5M2`。
- `xScaleOptional`、`weightScale`、`yScale` 仅支持 `FLOAT8_E8M0`。
- 当前所有 tensor 格式仅支持 ND。
- 当前 `roundMode` 仅支持 `"rint"`。
- 当前 `scaleAlg` 仅支持 `0`。
- `weight`、`weightScale`、`biasOptional` 均为 dynamic tensorlist 形式；调用方可以传入单个 tensor，也可以传入 tensorlist。`biasOptional` 支持为空。

## 调用示例

```Cpp
#include "aclnnop/aclnn_grouped_matmul_activation_quant.h"

// 省略 aclTensor/aclTensorList 创建和 device 内存申请。

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

auto ret = aclnnGroupedMatmulActivationQuantGetWorkspaceSize(
    x,
    groupList,
    xScale,
    weight,
    weightScale,
    bias,
    "gelu_tanh",
    0,
    false,
    0,
    tuningConfig,
    36,
    "rint",
    0,
    0.0,
    y,
    yScale,
    &workspaceSize,
    &executor);

// 根据 workspaceSize 申请 workspaceAddr。

ret = aclnnGroupedMatmulActivationQuant(workspaceAddr, workspaceSize, executor, stream);
```
