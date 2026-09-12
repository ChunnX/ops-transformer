# aclnnGroupedMatmulSwigluQuantWeightNzV3

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

- 接口功能：融合 GroupedMatmul、MXFP8 反量化、SwiGLU 和 MXFP8 量化。该接口是 `aclnnGroupedMatmulSwigluQuantV3` 的 weightNZ 特化接口，当前仅支持 Ascend 950（arch35）的单 Tensor MXFP8 weightNZ 场景。

- 与 V2 的主要区别：V3 增加了可配置的 SwiGLU 参数，包括 `activateDim`、`activateLeft`、`swigluMode`、`clampLimit`、`gluAlpha` 和 `gluBias`。这些参数只作用于 SwiGLU 计算阶段，不改变 GroupedMatmul 和 MXFP8 量化的数据格式约束。

- 输入和输出：

  - `x` 是形状为 `(M, K)` 的 MXFP8 激活矩阵。
  - `weight` 是 TensorList，长度必须为 1；其中唯一 Tensor 的逻辑 view 形状为 `(E, K, N)`，storage 形状为五维 FRACTAL_NZ。
  - `weightScale` 是长度为 1 的 TensorList，形状为 `(E, ceil(K / 64), N, 2)`。
  - `xScale` 的形状为 `(M, ceil(K / 64), 2)`。
  - `output` 的形状为 `(M, N / 2)`，`outputScale` 的形状为 `(M, ceil((N / 2) / 64), 2)`。

- 计算过程：

  1. 根据 `groupList` 和 `groupListType` 将 `x` 划分为不同的分组，并为每个分组选择对应的权重切片。
  2. 执行 grouped matmul，并结合 `xScale` 和 `weightScale` 完成 MXFP8 反量化，得到待进行 SwiGLU 计算的矩阵 `x_i`：

     $$x_i = (X_i \cdot W_i) \odot xScale_i \odot weightScale_i$$

     其中缩放因子按照 MX block 规则在 K 方向对应的 64 元素块上广播。

  3. 按 `activateDim=-1` 和 `activateLeft` 对 `x_i` 拆分，执行 `swigluMode` 对应的 SwiGLU 计算，得到 `swigluOut_i`。
  4. 对 `swigluOut_i` 执行 MXFP8 动态量化，输出量化结果 `output` 和 E8M0 格式的 `outputScale`：

     $$output, outputScale = DynamicMxQuant(swigluOut_i)$$

- `swigluMode` 取值和计算公式：

  - `swigluMode=0`：标准 SwiGLU。`A_i` 表示 `x_i` 的前半部分，`B_i` 表示 `x_i` 的后半部分。

    $$swigluOut_i = Swiglu(x_i) = Swish(A_i) * B_i$$

  - `swigluMode=1`：按奇偶位置交错拆分。`x_glu` 表示 `x_i` 的偶数索引部分，`x_linear` 表示 `x_i` 的奇数索引部分。

    $$x\_glu = x\_glu.clamp(min=None, max=clampLimit)$$

    $$x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)$$

    $$out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)$$

    $$swigluOut_i = out\_glu * (x\_linear + gluBias)$$

  - `swigluMode=2`：沿 `activateDim` 轴前后分半拆分，且不是交错分块。`activateLeft=true` 时 `x_glu` 为前半部分，否则为后半部分。

    $$x\_glu = x\_glu.clamp(min=None, max=clampLimit)$$

    $$x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)$$

    $$out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)$$

    $$swigluOut_i = out\_glu * (x\_linear + gluBias)$$

    此模式要求 `axis=-1`。

  - `swigluMode=3`：沿 `activateDim` 轴前后分半拆分，且不是交错分块。固定使用 `alpha=1`，不添加 bias。

    $$x\_glu = x\_glu * sigmoid(x\_glu)$$

    $$x\_glu = x\_glu.clamp(min=None, max=clampLimit)$$

    $$x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)$$

    $$swigluOut_i = x\_glu * x\_linear$$

    此模式忽略 `gluAlpha` 和 `gluBias`，并要求 `axis=-1`。

- `clampLimit` 表示变体 SwiGLU 使用的门限值，具体 clamp 范围由 `swigluMode` 决定：`x_glu` 使用上界 `clampLimit`，`x_linear` 使用对称范围 `[-clampLimit, clampLimit]`；该参数必须大于 0。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize` 获取计算所需的 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnGroupedMatmulSwigluQuantWeightNzV3` 执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
    const aclTensor *x,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *weightAssistMatrix,
    const aclTensor *bias,
    const aclTensor *xScale,
    const aclTensor *smoothScale,
    const aclTensor *groupList,
    int64_t dequantMode,
    int64_t dequantDtype,
    int64_t quantMode,
    int64_t groupListType,
    const aclIntArray *tuningConfigOptional,
    int64_t activateDim,
    bool activateLeft,
    int64_t swigluMode,
    double clampLimit,
    double gluAlpha,
    double gluBias,
    int64_t axis,
    const char *roundMode,
    int64_t scaleAlg,
    double dstTypeMax,
    aclTensor *output,
    aclTensor *outputScale,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize

该接口完成输入参数校验、输出 shape 校验、算子执行流程构造，并返回执行所需的 workspace 大小和 executor。

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | shape |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | 输入 | 激活矩阵。 | 必须为 MXFP8。 | `FLOAT8_E4M3FN`、`FLOAT8_E5M2` | ND | `(M, K)` |
| `weight` | 输入 | 分组权重矩阵。 | TensorList 长度必须为 1；唯一 Tensor 的逻辑 view 为 `(E, K, N)`，storage 为五维 FRACTAL_NZ。 | `FLOAT8_E4M3FN`、`FLOAT8_E5M2` | FRACTAL_NZ | view 3 维，storage 5 维 |
| `weightScale` | 输入 | 权重量化因子。 | TensorList 长度必须为 1；与 `weight` 的唯一 Tensor 对应。 | `FLOAT8_E8M0` | ND | `(E, ceil(K / 64), N, 2)` |
| `weightAssistMatrix` | 可选输入 | 权重辅助矩阵。 | 当前不支持，必须传 `nullptr` 或空 TensorList。 | - | - | - |
| `bias` | 可选输入 | 矩阵乘偏置。 | 当前不支持，必须传 `nullptr`。 | - | - | - |
| `xScale` | 输入 | 激活量化因子。 | 按 K 方向每 64 个元素对应两个 E8M0 scale。 | `FLOAT8_E8M0` | ND | `(M, ceil(K / 64), 2)` |
| `smoothScale` | 可选输入 | 平滑量化因子。 | 当前不支持，必须传 `nullptr`。 | - | - | - |
| `groupList` | 输入 | 分组信息。 | 长度必须为 `E`；其语义由 `groupListType` 决定。 | `INT64` | ND | `(E)` |
| `dequantMode` | 输入 | 反量化模式。 | 当前仅支持 `2`，表示 MX 量化反量化。 | `INT64` | - | - |
| `dequantDtype` | 输入 | GroupedMatmul 中间结果类型。 | 当前仅支持 `0`，即 `DT_FLOAT`。 | `INT64` | - | - |
| `quantMode` | 输入 | 输出量化模式。 | 当前仅支持 `2`，表示 MX 量化。 | `INT64` | - | - |
| `groupListType` | 输入 | 分组列表解释方式。 | `0` 表示 cumsum，`1` 表示 count。 | `INT64` | - | - |
| `tuningConfigOptional` | 可选输入 | tiling 调优配置。 | 当前不支持，必须传 `nullptr`。 | `aclIntArray` | - | - |
| `activateDim` | 输入 | SwiGLU 拆分维度。 | 当前仅支持 `-1`，表示最后一维。 | `INT64` | - | - |
| `activateLeft` | 输入 | 激活分支选择。 | `true` 表示拆分后的左分支作为激活分支，`false` 表示右分支作为激活分支。 | `BOOL` | - | - |
| `swigluMode` | 输入 | SwiGLU 计算模式。 | 支持 `0`、`1`、`2`、`3`。 | `INT64` | - | - |
| `clampLimit` | 输入 | 变体 SwiGLU 的 clamp 门限。 | `x_glu` 的上界为 `clampLimit`，`x_linear` 限制在 `[-clampLimit, clampLimit]`，必须大于 0，且能用 `float32` 表示。 | `DOUBLE` | - | - |
| `gluAlpha` | 输入 | SwiGLU alpha 参数。 | 用于带 alpha 的 SwiGLU 模式，必须能用 `float32` 表示。 | `DOUBLE` | - | - |
| `gluBias` | 输入 | SwiGLU bias 参数。 | 用于带 bias 的 SwiGLU 模式，必须能用 `float32` 表示。 | `DOUBLE` | - | - |
| `axis` | 输入 | MX 量化轴预留参数。 | 当前只校验默认值 `-1`，不改变量化逻辑。 | `INT64` | - | - |
| `roundMode` | 输入 | MX 量化舍入模式预留参数。 | 当前只支持 `"rint"`；传 `nullptr` 或空字符串时按默认值处理。 | `const char *` | - | - |
| `scaleAlg` | 输入 | MX 量化 scale 算法预留参数。 | 当前只校验默认值 `0`，不改变量化逻辑。 | `INT64` | - | - |
| `dstTypeMax` | 输入 | 目标类型最大值预留参数。 | 当前只校验默认值 `0.0`，不改变量化逻辑。 | `DOUBLE` | - | - |
| `output` | 输出 | MXFP8 量化结果。 | 调用方创建并传入，当前仅支持 `FLOAT8_E4M3FN`。 | `FLOAT8_E4M3FN` | ND | `(M, N / 2)` |
| `outputScale` | 输出 | MXFP8 输出 scale。 | 调用方创建并传入。 | `FLOAT8_E8M0` | ND | `(M, ceil((N / 2) / 64), 2)` |
| `workspaceSize` | 输出 | Device 侧 workspace 大小。 | 返回值。 | `uint64_t *` | - | - |
| `executor` | 输出 | 算子执行器。 | 返回值，传给第二段接口。 | `aclOpExecutor **` | - | - |

### 返回值

`aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

第一段接口会完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
<col style="width: 220px">
<col style="width: 120px">
<col style="width: 810px">
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
    <td>x、weight、weightScale、xScale、groupList、output、outputScale、workspaceSize 或 executor 为空指针，或者 TensorList 中包含空 Tensor。</td>
  </tr>
  <tr>
    <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="7">161002</td>
    <td>输入或输出的数据类型不在支持范围内；MXFP8 weightNZ 场景要求 x、weight 为 MXFP8，scale 为 FLOAT8_E8M0，output 为 FLOAT8_E4M3FN。</td>
  </tr>
  <tr>
    <td>输入或输出的参数维度、shape 或 format 不满足约束，例如 weight 的逻辑 view 不是三维、storage 不是五维 FRACTAL_NZ，或输出 shape 不匹配。</td>
  </tr>
  <tr>
    <td>当前设备不是 Ascend 950，或者 weight、weightScale 的 TensorList 长度不为 1。</td>
  </tr>
  <tr>
    <td>dequantMode、quantMode、dequantDtype 不满足 2、2、0 的约束。</td>
  </tr>
  <tr>
    <td>weightAssistMatrix、bias、smoothScale 或 tuningConfigOptional 传入了当前场景不支持的内容。</td>
  </tr>
  <tr>
    <td>activateDim 不为 -1，swigluMode 不在 [0, 3]，或 clampLimit、gluAlpha、gluBias 不满足取值约束。</td>
  </tr>
  <tr>
    <td>预留参数 axis、roundMode、scaleAlg、dstTypeMax 不满足默认值约束。</td>
  </tr>
</tbody>
</table>

## aclnnGroupedMatmulSwigluQuantWeightNzV3

该接口使用第一段接口返回的 executor，在指定 stream 上执行算子计算。

### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `workspace` | 输入 | Device 侧申请的 workspace 内存地址；大小由第一段接口返回的 `workspaceSize` 决定。 |
| `workspaceSize` | 输入 | Device 侧 workspace 大小，必须使用第一段接口返回的值。 |
| `executor` | 输入 | 第一段接口返回的 op executor。 |
| `stream` | 输入 | 指定算子执行的 ACL stream。 |

### 返回值

返回 `aclnnStatus` 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 平台和量化场景

- 仅支持 Ascend 950（arch35）。其他平台均不支持。
- 仅支持 MXFP8 weightNZ 场景，不支持 per-token、MXFP4、A8W4、A4W4 或其他量化场景。
- 仅支持单 Tensor：`weight` 和 `weightScale` 的 TensorList 长度必须均为 1；不支持将每个 expert 拆成多个 Tensor 的多 Tensor调用方式。
- `weight` 的逻辑 view 必须为三维，storage 必须为五维 FRACTAL_NZ；当前不支持 ND weight 和转置 weight。

### 数据类型和 shape

| 参数 | dtype | format | shape |
| --- | --- | --- | --- |
| `x` | `FLOAT8_E4M3FN` 或 `FLOAT8_E5M2` | ND | `(M, K)` |
| `weight` | `FLOAT8_E4M3FN` 或 `FLOAT8_E5M2` | FRACTAL_NZ | view `(E, K, N)`，storage 五维 |
| `weightScale` | `FLOAT8_E8M0` | ND | `(E, ceil(K / 64), N, 2)` |
| `xScale` | `FLOAT8_E8M0` | ND | `(M, ceil(K / 64), 2)` |
| `output` | `FLOAT8_E4M3FN` | ND | `(M, N / 2)` |
| `outputScale` | `FLOAT8_E8M0` | ND | `(M, ceil((N / 2) / 64), 2)` |
| `groupList` | `INT64` | ND | `(E)` |

其中：

- `E` 是 expert/group 数量，必须等于 `groupList` 的长度。
- `M` 是 token 数量，`K` 是输入特征维度，`N` 是 SwiGLU 拆分前的输出维度。
- `N` 必须大于 0 且为 64 的整数倍；SwiGLU 输出维度为 `N / 2`。
- `weightScale` 的最后一维固定为 2，表示每个 MX block 的 scale 数据。
- 输入 Tensor 可以是非连续 Tensor；接口内部会执行必要的 contiguous 处理，但 Tensor 的 dtype、shape 和 format 元数据必须满足上述约束。

### 属性和可选输入

- `dequantMode=2`、`quantMode=2`，且二者必须匹配。
- `dequantDtype=0`，表示中间计算使用 `DT_FLOAT`。
- `groupListType=0` 时，`groupList[i]` 表示累计 token 数，最后一个值不能大于 `M`。
- `groupListType=1` 时，`groupList[i]` 表示当前 group 的 token 数，所有值之和不能大于 `M`。
- `weightAssistMatrix`、`bias`、`smoothScale` 和 `tuningConfigOptional` 当前必须为空或 `nullptr`。
- `activateDim` 当前仅支持 `-1`。
- `swigluMode` 当前支持 `[0, 3]`。
- `clampLimit` 必须为有限值且大于 0；`gluAlpha` 和 `gluBias` 必须为可表示为 `float32` 的有限值。
- `axis`、`roundMode`、`scaleAlg`、`dstTypeMax` 目前仅作为预留参数校验默认值，分别只支持 `-1`、`"rint"`、`0`、`0.0`。

## 调用示例

示例代码如下，仅展示 V3 ACLNN 两段式接口的调用方式。完整的 Tensor 创建、Device 内存管理和资源释放代码请参考[示例源码](../examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_v3_nz.cpp)。

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

// 示例场景：E=4，M=512，K=2048，N=7168。
// x:            [M, K]，FLOAT8_E4M3FN，ND
// weight:       TensorList{[E, K, N]}，FLOAT8_E4M3FN，FRACTAL_NZ storage
// weightScale:  TensorList{[E, ceil(K/64), N, 2]}，FLOAT8_E8M0，ND
// xScale:       [M, ceil(K/64), 2]，FLOAT8_E8M0，ND
// output:       [M, N/2]，FLOAT8_E4M3FN，ND
// outputScale:  [M, ceil((N/2)/64), 2]，FLOAT8_E8M0，ND

aclTensorList *weight = nullptr;      // 使用 aclCreateTensorList 创建长度为 1 的 TensorList
aclTensorList *weightScale = nullptr; // 使用 aclCreateTensorList 创建长度为 1 的 TensorList
aclTensor *x = nullptr;               // 创建 FLOAT8_E4M3FN、ND、[M, K]
aclTensor *xScale = nullptr;          // 创建 FLOAT8_E8M0、ND、[M, ceil(K/64), 2]
aclTensor *groupList = nullptr;       // 创建 INT64、ND、[E]
aclTensor *output = nullptr;           // 创建 FLOAT8_E4M3FN、ND、[M, N/2]
aclTensor *outputScale = nullptr;      // 创建 FLOAT8_E8M0、ND、[M, ceil((N/2)/64), 2]
aclrtStream stream = nullptr;

// groupListType=1 表示 count；示例中四个 group 的 token 数之和为 M。
int64_t dequantMode = 2;
int64_t dequantDtype = 0;  // DT_FLOAT
int64_t quantMode = 2;
int64_t groupListType = 1;

int64_t activateDim = -1;
bool activateLeft = true;
int64_t swigluMode = 0;
double clampLimit = 7.0;
double gluAlpha = 1.702;
double gluBias = 1.0;
int64_t axis = -1;
const char *roundMode = "rint";
int64_t scaleAlg = 0;
double dstTypeMax = 0.0;

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
    x, weight, weightScale,
    nullptr,              // weightAssistMatrix
    nullptr,              // bias
    xScale,
    nullptr,              // smoothScale
    groupList,
    dequantMode, dequantDtype, quantMode, groupListType,
    nullptr,              // tuningConfigOptional
    activateDim, activateLeft, swigluMode,
    clampLimit, gluAlpha, gluBias,
    axis, roundMode, scaleAlg, dstTypeMax,
    output, outputScale, &workspaceSize, &executor);

// 根据 workspaceSize 在 Device 侧申请 workspace。
void *workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

ret = aclnnGroupedMatmulSwigluQuantWeightNzV3(workspace, workspaceSize, executor, stream);
// 检查 ret 后同步 stream，并按需读取 output/outputScale。
aclrtSynchronizeStream(stream);
```

当前 V3 的 tiling 和 kernel 仅保留 Ascend 950 的单入口占位实现，具体计算实现和 950 环境上的编译运行验证待后续补充。
