# GroupedMatmulSwigluQuantWeightNzV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：融合 GroupedMatmul、MXFP8 反量化、SwiGLU 和 MXFP8 动态量化。当前版本仅支持 Ascend 950（arch35）的单 Tensor MXFP8 weightNZ 场景；x 仅支持非转置输入，weight 支持非转置 NZ 和转置 ZN 两种布局。

- 当前支持的基本场景：
  - `weight` 和 `weightScale` 的 TensorList 长度均为 1；当前实现要求 `E=1`。
  - x、weight 和 output 为 `FLOAT8_E4M3FN`，xScale、weightScale 和 outputScale 为 `FLOAT8_E8M0`。
  - `weight` 为 `FRACTAL_NZ`，x、各 scale 和 output 为 `ND`。
  - 支持 `swigluMode=0/1/2/3`，以及 `activateLeft`、`clampLimit`、`gluAlpha` 和 `gluBias`。
  - `axis=-1`、`roundMode="rint"`、`dstTypeMax=0.0`；`scaleAlg` 支持 `0`（OCP）和 `1`（cuBLAS）。

- 计算公式：
  - 定义：$M$ 表示 token 数量，$K$ 表示输入特征维度，$N$ 表示 SwiGLU 拆分前的输出维度，`blocksize=64` 表示 MX 量化的分组大小。
  - 根据 `groupList` 确定每个 group 的 token 范围，并执行 GroupedMatmul 和 MXFP8 反量化：

    $$C_i = (X_i \cdot W_i) \odot xScale_{i\ Broadcast} \odot weightScale_{i\ Broadcast}$$

  - 沿最后一维执行 SwiGLU：
    - `swigluMode=0`：将结果前后分半，按 `activateLeft` 选择激活分支，计算 `SiLU(A) * B`。
    - `swigluMode=1`：按偶数/奇数位置交错拆分，计算
      `clip(x_glu, -∞, clampLimit) * sigmoid(gluAlpha * clip(x_glu, -∞, clampLimit)) * (clip(x_linear, -clampLimit, clampLimit) + gluBias)`；此模式忽略 `activateLeft`。
    - `swigluMode=2`：前后分半拆分，按 `activateLeft` 选择激活分支，对激活分支做上界裁剪、对线性分支做 `[-clampLimit, clampLimit]` 裁剪，再计算带 `gluAlpha` 和 `gluBias` 的 SwiGLU。
    - `swigluMode=3`：前后分半拆分，按 `activateLeft` 选择激活分支，计算 `min(SiLU(A), clampLimit) * clip(B, -clampLimit, clampLimit)`，忽略 `gluAlpha` 和 `gluBias`。
  - 对 SwiGLU 结果执行 MXFP8 动态量化，输出量化结果 `output` 和量化因子 `outputScale`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| ------ | -------------- | ---- | -------- | -------- | ----------- |
| x | 输入 | 激活矩阵。 | FLOAT8_E4M3FN | ND | 2维，`(M, K)` |
| weight | 动态输入 | 单个分组权重 Tensor；非转置 view 为 `(E, K, N)`，转置 view 为 `(E, N, K)`，storage 为 FRACTAL_NZ。 | FLOAT8_E4M3FN | FRACTAL_NZ | view 3维，storage 5维 |
| weight_scale | 动态输入 | weight 的 MXFP8 量化因子，TensorList 长度为 1。 | FLOAT8_E8M0 | ND | 非转置 `(E, ceil(K / 64), N, 2)`；转置 `(E, N, ceil(K / 64), 2)` |
| weight_assist_matrix | 动态输入 | 权重辅助矩阵；当前必须为 nullptr 或空 TensorList。 | - | - | - |
| bias | 可选输入 | GroupedMatmul 偏置；当前必须为 nullptr。 | - | - | - |
| x_scale | 输入 | x 的 MXFP8 量化因子。 | FLOAT8_E8M0 | ND | `(M, ceil(K / 64), 2)` |
| smooth_scale | 可选输入 | 平滑量化因子；当前必须为 nullptr。 | - | - | - |
| group_list | 输入 | 分组信息；当前长度为 1。 | INT64 | ND | `(E)` |
| dequant_mode | 属性 | 反量化模式，当前仅支持 `2`。 | INT64 | - | - |
| dequant_dtype | 属性 | 中间反量化类型，当前仅支持 `0`，即 `DT_FLOAT`。 | INT64 | - | - |
| quant_mode | 属性 | 输出量化模式，当前仅支持 `2`，即 MX 量化。 | INT64 | - | - |
| group_list_type | 属性 | `0` 表示 cumsum，`1` 表示 count，默认 `0`。 | INT64 | - | - |
| tuning_config_optional | 可选属性 | 预留调优参数；当前必须为 nullptr 或空数组。 | LIST_INT | - | - |
| activate_dim | 属性 | SwiGLU 拆分维度，当前仅支持 `-1`。 | INT64 | - | - |
| activate_left | 属性 | mode 0、2、3 中是否选择左半部分作为激活分支；mode 1 不生效。 | BOOL | - | - |
| swiglu_mode | 属性 | SwiGLU 计算模式，支持 `0`、`1`、`2`、`3`。 | INT64 | - | - |
| clamp_limit | 属性 | SwiGLU 裁剪上限；mode 1、2、3 时必须为正的有限 float32 可表示值，mode 0 不使用。 | FLOAT | - | - |
| glu_alpha | 属性 | mode 1、2 中 Sigmoid 输入的缩放系数，必须为有限且可用 float32 表示的值。 | FLOAT | - | - |
| glu_bias | 属性 | mode 1、2 中线性分支的偏置，必须为有限且可用 float32 表示的值。 | FLOAT | - | - |
| axis | 属性 | MX 量化轴，当前仅支持 `-1`。 | INT64 | - | - |
| round_mode | 属性 | MX 量化舍入模式，当前仅支持 `"rint"`。 | STRING | - | - |
| scale_alg | 属性 | MX 量化 scale 算法，支持 `0`（OCP）和 `1`（cuBLAS）；其他值报错。 | INT64 | - | - |
| dst_type_max | 属性 | 输出数据类型最大值，当前仅支持 `0.0`。 | FLOAT | - | - |
| output | 输出 | SwiGLU 后的 MXFP8 量化结果。 | FLOAT8_E4M3FN | ND | `(M, N / 2)` |
| output_scale | 输出 | output 的 MXFP8 量化因子。 | FLOAT8_E8M0 | ND | `(M, ceil((N / 2) / 64), 2)` |

## 约束说明

- 仅支持 Ascend 950（arch35）的 MXFP8 weightNZ 场景，不支持 MXFP4、per-token、A8W4、A4W4 或其他量化场景。
- 仅支持单 Tensor：`weight` 和 `weightScale` 的 TensorList 长度均为 1，当前实现要求 `groupList` 长度和 weight 的逻辑首维均为 1。
- x 仅支持非转置输入；weight 支持非转置和转置输入。非转置 weight 的 view/storage shape 分别为 `(E, K, N)` 和 `(E, ceil(K / 16), ceil(N / 32), 16, 32)`；转置 weight 的 view/storage shape 分别为 `(E, N, K)` 和 `(E, ceil(K / 32), ceil(N / 16), 16, 32)`。
- `M`、`K` 必须大于 0，`N` 必须为 64 的整数倍；SwiGLU 输出维度为 `N / 2`。
- `groupListType=0` 时，groupList 元素表示累计 token 数，序列必须非负单调递增且最后一个值不大于 `M`；`groupListType=1` 时，元素必须非负且总和不大于 `M`。
- `dequantMode=2`、`dequantDtype=0`、`quantMode=2`，且 `groupListType` 只能取 `0` 或 `1`。
- `weightAssistMatrix`、`bias`、`smoothScale` 必须为空或 nullptr；`tuningConfigOptional` 必须为 nullptr 或空数组。
- `activateDim=-1`、`axis=-1`、`roundMode="rint"`，`dstTypeMax=0.0`。
- `scaleAlg` 仅支持 `0` 或 `1`；本接口不支持 `scaleAlg=2`。
- 执行路径由 `K` 和 `swigluMode` 决定：mode 1 始终使用 CGMCT；mode 0、2、3 且 `K` 为 128 对齐时使用 Tensor API，否则使用 CGMCT 兼容路径。非转置 weight 对应 NZ 路径，转置 weight 对应 ZN 路径。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_grouped_matmul_swiglu_quant_v3_nz](examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_v3_nz.cpp) | 通过 [aclnnGroupedMatmulSwigluQuantWeightNzV3](docs/aclnnGroupedMatmulSwigluQuantWeightNzV3.md) 的两段式接口调用 V3 算子，支持四种 `swigluMode` 和 NZ/ZN 两种 weight 布局。 |
| PyTorch API | - | 上层 Torch 接口由 op-plugin 仓库提供；本目录只提供 aclnn 算子实现和直调示例。 |
