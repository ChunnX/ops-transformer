# grouped_matmul_activation_weight_nz

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

`cann_ops_transformer.grouped_matmul_activation_weight_nz` 是
`aclnnGroupedMatmulActivationWeightNz` 的 PyTorch 封装，融合 A4W4 WeightNZ grouped matmul、
MX 反量化和 SwiGLU，直接返回 BF16 `y`。

当前接口只支持 Ascend 950PR/Ascend 950DT、A4W4、WeightNZ、BF16 输出和
`swiglu_mode=2`。它不返回 MXFP8 `y`，不返回 `y_scale`，也不支持 bias。

## 函数原型

```python
cann_ops_transformer.grouped_matmul_activation_weight_nz(
    x,
    x_scale,
    group_list,
    weight,
    weight_scale,
    *,
    bias=None,
    dequant_mode=2,
    dequant_dtype=0,
    act_type=0,
    transpose_weight=False,
    group_list_type=0,
    tuning_config=None,
    swiglu_mode=2,
    clamp_limit=0.0,
    glu_alpha=1.0,
    glu_bias=0.0,
) -> Tensor
```

## 计算公式

对每个 expert 执行 MX A4W4 grouped matmul：

$$
C_i[m,n] = \sum_{k=0}^{K-1}
\left(X_i[m,k] \times S^X_i[m,\lfloor k/64\rfloor]\right)
\left(W_i[k,n] \times S^W_i[\lfloor k/64\rfloor,n]\right)
$$

GMM 结果 `C` 沿最后一维前后分半，前半为激活分支，后半为线性分支：

$$
u = \min(C[..., :N/2], clamp\_limit)
$$

$$
v = \operatorname{clip}(C[..., N/2:], -clamp\_limit, clamp\_limit) + glu\_bias
$$

$$
y = \frac{u}{1+\exp(-glu\_alpha\times u)} \times v
$$

中间结果保持 FP32，最终输出 `y` 为 BF16，shape 为 `(M,N/2)`。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型/存储 | shape/要求 |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | 激活矩阵。 | packed `torch.int8`/`torch.uint8`，逻辑类型 `FLOAT4_E2M1` | `(M,K)`，ND |
| `x_scale` | Tensor | 必选 | 激活 MX scale。 | `torch.int8`/`torch.uint8`，逻辑类型 `FLOAT8_E8M0` | `(M,ceil(K/64),2)`，ND |
| `group_list` | Tensor | 必选 | expert 分组信息。 | `torch.int64` | `(E)`，ND |
| `weight` | List[Tensor] | 必选 | WeightNZ 权重，长度必须为 1。 | packed `torch.int8`/`torch.uint8`，逻辑类型 `FLOAT4_E2M1` | 唯一 Tensor 为 FRACTAL_NZ |
| `weight_scale` | List[Tensor] | 必选 | 权重 MX scale，长度必须为 1。 | `torch.int8`/`torch.uint8`，逻辑类型 `FLOAT8_E8M0` | 逻辑 shape `(E,ceil(K/64),N,2)` |
| `bias` | Tensor/None | 可选 | 矩阵乘偏置，默认 `None`。 | - | 当前必须为 `None` |
| `dequant_mode` | int | 可选 | 反量化模式，默认 `2`。 | int | 仅支持 `2` |
| `dequant_dtype` | int | 可选 | GMM 中间累加类型，默认 `0`。 | int | 仅支持 `0`，表示 FLOAT32 |
| `act_type` | int | 可选 | 激活类型，默认 `0`。 | int | 仅支持 `0`，表示 SwiGLU |
| `transpose_weight` | bool | 可选 | WeightNZ 转置布局标识，默认 `False`。 | bool | 必须和 weight/scale 的实际布局一致 |
| `group_list_type` | int | 可选 | `group_list` 解释方式，默认 `0`。 | int | `0` 为 cumulative，`1` 为 count |
| `tuning_config` | List[int]/None | 可选 | tiling 调优配置，默认 `None`。 | - | 当前必须为 `None` 或空列表 |
| `swiglu_mode` | int | 可选 | SwiGLU 模式，默认 `2`。 | int | **仅支持 `2`** |
| `clamp_limit` | float | 必选 | 激活和线性分支的裁剪上限。 | float32 可表示 | 有限正数 |
| `glu_alpha` | float | 必选 | sigmoid 输入缩放系数。 | float32 可表示 | 有限值 |
| `glu_bias` | float | 必选 | 线性分支偏置。 | float32 可表示 | 有限值 |

## 返回值说明

| 返回值 | 类型 | shape | 说明 |
| --- | --- | --- | --- |
| `y` | Tensor | `(M,N/2)` | BF16 SwiGLU 结果，ND 格式 |

本接口只返回 `y`，不会返回 `y_scale` 或其他辅助 Tensor。

## 约束说明

- 仅支持 Ascend 950PR/Ascend 950DT。
- `x`、`weight` 的逻辑类型必须为 `FLOAT4_E2M1`；PyTorch 侧使用 `torch.int8` 或
  `torch.uint8` 承载 packed FP4 storage。
- `x_scale`、`weight_scale` 的逻辑类型必须为 `FLOAT8_E8M0`，PyTorch 侧使用
  `torch.int8` 或 `torch.uint8` 承载 storage。
- `weight` 和 `weight_scale` 必须是长度为 1 的 list，不能按 expert 拆分。
- `weight` 必须为 `FRACTAL_NZ`，转置场景需要同时按照 ZN storage 构造 weight 和 scale，
  并设置 `transpose_weight=True`。
- K 必须与 x 的第二维一致，逻辑 N 必须满足 A4W4 TensorAPI 的 tile 对齐要求，通常要求
  N 为 128 的整数倍；输出 shape 为 `(M,N/2)`。
- `group_list_type=0` 时，`group_list` 表示累计结束位置；`group_list_type=1` 时，表示
  各 expert 的 token 数量。两种模式下都不能出现负数，且不能超过 M。
- 固定属性为 `dequant_mode=2`、`dequant_dtype=0`、`act_type=0`、`swiglu_mode=2`。
- `clamp_limit`、`glu_alpha` 和 `glu_bias` 必须是有限值，`clamp_limit` 必须大于 0。
- `bias` 必须为 `None`，`tuning_config` 必须为 `None` 或空列表。
- 不支持 mode 0、其他 SwiGLU mode、FP8 输入、FP4 E1M2、WeightND、FLOAT16/FLOAT32 输出
  或 MXFP8 输出。

## 调用示例

```python
import torch
import cann_ops_transformer

M, K, N, E = 32, 128, 128, 1

# 实际工程中需要按 CANN 的 FP4/E8M0 编码和 FRACTAL_NZ storage 构造数据。
x = torch.zeros((M, K), dtype=torch.int8, device="npu")
x_scale = torch.zeros((M, K // 64, 2), dtype=torch.int8, device="npu")
weight = torch.zeros((E, K, N), dtype=torch.int8, device="npu")
weight_scale = torch.zeros((E, K // 64, N, 2), dtype=torch.int8, device="npu")
group_list = torch.tensor([M], dtype=torch.int64, device="npu")

y = cann_ops_transformer.grouped_matmul_activation_weight_nz(
    x,
    x_scale,
    group_list,
    [weight],
    [weight_scale],
    dequant_mode=2,
    dequant_dtype=0,
    act_type=0,
    transpose_weight=False,
    group_list_type=0,
    swiglu_mode=2,
    clamp_limit=7.0,
    glu_alpha=1.702,
    glu_bias=1.0,
)

assert y.dtype == torch.bfloat16
assert tuple(y.shape) == (M, N // 2)
```

## 确定性计算

默认支持确定性计算。中间 GMM 和 SwiGLU 保持 FP32 计算，最终仅在写回 `y` 时转换为 BF16。
