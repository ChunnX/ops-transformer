# grouped\_matmul\_activation\_quant

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

`grouped_matmul_activation_quant` 是 GroupedMatmulActivationQuant 的 torch 单算子接口，当前支持
`gelu_tanh` 激活函数与 MXFP8 量化组合，底层调用
`aclnnGroupedMatmulActivationQuantWeightNz`。

当前接口仅覆盖 WeightNz 路径，`weight` 需要先通过 `torch_npu.npu_format_cast(weight, 29)`
转换为 FRACTAL_NZ 格式。

## 函数原型

```python
grouped_matmul_activation_quant(
    x,
    group_list,
    weight,
    weight_scale,
    activation_type,
    *,
    bias=None,
    x_scale=None,
    group_list_type=0,
    tuning_config=None,
    quant_mode=None,
    y_dtype=None,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=None,
    x_scale_dtype=None,
) -> (Tensor, Tensor)
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `x` | 必选输入，Tensor，shape为`(M, K)`，dtype支持`torch.float8_e4m3fn`、`torch.float8_e5m2`。 |
| `group_list` | 必选输入，Tensor，1D，dtype为`torch.int64`，第一维表示group数E，E范围为`[1, 1024]`。 |
| `weight` | dynamic input，TensorList，当前MXFP8仅支持长度为1。元素为3D逻辑Tensor，支持`(E, K, N)`或`(E, N, K)`，实际格式必须为FRACTAL_NZ，dtype仅支持`torch.float8_e4m3fn`。 |
| `weight_scale` | dynamic input，TensorList，当前MXFP8仅支持长度为1。非转置weight对应shape为`(E, ceil(K / 64), N, 2)`，转置weight对应shape为`(E, N, ceil(K / 64), 2)`。 |
| `bias` | 可选dynamic input，TensorList，默认`None`。当前MXFP8不支持非空bias，支持`None`或空TensorList。 |
| `activation_type` | required attr，当前仅支持`"gelu_tanh"`。 |
| `x_scale` | optional input，当前MXFP8必须传入，shape为`(M, ceil(K / 64), 2)`。 |
| `group_list_type` | 可选属性，支持`0`或`1`，默认`0`。 |
| `tuning_config` | 可选属性，预留调优参数。 |
| `quant_mode` | 可选属性，默认`None`。输入FP8且`quant_mode`为`None`时，自动将`quant_mode`推导为`"MX"`；`quant_mode`显式传值时必须为`"MX"`。 |
| `y_dtype` | 可选属性，输出`y`的数据类型，支持`torch.float8_e4m3fn`和`torch.float8_e5m2`，默认`None`；为`None`时自动推导为与`x`相同的数据类型。 |
| `round_mode` | 可选属性，当前仅支持`"rint"`。 |
| `scale_alg` | 可选属性，支持`0`或`1`，默认`0`。 |
| `dst_type_max` | 可选属性，表示maxType的取值，对应公式中的Amax(DType)，默认`0.0`。支持取值`0.0`和`6.0-12.0`，取值为`0.0`代表Amax(DType)为量化结果数据类型的最大值；取值为`6.0-12.0`代表Amax(DType)为传入值。仅支持在FP4E2M1和blocksize取32时设置该值，当前MXFP8场景使用默认值`0.0`。 |
| `x_dtype` | 可选dtype wrapper，用于覆盖`x`构造aclTensor时使用的dtype。 |
| `weight_dtype` | 可选dtype wrapper，用于覆盖`weight`构造aclTensor时使用的dtype。 |
| `weight_scale_dtype` | 可选dtype wrapper，用于覆盖`weight_scale`构造aclTensor时使用的dtype。 |
| `x_scale_dtype` | 可选dtype wrapper，用于覆盖`x_scale`构造aclTensor时使用的dtype。 |

## 返回值说明

| 返回值 | shape | dtype |
| --- | --- | --- |
| `y` | `(M, N)` | 由`y_dtype`指定 |
| `y_scale` | `(M, ceil(N / 64), 2)` | FLOAT8_E8M0 |

## 约束说明

- 当前仅支持`activation_type="gelu_tanh"`。
- 当前仅支持MXFP8量化模式。
- `weight`需要以3D逻辑shape传入，并通过`torch_npu.npu_format_cast(weight, 29)`转换为FRACTAL_NZ格式。
- `N`必须为64的整数倍。
- 当前MXFP8场景下`bias`必须为空，支持`None`或空TensorList。

## 调用示例

```python
import math
import torch
import torch_npu
from cann_ops_transformer import grouped_matmul_activation_quant

E = 1
M = 64
K = 128
N = 128

x = torch.randint(-8, 8, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
weight = torch.randint(-8, 8, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
weight = torch_npu.npu_format_cast(weight, 29)
weight_scale = torch.randint(-8, 8, (E, math.ceil(K / 64), N, 2), dtype=torch.int8).npu()
x_scale = torch.randint(-8, 8, (M, math.ceil(K / 64), 2), dtype=torch.int8).npu()
group_list = torch.tensor([M], dtype=torch.int64).npu()

y, y_scale = grouped_matmul_activation_quant(
    x,
    group_list,
    [weight],
    [weight_scale],
    "gelu_tanh",
    bias=None,
    x_scale=x_scale,
    quant_mode=None,
    y_dtype=None,
    weight_scale_dtype=torch_npu.float8_e8m0fnu,
    x_scale_dtype=torch_npu.float8_e8m0fnu,
)
```
