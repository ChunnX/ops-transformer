# cann_ops_transformer.grouped_matmul_swiglu_quant_v2

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

`cann_ops_transformer.grouped_matmul_swiglu_quant_v2` 是 GroupedMatmul、MX
反量化、SwiGLU 和 MXFP8 量化的融合 Torch 接口。当前由
`ops-transformer_` 提供的 Torch Extension 只支持 V3 增量场景：单 Tensor、
MXFP8、FRACTAL_NZ weight，且 `swiglu_mode=2`。未传 `swiglu_mode` 的原 V2
场景不属于该扩展的支持范围。

对每个由 `group_list` 指定的 expert，先计算：

$$C=(XW)\odot XScale\odot WeightScale$$

再沿最后一维把 `C` 均分为 `x_glu` 和 `x_linear`，并按 mode 2 计算：

$$S=min(x_{glu},clampLimit)\cdot sigmoid(gluAlpha\cdot min(x_{glu},clampLimit))\cdot
(clip(x_{linear},-clampLimit,clampLimit)+gluBias)$$

最后将 `S` 动态 MX 量化为 FP8 E4M3FN 输出和 E8M0 scale。

## 函数原型

```python
cann_ops_transformer.grouped_matmul_swiglu_quant_v2(
    x, weight, weight_scale, x_scale, group_list, *,
    dequant_mode=2, dequant_dtype=torch.float32, quant_mode=2,
    quant_dtype=torch.float8_e4m3fn, group_list_type=0,
    weight_scale_dtype=torch_npu.float8_e8m0fnu,
    x_scale_dtype=torch_npu.float8_e8m0fnu,
    swiglu_mode=2, clamp_limit=None, glu_alpha=None, glu_bias=None,
    round_mode="rint", scale_alg=0, dst_type_max=0.0,
) -> (y, y_scale)
```

`x`、`weight` 使用 `torch.float8_e4m3fn`；E8M0 scale 用 `torch.int8` 存储，
并通过对应的 `*_dtype=torch_npu.float8_e8m0fnu` 指定逻辑 dtype。`weight` 和
`weight_scale` 均为长度为 1 的 TensorList。weight 必须为 FRACTAL_NZ；非转置逻辑
shape 分别为 `(E,K,N)`、`(E,ceil(K/64),N,2)`，转置时分别为 `(E,N,K)`、
`(E,N,ceil(K/64),2)`。转置场景应按 `(E,N,K)` 和 `(E,N,ceil(K/64),2)` 创建 NZ
Tensor，接口根据最后一维与 K 的关系识别该布局。输出 `y` 为 `(M,N/2)` 的 `torch.float8_e4m3fn`，`y_scale`
为 `(M,ceil((N/2)/64),2)` 的 E8M0。

`swiglu_mode` 是可选参数，但在本扩展中必须显式传 `2`；`clamp_limit`、
`glu_alpha`、`glu_bias` 缺省时分别取 `7.0`、`1.702`、`1.0`。`round_mode` 仅支持
`"rint"`，`scale_alg` 支持 `0` 或 `1`，`dst_type_max` 必须为 `0.0`。

## 调用示例

```python
import math
import torch
import torch_npu
import cann_ops_transformer

E, M, K, N = 1, 32, 128, 128
x = torch.randint(-4, 5, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
weight = torch.randint(-4, 5, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
weight = torch_npu.npu_format_cast(weight, 29, customize_dtype=torch.float8_e4m3fn)
weight_scale = torch.randint(120, 127, (E, math.ceil(K / 64), N, 2), dtype=torch.int8).npu()
x_scale = torch.randint(120, 127, (M, math.ceil(K / 64), 2), dtype=torch.int8).npu()
group_list = torch.tensor([M], dtype=torch.int64).npu()

y, y_scale = cann_ops_transformer.grouped_matmul_swiglu_quant_v2(
    x, [weight], [weight_scale], x_scale, group_list,
    swiglu_mode=2,
    weight_scale_dtype=torch_npu.float8_e8m0fnu,
    x_scale_dtype=torch_npu.float8_e8m0fnu,
    scale_alg=0,
)
```
