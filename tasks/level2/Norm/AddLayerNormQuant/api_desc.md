# aclnnAddLayerNormQuant

## 功能描述

### 算子功能
计算 (x1 + x2 + bias) 后做 LayerNorm（(x - mean) * rstd * gamma + beta），再按 quant_mode 做量化得到 int8 输出 y1、y2。支持 dynamic（仅 y1 有效、y2/out_scales2 为占位）或 static（双路 scales1/scales2）量化。输出 y1、y2、x（add 后未归一化结果）、out_scales1、out_scales2。

### 计算公式
- x = x1 + x2 + bias（bias 可选）
- mean = mean(x, dim=-1), var = var(x, dim=-1), rstd = 1/sqrt(var + epsilon)
- y_norm = (x - mean) * rstd * gamma + beta
- 动态量化：out_scale = max(|y_norm|)/127，y_int8 = round(y_norm / out_scale)
- 静态双路：y1 = quant(y_norm * scales1), y2 = quant(y_norm * scales2)，out_scales 为每行 scale

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.add_layer_norm_quant()` 函数形式提供：

```python
def add_layer_norm_quant(
    x1, x2, gamma, beta,
    bias=None, scales1=None, scales2=None,
    zero_points1=None, zero_points2=None,
    quant_mode="dynamic", epsilon=1e-5, additional_output=True, div_mode=True
):
    """
    实现 Add + LayerNorm + 量化。

    参数:
        x1 (Tensor): Device 侧输入，数据类型支持 FLOAT16、BF16、FLOAT32，数据格式支持 ND。
        x2 (Tensor): Device 侧输入，与 x1 同 dtype，数据格式支持 ND。
        gamma (Tensor): LayerNorm scale，最后一维与 x 一致，数据类型同 x1。
        beta (Tensor): LayerNorm bias，与 gamma 同 shape，数据类型同 x1。
        bias (Tensor, 可选): 加在 x1+x2 上的偏置，与 gamma 同 shape。
        scales1 (Tensor, 可选): 静态量化第一路 scale；dynamic 时可不传。
        scales2 (Tensor, 可选): 静态量化第二路 scale；仅传 scales1 不传 scales2 不支持。
        zero_points1 (Tensor, 可选): 第一路零点，当前动态模式不支持。
        zero_points2 (Tensor, 可选): 第二路零点，当前动态模式不支持。
        quant_mode (str, 可选): "dynamic" 或 "static"，默认 "dynamic"。
        epsilon (float, 可选): LayerNorm 的 epsilon，默认 1e-5。
        additional_output (bool, 可选): 是否输出额外信息，默认 True。
        div_mode (bool, 可选): 量化除/乘模式，默认 True。

    返回:
        List[Tensor]: [y1, y2, x, out_scales1, out_scales2]
        - y1, y2: int8，shape 与 x1 一致；dynamic 时 y2 为占位（如 shape [1]）。
        - x: add 结果（x1+x2+bias），与 x1 同 dtype、同 shape。
        - out_scales1, out_scales2: float32，每行 scale，shape 为 x1 除最后一维；dynamic 且无 scales2 时 out_scales2 为占位。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

x1 = torch.randn(2, 1024, dtype=torch.float16)
x2 = torch.randn(2, 1024, dtype=torch.float16)
gamma = torch.ones(1024, dtype=torch.float16)
beta = torch.zeros(1024, dtype=torch.float16)

# 动态量化
y1, y2, x_out, out_scale1, out_scale2 = kernel_gen_ops.add_layer_norm_quant(
    x1, x2, gamma, beta,
    None, None, None, None, None,
    "dynamic", 1e-5, True, True
)
```

## 约束与限制

- 张量数据格式支持 ND。
- x1、x2、gamma、beta 支持 FLOAT16、BF16、FLOAT32。
- dynamic 模式下不支持仅提供 scales2 或不支持 zero_points；仅提供 scales2 而不提供 scales1 会报错。
- 输出 y1、y2 为 INT8；out_scales1、out_scales2 为 FLOAT32。
