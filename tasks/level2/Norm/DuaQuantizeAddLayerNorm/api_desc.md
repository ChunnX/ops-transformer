# aclnnDuaQuantizeAddLayerNorm

## 功能描述

### 算子功能
计算 (x1 + x2 + bias) 后做 LayerNorm，再分别用 scales1、scales2（及可选 zero_points）做双路量化得到 int8 的 y1、y2。输出 y1、y2 与 x（add 后取整结果，用于与 tiling 一致）。与 AddLayerNormQuant 的区别为本算子 bias、scales1、scales2 为必选，且仅输出 y1、y2、x 三个张量。

### 计算公式
- x_sum = x1 + x2 + bias
- LayerNorm：mean、var、rstd = 1/sqrt(var + epsilon)，y_norm = (x_sum - mean) * rstd * gamma + beta
- 双路量化：div 模式（axis != 特殊值）y = round(y_norm / scale + zero_point)；mul 模式 y = round(y_norm * scale + zero_point)，再 clamp 到 int8
- 输出 x 为 round(x_sum) 写回输入 dtype（与 tiling 一致）

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.dua_quantize_add_layer_norm()` 函数形式提供：

```python
def dua_quantize_add_layer_norm(
    x1, x2, gamma, beta, bias,
    scales1, scales2,
    zero_points1=None, zero_points2=None,
    dtype, axis=-1, epsilon=1e-5, additional_output=False
):
    """
    实现双路量化 Add + LayerNorm。

    参数:
        x1 (Tensor): Device 侧输入，数据类型支持 FLOAT32、FLOAT16、BF16，数据格式支持 ND。
        x2 (Tensor): Device 侧输入，与 x1 同 dtype，数据格式支持 ND。
        gamma (Tensor): LayerNorm scale，与最后一维一致，数据类型同 x1。
        beta (Tensor): LayerNorm bias，与 gamma 同 shape。
        bias (Tensor): 加在 x1+x2 上的偏置，必选。
        scales1 (Tensor): 第一路量化 scale，必选。
        scales2 (Tensor): 第二路量化 scale，必选。
        zero_points1 (Tensor, 可选): 第一路零点。
        zero_points2 (Tensor, 可选): 第二路零点。
        dtype (int): 输出/内部 dtype 属性（算子 Attr）。
        axis (int, 可选): 归一化轴，默认 -1；特殊值（如 -65535）表示 mul_scale 模式。
        epsilon (float, 可选): LayerNorm 的 epsilon，默认 1e-5。
        additional_output (bool, 可选): 是否输出额外信息，默认 False。

    返回:
        List[Tensor]: [y1, y2, x]
        - y1, y2: int8，shape 与 x1 一致。
        - x: round(x1+x2+bias) 写回与 x1 同 dtype，shape 与 x1 一致。
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
bias = torch.zeros(1024, dtype=torch.float16)
scales1 = torch.tensor([0.01], dtype=torch.float16)
scales2 = torch.tensor([0.02], dtype=torch.float16)

y1, y2, x_out = kernel_gen_ops.dua_quantize_add_layer_norm(
    x1, x2, gamma, beta, bias,
    scales1, scales2,
    None, None,
    0, -1, 1e-5, False
)
```

## 约束与限制

- 张量数据格式支持 ND。
- x1、x2、gamma、beta、bias、scales1、scales2 支持 FLOAT32、FLOAT16、BF16。
- bias、scales1、scales2 为必选输入；zero_points1、zero_points2 为可选。
- 输出仅 y1、y2、x 三个张量；y1、y2 为 INT8，x 与输入同 dtype。
