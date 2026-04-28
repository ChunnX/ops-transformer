# Conv3DBackpropInputV2

## 功能描述

### 算子功能

Conv3DBackpropInputV2 是三维卷积的反向传播算子，用于计算卷积操作对输入的梯度。给定输出梯度、卷积核和输入尺寸，计算输入的梯度张量。

### 计算公式

对于3D卷积的前向传播：

\[
Y = \text{Conv3D}(X, W)
\]

反向传播时，给定输出梯度 \(\frac{\partial L}{\partial Y}\)，计算输入梯度：

\[
\frac{\partial L}{\partial X} = \text{Conv3DBackpropInput}(W, \frac{\partial L}{\partial Y}, \text{input\_size})
\]

其中：
- \(X\) 为输入张量，shape为 \((N, C_{in}, D, H, W)\)
- \(W\) 为卷积核，shape为 \((C_{out}, C_{in}/groups, K_d, K_h, K_w)\)
- \(Y\) 为输出张量
- \(\frac{\partial L}{\partial Y}\) 为输出梯度

## 接口定义

### Python 接口

该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.conv3d_backprop_input_v2()` 形式提供：

```python
def conv3d_backprop_input_v2(
    input_size: Tensor,
    filter: Tensor,
    out_backprop: Tensor,
    input_tensor: Tensor,
    strides: List[int],
    paddings: List[int],
    dilations: List[int],
    groups: int,
    data_format: str,
    enable_hf32: bool,
) -> Tensor:
    """
    计算3D卷积对输入的梯度。

    参数:
        input_size (Tensor): 输入尺寸，shape为(5,)，格式为[N, D, H, W, C]（NDHWC）或[N, C, D, H, W]（NCDHW）
        filter (Tensor): 卷积核，shape为(C_out, C_in/groups, K_d, K_h, K_w)
        out_backprop (Tensor): 输出梯度，shape为(N, C_out, D_out, H_out, W_out)（NCDHW格式）
        input_tensor (Tensor): 输入张量，用于确定输出shape和dtype
        strides (List[int]): 卷积步长，长度为3，格式为[stride_d, stride_h, stride_w]
        paddings (List[int]): 填充大小，长度为3，格式为[pad_d, pad_h, pad_w]
        dilations (List[int]): 膨胀系数，长度为3，格式为[dilation_d, dilation_h, dilation_w]
        groups (int): 分组卷积的组数
        data_format (str): 数据格式，支持"NCDHW"或"NDHWC"
        enable_hf32 (bool): 是否启用HF32精度模式

    返回:
        Tensor: 输入梯度，shape与input_tensor相同
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 设置参数
batch, cin, cout = 1, 16, 32
di, hi, wi = 8, 16, 16
dk, hk, wk = 3, 3, 3
stride = [1, 1, 1]
padding = [0, 0, 0]
dilation = [1, 1, 1]
groups = 1

# 创建输入张量
input_size = torch.tensor([batch, di, hi, wi, cin], dtype=torch.int32)
filter_tensor = torch.randn(cout, cin // groups, dk, hk, wk, dtype=torch.float16, device="npu")
input_tensor = torch.randn(batch, cin, di, hi, wi, dtype=torch.float16, device="npu")

# 计算输出尺寸
dout = (di + 2 * padding[0] - dilation[0] * (dk - 1) - 1) // stride[0] + 1
ho = (hi + 2 * padding[1] - dilation[1] * (hk - 1) - 1) // stride[1] + 1
wo = (wi + 2 * padding[2] - dilation[2] * (wk - 1) - 1) // stride[2] + 1

out_backprop = torch.randn(batch, cout, dout, ho, wo, dtype=torch.float16, device="npu")

# 调用算子
grad_input = kernel_gen_ops.conv3d_backprop_input_v2(
    input_size, filter_tensor, out_backprop, input_tensor,
    stride, padding, dilation, groups, "NCDHW", False
)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含 conda、CANN、shell、`run_multi_test`、设备错误或任何**运行环境 / 排障**说明。
- 支持 dtype：float16、float32、bfloat16。
- 数据格式：支持 NCDHW（N, C, D, H, W）格式。
- 输入尺寸 `input_size` 必须与 `input_tensor` 的实际shape一致。
- 卷积核的输入通道数必须等于 `input_tensor` 的通道数除以 `groups`。
- 输出梯度的通道数必须等于卷积核的输出通道数。
- 步长、填充、膨胀系数的长度必须为3。
- `groups` 必须能整除输入通道数和输出通道数。
