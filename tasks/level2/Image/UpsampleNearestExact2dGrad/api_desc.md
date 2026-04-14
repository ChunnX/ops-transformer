# aclnnUpsampleNearestExact2dGrad

## 功能描述

### 算子功能
计算最近邻精确（nearest exact）2D 上采样的反向传播，根据上采样输出的梯度（grad_output）计算上采样输入的梯度（grad_input）。

### 计算公式
对于 nearest exact 模式的反向传播，输出像素 (i, j) 的梯度累加到对应的输入像素上：

$$\text{grad\_input}[n, c, h, w] = \sum_{i,j} \text{grad\_output}[n, c, i, j] \quad \text{where} \quad h = \lfloor i \cdot \text{scale\_h} \rfloor, \; w = \lfloor j \cdot \text{scale\_w} \rfloor$$

其中 scale_h = iH / oH，scale_w = iW / oW。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_nearest_exact2d_grad()` 函数形式提供：

```python
def upsample_nearest_exact2d_grad(grad_output: torch.Tensor,
                                   output_size: List[int],
                                   input_size: List[int],
                                   scales_h: float = 0.0,
                                   scales_w: float = 0.0) -> torch.Tensor:
    """计算 nearest exact 2D 上采样的反向梯度。

    Args:
        grad_output: 上采样输出梯度，形状为 [N, C, oH, oW]
        output_size: 前向传播的输出尺寸 [N, C, oH, oW]
        input_size: 前向传播的输入尺寸 [N, C, iH, iW]
        scales_h: 高度缩放因子（可选，0 表示从尺寸计算）
        scales_w: 宽度缩放因子（可选，0 表示从尺寸计算）

    Returns:
        grad_input: 上采样输入梯度，形状为 [N, C, iH, iW]
    """
```

## 使用案例
```python
import torch
import kernel_gen_ops

# 前向传播: input [1,3,4,4] -> output [1,3,8,8] (2x上采样)
grad_output = torch.randn(1, 3, 8, 8)
output_size = [1, 3, 8, 8]
input_size = [1, 3, 4, 4]

grad_input = kernel_gen_ops.upsample_nearest_exact2d_grad(
    grad_output, output_size, input_size)
# grad_input shape: [1, 3, 4, 4]
```

## 约束与限制
- 支持的数据类型：float16、float32、bfloat16
- 支持的数据格式：ND（4维张量 NCHW）
- grad_output 的空间维度必须与 output_size 一致
- input_size 和 output_size 必须为 4 元素列表 [N, C, H, W]
- scales_h 和 scales_w 为可选参数，传入 0 时自动从 input_size/output_size 计算
