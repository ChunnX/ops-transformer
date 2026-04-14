# aclnnGridSampler2DGrad

## 功能描述

### 算子功能
该AscendC算子计算2D网格采样（Grid Sample）的反向梯度，对输入张量和网格坐标分别求导。

### 计算公式
$$
dx = \frac{\partial L}{\partial x}, \quad dgrid = \frac{\partial L}{\partial grid}
$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.grid_sampler2_d_grad()` 函数形式提供：

```python
def grid_sampler2_d_grad(grad, x, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    计算2D网格采样的反向梯度。
    
    参数:
        grad (Tensor): 上游梯度张量。
        x (Tensor): 前向传播的输入张量。
        grid (Tensor): 前向传播的网格坐标。
        interpolation_mode (str): 插值模式。
        padding_mode (str): 填充模式。
        align_corners (bool): 是否对齐角点。
    
    返回:
        tuple: (dx, dgrid) 输入和网格的梯度。
    """
```

## 使用案例
```python
import torch
import kernel_gen_ops

grad = torch.randn(1, 3, 8, 8)
x = torch.randn(1, 3, 4, 4)
grid = torch.rand(1, 8, 8, 2) * 2 - 1
dx, dgrid = kernel_gen_ops.grid_sampler2_d_grad(grad, x, grid, 'bilinear', 'zeros', False)
```

## 约束与限制
- 数据类型支持 float16、float32、bfloat16
- 数据格式仅支持 ND
