# aclnnGridSample

## 功能描述

### 算子功能
该AscendC算子实现空间网格采样（Grid Sample），根据输入的网格坐标对输入张量进行空间采样，支持双线性、最近邻和双三次插值模式。

### 计算公式
对于输出张量中每个位置，根据网格坐标从输入张量中采样：
$$
y[n, :, h, w] = \text{sample}(x[n, :, :, :], \text{grid}[n, h, w, :])
$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.grid_sample()` 函数形式提供：

```python
def grid_sample(x, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    对输入张量进行空间网格采样。
    
    参数:
        x (Tensor): 输入张量，形状为 (N, C, H_in, W_in) 或 (N, C, D_in, H_in, W_in)。
        grid (Tensor): 采样网格坐标，形状为 (N, H_out, W_out, 2) 或 (N, D_out, H_out, W_out, 3)。
        interpolation_mode (str): 插值模式，支持 'bilinear'、'nearest'、'bicubic'。
        padding_mode (str): 填充模式，支持 'zeros'、'border'、'reflection'。
        align_corners (bool): 是否对齐角点。
    
    返回:
        Tensor: 采样结果张量。
    """
```

## 使用案例
```python
import torch
import kernel_gen_ops

x = torch.randn(1, 3, 4, 4)
grid = torch.rand(1, 8, 8, 2) * 2 - 1
result = kernel_gen_ops.grid_sample(x, grid, 'bilinear', 'zeros', False)
```

## 约束与限制
- 输入数据类型支持 float16、float32、bfloat16
- 数据格式仅支持 ND
- 输入和输出在同一设备上
