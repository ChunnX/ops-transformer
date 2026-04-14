# aclnnThreeInterpolateBackward

## 功能描述

### 算子功能
该AscendC算子计算三点插值的反向传播，根据索引和权重将梯度分配回原始位置。

### 计算公式
$$
grad\_y[b, c, idx[b, n, k]] \mathrel{+}= weight[b, n, k] \cdot grad\_x[b, c, n]
$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.three_interpolate_backward()` 函数形式提供：

```python
def three_interpolate_backward(grad_x, idx, weight, m):
    """
    计算三点插值的反向梯度。
    
    参数:
        grad_x (Tensor): 输入梯度张量，形状为 (B, C, N)。
        idx (Tensor): 插值索引，形状为 (B, N, 3)。
        weight (Tensor): 插值权重，形状为 (B, N, 3)。
        m (int): 输出维度 M 的大小。
    
    返回:
        Tensor: 输出梯度张量，形状为 (B, C, M)。
    """
```

## 使用案例
```python
import torch
import kernel_gen_ops

grad_x = torch.randn(2, 16, 100)
idx = torch.randint(0, 200, (2, 100, 3))
weight = torch.rand(2, 100, 3)
result = kernel_gen_ops.three_interpolate_backward(grad_x, idx, weight, 200)
```

## 约束与限制
- 梯度数据类型支持 float32、float16
- 索引数据类型支持 int32、int64
- 数据格式仅支持 ND
