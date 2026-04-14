# aclnnUpsampleLinear1d

## 功能描述

### 算子功能
该AscendC算子实现1D线性插值上采样，将3D输入张量在长度维度上进行上采样。

### 计算公式
使用线性插值计算输出值：
$$
y[n, c, l_{out}] = (1-w) \cdot x[n, c, l_0] + w \cdot x[n, c, l_1]
$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_linear1d()` 函数形式提供：

```python
def upsample_linear1d(x, output_size, align_corners=False, scale=0.0):
    """
    使用线性插值进行1D上采样。
    
    参数:
        x (Tensor): 输入张量，形状为 (N, C, L_in)。
        output_size (int): 目标输出长度。
        align_corners (bool): 是否对齐角点。
        scale (float): 缩放因子。
    
    返回:
        Tensor: 上采样结果张量。
    """
```

## 使用案例
```python
import torch
import kernel_gen_ops

x = torch.randn(1, 3, 10)
result = kernel_gen_ops.upsample_linear1d(x, 20, align_corners=False)
```

## 约束与限制
- 数据类型支持 float16、bfloat16、float32
- size 参数类型为 int32
- 数据格式仅支持 ND
