# aclnnUpsampleNearest

## 功能描述

### 算子功能
该AscendC算子实现最近邻上采样，将输入张量在空间维度上进行上采样，支持普通最近邻和精确最近邻（nearest-exact）两种模式。

### 计算公式
最近邻模式：
$$
y[..., h_{out}, w_{out}] = x[..., \lfloor h_{out} \cdot scale_h \rfloor, \lfloor w_{out} \cdot scale_w \rfloor]
$$

精确最近邻模式：
$$
y[..., h_{out}, w_{out}] = x[..., \lfloor (h_{out} + 0.5) \cdot scale_h \rfloor, \lfloor (w_{out} + 0.5) \cdot scale_w \rfloor]
$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_nearest()` 函数形式提供：

```python
def upsample_nearest(x, output_size, scales_h=0.0, scales_w=0.0, exact_mode=False):
    """
    使用最近邻插值进行上采样。
    
    参数:
        x (Tensor): 输入张量。
        output_size (list): 目标输出尺寸。
        scales_h (float): 高度缩放因子。
        scales_w (float): 宽度缩放因子。
        exact_mode (bool): 是否使用精确最近邻模式。
    
    返回:
        Tensor: 上采样结果张量。
    """
```

## 使用案例
```python
import torch
import kernel_gen_ops

x = torch.randn(1, 3, 4, 4)
result = kernel_gen_ops.upsample_nearest(x, [8, 8], exact_mode=True)
```

## 约束与限制
- 数据类型支持 bfloat16、float16、float32
- 数据格式仅支持 ND
