# aclnnUpsampleBilinear2dGrad

## 功能描述

### 算子功能
该AscendC算子计算2D双线性上采样的反向梯度。

### 计算公式
$$
grad\_input = \frac{\partial L}{\partial input}
$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_bilinear2d_grad()` 函数形式提供：

```python
def upsample_bilinear2d_grad(grad_output, output_size, input_size, align_corners=False, scales_h=0.0, scales_w=0.0):
    """
    计算2D双线性上采样的反向梯度。
    
    参数:
        grad_output (Tensor): 上游梯度张量。
        output_size (list): 前向传播的输出尺寸 [H_out, W_out]。
        input_size (list): 前向传播的输入尺寸 [N, C, H_in, W_in]。
        align_corners (bool): 是否对齐角点。
        scales_h (float): 高度缩放因子。
        scales_w (float): 宽度缩放因子。
    
    返回:
        Tensor: 输入梯度张量。
    """
```

## 使用案例
```python
import torch
import kernel_gen_ops

grad = torch.randn(1, 3, 8, 8)
result = kernel_gen_ops.upsample_bilinear2d_grad(grad, [8, 8], [1, 3, 4, 4], align_corners=False)
```

## 约束与限制
- 数据类型支持 bfloat16、float16、float32
- 数据格式仅支持 ND
