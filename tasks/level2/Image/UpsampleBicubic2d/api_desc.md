# aclnnUpsampleBicubic2d

## 功能描述

### 算子功能
该AscendC算子实现2D双三次插值上采样，将4D输入张量在高度和宽度维度上进行上采样。

### 计算公式
使用双三次插值（bicubic interpolation）计算输出像素值，对每个输出位置使用 4x4 邻域进行加权计算。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_bicubic2d()` 函数形式提供：

```python
def upsample_bicubic2d(input, output_size, align_corners=False, scales_h=0.0, scales_w=0.0):
    """
    使用双三次插值进行2D上采样。
    
    参数:
        input (Tensor): 输入张量，形状为 (N, C, H_in, W_in)。
        output_size (list): 目标输出尺寸 [H_out, W_out]。
        align_corners (bool): 是否对齐角点。
        scales_h (float): 高度缩放因子。
        scales_w (float): 宽度缩放因子。
    
    返回:
        Tensor: 上采样结果张量。
    """
```

## 使用案例
```python
import torch
import kernel_gen_ops

x = torch.randn(1, 3, 4, 4)
result = kernel_gen_ops.upsample_bicubic2d(x, [8, 8], align_corners=False)
```

## 约束与限制
- 数据类型支持 bfloat16、float16、float32
- 数据格式仅支持 ND
