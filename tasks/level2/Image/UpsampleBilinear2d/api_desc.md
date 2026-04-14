# aclnnUpsampleBilinear2d

## 功能描述

### 算子功能
该AscendC算子实现2D双线性插值上采样，将4D输入张量在高度和宽度维度上进行上采样。

### 计算公式
使用双线性插值：
$$
y[n, c, h_{out}, w_{out}] = \sum_{i,j} w_i \cdot w_j \cdot x[n, c, h_i, w_j]
$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_bilinear2d()` 函数形式提供：

```python
def upsample_bilinear2d(input, size, align_corners=False, scales_h=0.0, scales_w=0.0):
    """
    使用双线性插值进行2D上采样。
    
    参数:
        input (Tensor): 输入张量，形状为 (N, C, H_in, W_in)。
        size (Tensor): 目标输出尺寸张量 [out_h, out_w]。
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
size = torch.tensor([8, 8], dtype=torch.int32)
result = kernel_gen_ops.upsample_bilinear2d(x, size, align_corners=False)
```

## 约束与限制
- 数据类型支持 bfloat16、float16、float32
- size 参数类型为 int32
- 数据格式仅支持 ND
