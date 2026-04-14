# aclnnResizeUpsampleTrilinear

## 功能描述

### 算子功能
该AscendC算子实现3D三线性插值上采样，将5D输入张量在深度、高度和宽度维度上进行上采样。

### 计算公式
使用三线性插值计算输出值：
$$
y[n, c, d_{out}, h_{out}, w_{out}] = \sum_{i,j,k} w_i \cdot w_j \cdot w_k \cdot x[n, c, d_i, h_j, w_k]
$$
其中权重由三线性插值确定。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.resize_upsample_trilinear()` 函数形式提供：

```python
def resize_upsample_trilinear(input, output_size, align_corners=False, scales_d=0.0, scales_h=0.0, scales_w=0.0):
    """
    使用三线性插值进行3D上采样。
    
    参数:
        input (Tensor): 输入张量，形状为 (N, C, D_in, H_in, W_in)。
        output_size (list): 目标输出尺寸 [D_out, H_out, W_out]。
        align_corners (bool): 是否对齐角点。
        scales_d (float): 深度方向缩放因子。
        scales_h (float): 高度方向缩放因子。
        scales_w (float): 宽度方向缩放因子。
    
    返回:
        Tensor: 上采样结果张量。
    """
```

## 使用案例
```python
import torch
import kernel_gen_ops

x = torch.randn(1, 3, 4, 4, 4)
result = kernel_gen_ops.resize_upsample_trilinear(x, [8, 8, 8], align_corners=False)
```

## 约束与限制
- 数据类型支持 float16、float32、bfloat16
- 数据格式仅支持 ND
