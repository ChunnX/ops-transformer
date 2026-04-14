# aclnnUpsampleTrilinear3dBackward

## 功能描述

### 算子功能
三线性插值上采样的反向传播算子。给定输出梯度张量，计算对应的输入梯度张量。

### 计算公式
三线性插值反向传播将输出梯度按照三线性插值权重分配到输入的8个相邻位置：

$$grad\_input[d_0,h_0,w_0] += grad\_output[d_{out},h_{out},w_{out}] \times (1-\lambda_d)(1-\lambda_h)(1-\lambda_w)$$

其中源索引计算方式：
- `align_corners=True`: $src = scale \times dst$
- `align_corners=False`: $src = \max(scale \times (dst + 0.5) - 0.5, 0)$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_trilinear3d_backward()` 函数形式提供：

```python
def upsample_trilinear3d_backward(grad_output: torch.Tensor,
                                   output_size: List[int],
                                   input_size: List[int],
                                   align_corners: bool = False,
                                   scale_d: float = 0.0,
                                   scale_h: float = 0.0,
                                   scale_w: float = 0.0) -> torch.Tensor:
    """三线性插值上采样反向传播。"""
```

## 使用案例
```python
import torch
import kernel_gen_ops

N, C, iD, iH, iW = 1, 3, 4, 4, 4
oD, oH, oW = 8, 8, 8
grad_output = torch.randn(N, C, oD, oH, oW, device='npu', dtype=torch.float32)
output_size = [N, C, oD, oH, oW]
input_size = [N, C, iD, iH, iW]
result = kernel_gen_ops.upsample_trilinear3d_backward(
    grad_output, output_size, input_size, align_corners=False)
print(result.shape)  # torch.Size([1, 3, 4, 4, 4])
```

## 约束与限制
- 支持的数据类型：float16、float32、bfloat16
- 支持的数据格式：ND（5维张量，NCDHW格式）
- input_size 和 output_size 必须为合法正整数
- scale 值过小（<0.02）时算子会返回失败
