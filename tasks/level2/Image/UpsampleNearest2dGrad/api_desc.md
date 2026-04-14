# aclnnUpsampleNearest2dGrad

## 功能描述

### 算子功能
计算二维最近邻上采样（UpsampleNearest2d）的反向梯度，将 grad_output 的梯度按照最近邻规则累加到 grad_input 对应位置。

### 计算公式
对于最近邻上采样的反向传播，每个输出位置的梯度累加到其对应的输入源位置：

$$\text{grad\_input}[n, c, h_{src}, w_{src}] \mathrel{+}= \text{grad\_output}[n, c, h_{dst}, w_{dst}]$$

其中源索引的计算方式为：

$$h_{src} = \lfloor h_{dst} \times \text{scale\_h} \rfloor, \quad w_{src} = \lfloor w_{dst} \times \text{scale\_w} \rfloor$$

$$\text{scale\_h} = \frac{iH}{oH}, \quad \text{scale\_w} = \frac{iW}{oW}$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_nearest2d_grad()` 函数形式提供：

```python
def upsample_nearest2d_grad(grad_output: torch.Tensor,
                             output_size: List[int],
                             input_size: List[int],
                             scales_h: float = 0.0,
                             scales_w: float = 0.0) -> torch.Tensor:
    """计算 UpsampleNearest2d 的反向梯度。"""
```

**参数说明：**
- `grad_output`：反向传播的输入梯度张量，形状为 [N, C, oH, oW]
- `output_size`：输出尺寸列表 [N, C, oH, oW]
- `input_size`：输入尺寸列表 [N, C, iH, iW]
- `scales_h`：高度方向的缩放因子（可选，默认为 0.0 表示自动计算）
- `scales_w`：宽度方向的缩放因子（可选，默认为 0.0 表示自动计算）

**返回值：**
- `grad_input`：反向传播的输出梯度张量，形状为 [N, C, iH, iW]

## 使用案例
```python
import torch
import kernel_gen_ops

# 定义输入输出尺寸
N, C, iH, iW = 1, 3, 4, 4
oH, oW = 8, 8

# 生成梯度输入
grad_output = torch.randn(N, C, oH, oW, dtype=torch.float32).npu()

# 调用算子
grad_input = kernel_gen_ops.upsample_nearest2d_grad(
    grad_output,
    output_size=[N, C, oH, oW],
    input_size=[N, C, iH, iW]
)
print(grad_input.shape)  # torch.Size([1, 3, 4, 4])
```

## 约束与限制
- 支持的数据类型：float16、float32、bfloat16
- 支持的数据格式：ND（NCHW）
- 输入梯度张量 grad_output 的形状必须为 4D
- output_size 和 input_size 必须为长度为 4 的列表
- scale 值为正数时使用显式指定值，否则自动通过 input_size/output_size 计算
