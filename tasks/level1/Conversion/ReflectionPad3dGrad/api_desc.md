# aclnnReflectionPad3dBackward

## 功能描述

### 算子功能
计算ReflectionPad3d的反向传播梯度。

### 计算公式
ReflectionPad3d的反向传播将填充区域的梯度反射回原始输入区域。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.reflection_pad3d_grad()` 函数形式提供：

```python
def reflection_pad3d_grad(grad_output: Tensor, self: Tensor, padding: list):
    """
    实现ReflectionPad3d的反向传播操作。

    参数:
        grad_output (Tensor): 反向传播的输入梯度张量，Device侧的张量，数据格式支持ND。
        self (Tensor): 正向传播的输入张量，Device侧的张量，数据格式支持ND。
        padding (list): 长度为6的填充参数，格式为(left, right, top, bottom, front, back)。

    返回:
        Tensor: 输出梯度张量，数据类型与输入一致，数据格式保持不变，输出shape与self相同。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
input_shape = [1, 1, 3, 3, 3]
grad_output_shape = [1, 1, 5, 5, 5]  # 假设padding为[1, 1, 1, 1, 1, 1]
padding = [1, 1, 1, 1, 1, 1]

# 创建随机输入张量
grad_output = torch.randn(grad_output_shape, dtype=torch.float16)
self_input = torch.randn(input_shape, dtype=torch.float16)

# 将张量转移到NPU设备
grad_output = grad_output.to('npu')
self_input = self_input.to('npu')

# 调用ReflectionPad3dGrad算子
result = kernel_gen_ops.reflection_pad3d_grad(grad_output, self_input, padding)
```

## 约束与限制

- gradOutput、self 和 gradInput 的维度需一致（支持四维或五维），且它们的形状需与 reflection_pad3d 正向传播的输出形状相互一致。
- 输入值域限制：padding前两个数值需小于self最后一维度的数值，中间两个数值需小于self倒数第二维度的数值，后两个数值需小于self倒数第三维度的数值。
- 数据类型支持：FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128。
- 数据格式支持：ND。