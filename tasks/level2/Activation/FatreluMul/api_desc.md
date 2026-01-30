# aclnnFatreluMul

## 功能描述

### 算子功能
该算子实现Threshold激活函数与Mul的计算功能。Threshold激活函数定义为 `Threshold(x, threshold) = 0 if x < threshold else x` 。该算子将输入input按照最后一个维度分为左右两个Tensor：x1和x2，输出结果为 $Threshold(x1, threshold) * x2$。

### 计算公式

给定输入张量input，最后一维的长度为2d，进行以下计算：

1. 将input分割为两部分：

    $$
    x1 = \text{input}[..., :d], \quad x2 = \text{input}[..., d:]
    $$

2. 对x1应用Threshold激活函数，定义如下：

    $$
    \text{Threshold}(x, \text{threshold}) =
    \begin{cases}
    0 & \text{if } x < \text{threshold} \\
    x & \text{if } x \geq \text{threshold}
    \end{cases}
    $$

    因此，计算：

    $$
    x1 = \text{Threshold}(x1, \text{threshold})
    $$

3. 最终输出是x1和x2的逐元素乘积：

    $$
    \text{out} = x1 \times x2
    $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.fatrelu_mul()` 函数形式提供：


```python
def fatrelu_mul(input: Tensor, threshold: Tensor):
    """
    实现 FatreluMul 激活函数操作。

    参数:
        input (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        threshold (Tensor): 输入张量，Device侧的张量，数据格式支持ND。

    返回:
        Tensor: 输出张量，为输入张量按照最后一个维度分为左右两个Tensor：x1和x2，对x1经过 Threshold 激活函数处理后的结果再与x2相乘得到最终输出结果。
                数据类型与输入一致，数据格式保持不变。

    注意:
        - Threshold 激活函数定义为:  `Threshold(x, threshold) = 0 if x < threshold else x`；
        - 输入 input 最后一维的长度必须为偶数且小于等于1024，其他维度的乘积小于等于200000；
        - 输入 threshold 中只有一个元素；
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
input = torch.randn(4, 8, dtype=torch.float32)
threshold = torch.randn(1, dtype=torch.float32)

# 使用 fatrelu_mul 执行激活操作
result = kernel_gen_ops.fatrelu_mul(input, threshold)
```
## 约束与限制

- 输入 input 最后一维的长度必须为偶数且小于等于1024，其他维度的乘积小于等于200000。
- 输入 threshold 中只有一个元素。
- 输出 out 最后一维的长度为输入 input 最后一维长度的二分之一，其余维度与输入 input 的维度保持一致。
- 张量数据格式支持ND。
- 输入 input 的数据类型只支持FLOAT16，FLOAT32, BF16。
- 输入 threshold 的数据类型需要与 输入 input 的数据类型一致。
- 输出 out 的数据类型需要与 输入 input 的数据类型一致。