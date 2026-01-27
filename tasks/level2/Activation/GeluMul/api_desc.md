# aclnnGeluMul

## 功能描述

### 算子功能
该算子实现Gelu激活函数与Mul的计算功能。该算子将输入input按照最后一个维度分为左右两个Tensor：x1和x2，输出结果为 $GELU(x1) * x2$。

### 计算公式

给定输入张量 `input`，最后一维的长度为 `2d`，函数 `GeluMul` 进行以下计算：

1. 将 `input` 分割为两部分：

    $$
    x_1 = \text{input}[..., :d], \quad x_2 = \text{input}[..., d:]
    $$

2. 对x1应用GELU激活函数，"tanh"模式公式如下：

    $$
    \text{GELU}(x) = 0.5 \cdot x \cdot \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \cdot \left( x + 0.044715 x^3 \right) \right) \right)
    $$

    “none”对应的erf模式公式如下：

    $$
    \text{GELU}(x) = 0.5 \cdot x \left( 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right)
    $$

    因此，计算：

    $$
    x_1 = \text{GELU}(x_1)
    $$

3. 最终输出是x1和x2的逐元素乘积：

    $$
    \text{out} = x_1 \times x_2
    $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.gelu_mul()` 函数形式提供：


```python
def gelu_mul(input: Tensor, approximate: str):
    """
    实现 GeluMul 激活函数操作。

    参数:
        input (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        approximate (str): GELU 近似方式，仅支持"tanh"和"none"。

    返回:
        Tensor: 输出张量，为输入张量按照最后一个维度分为左右两个Tensor：x1和x2，对x1经过 Gelu 激活函数处理后的结果再与x2相乘得到最终输出结果。
                数据类型与输入一致，数据格式保持不变。

    注意:
        - 输入 input 最后一维的长度必须为偶数且小于等于1024，其他维度的乘积小于等于200000；
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
input = torch.randn(4, 8, dtype=torch.float32)

# 使用 gelu_mul 执行激活操作
result = kernel_gen_ops.gelu_mul(input, "tanh")
```
## 约束与限制

- 输入 input 最后一维的长度必须为偶数且小于等于1024，其他维度的乘积小于等于200000。
- 输入 approximate 仅支持"tanh"和"none"。
- 输出 out 最后一维的长度为输入 input 最后一维长度的二分之一，其余维度与输入 input 的维度保持一致。
- 张量数据格式支持ND。
- 输入 input 的数据类型只支持FLOAT16，FLOAT32, BF16。
- 输出 out 的数据类型需要与 输入 input 的数据类型一致。
