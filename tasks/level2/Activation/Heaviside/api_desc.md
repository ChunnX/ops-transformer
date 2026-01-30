# aclnnHeaviside

## 功能描述

### 算子功能
该算子计算输入input中每个元素的Heaviside阶跃函数，作为模型的激活函数。

### 计算公式

$$
\text{Heaviside}(\text{input},   \text{values}) =
\begin{cases}
0, & \text{如果 input} < 0 \\
\text{values}, & \text{如果  input} = 0 \\
1, & \text{如果 input} > 0
\end{cases}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.heaviside()` 函数形式提供：


```python
def heaviside(input: Tensor, values: Tensor):
    """
    实现 Heaviside 激活函数操作。

    参数:
        input (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        values (Tensor): 输入张量，Device侧的张量，数据格式支持ND，input值为0时使用的值的张量。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变，输出shape和输入input与values经过broadcast后的shape一致。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
input = torch.randn(4, 8, dtype=torch.float32)
values = torch.randn(4, 8, dtype=torch.float32)

# 使用 heaviside 执行激活操作
result = kernel_gen_ops.heaviside(input, values)
```
## 约束与限制

- 输入 values 的元素数量为1，或者shape需要与 输入 input 的shape相同。
- 输出 out 的shape和输入 input 与 输入 values 经过broadcast后的shape一致。
- 张量数据格式支持ND。
- 输入 input 的数据类型只支持FLOAT16，FLOAT32, BF16。
- 输出 out 的数据类型和输入 values 的数据类型需要与 输入 input 的数据类型一致。
