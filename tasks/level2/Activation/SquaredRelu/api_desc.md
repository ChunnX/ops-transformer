# aclnnSquaredRelu

## 功能描述

### 算子功能
该算子SquaredRelu函数是一个基于标准Relu函数的变体，其主要特点是对Relu函数的输出进行平方，常作为模型的激活函数。

### 计算公式

$$
y_i=(Relu(input_i))^2
$$

其中Relu的计算公式：

$$
Relu(input_i)=max(0,input_i)
$$

其中，x是输入Tensor，y是输出Tensor，i是元素在Tensor中的位置索引。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.squared_relu()` 函数形式提供：


```python
def squared_relu(input: Tensor):
    """
    实现对Relu函数的输出进行平方。

    参数:
        input (Tensor): 输入张量，Device侧的张量，数据格式支持ND。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变，输出shape和输入x的shape一致。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
input = torch.randn(4, 8, dtype=torch.float32)

# 使用 squared_relu 执行激活操作
result = kernel_gen_ops.squared_relu(input)
```
## 约束与限制

- 输出 out 的shape需要与 输入 input 的shape相同。
- 输入 input 的数据类型只支持FLOAT16，FLOAT32, BF16。
- 输出 out 的数据类型与 输入 input 的数据类型一致。
- 张量数据格式支持ND。
