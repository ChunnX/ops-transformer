# aclnnPows

## 功能描述

### 算子功能
该算子对x1中的每个元素应用指数为x2的幂运算。

### 计算公式

$$
out_i = x1_i^{x2}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.pows()` 函数形式提供：


```python
def pows(x1: Tensor, x2: Tensor):
    """
    实现 pows 操作。

    参数:
        x1 (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        x2 (Tensor): 输入张量，Device侧的张量，数据格式支持ND。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变，输出shape和输入x1的shape一致。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x1 = torch.randn(4, 8, dtype=torch.float32)
x2 = torch.tensor([1.2], dtype=torch.float32)

result = kernel_gen_ops.pows(x1, x2)
```
## 约束与限制

- 输出 out 的shape和输入 x1 shape一致。
- 输入 x2 的元素数量必须为1，且数值只能为[-2, -1, -0.5, 0.5, 2, 3]。
- 张量数据格式支持ND。
- 输入 x1 的数据类型只支持FLOAT16，FLOAT32, BF16。
- 输出 out 的数据类型和输入 x1 的数据类型一致。
