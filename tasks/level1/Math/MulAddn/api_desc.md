# aclnnMulAddn

## 功能描述

### 算子功能
该算子实现 n>=2个mul和addn融合计算，减少搬运时间和内存的占用。

### 计算公式

输入x1，  x2为变长输入，为N个tensor组成的列表。x1输入中的每个tensor shape都为[B, M, 1], x2输入中的每个tensor shape都为[B, 1, K], y为输出，shape为[B, M, K]; N对应为addn算子的n数量，也为融合算子融合mul的数量。

$$
x1 * x2 = y
$$

$$
x1 = [[B, M, 1], [B, M, 1],...,[B, M, 1]] (共N个[B, M, 1])
$$

$$
x2 = [[B, 1, K],[B, 1, K],...,[B, 1, K]] (共N个[B, 1, K])
$$

$$
[[B, M, 1], [B, M, 1],...,[B, M, 1]] * [[B, 1, K],[B, 1, K],...,[B, 1, K]] = y
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.mul_addn()` 函数形式提供：


```python
def mul_addn(x1: List[Tensor], x2: List[Tensor], n: int):
    """
    实现n>=2个mul和addn融合计算，减少搬运时间和内存的占用。

    参数:
        x1 (List[Tensor]): 输入张量列表，Device侧的张量，数据格式支持ND。
        x2 (List[Tensor]): 输入张量列表，Device侧的张量，数据格式支持ND。

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

# 使用 mul_addn 执行激活操作
result = kernel_gen_ops.mul_addn(x1, x2)
```
## 约束与限制

- 输入 x1 中Tensor的shape为[B, M, 1]。
- 输入 x2 中Tensor的shape为[B, 1, K]。
- 输出 y 的shape为[B, M, K]。
- N >= 2。
- K <= 2040。
- 输入 x2 中Tensor的数据类型和输入 x1 的Tensor数据类型一致。
- 输出 y 的数据类型和输入 x1 的Tensor数据类型一致。
- 张量数据格式支持ND。
