# aclnnForeachPowList

## 功能描述

### 算子功能
`aclnnForeachPowList` 对两个张量列表对应位置元素执行插值计算，其中张量列表weight是插值系数。

### 计算公式

$$
x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}]\\
x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
$$

$$
y_i = x1_i^{x2_i}  (i=0,1,...n-1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_pow_list()` 函数形式提供：

```python
def foreach_pow_list(x1: List[Tensor], x2: List[Tensor]):
    """
    对张量列表中的每个张量执行取绝对值操作。
    
    参数:
        x1 (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型。
                           张量shape维度不高于8维且与入参x2shape一致，数据格式支持ND。支持非连续的Tensor。
        x2 (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型且数据类型需要与x1一致。
                           张量shape维度不高于8维且与入参x1的shape一致，数据格式支持ND。支持非连续的Tensor。
        
    返回:
        List[Tensor]: 一个新的Device侧张量列表。
    
    注意:
        - 张量shape维度不高于8维，数据格式支持ND
        - 支持非连续的Tensor
        - 输出张量与输入张量具有相同的形状、数据类型和数据格式
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建具有相同数据类型的张量列表
x1 = [
    torch.tensor([1.0, 2.0, 3.0], dtype=torch.float),
    torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float)
]

x2 = [
    torch.tensor([1.5, 2.5, 3.5], dtype=torch.float),
    torch.tensor([[4.5, 5.5], [6.5, 7.5]], dtype=torch.float)
]


result_list = kernel_gen_ops.foreach_pow_list(x1, x2)
```

## 约束与限制

- x1, x2的shape维度不高于8维。
- x1, x2的shape一致
- x1, x2的数据格式一致。
- x2取值的绝对值必须小于等于(2^31-1)
