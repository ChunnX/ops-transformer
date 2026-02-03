# aclnnForeachMinimumScalarList

## 功能描述

### 算子功能
`aclnnForeachMinimumScalarList` 对张量列表x和标量列表scalars执行逐元素比较，计算每个元素对应的最大值。

### 计算公式

$$
x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
$$

$$
{\rm y}_i = \min(x_i, \text{scalars}_i) (i=0,1,...n-1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_minimum_scalar_list()` 函数形式提供：

```python
def foreach_minimum_scalar_list(x: List[Tensor], scalars: List[Tensor]):
    """
    对张量列表中的每个张量执行取绝对值操作。
    
    参数:
        x (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型。
                          张量shape维度不高于8维，数据格式支持ND。支持非连续的Tensor。
        scalars (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须与x的数据类型具有一定对应关系：当入参x的数据类型为FLOAT32、BFLOAT16、FLOAT16时，scalars的数据类型支持FLOAT32、DOUBLE；当入参x的数据类型为INT32时，scalars的数据类型支持INT64。
                           元素个数与入参x中Tensor的个数相等。
        
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
x_list = [
    torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float),
    torch.tensor([[-4.0, 5.0], [-6.0, 7.0]], dtype=torch.float)
]

scalars_list = [
    torch.tensor([1.5], dtype=torch.float),
    torch.tensor([1.0], dtype=torch.float)
]

result_list = kernel_gen_ops.foreach_minimum_scalar_list(x_list, scalars_list)
```

## 约束与限制

- 入参 x 列表中所有张量必须具有相同的数据类型
- 当入参 x 的数据类型为FLOAT32、BFLOAT16、FLOAT16时，scalars的数据类型支持FLOAT32、DOUBLE；当入参x的数据类型为INT32时，scalars的数据类型支持INT64。
- 入参 scalars 的元素个数与入参 x 中Tensor的个数相等