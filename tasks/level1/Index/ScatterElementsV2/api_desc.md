# aclnnScatterElementsV2

## 功能描述

### 算子功能
将 `updates` 张量中的值按 `indices` 指定的位置写入（或聚合写入）到 `var` 张量的指定轴上，返回修改后的张量。

### 计算公式
对于 reduction="none"：
$$output[i_0, i_1, ..., i_{axis}, ..., i_{n-1}] = updates[i_0, i_1, ..., j, ..., i_{n-1}]$$
其中 $i_{axis} = indices[i_0, i_1, ..., j, ..., i_{n-1}]$，其余位置保持 var 原值不变。

对于 reduction="add"：
$$output[i_0, ..., i_{axis}, ..., i_{n-1}] = var[i_0, ..., i_{axis}, ..., i_{n-1}] + \sum_{j} updates[i_0, ..., j, ..., i_{n-1}]$$
其中 $i_{axis} = indices[i_0, ..., j, ..., i_{n-1}]$。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.scatter_elements_v2()` 函数形式提供：

```python
def scatter_elements_v2(var: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor,
                         axis: int = -1, reduction: str = "none") -> torch.Tensor:
    """将 updates 按 indices 散射写入 var 的指定轴。"""
```

## 使用案例
```python
import torch
import kernel_gen_ops

var = torch.randn(4, 8)
indices = torch.randint(0, 8, (4, 4), dtype=torch.int64)
updates = torch.randn(4, 4)
result = kernel_gen_ops.scatter_elements_v2(var, indices, updates, axis=-1, reduction="none")
```

## 约束与限制
- 支持的数据类型（var/updates）：float32、float16、bfloat16、int32、int8、uint8、bool
- 支持的索引数据类型（indices）：int64、int32
- 支持的数据格式：ND
- reduction 支持 "none" 和 "add" 两种模式
- indices 各维度大小 <= var 对应维度大小
- updates 形状与 indices 形状一致
