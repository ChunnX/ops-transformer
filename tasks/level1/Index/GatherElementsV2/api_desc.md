# aclnnGatherElementsV2

## 功能描述

### 算子功能
按照给定的维度 dim 和索引张量 index，从输入张量 x 中收集元素，输出到 y 中。

### 计算公式
$$y[i_0][i_1][\cdots][i_{d-1}][j][i_{d+1}][\cdots][i_{n-1}] = x[i_0][i_1][\cdots][i_{d-1}][\text{index}[i_0][i_1][\cdots][j][\cdots][i_{n-1}]]][i_{d+1}][\cdots][i_{n-1}]$$

其中 dim 为指定的维度 d。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.gather_elements_v2()` 函数形式提供：

```python
def gather_elements_v2(x: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """从输入张量x中按照index在指定维度dim上收集元素。"""
```

## 使用案例
```python
import torch
import kernel_gen_ops

x = torch.randn(4, 8, dtype=torch.float16).npu()
index = torch.randint(0, 8, (4, 8), dtype=torch.int32).npu()
dim = 1
result = kernel_gen_ops.gather_elements_v2(x, index, dim)
```

## 约束与限制
- 支持的数据类型：x 支持 float16、float32、int32、bfloat16；index 仅支持 int32
- 支持的数据格式：ND
- index 的形状必须与输出 y 的形状相同
- index 在非 dim 维度上的 size 不能超过 x 在对应维度上的 size
- index 在 dim 维度上的值范围必须在 [0, x.shape[dim]) 内
- 仅支持 ascend910b 和 ascend910_93 平台
