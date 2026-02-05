# aclnnSegsum (ExpSegsum)

## 功能描述

### 算子功能
进行分段和计算，生成对角线为 0 的半可分矩阵，上三角为 -inf，再对元素取 exp。输入 3D/4D (..., N)，输出 (..., N, N)。

### 计算公式
见 CANN 文档 aclnnExpSegsum：下三角为沿最后一维的 cumsum 后 exp，对角为 1，上三角为 0。

## 接口定义

### Python 接口
通过 PyBind11 封装，在 Python 中以 `kernel_gen_ops.segsum()` 提供：

```python
def segsum(x: Tensor) -> Tensor:
    """
    参数:
        x (Tensor): 输入张量，3D 或 4D，最后一维为 N，支持 float16/float32/bfloat16。
    返回:
        Tensor: 输出张量，形状 (..., N, N)。
    """
```

## 约束与限制
- 输入支持 FLOAT16、FLOAT、BFLOAT16。
- 输入为 3D 或 4D，输出比输入多一维，最后一维大小为输入的最后一维大小。
