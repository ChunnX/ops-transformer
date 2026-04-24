# aclnnGemmV2

## 功能描述

### 算子功能
计算矩阵乘结果与输入矩阵 `c` 的线性组合。

### 计算公式
$$out = \alpha \cdot (op(a) @ op(b)) + \beta \cdot c$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.gemm_v2()` 函数形式提供：

```python
def gemm_v2(a, b, alpha, beta, c, transpose_a=False, transpose_b=False):
    """Return alpha * matmul(op(a), op(b)) + beta * c."""
```

## 使用案例
```python
import torch
import kernel_gen_ops

a = torch.randn(64, 128, device="npu", dtype=torch.float16)
b = torch.randn(128, 32, device="npu", dtype=torch.float16)
alpha = torch.ones(64, device="npu", dtype=torch.float16)
beta = torch.zeros(64, device="npu", dtype=torch.float16)
c = torch.zeros(64, 32, device="npu", dtype=torch.float32)
out = kernel_gen_ops.gemm_v2(a, b, alpha, beta, c, False, False)
```

## 约束与限制
- `a`、`b`、`alpha`、`beta` 支持 `float16`、`bfloat16`。
- `c` 和输出为 `float32`。
- 输入采用 ND 格式，`a`、`b`、`c` 为二维矩阵。
- `transpose_a`、`transpose_b` 控制参与矩阵乘前是否转置 `a`、`b`。
