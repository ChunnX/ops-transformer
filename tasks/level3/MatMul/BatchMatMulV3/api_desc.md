# aclnnBatchMatMulV3

## 功能描述

### 算子功能
批量矩阵乘法算子，对两个三维张量进行批量矩阵乘法运算，支持转置和偏移。

### 计算公式
$$y = x1 \times x2$$

其中 $x1$ 的形状为 $(batch, M, K)$，$x2$ 的形状为 $(batch, K, N)$，输出 $y$ 的形状为 $(batch, M, N)$。当 `adj_x1=True` 时对 $x1$ 进行转置，当 `adj_x2=True` 时对 $x2$ 进行转置。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.batch_mat_mul_v3()` 函数形式提供：

```python
def batch_mat_mul_v3(x1: torch.Tensor,
                     x2: torch.Tensor,
                     bias: torch.Tensor,
                     offset_w: torch.Tensor,
                     adj_x1: bool,
                     adj_x2: bool,
                     offset_x: int,
                     enable_hf32: bool) -> torch.Tensor:
    """批量矩阵乘法"""
```

## 使用案例
```python
import torch
import torch_npu
import kernel_gen_ops

torch.npu.set_device(0)
device = torch.device("npu:0")

x1 = torch.randn(2, 128, 64, dtype=torch.float16, device=device)
x2 = torch.randn(2, 64, 128, dtype=torch.float16, device=device)
bias = torch.empty(0, dtype=torch.float16, device=device)
offset_w = torch.empty(0, dtype=torch.int8, device=device)

result = kernel_gen_ops.batch_mat_mul_v3(x1, x2, bias, offset_w, False, False, 0, False)
print(result.shape)  # torch.Size([2, 128, 128])
```

## 约束与限制
- 支持的数据类型：float16、float32、bfloat16
- 支持的数据格式：ND
- 输入 x1 和 x2 的 batch 维度必须相同
- x1 的最后一维（或倒数第二维，若 adj_x1=True）必须等于 x2 的倒数第二维（或最后一维，若 adj_x2=True）
