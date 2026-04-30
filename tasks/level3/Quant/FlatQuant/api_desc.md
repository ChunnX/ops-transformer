# FlatQuant 算子

## 功能描述

FlatQuant 算子通过 Kronecker 分解对输入张量进行矩阵变换后执行对称 int4 量化。对于每个批次 k，计算 y[k] = p1 @ x[k] @ p2^T，然后找到最大绝对值作为量化缩放因子，将变换结果量化到 int4 范围 [-7, 7]。

## 接口定义

```python
kernel_gen_ops.flat_quant(
    x: torch.Tensor,               # [K, M, N] 输入张量
    kronecker_p1: torch.Tensor,     # [M, M] Kronecker 因子 P1
    kronecker_p2: torch.Tensor,     # [N, N] Kronecker 因子 P2
    clip_ratio: float,              # 裁剪比例 (0, 1]
    dst_dtype: int                  # 目标量化类型 (DT_INT4=30)
) -> tuple[torch.Tensor, torch.Tensor]  # (quantized_out, quant_scale)
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `torch.Tensor` | 输入张量，形状 [K, M, N]，K 为批次数，M 和 N 为矩阵维度 |
| `kronecker_p1` | `torch.Tensor` | Kronecker 分解因子 P1，形状 [M, M] |
| `kronecker_p2` | `torch.Tensor` | Kronecker 分解因子 P2，形状 [N, N] |
| `clip_ratio` | `float` | 裁剪比例，取值范围 (0, 1]，默认 1.0 |
| `dst_dtype` | `int` | 目标量化数据类型，默认 DT_INT4 (30) |
| **返回值 1** | `torch.Tensor` | 量化输出 out，形状 [K, M, N]，int8 存储（int4 范围 [-7, 7]） |
| **返回值 2** | `torch.Tensor` | 量化缩放因子 quant_scale，形状 [K]，float32 |

## 使用示例

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops

K, M, N = 4, 32, 32
x = torch.randn(K, M, N, dtype=torch.float16, device="npu")
p1 = torch.randn(M, M, dtype=torch.float16, device="npu")
p2 = torch.randn(N, N, dtype=torch.float16, device="npu")

out, scale = kernel_gen_ops.flat_quant(x, p1, p2, clip_ratio=1.0, dst_dtype=30)

print(out.shape)   # torch.Size([4, 32, 32])
print(scale.shape) # torch.Size([4])
```

## 约束与限制

### 支持的数据类型
- 输入 x, kronecker_p1, kronecker_p2: float16, bfloat16
- 输出 out: int4 (存储为 int8)
- 输出 quant_scale: float32

### 张量格式
- 所有输入/输出必须为 ND 格式

### Shape 约束
- x 必须为 3D: [K, M, N]
- kronecker_p1 必须为 2D: [M, M]
- kronecker_p2 必须为 2D: [N, N]
- K <= 262144, M <= 256, N <= 256

### 其他限制
- clip_ratio 取值范围: (0, 1]
- 输入张量的数据类型必须一致
