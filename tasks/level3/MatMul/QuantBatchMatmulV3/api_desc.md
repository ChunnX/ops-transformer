# QuantBatchMatmulV3 算子

## 功能描述

完成量化的批量矩阵乘计算。支持 INT8/INT4 量化的输入，通过 scale 和 offset 参数进行反量化后执行矩阵乘法，支持 bias 加法和 per-token scale。

计算公式：
- 无 bias：`out = (x1 @ x2) * scale + offset`
- 有 INT32 bias：`out = ((x1 @ x2) + bias) * scale + offset`
- 有 BF16/FP32 bias：`out = (x1 @ x2) * scale + bias`

## 接口定义

```python
kernel_gen_ops.quant_batch_matmul_v3(
    x1,              # INT8/INT4 输入张量 (ND 或 FRACTAL_NZ)
    x2,              # INT8/INT4 量化权重张量
    scale,           # 反量化 scale (float32/bf16/int64/uint64)
    offset,          # 可选，反量化 offset (float32)
    bias,            # 可选，偏置 (bf16/float16/float32/int32)
    pertoken_scale,  # 可选，per-token scale (float32)
    dtype,           # 输出数据类型
    transpose_x1,    # 是否转置 x1
    transpose_x2,    # 是否转置 x2
    group_size       # 分组量化大小
)
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| x1 | INT8/INT4 Tensor | 输入矩阵，shape [..., M, K] |
| x2 | INT8/INT4 Tensor | 量化权重，shape [..., K, N] |
| scale | Float Tensor | 反量化 scale，可 per-channel |
| offset | Float Tensor (可选) | 反量化 offset |
| bias | Tensor (可选) | 偏置，shape [..., M, N] |
| pertoken_scale | Float Tensor (可选) | Per-token 缩放因子 |
| dtype | int | 输出数据类型 |
| transpose_x1 | bool | 是否转置 x1 (默认 false) |
| transpose_x2 | bool | 是否转置 x2 (默认 false) |
| group_size | int | 分组量化大小 (默认 0) |

## 约束限制

- 输入 x1 和 x2 维度范围：2D ~ 6D
- 支持 ND 和 FRACTAL_NZ 格式
- 输出数据类型由 dtype 参数决定

## 使用示例

```python
import torch
import kernel_gen_ops

x1 = torch.randint(-128, 127, (16, 64), dtype=torch.int8)
x2 = torch.randint(-128, 127, (64, 32), dtype=torch.int8)
scale = torch.rand(32, dtype=torch.float32) * 0.1

result = kernel_gen_ops.quant_batch_matmul_v3(
    x1, x2, scale, None, None, None,
    0, False, False, 0
)
```
