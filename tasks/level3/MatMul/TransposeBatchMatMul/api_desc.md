# TransposeBatchMatMul 算子说明

## 功能描述

TransposeBatchMatMul 是一个支持转置的批量矩阵乘法算子。它允许通过 `perm_x1`、`perm_x2`、`perm_y` 属性对输入和输出的维度进行重排，从而在一次矩阵乘法操作中同时完成转置和乘法计算。支持可选的 bias 加法和 scale 量化。

## 接口定义

```python
kernel_gen_ops.transpose_batch_mat_mul(
    x1,              # 输入矩阵 A，3D Tensor
    x2,              # 输入矩阵 B，3D Tensor
    bias,            # 可选 bias，Tensor 或 None
    scale,           # 可选 scale，Tensor 或 None
    perm_x1,         # ListInt，x1 的维度重排，如 [0,1,2] 或 [1,0,2]
    perm_x2,         # ListInt，x2 的维度重排，如 [0,1,2] 或 [0,2,1]
    perm_y,          # ListInt，y 的维度重排，如 [1,0,2]
    enable_hf32,     # Bool，是否启用 HF32 模式
    batch_split_factor  # Int，batch 分割因子
)
```

## 参数说明

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| x1 | Tensor | 是 | 输入矩阵 A，3D，支持 fp16/fp32/bf16 |
| x2 | Tensor | 是 | 输入矩阵 B，3D，支持 fp16/fp32/bf16 |
| bias | Tensor | 否 | 可选 bias 向量 |
| scale | Tensor | 否 | 可选 scale 用于量化 |
| perm_x1 | ListInt | 是 | x1 的维度排列，支持 {0,1,2} 和 {1,0,2} |
| perm_x2 | ListInt | 是 | x2 的维度排列，支持 {0,1,2} 和 {0,2,1} |
| perm_y | ListInt | 是 | 输出的维度排列，支持 {1,0,2} |
| enable_hf32 | Bool | 否 | 是否启用 HF32 计算模式，默认 False |
| batch_split_factor | Int | 否 | batch 维度的分割因子，默认 1 |

## 约束限制

1. 输入 x1 和 x2 必须是 3D 张量
2. perm_x1 支持 {0,1,2}（不转置）和 {1,0,2}（交换 batch 和 m 维度）
3. perm_x2 支持 {0,1,2}（不转置）和 {0,2,1}（交换 k 和 n 维度）
4. perm_y 仅支持 {1,0,2}
5. 内轴大小（k, n）必须对齐到 16 的倍数
6. batch * k 的乘积不能超过 65536（ND2NZ 转换限制）
7. Einsum 模式（fp32 + 3D 输入 + perm_x1={1,0,2}）不支持 bias 和 scale

## 使用示例

```python
import torch
import kernel_gen_ops

# 基本 batch matmul with transpose
x1 = torch.randn(2, 128, 64, dtype=torch.float16, device="npu")  # (batch, m, k)
x2 = torch.randn(2, 64, 128, dtype=torch.float16, device="npu")  # (batch, k, n)
y = kernel_gen_ops.transpose_batch_mat_mul(
    x1, x2, None, None,
    [0, 1, 2], [0, 1, 2], [1, 0, 2],
    False, 1
)
# y shape: (128, 2, 128) = (m, batch, n)
```
