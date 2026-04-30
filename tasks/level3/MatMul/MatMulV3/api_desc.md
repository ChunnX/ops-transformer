# MatMulV3 算子

## 功能描述

MatMulV3 实现通用矩阵乘法（GEMM），支持 FP16、FP32、BF16 数据类型，支持转置和偏置。该算子是昇腾 NPU 上的高性能矩阵乘法实现，支持动态 shape 和多核并行。

## 接口定义

```python
kernel_gen_ops.mat_mul_v3(x1, x2, bias, offset_w, transpose_x1, transpose_x2, offset_x, opImplMode)
```

## 参数说明

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| x1 | Tensor | 是 | 左矩阵，shape 为 (M, K) 或 (K, M)（若 transpose_x1=True） |
| x2 | Tensor | 是 | 右矩阵，shape 为 (K, N) 或 (N, K)（若 transpose_x2=True） |
| bias | Tensor | 否 | 偏置向量，shape 为 (N,)，无偏置时传入空 Tensor |
| offset_w | Tensor | 否 | 量化偏移权重，通常传入空 Tensor（int8 类型） |
| transpose_x1 | bool | 否 | 是否转置 x1，默认 False |
| transpose_x2 | bool | 否 | 是否转置 x2，默认 False |
| offset_x | int | 否 | 量化偏移值，默认 0 |
| opImplMode | int | 否 | 实现模式，默认 1（高性能模式） |

## 输出

| 参数 | 类型 | 说明 |
|------|------|------|
| y | Tensor | 矩阵乘法结果，shape 为 (M, N) |

## 约束限制

- 支持 dtype：FP16、FP32、BF16
- 输入 x1 和 x2 的 K 轴维度必须一致
- bias 的最后一维必须与 N 维度一致
- 仅支持 ascend910b 和 ascend910_93 平台
- bias 仅支持 FP16/FP32 类型

## 使用示例

```python
import torch
import torch_npu
import kernel_gen_ops

# 基本矩阵乘法 (M=64, K=128, N=64)
x1 = torch.randn(64, 128, dtype=torch.float16, device="npu:0")
x2 = torch.randn(128, 64, dtype=torch.float16, device="npu:0")
bias = torch.empty(0, dtype=torch.float16, device="npu:0")
offset_w = torch.empty(0, dtype=torch.int8, device="npu:0")
result = kernel_gen_ops.mat_mul_v3(x1, x2, bias, offset_w, False, False, 0, 1)
```
