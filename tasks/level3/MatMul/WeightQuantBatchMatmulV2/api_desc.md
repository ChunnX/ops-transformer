# WeightQuantBatchMatmulV2

## 功能描述

权重伪量化批量矩阵乘法算子。对 INT8/INT4 量化的权重矩阵进行反量化后，与输入矩阵执行批量矩阵乘法，可选地加上偏置。支持按通道或按组的伪量化参数。

## 接口定义

```python
kernel_gen_ops.weight_quant_batch_matmul_v2(
    x,                    # 输入矩阵 [batch, m, k] (float16/bfloat16)
    weight,               # 量化权重 [k, n] 或 [n, k] (int8/int4)
    antiquant_scale,      # 反量化缩放因子 (float16/bfloat16)
    antiquant_offset,     # 反量化偏移 (可选, float16/bfloat16)
    quant_scale,          # 量化缩放因子 (可选, float/uint64)
    quant_offset,         # 量化偏移 (可选, float)
    bias,                 # 偏置向量 (可选, float16/float)
    transpose_x,          # 是否转置 x (bool)
    transpose_weight,     # 是否转置 weight (bool)
    antiquant_group_size, # 反量化分组大小 (int)
    dtype,                # 数据类型指示 (int)
    inner_precise         # 内部精度模式 (int)
) -> Tensor              # 输出 [batch, m, n] (float16/bfloat16)
```

## 参数说明

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| x | Tensor | 是 | 输入张量，支持 float16/bfloat16 |
| weight | Tensor | 是 | 量化权重，支持 int8/int4 |
| antiquant_scale | Tensor | 是 | 反量化缩放因子 |
| antiquant_offset | Tensor | 否 | 反量化偏移 |
| quant_scale | Tensor | 否 | 输出量化缩放因子 |
| quant_offset | Tensor | 否 | 输出量化偏移 |
| bias | Tensor | 否 | 偏置向量 |
| transpose_x | bool | 否 | 是否转置 x 矩阵，默认 False |
| transpose_weight | bool | 否 | 是否转置 weight 矩阵，默认 False |
| antiquant_group_size | int | 否 | 反量化分组大小，0 表示按通道，默认 0 |
| dtype | int | 否 | 数据类型指示，默认 -1 |
| inner_precise | int | 否 | 内部精度模式，默认 0 |

## 约束限制

- 仅支持 ascend910b 和 ascend910_93 芯片
- x 支持 float16 和 bfloat16
- weight 支持 int8 和 int4
- 当 antiquant_group_size > 0 时为按组反量化模式
- 输出 shape: [batch, m, n]，其中 m/n 由输入和 transpose 参数决定

## 使用示例

```python
import torch
import kernel_gen_ops

x = torch.randn(1, 64, 128, dtype=torch.float16).npu()
weight = torch.randint(-127, 127, (128, 64), dtype=torch.int8).npu()
antiquant_scale = torch.randn(1, 128, dtype=torch.float16).npu()

result = kernel_gen_ops.weight_quant_batch_matmul_v2(
    x, weight, antiquant_scale, None, None, None, None,
    False, False, 0, -1, 0)
```
