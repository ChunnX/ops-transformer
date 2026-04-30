# QuantBatchMatmulV4

## 功能描述

量化矩阵乘算子，支持多种量化模式和缩放方式，完成 INT8/INT4 量化的矩阵乘计算。

## 接口定义

```python
kernel_gen_ops.quant_batch_matmul_v4(
    x1, x2, bias, x1_scale, x2_scale, y_scale,
    x1_offset, x2_offset, y_offset, x2_table,
    dtype, compute_type, transpose_x1, transpose_x2, group_size)
```

## 参数说明

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| x1 | Tensor (INT8) | 是 | 左矩阵输入 |
| x2 | Tensor (INT8) | 是 | 右矩阵输入 |
| bias | Tensor (INT32/BF16/FLOAT) | 否 | 偏置 |
| x1_scale | Tensor (FLOAT) | 是 | x1 量化缩放因子 |
| x2_scale | Tensor (FLOAT) | 是 | x2 量化缩放因子 |
| y_scale | Tensor (FLOAT) | 是 | 输出缩放因子 |
| x1_offset | Tensor (FLOAT) | 否 | x1 量化偏移 |
| x2_offset | Tensor (FLOAT) | 否 | x2 量化偏移 |
| y_offset | Tensor (FLOAT) | 否 | 输出偏移 |
| x2_table | Tensor (INT8) | 否 | x2 查找表 |
| dtype | int | 是 | 输出数据类型 |
| compute_type | int | 否 | 计算类型 |
| transpose_x1 | bool | 否 | 是否转置 x1 的最后两维 |
| transpose_x2 | bool | 否 | 是否转置 x2 的最后两维 |
| group_size | int | 否 | 分组量化的组大小 |

## 计算公式

基础公式（per-tensor 量化）：
- `out = x1 @ x2 * x2_scale * x1_scale`
- 有偏置时：`out = (x1 @ x2 + bias) * x2_scale * x1_scale`

## 约束限制

- x1 和 x2 的 shape 维度范围为 2-6 维
- 支持 ascend910b 和 ascend910_93 平台
- 输出为 FLOAT16 或 BFLOAT16
