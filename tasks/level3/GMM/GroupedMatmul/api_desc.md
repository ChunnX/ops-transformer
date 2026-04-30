# GroupedMatmul 算子

## 功能描述

GroupedMatmul（分组矩阵乘法）算子，支持将多个矩阵乘法操作按组批量执行。支持 BF16/FP16 非量化、INT8/INT4 量化、反量化（antiquant）等多种计算模式。主要用于 MoE（Mixture of Experts）模型中的专家并行矩阵乘法。

## 接口定义

```python
kernel_gen_ops.grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    antiquant_scale: torch.Tensor,
    antiquant_offset: torch.Tensor,
    group_list: torch.Tensor,
    per_token_scale: torch.Tensor,
    split_item: int,
    group_type: int,
    group_list_type: int,
    act_type: int,
    tuning_config: List[int]
) -> torch.Tensor
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `torch.Tensor` | 输入矩阵，形状 `(M, K)` 或 tensor list |
| `weight` | `torch.Tensor` | 权重矩阵，形状 `(N, K)` 或 tensor list |
| `bias` | `torch.Tensor` | 偏置（可选），形状 `(N,)` |
| `scale` | `torch.Tensor` | 量化 scale（量化模式使用） |
| `offset` | `torch.Tensor` | 量化 offset（量化模式使用） |
| `antiquant_scale` | `torch.Tensor` | 反量化 scale（反量化模式使用） |
| `antiquant_offset` | `torch.Tensor` | 反量化 offset（反量化模式使用） |
| `group_list` | `torch.Tensor` | 分组列表，INT64，cumsum 或 count 格式 |
| `per_token_scale` | `torch.Tensor` | 逐 token 的 scale（per-token 量化使用） |
| `split_item` | `int` | 分组策略：0=单X多W, 1=多X多W, 2=单X单W+group_list, 3=多X单W+group_list |
| `group_type` | `int` | 分组维度，默认 -1 |
| `group_list_type` | `int` | group_list 格式：0=cumsum, 2=count |
| `act_type` | `int` | 激活函数类型：0=无 |
| `tuning_config` | `List[int]` | 调优配置参数 |
| **返回值** | `torch.Tensor` | 输出矩阵，形状 `(M, N)` |

## 使用示例

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops

# 基本非量化矩阵乘法
M, K, N = 16, 64, 32
x = torch.randn(M, K, dtype=torch.float16, device="npu")
weight = torch.randn(N, K, dtype=torch.float16, device="npu")
bias = torch.empty(0, dtype=torch.float16, device="npu")
scale = torch.empty(0, dtype=torch.float32, device="npu")
offset = torch.empty(0, dtype=torch.float32, device="npu")
antiquant_scale = torch.empty(0, dtype=torch.float16, device="npu")
antiquant_offset = torch.empty(0, dtype=torch.float16, device="npu")
group_list = torch.empty(0, dtype=torch.int64, device="npu")
per_token_scale = torch.empty(0, dtype=torch.float32, device="npu")

y = kernel_gen_ops.grouped_matmul(
    x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
    group_list, per_token_scale,
    0, -1, 0, 0, [0])

print(y.shape)  # torch.Size([16, 32])
```

## 约束与限制

### 支持的数据类型
- x: float16, bfloat16, int8, int4
- weight: float16, bfloat16, int8, int4
- y: float16, bfloat16, int32

### 张量格式
- 输入/输出均为 ND 格式
- 部分量化模式支持 FRACTAL_NZ 权重格式

### 其他限制
- K 维度必须 16 字节对齐
- split_item=2/3 时需要提供 group_list
