# MultiScaleDeformableAttentionGrad 算子

## 功能描述

多尺度可变形注意力（Multi-Scale Deformable Attention）的反向传播算子。给定前向传播的输入和来自上层的梯度 `grad_output`，计算对 `value`、`sampling_locations` 和 `attention_weights` 三个输入的梯度。

核心计算是对双线性插值采样过程求梯度：在每个查询位置，根据归一化的采样坐标在多尺度特征图上进行双线性插值，反向传播时需计算对采样值、采样坐标和注意力权重的梯度。

## 接口定义

```python
kernel_gen_ops.multi_scale_deformable_attention_grad(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    grad_output: torch.Tensor
) -> List[torch.Tensor]
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | `torch.Tensor` (float32) | 输入特征图，形状 `[B, N, H, D]`，N 为总空间尺寸（所有层级 H_l×W_l 之和） |
| `value_spatial_shapes` | `torch.Tensor` (int32) | 各层级空间形状，形状 `[L, 2]`，每行为 `(H_l, W_l)` |
| `value_level_start_index` | `torch.Tensor` (int32) | 各层级在 N 维度的起始偏移，形状 `[L]` |
| `sampling_locations` | `torch.Tensor` (float32) | 采样坐标（归一化到 [0,1]），形状 `[B, nH, L, P, 2, nQ]` |
| `attention_weights` | `torch.Tensor` (float32) | 注意力权重，形状 `[B, nH, L, P, nQ]` |
| `grad_output` | `torch.Tensor` (float32) | 上层梯度，形状 `[B, nQ, nH, D]` |
| **返回值** | `List[torch.Tensor]` | 三个输出梯度张量 |

### 返回值

| 索引 | 名称 | 形状 | 说明 |
|------|------|------|------|
| 0 | grad_value | `[B, N, H, D]` | 对 value 的梯度 |
| 1 | grad_sampling_locations | `[B, nH, L, P, 2, nQ]` | 对采样坐标的梯度 |
| 2 | grad_attention_weights | `[B, nH, L, P, nQ]` | 对注意力权重的梯度 |

其中 B=批次大小，N=总空间尺寸，H=注意力头数，D=嵌入维度，L=尺度层级数，P=采样点数，nQ=查询数。

## 使用示例

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops

B, nH, D, L, nQ, nP = 1, 2, 8, 1, 3, 2
spatial_shapes = torch.tensor([[4, 4]], dtype=torch.int32)
level_start_index = torch.tensor([0], dtype=torch.int32)
N = 16  # 4*4

value = torch.rand(B, N, nH, D, dtype=torch.float32, device="npu")
sampling_locations = torch.rand(B, nH, L, nP, 2, nQ, dtype=torch.float32, device="npu")
attention_weights = torch.rand(B, nH, L, nP, nQ, dtype=torch.float32, device="npu")
grad_output = torch.ones(B, nQ, nH, D, dtype=torch.float32, device="npu")

grad_value, grad_loc, grad_weight = kernel_gen_ops.multi_scale_deformable_attention_grad(
    value, spatial_shapes, level_start_index,
    sampling_locations, attention_weights, grad_output)
```

## 约束与限制

### 支持的数据类型
- value, sampling_locations, attention_weights, grad_output, 所有输出: float32

### 张量格式
- 所有输入/输出为 ND 格式

### 其他限制
- embed_dims (D) 必须为 8 的倍数且 <= 256
- num_queries (nQ) < 500000
- num_levels (L) <= 16
- num_heads (nH) <= 16
- num_points (nP) <= 16
- sampling_locations 中的坐标值应在 [0, 1] 范围内
