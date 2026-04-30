# MultiScaleDeformableAttnFunction 算子

## 功能描述

多尺度可变形注意力（Multi-Scale Deformable Attention）算子。该算子根据采样位置和注意力权重，在多尺度特征图上进行双线性插值采样，并将所有层、所有采样点的结果加权求和，得到最终输出。

具体地，对于每个 query q、head h：
1. 将归一化采样坐标 (u,v) ∈ [0,1] 映射到第 ℓ 层特征图像素坐标系
2. 在 4 个相邻整数网格点上进行双线性插值，得到采样特征向量
3. 乘以注意力权重，对所有层、所有采样点求和

## 接口定义

```python
kernel_gen_ops.multi_scale_deformable_attn_function(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor
) -> torch.Tensor
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | `torch.Tensor (float32)` | Value 特征，形状 `[B, N, H, D]`，N 为所有层空间大小之和 |
| `value_spatial_shapes` | `torch.Tensor (int32)` | 各层特征图空间大小，形状 `[L, 2]`，每行为 `(H_ℓ, W_ℓ)` |
| `value_level_start_index` | `torch.Tensor (int32)` | 各层在 N 维度上的起始索引，形状 `[L]` |
| `sampling_locations` | `torch.Tensor (float32)` | 归一化采样坐标，形状 `[B, Q, H, L, P, 2]`，范围 [0,1] |
| `attention_weights` | `torch.Tensor (float32)` | 注意力权重，形状 `[B, Q, H, L, P]` |
| **返回值** | `torch.Tensor (float32)` | 输出，形状 `[B, Q, H*D]` |

## 使用示例

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops

B, L, Q, H, P, D = 1, 1, 32, 2, 4, 32
spatial_shapes = torch.tensor([[4, 4]], dtype=torch.int32, device="npu")
level_start_index = torch.tensor([0], dtype=torch.int32, device="npu")
value = torch.randn(B, 16, H, D, dtype=torch.float32, device="npu")
sampling_locations = torch.rand(B, Q, H, L, P, 2, dtype=torch.float32, device="npu")
attention_weights = torch.rand(B, Q, H, L, P, dtype=torch.float32, device="npu")

output = kernel_gen_ops.multi_scale_deformable_attn_function(
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights
)
print(output.shape)  # torch.Size([1, 32, 64])
```

## 约束与限制

### 支持的数据类型
- value / sampling_locations / attention_weights / output: float32
- value_spatial_shapes / value_level_start_index: int32

### shape 约束
- numQueries >= 32
- numHeads <= 16
- numLevels <= 16
- numPoints <= 16
- embedDims 必须能被 8 整除且 <= 256