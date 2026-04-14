# aclnnUpsampleNearestExact3dGrad

## 功能描述

### 算子功能
计算三维最近邻精确上采样（nearest exact）的反向传播梯度。对于最近邻精确模式，源索引计算公式为 `src_idx = floor((dst_idx + 0.5) * scale)`，其中 `scale = input_size / output_size`。

### 计算公式

对于输出位置 $(d_o, h_o, w_o)$，计算其在输入中的源索引：

$$d_{src} = \lfloor (d_o + 0.5) \times \frac{iD}{oD} \rfloor$$
$$h_{src} = \lfloor (h_o + 0.5) \times \frac{iH}{oH} \rfloor$$
$$w_{src} = \lfloor (w_o + 0.5) \times \frac{iW}{oW} \rfloor$$

然后将 `grad_output` 中对应位置的梯度累加到 `grad_input` 的对应源位置。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_nearest_exact3d_grad()` 函数形式提供：

```python
def upsample_nearest_exact3d_grad(grad_output: torch.Tensor,
                                   output_size: List[int],
                                   input_size: List[int],
                                   scales_d: float = 0.0,
                                   scales_h: float = 0.0,
                                   scales_w: float = 0.0) -> torch.Tensor:
    """计算三维最近邻精确上采样的反向梯度。"""
```

**参数说明**：
- `grad_output`：反向传播的输入梯度张量，形状为 `(N, C, oD, oH, oW)`
- `output_size`：输出尺寸 `[N, C, oD, oH, oW]`
- `input_size`：输入尺寸 `[N, C, iD, iH, iW]`
- `scales_d`：深度方向缩放因子（0 表示自动计算）
- `scales_h`：高度方向缩放因子（0 表示自动计算）
- `scales_w`：宽度方向缩放因子（0 表示自动计算）

**返回值**：
- 形状为 `(N, C, iD, iH, iW)` 的梯度张量

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

# 创建输入梯度
N, C, iD, iH, iW = 1, 3, 4, 4, 4
oD, oH, oW = 8, 8, 8
grad_output = torch.randn(N, C, oD, oH, oW, dtype=torch.float32).npu()

# 计算反向梯度
grad_input = kernel_gen_ops.upsample_nearest_exact3d_grad(
    grad_output,
    [N, C, oD, oH, oW],  # output_size
    [N, C, iD, iH, iW],  # input_size
)
print(grad_input.shape)  # (1, 3, 4, 4, 4)
```

## 约束与限制

- **支持的数据类型**：`float16`、`bfloat16`、`float32`
- **支持的数据格式**：`ND`（5维 NCDHW 布局）
- **缩放因子上限**：各维度缩放因子不超过 50
- **输入尺寸约束**：输入和输出的 D、H、W 维度必须大于 0
- **芯片支持**：ascend910b、ascend910_93
