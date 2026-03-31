# aclnnRoiAlignRotatedGrad

## 功能描述

### 算子功能

根据旋转 RoI Align **前向**输出端的梯度 `grad_output`，以及与前向相同的 `rois` 与超参，计算对**特征图 x** 的梯度 `y_grad`（形状与 forward 的 `x` 一致，即 `(B, H, W, C)`）。

### 计算公式

与 RoiAlignRotated 前向中的双线性采样链式法则一致；实现上与 `roi_align_rotated_grad` kernel 的权重 `w1..w4` 及 atomic add 累加语义对齐。

## 接口定义

### Python 接口

通过 PyBind 暴露，编译后函数名为 `kernel_gen_ops.roi_align_rotated_grad`。

```python
def roi_align_rotated_grad(
    grad_output: Tensor,
    rois: Tensor,
    y_grad_shape: list[int],
    pooled_h: int,
    pooled_w: int,
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    clockwise: bool,
):
    """
    参数:
        grad_output: 上游梯度，形状 (num_rois, pooled_h, pooled_w, C)，与 forward 输出一致。
        rois: (6, num_rois)，列为 [batch_idx, cx, cy, w, h, theta]。
        y_grad_shape: 长度 4 的列表 [B, H, W, C]，对应 forward 输入特征 x 的形状。
        pooled_h / pooled_w / spatial_scale / sampling_ratio / aligned / clockwise: 与前向一致。

    返回:
        Tensor: 对 x 的梯度，形状 (B, H, W, C)。
    """
```

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
B, H, W, C = 2, 24, 32, 8
N, ph, pw = 4, 4, 4
grad_out = torch.randn(N, ph, pw, C, dtype=torch.float32, device=device)
rois = torch.randn(6, N, dtype=torch.float32, device=device)
y_shape = [B, H, W, C]

g = kernel_gen_ops.roi_align_rotated_grad(
    grad_out, rois, y_shape, ph, pw, 0.25, 0, True, False
)
```

## 约束与限制

- `grad_output`、`rois` 为 **float32**，ND。
- `y_grad_shape` 必须为 **4 个整数**，顺序 **[B, H, W, C]**（与 infershape / tiling 中 `GetListInt` 约定一致）。
- `grad_output.size(0) == rois.size(1)`；`grad_output.size(3) == C`。
- 通道数 **C 建议为 8 的倍数**，便于与 kernel 通道对齐路径一致。
