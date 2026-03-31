# aclnnRoiAlignRotated

## 功能描述

### 算子功能

对输入特征图上的**旋转 ROI** 做 RoI Align：将每个旋转框区域双线性采样到固定的 `pooled_h × pooled_w` 网格，输出 `(num_rois, pooled_h, pooled_w, channels)`。

### 计算公式

对每个 RoI，在局部坐标系 `(xx, yy)` 上按 `sampling_ratio` 或 `ceil(bin_size)` 确定采样网格，将点映射到特征图坐标 `(x, y)` 后做双线性插值并取平均（与 kernel 中 `bilinear_interpolate` 及边界处理一致）。

## 接口定义

### Python 接口

该操作通过 PyBind11 封装 C++ 实现，编译后函数名为 `kernel_gen_ops.roi_align_rotated`（由 `batch_compile.py` 将 `custom_pybind_api` 替换为 `roi_align_rotated`）。

```python
def roi_align_rotated(
    x: Tensor,
    rois: Tensor,
    pooled_h: int,
    pooled_w: int,
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    clockwise: bool,
):
    """
    旋转 RoI Align。

    参数:
        x (Tensor): 特征图，形状 (B, H, W, C)，NPU 上 float32，ND。
        rois (Tensor): 形状 (6, num_rois)，每列为 [batch_idx, center_x, center_y, width, height, theta]，
            坐标与宽高为原图/上一级坐标系下的量，需与 spatial_scale 配合映射到特征图。
        pooled_h (int): 输出池化高度。
        pooled_w (int): 输出池化宽度。
        spatial_scale (float): 坐标到特征图尺度的缩放因子。
        sampling_ratio (int): 每维采样点数；0 表示按 bin 尺寸向上取整自动决定。
        aligned (bool): True 时中心坐标加 -0.5 偏移（与 PyTorch RoIAlign 对齐语义一致）。
        clockwise (bool): True 时对角度取反。

    返回:
        Tensor: (num_rois, pooled_h, pooled_w, C)，dtype 与 x 一致。
    """
```

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
x = torch.randn(2, 24, 32, 8, dtype=torch.float32, device=device)
rois = torch.zeros(6, 4, dtype=torch.float32, device=device)
rois[0] = torch.tensor([0, 1, 0, 1], device=device)
rois[1] = torch.tensor([16.0, 18.0, 20.0, 22.0], device=device)
rois[2] = torch.tensor([12.0, 14.0, 16.0, 18.0], device=device)
rois[3] = torch.tensor([4.0, 5.0, 6.0, 7.0], device=device)
rois[4] = torch.tensor([4.0, 5.0, 6.0, 7.0], device=device)
rois[5] = torch.tensor([0.1, 0.2, -0.1, 0.0], device=device)

y = kernel_gen_ops.roi_align_rotated(
    x, rois, 4, 4, 0.25, 0, True, False
)
```

## 约束与限制

- 输入 `x`、`rois` 数据类型为 **float32**（与 `op_host` 中 `DT_FLOAT` 一致）。
- `x` 布局为 **(B, H, W, C)**；`rois` 为 **(6, num_rois)**。
- `num_rois > 0`；`channels` 建议为 **8 的倍数**（与 tiling 中通道对齐逻辑一致）。
- 张量格式 ND；自定义 OPP 路径由框架设置 `ASCEND_CUSTOM_OPP_PATH`。
