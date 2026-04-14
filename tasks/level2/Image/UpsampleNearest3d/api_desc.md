# aclnnUpsampleNearest3d

## 功能描述

### 算子功能
对5D输入张量执行最近邻3D上采样，根据给定的输出尺寸或缩放因子对深度(D)、高度(H)、宽度(W)维度进行上采样。

### 计算公式
对于输出张量的每个位置 $(o_d, o_h, o_w)$，计算对应输入索引：
$$src\_idx = \lfloor dst\_idx \times scale \rfloor$$
其中 $scale = \frac{input\_size}{output\_size}$。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.upsample_nearest3d()` 函数形式提供：

```python
def upsample_nearest3d(x: torch.Tensor,
                        output_size: List[int],
                        scale_d: float = 0.0,
                        scale_h: float = 0.0,
                        scale_w: float = 0.0) -> torch.Tensor:
    """
    对输入张量执行最近邻3D上采样。

    参数:
        x: 输入张量，形状为 (N, C, D_in, H_in, W_in)
        output_size: 输出尺寸 [output_d, output_h, output_w]
        scale_d: 深度方向缩放因子（可选，默认0表示自动计算）
        scale_h: 高度方向缩放因子（可选，默认0表示自动计算）
        scale_w: 宽度方向缩放因子（可选，默认0表示自动计算）

    返回:
        上采样后的张量，形状为 (N, C, output_d, output_h, output_w)
    """
```

## 使用案例
```python
import torch
import torch_npu
import kernel_gen_ops

# 创建输入张量 (N=1, C=3, D=4, H=6, W=8)
x = torch.randn(1, 3, 4, 6, 8).npu()

# 上采样到 (D=8, H=12, W=16)
output = kernel_gen_ops.upsample_nearest3d(x, [8, 12, 16])
print(output.shape)  # torch.Size([1, 3, 8, 12, 16])
```

## 约束与限制
- 支持的数据类型：float16、float32、bfloat16
- 支持的数据格式：ND（5D张量，NCDHW格式）
- 输入张量必须是5D
- output_size 必须为长度3的列表，对应 [output_d, output_h, output_w]
- scale_d/scale_h/scale_w 为可选参数，默认值0表示自动根据 input_size/output_size 计算缩放因子
- 输入各维度大小不应超过 INT32_MAX
