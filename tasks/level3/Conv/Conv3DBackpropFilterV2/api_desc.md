# aclnnConv3DBackpropFilterV2

## 功能描述

### 算子功能
计算3D卷积的反向传播中对滤波器（filter）的梯度。

### 计算公式
$$
\frac{\partial L}{\partial W} = \sum_{n} \sum_{k} \sum_{d} \sum_{h} \sum_{w} \frac{\partial L}{\partial y_{n,k,d,h,w}} \cdot x_{n,:,d \cdot s_d - p_d + k \cdot d_d, h \cdot s_h - p_h + k \cdot d_h, w \cdot s_w - p_w + k \cdot d_w}
$$

其中：
- $W$ 是滤波器权重
- $x$ 是输入特征图
- $y$ 是输出特征图
- $s$ 是步长
- $p$ 是填充
- $d$ 是膨胀率

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.conv3d_backprop_filter_v2()` 函数形式提供：

```python
def conv3d_backprop_filter_v2(
    x: torch.Tensor,
    filter_size: torch.Tensor,
    out_backprop: torch.Tensor,
    strides: List[int],
    pads: List[int],
    dilations: List[int] = [1, 1, 1],
    groups: int = 1,
    data_format: str = "NDHWC",
    enable_hf32: bool = False
) -> torch.Tensor:
    """
    计算3D卷积的反向传播中对滤波器的梯度。

    参数:
        x: 输入特征图，形状为 (N, D, H, W, C) 或 (N, C, D, H, W)
        filter_size: 滤波器尺寸，形状为 (5,)，包含 [Dk, Hk, Wk, Cin, Cout]
        out_backprop: 输出的梯度，形状为 (N, D', H', W', Cout) 或 (N, Cout, D', H', W')
        strides: 卷积步长，长度为3的列表 [stride_d, stride_h, stride_w]
        pads: 填充大小，长度为3的列表 [pad_d, pad_h, pad_w]
        dilations: 膨胀率，长度为3的列表 [dilation_d, dilation_h, dilation_w]，默认为 [1, 1, 1]
        groups: 分组数，默认为1
        data_format: 数据格式，支持 "NDHWC" 或 "NCDHW"，默认为 "NDHWC"
        enable_hf32: 是否启用HF32精度模式，默认为False

    返回:
        滤波器的梯度，形状为 (Cout, Cin/groups, Dk, Hk, Wk)
    """
```

## 使用案例
```python
import torch
import torch_npu
import kernel_gen_ops

# 设置NPU设备
torch.npu.set_device(0)

# 创建输入张量
batch_size = 2
in_channels = 16
out_channels = 32
depth = 8
height = 16
width = 16
kernel_d, kernel_h, kernel_w = 3, 3, 3

# 输入特征图 (NDHWC格式)
x = torch.randn(batch_size, depth, height, width, in_channels, dtype=torch.float16).npu()

# 滤波器尺寸 [Dk, Hk, Wk, Cin, Cout]
filter_size = torch.tensor([kernel_d, kernel_h, kernel_w, in_channels, out_channels], dtype=torch.int32).npu()

# 输出梯度
out_depth = (depth - kernel_d + 2 * 1) // 1 + 1
out_height = (height - kernel_h + 2 * 1) // 1 + 1
out_width = (width - kernel_w + 2 * 1) // 1 + 1
out_backprop = torch.randn(batch_size, out_depth, out_height, out_width, out_channels, dtype=torch.float16).npu()

# 计算滤波器梯度
grad_filter = kernel_gen_ops.conv3d_backprop_filter_v2(
    x=x,
    filter_size=filter_size,
    out_backprop=out_backprop,
    strides=[1, 1, 1],
    pads=[1, 1, 1],
    dilations=[1, 1, 1],
    groups=1,
    data_format="NDHWC",
    enable_hf32=False
)

print(f"滤波器梯度形状: {grad_filter.shape}")  # (32, 16, 3, 3, 3)
```

## 约束与限制

### 支持的数据类型
- 输入 x: FLOAT16, FLOAT, BF16
- 输入 filter_size: INT32
- 输入 out_backprop: FLOAT16, FLOAT, BF16
- 输出 y: FLOAT

### 支持的数据格式
- NDHWC: (N, D, H, W, C)
- NCDHW: (N, C, D, H, W)

### 其他约束
1. 输入 x 和 out_backprop 的数据类型必须相同
2. filter_size 必须是长度为5的一维张量，包含 [Dk, Hk, Wk, Cin, Cout]
3. strides、pads、dilations 必须是长度为3的列表
4. groups 必须能整除 Cin 和 Cout
5. 支持动态形状和动态rank
6. 仅支持 ascend910b 和 ascend910_93 平台
