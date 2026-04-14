# aclnnScaledMaskedSoftmaxGradV2

## 功能描述

### 算子功能
计算带缩放和掩码的 Softmax 反向传播梯度。给定 Softmax 输出 y、其梯度 yGrad 以及掩码 mask，计算输入 x 的梯度 xGrad。

### 计算公式
$$xGrad = scale \cdot mask\_apply(y \cdot (yGrad - \sum(yGrad \cdot y)))$$

其中：
- $y$ 为 Softmax 前向输出
- $yGrad$ 为 Softmax 输出梯度
- $\sum$ 沿最后一维求和
- $scale$ 为缩放因子
- $mask\_apply$ 在 mask 为 True 的位置将输出置零

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.scaled_masked_softmax_grad_v2()` 函数形式提供：

```python
def scaled_masked_softmax_grad_v2(
    yGrad: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
    fixed_triu_mask: bool = False
) -> torch.Tensor:
    """计算带缩放和掩码的 Softmax 梯度。"""
```

**参数说明**：
| 参数 | 类型 | 说明 |
|:---|:---|:---|
| yGrad | Tensor | Softmax 输出梯度，shape 为 (B, N, S, D) |
| y | Tensor | Softmax 前向输出，shape 与 yGrad 相同 |
| mask | Tensor(bool) | 掩码张量，shape 为 (B, N, S, D) 或可广播形状 |
| scale | float | 缩放因子，默认 1.0 |
| fixed_triu_mask | bool | 是否使用固定上三角掩码，默认 False |

## 使用案例
```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops

B, N, S, D = 2, 4, 128, 64
yGrad = torch.randn(B, N, S, D, dtype=torch.float16).npu()
y = torch.softmax(torch.randn(B, N, S, D, dtype=torch.float16), dim=-1).npu()
mask = torch.zeros(B, N, S, D, dtype=torch.bool).npu()
scale = 1.0

xGrad = kernel_gen_ops.scaled_masked_softmax_grad_v2(yGrad, y, mask, scale, False)
```

## 约束与限制
- 支持的数据类型：float16、float32、bfloat16
- 支持的数据格式：ND（4 维）
- 输入 yGrad、y 和输出 xGrad 的 shape 必须相同：[B, N, S, D]
- mask 的最后两维必须与输入的最后两维相同
- mask 的前两维支持广播：可以与输入相同或为 1
- headDim（最后一维）必须 <= 8192
- 当 paddedHeadDim <= 1024 时使用 NormHeadDim kernel，否则使用 LargeHeadDim kernel
