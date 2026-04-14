# aclnnScaledMaskedSoftmaxV2

## 功能描述

### 算子功能
对输入张量 x 在最后一个维度上执行带缩放和掩码的 Softmax 操作。先将 x 乘以缩放因子 scale，然后在掩码为 True 的位置填充 -10000.0，最后对最后一维执行 Softmax。

### 计算公式
$$y = \text{Softmax}(scale \cdot x + mask \cdot (-10000.0))$$

其中：
- $x$ 为输入张量，shape 为 (B, N, S, D)
- $mask$ 为布尔掩码张量，True 表示被掩码的位置
- $scale$ 为缩放因子
- $\text{Softmax}$ 沿最后一维（D 维）执行

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.scaled_masked_softmax_v2()` 函数形式提供：

```python
def scaled_masked_softmax_v2(
    x: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
    fixed_triu_mask: bool = False
) -> torch.Tensor:
    """带缩放和掩码的 Softmax 前向计算。"""
```

**参数说明**：
| 参数 | 类型 | 说明 |
|:---|:---|:---|
| x | Tensor | 输入张量，shape 为 (B, N, S, D) |
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
x = torch.randn(B, N, S, D, dtype=torch.float16).npu()
mask = torch.zeros(B, N, S, D, dtype=torch.bool).npu()
scale = 1.0

y = kernel_gen_ops.scaled_masked_softmax_v2(x, mask, scale, False)
```

## 约束与限制
- 支持的数据类型：float16、float32、bfloat16
- 支持的数据格式：ND（4 维）
- 输入 x 和输出 y 的 shape 必须相同：[B, N, S, D]
- mask 的最后两维必须与输入的最后两维相同
- mask 的前两维支持广播：可以与输入相同或为 1
- width（最后一维 D）必须 <= 4096
