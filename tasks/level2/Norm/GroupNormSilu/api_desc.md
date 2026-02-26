# aclnnGroupNormSilu

## 功能描述

### 算子功能
对输入在通道维度上做 GroupNorm（分组归一化），并可选地施加 SiLU 激活（x * sigmoid(x)）。输出归一化并激活后的张量 y，以及分组内的 mean 和 rstd（用于反向等）。

### 计算公式
- 将输入按通道分为 num_groups 组，每组内做 LayerNorm：mean、var 在 (C/G * H*W) 上计算。
- x_norm = (x - mean) * rstd，其中 rstd = 1/sqrt(var + eps)。
- y_gn = x_norm * gamma + beta。
- 若 activate_silu 为 True：y = y_gn * sigmoid(y_gn)；否则 y = y_gn。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.group_norm_silu()` 函数形式提供：

```python
def group_norm_silu(x, gamma, beta, num_groups, eps=1e-5, activate_silu=True):
    """
    实现 GroupNorm + 可选 SiLU 激活。

    参数:
        x (Tensor): Device 侧输入，数据类型支持 FLOAT16、FLOAT32、BF16，数据格式支持 ND。
        gamma (Tensor): Device 侧 scale，可选，与通道维度一致，数据类型同 x。
        beta (Tensor): Device 侧 bias，可选，与通道维度一致，数据类型同 x。
        num_groups (int): 分组数量，必须能整除通道数 C。
        eps (float, 可选): 归一化中的 epsilon，默认 1e-5。
        activate_silu (bool, 可选): 是否在归一化后施加 SiLU 激活，默认 True。

    返回:
        Tuple[Tensor, Tensor, Tensor]: (y, mean, rstd)
        - y: 与 x 同 shape、同 dtype。
        - mean: shape [N, num_groups]，与输入同 dtype。
        - rstd: shape [N, num_groups]，与输入同 dtype。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入：NCHW 或 ND
x = torch.randn(4, 32, 14, 14, dtype=torch.float16)
num_groups = 8
C = x.size(1)
gamma = torch.ones(C, dtype=torch.float16)
beta = torch.zeros(C, dtype=torch.float16)

y, mean, rstd = kernel_gen_ops.group_norm_silu(
    x, gamma, beta, num_groups, eps=1e-5, activate_silu=True
)
```

## 约束与限制

- 张量数据格式支持 ND。
- 输入 x 支持 FLOAT16、FLOAT32、BF16；gamma、beta 可选，与 x 同 dtype。
- num_groups 必须能整除通道数 C。
- 支持动态 shape、动态 rank（依平台配置）。
