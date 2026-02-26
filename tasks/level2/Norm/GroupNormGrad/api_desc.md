# aclnnGroupNormGrad

## 功能描述

### 算子功能
对 GroupNorm 的反向：根据输出的梯度 dy、前向的 mean 与 rstd、输入 x、参数 gamma，计算对输入的梯度 dx、对 gamma 的梯度 dgamma、对 beta 的梯度 dbeta。

### 计算公式
在每组内（通道分组），设 x_norm = (x - mean) * rstd，则：
- 对 gamma、beta 的梯度为组内对 x_norm 与 1 的加权和。
- 对 x 的梯度 dx 由 dy、gamma、x_norm、mean、rstd 及组内统计得到，与标准 GroupNorm 反向一致。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.group_norm_grad()` 函数形式提供：

```python
def group_norm_grad(
    dy, mean, rstd, x, gamma,
    num_groups, data_format="NCHW",
    dx_is_require=True, dgamma_is_require=True, dbeta_is_require=True
):
    """
    实现 GroupNorm 反向，计算 dx、dgamma、dbeta。

    参数:
        dy (Tensor): 输出梯度，数据类型支持 FLOAT32、FLOAT16、BF16，数据格式支持 ND。
        mean (Tensor): 前向 GroupNorm 的 mean，shape [N, num_groups]，数据类型同 dy。
        rstd (Tensor): 前向 GroupNorm 的 rstd，shape [N, num_groups]，数据类型同 dy。
        x (Tensor): 前向输入，与 dy 同 shape，数据类型同 dy。
        gamma (Tensor): 前向 scale，与通道维度一致，数据类型同 dy。
        num_groups (int): 分组数量。
        data_format (str, 可选): 数据格式，默认 "NCHW"。
        dx_is_require (bool, 可选): 是否计算 dx，默认 True。
        dgamma_is_require (bool, 可选): 是否计算 dgamma，默认 True。
        dbeta_is_require (bool, 可选): 是否计算 dbeta，默认 True。

    返回:
        List[Tensor]: 按需返回的列表，顺序依次为 dx、dgamma、dbeta（仅包含对应 dx_is_require、dgamma_is_require、dbeta_is_require 为 True 的项，见 validation/pybind.cpp）。
        - dx: 与 x 同 shape、同 dtype。
        - dgamma, dbeta: 与 gamma 同 shape、同 dtype。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

dy = torch.randn(4, 32, 14, 14, dtype=torch.float16)
x = torch.randn(4, 32, 14, 14, dtype=torch.float16)
num_groups = 8
mean = torch.randn(4, num_groups, dtype=torch.float16)
rstd = torch.randn(4, num_groups, dtype=torch.float16)
gamma = torch.ones(32, dtype=torch.float16)

dx, dgamma, dbeta = kernel_gen_ops.group_norm_grad(
    dy, mean, rstd, x, gamma,
    num_groups, "NCHW",
    True, True, True
)
```

## 约束与限制

- 张量数据格式支持 ND。
- dy、mean、rstd、x、gamma 支持 FLOAT32、FLOAT16、BF16。
- num_groups 必须能整除通道数 C。
- mean、rstd 的 shape 为 [N, num_groups]。
