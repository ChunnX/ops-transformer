# aclnnAddRmsNormDynamicQuantV2

## 功能描述

### 算子功能
计算 (x1 + x2) 后做 RMSNorm，再按行做动态量化得到 int8 输出；可选 smooth_scale1/smooth_scale2。输出 y1(int8)、y2(int8)、y3(float32)、y4(fp16)、x(add 结果)、scale1(每行动态量化 scale)、scale2。

### 计算公式
- x = x1 + x2
- rms = sqrt(mean(x^2, dim=-1) + epsilon)
- x_norm = x / rms
- y = x_norm * gamma
- 每行动态量化：scale = max(|y|)/127，q = round(y/scale)，y1 = clamp(q, -128, 127)，scale1 = scale

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.add_rms_norm_dynamic_quant_v2()` 函数形式提供：

```python
def add_rms_norm_dynamic_quant_v2(
    x1, x2, gamma,
    smooth_scale1=None, smooth_scale2=None,
    epsilon=1e-6
):
    """
    实现 Add + RMSNorm + 动态量化（V2）。

    参数:
        x1 (Tensor): Device 侧输入，公式中的 x1，数据类型支持 FLOAT16、BF16，数据格式支持 ND。
        x2 (Tensor): Device 侧输入，公式中的 x2，数据类型支持 FLOAT16、BF16，数据格式支持 ND。
        gamma (Tensor): Device 侧输入，归一化后的缩放，数据类型支持 FLOAT16、BF16，数据格式支持 ND。
        smooth_scale1 (Tensor, 可选): 第一路动态量化的 smooth 系数，与 gamma 同 shape。
        smooth_scale2 (Tensor, 可选): 第二路动态量化的 smooth 系数，与 gamma 同 shape。
        epsilon (float, 可选): RMSNorm 中的 epsilon，默认 1e-6。

    返回:
        Tuple[Tensor, ...]: (y1, y2, y3, y4, x, scale1, scale2)
        - y1, y2: int8 量化结果，shape 与 x1 一致。
        - y3: float32，y4: 与输入同 dtype，shape 与 x1 一致。
        - x: add 结果，与 x1 同 dtype、同 shape。
        - scale1, scale2: float32 每行 scale，shape 为 x1 除最后一维外与 x1 一致、最后一维为 1。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x1 = torch.randn(2, 1024, dtype=torch.float16)
x2 = torch.randn(2, 1024, dtype=torch.float16)
gamma = torch.ones(1024, dtype=torch.float16)

# 可选 smooth
smooth_scale1 = torch.ones(1024, dtype=torch.float16)
smooth_scale2 = torch.ones(1024, dtype=torch.float16)

# 调用算子
y1, y2, y3, y4, x_out, scale1, scale2 = kernel_gen_ops.add_rms_norm_dynamic_quant_v2(
    x1, x2, gamma, smooth_scale1, smooth_scale2, epsilon=1e-6
)
```

## 约束与限制

- 张量数据格式支持 ND。
- x1、x2、gamma 支持 float16 / bfloat16。
- normalized_shape 与 gamma 的 shape 一致，为最后一维。
- 未提供 smooth 时，y2、scale2 由实现定义（参考返回 0 与 1）。
- 不支持仅提供 smooth_scale2 而不提供 smooth_scale1。
