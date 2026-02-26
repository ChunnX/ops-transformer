# aclnnAddRmsNormDynamicQuant

## 功能描述

### 算子功能
计算 (x1 + x2) 后做 RMSNorm，再按行做动态量化；可选 smooth_scale1、smooth_scale2、beta。输出 y1、y2（量化结果）、x（add 结果）、scale1、scale2。与 V2 相比无 y3/y4 等中间 float 输出，且 y1/y2 可为 int8 或 int4 等（由 dst_type 控制）。

### 计算公式
- x = x1 + x2
- rms = sqrt(mean(x^2, dim=-1) + epsilon)
- x_norm = x / rms
- y = x_norm * gamma + beta（beta 可选）
- 每行动态量化：smooth_x = y 或 y * smooth；scale = max(|smooth_x|)/127，q = round(smooth_x/scale)，y1/y2 为 clamp(q, -128, 127) 等，scale1/scale2 为对应 scale。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.add_rms_norm_dynamic_quant()` 函数形式提供：

```python
def add_rms_norm_dynamic_quant(
    x1, x2, gamma,
    smooth_scale1=None, smooth_scale2=None,
    epsilon=1e-6
):
    """
    实现 Add + RMSNorm + 动态量化。

    参数:
        x1 (Tensor): Device 侧输入，数据类型支持 FLOAT16、BF16，数据格式支持 ND。
        x2 (Tensor): Device 侧输入，数据类型支持 FLOAT16、BF16，数据格式支持 ND。
        gamma (Tensor): Device 侧 scale，数据类型支持 FLOAT16、BF16，数据格式支持 ND。
        smooth_scale1 (Tensor, 可选): 第一路量化的 smooth 系数，与 gamma 同 shape。
        smooth_scale2 (Tensor, 可选): 第二路量化的 smooth 系数，与 gamma 同 shape。
        epsilon (float, 可选): RMSNorm 的 epsilon，默认 1e-6。

    返回:
        Tuple[Tensor, ...]: (y1, y2, x, scale1, scale2)
        - y1, y2: 量化结果（如 int8），y1 shape 与 x1 一致；无 smooth2 时 y2 为 shape [1]。
        - x: add 结果，与 x1 同 dtype、同 shape。
        - scale1, scale2: float32 每行 scale，无 smooth2 时 scale2 为 shape [1]。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

x1 = torch.randn(2, 1024, dtype=torch.float16)
x2 = torch.randn(2, 1024, dtype=torch.float16)
gamma = torch.ones(1024, dtype=torch.float16)

y1, y2, x_out, scale1, scale2 = kernel_gen_ops.add_rms_norm_dynamic_quant(
    x1, x2, gamma, None, None, epsilon=1e-6
)
```

## 约束与限制

- 张量数据格式支持 ND。
- x1、x2、gamma 支持 FLOAT16、BF16。
- 不支持仅提供 smooth_scale2 而不提供 smooth_scale1。
- 输出 y1/y2 类型可由算子属性 dst_type 指定（如 int8、int4 等，依平台支持）。
