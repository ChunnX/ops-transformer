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
```python
def add_rms_norm_dynamic_quant_v2(
    x1: Tensor, x2: Tensor, gamma: Tensor,
    smooth_scale1: Optional[Tensor], smooth_scale2: Optional[Tensor],
    epsilon: float = 1e-6
) -> Tuple[Tensor, ...]:
    """
    返回: (y1, y2, y3, y4, x, scale1, scale2)
    - y1, y2: int8
    - y3: float32, y4: 与输入同 dtype
    - x: add 结果
    - scale1, scale2: float32 每行 scale
    """
```

## 约束与限制
- x1、x2、gamma 支持 float16 / bfloat16。
- normalized_shape 与 gamma 的 shape 一致，为最后一维。
- 未提供 smooth 时，y2、scale2 由实现定义（参考返回 0 与 1）。
