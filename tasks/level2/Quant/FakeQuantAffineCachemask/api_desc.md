# FakeQuantAffineCachemask

## 功能

对输入 `x` 在通道维（与 tiling 一致：`headNum = x.shape[0]`）上做伪量化，输出 `y` 与同形 `mask`（bool）。

## 输入输出

| 名称 | 说明 |
|------|------|
| `x` | FLOAT16 / FLOAT32 |
| `scale` | 与 `x` 同 dtype，1D，长度等于 `x.shape[0]` |
| `zero_point` | INT32，1D，长度等于 `x.shape[0]` |
| `y` | 与 `x` 同形、同 dtype |
| `mask` | BOOL，与 `x` 同形 |

## 属性

- `axis`：整型，本 validation 固定为 `0`（与 kernel tiling 对 `dim0` 分通道一致）。
- `quant_min` / `quant_max`：伪量化整型网格上下界。

## Python 示例（编译并安装自定义 OPP 后）

```python
import torch
import kernel_gen_ops

y, mask = kernel_gen_ops.fake_quant_affine_cachemask(
    x, scale, zero_point, axis=0, quant_min=0, quant_max=255
)
```
