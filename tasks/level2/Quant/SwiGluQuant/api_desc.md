# aclnnSwiGluQuantV2

## 功能描述

### 算子功能

对输入 `x` 在最后一维二等分后做 SwiGLU（SiLU 一侧 × 另一侧），再按 `quant_mode` 做动态或静态量化，输出整型 `y` 与浮点 `scale`（动态场景 `scale` 为每「行」量化因子；静态场景实现中 `scale` 可全零）。

### 计算公式

记最后一维前半为 A、后半为 B，`activate_left=true` 时对 A 做 SiLU，否则对 B 做 SiLU，再与另一半逐元素相乘得到 `Act`。动态量化下先按 `smooth_scales`（及可选 MoE `group_index`）缩放，再按行最大绝对值与 `dst_type`（int8→127、int4→7）求 scale 并映射到整型；静态量化下为 `Act * scale + offset` 后直接整型化（以实际 kernel 为准）。

## 接口定义

### Python 接口

PyBind 封装后通过 `kernel_gen_ops.swi_glu_quant()` 调用（与 msopgen 替换后的符号名一致），语义对齐 **aclnnSwiGluQuantV2**：

```python
def swi_glu_quant(
    x: Tensor,
    smooth_scales: Tensor,
    offsets_optional: Optional[Tensor],
    group_index_optional: Optional[Tensor],
    activate_left: bool,
    quant_mode: str,
    group_list_type: int,
    dst_type: int,
) -> List[Tensor]:
    """
    返回 [y, scale]；y 最后一维为 x 最后一维的一半；scale 的 shape 与 x 去掉最后一维一致。
    smooth_scales / offsets 在图编译侧为 float32；dst_type 取 ge 枚举值 2（int8）或 29（int4）。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

x = torch.randn(4, 32, dtype=torch.float16, device="npu")
smooth = torch.rand(1, 16, dtype=torch.float32, device="npu")
y, scale = kernel_gen_ops.swi_glu_quant(
    x, smooth, None, None, False, "dynamic", 0, 2
)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含运行环境或排障说明。
- `x` 至少 2 维，最后一维为偶数；int4 时最后一维须为 4 的倍数。
- `quant_mode` 为 `"dynamic"` 或 `"static"`；静态场景需提供与规范一致的 `offsets`。
- MoE 场景下 `group_index` 与 `smooth_scales` 第一维需满足 tiling 校验；`group_list_type` 为 0（cumsum）或 1（count）。
