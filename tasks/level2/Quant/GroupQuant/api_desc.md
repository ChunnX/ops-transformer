# aclnnGroupQuant

## 功能描述

### 算子功能

按 `group_index` 将输入 `x` 的 S 维（第 0 维）划分为 E 段，每段对应 `scale` 的一行；对段内元素做逐列缩放，可选加标量 `offset`，再量化到 int8 或 int4。

### 计算公式

对第 `e` 个 expert，其负责行区间为 \([g_{e-1}, g_e)\)（约定 \(g_{-1}=0\)），\(g\) 为单调非减的 `group_index`，且 \(g_{E-1}=S\)：

\[
y_{s,h} = \mathrm{Quant}\bigl(x_{s,h} \cdot \mathrm{scale}_{e,h} + \mathrm{offset}\bigr)
\]

其中 `Quant` 与 AscendC 实现一致：fp32 乘加后经向最近整数取整、再经 fp16 与再次取整得到 int8 或 int4 表示。

## 接口定义

### Python 接口

通过 PyBind 封装 C++，在 Python 中以 `kernel_gen_ops.group_quant()` 提供：

```python
def group_quant(
    x: Tensor,
    scale: Tensor,
    group_index: Tensor,
    offset_optional: Optional[Tensor],
    dst_type: int,
) -> List[Tensor]:
    """
    GroupQuant：按组缩放并量化。

    参数:
        x: [S, H]，float16 / float32 / bfloat16，ND。
        scale: [E, H]，类型须与 x 组合在算子支持表内；第 1 维与 x 的 H 一致。
        group_index: [E]，int32 或 int64；单调非减，最后一元等于 S。
        offset_optional: 标量或 shape [1]，类型与 scale 相同；可为 None。
        dst_type: 输出数据类型枚举，int8 为 2，int4 为 29（与 ge 枚举一致）。

    返回:
        单元素列表 [y]，y 与 x 同 shape，dtype 为 int8（int4 仍以 int8 存储布局）。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

x = torch.randn(16, 32, dtype=torch.float16, device="npu")
scale = torch.rand(2, 32, dtype=torch.float16, device="npu") * 0.5 + 0.1
group_index = torch.tensor([8, 16], dtype=torch.int32, device="npu")
dst_type = 2
(y,) = kernel_gen_ops.group_quant(x, scale, group_index, None, dst_type)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含运行环境、shell 或排障说明。
- `x` 为 2 维，`scale` 为 2 维且 `scale.shape[1] == x.shape[1]`，`group_index` 为 1 维且长度等于 `scale.shape[0]`。
- `group_index` 每个元素在 \([0,S]\) 内，非递减，且最后一个等于 S。
- 若 `dst_type` 为 int4，则 H 须为偶数。
- 若提供 `offset`，其 dtype 必须与 `scale` 一致。
