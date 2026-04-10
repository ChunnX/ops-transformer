# aclnnInterleaveRope

## 功能描述

### 算子功能

对四维输入 `x`（形状 `[B,N,S,D]`）先做最后一维上的 **interleave**（将 `x` 视为 `[B,N,S,D/2,2]`，交换最后两维后再 reshape 回 `[B,N,S,D]`），再按 RoPE 形式与 `cos`、`sin` 组合得到输出 `y`。

### 计算公式

记 interleave 后的张量为 `q`，则：

\[
y = q \cdot \cos + \mathrm{RotateHalf}(q) \cdot \sin
\]

其中 \(\mathrm{RotateHalf}(q)\) 将 `q` 在最后一维上后半段移至前半并取负、前半段移至后半（与产品文档一致）。

## 接口定义

### Python 接口

该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.interleave_rope()` 形式提供（由编译流程将 `custom_pybind_api` 绑定为 `interleave_rope`）：

```python
def interleave_rope(x: Tensor, cos: Tensor, sin: Tensor, out: Tensor) -> Tensor:
    """
    参数:
        x (Tensor): `[B,N,S,D]`，D 须为 64，dtype 为 float16 或 bfloat16。
        cos (Tensor): `[B,1,S',D]`，S' 为 1 或与 S 相同；dtype 与 x 一致。
        sin (Tensor): 与 cos 同 shape、同 dtype。
        out (Tensor): 与 x 同 shape、同 dtype，预分配的输出缓冲区。

    返回:
        Tensor: 与 out 为同一 Tensor（就地写入）。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

B, N, S, D = 2, 4, 8, 64
x = torch.randn(B, N, S, D, dtype=torch.float16, device="npu")
cos = torch.randn(B, 1, S, D, dtype=torch.float16, device="npu")
sin = torch.randn(B, 1, S, D, dtype=torch.float16, device="npu")
out = torch.empty_like(x)
kernel_gen_ops.interleave_rope(x, cos, sin, out)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含 conda、CANN、shell、`run_multi_test`、设备错误或任何运行环境 / 排障说明。
- `x`、`y` 为四维 ND，`shape` 为 `(B,N,S,D)`，且 **D 必须为 64**。
- `cos`、`sin` 为四维，`shape` 为 `(B,1,S',D)`，其中 **N 维为 1**；**S' 为 1 或与 S 相同**；**B 与 x 的 B 一致**；**D 与 x 的 D 一致**；dtype 与 `x` 一致。
- 支持 **FLOAT16、BFLOAT16**。
