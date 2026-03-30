# aclnnAddLora

## 功能描述

### 算子功能

按 `indices` 为每条样本选择 LoRA 权重组，对 `x` 依次与 `weightA`、`weightB` 做矩阵乘并乘以 `scale`，将结果累加到 `y` 的 `[y_offset : y_offset + y_slice_size)` 列上。

### 计算公式

见官方文档：\(Z_1 = x \cdot W_A\)，\(Z_2 = Z_1 \cdot W_B \times scale\)，`out = y` 对应切片加上 `Z2`。

## 接口定义

### Python 接口

该操作通过 PyBind11 封装，编译替换后为 `kernel_gen_ops.add_lora()`。

```python
def add_lora(
    y: Tensor,
    x: Tensor,
    weight_b: Tensor,
    indices: Tensor,
    weight_a: Tensor,
    layer_idx: int,
    scale: float,
    y_offset: int,
    y_slice_size: int,
) -> Tensor:
    """
    参数均为 float16（indices 为 int32），shape 满足 aclnn 文档：y/x 为 [B,H3]/[B,H1]，
    weight_b [W,L,H2,R]，weight_a [W,L,R,H1]，indices [B]。
    """
```

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

B, H1, H3, W, L, H2, R = 8, 32, 64, 4, 2, 32, 16
device = "npu:0"
y = torch.randn(B, H3, device=device, dtype=torch.float16)
x = torch.randn(B, H1, device=device, dtype=torch.float16)
weight_b = torch.randn(W, L, H2, R, device=device, dtype=torch.float16)
weight_a = torch.randn(W, L, R, H1, device=device, dtype=torch.float16)
indices = torch.randint(0, W, (B,), device=device, dtype=torch.int32)
out = kernel_gen_ops.add_lora(
    y, x, weight_b, indices, weight_a, 0, 1e-3, 0, H2
)
```

## 约束与限制

- PyBind 侧实现中 **`out` 须先为 `y` 的拷贝**（`y.clone()`），再在切片上累加 LoRA 项；勿用未初始化的 `empty_like(y)`，否则非更新列可能为随机值。
- 仅支持 **float16**（与算子定义一致）；`H1`、`H2`、`H3`、`R` 为 **16 的倍数**，`R≤128`，`W,L≤32` 等见 aclnn 文档。
- `layer_idx < L`；`y_offset + y_slice_size` 不超过 `H3`。
- 在 **Atlas A2 / 910B** 等路径下，tiling 在 `batch` 较大与较小时会走不同 kernel 分支（如 `100000` / `100001`），validation 用例已分别覆盖。
