# aclnnUnfoldGrad

## 功能描述

### 算子功能

实现 `torch.Tensor.unfold` 的反向：根据 Unfold 正向输出的梯度 `gradOut` 与原输入形状 `inputSizes`，计算对原输入的梯度 `gradIn`。

### 计算公式

与 PyTorch `unfold` 的反向一致；等价于对展开维做滑窗梯度的折叠累加。

## 接口定义

### Python 接口

该操作通过 PyBind11 封装 C++ 实现，编译替换后在 Python 中以 `kernel_gen_ops.unfold_grad()` 提供。

```python
def unfold_grad(
    grad_out: Tensor,
    input_sizes: list[int],
    dim: int,
    size: int,
    step: int,
) -> Tensor:
    """
    参数:
        grad_out: Unfold 正向输出形状的梯度，NPU，dtype 为 float16 / bfloat16 / float32。
        input_sizes: 原输入 self 的各维长度（Python list[int]），PyBind 侧转为 aclIntArray。
        dim: 展开所在维，仅支持 len(input_sizes)-1 或 len(input_sizes)-2。
        size: 窗口长度，需满足 0 < size <= input_sizes[dim]。
        step: 步长，需 > 0。

    返回:
        grad_in: 与 input_sizes 一致形状的梯度张量，dtype 与 grad_out 一致。
    """
```

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
input_sizes = [2, 3, 16]
dim, size, step = 2, 4, 2
grad_shape = list(input_sizes)
grad_shape[dim] = (input_sizes[dim] - size) // step + 1
grad_shape.append(size)
g = torch.randn(grad_shape, device=device, dtype=torch.float16)
out = kernel_gen_ops.unfold_grad(g, input_sizes, dim, size, step)
```

## 约束与限制

- `grad_out` 维数 = `len(input_sizes) + 1`；`grad_out.shape[dim] == (input_sizes[dim]-size)//step+1`；最后一维 = `size`。
- `dim` 仅能为最后一维或倒数第二维。
- 尾轴展开时 fp16 的 `size`、`step` 等上界见官方 aclnn 文档；倒数第二轴时 fp16 上界更严（如 size、step ≤ 72）。
- `input_sizes` 以 Host 可读的形状信息传入（validation 使用 `list[int]` → `aclIntArray`），Tiling 从常量 Tensor 读指针时须保证图为 Host 侧数据；勿将设备指针当 Host 解引用。
