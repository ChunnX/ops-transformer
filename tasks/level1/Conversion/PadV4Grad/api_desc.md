# aclnnPadV4Grad

## 功能描述

### 算子功能

Pad v4 的反向：输入为 **已 pad 张量** 上的梯度 `x`，按 GE 风格 8 元 `paddings` 与 `mode`（如 `reflect`）将梯度折叠为 **原始未 pad 形状** 的 `y`。

### 计算公式

与 `torch.nn.functional.pad` 对应 `mode` 的反传一致：对 pad 区域按规则把 `grad_output` 累加回 `grad_input` 的有效区域。

## 接口定义

### Python 接口

编译替换后：`kernel_gen_ops.pad_v4_grad`（C++ 内固定 `mode="reflect"`、`paddings_contiguous=True` 传入 aclnn）。

```python
def pad_v4_grad(grad_output: Tensor, padding: Tensor) -> Tensor:
    """
    参数:
        grad_output: 4D NCHW，已 pad 尺寸的梯度，device 为 NPU；dtype 支持 fp32/fp16/bf16。
        padding: 8 个元素的一维张量，GE 约定 [N前,N后,C前,C后, top, bottom, left, right]；
                 建议在 CPU 上 int64 且 contiguous（与 validation/pybind 一致）。

    返回:
        与原始输入同形状的梯度，4D NCHW，dtype 与 grad_output 一致。
    """
```

## 使用案例

```
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
# 例：N,C,H_pad,W_pad；未 pad 前 H=62,W=62，各向 reflect pad 1
grad_out = torch.randn(2, 3, 64, 64, device=device, dtype=torch.float32)
padding = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64, device="cpu")

grad_in = kernel_gen_ops.pad_v4_grad(grad_out, padding)
```

## 约束与限制

- 仅支持 **4D NCHW**；`padding` 元素个数须为 **8**。
- `paddings` 若需在 Host tiling 中读指针，须 **CPU、整型、contiguous**。
- 半精度在 CPU 参考中可能需先升 float32 再 cast，与 validation 中 `Model` 一致。
