# aclnnPadV3GradReplicate

## 功能描述

### 算子功能

**Replicate / edge** 类 Pad v3 的反向：输入为已 pad 张量的梯度 `x`，按 8 元 `paddings` 与 `mode`（pybind 中固定为 **`edge`**，与 CANN replicate 路径一致）得到原始形状的梯度 `y`。

### 计算公式

与 `torch.nn.functional.pad(..., mode="replicate")` 对 4D NCHW 的反传一致。

## 接口定义

### Python 接口

编译替换后：`kernel_gen_ops.pad_v3_grad_replicate`（C++ 内 `mode="edge"`、`paddings_contiguous=True`）。

```python
def pad_v3_grad_replicate(grad_output: Tensor, padding: Tensor) -> Tensor:
    """
    参数:
        grad_output: 4D NCHW，已 pad 尺寸梯度，NPU；dtype fp32/fp16/bf16。
        padding: **8** 元 int64，GE 约定 [N前,N后,C前,C后, top, bottom, left, right]；
                 建议 CPU contiguous。

    返回:
        与原始输入同形状的梯度，4D NCHW。
    """
```

## 使用案例

```
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
grad_out = torch.randn(2, 3, 66, 66, device=device, dtype=torch.float16)
padding = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2], dtype=torch.int64, device="cpu")

grad_in = kernel_gen_ops.pad_v3_grad_replicate(grad_out, padding)
```

## 约束与限制

- 当前 validation 为 **4D NCHW**；`padding` 须 **8** 个元素。
- `paddings` 在 pybind 中重建为 CPU `int64` 向量再入 aclnn，与 tiling Host 读参习惯一致。
- 半精度 CPU 上 `F.pad(..., replicate)` 可能不可用，金标准可在 float32 上反传再 cast。
