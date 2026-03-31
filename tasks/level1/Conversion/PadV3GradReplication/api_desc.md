# aclnnPadV3GradReplication

## 功能描述

### 算子功能

**Replication（复制边界）** pad 的反向：输入为 pad 后张量的梯度 `x`，按 GE 风格 **10 元** `paddings` 将梯度累加回原始形状的 `z`（与 `replicate` 正向对应）。

### 计算公式

与 `torch.nn.functional.pad(..., mode="replicate")` 或 `ReplicationPad3d` 的反传一致：边界复制带来的梯度沿 pad 方向回灌到有效区域。

## 接口定义

### Python 接口

编译替换后：`kernel_gen_ops.pad_v3_grad_replication`。

```python
def pad_v3_grad_replication(grad_output: Tensor, padding: Tensor) -> Tensor:
    """
    参数:
        grad_output: 4D 或 5D（ND），已 pad 尺寸梯度，NPU；dtype fp16/fp32/bf16。
        padding: 展平后 **10** 个 int64，GE 顺序自最后一维向前（与 validation 中 prepare 一致）；
                 建议在 CPU contiguous。

    返回:
        z: 与正向 self 同形状的梯度，维数与 grad_output 一致。
    """
```

## 使用案例

```
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
# 4D 示例：10 元 padding 中与 H,W 相关部分在后 4 位中体现（与任务 prepare_inputs 约定一致）
grad_out = torch.randn(1, 3, 34, 34, device=device, dtype=torch.float32)
padding = torch.tensor(
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.int64, device="cpu"
)

grad_in = kernel_gen_ops.pad_v3_grad_replication(grad_out, padding)
```

## 约束与限制

- validation 当前支持 **4D / 5D**；`padding` 总长度须与算子 **10 元** 约定一致。
- `padding` 与 autogen 中类型一致；Host 读参时使用 **CPU int64 contiguous**。
- 5D 场景与 GE 维序对应关系以 `prepare_inputs` / tiling 为准。
