import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """CPU 金标准：ReflectionPad3d 前向 + backward，得到对 self 的梯度。"""

    def __init__(self, padding):
        super().__init__()
        self.pad = nn.ReflectionPad3d(padding)

    def forward(self, grad_output: torch.Tensor, self_input: torch.Tensor) -> torch.Tensor:
        x = self_input.detach().clone().requires_grad_(True)
        y = self.pad(x)
        y.backward(grad_output)
        return x.grad


class ModelNew(nn.Module):
    """NPU 自定义 ReflectionPad3dGrad（与 aclnn 入参顺序一致）。"""

    def __init__(self, padding):
        super().__init__()
        self.padding = list(padding) if not isinstance(padding, list) else padding

    def forward(self, grad_output: torch.Tensor, self_input: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.reflection_pad3d_grad(grad_output, self_input, self.padding)
