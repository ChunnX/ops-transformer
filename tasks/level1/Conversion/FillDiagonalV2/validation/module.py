import torch
import torch.nn as nn


class Model(nn.Module):
    """CPU 参考：与 torch.fill_diagonal_ 语义一致（就地填主对角）。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, fill_tensor: torch.Tensor, wrap: bool) -> torch.Tensor:
        out = x.clone()
        f = fill_tensor.to(device=out.device, dtype=out.dtype).contiguous().reshape(())
        out.fill_diagonal_(f, wrap=wrap)
        return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, fill_tensor: torch.Tensor, wrap: bool) -> torch.Tensor:
        import kernel_gen_ops

        y = x.clone()
        fill = fill_tensor.to(device=y.device, dtype=y.dtype).contiguous().reshape(1)
        wrap_b = bool(wrap)
        return kernel_gen_ops.fill_diagonal_v2(y, fill, wrap_b)
