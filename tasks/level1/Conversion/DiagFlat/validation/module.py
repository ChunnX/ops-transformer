import torch
import torch.nn as nn


class Model(nn.Module):
    """CPU 参考：与 aclnnDiagFlat 语义一致（先展平再填对角）。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
        return torch.diagflat(x, offset=int(diagonal))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.diag_flat(x, int(diagonal))
