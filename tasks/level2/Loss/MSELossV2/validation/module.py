import torch
import torch.nn as nn


class Model(nn.Module):
    """CPU 参考：MSE loss，与 PyTorch functional 语义一致。"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        y = (predict - label) * (predict - label)
        if self.reduction == "sum":
            return y.sum()
        if self.reduction == "mean":
            return y.mean()
        return y


class ModelNew(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.mse_loss_v2(predict, label, self.reduction)
