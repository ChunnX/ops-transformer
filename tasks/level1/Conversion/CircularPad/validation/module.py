import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """CPU 参考：与 aclnn CircularPad2d / CircularPad3d 语义一致（circular 填充最后两维或三维）。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, padding):
        pad = tuple(int(p) for p in padding)
        return F.pad(x, pad, mode="circular")


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, padding):
        import kernel_gen_ops

        # padding 保持 CPU int64，与 pybind 中传入 aclnn 的 Host tensor 一致，避免 tiling 在 Host 读 GetData 崩溃
        if not isinstance(padding, torch.Tensor):
            padding = torch.tensor(padding, dtype=torch.int64, device="cpu")
        else:
            padding = padding.reshape(-1).to(dtype=torch.int64, device="cpu")
        return kernel_gen_ops.circular_pad(x, padding)
