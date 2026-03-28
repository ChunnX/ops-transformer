import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, perm_cpu: torch.Tensor):
        super().__init__()
        self._perm = tuple(int(x) for x in perm_cpu.detach().cpu().tolist())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self._perm).contiguous()


class ModelNew(nn.Module):
    """perm 保持 CPU 张量，避免 ModelNew.to(npu) 把整型 perm 搬到设备侧。"""

    def __init__(self, perm_cpu: torch.Tensor):
        super().__init__()
        self._perm = perm_cpu.detach().cpu().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.transpose_v2(x, self._perm)
