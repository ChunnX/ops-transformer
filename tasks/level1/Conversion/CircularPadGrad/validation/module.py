import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """CPU 参考：与 ops ST executor 一致，对 circular pad 做反向（grad 对 unpadded 输入）。"""

    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, padding):
        pad = tuple(int(p) for p in padding)
        nd = grad_output.dim()
        if len(pad) == 4:
            h = grad_output.shape[nd - 2] - pad[2] - pad[3]
            w = grad_output.shape[nd - 1] - pad[0] - pad[1]
            self_shape = tuple(grad_output.shape[:-2]) + (h, w)
        else:
            d = grad_output.shape[nd - 3] - pad[4] - pad[5]
            h = grad_output.shape[nd - 2] - pad[2] - pad[3]
            w = grad_output.shape[nd - 1] - pad[0] - pad[1]
            self_shape = tuple(grad_output.shape[:-3]) + (d, h, w)

        grad_input = torch.zeros(self_shape, dtype=grad_output.dtype, device=grad_output.device, requires_grad=True)
        out = F.pad(grad_input, pad, mode="circular")
        out.backward(grad_output)
        return grad_input.grad


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, padding):
        import kernel_gen_ops

        if not isinstance(padding, torch.Tensor):
            padding = torch.tensor(padding, dtype=torch.int64, device="cpu")
        else:
            padding = padding.reshape(-1).to(dtype=torch.int64, device="cpu")
        return kernel_gen_ops.circular_pad_grad(grad_output, padding)
