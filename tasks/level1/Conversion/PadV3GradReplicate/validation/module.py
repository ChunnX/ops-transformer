import torch
import torch.nn as nn
import torch.nn.functional as F


def _pytorch_pad_tuple_from_ge_eight(ge_pad):
    """ge_pad: [0,0,0,0, top, bottom, left, right] -> F.pad 用 (left, right, top, bottom)。"""
    pt, pb, pl, pr = int(ge_pad[4]), int(ge_pad[5]), int(ge_pad[6]), int(ge_pad[7])
    return (pl, pr, pt, pb)


class Model(nn.Module):
    """CPU 参考：replication pad 反向；半精度在 CPU 上无 replicate 实现，先 float32 反传再 cast 回原 dtype。"""

    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, ge_padding):
        pad = _pytorch_pad_tuple_from_ge_eight(ge_padding)
        nd = grad_output.dim()
        if nd != 4:
            raise ValueError("PadV3GradReplicate validation 仅支持 4D NCHW")
        h = grad_output.shape[2] - pad[2] - pad[3]
        w = grad_output.shape[3] - pad[0] - pad[1]
        self_shape = (grad_output.shape[0], grad_output.shape[1], h, w)
        orig_dtype = grad_output.dtype
        g = grad_output.to(dtype=torch.float32)
        grad_input = torch.zeros(self_shape, dtype=torch.float32, device=g.device, requires_grad=True)
        out = F.pad(grad_input, pad, mode="replicate")
        out.backward(g)
        return grad_input.grad.to(dtype=orig_dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, ge_padding):
        import kernel_gen_ops

        if not isinstance(ge_padding, torch.Tensor):
            padding = torch.tensor(ge_padding, dtype=torch.int64, device="cpu")
        else:
            padding = ge_padding.reshape(-1).to(dtype=torch.int64, device="cpu")
        return kernel_gen_ops.pad_v3_grad_replicate(grad_output, padding)
