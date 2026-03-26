import torch
import torch.nn as nn
import torch.nn.functional as F


def _pytorch_pad_from_ge10(ge_pad):
    """ge 10 元组 -> torch.nn.functional.pad 的 pad（自最后一维向前：W,H,D,C,N）。"""
    if len(ge_pad) != 10:
        raise ValueError("ge_pad 长度须为 10")
    return (
        int(ge_pad[8]),
        int(ge_pad[9]),
        int(ge_pad[6]),
        int(ge_pad[7]),
        int(ge_pad[4]),
        int(ge_pad[5]),
        int(ge_pad[2]),
        int(ge_pad[3]),
        int(ge_pad[0]),
        int(ge_pad[1]),
    )


class Model(nn.Module):
    """CPU 参考：replicate 反传；4D 用 F.pad；5D 用 ReplicationPad3d（F.pad replicate 在部分 CPU 上未实现）。"""

    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, ge_padding):
        ge_list = list(ge_padding)
        nd = grad_output.dim()
        if nd not in (4, 5):
            raise ValueError("PadV3GradReplication validation 仅支持 4D / 5D")

        orig_dtype = grad_output.dtype
        g = grad_output.to(dtype=torch.float32)

        if nd == 4:
            pad = _pytorch_pad_from_ge10(ge_list)
            eff = pad[:4]
            out_sizes = list(g.shape)
            out_sizes[-2] -= eff[2] + eff[3]
            out_sizes[-1] -= eff[0] + eff[1]
            grad_input = torch.zeros(out_sizes, dtype=torch.float32, device=g.device, requires_grad=True)
            out = F.pad(grad_input, eff, mode="replicate")
            out.backward(g)
            return grad_input.grad.to(dtype=orig_dtype)

        # 5D N,C,D,H,W：与 prepare 一致 ge[4..9] = (dl,dr,hl,hr,wl,wr)
        dl, dr, hl, hr, wl, wr = (int(ge_list[i]) for i in range(4, 10))
        out_sizes = [
            g.shape[0],
            g.shape[1],
            g.shape[2] - dl - dr,
            g.shape[3] - hl - hr,
            g.shape[4] - wl - wr,
        ]
        # ReplicationPad3d: (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_front, pad_d_back)
        pad3d = (wl, wr, hl, hr, dl, dr)
        m = nn.ReplicationPad3d(pad3d)
        grad_input = torch.zeros(out_sizes, dtype=torch.float32, device=g.device, requires_grad=True)
        out = m(grad_input)
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
        return kernel_gen_ops.pad_v3_grad_replication(grad_output, padding)
