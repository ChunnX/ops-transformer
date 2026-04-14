"""
NsaCompressGrad：CPU 金标准通过对可微的 forward（与 NsaCompress 参考同构的树形规约）做 autograd 得到 inputGrad / weightGrad。
"""

import torch
import torch.nn as nn


def _compress_out_rows(act_cumsum, compress_block_size: int, compress_stride: int) -> int:
    pre = 0
    total = 0
    for end in act_cumsum:
        cur = end - pre
        if cur >= compress_block_size:
            total += (cur - compress_block_size + compress_stride) // compress_stride
        pre = end
    return total


def _ceil_power2_u32(n: int) -> int:
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def reduce_block_like_torch(mul_rows: torch.Tensor) -> torch.Tensor:
    """与 NsaCompress `reduce_block_like_kernel` 同构、可反传。"""
    x = mul_rows
    l_len, dim = x.shape
    if l_len == 1:
        return x[0]
    align = _ceil_power2_u32(l_len) // 2
    add_n = l_len - align
    new = x.clone()
    new[0:add_n] = x[0:add_n] + x[align : align + add_n]
    x = new
    while align > 1:
        align = align >> 1
        new = x.clone()
        new[0:align] = x[0:align] + x[align : 2 * align]
        x = new
    return x[0]


def nsa_compress_forward_torch(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    act_seq_len_cumsum: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
) -> torch.Tensor:
    """可微 forward（float32），与 `NsaCompress` 参考语义一致。"""
    t, n, d = input_tensor.shape
    l_blk, n_w = weight.shape
    assert l_blk == compress_block_size and n_w == n
    cum = act_seq_len_cumsum.detach().cpu().tolist()
    m = _compress_out_rows(cum, compress_block_size, compress_stride)
    out_rows = []
    inp_f = input_tensor
    w_f = weight
    pre = 0
    flat_dim = n * d
    for end in cum:
        seg = inp_f[pre:end]
        t_seg = seg.shape[0]
        k = 0
        while k * compress_stride + compress_block_size <= t_seg:
            s = k * compress_stride
            block = seg[s : s + compress_block_size]
            mul = block * w_f.unsqueeze(-1)
            mul_flat = mul.reshape(compress_block_size, flat_dim)
            red = reduce_block_like_torch(mul_flat)
            out_rows.append(red.view(n, d))
            k += 1
        pre = end
    assert len(out_rows) == m
    return torch.stack(out_rows, dim=0)


def nsa_compress_grad_reference(
    output_grad: torch.Tensor,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    act_seq_len_cumsum: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    act_seq_len_type: int,
):
    del act_seq_len_type
    inp = input_tensor.float().detach().clone().requires_grad_(True)
    w = weight.float().detach().clone().requires_grad_(True)
    out = nsa_compress_forward_torch(inp, w, act_seq_len_cumsum, compress_block_size, compress_stride)
    loss = (out * output_grad.float()).sum()
    gi, gw = torch.autograd.grad(loss, (inp, w), retain_graph=False, create_graph=False)
    return gi.to(input_tensor.dtype), gw.to(weight.dtype)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        output_grad: torch.Tensor,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        act_seq_len_cumsum: torch.Tensor,
        compress_block_size: int,
        compress_stride: int,
        act_seq_len_type: int,
    ):
        ig, wg = nsa_compress_grad_reference(
            output_grad,
            input_tensor,
            weight,
            act_seq_len_cumsum,
            compress_block_size,
            compress_stride,
            act_seq_len_type,
        )
        return [ig, wg]


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        output_grad: torch.Tensor,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        act_seq_len_cumsum: torch.Tensor,
        compress_block_size: int,
        compress_stride: int,
        act_seq_len_type: int,
    ):
        import kernel_gen_ops

        return kernel_gen_ops.nsa_compress_grad(
            output_grad,
            input_tensor,
            weight,
            act_seq_len_cumsum,
            compress_block_size,
            compress_stride,
            act_seq_len_type,
        )
