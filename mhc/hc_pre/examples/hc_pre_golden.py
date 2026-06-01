import numpy as np
import torch
import torch.nn.functional as F


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), np.abs(expect_data))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    return np.where(a < diff_thd, a, a / b)


def data_compare(npu_out, cpu_out, diff_thd=0.005, pct_thd=0.005):
    real_data = npu_out.flatten()
    data_compe = cpu_out.flatten()
    if real_data.size != data_compe.size:
        return "Failed", 0.0, 0
    split_count = real_data.size
    diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(
        real_data[diff_index].astype(np.float32), data_compe[diff_index].astype(np.float32), diff_thd)
    error_cnt = rdiff[rdiff > diff_thd].size
    fulfill_percent = float(split_count - error_cnt) / float(split_count) * 100.0
    return ("Pass" if fulfill_percent >= (1 - pct_thd) * 100.0 else "Failed"), fulfill_percent, 0


def hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))
    pre = F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
    post = 2 * F.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre_golden(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
    shape, dtype = x.size(), x.dtype
    flat_x = x.flatten(2).float() if x.dim() == 4 else x.flatten(1).float()
    rsqrt = torch.rsqrt(flat_x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat_x, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * flat_x.view(shape), dim=-2)
    return y.to(dtype), post, comb


def create_hc_pre_inputs(shape, hc_mix, hc_mult, seed=42):
    np.random.seed(seed)
    hc_scale = torch.tensor(np.random.uniform(0, 2, (3))).to(torch.float32)
    hc_base = torch.tensor(np.random.uniform(0, 2, (hc_mix))).to(torch.float32)
    hc_fn = torch.tensor(np.random.uniform(0, 2, (hc_mix, hc_mult * shape[-1]))).to(torch.float32)
    x = torch.tensor(np.random.uniform(0, 2, shape)).to(torch.bfloat16)
    return x, hc_fn, hc_scale, hc_base
