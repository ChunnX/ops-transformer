#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Kernel-level golden for MhcPreSinkhornBackward.
# Math = /workspace/ops-transformer/mhc/mhc_pre_sinkhorn_backward/tests/assets/golden_src.py
# (SinkhornGrad adapted to the op's sum_out/norm_out layout (2*iters,B,S,n[,n]) via reshape.)
# Kernel goldens receive numpy arrays; we upcast to torch fp32, compute, return numpy.
__golden__ = {
    "kernel": {"mhc_pre_sinkhorn_backward": "mhc_pre_sinkhorn_backward_golden"}
}

import numpy as np
import torch


def SigmoidGrad(z, dy, is_pre=False, hc_eps=1e-6):
    y = torch.sigmoid(z)
    sigma = (y - hc_eps) if is_pre else y
    return dy * sigma * (1 - sigma)


def RMSNormGrad(x, inv_rms, grad_inv_rms):
    NC = x.size(-1)
    return -(inv_rms ** 3) * x * grad_inv_rms / float(NC)


def ExpGrad(x, y_grad):
    n = x.size(-1)
    x_max, idx = x.max(dim=-1, keepdim=True)
    arange_idx = torch.arange(n, device=x.device)[None, :]
    is_max = (idx == arange_idx)
    y = (x - x_max).exp()
    sum_all = (y * y_grad).sum(dim=-1, keepdim=True)
    return y * y_grad - is_max * sum_all


def SinkhornGrad(grad_h_res, sum_out, norm_out):
    bs, seq_len, n, _ = grad_h_res.shape
    x_grad = grad_h_res
    two_it = sum_out.shape[0]
    iters = two_it // 2
    sum_out = sum_out.reshape(iters, 2, bs, seq_len, n)
    norm_out = norm_out.reshape(iters, 2, bs, seq_len, n, n)
    for i in reversed(range(iters)):
        row_sum = sum_out[i][0].view(bs, seq_len, n, 1)
        col_sum = sum_out[i][1].view(bs, seq_len, 1, n)
        x_row_normed = norm_out[i][0].view(bs, seq_len, n, n)
        grad_x_row_normed = x_grad / col_sum - (x_grad * x_row_normed / (col_sum ** 2)).sum(dim=-2, keepdim=True)
        x_grad = grad_x_row_normed / row_sum - (grad_x_row_normed * x_row_normed / row_sum).sum(dim=-1, keepdim=True)
    return x_grad


def _core(grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
          h_pre, hc_before_norm, inv_rms, sum_out, norm_out, hc_eps):
    B, S, n, d = x.shape
    grad_hin_fp32 = grad_hin.float()
    x_fp32 = x.float()

    grad_x_from_hin = grad_hin_fp32[..., None, :] * h_pre[..., None]
    grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)

    norm_out_forward = hc_before_norm * inv_rms
    z_pre = norm_out_forward[..., :n] * alpha[0] + bias[:n]
    z_post = norm_out_forward[..., n:2 * n] * alpha[1] + bias[n:2 * n]
    z_res = (norm_out_forward[..., 2 * n:] * alpha[2] + bias[2 * n:]).reshape(B, S, n, n)

    grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=hc_eps)
    grad_z_post = 2 * SigmoidGrad(z_post, grad_h_post, is_pre=False, hc_eps=hc_eps)
    sk_grad = SinkhornGrad(grad_h_res, sum_out, norm_out)
    grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)

    grad_bias = torch.cat([grad_z_pre.sum((0, 1)), grad_z_post.sum((0, 1)), grad_z_res.sum((0, 1))])

    grad_alpha = torch.tensor([
        (norm_out_forward[..., :n] * grad_z_pre).sum(),
        (norm_out_forward[..., n:2 * n] * grad_z_post).sum(),
        (norm_out_forward[..., 2 * n:] * grad_z_res).sum(),
    ], dtype=torch.float32)

    grad_norm_out = torch.cat([grad_z_pre * alpha[0], grad_z_post * alpha[1], grad_z_res * alpha[2]], dim=-1)
    grad_hc_before_norm = grad_norm_out * inv_rms
    grad_inv_rms = (hc_before_norm * grad_norm_out).sum(-1, keepdim=True)

    x_flat = x_fp32.flatten(2)
    grad_x_from_rms = RMSNormGrad(x_flat, inv_rms, grad_inv_rms)
    grad_x_from_matmul = grad_hc_before_norm @ phi
    grad_phi = grad_hc_before_norm.flatten(0, 1).T @ x_flat.flatten(0, 1)

    grad_x = (grad_x_from_hin.view(x.shape)
              + grad_x_from_rms.view(x.shape)
              + grad_x_from_matmul.view(x.shape))
    return grad_x, grad_phi, grad_alpha, grad_bias


def mhc_pre_sinkhorn_backward_golden(grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
                                     h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
                                     hc_eps=1e-6, **kwargs):
    x_np_dtype = np.asarray(x).dtype          # bf16/fp16 — preserve for grad_x output

    def T(a):
        return torch.from_numpy(np.ascontiguousarray(np.asarray(a)).astype(np.float32))

    args = [T(a) for a in (grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
                           h_pre, hc_before_norm, inv_rms, sum_out, norm_out)]
    grad_x, grad_phi, grad_alpha, grad_bias = _core(*args, hc_eps=float(hc_eps))

    grad_x_np = grad_x.detach().numpy().astype(x_np_dtype)      # match grad_x output dtype
    grad_phi_np = grad_phi.detach().numpy().astype(np.float32)
    grad_alpha_np = grad_alpha.detach().numpy().astype(np.float32)
    grad_bias_np = grad_bias.detach().numpy().astype(np.float32)
    return [grad_x_np, grad_phi_np, grad_alpha_np, grad_bias_np]
