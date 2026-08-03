# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

__input__ = {
    "aclnn": {"aclnnMhcPreSinkhornBackward": "mhc_pre_sinkhorn_backward_input"},
}


def mhc_pre_sinkhorn_backward_input(
    grad_hin,
    grad_h_post,
    grad_h_res,
    x,
    phi,
    alpha,
    bias,
    h_pre,
    hc_before_norm,
    inv_rms,
    sum_out,
    norm_out,
    hc_eps,
    grad_x_out,
    grad_phi_out,
    grad_alpha_out,
    grad_bias_out,
    **kwargs,
):
    """
    Generate test inputs for aclnnMhcPreSinkhornBackward.
    Supports both BSND (4D) and TND (3D) input formats.

    Forward pass is computed in torch to produce the cached intermediate tensors
    (hPre, hcBeforeNorm, invRms, sumOut, normOut). Random gradients are created
    for grad_hin / grad_h_post / grad_h_res, then autograd is used to obtain the
    reference gradients (grad_x, grad_phi, grad_alpha, grad_bias).

    All tensor parameters are torch.Tensor on NPU. They are filled in-place.
    """
    is_3d = x.dim() == 3
    n = x.shape[-2]
    c = x.shape[-1]
    sinkhorn_iters = int(sum_out.shape[0] // 2)

    x.copy_(torch.randn_like(x))
    phi.copy_(torch.randn(phi.shape, dtype=torch.float32, device=x.device))
    alpha.copy_(torch.randn(3, dtype=torch.float32, device=x.device) * 0.1)
    bias.copy_(torch.randn(n * n + 2 * n, dtype=torch.float32, device=x.device))

    x.requires_grad_(True)
    phi.requires_grad_(True)
    alpha.requires_grad_(True)
    bias.requires_grad_(True)

    h_pre_fwd, hc_before_norm_fwd, inv_rms_fwd, sum_out_fwd, norm_out_fwd, hin_fwd = (
        _mhc_pre_sinkhorn_forward(x, phi, alpha, bias, sinkhorn_iters, n, float(hc_eps))
    )

    grad_hin.copy_(torch.randn(hin_fwd.shape, dtype=x.dtype, device=x.device))
    grad_h_post.copy_(
        torch.randn(h_pre_fwd.shape, dtype=torch.float32, device=x.device)
    )
    grad_h_res.copy_(
        torch.randn(h_pre_fwd.shape[:-1] + (n, n), dtype=torch.float32, device=x.device)
    )

    hin_fwd.backward(grad_hin.to(torch.float32), retain_graph=True)
    h_pre_fwd.backward(grad_h_post, retain_graph=True)
    grad_h_res_fwd = grad_h_res.reshape(h_pre_fwd.shape[:-1] + (n, n))
    _sinkhorn_backward_autograd(
        h_pre_fwd,
        grad_h_res_fwd,
        sum_out_fwd,
        norm_out_fwd,
        sinkhorn_iters,
        float(hc_eps),
    )

    h_pre.copy_(h_pre_fwd.detach())
    hc_before_norm.copy_(hc_before_norm_fwd.detach())
    inv_rms.copy_(inv_rms_fwd.detach())
    sum_out.copy_(sum_out_fwd.detach())
    norm_out.copy_(norm_out_fwd.detach())

    grad_x_out.copy_(x.grad.to(x.dtype))
    grad_phi_out.copy_(phi.grad)
    grad_alpha_out.copy_(alpha.grad)
    grad_bias_out.copy_(bias.grad)

    return [
        grad_hin,
        grad_h_post,
        grad_h_res,
        x.detach(),
        phi.detach(),
        alpha.detach(),
        bias.detach(),
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        hc_eps,
        grad_x_out,
        grad_phi_out,
        grad_alpha_out,
        grad_bias_out,
    ]


def _mhc_pre_sinkhorn_forward(x, phi, alpha, bias, sinkhorn_iters, n, hc_eps):
    """
    Forward pass (torch reference) producing all cached intermediates.
    Internally handles both 3D (TND) and 4D (BSND) input by operating on
    the last dimensions; sum_out/norm_out are stacked on dim=0.

    Returns: h_pre, hc_before_norm, inv_rms, sum_out, norm_out, hin
    """
    x_fp32 = x.float()
    x_flat = x_fp32.reshape(*x_fp32.shape[:-2], -1)

    hc_before_norm = torch.nn.functional.linear(x_flat, phi) + bias
    inv_rms = torch.rsqrt(x_flat.float().square().mean(-1, keepdim=True) + hc_eps)
    norm_out_forward = hc_before_norm * inv_rms

    z_pre = norm_out_forward[..., :n] * alpha[0] + bias[:n]
    z_post = norm_out_forward[..., n : 2 * n] * alpha[1] + bias[n : 2 * n]
    z_res = norm_out_forward[..., 2 * n :] * alpha[2] + bias[2 * n :]

    h_pre = torch.sigmoid(z_pre + hc_eps)
    h_post = 2 * torch.sigmoid(z_post)

    comb = z_res.reshape(*z_res.shape[:-1], n, n)
    comb = torch.exp(comb - comb.max(dim=-1, keepdim=True).values)

    sum_out_list = []
    norm_out_list = []
    for _ in range(sinkhorn_iters):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + hc_eps)
        norm_out_list.append(comb)
        sum_out_list.append(row_sum.squeeze(-1) + hc_eps)

        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + hc_eps)
        sum_out_list.append(col_sum.squeeze(-2) + hc_eps)
        norm_out_list.append(comb)

    sum_out = torch.stack(sum_out_list, dim=0)
    norm_out = torch.stack(norm_out_list, dim=0)

    hin = (h_pre.unsqueeze(-1) * x_flat).sum(-1).to(x.dtype)

    return h_pre, hc_before_norm, inv_rms, sum_out, norm_out, hin


def _sinkhorn_backward_autograd(
    h_pre, grad_h_res, sum_out, norm_out, sinkhorn_iters, hc_eps
):
    """
    Trigger autograd on the sinkhorn path by back-propagating grad_h_res
    through the cached forward graph.
    """
    grad_h_res_flat = grad_h_res.reshape(*grad_h_res.shape[:-2], -1)
    h_pre.grad = (
        h_pre.grad + grad_h_res_flat if h_pre.grad is not None else grad_h_res_flat
    )
