#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
    "kernel": {
        "mhc_pre_sinkhorn_backward": "mhc_pre_sinkhorn_backward_golden"
    }
}

import numpy as np
import torch


def _to_float32(arr):
    if arr is None:
        return None
    t = torch.from_numpy(arr)
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        t = t.to(torch.float32)
    return t


def mhc_pre_sinkhorn_backward_golden(
    gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre,
    hcBeforeNorm, invRms, sumOut, normOut, hcEps=1e-6, **kwargs
):
    gradHin_f = _to_float32(gradHin)
    gradHPost_f = _to_float32(gradHPost)
    gradHRes_f = _to_float32(gradHRes)
    x_f = _to_float32(x)
    phi_f = _to_float32(phi)
    alpha_f = _to_float32(alpha)
    bias_f = _to_float32(bias)
    hPre_f = _to_float32(hPre)
    hcBeforeNorm_f = _to_float32(hcBeforeNorm)
    invRms_f = _to_float32(invRms)
    sumOut_f = _to_float32(sumOut)
    normOut_f = _to_float32(normOut)

    eps = float(hcEps)
    N = 4
    sk_iter_count = 20
    B = x_f.shape[0]
    S = x_f.shape[1]
    C = x_f.shape[3]
    hcMix = 2 * N + N * N

    alphaPre = alpha_f[0]
    alphaPost = alpha_f[1]
    alphaRes = alpha_f[2]
    biasPre = bias_f[:N]
    biasPost = bias_f[N:2 * N]
    biasRes = bias_f[2 * N:]

    # ---- Phase 1: Sinkhorn backward ----
    grad_curr = gradHRes_f.clone()
    for i in range(sk_iter_count - 1, 0, -1):
        # Column normalization backward
        dotProd_col = (grad_curr * normOut_f[2 * i + 1]).sum(dim=-2, keepdim=True)
        grad_curr = (grad_curr - dotProd_col) / (sumOut_f[2 * i + 1].unsqueeze(-2) + eps)

        # Row normalization backward
        dotProd_row = (grad_curr * normOut_f[2 * i]).sum(dim=-1, keepdim=True)
        grad_curr = (grad_curr - dotProd_row) / (sumOut_f[2 * i].unsqueeze(-1) + eps)

    # Last iteration (i=0)
    dotProd_col0 = (grad_curr * normOut_f[1]).sum(dim=-2, keepdim=True)
    grad_curr = (grad_curr - dotProd_col0) / (sumOut_f[1].unsqueeze(-2) + eps)
    dotProd_row0 = (grad_curr * normOut_f[0]).sum(dim=-1, keepdim=True)
    HResGrad = (grad_curr - dotProd_row0) * normOut_f[0]

    # ---- Phase 2: Output combination backward ----
    # hIn = sum_i x[:, :, i, :] * hPre[:, :, i] (forward)
    HPreGrad = (gradHin_f.unsqueeze(-2) * x_f).sum(dim=-1)
    xGradVec3 = gradHin_f.unsqueeze(-2) * hPre_f.unsqueeze(-1)

    # ---- Phase 3: Sigmoid gate backward (H_pre) ----
    # Forward: hPre = sigmoid(alphaPre * hPre1 + biasPre) + hcEps
    hMixTmp = hcBeforeNorm_f * invRms_f
    hPre1 = hMixTmp[..., :N]
    hPost1 = hMixTmp[..., N:2 * N]
    hRes1 = hMixTmp[..., 2 * N:]

    s = hPre_f - eps
    HPre2Grad = HPreGrad * s * (1 - s)
    HPre1Grad = HPre2Grad * alphaPre
    alphaPreGrad = (HPre2Grad * hPre1).sum()
    biasPreGrad = HPre2Grad.sum(dim=(0, 1))

    # ---- Phase 4: Sigmoid gate backward (H_post) ----
    # Forward: hPost = 2 * sigmoid(alphaPost * hPost1 + biasPost)
    hPost = 2 * torch.sigmoid(alphaPost * hPost1 + biasPost)
    HPost2Grad = gradHPost_f * hPost * (1 - hPost / 2)
    HPost1Grad = HPost2Grad * alphaPost
    alphaPostGrad = (HPost2Grad * hPost1).sum()
    biasPostGrad = HPost2Grad.sum(dim=(0, 1))

    # ---- Phase 5: Residual linear backward (H_res) ----
    # Forward: hRes2 = alphaRes * hRes1_reshaped + biasRes_reshaped
    hRes1_mat = hRes1.reshape(B, S, N, N)
    hRes2 = alphaRes * hRes1_mat + biasRes.reshape(N, N)
    HRes2Grad = HResGrad * alphaRes
    alphaResGrad = (HResGrad * hRes2).sum()
    biasResGrad = HResGrad.sum(dim=(0, 1)).reshape(N * N)
    HRes1Grad = HRes2Grad.reshape(B, S, N * N)

    # ---- Phase 6: RMSNorm fusion backward ----
    hMixTmpGrad = torch.cat([HPre1Grad, HPost1Grad, HRes1Grad], dim=-1)
    hMixGrad = hMixTmpGrad * invRms_f
    invRmsGrad = (hMixTmpGrad * hcBeforeNorm_f).sum(dim=-1, keepdim=True)

    # ---- Phase 7: Linear projection backward ----
    # Forward: hMix = xRs @ phi^T, xRs = x * gamma (gamma = invRms)
    gamma = invRms_f
    xRs = x_f * gamma

    xRs_2d = xRs.reshape(B * S, N * C)
    hMixGrad_2d = hMixGrad.reshape(B * S, hcMix)

    xRsGrad_2d = hMixGrad_2d @ phi_f
    phiGrad = hMixGrad_2d.T @ xRs_2d

    xRsGrad = xRsGrad_2d.reshape(B, S, N, C)

    # ---- Phase 8: Feature scaling backward ----
    xGradMm = xRsGrad * gamma

    # ---- Phase 9: RMSNorm gradient backward ----
    xRsGradInv = -(invRmsGrad * invRms_f ** 3 / (N * C)) * xRs
    xRsGrad_total = xGradMm + xRsGradInv
    xGradVec1 = xRsGrad_total

    # ---- Phase 10: Final x gradient ----
    xGrad = xGradVec3 + xGradVec1

    # ---- Build output ----
    gradAlpha_out = torch.stack([alphaPreGrad, alphaPostGrad, alphaResGrad])
    gradBias_out = torch.cat([biasPreGrad, biasPostGrad, biasResGrad])

    output_dtypes = kwargs.get('output_dtypes', None)
    if output_dtypes is not None:
        gradX_dtype = output_dtypes[0]
        if gradX_dtype == 'bfloat16' or gradX_dtype == 'bf16':
            xGrad = xGrad.to(torch.bfloat16)
        elif gradX_dtype == 'float16' or gradX_dtype == 'fp16':
            xGrad = xGrad.to(torch.float16)

    gradX_out = xGrad.numpy()
    gradPhi_out = phiGrad.numpy()
    gradAlpha_out_np = gradAlpha_out.numpy()
    gradBias_out_np = gradBias_out.numpy()

    return gradX_out, gradPhi_out, gradAlpha_out_np, gradBias_out_np