# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mla_prolog_v5 CPU golden 实现（fp32）。

公式（与 mla_prolog_v3 README 一致）：
    cQ  = alpha_q  * RmsNorm(x @ Wdq, gamma_cq, eps)
    qN  = per-head(cQ @ Wuq) @ Wuk
    qR  = RoPE(cQ @ Wqr)
    cKV = alpha_kv * RmsNorm(x @ Wdkv, gamma_ckv, eps)
    kR  = RoPE(x @ Wkr)

RoPE 语义（与 arch22 kernel service_rope_arch22.h 的奇偶拆分实现一致，
且约定 ropeSin/ropeCos 每行 Dr 个元素、前后两半重复）：
    out[2j]   = x[2j] * sin_j + x[2j+1] * cos_j
    out[2j+1] = x[2j] * cos_j + x[2j+1] * sin_j
"""

import torch


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """按最后一维 RmsNorm。x: (..., n), gamma: (n,)"""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x / torch.sqrt(var + eps) * gamma


def rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """x/sin/cos: (..., Dr)，sin/cos 前后半重复布局。"""
    d = x.shape[-1]
    half = d // 2
    even = x[..., 0::2]  # (..., d/2) 偶数位
    odd = x[..., 1::2]  # (..., d/2) 奇数位
    s = sin[..., :half]
    c = cos[..., :half]
    out = torch.empty_like(x)
    out[..., 0::2] = even * s + odd * c
    out[..., 1::2] = even * c + odd * s
    return out


def mla_prolog_golden(
    x: torch.Tensor,  # (T, He) bf16->fp32
    w_dq: torch.Tensor,  # (He, Hcq)
    w_uq_qr: torch.Tensor,  # (Hcq, N*(D+Dr))
    w_uk: torch.Tensor,  # (N, D, Hckv)
    w_dkv_kr: torch.Tensor,  # (He, Hckv+Dr)
    gamma_cq: torch.Tensor,  # (Hcq,)
    gamma_ckv: torch.Tensor,  # (Hckv,)
    rope_sin: torch.Tensor,  # (T, Dr) or None
    rope_cos: torch.Tensor,  # (T, Dr) or None
    *,
    n: int,
    d: int,
    hckv: int,
    dr: int,
    eps_cq: float = 1e-5,
    eps_ckv: float = 1e-5,
    alpha_q: float = 1.0,
    alpha_kv: float = 1.0,
    do_rope: bool = True,
) -> dict:
    """返回 dict: query(T,N,Hckv), query_rope(T,N,Dr), query_norm(T,Hcq), c_kv(T,Hckv), k_r(T,Dr)"""
    t = x.shape[0]
    hckv_plus_dr = w_dkv_kr.shape[1]

    # cQ
    cq = alpha_q * rmsnorm(x @ w_dq, gamma_cq, eps_cq)  # (T, Hcq)
    # qN: cQ @ Wuq -> (T, N*D) -> per-head @ Wuk -> (T, N, Hckv)
    w_uq = w_uq_qr[:, : n * d]  # (Hcq, N*D)
    q_c = (cq @ w_uq).view(t, n, d)  # (T, N, D)
    query = torch.einsum("tnd,ndh->tnh", q_c, w_uk.to(q_c.dtype))  # (T, N, Hckv)
    # qR
    w_qr = w_uq_qr[:, n * d :]  # (Hcq, N*Dr)
    q_r = (cq @ w_qr).view(t, n, dr)  # (T, N, Dr)
    if do_rope:
        query_rope = rope(q_r, rope_sin.unsqueeze(1), rope_cos.unsqueeze(1))
    else:
        query_rope = q_r
    # cKV / kR
    w_dkv = w_dkv_kr[:, :hckv]  # (He, Hckv)
    w_kr = w_dkv_kr[:, hckv:]  # (He, Dr)
    c_kv = alpha_kv * rmsnorm(x @ w_dkv, gamma_ckv, eps_ckv)  # (T, Hckv)
    k_r_pre = x @ w_kr  # (T, Dr)
    if do_rope:
        k_r = rope(k_r_pre, rope_sin, rope_cos)
    else:
        k_r = k_r_pre
    return {
        "query": query,
        "query_rope": query_rope,
        "query_norm": cq,
        "c_kv": c_kv,
        "k_r": k_r,
    }
