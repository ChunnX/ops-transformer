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

关键布局契约（依据 kernel 实现与 NPU 实测标定，见 docs/LOG.md 2026-08-31 记录）：
1. weight_uq_qr 的列布局为 **按头交错**：[Qc_h(D) | Qr_h(Dr)] × N（注意不是 [Wuq|Wqr] 分离拼接）。
2. weight_dkv_kr 的列布局为 **分离**：[W_dkv(Hckv) | W_kr(Dr)]。
3. RoPE：输入 x 视为 32 个 (x[2j], x[2j+1]) 旋转对；sin/cos 形状 (..., Dr) 且前后半重复
   （标准 RoPE 频率表形态）；输出为 [前半分量 | 后半分量] 布局：
       out[j]      = x[2j]*cos[j]   - x[2j+1]*sin[j]   (j < Dr/2)
       out[Dr/2+j] = x[2j]*sin[j]   + x[2j+1]*cos[j]
"""

import torch


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """按最后一维 RmsNorm。x: (..., n), gamma: (n,)"""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x / torch.sqrt(var + eps) * gamma


def rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """RoPE 旋转（kernel 语义，见模块 docstring 第 3 条）。

    x: (..., Dr)；sin/cos: (..., Dr) 前后半重复。输出与 x 同形（[前半|后半] 布局）。
    """
    d = x.shape[-1]
    half = d // 2
    a = x[..., 0::2]  # 偶数位 (..., d/2)
    b = x[..., 1::2]  # 奇数位 (..., d/2)
    s = sin[..., :half]
    c = cos[..., :half]
    out = torch.empty_like(x)
    out[..., :half] = a * c - b * s
    out[..., half:] = a * s + b * c
    return out


def mla_prolog_golden(
    x: torch.Tensor,  # (T, He)
    w_dq: torch.Tensor,  # (He, Hcq)
    w_uq_qr: torch.Tensor,  # (Hcq, N*(D+Dr)) 按头交错 [Qc_h|Qr_h]
    w_uk: torch.Tensor,  # (N, D, Hckv)
    w_dkv_kr: torch.Tensor,  # (He, Hckv+Dr) 分离 [W_dkv|W_kr]
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

    # cQ（下采样 + RmsNorm）
    cq = alpha_q * rmsnorm(x @ w_dq, gamma_cq, eps_cq)  # (T, Hcq)

    # mm3：cQ @ W_uq_qr，结果列布局与权重一致（按头交错）
    mm3 = cq @ w_uq_qr  # (T, N*(D+Dr))
    mm3 = mm3.view(t, n, d + dr)
    qc = mm3[:, :, :d]  # (T, N, D) 各头 Qc
    qr = mm3[:, :, d:]  # (T, N, Dr) 各头 Qr

    # query：per-head (Qc @ W_uk)
    query = torch.einsum("tnd,ndh->tnh", qc, w_uk.to(qc.dtype))  # (T, N, Hckv)

    # query_rope
    if do_rope:
        query_rope = rope(
            qr, rope_sin.unsqueeze(1), rope_cos.unsqueeze(1)
        )  # (T, N, Dr)
    else:
        query_rope = qr

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
