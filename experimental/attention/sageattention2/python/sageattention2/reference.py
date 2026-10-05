# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Explicit CPU reference for the revised per-token INT8 / MXFP8 algorithm.

Never used as an NPU fallback. Streaming KV blocks bound score storage.
"""

import math

import torch


def quantize_v_mx(v):
    """Independent MX reference: E4M3 data and one E8M0 scale per 32 tokens/channel.

    Return unpadded data and FP32 decoded scales shaped [B,H,ceil(S/32),D].
    Zero-padding contributes nothing to a group's magnitude maximum.
    """
    seq = v.shape[-2]
    padded = torch.nn.functional.pad(v.float(), (0, 0, 0, (-seq) % 32))
    groups = padded.reshape(*v.shape[:-2], -1, 32, v.shape[-1])
    amax = groups.abs().amax(-2, keepdim=True)
    scales = torch.exp2(torch.ceil(torch.log2((amax / 448.0).clamp_min(1e-30))))
    data = (groups / scales).to(torch.float8_e4m3fn)
    return data.reshape_as(padded)[..., :seq, :], scales.squeeze(-2)


def quantize(q, k, v, *, smooth_k=True):
    q, k, v = q.float(), k.float(), v.float()
    mean = k.mean(-2, keepdim=True) if smooth_k else torch.zeros_like(k[..., :1, :])
    kc = k - mean

    def token(x):
        a = x.abs().amax(-1, keepdim=True)
        scale = torch.where(a > 0, a / 127, torch.ones_like(a))
        return (x / scale).round().clamp(-127, 127).to(torch.int8), scale.squeeze(-1)

    q8, qs = token(q)
    k8, ks = token(kc)
    v8, vs = quantize_v_mx(v)
    return q8, k8, v8, qs, ks, mean.squeeze(-2), vs


def prepare_scales(qs, ks, sm_scale):
    """Fixed FP16 factors; no statistics across tokens or heads."""
    factor = (
        torch.as_tensor(sm_scale, dtype=qs.dtype, device=qs.device).abs() * 2048
    ).sqrt()
    qfactor = -factor if sm_scale < 0 else factor
    return (qs * qfactor).half(), (ks * factor).half()


def attention(
    q,
    k,
    v,
    *,
    sm_scale=None,
    is_causal=False,
    smooth_k=True,
    tensor_layout="HND",
    return_lse=False,
    block_q=128,
    block_k=256,
    quantized=None,
    pv_storage="fp32",
    score_mode="fp16",
):
    """CPU golden; quantized optionally isolates attention from quantization error.

    quantized is an unpadded HND tuple in the order returned by quantize().
    """
    if tensor_layout == "NHD":
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    b, hq, sq, d = q.shape
    hk, sk = k.shape[1:3]
    if is_causal and sq != sk:
        raise ValueError("causal requires Sq == Sk")
    scale = 1 / math.sqrt(d) if sm_scale is None else float(sm_scale)
    packed = (
        quantize(q, k, v, smooth_k=smooth_k)
        if quantized is None
        else quantized
    )
    q8, k8, v8, qs, ks, mean, vs = packed[:7]
    v_dequant = v8.float() * vs.repeat_interleave(32, dim=-2)[..., :sk, :]
    if score_mode not in ("fp16", "fp32"):
        raise ValueError("score_mode must be fp16 or fp32")
    if score_mode == "fp16":
        block_q = 64
        qhalf, khalf = prepare_scales(qs, ks, scale)
        log_scale = torch.tensor(math.log(448), dtype=torch.float16, device=q.device)
    output = torch.empty_like(q)
    lse = torch.empty((b, hq, sq), dtype=torch.float32, device=q.device)
    for h in range(hq):
        kh = h // (hq // hk)
        for start in range(0, sq, block_q):
            end = min(start + block_q, sq)
            m = torch.full((b, end - start, 1), -torch.inf, device=q.device)
            den = torch.zeros_like(m)
            acc = torch.zeros((b, end - start, d), device=q.device)
            limit = min(sk, ((start // 128) + 1) * 128) if is_causal else sk
            for j in range(0, limit, block_k):
                je = min(j + block_k, sk)
                scores = q8[:, h, start:end].float() @ k8[
                    :, kh, j:je
                ].float().transpose(-2, -1)
                if score_mode == "fp16":
                    qh = qhalf[:, h, start:end, None]
                    khs = khalf[:, kh, None, j:je]
                    coeff = (qh.float() * khs.float()).half()
                    scores = ((scores / 2048).half().float() * coeff.float()).half()
                else:
                    scores *= qs[:, h, start:end, None] * ks[:, kh, None, j:je] * scale
                if is_causal:
                    mask = (
                        torch.arange(j, je, device=q.device)[None, :]
                        > torch.arange(start, end, device=q.device)[:, None]
                    )
                    scores.masked_fill_(mask, -torch.inf)
                if score_mode == "fp16":
                    # FP16 shifted maxima are the online normalization state.
                    block_max = (scores.amax(-1, keepdim=True) - log_scale).half()
                    nm = torch.maximum(m.half(), block_max).float()
                    scores = scores.float()
                else:
                    nm = torch.maximum(m, scores.amax(-1, keepdim=True) - 6.10450458)
                alpha = (m - nm).exp()
                prob = (scores - nm).exp()
                den = den * alpha + prob.sum(-1, keepdim=True)
                # Device FP8 Cast saturates; torch's E4M3 conversion alone can
                # produce NaN when a rounded FP16 max makes P exceed its range.
                p8 = prob.clamp(max=448.0).to(torch.float8_e4m3fn).float()
                partial = p8 @ v_dequant[:, kh, j:je]
                if pv_storage == "bf16":
                    partial = partial.to(torch.bfloat16).float()
                elif pv_storage != "fp32":
                    raise ValueError("pv_storage must be bf16 or fp32")
                acc = acc * alpha + partial
                m = nm
            result = acc / den
            output[:, h, start:end] = result.to(q.dtype)
            lse[:, h, start:end] = (m + den.log()).squeeze(-1)
            if smooth_k:
                lse[:, h, start:end] += (
                    q[:, h, start:end].float() * mean[:, kh, None, :]
                ).sum(-1) * scale
    if tensor_layout == "NHD":
        output = output.transpose(1, 2)
    return (output, lse) if return_lse else output
