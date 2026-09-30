# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU reference for the quantized attention algorithm delivered by this operator.

Never used as an NPU fallback. Streaming KV blocks bound score storage.
"""

import math

import torch

# Tokens sharing one INT8 scale, mirroring the kernel's SageParams qg/kg.
QUAN_GRAN_GROUPS = {"per_thread": (4, 16), "per_warp": (32, 64)}

# The kernel's PV MMA carries 2^-PV_PARTIAL_SHIFT on P's E8M0 scale so that a
# full 256-token block of ceiling probabilities cannot overflow the fp32+fp16
# fold; the reciprocal at finalize multiplies it back. A power of two, so the
# golden stays bit-exact everywhere except the FP16 fold it models.
PV_PARTIAL_SHIFT = 8
PV_PARTIAL_SCALE = 2.0**-PV_PARTIAL_SHIFT
PV_PARTIAL_UNDO = 2.0**PV_PARTIAL_SHIFT


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


def _group_amax(a, group):
    """Max-reduce a [B,H,S,1] row magnitude over consecutive token groups."""
    if group <= 1:
        return a
    seq = a.shape[-2]
    padded = torch.nn.functional.pad(a, (0, 0, 0, (-seq) % group))
    per_group = padded.reshape(*a.shape[:-2], -1, group).amax(-1)
    return torch.repeat_interleave(per_group, group, dim=-1)[..., :seq].unsqueeze(-1)


def quantize(q, k, v, *, smooth_k=True, qg=1, kg=1, smooth_v=False):
    """Quantize Q/K to INT8 and V to MXFP8 with the kernel's scale sharing."""
    q, k, v = q.float(), k.float(), v.float()
    mean = k.mean(-2, keepdim=True) if smooth_k else torch.zeros_like(k[..., :1, :])
    kc = k - mean
    mean_v = v.mean(-2, keepdim=True) if smooth_v else torch.zeros_like(v[..., :1, :])

    def token(x, group):
        a = _group_amax(x.abs().amax(-1, keepdim=True), group)
        scale = torch.where(a > 0, a / 127, torch.ones_like(a))
        return (x / scale).round().clamp(-127, 127).to(torch.int8), scale.squeeze(-1)

    q8, qs = token(q, qg)
    k8, ks = token(kc, kg)
    v8, vs = quantize_v_mx(v - mean_v)
    return q8, k8, v8, qs, ks, mean.squeeze(-2), vs, mean_v.squeeze(-2)


def prepare_scales(qs, ks, sm_scale):
    """Fixed FP16 factors; no statistics across tokens or heads."""
    factor = (
        torch.as_tensor(sm_scale, dtype=qs.dtype, device=qs.device).abs() * 2048
    ).sqrt()
    qfactor = -factor if sm_scale < 0 else factor
    return (qs * qfactor).half(), (ks * factor).half()


# The FP16 scale path mirrors the kernel's 64-row query tile; block_k follows
# the kernel's key tile, so the golden reproduces its online-softmax grouping.
def attention(
    q,
    k,
    v,
    *,
    sm_scale=None,
    is_causal=False,
    smooth_k=True,
    smooth_v=False,
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32+fp16",
    tensor_layout="HND",
    return_lse=False,
    block_k=256,
):
    """CPU golden for the INT8 QK / MXFP8 PV algorithm at the requested granularities."""
    if tensor_layout == "NHD":
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    b, hq, sq, d = q.shape
    hk, sk = k.shape[1:3]
    if is_causal and sq != sk:
        raise ValueError("causal requires Sq == Sk")
    if qk_quant_gran not in QUAN_GRAN_GROUPS:
        raise ValueError(f"qk_quant_gran must be one of {sorted(QUAN_GRAN_GROUPS)}")
    qg, kg = QUAN_GRAN_GROUPS[qk_quant_gran]
    fp16_partial = pv_accum_dtype == "fp32+fp16"
    smooth_v = bool(smooth_v) and pv_accum_dtype == "fp32"
    scale = 1 / math.sqrt(d) if sm_scale is None else float(sm_scale)
    q8, k8, v8, qs, ks, mean, vs, mean_v = quantize(
        q, k, v, smooth_k=smooth_k, qg=qg, kg=kg, smooth_v=smooth_v
    )
    v_dequant = v8.float() * vs.repeat_interleave(32, dim=-2)[..., :sk, :]
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
                qh = qhalf[:, h, start:end, None]
                khs = khalf[:, kh, None, j:je]
                coeff = (qh.float() * khs.float()).half()
                scores = ((scores / 2048).half().float() * coeff.float()).half()
                if is_causal:
                    mask = (
                        torch.arange(j, je, device=q.device)[None, :]
                        > torch.arange(start, end, device=q.device)[:, None]
                    )
                    scores.masked_fill_(mask, -torch.inf)
                # FP16 shifted maxima are the online normalization state.
                block_max = (scores.amax(-1, keepdim=True) - log_scale).half()
                nm = torch.maximum(m.half(), block_max).float()
                scores = scores.float()
                alpha = (m - nm).exp()
                prob = (scores - nm).exp()
                den = den * alpha + prob.sum(-1, keepdim=True)
                # Device FP8 Cast saturates; torch's E4M3 conversion alone can
                # produce NaN when a rounded FP16 max makes P exceed its range.
                p8 = prob.clamp(max=448.0).to(torch.float8_e4m3fn).float()
                partial = (p8 * PV_PARTIAL_SCALE) @ v_dequant[:, kh, j:je]
                if fp16_partial:
                    partial = partial.half().float()
                acc = acc * alpha + partial
                m = nm
            result = acc * (PV_PARTIAL_UNDO / den)
            if smooth_v:
                result = result + mean_v[:, kh, None, :]
            output[:, h, start:end] = result.to(q.dtype)
            lse[:, h, start:end] = (m + den.log()).squeeze(-1)
            if smooth_k:
                lse[:, h, start:end] += (
                    q[:, h, start:end].float() * mean[:, kh, None, :]
                ).sum(-1) * scale
    if tensor_layout == "NHD":
        output = output.transpose(1, 2)
    return (output, lse) if return_lse else output
