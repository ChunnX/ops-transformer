"""CPU reference for the quantized attention algorithm delivered by this operator.

Never used as an NPU fallback. Streaming KV blocks bound score storage.
"""

import math
import torch

QUAN_GRAN_GROUPS = {"per_thread": (4, 16), "per_warp": (32, 64)}

PV_PARTIAL_SHIFT = 8
PV_PARTIAL_SCALE = 2.0**-PV_PARTIAL_SHIFT
PV_PARTIAL_UNDO = 2.0**PV_PARTIAL_SHIFT


def quantize_v_mx(v):
    seq = v.shape[-2]
    padded = torch.nn.functional.pad(v.float(), (0, 0, 0, (-seq) % 32))
    groups = padded.reshape(*v.shape[:-2], -1, 32, v.shape[-1])
    amax = groups.abs().amax(-2, keepdim=True)
    scales = torch.exp2(torch.ceil(torch.log2((amax / 448.0).clamp_min(1e-30))))
    data = (groups / scales).to(torch.float8_e4m3fn)
    return data.reshape_as(padded)[..., :seq, :], scales.squeeze(-2)


def _group_amax(a, group):
    if group <= 1:
        return a
    seq = a.shape[-2]
    padded = torch.nn.functional.pad(a, (0, 0, 0, (-seq) % group))
    per_group = padded.reshape(*a.shape[:-2], -1, group).amax(-1)
    return torch.repeat_interleave(per_group, group, dim=-1)[..., :seq].unsqueeze(-1)


def quantize(q, k, v, *, smooth_k=True, qg=1, kg=1, smooth_v=False):
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
    smooth_v=False,
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32+fp16",
    tensor_layout="HND",
    return_lse=False,
    block_k=256,
):
    if tensor_layout == "NHD":
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    b, hq, sq, d = q.shape
    hk, sk = k.shape[1:3]
    if is_causal and sq != sk:
        raise ValueError("causal requires Sq == Sk")
    if qk_quant_gran not in QUAN_GRAN_GROUPS:
        raise ValueError(f"qk_quant_gran must be one of {sorted(QUAN_GRAN_GROUPS)}")
    qg, kg = QUAN_GRAN_GROUPS[qk_quant_gran]
    if pv_accum_dtype not in ("fp32", "fp32+fp32", "fp32+fp16"):
        raise ValueError(f"unknown pv_accum_dtype {pv_accum_dtype}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    q8, k8, v8, qs, ks, mean_k, vs, mean_v = quantize(
        q, k, v, smooth_k=smooth_k, qg=qg, kg=kg, smooth_v=smooth_v
    )
    qs_h, ks_h = prepare_scales(qs, ks, sm_scale)

    o = torch.zeros(b, hq, sq, d, dtype=torch.float32, device=q.device)
    lse = torch.full((b, hq, sq), -float("inf"), dtype=torch.float32, device=q.device) if return_lse else None
    gqa_ratio = hq // hk

    for i in range(0, sq, 64):
        q_len = min(64, sq - i)
        q_chunk = q8[:, :, i:i + q_len]
        qs_chunk = qs_h[:, :, i:i + q_len].float()
        
        m_i = torch.full((b, hq, q_len), -float("inf"), dtype=torch.float32, device=q.device)
        l_i = torch.zeros((b, hq, q_len), dtype=torch.float32, device=q.device)
        acc = torch.zeros((b, hq, q_len, d), dtype=torch.float32, device=q.device)
        
        k_limit = min(sk, i + 64) if is_causal else sk
        for j in range(0, k_limit, block_k):
            k_len = min(block_k, sk - j)
            k_chunk = k8[:, :, j:j + k_len]
            ks_chunk = ks_h[:, :, j:j + k_len].float()
            v_chunk = v8[:, :, j:j + k_len]
            vs_chunk = vs[:, :, j // 32:(j + k_len + 31) // 32]
            
            if gqa_ratio > 1:
                k_chunk = k_chunk.repeat_interleave(gqa_ratio, dim=1)
                ks_chunk = ks_chunk.repeat_interleave(gqa_ratio, dim=1)
                v_chunk = v_chunk.repeat_interleave(gqa_ratio, dim=1)
                vs_chunk = vs_chunk.repeat_interleave(gqa_ratio, dim=1)
            
            s_int = torch.matmul(q_chunk.float(), k_chunk.float().transpose(-1, -2))
            deq = (qs_chunk.unsqueeze(-1) * ks_chunk.unsqueeze(-2)) * (2.0**-11)
            s = s_int * deq
            
            if is_causal:
                mask = (torch.arange(i, i + q_len, device=q.device).unsqueeze(-1) >= 
                        torch.arange(j, j + k_len, device=q.device).unsqueeze(-2))
                s = s.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -float("inf"))
            
            m_curr = s.amax(dim=-1)
            m_new = torch.maximum(m_i, m_curr)
            
            alpha = torch.exp(m_i - m_new)
            p = torch.exp(s - m_new.unsqueeze(-1))
            
            p_quant = (p * (448.0 * PV_PARTIAL_SCALE)).to(torch.float8_e4m3fn).float() / (448.0 * PV_PARTIAL_SCALE)
            v_dequant = v_chunk.float() * vs_chunk.repeat_interleave(32, dim=-2)[:, :, :k_len]
            
            pv = torch.matmul(p_quant, v_dequant)
            if pv_accum_dtype == "fp32+fp16":
                pv = pv.half().float()
            
            l_i = l_i * alpha + p.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + pv
            m_i = m_new

        out_chunk = acc / l_i.unsqueeze(-1)
        if smooth_v:
            mv = mean_v.repeat_interleave(gqa_ratio, dim=1) if gqa_ratio > 1 else mean_v
            out_chunk = out_chunk + mv.unsqueeze(-2)
            
        o[:, :, i:i + q_len] = out_chunk
        if return_lse:
            lse[:, :, i:i + q_len] = (m_i + torch.log(l_i)) * 1.44269504

    if tensor_layout == "NHD":
        o = o.transpose(1, 2)
    return (o.to(q.dtype), lse) if return_lse else o.to(q.dtype)
