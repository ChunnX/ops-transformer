import math
import warnings

import torch

from ._constants import (
    E4M3_MAX,
    E8M0_MIN_EXPONENT,
    KAPPA,
    KV_TILE_ROWS,
    MX_GROUP_TOKENS,
    PER_THREAD_SEG,
    QUERY_TILE_ROWS,
)
from ._utils import (
    pad_to_bucket,
    padded_head_dim,
    sanitize_smooth_v,
    validate_all_inputs,
)


def _e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 码字舍入（往返模拟）；先饱和到 ±448，避免溢出转 NaN。"""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def _int8_quant(x: torch.Tensor, seg: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 对称量化：scale = amax/127，零行（段）→ 1.0（DESIGN_SPEC §1）。

    x: [R, D] FP32。seg=None → per_warp（每 token 一个 scale，返回 [R]）；
    seg=16 → per_thread（每 16 列段一个 scale，返回 [R, D/16]）。
    返回 (q8 整数值 FP32 张量 [R, D], scale)。模型近似：内核将 scale 以 FP16 存储
    （DESIGN_SPEC §5.1/§5.3），参考实现全程 FP32。
    """
    rows, dim = x.shape
    if seg is None:
        amax = x.abs().amax(dim=-1)                     # [R]
        scale = amax / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q8 = torch.round(x / scale.unsqueeze(-1)).clamp(-127.0, 127.0)
        return q8, scale
    groups = dim // seg
    xg = x.view(rows, groups, seg)
    amax = xg.abs().amax(dim=-1)                        # [R, G]
    scale = amax / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q8g = torch.round(xg / scale.unsqueeze(-1)).clamp(-127.0, 127.0)
    return q8g.view(rows, dim), scale


def _qk_score(q8: torch.Tensor, q_scale: torch.Tensor, k8: torch.Tensor, k_scale: torch.Tensor, sm_scale: float,
              seg: int | None) -> torch.Tensor:
    """QK^T 反量化得分；q8/k8 为整数值 FP32，矩阵乘按整值精确（127*127*128 < 2^24）。

    舍入点（DESIGN_SPEC §4.1）：QK^T 结果以 FP16 存储。模型近似：内核按 §4.2 将 scale
    拆为 QH/KH 因子在 FP16 域逐段相乘，参考实现把全部 scale 因子在 FP32 域合并后仅做
    一次 FP16 舍入。
    """
    if seg is None:
        acc = q8 @ k8.transpose(0, 1)                   # [R, C] 整值精确
        score = sm_scale * acc * q_scale.unsqueeze(-1) * k_scale.unsqueeze(0)
    else:
        rows, dim = q8.shape
        cols = k8.shape[0]
        groups = dim // seg
        acc = torch.einsum(
            "igd,jgd->ijg", q8.view(rows, groups, seg), k8.view(cols, groups, seg)
        )                                               # [R, C, G] 整值精确
        score = sm_scale * (
            acc * q_scale.view(rows, groups).unsqueeze(1) * k_scale.view(cols, groups).unsqueeze(0)
        ).sum(dim=-1)
    return score.half().float()


def _mx_quant_v(v: torch.Tensor) -> torch.Tensor:
    """V 的 MX 量化（DESIGN_SPEC §4.8）：32 token × channel 一组，E4M3 码字 + E8M0 组 scale。

    scale = 2^ceil(log2(amax/448))（向上取 2 的幂），全零组 → E8M0 最小正。
    v: [B, Hk, Sk, D] FP32（smooth_v 时已中心化）。返回反量化后的 v_deq，形状同输入。
    """
    batch, head, seq, dim = v.shape
    pad_tokens = (-seq) % MX_GROUP_TOKENS
    if pad_tokens > 0:
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_tokens))
    vg = v.view(batch, head, -1, MX_GROUP_TOKENS, dim)
    amax = vg.abs().amax(dim=3)                         # [B, Hk, G, D]
    # 与内核逐位一致（P1-6 对齐）：内核用 float32 商 x = amax/448 后做 ceil(log2) 位技巧；
    # 参考实现对同一 float32 商取 float64 对数再 ceil（2^k 边界处两者逐位相等，
    # 避免 float32 log2 直接作用于 amax 时除法/对数双重舍入分叉）。
    exponent = torch.ceil(torch.log2((amax / E4M3_MAX).double())).float()
    exponent = torch.where(
        amax > 0, exponent, torch.full_like(exponent, E8M0_MIN_EXPONENT)
    )
    # P1-6 幂边界兜底（与内核 ComputeE8m0Exponent 对称）：若量化后 amax * invScale > 448.0f，自动执行 exponent += 1.0 兜底
    scale = torch.exp2(exponent)
    overflow = (amax / scale > E4M3_MAX) | (amax > (E4M3_MAX * torch.exp2(exponent.double())).float())
    exponent = torch.where(overflow, exponent + 1.0, exponent).clamp(E8M0_MIN_EXPONENT, 127.0)
    scale = torch.exp2(exponent)                        # E8M0：2 的幂
    code = _e4m3_round(vg / scale.unsqueeze(3))
    return (code * scale.unsqueeze(3)).view(batch, head, -1, dim)[:, :, :seq, :]


def sageattn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: float | None = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """CPU 参考实现：与内核同配置（gran × accum × smooth）的数值 golden，仅前向。

    说明：head_dim pad 列在 QK^T、corr 与 PV 中贡献恒为零，故参考实现 pad 后计算、
    输出裁回 head_dim_og 与内核语义等价。
    """
    validate_all_inputs(q, k, v, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype, sm_scale)
    smooth_v = sanitize_smooth_v(smooth_v, pv_accum_dtype)
    kappa = KAPPA[pv_accum_dtype]
    ln_kappa = math.log(kappa)
    seg = None if qk_quant_gran == "per_warp" else PER_THREAD_SEG
    head_dim_og = q.shape[3]
    sm_scale_eff = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim_og)

    def to_hnd(t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 2, 1, 3) if tensor_layout == "NHD" else t

    q_hnd = to_hnd(q).float()
    k_hnd = to_hnd(k).float()
    v_hnd = to_hnd(v).float()
    batch, num_q_heads, seq_q, _ = q_hnd.shape
    num_kv_heads, seq_k = k_hnd.shape[1], k_hnd.shape[2]
    group = num_q_heads // num_kv_heads

    # head_dim pad 到 64/128（pad 列为零，不影响量化 amax 与得分）
    padded = padded_head_dim(head_dim_og)
    if padded != head_dim_og:
        pad_len = pad_to_bucket(head_dim_og, padded)
        q_hnd = torch.nn.functional.pad(q_hnd, (0, pad_len))
        k_hnd = torch.nn.functional.pad(k_hnd, (0, pad_len))
        v_hnd = torch.nn.functional.pad(v_hnd, (0, pad_len))


    # smooth_k：K 沿序列维减逐通道均值（FP32 统计，§4.7）；GQA 时 μ 按组广播（repeat_interleave）
    if smooth_k:
        mu_k = k_hnd.mean(dim=2, keepdim=True)          # [B, Hk, 1, D] FP32
        mu_for_q = mu_k.repeat_interleave(group, dim=1)  # [B, Hq, 1, D]
        k_centered = k_hnd - mu_k
    else:
        mu_for_q = torch.zeros_like(q_hnd[:, :, :1, :])
        k_centered = k_hnd

    # K 一次性量化（阶段 A，§3.1）；行 = (b, kh, token) 展开为 [B*Hk*Sk, D]
    k_rows = k_centered.reshape(batch * num_kv_heads * seq_k, -1)
    k8, k_scale = _int8_quant(k_rows, seg)
    k8 = k8.view(batch, num_kv_heads, seq_k, -1)
    if seg is None:
        k_scale = k_scale.view(batch, num_kv_heads, seq_k)          # per_warp：[B, Hk, Sk]
    else:
        k_scale = k_scale.view(batch, num_kv_heads, seq_k, -1)      # per_thread：[B, Hk, Sk, G]

    # V 的 MX 量化；smooth_v 仅 fp32 档生效（§4.5），输出侧补回 μ_v
    smooth_v_active = smooth_v and pv_accum_dtype == "fp32"
    if smooth_v_active:
        mu_v = v_hnd.mean(dim=2, keepdim=True)          # [B, Hk, 1, D]
        v_deq = _mx_quant_v(v_hnd - mu_v)
    else:
        mu_v = torch.zeros_like(v_hnd[:, :, :1, :])
        v_deq = _mx_quant_v(v_hnd)

    out_f32 = torch.empty(batch, num_q_heads, seq_q, padded, dtype=torch.float32)
    lse_f32 = torch.empty(batch, num_q_heads, seq_q, dtype=torch.float32)

    for b in range(batch):
        for h in range(num_q_heads):
            kh = h // group
            q_rows = q_hnd[b, h]                        # [Sq, D]
            k8_bh = k8[b, kh]
            k_scale_bh = k_scale[b, kh]
            v_bh = v_deq[b, kh]
            mu_bh = mu_for_q[b, h, 0]
            mu_v_bh = mu_v[b, kh, 0]
            for m0 in range(0, seq_q, QUERY_TILE_ROWS):
                rows = min(QUERY_TILE_ROWS, seq_q - m0)
                q_tile = q_rows[m0:m0 + rows]
                # Q 每 tile 量化一次（§3.3）；corr = q·μ 用原始 FP32 q（GQA 自动取本 kh 的 μ）
                q8, q_scale = _int8_quant(q_tile, seg)
                corr = (q_tile * mu_bh).sum(dim=-1)     # [rows] FP32
                m_run = torch.full((rows,), -float("inf"), dtype=torch.float32)
                denom = torch.zeros(rows, dtype=torch.float32)
                acc_o = torch.zeros(rows, padded, dtype=torch.float32)
                for n0 in range(0, seq_k, KV_TILE_ROWS):
                    cols = min(KV_TILE_ROWS, seq_k - n0)
                    score = _qk_score(q8, q_scale, k8_bh[n0:n0 + cols], k_scale_bh[n0:n0 + cols], sm_scale_eff, seg)
                    if is_causal:
                        q_idx = torch.arange(m0, m0 + rows).unsqueeze(-1)
                        k_idx = torch.arange(n0, n0 + cols).unsqueeze(0)
                        score = score.masked_fill(k_idx > q_idx, -float("inf"))
                    # 自然域 online softmax（§4.3）：m' 存平移后的值，p = exp(S - m') ≤ κ
                    m_blk = score.amax(dim=-1)
                    m_new = torch.maximum(m_run, m_blk - ln_kappa)
                    alpha = torch.exp(m_run - m_new)
                    p = torch.exp(score - m_new.unsqueeze(-1))
                    p8 = _e4m3_round(p)                 # P 码字（MX scale 恒为 2^0，§4.3）
                    partial = p8 @ v_bh[n0:n0 + cols]   # FP32 部分和
                    if pv_accum_dtype == "fp32+fp16":
                        # 舍入点（§4.4 两级累加）：部分和经 FP16 往返后折叠进 O
                        partial = partial.half().float()
                    # 同源分母（§4.5）：累加反量化后的 E4M3 码字
                    denom = alpha * denom + p8.sum(dim=-1)
                    acc_o = alpha.unsqueeze(-1) * acc_o + partial
                    m_run = m_new
                # O 归一化 + smooth_v 补偿（§4.5）：Σp·(v-μ_v)/denom + μ_v = Σp·v/denom
                safe_denom = denom.unsqueeze(-1)
                inv_denom = torch.where(safe_denom > 0, 1.0 / safe_denom, torch.zeros_like(safe_denom))
                out_f32[b, h, m0:m0 + rows] = acc_o * inv_denom + mu_v_bh.unsqueeze(0)
                # LSE 全核内闭环（§4.6）：lse = ln(denom) + m' + s·(q·μ)
                safe_log_denom = torch.where(denom > 0, torch.log(denom), torch.full_like(denom, float("-inf")))
                lse_f32[b, h, m0:m0 + rows] = safe_log_denom + m_run + sm_scale_eff * corr

    # 舍入点：epilogue cast 到输入 dtype；裁回 head_dim_og
    out = out_f32[..., :head_dim_og].to(q.dtype).to(q.device)
    lse = lse_f32.to(q.device)

    def from_hnd(t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 2, 1, 3) if tensor_layout == "NHD" else t

    out = from_hnd(out).contiguous()
    if return_lse:
        return out, lse
    return out
