"""
Flash Attention CPU Golden Reference (FP32, 无量化)
======================================================
作为 MXFP4 版本的"控制组": 同样的算法骨架, 但 Q/K/V/P 全程保持 FP32。

特性 (与 flash_attention_mxfp4_golden.py 对齐):
  - MHA / GQA / MQA
  - Causal mask, 支持 Q 对应 KV 末尾的 offset (decoding / prefix caching)
  - Tiled flash attention v2 (online softmax)
  - 两套入口:
      Padded: [batch, num_heads, seq, head_dim]
      Varlen: [total_seq, num_heads, head_dim] + cu_seqlens + seq_used (左对齐)

用途:
  1. 单独验证 flash 算法本身的正确性 (ref vs flash 应 < 1e-5, 只差 FP 求和顺序)
  2. 给 MXFP4 版本提供"无量化"对比基线 — 把两者的输出做差, 即可隔离出量化误差
"""

import math
from typing import Optional

import torch


# ============================================================
# 通用辅助
# ============================================================

def _expand_kv_for_gqa(q: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor,
                       head_dim: int = -3) -> tuple:
    """
    GQA/MQA 适配: K/V 头数若少于 Q, 沿指定 head 维 repeat_interleave。

    head_dim:
        -3  适用 [B, H, S, D] padded 布局
        -2  适用 [total_seq, H, D] packed 布局
    """
    if q.dim() < abs(head_dim) or k.dim() < abs(head_dim):
        return k, v
    nh_q = q.shape[head_dim]
    nh_kv = k.shape[head_dim]
    if nh_q == nh_kv:
        return k, v
    if nh_q % nh_kv != 0:
        raise ValueError(
            f"GQA: num_heads_q ({nh_q}) 必须能被 num_heads_kv ({nh_kv}) 整除"
        )
    if v.shape[head_dim] != nh_kv:
        raise ValueError(
            f"K 头数 ({nh_kv}) 与 V 头数 ({v.shape[head_dim]}) 不一致"
        )
    gs = nh_q // nh_kv
    return (k.repeat_interleave(gs, dim=head_dim),
            v.repeat_interleave(gs, dim=head_dim))


def _normalize_cu_seqlens(cu_seqlens, seq_used, batch_size, tag=""):
    """Validate/convert cu_seqlens + seq_used -> int lists."""
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = cu_seqlens.tolist()
    cu_seqlens = [int(x) for x in cu_seqlens]
    if len(cu_seqlens) != batch_size + 1:
        raise ValueError(
            f"cu_seqlens_{tag} 长度应为 batch+1={batch_size+1}, 实际 {len(cu_seqlens)}"
        )
    storage = [cu_seqlens[i+1] - cu_seqlens[i] for i in range(batch_size)]
    for i, s in enumerate(storage):
        if s < 0:
            raise ValueError(
                f"cu_seqlens_{tag}[{i+1}]={cu_seqlens[i+1]} < cu_seqlens_{tag}[{i}]={cu_seqlens[i]}"
            )

    if seq_used is None:
        seq_used = storage
    else:
        if isinstance(seq_used, torch.Tensor):
            seq_used = seq_used.tolist()
        seq_used = [int(x) for x in seq_used]
        if len(seq_used) != batch_size:
            raise ValueError(
                f"seq_used_{tag} 长度应为 batch={batch_size}, 实际 {len(seq_used)}"
            )
        for i in range(batch_size):
            if seq_used[i] < 0 or seq_used[i] > storage[i]:
                raise ValueError(
                    f"batch {i}: seq_used_{tag}[{i}]={seq_used[i]} 越界 [0, {storage[i]}]"
                )
    return cu_seqlens, seq_used


def _flash_attn_kernel_fp32(
    Q: torch.Tensor,        # [H, sq, D]
    K: torch.Tensor,        # [H, sk, D]
    V: torch.Tensor,        # [H, sk, D_v]
    softmax_scale: float,
    causal: bool,
    causal_offset: int,     # K 可见于 Q 当 k <= q + causal_offset
    block_q: int,
    block_kv: int,
) -> torch.Tensor:
    """单 batch 的 tiled flash attention v2 内核 (FP32 全程, 无量化)。"""
    H, sq, _ = Q.shape
    sk = K.shape[1]
    d_v = V.shape[-1]
    device = Q.device
    NEG_INF = float('-inf')

    out = torch.zeros(H, sq, d_v, dtype=torch.float32, device=device)
    num_q_tiles = (sq + block_q - 1) // block_q
    num_kv_tiles = (sk + block_kv - 1) // block_kv

    for h in range(H):
        Qh, Kh, Vh = Q[h], K[h], V[h]

        for i in range(num_q_tiles):
            q_lo = i * block_q
            q_hi = min(q_lo + block_q, sq)
            Q_i = Qh[q_lo:q_hi]
            bq = q_hi - q_lo

            m_i = torch.full((bq,), NEG_INF, dtype=torch.float32, device=device)
            l_i = torch.zeros(bq, dtype=torch.float32, device=device)
            O_i = torch.zeros(bq, d_v, dtype=torch.float32, device=device)

            for j in range(num_kv_tiles):
                k_lo = j * block_kv
                k_hi = min(k_lo + block_kv, sk)

                if causal and k_lo > (q_hi - 1) + causal_offset:
                    break

                K_j = Kh[k_lo:k_hi]
                V_j = Vh[k_lo:k_hi]

                S_ij = (Q_i @ K_j.t()) * softmax_scale

                if causal:
                    q_pos = (torch.arange(q_lo, q_hi, device=device)
                             .unsqueeze(1) + causal_offset)
                    k_pos = torch.arange(k_lo, k_hi, device=device).unsqueeze(0)
                    S_ij = S_ij.masked_fill(k_pos > q_pos, NEG_INF)

                m_ij = S_ij.max(dim=-1).values
                m_new = torch.maximum(m_i, m_ij)
                m_safe = torch.where(
                    torch.isinf(m_new) & (m_new < 0),
                    torch.zeros_like(m_new), m_new,
                )

                alpha = torch.exp(m_i - m_safe)
                P_ij = torch.exp(S_ij - m_safe.unsqueeze(-1))

                l_i = alpha * l_i + P_ij.sum(dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i + P_ij @ V_j
                m_i = m_new

            l_safe = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
            O_i = O_i / l_safe.unsqueeze(-1)
            out[h, q_lo:q_hi] = O_i

    return out


# ============================================================
# Padded 布局: [batch, num_heads, seq, head_dim]
# ============================================================

def attention_fp32_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    标准 (非分块) FP32 attention, 支持 MHA/GQA/MQA。
    输入形状: [..., num_heads, seq, head_dim]
    """
    d = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    k_exp, v_exp = _expand_kv_for_gqa(q, k, v, head_dim=-3)

    scores = (q @ k_exp.transpose(-1, -2)) * softmax_scale
    if causal:
        seq_q, seq_k = scores.shape[-2:]
        mask = torch.triu(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs @ v_exp


def flash_attention_fp32_cpu_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 64,
    block_kv: int = 64,
) -> torch.Tensor:
    """
    Tiled FlashAttention v2 (FP32, 无量化), 支持 MHA/GQA/MQA。
    输入形状: [..., num_heads, seq, head_dim], 任意 batch 前缀。
    """
    *batch_dims, seq_q, d = q.shape
    seq_k = k.shape[-2]
    d_v = v.shape[-1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    k_exp, v_exp = _expand_kv_for_gqa(q, k, v, head_dim=-3)

    # Flatten batch & head 维 -> [B*H, seq, D]
    Q = q.reshape(-1, seq_q, d).float()
    K = k_exp.reshape(-1, seq_k, d).float()
    V = v_exp.reshape(-1, seq_k, d_v).float()
    B = Q.shape[0]

    out = torch.zeros(B, seq_q, d_v, dtype=torch.float32, device=q.device)
    num_q_tiles = (seq_q + block_q - 1) // block_q
    num_kv_tiles = (seq_k + block_kv - 1) // block_kv
    NEG_INF = float('-inf')

    for b in range(B):
        Qb, Kb, Vb = Q[b], K[b], V[b]

        for i in range(num_q_tiles):
            q_lo = i * block_q
            q_hi = min(q_lo + block_q, seq_q)
            Q_i = Qb[q_lo:q_hi]
            bq = q_hi - q_lo

            m_i = torch.full((bq,), NEG_INF, dtype=torch.float32, device=q.device)
            l_i = torch.zeros(bq, dtype=torch.float32, device=q.device)
            O_i = torch.zeros(bq, d_v, dtype=torch.float32, device=q.device)

            for j in range(num_kv_tiles):
                k_lo = j * block_kv
                k_hi = min(k_lo + block_kv, seq_k)

                if causal and k_lo > q_hi - 1:
                    break

                K_j = Kb[k_lo:k_hi]
                V_j = Vb[k_lo:k_hi]

                S_ij = (Q_i @ K_j.t()) * softmax_scale

                if causal:
                    q_pos = torch.arange(q_lo, q_hi, device=q.device).unsqueeze(1)
                    k_pos = torch.arange(k_lo, k_hi, device=q.device).unsqueeze(0)
                    S_ij = S_ij.masked_fill(k_pos > q_pos, NEG_INF)

                m_ij = S_ij.max(dim=-1).values
                m_new = torch.maximum(m_i, m_ij)
                m_safe = torch.where(
                    torch.isinf(m_new) & (m_new < 0),
                    torch.zeros_like(m_new), m_new,
                )

                alpha = torch.exp(m_i - m_safe)
                P_ij = torch.exp(S_ij - m_safe.unsqueeze(-1))

                l_i = alpha * l_i + P_ij.sum(dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i + P_ij @ V_j
                m_i = m_new

            l_safe = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
            O_i = O_i / l_safe.unsqueeze(-1)
            out[b, q_lo:q_hi] = O_i

    return out.reshape(*batch_dims, seq_q, d_v).to(q.dtype)


# ============================================================
# Varlen 布局: [total_seq, num_heads, head_dim]
# ============================================================

def attention_fp32_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_kv,
    seq_used_q=None,
    seq_used_kv=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Varlen 非分块参考 (FP32): 逐 batch 切片做标准 attention。
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("Varlen 期望 3D packed 输入 [total_seq, num_heads, head_dim]")

    total_q, num_heads_q, head_dim = q.shape
    num_heads_kv = k.shape[1]
    d_v = v.shape[-1]
    device = q.device

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if isinstance(cu_seqlens_q, torch.Tensor):
        batch_size = cu_seqlens_q.numel() - 1
    else:
        batch_size = len(cu_seqlens_q) - 1
    cu_q, used_q = _normalize_cu_seqlens(cu_seqlens_q, seq_used_q, batch_size, "q")
    cu_kv, used_kv = _normalize_cu_seqlens(cu_seqlens_kv, seq_used_kv, batch_size, "kv")

    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"GQA: num_heads_q ({num_heads_q}) 必须能被 num_heads_kv ({num_heads_kv}) 整除"
        )
    group_size = num_heads_q // num_heads_kv
    out = torch.zeros(total_q, num_heads_q, d_v, dtype=torch.float32, device=device)

    for b in range(batch_size):
        q_off, sq = cu_q[b], used_q[b]
        kv_off, sk = cu_kv[b], used_kv[b]
        if sq == 0 or sk == 0:
            continue

        Qb = q[q_off : q_off + sq].float()         # [sq, Hq, D]
        Kb = k[kv_off : kv_off + sk].float()       # [sk, Hkv, D]
        Vb = v[kv_off : kv_off + sk].float()       # [sk, Hkv, D_v]

        if group_size > 1:
            Kb = Kb.repeat_interleave(group_size, dim=1)
            Vb = Vb.repeat_interleave(group_size, dim=1)

        # [seq, H, D] -> [H, seq, D]
        Qh = Qb.transpose(0, 1)
        Kh = Kb.transpose(0, 1)
        Vh = Vb.transpose(0, 1)

        S = (Qh @ Kh.transpose(-1, -2)) * softmax_scale         # [H, sq, sk]
        if causal:
            offset = sk - sq
            q_pos = torch.arange(sq, device=device).unsqueeze(1) + offset
            k_pos = torch.arange(sk, device=device).unsqueeze(0)
            S = S.masked_fill(k_pos > q_pos, float('-inf'))

        P = torch.softmax(S, dim=-1)
        P = torch.nan_to_num(P, nan=0.0)
        Ob = P @ Vh

        out[q_off : q_off + sq] = Ob.transpose(0, 1)

    return out


def flash_attention_fp32_cpu_golden_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_kv,
    seq_used_q=None,
    seq_used_kv=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 64,
    block_kv: int = 64,
) -> torch.Tensor:
    """
    Varlen tiled FlashAttention v2 (FP32, 无量化)。

    参数:
        q:              [total_seq_q,  num_heads,    head_dim]   packed
        k:              [total_seq_kv, num_heads_kv, head_dim]   packed
        v:              [total_seq_kv, num_heads_kv, head_dim_v] packed
        cu_seqlens_q:   [batch+1] 存储边界
        cu_seqlens_kv:  [batch+1] 存储边界
        seq_used_q:     [batch] 有效长度 (<= 存储), None 表示满
        seq_used_kv:    [batch] 同上
        causal:         True 时 K 可见于 Q 当且仅当 k <= q + (seq_used_kv-seq_used_q)
                        (Q 对应 KV 末尾, 适配 decoding/prefix caching)

    返回: [total_seq_q, num_heads, head_dim_v]
        - valid 区域是 attention 输出
        - unused slot (seq_used 到存储末尾) 填 0
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"Varlen 期望 3D 输入, 实际 q.dim()={q.dim()}, k.dim()={k.dim()}, v.dim()={v.dim()}"
        )

    total_q, num_heads_q, head_dim = q.shape
    total_kv, num_heads_kv, _ = k.shape
    d_v = v.shape[-1]
    device = q.device

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if isinstance(cu_seqlens_q, torch.Tensor):
        batch_size = cu_seqlens_q.numel() - 1
    else:
        batch_size = len(cu_seqlens_q) - 1
    cu_q, used_q = _normalize_cu_seqlens(cu_seqlens_q, seq_used_q, batch_size, "q")
    cu_kv, used_kv = _normalize_cu_seqlens(cu_seqlens_kv, seq_used_kv, batch_size, "kv")
    if cu_q[-1] > total_q:
        raise ValueError(f"cu_seqlens_q[-1]={cu_q[-1]} > total_seq_q={total_q}")
    if cu_kv[-1] > total_kv:
        raise ValueError(f"cu_seqlens_kv[-1]={cu_kv[-1]} > total_seq_kv={total_kv}")

    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"GQA: num_heads_q ({num_heads_q}) 必须能被 num_heads_kv ({num_heads_kv}) 整除"
        )
    group_size = num_heads_q // num_heads_kv

    # GQA 展开 (packed 布局 head 维在 dim=1)
    k_exp = k.repeat_interleave(group_size, dim=1) if group_size > 1 else k
    v_exp = v.repeat_interleave(group_size, dim=1) if group_size > 1 else v

    out = torch.zeros(total_q, num_heads_q, d_v, dtype=torch.float32, device=device)

    for b in range(batch_size):
        q_off, sq = cu_q[b], used_q[b]
        kv_off, sk = cu_kv[b], used_kv[b]

        if sq == 0 or sk == 0:
            continue

        # [seq, H, D] -> [H, seq, D]
        Qb = q[q_off : q_off + sq].transpose(0, 1).float().contiguous()
        Kb = k_exp[kv_off : kv_off + sk].transpose(0, 1).float().contiguous()
        Vb = v_exp[kv_off : kv_off + sk].transpose(0, 1).float().contiguous()

        causal_offset = (sk - sq) if causal else 0

        Ob = _flash_attn_kernel_fp32(
            Qb, Kb, Vb,
            softmax_scale=softmax_scale,
            causal=causal,
            causal_offset=causal_offset,
            block_q=block_q,
            block_kv=block_kv,
        )  # [H, sq, D_v]

        out[q_off : q_off + sq] = Ob.transpose(0, 1)

    return out.to(q.dtype)


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 64)
    print("Flash Attention FP32 Golden 自检 (无量化)")
    print("=" * 64)

    def _check(name, a, b, atol=1e-5, rtol=1e-4):
        """对算法不变式做 allclose 断言, 同时打印量化误差."""
        diff = (a - b).abs().max().item()
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: max diff {diff:.3e}")
        assert ok, f"{name}: max diff {diff:.3e} > atol={atol:.0e}"
        return diff

    def _check_true(name, cond):
        """对布尔不变式做断言."""
        ok = bool(cond)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: {cond}")
        assert ok, f"{name} failed"

    # 公共数据
    batch, heads, seq, d = 2, 4, 128, 64
    q = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    v = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)

    # PyTorch 内置作为 ground truth
    scores_pt = q @ k.transpose(-1, -2) * scale
    out_pt = torch.softmax(scores_pt, dim=-1) @ v

    # ---------- 测试 1: 非因果, ref 应严格等于 PyTorch 内置 ----------
    print("\n--- 测试 1: 非因果 ---")
    out_ref = attention_fp32_reference(q, k, v)
    out_flash = flash_attention_fp32_cpu_golden(q, k, v, block_q=64, block_kv=32)
    _check("PT vs ref",    out_pt, out_ref,   atol=1e-6)   # 同一算法, 应 ~0
    _check("PT vs flash",  out_pt, out_flash, atol=1e-5)
    _check("ref vs flash", out_ref, out_flash, atol=1e-5)

    # ---------- 测试 2: 因果 ----------
    print("\n--- 测试 2: 因果 ---")
    out_ref_c = attention_fp32_reference(q, k, v, causal=True)
    out_flash_c = flash_attention_fp32_cpu_golden(
        q, k, v, causal=True, block_q=32, block_kv=32,
    )
    _check("ref vs flash (causal)", out_ref_c, out_flash_c, atol=1e-5)

    # ---------- 测试 3: 不同 tile 大小一致性 ----------
    print("\n--- 测试 3: 不同 tile 大小输出一致性 ---")
    cfgs = [(32, 32), (64, 32), (128, 64), (seq, seq)]
    base = None
    for bq, bkv in cfgs:
        o = flash_attention_fp32_cpu_golden(
            q, k, v, causal=True, block_q=bq, block_kv=bkv,
        )
        if base is None:
            base = o
            print(f"  block_q={bq:3d}, block_kv={bkv:3d}: (基准)")
        else:
            _check(f"block_q={bq}, block_kv={bkv} vs base", o, base, atol=1e-5)

    # ---------- 测试 4: 解码场景 (sq=1, sk=256) ----------
    print("\n--- 测试 4: 解码场景 (sq=1, sk=256) ---")
    q1 = torch.randn(1, 4, 1, d)
    k_cache = torch.randn(1, 4, 256, d)
    v_cache = torch.randn(1, 4, 256, d)
    o_pt = torch.softmax(q1 @ k_cache.transpose(-1, -2) * scale, dim=-1) @ v_cache
    o_fl = flash_attention_fp32_cpu_golden(
        q1, k_cache, v_cache, block_q=1, block_kv=64,
    )
    _check_true(f"输出 shape == Q shape ({tuple(o_fl.shape)})", o_fl.shape == q1.shape)
    _check("PT vs flash (decoding)", o_pt, o_fl, atol=1e-5)

    # ---------- 测试 5: GQA / MQA ----------
    print("\n--- 测试 5: GQA / MQA ---")
    head_dim = 64
    seq_len = 128
    configs = [
        ("MHA(8/8)", 8, 8),
        ("GQA(8/4)", 8, 4),
        ("GQA(8/2)", 8, 2),
        ("MQA(8/1)", 8, 1),
    ]
    for name, nh_q, nh_kv in configs:
        qg = torch.randn(2, nh_q,  seq_len, head_dim)
        kg = torch.randn(2, nh_kv, seq_len, head_dim)
        vg = torch.randn(2, nh_kv, seq_len, head_dim)
        out_ref_g = attention_fp32_reference(qg, kg, vg)
        out_flash_g = flash_attention_fp32_cpu_golden(
            qg, kg, vg, block_q=64, block_kv=32,
        )
        _check_true(f"{name} shape == Q shape", out_flash_g.shape == qg.shape)
        _check(f"{name} ref vs flash", out_ref_g, out_flash_g, atol=1e-5)

    # ---------- 测试 6: Varlen (packed seq + cu_seqlens + seq_used) ----------
    print("\n--- 测试 6: Varlen (cu_seqlens + seq_used) ---")
    cu_q_t  = [0, 3, 4, 6, 9]
    used_q_t = [2, 1, 2, 3]
    cu_kv_t  = [0, 5, 7, 12, 16]
    used_kv_t = [4, 1, 5, 3]
    total_q_t = cu_q_t[-1]
    total_kv_t = cu_kv_t[-1]
    nh_q, nh_kv, hd = 8, 2, 64

    q_pack = torch.randn(total_q_t, nh_q, hd) * 10.0
    k_pack = torch.randn(total_kv_t, nh_kv, hd) * 10.0
    v_pack = torch.randn(total_kv_t, nh_kv, hd) * 10.0
    q_orig, k_orig, v_orig = q_pack.clone(), k_pack.clone(), v_pack.clone()

    out_v = flash_attention_fp32_cpu_golden_varlen(
        q_pack, k_pack, v_pack,
        cu_q_t, cu_kv_t, used_q_t, used_kv_t,
        causal=True, block_q=32, block_kv=32,
    )
    out_v_ref = attention_fp32_varlen_reference(
        q_pack, k_pack, v_pack,
        cu_q_t, cu_kv_t, used_q_t, used_kv_t,
        causal=True,
    )

    _check_true(f"varlen 输出 shape == Q shape ({tuple(out_v.shape)})",
                out_v.shape == q_pack.shape)
    _check("varlen flash vs ref", out_v, out_v_ref, atol=1e-5)

    # unused slot 全 0
    unused_ok = True
    for b in range(len(used_q_t)):
        slot_end = cu_q_t[b+1]
        valid_end = cu_q_t[b] + used_q_t[b]
        if valid_end < slot_end:
            if out_v[valid_end:slot_end].abs().max().item() > 0:
                unused_ok = False
    _check_true("unused slot 全 0", unused_ok)

    _check_true(
        "输入未被原地修改",
        torch.equal(q_pack, q_orig) and torch.equal(k_pack, k_orig) and torch.equal(v_pack, v_orig),
    )

    # 逐 batch PT 对比
    print(f"  逐 batch PT 对比:")
    for b in range(len(used_q_t)):
        q_off, sq = cu_q_t[b], used_q_t[b]
        kv_off, sk = cu_kv_t[b], used_kv_t[b]
        Qb_fp = q_pack[q_off : q_off + sq].transpose(0, 1)
        Kb_fp = k_pack[kv_off : kv_off + sk].transpose(0, 1)
        Vb_fp = v_pack[kv_off : kv_off + sk].transpose(0, 1)
        gs = nh_q // nh_kv
        if gs > 1:
            Kb_fp = Kb_fp.repeat_interleave(gs, dim=0)
            Vb_fp = Vb_fp.repeat_interleave(gs, dim=0)
        S_fp = Qb_fp @ Kb_fp.transpose(-1, -2) / math.sqrt(hd)
        offset = sk - sq
        q_pos = torch.arange(sq).unsqueeze(1) + offset
        k_pos = torch.arange(sk).unsqueeze(0)
        S_fp = S_fp.masked_fill(k_pos > q_pos, float('-inf'))
        Ob_fp = torch.softmax(S_fp, dim=-1) @ Vb_fp
        out_b = out_v[q_off : q_off + sq].transpose(0, 1)
        _check(f"  batch {b} (sq={sq}, sk={sk}, offset={offset}) PT vs flash",
               out_b, Ob_fp, atol=1e-5)

    # ---------- 测试 7: padded API ≡ varlen API ----------
    print("\n--- 测试 7: padded API ≡ varlen API (退化等长批) ---")
    seq_each = 64
    B_test = 2
    q_pad = torch.randn(B_test, nh_q,  seq_each, hd)
    k_pad = torch.randn(B_test, nh_kv, seq_each, hd)
    v_pad = torch.randn(B_test, nh_kv, seq_each, hd)
    out_pad = flash_attention_fp32_cpu_golden(
        q_pad, k_pad, v_pad, causal=True, block_q=32, block_kv=32,
    )

    def _to_packed(t):
        B, H, S, D = t.shape
        return t.transpose(1, 2).reshape(B * S, H, D)

    q_packed = _to_packed(q_pad)
    k_packed = _to_packed(k_pad)
    v_packed = _to_packed(v_pad)
    cu_q_eq = [i * seq_each for i in range(B_test + 1)]
    cu_kv_eq = cu_q_eq
    out_var = flash_attention_fp32_cpu_golden_varlen(
        q_packed, k_packed, v_packed,
        cu_q_eq, cu_kv_eq, causal=True,
        block_q=32, block_kv=32,
    )
    _check("padded API ≡ varlen API", _to_packed(out_pad), out_var, atol=1e-5)

    # ---------- 测试 8: 退化空 batch & 空 KV ----------
    print("\n--- 测试 8: 边界 (空 batch / 空 KV) ---")
    out_e = flash_attention_fp32_cpu_golden_varlen(
        torch.randn(5, nh_q, hd), torch.randn(8, nh_kv, hd), torch.randn(8, nh_kv, hd),
        [0, 5, 5], [0, 4, 8], [3, 0], [4, 4],
    )
    _check_true("空 batch: valid 部分非零", (out_e[:3].abs().sum() > 0).item())
    _check_true("空 batch: unused 全 0",   (out_e[3:5].abs().sum() == 0).item())

    out_ek = flash_attention_fp32_cpu_golden_varlen(
        torch.randn(5, nh_q, hd), torch.randn(4, nh_kv, hd), torch.randn(4, nh_kv, hd),
        [0, 3, 5], [0, 4, 4], None, None,
    )
    _check_true("空 KV (sk=0) -> 对应 Q 输出全 0", (out_ek[3:5].abs().sum() == 0).item())

    print("\n" + "=" * 64)
    print("自检完成")
    print("=" * 64)