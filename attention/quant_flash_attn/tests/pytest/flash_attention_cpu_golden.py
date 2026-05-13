"""
Flash Attention Varlen CPU Golden (FP32 / MXFP4 统一)
=======================================================
Tiled FlashAttention v2 的 CPU golden, 仅 varlen (packed seq) 布局,
通过 `quantize` 开关切换 FP32 与 MXFP4 (Q/K/V + 可选 P) 两种模式:

    quantize=False  -> 全 FP32 (无量化), 用于校验 flash 算法骨架本身
    quantize=True   -> Q/K/V (可选 P) MXFP4 qdq, 用于推理量化建模

两种模式共享单一 kernel; MXFP4 只是在 kernel 之前 (Q/K/V) 与循环内
(P, 可选) 插入 qdq 步骤; 因此 ref vs flash 的算法一致性结论对两种
模式同样成立, 而 FP32 vs MXFP4 的差异即纯量化噪声。

两个入口:
  attention_cpu_golden_varlen        非分块 reference (flash 的对照)
  flash_attention_cpu_golden_varlen  Tiled flash v2

布局:
  q: [total_seq_q,  num_heads,    head_dim]
  k: [total_seq_kv, num_heads_kv, head_dim]
  v: [total_seq_kv, num_heads_kv, head_dim_v]
  cu_seqlens_q/kv: [batch+1] int, 存储边界. 例 [0,3,4,6,9] -> 存储 3/1/2/3
  seq_used_q/kv:   [batch] int, 有效长度 (左对齐, <= 存储长度); None = 满
  causal: K 可见于 Q 当且仅当 k <= q + (seq_used_kv - seq_used_q)
          (Q 对应 KV 末尾, 适配 decoding / prefix caching)
"""

import math
from typing import Optional

import torch

from mx_quant_fp4_tool import mx_quantize, mx_dequantize


# ============================================================
# MXFP4: 沿任意轴 qdq
# ============================================================

def _mxfp4_qdq_lastdim(t: torch.Tensor,
                       block_size: int = 32,
                       mode: str = "baseline") -> torch.Tensor:
    q, s = mx_quantize(t, block_size=block_size, mode=mode)
    return mx_dequantize(q, s, block_size=block_size)


def mxfp4_qdq(t: torch.Tensor,
              axis: int = -1,
              block_size: int = 32,
              mode: str = "baseline") -> torch.Tensor:
    """沿指定轴对张量做 MXFP4 quantize + dequantize."""
    ndim = t.dim()
    if axis < 0:
        axis += ndim
    if axis == ndim - 1:
        return _mxfp4_qdq_lastdim(t, block_size, mode)

    perm = list(range(ndim))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    t_p = t.permute(perm).contiguous()
    dq_p = _mxfp4_qdq_lastdim(t_p, block_size, mode)

    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return dq_p.permute(inv_perm).contiguous()


# ============================================================
# Varlen 公共辅助
# ============================================================

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
                f"cu_seqlens_{tag}[{i+1}]={cu_seqlens[i+1]} < "
                f"cu_seqlens_{tag}[{i}]={cu_seqlens[i]}"
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
                    f"batch {i}: seq_used_{tag}[{i}]={seq_used[i]} "
                    f"越界 [0, {storage[i]}]"
                )
    return cu_seqlens, seq_used


def _apply_input_quant(q, k, v, cu_kv, used_kv,
                       mx_block_size, mx_mode, v_quant_axis):
    """
    Packed Q/K/V 整体 MXFP4 qdq.
    V@seq_k 时 per-batch 处理, 避免 32-block 跨 batch 边界污染 scale。
    """
    q_dq = mxfp4_qdq(q, axis=-1, block_size=mx_block_size, mode=mx_mode)
    k_dq = mxfp4_qdq(k, axis=-1, block_size=mx_block_size, mode=mx_mode)
    if v_quant_axis == "head_dim":
        v_dq = mxfp4_qdq(v, axis=-1, block_size=mx_block_size, mode=mx_mode)
    elif v_quant_axis == "seq_k":
        v_dq = v.clone()
        for off, sk in zip(cu_kv, used_kv):
            if sk == 0:
                continue
            v_dq[off : off + sk] = mxfp4_qdq(
                v[off : off + sk], axis=0,
                block_size=mx_block_size, mode=mx_mode,
            )
    else:
        raise ValueError(f"unknown v_quant_axis: {v_quant_axis}")
    return q_dq, k_dq, v_dq


def _expand_kv_for_gqa_packed(k, v, num_heads_q):
    """Packed 布局 [seq, H, D]: K/V 沿 dim=1 repeat_interleave 到 num_heads_q."""
    num_heads_kv = k.shape[1]
    if num_heads_q == num_heads_kv:
        return k, v
    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"GQA: num_heads_q ({num_heads_q}) 必须能被 "
            f"num_heads_kv ({num_heads_kv}) 整除"
        )
    if v.shape[1] != num_heads_kv:
        raise ValueError(
            f"K 头数 ({num_heads_kv}) 与 V 头数 ({v.shape[1]}) 不一致"
        )
    gs = num_heads_q // num_heads_kv
    return k.repeat_interleave(gs, dim=1), v.repeat_interleave(gs, dim=1)


# ============================================================
# 单 batch tiled flash kernel (FA v2)
# ============================================================

def _flash_attn_single_batch_kernel(
    Q: torch.Tensor,        # [H, sq, D]
    K: torch.Tensor,        # [H, sk, D]
    V: torch.Tensor,        # [H, sk, D_v]
    softmax_scale: float,
    causal: bool,
    causal_offset: int,
    block_q: int,
    block_kv: int,
    quantize_p: bool,
    mx_block_size: int,
    mx_mode: str,
) -> torch.Tensor:
    """单 batch tiled FA v2; 输入已 dequant (FP32). 可选 P MXFP4 qdq."""
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

                if quantize_p:
                    P_ij = mxfp4_qdq(
                        P_ij, axis=-1,
                        block_size=mx_block_size, mode=mx_mode,
                    )

                l_i = alpha * l_i + P_ij.sum(dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i + P_ij @ V_j
                m_i = m_new

            l_safe = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
            O_i = O_i / l_safe.unsqueeze(-1)
            out[h, q_lo:q_hi] = O_i

    return out


# ============================================================
# 入口 1: 非分块 Reference (flash 的对照基线)
# ============================================================

def attention_cpu_golden_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_kv,
    seq_used_q=None,
    seq_used_kv=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    quantize: bool = False,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    v_quant_axis: str = "head_dim",
    quantize_p: bool = False,
) -> torch.Tensor:
    """
    Varlen 非分块 attention reference, 用作 flash 的对照。
    `quantize=True` 时启用 Q/K/V (可选 P) MXFP4 qdq。
    Unused slot (seq_used 到存储末尾) 填 0。
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("Varlen 期望 3D packed 输入 [total_seq, num_heads, head_dim]")

    total_q, num_heads_q, head_dim = q.shape
    num_heads_kv = k.shape[1]
    d_v = v.shape[-1]
    device = q.device

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    batch_size = (cu_seqlens_q.numel() - 1
                  if isinstance(cu_seqlens_q, torch.Tensor)
                  else len(cu_seqlens_q) - 1)
    cu_q, used_q = _normalize_cu_seqlens(cu_seqlens_q, seq_used_q, batch_size, "q")
    cu_kv, used_kv = _normalize_cu_seqlens(cu_seqlens_kv, seq_used_kv, batch_size, "kv")

    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"GQA: num_heads_q ({num_heads_q}) 必须能被 "
            f"num_heads_kv ({num_heads_kv}) 整除"
        )
    gs = num_heads_q // num_heads_kv

    out = torch.zeros(total_q, num_heads_q, d_v,
                      dtype=torch.float32, device=device)

    for b in range(batch_size):
        q_off, sq = cu_q[b], used_q[b]
        kv_off, sk = cu_kv[b], used_kv[b]
        if sq == 0 or sk == 0:
            continue

        Qb = q[q_off : q_off + sq].float()         # [sq, Hq,  D]
        Kb = k[kv_off : kv_off + sk].float()       # [sk, Hkv, D]
        Vb = v[kv_off : kv_off + sk].float()       # [sk, Hkv, D_v]

        if quantize:
            Qb = mxfp4_qdq(Qb, axis=-1,
                           block_size=mx_block_size, mode=mx_mode)
            Kb = mxfp4_qdq(Kb, axis=-1,
                           block_size=mx_block_size, mode=mx_mode)
            if v_quant_axis == "head_dim":
                Vb = mxfp4_qdq(Vb, axis=-1,
                               block_size=mx_block_size, mode=mx_mode)
            elif v_quant_axis == "seq_k":
                Vb = mxfp4_qdq(Vb, axis=0,
                               block_size=mx_block_size, mode=mx_mode)
            else:
                raise ValueError(f"unknown v_quant_axis: {v_quant_axis}")

        if gs > 1:
            Kb = Kb.repeat_interleave(gs, dim=1)
            Vb = Vb.repeat_interleave(gs, dim=1)

        Qh = Qb.transpose(0, 1)                    # [Hq, sq, D]
        Kh = Kb.transpose(0, 1)                    # [Hq, sk, D]
        Vh = Vb.transpose(0, 1)                    # [Hq, sk, D_v]

        S = (Qh @ Kh.transpose(-1, -2)) * softmax_scale     # [Hq, sq, sk]
        if causal:
            offset = sk - sq
            q_pos = torch.arange(sq, device=device).unsqueeze(1) + offset
            k_pos = torch.arange(sk, device=device).unsqueeze(0)
            S = S.masked_fill(k_pos > q_pos, float('-inf'))

        if quantize and quantize_p:
            m = S.max(dim=-1, keepdim=True).values
            m_safe = torch.where(torch.isinf(m) & (m < 0),
                                 torch.zeros_like(m), m)
            P = torch.exp(S - m_safe)
            P = torch.nan_to_num(P, nan=0.0)
            P = mxfp4_qdq(P, axis=-1,
                          block_size=mx_block_size, mode=mx_mode)
            l = P.sum(dim=-1, keepdim=True)
            l_safe = torch.where(l > 0, l, torch.ones_like(l))
            Ob = (P @ Vh) / l_safe
        else:
            P = torch.softmax(S, dim=-1)
            P = torch.nan_to_num(P, nan=0.0)
            Ob = P @ Vh

        out[q_off : q_off + sq] = Ob.transpose(0, 1)

    return out


# ============================================================
# 入口 2: Tiled Flash (主要的 golden)
# ============================================================

def flash_attention_cpu_golden_varlen(
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
    quantize: bool = False,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    v_quant_axis: str = "head_dim",
    quantize_p: bool = False,
) -> torch.Tensor:
    """
    Varlen tiled FlashAttention v2, `quantize` 开关切换 FP32 / MXFP4。

    参数:
        quantize:       False -> 全程 FP32 无量化
                        True  -> Q/K/V (可选 P) MXFP4 qdq
        当 quantize=False 时, 后面所有 mx_* / v_quant_axis / quantize_p 被忽略

        其余: 见模块 docstring。

    返回: [total_seq_q, num_heads, head_dim_v]
        valid 区域为 attention 输出, unused slot 全 0。
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"Varlen 期望 3D 输入, "
            f"实际 q.dim()={q.dim()}, k.dim()={k.dim()}, v.dim()={v.dim()}"
        )

    total_q, num_heads_q, head_dim = q.shape
    total_kv, num_heads_kv, _ = k.shape
    d_v = v.shape[-1]
    device = q.device

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    batch_size = (cu_seqlens_q.numel() - 1
                  if isinstance(cu_seqlens_q, torch.Tensor)
                  else len(cu_seqlens_q) - 1)
    cu_q, used_q = _normalize_cu_seqlens(cu_seqlens_q, seq_used_q, batch_size, "q")
    cu_kv, used_kv = _normalize_cu_seqlens(cu_seqlens_kv, seq_used_kv, batch_size, "kv")
    if cu_q[-1] > total_q:
        raise ValueError(f"cu_seqlens_q[-1]={cu_q[-1]} > total_seq_q={total_q}")
    if cu_kv[-1] > total_kv:
        raise ValueError(f"cu_seqlens_kv[-1]={cu_kv[-1]} > total_seq_kv={total_kv}")

    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"GQA: num_heads_q ({num_heads_q}) 必须能被 "
            f"num_heads_kv ({num_heads_kv}) 整除"
        )

    # ---- 可选 MXFP4 qdq (Q/K/V) ----
    if quantize:
        q_in, k_in, v_in = _apply_input_quant(
            q, k, v, cu_kv, used_kv,
            mx_block_size, mx_mode, v_quant_axis,
        )
    else:
        q_in, k_in, v_in = q, k, v

    # ---- GQA 展开 ----
    k_exp, v_exp = _expand_kv_for_gqa_packed(k_in, v_in, num_heads_q)

    # ---- 输出 buffer, unused slot 默认 0 ----
    out = torch.zeros(total_q, num_heads_q, d_v,
                      dtype=torch.float32, device=device)

    # ---- 逐 batch kernel ----
    for b in range(batch_size):
        q_off, sq = cu_q[b], used_q[b]
        kv_off, sk = cu_kv[b], used_kv[b]
        if sq == 0 or sk == 0:
            continue

        Qb = q_in[q_off : q_off + sq].transpose(0, 1).float().contiguous()
        Kb = k_exp[kv_off : kv_off + sk].transpose(0, 1).float().contiguous()
        Vb = v_exp[kv_off : kv_off + sk].transpose(0, 1).float().contiguous()

        causal_offset = (sk - sq) if causal else 0

        Ob = _flash_attn_single_batch_kernel(
            Qb, Kb, Vb,
            softmax_scale=softmax_scale,
            causal=causal,
            causal_offset=causal_offset,
            block_q=block_q,
            block_kv=block_kv,
            quantize_p=(quantize and quantize_p),
            mx_block_size=mx_block_size,
            mx_mode=mx_mode,
        )

        out[q_off : q_off + sq] = Ob.transpose(0, 1)

    return out.to(q.dtype)


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # -------- 精度比较 helpers --------
    def _pass_rate(a, b, atol, rtol):
        """逐元素 |a-b| <= atol + rtol*|b| 的通过率."""
        diff = (a - b).abs()
        tol = atol + rtol * b.abs()
        passed = int((diff <= tol).sum().item())
        total = a.numel()
        pct = passed / total * 100.0
        return passed, total, pct, diff.max().item()

    def _check(name, a, b, atol=1e-5, rtol=1e-4):
        """算法不变式: 全部元素必须满足允差; pass% 必为 100%."""
        passed, total, pct, max_diff = _pass_rate(a, b, atol, rtol)
        ok = (passed == total)
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] {name}: pass {passed}/{total} ({pct:.2f}%), "
              f"max diff {max_diff:.3e}")
        assert ok, (f"{name}: only {pct:.2f}% pass, "
                    f"max diff {max_diff:.3e} > atol={atol:.0e}")
        return max_diff

    def _check_true(name, cond):
        ok = bool(cond)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: {cond}")
        assert ok, f"{name} failed"

    def _info_cmp(name, a, b, atol=5e-2, rtol=1e-2):
        """量化噪声描述性指标 (不断言): pass@atol 百分比 + max/mean/cos."""
        passed, total, pct, max_diff = _pass_rate(a, b, atol, rtol)
        mean_err = (a - b).abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0).item()
        print(f"  [info] {name}: pass@{atol:.0e} {passed}/{total} ({pct:.2f}%), "
              f"max {max_diff:.3e}, mean {mean_err:.3e}, cos {cos:.4f}")

    print("=" * 72)
    print("Flash Attention Varlen CPU Golden 自检 (FP32 / MXFP4)")
    print("=" * 72)

    # -------- 公共测试数据 --------
    # 主 varlen 配置: 4 batch, 不等长 + 含 unused slot + KV > Q
    CU_Q    = [0, 3, 4, 6, 9]
    USED_Q  = [2, 1, 2, 3]        # batch0 sq=2(<3 含 unused), batch1=1, b2=2, b3=3
    CU_KV   = [0, 5, 7, 12, 16]
    USED_KV = [4, 1, 5, 3]        # KV 长度 >= Q (模拟 prefix cache / decoding)
    BATCH   = len(USED_Q)
    NH_Q, NH_KV, HD = 8, 2, 64    # GQA 8/2
    TOTAL_Q  = CU_Q[-1]
    TOTAL_KV = CU_KV[-1]

    Q = torch.randn(TOTAL_Q,  NH_Q,  HD)
    K = torch.randn(TOTAL_KV, NH_KV, HD)
    V = torch.randn(TOTAL_KV, NH_KV, HD)

    # ==========================================================
    # Group A: 算法骨架正确性 (ref vs flash)
    # ==========================================================
    # 测试意图:
    #   在 FP32 / MXFP4 两种模式下, tiled flash 与非分块 reference 走的是
    #   同一个数学路径, 仅相差 FP 求和顺序, 因此应严格 < 1e-5.
    #   这是 flash 算法骨架的核心不变式, 与量化噪声无关.
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group A | 算法骨架正确性 (ref vs flash)")
    print("意图: 两种模式下 tiled flash 与非分块 ref 应仅差 FP 求和顺序 (<1e-5)")
    print("=" * 72)

    print("\n--- A.1 FP32 / 非因果 ---")
    ref = attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, quantize=False)
    fl = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, quantize=False,
        block_q=32, block_kv=32)
    _check("A.1 FP32 非因果 ref vs flash", ref, fl, atol=1e-5)

    print("\n--- A.2 FP32 / 因果 ---")
    ref = attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=False)
    fl = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=False, block_q=32, block_kv=32)
    _check("A.2 FP32 因果 ref vs flash", ref, fl, atol=1e-5)

    print("\n--- A.3 MXFP4 / 非因果 ---")
    ref = attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, quantize=True)
    fl = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, quantize=True,
        block_q=32, block_kv=32)
    _check("A.3 MXFP4 非因果 ref vs flash", ref, fl, atol=1e-5)

    print("\n--- A.4 MXFP4 / 因果 ---")
    ref = attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=True)
    fl = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=True, block_q=32, block_kv=32)
    _check("A.4 MXFP4 因果 ref vs flash", ref, fl, atol=1e-5)

    print("\n--- A.5 不同 tile 大小一致性 ---")
    for mode_name, q_flag in [("FP32", False), ("MXFP4", True)]:
        print(f"  [{mode_name}]")
        base = None
        for bq, bkv in [(32, 32), (64, 32), (128, 64), (256, 256)]:
            o = flash_attention_cpu_golden_varlen(
                Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
                causal=True, quantize=q_flag, block_q=bq, block_kv=bkv)
            if base is None:
                base = o
                print(f"     block_q={bq:3d}, block_kv={bkv:3d}: (基准)")
            else:
                _check(f"     {mode_name} bq={bq} bkv={bkv} vs base",
                       o, base, atol=1e-5)

    # ==========================================================
    # Group B: GQA / MQA 适配
    # ==========================================================
    # 测试意图:
    #   num_heads_q != num_heads_kv 时, K/V 沿 head 维 repeat_interleave
    #   展开后参与计算, 输出 shape 与 Q 对齐.
    #   两种模式 (FP32 / MXFP4) 都验证 ref ≡ flash, 因为 GQA 展开发生在
    #   量化之后, 不影响算法骨架的一致性.
    #   不整除时必须 raise ValueError, 避免静默错误.
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group B | GQA / MQA 适配")
    print("意图: K/V 头数 < Q 头数时正确展开; 不整除时触发 ValueError")
    print("=" * 72)

    seq_each, batch_n = 64, 2
    cu_eq = [i * seq_each for i in range(batch_n + 1)]
    gqa_configs = [
        ("MHA(8/8)", 8, 8),
        ("GQA(8/4)", 8, 4),
        ("GQA(8/2)", 8, 2),
        ("MQA(8/1)", 8, 1),
    ]
    for tag, nh_q, nh_kv in gqa_configs:
        print(f"\n--- B.{tag} ---")
        qg = torch.randn(seq_each * batch_n, nh_q,  HD)
        kg = torch.randn(seq_each * batch_n, nh_kv, HD)
        vg = torch.randn(seq_each * batch_n, nh_kv, HD)
        for mode_name, q_flag in [("FP32", False), ("MXFP4", True)]:
            ref = attention_cpu_golden_varlen(
                qg, kg, vg, cu_eq, cu_eq, quantize=q_flag)
            fl = flash_attention_cpu_golden_varlen(
                qg, kg, vg, cu_eq, cu_eq, quantize=q_flag,
                block_q=32, block_kv=32)
            _check_true(f"  {tag} {mode_name} 输出 shape == Q shape",
                        fl.shape == qg.shape)
            _check(f"  {tag} {mode_name} ref vs flash", ref, fl, atol=1e-5)

    print("\n--- B.整除性校验 ---")
    raised = False
    try:
        flash_attention_cpu_golden_varlen(
            torch.randn(32, 7, HD),
            torch.randn(32, 4, HD),
            torch.randn(32, 4, HD),
            [0, 32], [0, 32],
        )
    except ValueError:
        raised = True
    _check_true("B 不整除 heads_q=7 vs heads_kv=4 触发 ValueError", raised)

    # ==========================================================
    # Group C: Varlen 不变式
    # ==========================================================
    # 测试意图:
    #   varlen 包含的关键不变式 — unused slot 不参与计算且输出 0;
    #   输入张量不被原地修改; 空 batch / 空 KV 不崩溃且语义正确;
    #   decoding (sq=1, sk=N) 是 prefix-cache 推理的核心场景, 必须可用.
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group C | Varlen 不变式 (unused/原地不变/空边界/decoding)")
    print("意图: unused slot 输出 0; 输入不变; 空 batch/KV 不崩溃; decoding 通过")
    print("=" * 72)

    print("\n--- C.1 unused slot 全 0 ---")
    Q_clone, K_clone, V_clone = Q.clone(), K.clone(), V.clone()
    out_main = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=True, block_q=32, block_kv=32)
    unused_max = 0.0
    for b in range(BATCH):
        valid_end = CU_Q[b] + USED_Q[b]
        slot_end = CU_Q[b+1]
        if valid_end < slot_end:
            unused_max = max(
                unused_max,
                out_main[valid_end:slot_end].abs().max().item(),
            )
    _check_true(f"C.1 unused slot max abs = {unused_max:.3e}", unused_max == 0.0)

    print("\n--- C.2 输入未被原地修改 ---")
    _check_true(
        "C.2 输入 Q/K/V 未变",
        torch.equal(Q, Q_clone)
        and torch.equal(K, K_clone)
        and torch.equal(V, V_clone),
    )

    print("\n--- C.3 空 batch (sq=0) / 空 KV (sk=0) ---")
    out_e = flash_attention_cpu_golden_varlen(
        torch.randn(5, NH_Q, HD),
        torch.randn(8, NH_KV, HD),
        torch.randn(8, NH_KV, HD),
        [0, 5, 5], [0, 4, 8], [3, 0], [4, 4],
    )
    _check_true("C.3a 空 batch (sq=0): valid 部分非零",
                (out_e[:3].abs().sum() > 0).item())
    _check_true("C.3b 空 batch (sq=0): unused 全 0",
                (out_e[3:5].abs().sum() == 0).item())

    out_ek = flash_attention_cpu_golden_varlen(
        torch.randn(5, NH_Q, HD),
        torch.randn(4, NH_KV, HD),
        torch.randn(4, NH_KV, HD),
        [0, 3, 5], [0, 4, 4], None, None,
    )
    _check_true("C.3c 空 KV (sk=0) 对应 Q 输出全 0",
                (out_ek[3:5].abs().sum() == 0).item())

    print("\n--- C.4 Decoding (sq=1, sk=N) ---")
    Q_dec = torch.randn(1,  NH_Q,  HD)
    K_dec = torch.randn(10, NH_KV, HD)
    V_dec = torch.randn(10, NH_KV, HD)
    for mode_name, q_flag in [("FP32", False), ("MXFP4", True)]:
        ref = attention_cpu_golden_varlen(
            Q_dec, K_dec, V_dec, [0, 1], [0, 10],
            causal=True, quantize=q_flag)
        fl = flash_attention_cpu_golden_varlen(
            Q_dec, K_dec, V_dec, [0, 1], [0, 10],
            causal=True, quantize=q_flag, block_q=1, block_kv=4)
        _check(f"C.4 {mode_name} decoding (sq=1, sk=10) ref vs flash",
               ref, fl, atol=1e-5)

    # ==========================================================
    # Group D: MXFP4 专属
    # ==========================================================
    # 测试意图:
    #   验证 MXFP4 模式下特有开关的语义:
    #   - V@seq_k 与 V@head_dim 都能正确走 ref vs flash 一致性
    #   - quantize_p 单 tile (bkv >= sk) 时, flash 等价于 ref (单 tile 退化)
    #   - quantize_p 多 tile 时 tile 内用 running max, 与单 tile 必有差异
    #     (这是硬件真实行为, 不是 bug; 仅描述性观察)
    #   - 量化路径的确定性 (相同输入跑两次必须 bit-identical)
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group D | MXFP4 专属 (V 轴 / P 量化 / 确定性)")
    print("意图: V@seq_k 一致性; quantize_p 单tile≡ref / 多tile 描述性; 路径确定性")
    print("=" * 72)

    print("\n--- D.1 V 量化轴 (head_dim vs seq_k) ---")
    for axis in ("head_dim", "seq_k"):
        ref = attention_cpu_golden_varlen(
            Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
            quantize=True, v_quant_axis=axis)
        fl = flash_attention_cpu_golden_varlen(
            Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
            quantize=True, v_quant_axis=axis,
            block_q=32, block_kv=32)
        _check(f"D.1 V@{axis} ref vs flash", ref, fl, atol=1e-5)

    print("\n--- D.2 P 量化单 tile (bkv >= max sk) ---")
    # max sk = 5; block_kv=64 -> 单 tile, flash 应严格等价于非分块 ref
    ref = attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        quantize=True, quantize_p=True)
    fl = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        quantize=True, quantize_p=True,
        block_q=64, block_kv=64)
    _check("D.2 quantize_p 单 tile ref vs flash", ref, fl, atol=1e-5)

    print("\n--- D.3 P 量化多 tile (描述性, 不断言) ---")
    seq_big = 128
    cu_big = [0, seq_big]
    Qb_big = torch.randn(seq_big, NH_Q,  HD)
    Kb_big = torch.randn(seq_big, NH_KV, HD)
    Vb_big = torch.randn(seq_big, NH_KV, HD)
    single = flash_attention_cpu_golden_varlen(
        Qb_big, Kb_big, Vb_big, cu_big, cu_big,
        quantize=True, quantize_p=True,
        block_q=64, block_kv=seq_big)
    for bkv in [16, 32, 64]:
        multi = flash_attention_cpu_golden_varlen(
            Qb_big, Kb_big, Vb_big, cu_big, cu_big,
            quantize=True, quantize_p=True,
            block_q=64, block_kv=bkv)
        _info_cmp(f"D.3 block_kv={bkv:3d} 多tile vs 单tile",
                  single, multi, atol=5e-2)

    print("\n--- D.4 确定性 ---")
    o1 = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=True, quantize_p=True,
        block_q=32, block_kv=32)
    o2 = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=True, quantize_p=True,
        block_q=32, block_kv=32)
    _check_true("D.4 同配置两次 bit-identical", torch.equal(o1, o2))

    # ==========================================================
    # Group E: 量化噪声 (FP32 vs MXFP4 对比, 描述性)
    # ==========================================================
    # 测试意图:
    #   这是合并脚本相对原来两个独立脚本的核心增值 — 把 quantize=False
    #   输出作为 ground truth, 直接对比 quantize=True 输出, 隔离出
    #   "纯量化噪声". pass% (在松允差下) / 平均误差 / 余弦相似度
    #   反映量化对 attention 输出的扰动幅度, 用于工程上评估精度损失.
    #   不做断言, 因为这些是数据相关的描述性指标.
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group E | 量化噪声 (FP32 vs MXFP4, 描述性, 不断言)")
    print("意图: FP32 输出作为基线, 直接量化 quantize=True 引入的扰动幅度")
    print("=" * 72)

    print("\n--- E.1 非因果 ---")
    fp32_a = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        quantize=False, block_q=32, block_kv=32)
    mx_a = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        quantize=True, block_q=32, block_kv=32)
    _info_cmp("E.1 FP32 vs MXFP4 非因果", fp32_a, mx_a, atol=5e-2)

    print("\n--- E.2 因果 + GQA(8/2) ---")
    fp32_b = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=False, block_q=32, block_kv=32)
    mx_b = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=True, block_q=32, block_kv=32)
    _info_cmp("E.2 FP32 vs MXFP4 因果+GQA(8/2)", fp32_b, mx_b, atol=5e-2)

    print("\n--- E.3 Decoding (sq=1, sk=10) ---")
    fp32_c = flash_attention_cpu_golden_varlen(
        Q_dec, K_dec, V_dec, [0, 1], [0, 10],
        causal=True, quantize=False, block_q=1, block_kv=4)
    mx_c = flash_attention_cpu_golden_varlen(
        Q_dec, K_dec, V_dec, [0, 1], [0, 10],
        causal=True, quantize=True, block_q=1, block_kv=4)
    _info_cmp("E.3 FP32 vs MXFP4 decoding", fp32_c, mx_c, atol=5e-2)

    print("\n--- E.4 P 量化 + V@seq_k 组合 ---")
    mx_d = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV,
        causal=True, quantize=True, quantize_p=True,
        v_quant_axis="seq_k", block_q=32, block_kv=32)
    _info_cmp("E.4 FP32 vs MXFP4(P+V@seq_k) 因果",
              fp32_b, mx_d, atol=1e-1)

    print("\n" + "=" * 72)
    print("自检完成")
    print("=" * 72)