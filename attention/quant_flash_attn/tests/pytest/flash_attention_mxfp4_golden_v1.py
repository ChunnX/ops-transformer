"""
Flash Attention CPU Golden Reference with MXFP4 Quantized Q/K/V
=================================================================
推理时 Q/K/V 全部用 MXFP4 量化的 FlashAttention v2 数值参考实现 (CPU)。

设计要点:
  1. 量化轴:
        Q 沿 head_dim 量化  (Q @ K^T 的收缩轴)
        K 沿 head_dim 量化  (Q @ K^T 的收缩轴)
        V 默认沿 head_dim 量化 (per-token, 适合 KV cache 增量更新);
          也可选 v_quant_axis="seq_k" 让 V 沿 seq_k 量化
          (P @ V 收缩轴, 数值上更贴近 MX 规范但需要全量重量化)
  2. 量化-反量化 (qdq) 一次性整体做完, 然后在 dequantized FP32 上
     按 FlashAttention v2 的 tiled online-softmax 流程累加。
     这是 hardware 的"高精度等价"参考: dequant 后用 FP32 做累加,
     给出"假如硬件无累加误差"的理想结果。
  3. 输出累加器 / softmax 状态全部 FP32, 与 vendor 内核做 bit-level
     对比时, 差异主要源自 FP32 求和顺序 + softmax 数值实现。

接口形状: [..., seq, head_dim], 任意 batch/head 前缀维度。
"""

import math
from typing import Optional

import torch

from mx_quant_fp4_tool import mx_quantize, mx_dequantize
from quant_flash_attn_golden import check_result, rand_range


# ============================================================
# 工具: 沿任意轴做 MXFP4 qdq
# ============================================================

def _mxfp4_qdq_lastdim(t: torch.Tensor,
                      block_size: int = 32,
                      mode: str = "baseline") -> torch.Tensor:
    """沿最后一维做 MXFP4 quantize + dequantize。"""
    q, s = mx_quantize(t, block_size=block_size, mode=mode)
    return mx_dequantize(q, s, block_size=block_size)


def mxfp4_qdq(t: torch.Tensor,
              axis: int = -1,
              block_size: int = 32,
              mode: str = "baseline") -> torch.Tensor:
    """
    沿指定轴对张量做 MXFP4 quantize+dequantize。
    实现方式: 将目标轴 permute 到最后一维, 复用 last-dim 量化函数, 再 permute 回去。
    """
    ndim = t.dim()
    if axis < 0:
        axis += ndim
    if axis == ndim - 1:
        return _mxfp4_qdq_lastdim(t, block_size, mode)

    perm = list(range(ndim))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    t_p = t.permute(perm).contiguous()
    dq_p = _mxfp4_qdq_lastdim(t_p, block_size, mode)

    # 逆 permute
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return dq_p.permute(inv_perm).contiguous()


# ============================================================
# 参考实现 (非分块): MXFP4 qdq + 标准 attention
# ============================================================

def attention_mxfp4_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    v_quant_axis: str = "head_dim",
) -> torch.Tensor:
    """
    标准 (非分块) attention, 仅在 Q/K/V 上做 MXFP4 qdq。
    用于校验分块 flash 版本: 两者输出应仅相差 FP 求和顺序导致的微小误差。
    """
    d = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    q_dq = mxfp4_qdq(q, axis=-1, block_size=mx_block_size, mode=mx_mode)
    k_dq = mxfp4_qdq(k, axis=-1, block_size=mx_block_size, mode=mx_mode)

    if v_quant_axis == "head_dim":
        v_dq = mxfp4_qdq(v, axis=-1, block_size=mx_block_size, mode=mx_mode)
    elif v_quant_axis == "seq_k":
        v_dq = mxfp4_qdq(v, axis=-2, block_size=mx_block_size, mode=mx_mode)
    else:
        raise ValueError(f"unknown v_quant_axis: {v_quant_axis}")

    scores = (q_dq @ k_dq.transpose(-1, -2)) * softmax_scale
    if causal:
        seq_q, seq_k = scores.shape[-2:]
        mask = torch.triu(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    # 全行被 mask 的情况下 softmax 会产生 NaN, 视情形清零
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs @ v_dq


# ============================================================
# Flash Attention v2 (tiled) golden
# ============================================================

def flash_attention_mxfp4_cpu_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 64,
    block_kv: int = 64,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    v_quant_axis: str = "head_dim",
) -> torch.Tensor:
    """
    FlashAttention v2 算法的 CPU golden, Q/K/V 均 MXFP4 量化。

    参数:
        q, k, v:        [..., seq, head_dim], 任意 batch/head 前缀维, FP32
        softmax_scale:  默认 1/sqrt(head_dim)
        causal:         是否使用上三角因果 mask
        block_q:        Q tile 大小 (沿 seq_q 切)
        block_kv:       K/V tile 大小 (沿 seq_k 切)
        mx_block_size:  MX 分组大小 (规范固定为 32)
        mx_mode:        scale 计算策略 ("baseline" 或 "oas")
        v_quant_axis:   V 量化轴, "head_dim" (默认, 适配 KV cache) 或 "seq_k"

    返回:
        attention 输出, shape 同 Q
    """
    *batch_dims, seq_q, d = q.shape
    seq_k = k.shape[-2]
    d_v = v.shape[-1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    # ---- Step 1: 整体 MXFP4 qdq ----
    q_dq = mxfp4_qdq(q, axis=-1, block_size=mx_block_size, mode=mx_mode)
    k_dq = mxfp4_qdq(k, axis=-1, block_size=mx_block_size, mode=mx_mode)
    if v_quant_axis == "head_dim":
        v_dq = mxfp4_qdq(v, axis=-1, block_size=mx_block_size, mode=mx_mode)
    elif v_quant_axis == "seq_k":
        v_dq = mxfp4_qdq(v, axis=-2, block_size=mx_block_size, mode=mx_mode)
    else:
        raise ValueError(f"unknown v_quant_axis: {v_quant_axis}")

    # ---- Step 2: flatten 所有 batch 维 ----
    Q = q_dq.reshape(-1, seq_q, d).float()
    K = k_dq.reshape(-1, seq_k, d).float()
    V = v_dq.reshape(-1, seq_k, d_v).float()
    B = Q.shape[0]

    out = torch.zeros(B, seq_q, d_v, dtype=torch.float32, device=q.device)

    num_q_tiles = (seq_q + block_q - 1) // block_q
    num_kv_tiles = (seq_k + block_kv - 1) // block_kv
    NEG_INF = float('-inf')

    # ---- Step 3: tiled flash attention v2 ----
    for b in range(B):
        Qb, Kb, Vb = Q[b], K[b], V[b]

        for i in range(num_q_tiles):
            q_lo = i * block_q
            q_hi = min(q_lo + block_q, seq_q)
            Q_i = Qb[q_lo:q_hi]                       # [bq, d]
            bq = q_hi - q_lo

            # FlashAttention v2 状态: 每行的 running max / sum / 输出累加
            m_i = torch.full((bq,), NEG_INF, dtype=torch.float32, device=q.device)
            l_i = torch.zeros(bq, dtype=torch.float32, device=q.device)
            O_i = torch.zeros(bq, d_v, dtype=torch.float32, device=q.device)

            for j in range(num_kv_tiles):
                k_lo = j * block_kv
                k_hi = min(k_lo + block_kv, seq_k)

                # 因果剪枝: 整 tile 都在对角线之上, 提前结束
                if causal and k_lo > q_hi - 1:
                    break

                K_j = Kb[k_lo:k_hi]                   # [bkv, d]
                V_j = Vb[k_lo:k_hi]                   # [bkv, d_v]

                # (1) QK^T scaled
                S_ij = (Q_i @ K_j.t()) * softmax_scale   # [bq, bkv]

                # (2) 因果 mask
                if causal:
                    q_pos = torch.arange(q_lo, q_hi, device=q.device).unsqueeze(1)
                    k_pos = torch.arange(k_lo, k_hi, device=q.device).unsqueeze(0)
                    S_ij = S_ij.masked_fill(k_pos > q_pos, NEG_INF)

                # (3) online softmax 更新
                m_ij = S_ij.max(dim=-1).values            # [bq]
                m_new = torch.maximum(m_i, m_ij)

                # 整行被 mask 时 m_new = -inf, 用 0 替换避免 NaN 传播
                # (此时该行的 alpha 和 P_ij 都会是 0, 累加器保持为 0)
                m_safe = torch.where(
                    torch.isinf(m_new) & (m_new < 0),
                    torch.zeros_like(m_new), m_new,
                )

                # alpha = exp(m_old - m_new): 旧累加器的修正系数
                # 首次迭代 m_i=-inf, alpha = exp(-inf - 0) = 0, 自然清零旧累加
                alpha = torch.exp(m_i - m_safe)

                # P_ij = exp(S - m_new)
                P_ij = torch.exp(S_ij - m_safe.unsqueeze(-1))    # [bq, bkv]

                # 累加更新
                l_i = alpha * l_i + P_ij.sum(dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i + P_ij @ V_j
                m_i = m_new

            # tile 完成: 用 sum 归一化 (整行被 mask 的行 l_i=0, 输出保持 0)
            l_safe = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
            O_i = O_i / l_safe.unsqueeze(-1)

            out[b, q_lo:q_hi] = O_i

    return out.reshape(*batch_dims, seq_q, d_v).to(q.dtype)


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 64)
    print("Flash Attention MXFP4 CPU Golden 自检")
    print("=" * 64)

    # ---------- 配置 ----------
    batch, heads, seq, d = 2, 4, 128, 64
    # # q = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    # k = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    # v = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    q = rand_range((batch, heads, seq, d), data_range=[-10, 10], dtype = torch.float32)
    k = rand_range((batch, heads, seq, d), data_range=[-10, 10], dtype = torch.float32)
    v = rand_range((batch, heads, seq, d), data_range=[-10, 10], dtype = torch.float32)
    scale = 1.0 / math.sqrt(d)

    # ---------- FP32 标准 attention (无量化基线) ----------
    scores_fp32 = q @ k.transpose(-1, -2) * scale
    out_fp32 = torch.softmax(scores_fp32, dim=-1) @ v

    # ---------- 测试 1: 非因果 ----------
    print("\n--- 测试 1: 非因果 attention ---")
    out_ref = attention_mxfp4_reference(q, k, v)
    out_flash = flash_attention_mxfp4_cpu_golden(
        q, k, v, block_q=64, block_kv=32,
    )
    print("----------out_fp32 vs out_ref------------------------")
    check_result(out_fp32.to(torch.float32), out_ref.cpu().to(torch.float32), 'fp4_e2m1')
    print("----------out_fp32 vs out_flash------------------------")
    check_result(out_fp32.to(torch.float32), out_flash.cpu().to(torch.float32), 'fp4_e2m1')
    print("----------out_ref vs out_flash------------------------")
    check_result(out_ref.to(torch.float32), out_flash.cpu().to(torch.float32), 'fp4_e2m1')

    err_fp32_vs_ref = (out_fp32 - out_ref).abs().mean().item()
    err_fp32_vs_flash = (out_fp32 - out_flash).abs().mean().item()
    err_ref_vs_flash = (out_ref - out_flash).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out_fp32.flatten(), out_flash.flatten(), dim=0
    ).item()
    print(f"  FP32 vs MXFP4-ref     平均误差: {err_fp32_vs_ref:.6f}")
    print(f"  FP32 vs MXFP4-flash   平均误差: {err_fp32_vs_flash:.6f}")
    print(f"  ref vs flash          最大误差: {err_ref_vs_flash:.3e}  (~0 = 算法一致)")
    print(f"  FP32 vs flash         余弦相似度: {cos:.6f}")

    # ---------- 测试 2: 因果 ----------
    print("\n--- 测试 2: 因果 attention (decoder) ---")
    scores_c = scores_fp32.masked_fill(
        torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1),
        float('-inf'),
    )
    out_fp32_c = torch.softmax(scores_c, dim=-1) @ v

    out_ref_c = attention_mxfp4_reference(q, k, v, causal=True)
    out_flash_c = flash_attention_mxfp4_cpu_golden(
        q, k, v, causal=True, block_q=32, block_kv=32,
    )
    err_ref_vs_flash_c = (out_ref_c - out_flash_c).abs().max().item()
    err_fp32_vs_flash_c = (out_fp32_c - out_flash_c).abs().mean().item()
    cos_c = torch.nn.functional.cosine_similarity(
        out_fp32_c.flatten(), out_flash_c.flatten(), dim=0
    ).item()
    print(f"  FP32 vs MXFP4-flash   平均误差: {err_fp32_vs_flash_c:.6f}")
    print(f"  ref vs flash          最大误差: {err_ref_vs_flash_c:.3e}  (~0 = 算法一致)")
    print(f"  FP32 vs flash         余弦相似度: {cos_c:.6f}")

    # ---------- 测试 3: V 沿 seq_k 量化 ----------
    print("\n--- 测试 3: V 沿 seq_k 量化 (MX 规范方案) ---")
    out_flash_v_seqk = flash_attention_mxfp4_cpu_golden(
        q, k, v, v_quant_axis="seq_k",
    )
    out_ref_v_seqk = attention_mxfp4_reference(
        q, k, v, v_quant_axis="seq_k",
    )
    diff = (out_flash_v_seqk - out_ref_v_seqk).abs().max().item()
    err_fp32 = (out_fp32 - out_flash_v_seqk).abs().mean().item()
    print(f"  V quant axis=seq_k:   FP32 平均误差: {err_fp32:.6f}")
    print(f"                        ref vs flash 最大误差: {diff:.3e}")

    # ---------- 测试 4: 解码场景 (seq_q=1, seq_k=N) ----------
    print("\n--- 测试 4: 解码场景 (seq_q=1, KV cache seq_k=256) ---")
    q1 = torch.randn(1, 4, 1, d)
    k_cache = torch.randn(1, 4, 256, d)
    v_cache = torch.randn(1, 4, 256, d)

    scores_d = q1 @ k_cache.transpose(-1, -2) * scale
    out_d_fp32 = torch.softmax(scores_d, dim=-1) @ v_cache

    out_d_flash = flash_attention_mxfp4_cpu_golden(
        q1, k_cache, v_cache, block_q=1, block_kv=64,
    )
    err_d = (out_d_fp32 - out_d_flash).abs().mean().item()
    cos_d = torch.nn.functional.cosine_similarity(
        out_d_fp32.flatten(), out_d_flash.flatten(), dim=0
    ).item()
    print(f"  输出 shape: {tuple(out_d_flash.shape)}")
    print(f"  FP32 平均误差: {err_d:.6f}")
    print(f"  余弦相似度:    {cos_d:.6f}")

    # ---------- 测试 5: 不同 tile 大小一致性 ----------
    print("\n--- 测试 5: 不同 tile 大小输出应一致 ---")
    cfgs = [(32, 32), (64, 32), (128, 64), (seq, seq)]
    base = None
    for bq, bkv in cfgs:
        o = flash_attention_mxfp4_cpu_golden(
            q, k, v, causal=True, block_q=bq, block_kv=bkv,
        )
        if base is None:
            base = o
            print(f"  block_q={bq:3d}, block_kv={bkv:3d}: (基准)")
        else:
            d_max = (o - base).abs().max().item()
            print(f"  block_q={bq:3d}, block_kv={bkv:3d}: 与基准最大差异 {d_max:.3e}")

    print("\n" + "=" * 64)
    print("自检完成")
    print("=" * 64)