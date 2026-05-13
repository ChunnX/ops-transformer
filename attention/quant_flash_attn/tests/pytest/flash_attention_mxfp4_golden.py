"""
Flash Attention CPU Golden Reference with MXFP4 Quantized Q/K/V
=================================================================
推理时 Q/K/V 全部用 MXFP4 量化的 FlashAttention v2 数值参考实现 (CPU)。
支持 MHA / GQA / MQA。

两套入口:
  flash_attention_mxfp4_cpu_golden        - Padded 布局 [B, H, S, D]
                                            适合等长批 / 静态 shape 调试
  flash_attention_mxfp4_cpu_golden_varlen - Packed 布局 [total_seq, H, D]
                                            + cu_seqlens (存储边界)
                                            + seq_used   (有效长度, 可 < 存储)
                                            适合真实推理 (continuous batching,
                                            KV cache, decoding 时 sq < sk)

设计要点:
  0. 张量布局: [..., num_heads, seq, head_dim], head 维位于倒数第 3 位。
        MHA: num_heads_q == num_heads_kv
        GQA: num_heads_q  > num_heads_kv (且整除), 每 num_heads_q/num_heads_kv
             个 Q 头共享一个 KV 头
        MQA: num_heads_kv == 1
     K/V 在原本的 num_kv_heads 维度上做 MXFP4 量化 (量化只算一次),
     再 repeat_interleave 沿 head 维展开到 num_heads_q 后参与计算。
  1. 量化轴:
        Q 沿 head_dim 量化  (Q @ K^T 的收缩轴)
        K 沿 head_dim 量化  (Q @ K^T 的收缩轴)
        V 默认沿 head_dim 量化 (per-token, 适合 KV cache 增量更新);
          也可选 v_quant_axis="seq_k" 让 V 沿 seq_k 量化
          (P @ V 收缩轴, 数值上更贴近 MX 规范但需要全量重量化)
  2. Q/K/V 的 qdq 一次性整体做完; P (softmax 输出) 可选在 tile 内
     沿 bkv 轴量化为 MXFP4 (quantize_p=True), 模拟 P@V 的 MX 矩阵乘。
     P 量化的语义说明:
       - 量化对象是 P_ij = exp(S_ij - m_new) (per-tile 未归一化 exp)
       - 量化后的 P 同时用于 l_i 累加和 O_i 累加, 保证最终 O/l 一致
       - 因为每个 tile 用的是 running max (m_new) 而非全局 max,
         不同 block_kv 切分下结果会有差异 (这就是硬件真实行为)
       - block_kv >= seq_k (单 tile) 时, 等价于非分块 reference
  3. 输出累加器 / softmax 状态全部 FP32, 与 vendor 内核做 bit-level
     对比时, 差异主要源自 FP32 求和顺序 + softmax 数值实现。

接口形状: [..., seq, head_dim], 任意 batch/head 前缀维度。
"""

import math
from typing import Optional

import torch

from mx_quant_fp4_tool import mx_quantize, mx_dequantize


# ============================================================
# 工具: 沿任意轴做 MXFP4 qdq
# ============================================================

def _mxfp4_qdq_lastdim(t: torch.Tensor,
                      block_size: int = 32,
                      mode: str = "baseline") -> torch.Tensor:
    """沿最后一维做 MXFP4 quantize + dequantize。"""
    q, s = mx_quantize(t, block_size=block_size, mode=mode)
    return mx_dequantize(q, s, block_size=block_size)


def _expand_kv_for_gqa(q: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor) -> tuple:
    """
    GQA/MQA 适配: K/V 头数若少于 Q 头数, 沿 head 维 (倒数第 3 维) repeat_interleave。

    约定输入布局: [..., num_heads, seq, head_dim], 头维位于 -3。
      MHA: num_heads_q == num_heads_kv, 不展开
      GQA: num_heads_q > num_heads_kv, 每个 KV 头被 group_size 个 Q 头共享
      MQA: num_heads_kv = 1, GQA 的特例

    返回 (k_expanded, v_expanded), 头数与 Q 对齐。
    """
    if q.dim() < 3 or k.dim() < 3:
        return k, v
    num_heads_q = q.shape[-3]
    num_heads_kv = k.shape[-3]
    if num_heads_q == num_heads_kv:
        return k, v
    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"GQA: num_heads_q ({num_heads_q}) 必须能被 "
            f"num_heads_kv ({num_heads_kv}) 整除"
        )
    if v.shape[-3] != num_heads_kv:
        raise ValueError(
            f"K 头数 ({num_heads_kv}) 与 V 头数 ({v.shape[-3]}) 不一致"
        )
    group_size = num_heads_q // num_heads_kv
    k = k.repeat_interleave(group_size, dim=-3)
    v = v.repeat_interleave(group_size, dim=-3)
    return k, v


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
    quantize_p: bool = False,
) -> torch.Tensor:
    """
    标准 (非分块) attention, Q/K/V 做 MXFP4 qdq; 可选 P 也做 MXFP4 qdq。

    P 量化的实现 (等价于 flash 在 block_kv >= seq_k 时的单 tile 情形):
      手工做 softmax (减最大值 -> exp -> 量化未归一化 P -> sum 归一化),
      量化对象是 exp(S - row_max), 而非已归一化的 softmax 输出。
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

    # GQA/MQA: K/V 量化后再沿 head 维展开到 Q 的头数 (量化只做一次)
    k_dq, v_dq = _expand_kv_for_gqa(q_dq, k_dq, v_dq)

    scores = (q_dq @ k_dq.transpose(-1, -2)) * softmax_scale
    if causal:
        seq_q, seq_k = scores.shape[-2:]
        mask = torch.triu(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float('-inf'))

    if not quantize_p:
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        return probs @ v_dq

    # 手工 softmax 以便在归一化前插入 P 量化
    m = scores.max(dim=-1, keepdim=True).values
    m_safe = torch.where(torch.isinf(m) & (m < 0), torch.zeros_like(m), m)
    P = torch.exp(scores - m_safe)                       # 未归一化 exp
    P = torch.nan_to_num(P, nan=0.0)                     # 全 -inf 行 -> 0

    # 沿 seq_k (最后一维) 做 MXFP4 qdq
    P = mxfp4_qdq(P, axis=-1, block_size=mx_block_size, mode=mx_mode)

    l = P.sum(dim=-1, keepdim=True)
    l_safe = torch.where(l > 0, l, torch.ones_like(l))
    return (P @ v_dq) / l_safe


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
    quantize_p: bool = False,
) -> torch.Tensor:
    """
    FlashAttention v2 算法的 CPU golden, Q/K/V 均 MXFP4 量化, 可选 P 量化。

    参数:
        q, k, v:        [..., seq, head_dim], 任意 batch/head 前缀维, FP32
        softmax_scale:  默认 1/sqrt(head_dim)
        causal:         是否使用上三角因果 mask
        block_q:        Q tile 大小 (沿 seq_q 切)
        block_kv:       K/V tile 大小 (沿 seq_k 切, 建议为 mx_block_size 倍数)
        mx_block_size:  MX 分组大小 (规范固定为 32)
        mx_mode:        scale 计算策略 ("baseline" 或 "oas")
        v_quant_axis:   V 量化轴, "head_dim" (默认, 适配 KV cache) 或 "seq_k"
        quantize_p:     是否在每个 tile 内对 P_ij 沿 bkv 量化为 MXFP4
                        (模拟 P @ V 的 MX 矩阵乘 P 端量化)

    返回:
        attention 输出, shape 同 Q
    """
    *batch_dims, seq_q, d = q.shape
    seq_k = k.shape[-2]
    d_v = v.shape[-1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    # ---- Step 1: 整体 MXFP4 qdq (K/V 在原本 num_kv_heads 维度上量化, 只算一次) ----
    q_dq = mxfp4_qdq(q, axis=-1, block_size=mx_block_size, mode=mx_mode)
    k_dq = mxfp4_qdq(k, axis=-1, block_size=mx_block_size, mode=mx_mode)
    if v_quant_axis == "head_dim":
        v_dq = mxfp4_qdq(v, axis=-1, block_size=mx_block_size, mode=mx_mode)
    elif v_quant_axis == "seq_k":
        v_dq = mxfp4_qdq(v, axis=-2, block_size=mx_block_size, mode=mx_mode)
    else:
        raise ValueError(f"unknown v_quant_axis: {v_quant_axis}")

    # ---- Step 2: GQA/MQA 展开 K/V 头数到 num_heads_q ----
    k_dq, v_dq = _expand_kv_for_gqa(q_dq, k_dq, v_dq)

    # ---- Step 3: flatten 所有 batch 维 (包含已对齐的 head 维) ----
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

                # 可选: 对 P_ij 沿 bkv 做 MXFP4 qdq, 模拟 P@V 的 MX 矩阵乘
                # 量化后的 P 同时用于 l_i 和 O_i, 保证最终 O/l 一致
                if quantize_p:
                    P_ij = mxfp4_qdq(
                        P_ij, axis=-1,
                        block_size=mx_block_size, mode=mx_mode,
                    )

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
# Varlen (packed) Flash Attention: cu_seqlens + seq_used
# ============================================================
# 布局约定:
#   Q/K/V 形状 [total_seq, num_heads, head_dim] (沿 seq 维 packed)
#   cu_seqlens_q/kv: [batch+1] int, 存储边界. 例 [0,3,4,6,9] -> 4 个 batch,
#                    存储长度 3/1/2/3
#   seq_used_q/kv:   [batch] int, 每 batch 有效长度 (<= 存储长度)
#                    None 表示存储长度全部有效
#   对齐: 左对齐 (有效数据在 slot 起始)
#   Causal: Q 对应 KV 末尾, k 可见于 q 当且仅当 k <= q + (seq_used_kv - seq_used_q)
#   MXFP4: head_dim 轴量化 packed tensor 整体; seq_k 轴量化 per-batch (不跨 batch)
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


def _flash_attn_single_batch_kernel(
    Q: torch.Tensor,        # [H, sq, D]
    K: torch.Tensor,        # [H, sk, D]
    V: torch.Tensor,        # [H, sk, D_v]
    softmax_scale: float,
    causal: bool,
    causal_offset: int,     # K 可见于 Q 当 k <= q + causal_offset
    block_q: int,
    block_kv: int,
    quantize_p: bool,
    mx_block_size: int,
    mx_mode: str,
) -> torch.Tensor:
    """单 batch 的 tiled flash attention v2 内核, 输入已是 FP32 dequant。"""
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

                # 因果剪枝 (考虑 causal_offset)
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


def flash_attention_mxfp4_cpu_golden_varlen(
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
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    v_quant_axis: str = "head_dim",
    quantize_p: bool = False,
) -> torch.Tensor:
    """
    Varlen FlashAttention v2 CPU golden, MXFP4 Q/K/V (可选 P) 量化。

    参数:
        q:              [total_seq_q,  num_heads,    head_dim]  packed FP32
        k:              [total_seq_kv, num_heads_kv, head_dim]  packed FP32
        v:              [total_seq_kv, num_heads_kv, head_dim_v] packed FP32
        cu_seqlens_q:   [batch+1] int. e.g. [0,3,4,6,9] -> 存储长度 [3,1,2,3]
        cu_seqlens_kv:  [batch+1] int. 同上
        seq_used_q:     [batch] int, 每 batch Q 的有效长度. None 表示满
        seq_used_kv:    [batch] int, 每 batch KV 的有效长度. None 表示满
        其余参数同 flash_attention_mxfp4_cpu_golden

    返回: [total_seq_q, num_heads, head_dim_v] FP32
           - valid 范围 [cu_seqlens_q[i], cu_seqlens_q[i] + seq_used_q[i]) 是 attention 结果
           - 其余 (unused slot) 填 0
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"Varlen 期望 3D 输入 [total_seq, num_heads, head_dim], "
            f"实际 q.dim()={q.dim()}, k.dim()={k.dim()}, v.dim()={v.dim()}"
        )

    total_q, num_heads_q, head_dim = q.shape
    total_kv, num_heads_kv, _ = k.shape
    d_v = v.shape[-1]
    device = q.device

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # 解析 cu_seqlens / seq_used
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

    # GQA 校验
    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"GQA: num_heads_q ({num_heads_q}) 必须能被 num_heads_kv ({num_heads_kv}) 整除"
        )
    group_size = num_heads_q // num_heads_kv

    # ---- MXFP4 量化 ----
    # Q, K 沿 head_dim: 整体 packed 一次性 qdq (每 token 独立, 不跨 batch)
    q_dq = mxfp4_qdq(q, axis=-1, block_size=mx_block_size, mode=mx_mode)
    k_dq = mxfp4_qdq(k, axis=-1, block_size=mx_block_size, mode=mx_mode)

    if v_quant_axis == "head_dim":
        v_dq = mxfp4_qdq(v, axis=-1, block_size=mx_block_size, mode=mx_mode)
    elif v_quant_axis == "seq_k":
        # Per-batch 量化, 避免 32-block 跨 batch 边界污染 scale
        # unused slot 区域不参与量化, 保留原值 (反正后续 attention 不会读)
        v_dq = v.clone()
        for b in range(batch_size):
            kv_off, sk = cu_kv[b], used_kv[b]
            if sk == 0:
                continue
            v_slice = v[kv_off : kv_off + sk]                 # [sk, Hkv, D_v]
            # axis=0 即 slice 的 seq_k 维
            v_dq[kv_off : kv_off + sk] = mxfp4_qdq(
                v_slice, axis=0,
                block_size=mx_block_size, mode=mx_mode,
            )
    else:
        raise ValueError(f"unknown v_quant_axis: {v_quant_axis}")

    # GQA 展开: packed 布局 head 维位于 dim=1
    if group_size > 1:
        k_dq = k_dq.repeat_interleave(group_size, dim=1)
        v_dq = v_dq.repeat_interleave(group_size, dim=1)

    # ---- 输出 buffer, unused slot 默认 0 ----
    out = torch.zeros(total_q, num_heads_q, d_v, dtype=torch.float32, device=device)

    # ---- 逐 batch kernel ----
    for b in range(batch_size):
        q_off, sq = cu_q[b], used_q[b]
        kv_off, sk = cu_kv[b], used_kv[b]

        if sq == 0:
            continue
        if sk == 0:
            # 无 key 可 attend, 输出保持 0
            continue

        # [seq, H, D] -> [H, seq, D]
        Qb = q_dq[q_off : q_off + sq].transpose(0, 1).float().contiguous()
        Kb = k_dq[kv_off : kv_off + sk].transpose(0, 1).float().contiguous()
        Vb = v_dq[kv_off : kv_off + sk].transpose(0, 1).float().contiguous()

        causal_offset = (sk - sq) if causal else 0

        Ob = _flash_attn_single_batch_kernel(
            Qb, Kb, Vb,
            softmax_scale=softmax_scale,
            causal=causal,
            causal_offset=causal_offset,
            block_q=block_q,
            block_kv=block_kv,
            quantize_p=quantize_p,
            mx_block_size=mx_block_size,
            mx_mode=mx_mode,
        )  # [H, sq, D_v]

        out[q_off : q_off + sq] = Ob.transpose(0, 1)

    return out.to(q.dtype)


def attention_mxfp4_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_kv,
    seq_used_q=None,
    seq_used_kv=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    v_quant_axis: str = "head_dim",
    quantize_p: bool = False,
) -> torch.Tensor:
    """
    Varlen 非分块参考: 逐 batch 切片, MXFP4 qdq, 标准 attention。
    用于校验 flash varlen 版本: 两者应仅相差 FP 求和顺序。
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("Varlen reference 期望 3D packed 输入")

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

    group_size = num_heads_q // num_heads_kv
    out = torch.zeros(total_q, num_heads_q, d_v, dtype=torch.float32, device=device)

    for b in range(batch_size):
        q_off, sq = cu_q[b], used_q[b]
        kv_off, sk = cu_kv[b], used_kv[b]
        if sq == 0 or sk == 0:
            continue

        Qb = q[q_off : q_off + sq].float()        # [sq, Hq, D]
        Kb = k[kv_off : kv_off + sk].float()      # [sk, Hkv, D]
        Vb = v[kv_off : kv_off + sk].float()      # [sk, Hkv, D_v]

        Qb_dq = mxfp4_qdq(Qb, axis=-1, block_size=mx_block_size, mode=mx_mode)
        Kb_dq = mxfp4_qdq(Kb, axis=-1, block_size=mx_block_size, mode=mx_mode)
        if v_quant_axis == "head_dim":
            Vb_dq = mxfp4_qdq(Vb, axis=-1, block_size=mx_block_size, mode=mx_mode)
        elif v_quant_axis == "seq_k":
            Vb_dq = mxfp4_qdq(Vb, axis=0, block_size=mx_block_size, mode=mx_mode)
        else:
            raise ValueError(f"unknown v_quant_axis: {v_quant_axis}")

        if group_size > 1:
            Kb_dq = Kb_dq.repeat_interleave(group_size, dim=1)
            Vb_dq = Vb_dq.repeat_interleave(group_size, dim=1)

        # [seq, H, D] -> [H, seq, D]
        Qh = Qb_dq.transpose(0, 1)
        Kh = Kb_dq.transpose(0, 1)
        Vh = Vb_dq.transpose(0, 1)

        S = (Qh @ Kh.transpose(-1, -2)) * softmax_scale       # [H, sq, sk]
        if causal:
            offset = sk - sq
            q_pos = torch.arange(sq, device=device).unsqueeze(1) + offset
            k_pos = torch.arange(sk, device=device).unsqueeze(0)
            S = S.masked_fill(k_pos > q_pos, float('-inf'))

        if not quantize_p:
            P = torch.softmax(S, dim=-1)
            P = torch.nan_to_num(P, nan=0.0)
            Ob = P @ Vh
        else:
            m = S.max(dim=-1, keepdim=True).values
            m_safe = torch.where(torch.isinf(m) & (m < 0), torch.zeros_like(m), m)
            P = torch.exp(S - m_safe)
            P = torch.nan_to_num(P, nan=0.0)
            P = mxfp4_qdq(P, axis=-1, block_size=mx_block_size, mode=mx_mode)
            l = P.sum(dim=-1, keepdim=True)
            l_safe = torch.where(l > 0, l, torch.ones_like(l))
            Ob = (P @ Vh) / l_safe

        out[q_off : q_off + sq] = Ob.transpose(0, 1)

    return out


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
    q = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    v = torch.randn(batch, heads, seq, d, dtype=torch.float32)
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

    # ---------- 测试 5: 不同 tile 大小一致性 (不量化 P) ----------
    print("\n--- 测试 5: 不同 tile 大小输出应一致 (quantize_p=False) ---")
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

    # ---------- 测试 6: P 也量化为 MXFP4 ----------
    print("\n--- 测试 6: P 沿 seq_k 量化为 MXFP4 ---")
    # 单 tile (block_kv >= seq_k): flash 应该和 reference 严格匹配
    out_ref_qp = attention_mxfp4_reference(
        q, k, v, quantize_p=True,
    )
    out_flash_qp_single = flash_attention_mxfp4_cpu_golden(
        q, k, v, block_q=seq, block_kv=seq, quantize_p=True,
    )
    diff_single = (out_ref_qp - out_flash_qp_single).abs().max().item()
    err_fp32_qp = (out_fp32 - out_flash_qp_single).abs().mean().item()
    cos_qp = torch.nn.functional.cosine_similarity(
        out_fp32.flatten(), out_flash_qp_single.flatten(), dim=0,
    ).item()
    print(f"  单 tile (bkv=seq_k):")
    print(f"    ref vs flash 最大差异: {diff_single:.3e}  (应 ~0)")
    print(f"    FP32 vs flash 平均误差: {err_fp32_qp:.6f}")
    print(f"    FP32 vs flash 余弦相似度: {cos_qp:.6f}")

    # 多 tile: 每 tile 用 running max 量化, 与 reference (全局 max) 有差异
    # 这就是硬件实际行为, 不是 bug
    print(f"  多 tile (展示 tile-quant 的真实行为):")
    for bkv in [32, 64, 128]:
        out_qp = flash_attention_mxfp4_cpu_golden(
            q, k, v, block_q=64, block_kv=bkv, quantize_p=True,
        )
        d_ref = (out_ref_qp - out_qp).abs().max().item()
        d_fp32 = (out_fp32 - out_qp).abs().mean().item()
        print(f"    block_kv={bkv:3d}: 与单tile差异 {d_ref:.3e}, "
              f"FP32 平均误差 {d_fp32:.6f}")

    # 多 tile 内部一致性: 同样的 (bq, bkv) 跑两次应完全相同 (确定性检查)
    o1 = flash_attention_mxfp4_cpu_golden(
        q, k, v, causal=True, block_q=64, block_kv=32, quantize_p=True,
    )
    o2 = flash_attention_mxfp4_cpu_golden(
        q, k, v, causal=True, block_q=64, block_kv=32, quantize_p=True,
    )
    print(f"  确定性检查 (同配置跑两次): 差异 {(o1-o2).abs().max().item():.3e}")

    # ---------- 测试 7: GQA / MQA 适配 ----------
    print("\n--- 测试 7: GQA / MQA 适配 ---")

    def _ref_fp32_gqa(q, k, v, causal=False):
        """FP32 reference, 手动展开 K/V 头数."""
        nh_q = q.shape[-3]
        nh_kv = k.shape[-3]
        if nh_q != nh_kv:
            gs = nh_q // nh_kv
            k = k.repeat_interleave(gs, dim=-3)
            v = v.repeat_interleave(gs, dim=-3)
        s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if causal:
            sq, sk = s.shape[-2:]
            mask = torch.triu(torch.ones(sq, sk, dtype=torch.bool), diagonal=1)
            s = s.masked_fill(mask, float('-inf'))
        return torch.softmax(s, dim=-1) @ v

    head_dim = 64
    seq_len = 128
    configs = [
        ("MHA  (heads_q=8,  heads_kv=8)", 8, 8),
        ("GQA  (heads_q=8,  heads_kv=4)", 8, 4),
        ("GQA  (heads_q=8,  heads_kv=2)", 8, 2),
        ("MQA  (heads_q=8,  heads_kv=1)", 8, 1),
    ]
    for name, nh_q, nh_kv in configs:
        qg = torch.randn(2, nh_q,  seq_len, head_dim, dtype=torch.float32)
        kg = torch.randn(2, nh_kv, seq_len, head_dim, dtype=torch.float32)
        vg = torch.randn(2, nh_kv, seq_len, head_dim, dtype=torch.float32)

        out_fp32_g = _ref_fp32_gqa(qg, kg, vg)
        out_ref_g = attention_mxfp4_reference(qg, kg, vg)
        out_flash_g = flash_attention_mxfp4_cpu_golden(
            qg, kg, vg, block_q=64, block_kv=32,
        )

        # 形状检查
        assert out_flash_g.shape == qg.shape, \
            f"输出 shape {out_flash_g.shape} != Q shape {qg.shape}"

        err_fp32 = (out_fp32_g - out_flash_g).abs().mean().item()
        err_ref_vs_flash = (out_ref_g - out_flash_g).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            out_fp32_g.flatten(), out_flash_g.flatten(), dim=0,
        ).item()
        print(f"  {name}: 输出 {tuple(out_flash_g.shape)}")
        print(f"      FP32 平均误差 {err_fp32:.6f}, "
              f"ref vs flash {err_ref_vs_flash:.3e}, "
              f"余弦相似度 {cos:.6f}")

    # 错误用例: heads_q 不能被 heads_kv 整除
    try:
        bad_q = torch.randn(1, 7, 32, 64)
        bad_k = torch.randn(1, 4, 32, 64)
        bad_v = torch.randn(1, 4, 32, 64)
        flash_attention_mxfp4_cpu_golden(bad_q, bad_k, bad_v)
        print("  [错误] 7/4 整除检查未触发")
    except ValueError as e:
        print(f"  整除检查 OK: {e}")

    # GQA + causal + quantize_p 综合
    nh_q, nh_kv = 8, 2
    qg = torch.randn(1, nh_q,  seq_len, head_dim)
    kg = torch.randn(1, nh_kv, seq_len, head_dim)
    vg = torch.randn(1, nh_kv, seq_len, head_dim)
    out_combo = flash_attention_mxfp4_cpu_golden(
        qg, kg, vg, causal=True, block_q=32, block_kv=32,
        quantize_p=True, v_quant_axis="seq_k",
    )
    out_combo_fp32 = _ref_fp32_gqa(qg, kg, vg, causal=True)
    err = (out_combo_fp32 - out_combo).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        out_combo_fp32.flatten(), out_combo.flatten(), dim=0,
    ).item()
    print(f"  GQA(8/2) + causal + quantize_p + V@seq_k: "
          f"FP32 误差 {err:.6f}, 余弦 {cos:.6f}")

    # ---------- 测试 8: Varlen packed 布局 ----------
    print("\n--- 测试 8: Varlen (packed seq, cu_seqlens + seq_used) ---")

    # 用户例子: cu_seqlens_q = [0,3,4,6,9], 即 4 个 batch, 存储长度 3/1/2/3
    # seq_used 让部分 slot 不满, 验证 unused 区域被正确忽略
    cu_q_test  = [0, 3, 4, 6, 9]
    used_q_test = [2, 1, 2, 3]            # batch 0 只用 2/3, batch 2 满, 等等
    # KV 比 Q 长 (模拟 prefix cache / decoding)
    cu_kv_test  = [0, 5, 7, 12, 16]
    used_kv_test = [4, 1, 5, 3]

    total_q_test = cu_q_test[-1]
    total_kv_test = cu_kv_test[-1]
    n_heads_q, n_heads_kv, hd = 8, 2, 64

    # 包含 unused slot 的 garbage (用大幅值 randn 模拟, 应被忽略)
    q_pack = torch.randn(total_q_test, n_heads_q, hd) * 10.0
    k_pack = torch.randn(total_kv_test, n_heads_kv, hd) * 10.0
    v_pack = torch.randn(total_kv_test, n_heads_kv, hd) * 10.0
    # 复制一份用于"valid 区域不变"对比
    q_orig = q_pack.clone()
    k_orig = k_pack.clone()
    v_orig = v_pack.clone()

    out_varlen = flash_attention_mxfp4_cpu_golden_varlen(
        q_pack, k_pack, v_pack,
        cu_q_test, cu_kv_test,
        used_q_test, used_kv_test,
        causal=True,
        block_q=32, block_kv=32,
    )
    out_ref_varlen = attention_mxfp4_varlen_reference(
        q_pack, k_pack, v_pack,
        cu_q_test, cu_kv_test,
        used_q_test, used_kv_test,
        causal=True,
    )

    # (a) Shape 校验
    assert out_varlen.shape == q_pack.shape, \
        f"输出 shape {out_varlen.shape} != Q shape {q_pack.shape}"
    print(f"  输出 shape: {tuple(out_varlen.shape)}  (= Q shape)")

    # (b) flash vs reference: 应仅相差 FP 求和顺序
    diff_max = (out_varlen - out_ref_varlen).abs().max().item()
    print(f"  flash vs reference 最大差异: {diff_max:.3e}")

    # (c) unused slot 必须为 0
    unused_ok = True
    for b in range(len(used_q_test)):
        q_off = cu_q_test[b]
        sq = used_q_test[b]
        slot_end = cu_q_test[b+1]
        if sq < slot_end - q_off:
            unused = out_varlen[q_off + sq : slot_end]
            if unused.abs().max().item() > 0:
                unused_ok = False
                print(f"  batch {b}: unused slot 非零! max={unused.abs().max():.3e}")
    print(f"  unused slot 全 0: {unused_ok}")

    # (d) 输入未被原地修改
    inputs_unchanged = (
        torch.equal(q_pack, q_orig)
        and torch.equal(k_pack, k_orig)
        and torch.equal(v_pack, v_orig)
    )
    print(f"  输入未被修改: {inputs_unchanged}")

    # (e) 对每个 batch 单独算 FP32 attention, 验证 valid 区域语义正确
    print(f"  逐 batch FP32 对比:")
    for b in range(len(used_q_test)):
        q_off, sq = cu_q_test[b], used_q_test[b]
        kv_off, sk = cu_kv_test[b], used_kv_test[b]
        Qb_fp = q_pack[q_off : q_off + sq].transpose(0, 1)        # [Hq, sq, D]
        Kb_fp = k_pack[kv_off : kv_off + sk].transpose(0, 1)      # [Hkv, sk, D]
        Vb_fp = v_pack[kv_off : kv_off + sk].transpose(0, 1)
        # GQA expand
        gs = n_heads_q // n_heads_kv
        if gs > 1:
            Kb_fp = Kb_fp.repeat_interleave(gs, dim=0)
            Vb_fp = Vb_fp.repeat_interleave(gs, dim=0)
        S_fp = Qb_fp @ Kb_fp.transpose(-1, -2) / math.sqrt(hd)
        offset = sk - sq
        q_pos = torch.arange(sq).unsqueeze(1) + offset
        k_pos = torch.arange(sk).unsqueeze(0)
        S_fp = S_fp.masked_fill(k_pos > q_pos, float('-inf'))
        Ob_fp = torch.softmax(S_fp, dim=-1) @ Vb_fp                # [Hq, sq, D]
        out_b = out_varlen[q_off : q_off + sq].transpose(0, 1)
        err = (out_b - Ob_fp).abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(
            Ob_fp.flatten(), out_b.flatten(), dim=0,
        ).item()
        print(f"    batch {b} (sq={sq}, sk={sk}, offset={offset}): "
              f"FP32 误差 {err:.4f}, 余弦 {cos:.4f}")

    # (f) Varlen 退化到等长批: 应与原 flash_attention_mxfp4_cpu_golden 一致
    print(f"  Varlen 退化等长批 (batch=2, seq=64) -> 与 padded API 对比:")
    seq_each = 64
    B_test = 2
    q_pad = torch.randn(B_test, n_heads_q, seq_each, hd)
    k_pad = torch.randn(B_test, n_heads_kv, seq_each, hd)
    v_pad = torch.randn(B_test, n_heads_kv, seq_each, hd)
    out_pad = flash_attention_mxfp4_cpu_golden(
        q_pad, k_pad, v_pad, causal=True, block_q=32, block_kv=32,
    )
    # 转 packed: [B, H, S, D] -> [B*S, H, D]
    def _to_packed(t):
        B, H, S, D = t.shape
        return t.transpose(1, 2).reshape(B * S, H, D)
    q_packed = _to_packed(q_pad)
    k_packed = _to_packed(k_pad)
    v_packed = _to_packed(v_pad)
    cu_q_pad = [i * seq_each for i in range(B_test + 1)]
    cu_kv_pad = cu_q_pad
    out_varlen_eq = flash_attention_mxfp4_cpu_golden_varlen(
        q_packed, k_packed, v_packed,
        cu_q_pad, cu_kv_pad, causal=True,
        block_q=32, block_kv=32,
    )
    # padded 输出 [B, H, S, D] -> packed [B*S, H, D]
    out_pad_packed = _to_packed(out_pad)
    diff_pad_vs_varlen = (out_pad_packed - out_varlen_eq).abs().max().item()
    print(f"    padded API vs varlen API: 最大差异 {diff_pad_vs_varlen:.3e}")

    # (g) seq_used_kv > seq_used_q 触发 causal_offset (decoding 场景)
    print(f"  Causal offset (decoding: sq=1, sk=10):")
    cu_q_dec = [0, 1]
    cu_kv_dec = [0, 10]
    q_dec = torch.randn(1, n_heads_q, hd)
    k_dec = torch.randn(10, n_heads_kv, hd)
    v_dec = torch.randn(10, n_heads_kv, hd)
    out_dec = flash_attention_mxfp4_cpu_golden_varlen(
        q_dec, k_dec, v_dec,
        cu_q_dec, cu_kv_dec, causal=True,
        block_q=1, block_kv=4,
    )
    out_dec_ref = attention_mxfp4_varlen_reference(
        q_dec, k_dec, v_dec,
        cu_q_dec, cu_kv_dec, causal=True,
    )
    print(f"    flash vs ref: 最大差异 {(out_dec - out_dec_ref).abs().max().item():.3e}")

    # (h) seq_used = 0 边界
    print(f"  空 batch 边界 (seq_used_q=0): ", end="")
    cu_q_empty = [0, 5, 5]   # batch 1 有 0 个 Q
    used_q_empty = [3, 0]
    cu_kv_empty = [0, 4, 8]
    used_kv_empty = [4, 4]
    q_e = torch.randn(5, n_heads_q, hd)
    k_e = torch.randn(8, n_heads_kv, hd)
    v_e = torch.randn(8, n_heads_kv, hd)
    try:
        out_e = flash_attention_mxfp4_cpu_golden_varlen(
            q_e, k_e, v_e,
            cu_q_empty, cu_kv_empty,
            used_q_empty, used_kv_empty,
        )
        print(f"OK, 空 batch 输出全 0: {(out_e[5:5].abs().sum() == 0).item()}")
    except Exception as e:
        print(f"失败: {e}")

    print("\n" + "=" * 64)
    print("自检完成")
    print("=" * 64)