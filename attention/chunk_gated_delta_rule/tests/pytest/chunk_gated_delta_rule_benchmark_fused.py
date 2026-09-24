# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""NPU 小算子融合 benchmark，模拟当前 CGDR 的舍入与规约位置。

计算和输出留在输入 NPU；只有 batch 长度元数据读回主机以驱动 Python 循环。
统一符号：T=总 token 数，B=batch 数，Nk/Nv=QK/V 头数，Dk/Dv=特征维，C=64。
状态方向为 S:[Dv,Dk]；g 是 log decay，实际衰减为 exp(g)，缺省 g=0。

R 在 FP32 累加后写 UB，再生成 BF16 Cube 副本；跨 chunk 主状态保持 FP32。
Streaming 点积统一使用 NPU torch.sum；与 kernel 的末位差异由原三方判据检验。
该模型是三方比较中的低精度 benchmark，不替代独立公式 golden。
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F


SOURCE_COMMIT = "dbe6a50a2"
MODEL_REVISION = "r32_ub_state32_npu_sum_v3"
CHUNK_SIZE = 64
INVERSE_BLOCK_SIZE = CHUNK_SIZE // 2
# 与 FusedStage123Exact::UseStreamingExact 保持一致，均为静态代价权重。
STREAM_ARITHMETIC_FACTOR = 7
STREAM_VECTOR_COST = 256
CUBE_MAC_PER_CHUNK = 5832704


class ChunkPrefix(NamedTuple):
    """与状态无关的 chunk 中间量；P=Nv*chunk_count，下面省略共同前缀 [P]。

    数学符号与 C++ 对应：value_projection=U，state_projection=Kcd，
    gated_query=Q'，decayed_key=Kg，causal_scores=M，chunk_decay=exp(G_last)。
    除 U/decay 外，成员均已在对应发布点舍入 BF16，再升为 FP32 供小算子计算。
    """

    value_projection: torch.Tensor  # U=A@(beta*V), [C,Dv]，FP32 L0C 累加量
    state_projection: torch.Tensor  # Kcd=A@(-beta*exp(G)*K), [C,Dk]，已包含负号
    gated_query: torch.Tensor  # Q'=scale*exp(G)*Q, [C,Dk]
    decayed_key: torch.Tensor  # Kg=exp(G_last-G)*K, [C,Dk]
    causal_scores: torch.Tensor  # M_ij=scale*(Q_i K_j^T)*exp(G_i-G_j), j<=i, [C,C]
    chunk_decay: torch.Tensor  # exp(G_last), 标量


def _bf16(x):
    """显式落地 BF16，再升 FP32 供 NPU 运算；升精度不恢复丢失的位。"""
    return x.bfloat16().float()


def uses_streaming(seq_len, dk, dv, C=CHUNK_SIZE):
    """优化方案：小工作量用纯 AIV 递推，省去 Cube 固定 tile 的计算和搬运。

    比较 7*seq_len*Dk*Dv*256 与 ceil(seq_len/C)*2*5832704；分数不是实测耗时。
    保留 C 关键字兼容现有调用；实际融合路径固定 C=64，空序列只复制初态。
    """
    chunk_count = (seq_len + C - 1) // C
    cube_score = chunk_count * 2 * CUBE_MAC_PER_CHUNK
    streaming_score = seq_len * dk * dv * STREAM_ARITHMETIC_FACTOR * STREAM_VECTOR_COST
    return streaming_score < cube_score or seq_len == 0


def _inclusive_log_decay_scan(log_decay):
    """G_i=sum(g_0..g_i)，输入/输出均为 [P,C] FP32。

    优化方案：使用 Hillis–Steele 倍增顺序
    C = 64, 64 轮 -> 6 轮并行前缀和；
    改成串行 cumsum 会改变浮点加法顺序，clone 保证每轮读取更新前的值。
    """
    cumulative = log_decay.clone()
    for stride in (1, 2, 4, 8, 16, 32):
        cumulative[..., stride:] = cumulative[..., stride:] \
                                 + cumulative[..., :-stride].clone()
    return cumulative


def _invert_unit_lower_block(lower):
    """求 A=(I+L)^(-1)，输入严格下三角 L、输出 A 均为 [P,32,32]。

    由 (I+L)@A=I 逐行前代：A_i=e_i-sum_{j<i}(L_ij*A_j)。
    I+L 的对角线全为 1，因此无需除法或选主元。对应 InverseAIVPanel8DirectBrcVFImpl，
    kernel 每 8 行复用已求逆行；此模型保留每行按列号递增的 FP32 FMA 顺序。
    """
    inverse = torch.zeros_like(lower)
    accumulated = torch.zeros_like(lower)
    identity = torch.eye(INVERSE_BLOCK_SIZE, device=lower.device)
    for col in range(INVERSE_BLOCK_SIZE):
        inverse_row = identity[col] - accumulated[..., col, :]
        inverse[..., col, :] = inverse_row
        if col < INVERSE_BLOCK_SIZE - 1:
            accumulated[..., col + 1:, :] = torch.addcmul(
                accumulated[..., col + 1:, :],
                lower[..., col + 1:, col, None], inverse_row[..., None, :],
            )
    return inverse


def _invert_unit_lower_chunk(lower):
    """分块求 A=(I+L)^(-1)，模拟 kernel 的 BF16 舍入过程。

    输入 lower 是严格下三角 L:[P,C,C] FP32，不含单位阵；C=64。
    输出同 shape，以 FP32 存储各块舍入 BF16 后的值，是精确逆的低精度近似。
    P=Nv*chunk_count；以下公式省略 P，每个子块均为 [32,32]。

    将单位下三角矩阵及其逆按 2x2 分块：
        I+L = [[I+L00,   0  ],     A = [[A00,  0 ],
               [ L10,  I+L11]]          [A10, A11]]

    1. 两个对角块独立前代：A00=(I+L00)^(-1)，A11=(I+L11)^(-1)。
    2. 由 (I+L)@A=I 的左下块得 L10@A00+(I+L11)@A10=0，
       因此 A10=A11@(-L10)@A00；右上块保持 0。

    优化方案：kernel 的两个 AIV 分别求解一个 32x32 对角块，再用 Cube 的
    两次矩阵乘拼出左下块，避免对整个 64x64 矩阵逐行前代。
    本模型按发布顺序计算；记 R(x)=FP32(BF16(x))：
        A00=R(A00)，A11=R(A11)
        intermediate=R(R(-L10)@A00)
        A10=R(A11@intermediate)
    中间结果有独立 BF16 舍入，不能重排矩阵乘或替换为 torch.linalg.inv，
    否则虽在实数域公式等价，却无法复现 kernel 的低精度数值行为。
    """
    split = INVERSE_BLOCK_SIZE
    inverse = torch.zeros_like(lower)  # [P,64,64]；右上块保持为 0。

    # 对角块：[P,32,32]，FP32 前代完成后各舍入一次 BF16。
    inverse[..., :split, :split] = _bf16(
        _invert_unit_lower_block(lower[..., :split, :split])
    )
    inverse[..., split:, split:] = _bf16(
        _invert_unit_lower_block(lower[..., split:, split:])
    )

    # 左下块：先 (-L10)@A00，再 A11@intermediate；各乘积均为 [P,32,32]。
    lower_times_inverse = _bf16(
        _bf16(-lower[..., split:, :split]) @ inverse[..., :split, :split]
    )
    inverse[..., split:, :split] = _bf16(
        inverse[..., split:, split:] @ lower_times_inverse
    )
    return inverse


def _prepare_chunk_prefix(q, # [P,C,Dk]
                          k, # [P,C,Dk]
                          v, # [P,C,Dv]
                          beta, # [P,C]
                          g, # [P,C]
                          scale, nv, chunk_count):
    """
    准备不依赖 S 的前缀。

    kernel优化方案：kernel 用一次合并矩阵乘获得 KKT/QKT，并提前准备下一 chunk；
    这里分别计算两者以展示公式，在与 FIX->UB 相同的位置舍入 BF16。
    """

    kkt = _bf16(k @ k.transpose(-1, -2))  # KKT:[P,C,C]
    qkt = _bf16(q @ k.transpose(-1, -2))  # QKT:[P,C,C]

    cumulative = _inclusive_log_decay_scan(g)  # G:[P,C]
    exp_cumulative = cumulative.exp()
    log_decay_difference = cumulative[..., :, None] - cumulative[..., None, :]
    # 上三角不参与后续计算，先置 0 再 exp，避免无效位置溢出；下三角保留 exp(G_i-G_j)。
    pairwise_decay = torch.tril(log_decay_difference).exp()

    # L_ij=beta_i*KKT_ij*exp(G_i-G_j)，仅 j<i 非零；L/A 均为 [P,C,C]。
    lower = torch.tril(kkt * pairwise_decay, diagonal=-1) * beta[..., None]
    inverse = _invert_unit_lower_chunk(lower)

    raw_kcd = _bf16(k * (-(beta * exp_cumulative))[..., None])
    v_beta = _bf16(v * beta[..., None])
    q_prime = _bf16((q * scale) * exp_cumulative[..., None])
    kg = _bf16(k * (cumulative[..., -1, None] - cumulative).exp()[..., None])
    kcd = _bf16(inverse @ raw_kcd)
    u = inverse @ v_beta  # U 保留 L0C FP32，尚未发布残差 R。
    masked_qkt = _bf16(torch.tril((qkt * scale) * pairwise_decay))

    prefix = u, kcd, q_prime, kg, masked_qkt, cumulative[..., -1].exp()
    # [Nv*chunk_count,...] -> [Nv,chunk_count,...]，下一步按 chunk 取出所有头。
    prefix = ChunkPrefix(*(tensor.reshape(nv, chunk_count, *tensor.shape[1:]) for tensor in prefix))
    return prefix


def _stream_sequence(q, # (Si, Nv, Dk)
                     k, # (Si, Nv, Dk)
                     v, # (Si, Nv, Dv)
                     beta, # (Si, Nv)
                     state, # (Nv, Dv, Dk)
                     g, # (Si, Nv)
                     scale):
    """
    共享 QK 头已展开；state:[Nv,Dv,Dk] 始终为 FP32，输出为 BF16 [Si,Nv,Dv]。
    以下 r/S/o 公式按单个头书写，代码通过 Nv 维并行处理全部 value 头。
    """
    def dot(a, b):
        return (a * b).sum(dim=-1)

    output = torch.empty_like(v)
    # 逐token计算
    for token in range(len(q)):
        decay = g[token].exp()  # alpha:[Nv]
        token_beta = beta[token]  # [Nv]
        token_query = q[token]  # [Nv,Dk]
        token_key = k[token]  # [Nv,Dk]
        token_value = v[token]  # [Nv,Dv]

        # r=beta*(v-alpha*S_prev@k)，沿 Dk 点积：[Nv,Dv,Dk] x [Nv,1,Dk] -> [Nv,Dv]。
        residual = (token_value - decay[:, None] * dot(state, token_key[:, None, :])) * token_beta[:, None]
        # S=alpha*S_prev+r[:,None]*k[None,:]，外积广播为 [Nv,Dv,Dk]。
        state = state * decay[:, None, None] + token_key[:, None, :] * residual[..., None]
        # o=S@(scale*q)，读取更新后的 S，结果为 [Nv,Dv]。
        output[token] = dot(state, (token_query * scale)[:, None, :])
    return output.bfloat16(), state


def _pack_sequence_chunks(tensor, chunk_count):
    """[Si,Nv,*features] -> [Nv*chunk_count,C,*features]，先按头再按 chunk 排列。

    仅沿序列维度补 0，Q/K/V需要跳过feature维度
    """
    head_major = tensor.transpose(0, 1)
    # feature维度 无padding + S轴补0
    padding = [0, 0] * (head_major.ndim - 2) \
                + [0, chunk_count * CHUNK_SIZE - len(tensor)]
    # 补0并拆分chunk
    return F.pad(head_major, padding).reshape(-1, CHUNK_SIZE, *head_major.shape[2:]) # [Nv*chunk_count, C, *feature]


def _advance_chunk(prefix, state, state_dtype):
    """
    单个 chunk 的状态相关计算：prefix 各成员以前缀 [Nv] 索引，State:[Nv,Dv,Dk]。

    R=U+Kcd@S^T
    O=Q'@S^T+M@R
    S_next=exp(G_last)*S+R^T@Kg。
    返回 BF16 O:[Nv,C,Dv] 和 FP32 S_next:[Nv,Dv,Dk]。
    """
    # 优化方案：两次状态矩阵乘共享同一份 BF16 S^T；kernel 只搬一次 L1->L0B。
    published_state = _bf16(state).transpose(-1, -2)  # [Nv,Dk,Dv]
    # U 的 FP32 累加量直接叠加 Kcd@S^T，再写 UB
    residual_fp32 = torch.baddbmm(prefix.value_projection, prefix.state_projection, published_state)
    # R 转 BF16 进行 Cube 计算
    residual_bf16 = _bf16(residual_fp32)  # [Nv,C,Dv]
    inter_chunk_output = prefix.gated_query @ published_state  # [Nv,C,Dv]
    output = torch.baddbmm(inter_chunk_output, prefix.causal_scores, residual_bf16).bfloat16()
    state_delta = residual_bf16.transpose(-1, -2) @ prefix.decayed_key  # [Nv,Dv,Dk]
    if state_dtype == torch.bfloat16:
        # BF16 接口保留 delta 的既有 BF16 舍入；主状态的 FP32 精度与此独立。
        state_delta = _bf16(state_delta)
    next_state = state * prefix.chunk_decay[:, None, None] + state_delta
    return output, next_state


def _chunk_sequence(q, # (Si, Nv, Dk)
                    k, # (Si, Nv, Dk)
                    v, # (Si, Nv, Dv)
                    beta, # (Si, Nv)
                    state, # (Nv, Dv, Dk)
                    g, # (Si, Nv)
                    scale, state_dtype):
    """按 chunk 顺序递推一个序列；输入 shape 同 _stream_sequence，主状态跨块保留 FP32。"""
    seq_len, value_heads = v.shape[:2]
    chunk_count = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    pad_q, pad_k, pad_v, pad_beta, pad_g = [_pack_sequence_chunks(tensor, chunk_count) for tensor in (q, k, v, beta, g)]

    prefix = _prepare_chunk_prefix(pad_q, pad_k, pad_v, pad_beta, pad_g, scale, value_heads, chunk_count)
    output = torch.empty_like(v, dtype=torch.bfloat16)
    # 逐chunk计算，[Nv, chunk_count, C, D]
    for chunk in range(chunk_count):
        chunk_prefix = ChunkPrefix(*(tensor[:, chunk] for tensor in prefix))
        chunk_output, state = _advance_chunk(chunk_prefix, state, state_dtype)

        token_start = chunk * CHUNK_SIZE
        valid_tokens = min(CHUNK_SIZE, seq_len - token_start)
        output[token_start:token_start + valid_tokens] = chunk_output[:, :valid_tokens].transpose(0, 1)
    return output, state


@torch.no_grad()
def chunk_gdn_benchmark_fused(
    q, # (T, Nk, Dk)
    k, # (T, Nk, Dk)
    v, # (T, Nv, Dv)
    beta, # (T, Nv)
    scale, # float
    initial_state, # (B, Nv, Dv, Dk)
    actual_seq_lengths, # (B,)
    g=None, # (T, Nv)
    chunk_size=CHUNK_SIZE
):
    """
    actual_seq_lengths 保存各序列长度（非累积终点），总和等于 T，可含空序列。
    Nv 是 Nk 的整数倍；scale 为调用方给定的 query 缩放，chunk_size 固定为 64。
    返回 output:[T,Nv,Dv] BF16 和 states:[B,Nv,Dv,Dk]（与初态 dtype 一致）。
    """
    if chunk_size != CHUNK_SIZE:
        raise ValueError("融合模型只支持 chunk_size=64")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, beta)):
        raise ValueError("q/k/v/beta 必须是接口要求的 BF16，不在模型入口隐式改写输入")
    if initial_state.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("initial_state 必须是 BF16 或 FP32")
    if q.device.type != "npu":
        raise ValueError("fused benchmark 必须接收 NPU tensor")
    if any(t.device != q.device for t in (k, v, beta, initial_state)) or (g is not None and g.device != q.device):
        raise ValueError("benchmark 输入必须位于同一 NPU")
    lengths = actual_seq_lengths.detach().cpu().tolist()
    if any(length < 0 for length in lengths) or sum(lengths) != len(q):
        raise ValueError("各 batch 长度必须非负且总和等于 T")

    state_dtype = initial_state.dtype
    q, k, v, beta, initial_state = [t.detach().float() for t in (q, k, v, beta, initial_state)]
    g = torch.zeros(v.shape[:2], device=v.device) if g is None else g.detach().float()
    nv, nk = v.shape[1], q.shape[1]
    # QK 共享：value_head h 读取 qk_head=floor(h/group_size)，展开后 Q/K:[T,Nv,Dk]。
    head_group_size = nv // nk
    q = q.repeat_interleave(head_group_size, dim=1)
    k = k.repeat_interleave(head_group_size, dim=1)
    output = torch.empty_like(v, dtype=torch.bfloat16)
    states = torch.empty_like(initial_state, dtype=state_dtype)

    token_start = 0
    for batch, seq_len in enumerate(lengths):
        token_end = token_start + seq_len
        state = initial_state[batch].clone()  # [Nv,Dv,Dk]，空序列原样返回
        cur_q, cur_k, cur_v, cur_beta, cur_g = [tensor[token_start:token_end] for tensor in (q, k, v, beta, g)]
        if uses_streaming(seq_len, k.shape[-1], v.shape[-1]):
            # 小数据量走逐token计算分支
            sequence_output, state = _stream_sequence(
                cur_q, cur_k, cur_v, cur_beta, state, cur_g, scale,
            )
        else:
            sequence_output, state = _chunk_sequence(
                cur_q, cur_k, cur_v, cur_beta, state, cur_g, scale, state_dtype,
            )
        output[token_start:token_end] = sequence_output
        states[batch] = state.to(state_dtype)
        token_start = token_end
    return output, states
