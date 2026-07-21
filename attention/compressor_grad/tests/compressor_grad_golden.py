# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""
Compressor 反向传播独立实现。

按 param.txt 定义的接口，与 torch autograd 推导的反向结果对齐。

═══════════════════════════════════════════════════════════════════════════════
                            反 向 传 播 推 导 链
═══════════════════════════════════════════════════════════════════════════════

【前向过程回顾】(对应 compressor_core.py 中 compressor_forward)

  (1)  Matmul（与位置无关，APE 单独加在 score 上）：
         new_kv    = x @ wkv.T         → (T, coff * head_dim)   纯 matmul，无 APE
         new_score = x @ wgate.T       → (T, coff * head_dim)   纯 matmul，无 APE
         new_score += ape[global_pos % cmp_ratio, :]             APE 按位置累加

      关键：new_score = x @ wgate.T + ape，反向时梯度会分流到 dwgate/dx 和 dape

  (2)  数据重组（仅 coff=2 时需要，coff=1 可以认为是 coff=2 的特例）：
       对每个压缩块，收集 coff 组各 cmp_ratio 个 token 的特征：
         kv_2d    = reshape([kv_block_0, ..., kv_block_{coff-1}], (coff*cmp_ratio, head_dim))
         sc_score_2d = reshape([score_block_0, ..., score_block_{coff-1}], ...)

       其中：
         - block_{coff-1} (当前块)：来自 new_kv/new_score 的 [当前偏移, :]，列 [当前头维, :]
         - block_0       (前一块)：来自 new_kv/new_score 的 [前面偏移, :]，列 [前一头维,  :]
         - 如果块是 batch 开头，缺失的 token 从 page cache 补（反向时不涉及 cache，因为
           cache 不在计算图中，梯度直接流向 new_kv/new_score）

      数据布局示意（coff=2, cmp_ratio=4 时 kv_2d 的行结构）：
         行 0-3:  前一块 token（overlap_id=0, 列 head_dim*0 : head_dim*1）
         行 4-7:  当前块 token（overlap_id=1, 列 head_dim*1 : head_dim*2）

  (3)  压缩计算：
         softmax_score  = softmax(sc_score_2d, dim=0)  按列 softmax → 每列是概率分布
         compressed = sum(kv_2d * softmax_score, dim=0) → (1, head_dim)

【反向传播推导】

  已知：上游梯度 d_compressed_kv = ∂Loss/∂compressed，形状 (1, head_dim)

  ┌─────────────────────────────────────────────────────────────────────┐
  │ 反步 1:  通过 element-wise mul + reduce_sum 反向                   │
  │                                                                     │
  │   前向: C = Σᵢ (Kᵢ ⊙ Wᵢ)   其中 K=kv_2d, W=softmax_score           │
  │                                                                     │
  │   由链式法则:                                                        │
  │     ∂L/∂Kᵢ = ∂L/∂C · ∂C/∂Kᵢ = dC ⊙ Wᵢ           (标量*向量, 广播) │
  │     ∂L/∂Wᵢ = ∂L/∂C · ∂C/∂Wᵢ = dC ⊙ Kᵢ                            │
  │                                                                     │
  │   即 (设 dC = d_compressed_kv.unsqueeze(0)):                        │
  │     d_kv_block      = dC * W    — 传给 new_kv 的梯度                │
  │     d_score_weighted = dC * K   — softmax 反向的输入端               │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ 反步 2:  通过 softmax 反向 (dim=0, 每列独立)                        │
  │                                                                     │
  │   softmax 定义: sⱼ = exp(zⱼ) / Σₖ exp(zₖ)                          │
  │                                                                     │
  │   雅可比矩阵: ∂sⱼ/∂zᵢ = sⱼ(δⱼᵢ - sᵢ)                              │
  │                                                                     │
  │   反向传播: ∂L/∂zᵢ = Σⱼ ∂L/∂sⱼ · ∂sⱼ/∂zᵢ = sᵢ · (∂L/∂sᵢ - Σⱼ sⱼ·∂L/∂sⱼ)│
  │                                                                     │
  │   写成向量形式 (每列独立):                                            │
  │     d_score = softmax_score * (d_score_weighted - sum(softmax_score * d_score_weighted, dim=0)) │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ 反步 3:  梯度路由 — 将 kv_2d/sc_score_2d 的梯度映射回            │
  │           flat new_kv / new_score 数组的对应位置                     │
  │                                                                     │
  │   路由规则（以 coff=2 为例）:                                        │
  │     kv_2d 行 0-3 (前一块) → flat 的 prev_tokens 行, 列 0:head_dim   │
  │     kv_2d 行 4-7 (当前块) → flat 的 cur_tokens  行, 列 head_dim:2*head_dim │
  │                                                                     │
  │   其中 prev_tokens 的 flat 偏移 = cur_offset - (cmp_ratio - cfs_p)   │
  │   cfs_p = 从 cache 补的 token 数（cache 在计算图外，梯度流向 flat）  │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ 反步 4:  APE 梯度累积                                               │
  │                                                                     │
  │   前向: new_score[t] += ape[t_global % cmp_ratio, :]                 │
  │                                                                     │
  │   反向: d_ape 从 d_new_score 中按位置模 cmp_ratio 分组累加           │
  │     d_ape[pos_mod, :] += Σ_{t: t%cmp_ratio==pos_mod} d_new_score[t] │
  │                                                                     │
  │   注意: d_new_score 同时流向 d_wgate/d_x (matmul 反向) 和 d_ape      │
  │         因为前向是 new_score = x@wgate + ape，两个操作是并行的加法    │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ 反步 5:  Matmul 反向                                                │
  │                                                                     │
  │   前向: Y = X @ W.T     (new_kv = x @ wkv.T, new_score = x @ wgate.T) │
  │   反向: dX = dY @ W   ,  dW = dY.T @ X                              │
  │                                                                     │
  │   由于 x 同时流入两条 matmul 路径:                                    │
  │     d_x = d_new_kv @ wkv + d_new_score @ wgate                       │
  │     d_wkv   = d_new_kv.T @ x                                         │
  │     d_wgate = d_new_score.T @ x                                      │
  └─────────────────────────────────────────────────────────────────────┘
"""
import torch
from typing import Optional, List, Tuple


def compressor_grad_golden(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    d_cmp_kv: torch.Tensor,
    softmax_score: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    coff: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compressor 算子的反向传播。

    参数:
        x:            输入张量 [B, S, H] 或 [T, H]（BSH 布局或 TH 布局）
        wkv:          KV 投影权重 (coff * head_dim, H)
        wgate:        Gate 投影权重 (coff * head_dim, H)
        d_cmp_kv:     前向输出 compressed KV 的上游梯度 ∂Loss/∂cmp_kv
                      BSH: (B, ceil(S/cmp_ratio), head_dim)
                      TH:  (min(T, T//cmp_ratio+B), head_dim)
        kv:           前向保存的中间变量（reshape 后的 KV 数据）
                      BSH: (B, ceil(S/cmp_ratio), coff*cmp_ratio, head_dim)
                      TH:  (min(T, T//cmp_ratio+B), coff*cmp_ratio, head_dim)
        softmax_score: 前向保存的 softmax 权重
                      形状同 kv
        start_pos:    每个 batch 的起始全局序列位置 (B,) int32。
                      None 时默认全为 0（即所有 batch 从头开始）
        cmp_ratio:    压缩率：每 cmp_ratio 个 token 压缩为 1 个
        coff:         重叠因子（1=无重叠, 2=相邻块有 50% 重叠）
        cu_seqlens:   TH 布局下的累积序列长度 (B+1,) int32。
                      BSH 布局时传 None，函数内部不依赖此参数定位 token
        seqused:      每个 batch 实际参与压缩的 token 数 (B,) int32。
                      None 表示使用全部 sequence length

    返回:
        d_x:      输入 x 的梯度 ∂Loss/∂x
        d_wkv:    权重 wkv 的梯度 ∂Loss/∂wkv
        d_wgate:  权重 wgate 的梯度 ∂Loss/∂wgate
        d_ape:    APE 位置编码的梯度 ∂Loss/∂ape, (cmp_ratio, coff * head_dim)
    """
    # ══════════════════════════════════════════════════════════════════════
    # 阶段 A: 解析输入参数，确定数据布局
    # ══════════════════════════════════════════════════════════════════════

    # head_dim: 每个注意力头的维度 D。
    # wkv 的 shape 是 (coff * head_dim, H)，所以 head_dim = wkv.shape[0] // coff
    head_dim = wkv.shape[0] // coff

    # 两种数据布局：
    #   TH 布局 (is_th_layout=True):  x shape (T, H)，B 和 S 合并到 T 维
    #       此时 cu_seqlens 提供每个 batch 在 T 维的起止边界
    #   BSH 布局 (is_th_layout=False): x shape (B, S, H)，batch 独立存储
    #       此时不需要 cu_seqlens，直接用 b_idx * S 定位
    is_th_layout = (cu_seqlens is not None)

    # B: batch 数量。start_pos 为 None 时默认全 0
    if start_pos is None:
        if is_th_layout:
            num_batches = cu_seqlens.shape[0] - 1
        else:
            num_batches = x.shape[0]
        start_positions = [0] * num_batches
    else:
        num_batches = start_pos.shape[0]
        start_positions = start_pos.tolist()

    # 前向计算全部使用 float32（与 golden 一致，保证数值精度）
    x_f32 = x.float()
    wkv_f32 = wkv.float()
    wgate_f32 = wgate.float()

    # 确定总 token 数和 batch/seq 维度（仅 BSH 布局需要保留 shape 信息用于最后 reshape）
    if is_th_layout:
        total_tokens = x.shape[0]
        batch_size = seq_len = None  # TH 布局下不需要恢复 BSH shape
    else:
        batch_size, seq_len, hidden_size = x.shape
        total_tokens = batch_size * seq_len

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 B: 分配梯度累积缓冲区
    # ══════════════════════════════════════════════════════════════════════
    #
    # 前向计算中有三个"源头"张量会产生梯度：
    #   (a) new_kv    = x @ wkv.T         — 仅仅来自 matmul
    #   (b) new_score = x @ wgate.T + APE — 来自 matmul + 加法
    #   (c) APE 本身                        — 位置编码参数
    #
    # 在反向过程中，我们先将每个压缩块的梯度反向传播到 (a)(b)(c) 的对应
    # token/位置，得到 d_new_kv、d_new_score、d_ape。最后一步再将这些
    # 累积梯度通过 matmul 反向得到 d_x, d_wkv, d_wgate。

    # 维度约定: T=total_tokens, C=coff*head_dim, H=hidden_size
    #
    # d_new_kv:    (T, C)  ∂Loss/∂new_kv — 仅来自 matmul 反向
    d_new_kv = torch.zeros(total_tokens, coff * head_dim, device=x.device)

    # d_new_score: (T, C)  ∂Loss/∂new_score — 同时流入 matmul 反向和 APE 反向
    d_new_score = torch.zeros(total_tokens, coff * head_dim, device=x.device)

    # d_ape:       (cmp_ratio, C)  ∂Loss/∂ape
    d_ape = torch.zeros(cmp_ratio, coff * head_dim, device=x.device)

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 C: 逐 batch、逐压缩块进行反向传播
    # ══════════════════════════════════════════════════════════════════════
    #
    # 遍历顺序与前向完全一致：先遍历 batch，再在每个 batch 内按 cmp_ratio
    # 步长遍历压缩块。对于每个压缩块，执行反步 1→3（参见文件头部推导）。
    #
    # grad_output_index / bsh_block_idx: 按压缩块顺序消费三个输入（见 C.4a）
    grad_output_index = 0

    for batch_idx in range(num_batches):
        # ── C.1  确定当前 batch 的序列范围 ────────────────────────────────
        # batch_start: 当前 batch 在全局序列中的绝对起始位置
        batch_start = start_positions[batch_idx]

        # batch_seq_used: 当前 batch 实际参与计算的 token 数
        # seqused 非 None 时优先使用，否则用 sequence length
        if seqused is not None:
            batch_seq_used = seqused[batch_idx].item()
        elif is_th_layout:
            batch_seq_used = int(cu_seqlens[batch_idx + 1].item() - cu_seqlens[batch_idx].item())
        else:
            batch_seq_used = x.shape[1]

        # compress_limit: 需要压缩的序列位置上限（向下对齐到 cmp_ratio 的整数倍）
        # 例如: batch_start=5, seq_used=10, cmp_ratio=4 → limit = (5+10)//4*4 = 12
        #       位置 5,6,7,8,9,10,11 会被压缩，12,13,14 不压缩（不足一个完整块）
        compress_limit = (batch_start + batch_seq_used) // cmp_ratio * cmp_ratio

        # batch_seq_idx: 当前在 batch 内已处理到的相对位置（相对于 batch_start）
        batch_seq_idx = 0
        # bsh_block_idx: BSH 布局下当前 batch 的压缩块序号（从 0 开始）
        bsh_block_idx = 0

        while batch_seq_idx < batch_seq_used:
            # ── C.2  当前压缩块的全局序列范围 ─────────────────────────────
            # block_seq_start: 当前块在全局序列中的起始位置
            block_seq_start = batch_start + batch_seq_idx

            # block_seq_end: 当前块在全局序列中的结束位置（向前对齐到 cmp_ratio 整数倍）
            # 对于最后一个不完整块，end 会被截断到 batch 结尾
            block_seq_end = (block_seq_start // cmp_ratio) * cmp_ratio + cmp_ratio
            if block_seq_end > batch_start + batch_seq_used:
                block_seq_end = batch_start + batch_seq_used

            # 推进到下一块（在 guard clause 之前更新，避免重复）
            batch_seq_idx = block_seq_end - batch_start

            # ── C.4  尾块跳过（不足 cmp_ratio 个 token，仅写 cache）──
            if block_seq_start >= compress_limit:
                break

            # ═══════════ 反向传播：反步 1→3 ═══════════

            # ── 获取当前块的上游梯度和前向中间变量 ──
            if is_th_layout:
                d_compressed_kv = d_cmp_kv[grad_output_index].float()
                saved_kv = kv[grad_output_index].float()
                saved_softmax = softmax_score[grad_output_index].float()
                grad_output_index += 1
            else:
                d_compressed_kv = d_cmp_kv[batch_idx, bsh_block_idx].float()
                saved_kv = kv[batch_idx, bsh_block_idx].float()
                saved_softmax = softmax_score[batch_idx, bsh_block_idx].float()
                bsh_block_idx += 1

            # ══════════════════════════════════════════════════════════
            # 【反步 1】element-wise mul + reduce_sum 反向
            # ══════════════════════════════════════════════════════════
            #
            # 前向: C = Σ_i (K_i ⊙ W_i)   K=(N, hd), W=(N, hd), C=(1, hd)
            #   其中 N=coff*cmp_ratio, hd=head_dim
            # 反向: dK_i = dC ⊙ W_i,  dW_i = dC ⊙ K_i  (广播: dC (1,hd)→(N,hd))

            d_compressed_kv_2d = d_compressed_kv.unsqueeze(0)  # (1,hd) → (N,hd)
            d_kv_block = d_compressed_kv_2d * saved_softmax     # (N,hd)
            d_score_weighted = d_compressed_kv_2d * saved_kv    # (N,hd)

            # ══════════════════════════════════════════════════════════
            # 【反步 2】softmax 反向 (dim=0, 每列独立)
            # ══════════════════════════════════════════════════════════
            #
            # 前向: S = softmax(Z, dim=0)   S,Z ∈ (N,hd)
            # 反向: dZ = S ⊙ (dS - column_sum(S ⊙ dS))
            #   S ⊙ dS → (N,hd)
            #   column_sum → (1,hd)
            #   dZ → (N,hd)

            softmax_backward_sum = (saved_softmax * d_score_weighted).sum(dim=0, keepdim=True)  # (1,hd)
            d_score_block = saved_softmax * (d_score_weighted - softmax_backward_sum)  # (N,hd)

            # ══════════════════════════════════════════════════════════
            # 【反步 3】梯度路由 → 映射回 flat new_kv/new_score 数组
            # ══════════════════════════════════════════════════════════
            #
            # d_kv_block, d_score_block ∈ (N, hd)  其中 N=coff*cmp_ratio
            # 路由目标: d_new_kv, d_new_score ∈ (T, C)  其中 C=coff*hd
            #
            # kv_2d 的 N 行按 overlap_id 分组:
            #   overlap_id=0:       行 [0, cmp_ratio)       → 列 [0, hd)
            #   overlap_id=coff-1:  行 [(coff-1)*cr, N)    → 列 [(coff-1)*hd, C)

            # flat 偏移: 当前块在展平 (T, C) 数组中的 token 行范围
            flat_offset_base = (
                int(cu_seqlens[batch_idx].item()) if is_th_layout
                else batch_idx * x.shape[1]
            )
            flat_offset_start = flat_offset_base + (block_seq_start - batch_start)
            flat_offset_end = flat_offset_base + (block_seq_end - batch_start)

            # ── 3a  当前 overlap 槽位 (overlap_id = coff - 1) ──
            current_overlap_id = coff - 1
            current_dim_start = current_overlap_id * head_dim
            current_dim_end = (current_overlap_id + 1) * head_dim

            current_tokens_from_cache = 0
            if batch_start == block_seq_start:
                current_tokens_from_cache = batch_start % cmp_ratio

            block_row_start = current_overlap_id * cmp_ratio + current_tokens_from_cache
            block_row_end = current_overlap_id * cmp_ratio + cmp_ratio

            # d_kv_block[rows, :] → d_new_kv[tokens, cols]   (N,hd)子块 → (T,C)子块
            d_new_kv[
                flat_offset_start:flat_offset_end,
                current_dim_start:current_dim_end,
            ] += d_kv_block[block_row_start:block_row_end, :]

            d_new_score[
                flat_offset_start:flat_offset_end,
                current_dim_start:current_dim_end,
            ] += d_score_block[block_row_start:block_row_end, :]

            # d_ape: (cmp_ratio, C)  按 pos%cmp_ratio 累加 d_score_block 对应行
            ape_offset = block_seq_start % cmp_ratio
            n_pos = block_seq_end - block_seq_start
            d_ape[ape_offset:ape_offset + n_pos, current_dim_start:current_dim_end] += d_score_block[block_row_start:block_row_start + n_pos, :]

            # ── 3b  前一个 overlap 槽位 (overlap_id = 0, 仅 coff == 2) ──
            if coff == 2:
                prev_tokens_from_cache = 0
                if batch_start == block_seq_start:
                    prev_tokens_from_cache = cmp_ratio
                elif block_seq_start - cmp_ratio < batch_start:
                    prev_tokens_from_cache = batch_start % cmp_ratio

                # 全来自 cache，无需路由
                if prev_tokens_from_cache >= cmp_ratio:
                    continue

                prev_dim_start = 0
                prev_dim_end = head_dim
                block_row_start = prev_tokens_from_cache
                block_row_end = cmp_ratio

                prev_flat_offset_start = flat_offset_start - (cmp_ratio - prev_tokens_from_cache)
                prev_flat_offset_end = flat_offset_start

                # (N,hd)子块 → (T,C)子块，前一块列偏移 [0, hd)
                d_new_kv[
                    prev_flat_offset_start:prev_flat_offset_end,
                    prev_dim_start:prev_dim_end,
                ] += d_kv_block[block_row_start:block_row_end, :]

                d_new_score[
                    prev_flat_offset_start:prev_flat_offset_end,
                    prev_dim_start:prev_dim_end,
                ] += d_score_block[block_row_start:block_row_end, :]

                n_pos = cmp_ratio - prev_tokens_from_cache
                d_ape[prev_tokens_from_cache:cmp_ratio, prev_dim_start:prev_dim_end] += d_score_block[block_row_start:block_row_start + n_pos, :]

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 D: Matmul 反向
    # ══════════════════════════════════════════════════════════════════════
    #
    # 维度约定: T=total_tokens, C=coff*head_dim, H=hidden_size
    #
    # 前向有两条独立的 matmul:
    #   new_kv    = x @ wkv.T      → (T, H) @ (H, C) = (T, C)   不含 APE
    #   new_score = x @ wgate.T    → (T, H) @ (H, C) = (T, C)   APE 后加
    #
    # Matmul 反向公式 (Y = X @ W.T):
    #   dX = dY @ W       → (T, C) @ (C, H) = (T, H)
    #   dW = dY.T @ X     → (C, T) @ (T, H) = (C, H)
    #
    # x 同时流入两条路径，d_x 是两条之和:
    #   d_x     = d_new_kv @ wkv   + d_new_score @ wgate
    #             (T,C)@(C,H)→(T,H)   (T,C)@(C,H)→(T,H)
    #   d_wkv   = d_new_kv.T @ x_flat   → (C, T) @ (T, H) = (C, H)
    #   d_wgate = d_new_score.T @ x_flat → (C, T) @ (T, H) = (C, H)

    x_flat = x_f32 if is_th_layout else x_f32.reshape(total_tokens, x.shape[-1])  # (T, H)

    d_x_flat = d_new_kv @ wkv_f32 + d_new_score @ wgate_f32  # (T, H)
    d_wkv = d_new_kv.T @ x_flat                                # (C, H)
    d_wgate = d_new_score.T @ x_flat                            # (C, H)

    # 根据原始输入布局恢复 d_x 的形状，并转换回输入的 dtype
    if is_th_layout:
        d_x = d_x_flat.to(x.dtype)
    else:
        d_x = d_x_flat.reshape(batch_size, seq_len, hidden_size).to(x.dtype)

    return d_x, d_wkv.to(wkv.dtype), d_wgate.to(wgate.dtype), d_ape
