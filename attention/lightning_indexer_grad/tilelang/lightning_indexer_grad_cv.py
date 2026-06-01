#!/usr/bin/env python3
"""
TileLang-Ascend Lightning Indexer Gradient 算子 (CV 分离架构)

单 kernel 设计，使用 T.Scope("C")/T.Scope("V") 分离 Cube/Vector 阶段，
通过 workspace 在 CV 间传递中间结果。

计算流程:
    Vec1:  GatherTopk (按 indices 从 Key gather)
    Cube:  Cube1: Q @ K_gather^T -> reluIn (FP32 workspace)
           Cube2: weights ⊗ dy -> reluGradIn (FP16 workspace)
    Vec2:  ReLU反向 + dWeights
    Cube:  Cube3: reluGradOut @ K_gather -> dQuery
           Cube4: reluGradOut^T @ Q -> scatterAdd (FP32 workspace)
    Vec3:  ScatterAdd -> dKey

输入 (BSND 格式):
    Query:    (B, S1, N1, D)   FP16
    Key:      (B, S2, N2, D)   FP16
    Dy:       (B, S1, topK)    FP16
    Weights:  (B, S1, N1)      FP16
    Indices:  (B, S1, topK)    INT32
"""

import tilelang
from tilelang import jit
import tilelang.language as T
import torch
import torch_npu
import numpy as np

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

C0_SIZE = 16


@jit(out_idx=[5, 6, 7], workspace_idx=[8, 9, 10, 11], pass_configs=pass_configs)
def lightning_indexer_grad_kernel(
    B: int, S1: int, S2: int, N1: int, N2: int, G: int, D: int, TOPK: int,
    BLOCK_M: int, BLOCK_N: int, BLOCK_D: int,
    input_dtype: str = 'float16',
    calc_dtype: str = 'float32',
):
    """
    Lightning Indexer Grad 单 kernel (CV 分离架构)

    分块参数:
        BLOCK_M: 切分 G 维度 (group 方向)
        BLOCK_N: 切分 TOPK 维度
        BLOCK_D: 切分 D 维度 (head_dim 方向)
    """

    m_num = G // BLOCK_M
    n_num = TOPK // BLOCK_N
    d_num = D // BLOCK_D

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, D), input_dtype),
        Key: T.Tensor((B, S2, N2, D), input_dtype),
        Dy: T.Tensor((B, S1, TOPK), input_dtype),
        Weights: T.Tensor((B, S1, N1), input_dtype),
        Indices: T.Tensor((B, S1, TOPK), 'int32'),
        dQuery: T.Tensor((B, S1, N1, D), input_dtype),
        dKey: T.Tensor((B, S2, N2, D), input_dtype),
        dWeights: T.Tensor((B, S1, N1), input_dtype),
        K_gather_ws: T.Tensor((B, S1, N2, TOPK, D), input_dtype),
        reluIn_ws: T.Tensor((B, S1, N2, G, TOPK), calc_dtype),
        reluGrad_ws: T.Tensor((B, S1, N2, G, TOPK), input_dtype),
        scatterAdd_ws: T.Tensor((B, S1, N2, TOPK, D), calc_dtype),
    ):
        with T.Kernel(B * S1, is_npu=True) as (cid, _):
            b_idx = cid // S1
            s1_idx = cid % S1

            # ========== Vec1: GatherTopk ==========
            with T.Scope("V"):
                idx_ub = T.alloc_ub(TOPK, 'int32')
                key_tile_ub = T.alloc_ub(D, input_dtype)
                gather_dst_ub = T.alloc_ub(D, input_dtype)

                for n2 in T.serial(N2):
                    T.copy(Indices[b_idx, s1_idx, 0:TOPK], idx_ub)
                    for t in T.serial(TOPK):
                        s2_pos = idx_ub[t]
                        T.copy(Key[b_idx, s2_pos, n2, 0:D], key_tile_ub)
                        for d_i in T.serial(D):
                            gather_dst_ub[d_i] = key_tile_ub[d_i]
                        T.copy(gather_dst_ub, K_gather_ws[b_idx, s1_idx, n2, t, 0:D])

            # ========== Cube1 + Cube2 ==========
            with T.Scope("C"):
                for n2 in T.serial(N2):
                    n1_start = n2 * G

                    # --- Cube1: Q_group(G,D) @ K_gather^T(TOPK,D) -> reluIn(G,TOPK) ---
                    Q_L1 = T.alloc_L1((BLOCK_M, BLOCK_D), input_dtype)
                    K_L1 = T.alloc_L1((BLOCK_N, BLOCK_D), input_dtype)
                    C_L0C = T.alloc_L0C((BLOCK_M, BLOCK_N), calc_dtype)

                    for m_idx in T.serial(m_num):
                        for n_idx in T.serial(n_num):
                            for k_idx in T.serial(d_num):
                                T.copy(Query[b_idx, s1_idx, n1_start + m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, k_idx * BLOCK_D:(k_idx + 1) * BLOCK_D], Q_L1)
                                T.copy(K_gather_ws[b_idx, s1_idx, n2, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N, k_idx * BLOCK_D:(k_idx + 1) * BLOCK_D], K_L1)
                                T.gemm_v0(Q_L1, K_L1, C_L0C, transpose_B=True, init=(k_idx == 0))
                            T.copy(C_L0C, reluIn_ws[b_idx, s1_idx, n2, m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N])

                    # --- Cube2: weights(G,1) ⊗ dy(1,TOPK) -> reluGradIn(G,TOPK) ---
                    W_L1 = T.alloc_L1((BLOCK_M, C0_SIZE), input_dtype)
                    DY_L1 = T.alloc_L1((C0_SIZE, BLOCK_N), input_dtype)
                    C2_L0C = T.alloc_L0C((BLOCK_M, BLOCK_N), calc_dtype)

                    for m_idx in T.serial(m_num):
                        for n_idx in T.serial(n_num):
                            T.copy(Weights[b_idx, s1_idx, n1_start + m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M], W_L1)
                            T.copy(Dy[b_idx, s1_idx, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N], DY_L1)
                            T.gemm_v0(W_L1, DY_L1, C2_L0C, init=True)
                            T.copy(C2_L0C, reluGrad_ws[b_idx, s1_idx, n2, m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N])

                T.set_cross_flag("FIX", 0)

            # ========== Vec2: ReLU反向 + dWeights ==========
            with T.Scope("V"):
                T.wait_cross_flag(0)

                relu_in_ub = T.alloc_ub((BLOCK_M, BLOCK_N), calc_dtype)
                relu_grad_ub = T.alloc_ub((BLOCK_M, BLOCK_N), input_dtype)
                dy_ub = T.alloc_ub(BLOCK_N, input_dtype)
                dy_fp32_ub = T.alloc_ub(BLOCK_N, calc_dtype)
                mask_ub = T.alloc_ub((BLOCK_M, BLOCK_N), calc_dtype)
                zero_ub = T.alloc_ub(1, calc_dtype)
                reduce_in_ub = T.alloc_ub((BLOCK_M, BLOCK_N), calc_dtype)
                reduce_out_ub = T.alloc_ub(BLOCK_M, calc_dtype)
                dw_fp32_ub = T.alloc_ub(BLOCK_M, calc_dtype)
                dw_ub = T.alloc_ub(BLOCK_M, input_dtype)

                T.tile.fill(zero_ub, 0.0)

                for n2 in T.serial(N2):
                    n1_start = n2 * G

                    for m_idx in T.serial(m_num):
                        for n_idx in T.serial(n_num):
                            T.copy(reluIn_ws[b_idx, s1_idx, n2, m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N], relu_in_ub)
                            T.copy(reluGrad_ws[b_idx, s1_idx, n2, m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N], relu_grad_ub)

                            T.tile.compare(mask_ub, relu_in_ub, zero_ub, "GT")
                            T.tile.select(relu_grad_ub, mask_ub, relu_grad_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                            T.copy(relu_grad_ub, reluGrad_ws[b_idx, s1_idx, n2, m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N])

                    for m_idx in T.serial(m_num):
                        T.tile.fill(reduce_in_ub, 0.0)
                        for n_idx in T.serial(n_num):
                            T.copy(reluIn_ws[b_idx, s1_idx, n2, m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N], relu_in_ub)
                            T.copy(Dy[b_idx, s1_idx, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N], dy_ub)
                            T.tile.cast(dy_fp32_ub, dy_ub, "CAST_NONE", BLOCK_N)
                            for row in T.serial(BLOCK_M):
                                T.tile.mul(reduce_in_ub[row, :], relu_in_ub[row, :], dy_fp32_ub)
                        T.reduce_sum(reduce_in_ub, reduce_out_ub, dim=-1, clear=True)
                        T.tile.cast(dw_ub, reduce_out_ub, "CAST_ROUND", BLOCK_M)
                        T.copy(dw_ub, dWeights[b_idx, s1_idx, n1_start + m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M])

                T.set_cross_flag("FIX", 1)

            # ========== Cube3 + Cube4 ==========
            with T.Scope("C"):
                T.wait_cross_flag(1)

                for n2 in T.serial(N2):
                    n1_start = n2 * G

                    # --- Cube3: reluGradOut(G,TOPK) @ K_gather(TOPK,D) -> dQuery(G,D) ---
                    RG_L1 = T.alloc_L1((BLOCK_M, BLOCK_N), input_dtype)
                    KG_L1 = T.alloc_L1((BLOCK_N, BLOCK_D), input_dtype)
                    DQ_L0C = T.alloc_L0C((BLOCK_M, BLOCK_D), calc_dtype)
                    DQ_fp32_ub = T.alloc_ub((BLOCK_M, BLOCK_D), calc_dtype)
                    DQ_ub = T.alloc_ub((BLOCK_M, BLOCK_D), input_dtype)

                    for m_idx in T.serial(m_num):
                        for d_idx in T.serial(d_num):
                            for k_idx in T.serial(n_num):
                                T.copy(reluGrad_ws[b_idx, s1_idx, n2, m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, k_idx * BLOCK_N:(k_idx + 1) * BLOCK_N], RG_L1)
                                T.copy(K_gather_ws[b_idx, s1_idx, n2, k_idx * BLOCK_N:(k_idx + 1) * BLOCK_N, d_idx * BLOCK_D:(d_idx + 1) * BLOCK_D], KG_L1)
                                T.gemm_v0(RG_L1, KG_L1, DQ_L0C, init=(k_idx == 0))
                            T.copy(DQ_L0C, DQ_fp32_ub)
                            T.tile.cast(DQ_ub, DQ_fp32_ub, "CAST_ROUND", BLOCK_M * BLOCK_D)
                            T.copy(DQ_ub, dQuery[b_idx, s1_idx, n1_start + m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, d_idx * BLOCK_D:(d_idx + 1) * BLOCK_D])

                    # --- Cube4: reluGradOut^T(TOPK,G) @ Q(G,D) -> scatterAdd(TOPK,D) ---
                    RGT_L1 = T.alloc_L1((BLOCK_N, BLOCK_M), input_dtype)
                    Q4_L1 = T.alloc_L1((BLOCK_M, BLOCK_D), input_dtype)
                    SA_L0C = T.alloc_L0C((BLOCK_N, BLOCK_D), calc_dtype)

                    for n_idx in T.serial(n_num):
                        for d_idx in T.serial(d_num):
                            for m_idx in T.serial(m_num):
                                T.copy(reluGrad_ws[b_idx, s1_idx, n2, m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N], RGT_L1)
                                T.copy(Query[b_idx, s1_idx, n1_start + m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M, d_idx * BLOCK_D:(d_idx + 1) * BLOCK_D], Q4_L1)
                                T.gemm_v0(RGT_L1, Q4_L1, SA_L0C, init=(m_idx == 0))
                            T.copy(SA_L0C, scatterAdd_ws[b_idx, s1_idx, n2, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N, d_idx * BLOCK_D:(d_idx + 1) * BLOCK_D])

                T.set_cross_flag("FIX", 2)

            # ========== Vec3: ScatterAdd -> dKey ==========
            with T.Scope("V"):
                T.wait_cross_flag(2)

                sa_ub = T.alloc_ub((BLOCK_N, BLOCK_D), calc_dtype)
                sa_fp16_ub = T.alloc_ub((BLOCK_N, BLOCK_D), input_dtype)
                idx_ub3 = T.alloc_ub(TOPK, 'int32')

                for n2 in T.serial(N2):
                    T.copy(Indices[b_idx, s1_idx, 0:TOPK], idx_ub3)
                    for n_idx in T.serial(n_num):
                        for d_idx in T.serial(d_num):
                            T.copy(scatterAdd_ws[b_idx, s1_idx, n2, n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N, d_idx * BLOCK_D:(d_idx + 1) * BLOCK_D], sa_ub)
                            T.tile.cast(sa_fp16_ub, sa_ub, "CAST_ROUND", BLOCK_N * BLOCK_D)
                            for t in T.serial(BLOCK_N):
                                global_t = n_idx * BLOCK_N + t
                                s2_pos = idx_ub3[global_t]
                                T.copy(sa_fp16_ub[t, 0:BLOCK_D], dKey[b_idx, s2_pos, n2, d_idx * BLOCK_D:(d_idx + 1) * BLOCK_D])

    return main


class LightningIndexerGradCV:
    """
    Lightning Indexer Grad 算子 (CV 分离架构)

    使用单 kernel + T.Scope("C")/T.Scope("V") 实现 CV 分离，
    通过 workspace 在 Cube/Vector 间传递中间结果。
    """

    def __init__(
        self,
        seqlen_q: int,
        seqlen_k: int,
        head_dim: int,
        topk: int,
        head_num_q: int,
        head_num_k: int,
        group_num: int,
        block_m: int = 64,
        block_n: int = 128,
        block_d: int = 128,
    ):
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.head_dim = head_dim
        self.topk = topk
        self.head_num_q = head_num_q
        self.head_num_k = head_num_k
        self.group_num = group_num

        assert head_num_q == head_num_k * group_num, \
            f"head_num_q({head_num_q}) != head_num_k({head_num_k}) * group_num({group_num})"
        assert head_dim == 128, f"head_dim must be 128, got {head_dim}"
        assert group_num == 64, f"group_num must be 64, got {group_num}"

        self.kernel = lightning_indexer_grad_kernel(
            B=1, S1=seqlen_q, S2=seqlen_k,
            N1=head_num_q, N2=head_num_k, G=group_num, D=head_dim,
            TOPK=topk,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
            input_dtype='float16',
            calc_dtype='float32',
        )
        print("TileLang CV-Separated 内核编译成功!")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        dy: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        head_num: int = None,
        layout: str = "BSND",
        sparse_mode: int = 3,
    ):
        """
        前向计算 (梯度反向传播)

        Args:
            query:   (B, S1, N1, D) FP16
            key:     (B, S2, N2, D) FP16
            dy:      (B, S1, topK)  FP16
            indices: (B, S1, topK)  INT32
            weights: (B, S1, N1)    FP16
        Returns:
            dQuery:   (B, S1, N1, D) FP16
            dKey:     (B, S2, N2, D) FP16
            dWeights: (B, S1, N1)    FP16
        """
        B = query.shape[0]
        S1 = self.seqlen_q
        S2 = self.seqlen_k
        N1 = self.head_num_q
        N2 = self.head_num_k
        G = self.group_num
        D = self.head_dim
        TOPK = self.topk
        device = query.device
        dtype = query.dtype

        d_query = torch.zeros(B, S1, N1, D, dtype=dtype, device=device)
        d_key = torch.zeros(B, S2, N2, D, dtype=dtype, device=device)
        d_weights = torch.zeros(B, S1, N1, dtype=dtype, device=device)

        k_gather_ws = torch.zeros(B, S1, N2, TOPK, D, dtype=dtype, device=device)
        relu_in_ws = torch.zeros(B, S1, N2, G, TOPK, dtype=torch.float32, device=device)
        relu_grad_ws = torch.zeros(B, S1, N2, G, TOPK, dtype=dtype, device=device)
        scatter_add_ws = torch.zeros(B, S1, N2, TOPK, D, dtype=torch.float32, device=device)

        self.kernel(
            query, key, dy, weights, indices,
            d_query, d_key, d_weights,
            k_gather_ws, relu_in_ws, relu_grad_ws, scatter_add_ws,
        )

        return d_query, d_key, d_weights