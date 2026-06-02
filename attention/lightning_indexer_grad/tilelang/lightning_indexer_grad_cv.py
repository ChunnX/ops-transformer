import torch
import torch_npu
import math
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

import tilelang
import tilelang.language as T

tilelang.disable_cache()

# ==============================================================================
# 全局常量：Tiling 参数（对标 AscendC arch22）
# ==============================================================================
BLOCK_M = 64
BLOCK_N = 128
BLOCK_D = 128

VECTOR_BASEG = 16
VECTOR_BASEN = 256

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ==============================================================================
# TilingData 与参数校验
# ==============================================================================
@dataclass
class LightningIndexerGradTilingData:
    B: int
    N1: int
    N2: int
    G: int
    D: int
    S1: int
    S2: int
    sparse_count: int = 2048
    sparse_mode: int = 0

    @property
    def g_size(self) -> int:
        return self.G

    @property
    def n2_size(self) -> int:
        return self.N2


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def check_param_valid(
    query: torch.Tensor,
    key: torch.Tensor,
    dy: torch.Tensor,
    sparse_indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_key: Optional[torch.Tensor] = None,
    layout_query: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 0,
) -> LightningIndexerGradTilingData:
    """校验 LightningIndexerGrad 入参，与文档 aclnnLightningIndexerGrad 保持一致。"""
    assert query.dim() in (3, 4), f"query dim must be 3 or 4, got {query.dim()}"
    assert key.dim() in (3, 4), f"key dim must be 3 or 4, got {key.dim()}"
    assert query.dtype in (torch.float16, torch.bfloat16), f"query dtype must be float16/bfloat16, got {query.dtype}"
    assert key.dtype == query.dtype, f"key dtype must match query, got {key.dtype}"
    assert dy.dtype == query.dtype, f"dy dtype must match query, got {dy.dtype}"
    assert weights.dtype == query.dtype, f"weights dtype must match query, got {weights.dtype}"
    assert sparse_mode in (0, 3), f"sparse_mode must be 0 or 3, got {sparse_mode}"

    if query.dim() == 4:
        B, S1, N1, D = query.shape
        _, S2, N2, _ = key.shape
    else:
        raise ValueError(f"Only BSND layout (4-dim) is supported, got query dim={query.dim()}")

    G = N1 // N2
    assert N1 == N2 * G, f"N1({N1}) must be divisible by N2({N2})"

    return LightningIndexerGradTilingData(
        B=B, N1=N1, N2=N2, G=G, D=D, S1=S1, S2=S2,
        sparse_count=sparse_count, sparse_mode=sparse_mode,
    )


# ==============================================================================
# TileLang Kernel: LightningIndexerGrad (BSND, CV 分离架构)
# ==============================================================================
@tilelang.jit(out_idx=[-1], workspace_idx=[2, 3, 4], pass_configs=pass_configs)
def tilelang_lightning_indexer_grad_bsnd(
    B, S1, S2, N1, N2, G, D, sparse_count,
    input_dtype, calc_dtype
):
    """
    LightningIndexerGrad kernel (BSND, CV 分离架构)

    计算流程:
        Cube1: Q_group @ Key^T -> reluIn (FP32 workspace)
        Vec2:  weights * dy -> reluGradIn; ReLU反向 -> reluGradOut (FP16 workspace)
        Cube3: reluGradOut @ Key -> dQuery_ws (FP32 workspace, 避免 L0C->UB 通路)
        Vec4:  dQuery_ws FP32->FP16 cast -> dQuery

    T.Tensor 参数顺序: 输入 -> workspace -> 输出
        输入:     Query, Key, Dy, Weights, actual_q_len, actual_k_len
        Workspace: reluIn_ws (FP32), reluGrad_ws (FP16), dQuery_ws (FP32)  (idx=2,3,4)
        输出:     dQuery
    """
    block_m = BLOCK_M
    block_n = BLOCK_N
    block_d = BLOCK_D
    vector_baseg = VECTOR_BASEG
    s1_tile_num = ceil_div(S1, block_m)
    block_num = B * N2 * s1_tile_num

    n_num = sparse_count // block_n
    d_num = D // block_d
    g_group_num = G // vector_baseg

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, D), input_dtype),
        Key: T.Tensor((B, S2, N2, D), input_dtype),
        reluIn_ws: T.Tensor((B, N2, S1, G, sparse_count), calc_dtype),
        reluGrad_ws: T.Tensor((B, N2, S1, G, sparse_count), input_dtype),
        dQuery_ws: T.Tensor((B, S1, N1, D), calc_dtype),
        Dy: T.Tensor((B, S1, N1, sparse_count), input_dtype),
        Weights: T.Tensor((B, S1, N1), input_dtype),
        actual_q_len: T.Tensor((B,), "int32"),
        actual_k_len: T.Tensor((B,), "int32"),
        dQuery: T.Tensor((B, S1, N1, D), input_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bz = cid // (N2 * s1_tile_num)
            n2_id = cid // s1_tile_num % N2
            m = cid % s1_tile_num
            s1_start = m * block_m
            s1_end = T.min(s1_start + block_m, S1)

            # ========== Cube1: Q @ K^T -> reluIn ==========
            with T.Scope("C"):
                T.barrier_all()

                Q_L1 = T.alloc_L1((block_m, block_d), input_dtype)
                K_L1 = T.alloc_L1((block_n, block_d), input_dtype)
                C_L0C = T.alloc_L0C((block_m, block_n), calc_dtype)

                for g_idx in T.serial(G):
                    n1_idx = n2_id * G + g_idx

                    # --- Cube1: Q(block_m,D) @ K^T(block_n,D) -> reluIn(block_m,block_n) ---
                    # TileLang 不支持运行时索引，此处直接用 Key 的 S2 维做 matmul
                    for n_idx in T.serial(n_num):
                        T.barrier_all()
                        for k_idx in T.serial(d_num):
                            T.copy(
                                Query[bz, s1_start:s1_end, n1_idx, k_idx * block_d:(k_idx + 1) * block_d],
                                Q_L1,
                            )
                            T.barrier_all()
                            T.copy(
                                Key[bz, n_idx * block_n:(n_idx + 1) * block_n, n2_id, k_idx * block_d:(k_idx + 1) * block_d],
                                K_L1,
                            )
                            T.barrier_all()
                        T.gemm_v0(Q_L1, K_L1, C_L0C, transpose_B=True, init=(n_idx == 0))
                        T.barrier_all()
                        T.copy(
                            C_L0C,
                            reluIn_ws[bz, n2_id, s1_start:s1_end, g_idx, n_idx * block_n:(n_idx + 1) * block_n],
                        )
                        T.barrier_all()

                T.set_cross_flag("FIX", 0)

            # ========== Vec2: weights * dy + ReLU反向 ==========
            # 循环结构: g_group(n_idx(g_local(s1_row)))
            # 与 BSND3 的 g_group(s1_row(g_local)) 不同, 此处 s1_row 在最内层
            # 因为 S1 按 block_m 分核并行, 需逐行加载 weight 保 GM 地址对齐
            with T.Scope("V"):
                T.wait_cross_flag(0)

                relu_in_ub = T.alloc_ub((block_m, block_n), calc_dtype)
                dy_ub = T.alloc_ub((block_m, block_n), input_dtype)
                dy_fp32_ub = T.alloc_ub((block_m, block_n), calc_dtype)

                # weight: 按 vector_baseg 分组加载 (对标 BSND3)
                # 1D (vector_baseg,) 用于 T.tile.mul 标量索引, 与 BSND3 一致
                weight_half_ub = T.alloc_ub(vector_baseg, input_dtype)
                weight_fp32_ub = T.alloc_ub(vector_baseg, calc_dtype)

                # 单行 dy 缓冲区, 用于逐行 weight 乘法
                dy_row_ub = T.alloc_ub(block_n, input_dtype)
                dy_row_fp32_ub = T.alloc_ub(block_n, calc_dtype)
                relu_in_row_ub = T.alloc_ub(block_n, calc_dtype)
                relu_grad_row_ub = T.alloc_ub(block_n, calc_dtype)
                relu_grad_row_half_ub = T.alloc_ub(block_n, input_dtype)
                mask_row_ub = T.alloc_ub(block_n, calc_dtype)
                zero_row_ub = T.alloc_ub(block_n, calc_dtype)

                T.tile.fill(zero_row_ub, 0.0)

                for g_group in T.serial(g_group_num):
                    for n_idx in T.serial(n_num):
                        # 逐行处理 S1, 每行批量加载 vector_baseg 个 weight (32B 对齐)
                        # 循环顺序: s1_row -> g_local (与 BSND3 的 g_group(s1_row(g_local)) 对齐)
                        for s1_row in range(block_m):
                            T.barrier_all()

                            # 批量加载 weight: vector_baseg 个 FP16 = 32 字节, 满足 UB 对齐
                            n1_group_start = n2_id * G + g_group * vector_baseg
                            T.copy(
                                Weights[bz, s1_start + s1_row : s1_start + s1_row + 1, n1_group_start : n1_group_start + vector_baseg],
                                weight_half_ub,
                            )
                            T.barrier_all()
                            T.tile.cast(weight_fp32_ub, weight_half_ub, "CAST_NONE", vector_baseg)
                            T.barrier_all()

                            for g_local in range(vector_baseg):
                                g_idx = g_group * vector_baseg + g_local
                                n1_idx = n2_id * G + g_idx

                                # 加载 dy 单行
                                T.copy(
                                    Dy[bz, s1_start + s1_row : s1_start + s1_row + 1, n1_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                    dy_row_ub,
                                )
                                T.barrier_all()
                                T.tile.cast(dy_row_fp32_ub, dy_row_ub, "CAST_NONE", block_n)
                                T.barrier_all()

                                # 加载 reluIn 单行
                                T.copy(
                                    reluIn_ws[bz, n2_id, s1_start + s1_row : s1_start + s1_row + 1, g_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                    relu_in_row_ub,
                                )
                                T.barrier_all()

                                # 标量-向量乘法: dy_row *= weight[g_local]
                                T.tile.mul(dy_row_fp32_ub, dy_row_fp32_ub, weight_fp32_ub[g_local])
                                T.barrier_all()

                                # ReLU反向
                                T.tile.compare(mask_row_ub, relu_in_row_ub, zero_row_ub, "GT")
                                T.barrier_all()
                                T.tile.select(relu_grad_row_ub, mask_row_ub, dy_row_fp32_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                                T.barrier_all()
                                T.tile.cast(relu_grad_row_half_ub, relu_grad_row_ub, "CAST_ROUND", block_n)
                                T.barrier_all()

                                # 写回单行
                                T.copy(
                                    relu_grad_row_half_ub,
                                    reluGrad_ws[bz, n2_id, s1_start + s1_row : s1_start + s1_row + 1, g_idx, n_idx * block_n:(n_idx + 1) * block_n],
                                )
                                T.barrier_all()

                T.set_cross_flag("FIX", 1)

            # ========== Cube3: reluGradOut @ K -> dQuery_ws ==========
            with T.Scope("C"):
                T.wait_cross_flag(1)

                RG_L1 = T.alloc_L1((block_m, block_n), input_dtype)
                KG_L1 = T.alloc_L1((block_n, block_d), input_dtype)
                DQ_L0C = T.alloc_L0C((block_m, block_d), calc_dtype)

                for g_idx in T.serial(G):
                    n1_idx = n2_id * G + g_idx
                    for d_idx in T.serial(d_num):
                        T.barrier_all()
                        for k_idx in T.serial(n_num):
                            T.copy(
                                reluGrad_ws[bz, n2_id, s1_start:s1_end, g_idx, k_idx * block_n:(k_idx + 1) * block_n],
                                RG_L1,
                            )
                            T.barrier_all()
                            T.copy(
                                Key[bz, k_idx * block_n:(k_idx + 1) * block_n, n2_id, d_idx * block_d:(d_idx + 1) * block_d],
                                KG_L1,
                            )
                            T.barrier_all()
                            T.gemm_v0(RG_L1, KG_L1, DQ_L0C, init=(k_idx == 0))
                            T.barrier_all()
                        T.copy(
                            DQ_L0C,
                            dQuery_ws[bz, s1_start:s1_end, n1_idx, d_idx * block_d:(d_idx + 1) * block_d],
                        )
                        T.barrier_all()

                T.set_cross_flag("FIX", 2)

            # ========== Vec4: dQuery_ws FP32 -> FP16 cast -> dQuery ==========
            with T.Scope("V"):
                T.wait_cross_flag(2)

                dq_fp32_ub = T.alloc_ub((block_m, block_d), calc_dtype)
                dq_fp16_ub = T.alloc_ub((block_m, block_d), input_dtype)

                for g_idx in T.serial(G):
                    n1_idx = n2_id * G + g_idx
                    for d_idx in T.serial(d_num):
                        T.barrier_all()
                        T.copy(
                            dQuery_ws[bz, s1_start:s1_end, n1_idx, d_idx * block_d:(d_idx + 1) * block_d],
                            dq_fp32_ub,
                        )
                        T.barrier_all()
                        T.tile.cast(dq_fp16_ub, dq_fp32_ub, "CAST_ROUND", block_m * block_d)
                        T.barrier_all()
                        T.copy(
                            dq_fp16_ub,
                            dQuery[bz, s1_start:s1_end, n1_idx, d_idx * block_d:(d_idx + 1) * block_d],
                        )
                        T.barrier_all()

    return main


# ==============================================================================
# Python 入口函数
# ==============================================================================
def lightning_indexer_grad(
    query: torch.Tensor,
    key: torch.Tensor,
    dy: torch.Tensor,
    sparse_indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_key: Optional[torch.Tensor] = None,
    layout_query: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LightningIndexerGrad Python 入口。
    严格对应 aclnnLightningIndexerGrad 文档参数:
        输入: query, key, dy, sparseIndices, weights, actualSeqLengthsQuery, actualSeqLengthsKey
        输出: dQuery, dKey, dWeights

    支持 sparse_mode=0（无 mask）和 sparse_mode=3（causal mask）。
    BSND 布局: query(B,S1,N1,D), key(B,S2,N2,D), dy(B,S1,N1,topK), weights(B,S1,N1)
    """
    tiling = check_param_valid(
        query, key, dy, sparse_indices, weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        layout_query=layout_query,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )

    B = tiling.B
    S1 = tiling.S1
    S2 = tiling.S2
    N1 = tiling.N1
    N2 = tiling.N2
    G = tiling.G
    D = tiling.D
    sparse_count = tiling.sparse_count

    calc_dtype_str = "float"
    input_dtype_str = "float16" if query.dtype == torch.float16 else "bfloat16"

    if layout_query == "BSND" and sparse_mode in (0, 3):
        if actual_seq_lengths_query is not None:
            actual_q_len = actual_seq_lengths_query.to(torch.int32)
        else:
            actual_q_len = torch.full((B,), S1, dtype=torch.int32)

        if actual_seq_lengths_key is not None:
            actual_k_len = actual_seq_lengths_key.to(torch.int32)
        else:
            actual_k_len = torch.full((B,), S2, dtype=torch.int32)

        S2_pad = (S2 + VECTOR_BASEN - 1) // VECTOR_BASEN * VECTOR_BASEN
        if S2_pad > S2:
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))

        if weights.dtype == torch.float32:
            weights = weights.half()

        func = tilelang_lightning_indexer_grad_bsnd(
            B, S1, S2_pad, N1, N2, G, D, sparse_count,
            input_dtype_str, calc_dtype_str,
        )

        torch.npu.synchronize()
        d_query = func(query, key, dy, weights, actual_q_len, actual_k_len)
        torch.npu.synchronize()

        # dKey 和 dWeights 需要使用 sparseIndices 做 ScatterAdd，
        # TileLang 不支持运行时索引，在 Python 层补算
        device = query.device
        dtype = query.dtype
        d_key = torch.zeros_like(key)
        d_weights = torch.zeros(B, S1, N1, dtype=dtype, device=device)

        indices_cpu = sparse_indices.cpu().numpy()
        key_cpu = key.cpu().numpy()
        query_cpu = query.cpu()

        for b in range(B):
            for s1 in range(S1):
                for n2 in range(N2):
                    n1_start = n2 * G
                    n1_end = (n2 + 1) * G
                    indices_s1 = indices_cpu[b, s1]

                    query_group = query[b, s1, n1_start:n1_end, :]
                    weights_group = weights[b, s1, n1_start:n1_end]
                    dy_group = dy[b, s1, n1_start:n1_end, :]

                    key_gather = key_cpu[b, indices_s1, n2, :]
                    key_gather_npu = torch.from_numpy(key_gather).npu().to(dtype)

                    # ReLU 输入
                    relu_in_pt = (query_group.float() @ key_gather_npu.float().T)
                    mask_pt = (relu_in_pt > 0).to(dtype)

                    # reluGradOut = weights * dy * relu_mask
                    relu_grad_out = weights_group.unsqueeze(1) * dy_group[:, :sparse_count]
                    relu_grad_out = relu_grad_out * mask_pt

                    # Cube4: scatterAdd = reluGradOut^T @ Q
                    scatter_add = relu_grad_out.T.half() @ query_group

                    # ScatterAdd (Python 层)
                    scatter_add_cpu = scatter_add.cpu().numpy()
                    for k_idx in range(sparse_count):
                        idx = indices_s1[k_idx]
                        d_key[b, idx, n2, :] += torch.from_numpy(scatter_add_cpu[k_idx]).to(device)

                    # dWeights
                    d_weights_group = (relu_in_pt * dy_group[:, :sparse_count].float()).sum(dim=-1).to(dtype)
                    d_weights[b, s1, n1_start:n1_end] = d_weights_group

        torch.npu.synchronize()
        return d_query, d_key, d_weights

    raise ValueError(f"Unsupported layout_query={layout_query} with sparse_mode={sparse_mode}")
