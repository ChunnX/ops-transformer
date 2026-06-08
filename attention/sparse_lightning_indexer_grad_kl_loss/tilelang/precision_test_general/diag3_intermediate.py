#!/usr/bin/env python3
"""诊断脚本 - 修改kernel输出P_reduced/SoftmaxI/ReduceSumY，与Python参考逐层对比"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sparse_lightning_indexer_grad_kl_loss import (
    _preprocess_gather,
    check_param_valid,
)
import tilelang
import tilelang.language as T

tilelang.disable_cache()

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
VECTOR_BASEG = 16
DQUERY = 512
DQUERY_INDEX = 128

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

def ceil_div(a, b):
    return (a + b - 1) // b

# Modified kernel: output P_reduced, ReduceSumY, SoftmaxI as additional outputs
@tilelang.jit(
    out_idx=[-8, -7, -6, -5, -4, -3, -2],
    workspace_idx=[10, 11, 19],
    pass_configs=pass_configs,
)
def diag_kernel(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
    d_rope, input_dtype, calc_dtype, scale_value
):
    s1_block = BLOCK_M
    n_block = BLOCK_N
    k_block = BLOCK_K
    s1_tile_num = ceil_div(S1, s1_block)
    block_num = B * s1_tile_num
    n_num = K // n_block
    d_num_query = DQuery // k_block
    d_num_rope = ceil_div(d_rope, k_block)
    d_num_total = d_num_query + d_num_rope
    d_rope_last = d_rope - (d_num_rope - 1) * k_block
    d_num_query_index = DQueryIndex // k_block
    vector_baseg = Nidx1 if Nidx1 <= VECTOR_BASEG else VECTOR_BASEG
    g_num = Nidx1 // vector_baseg

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),
        QueryRope: T.Tensor((B, S1, N1, d_rope), input_dtype),
        GatherKey: T.Tensor((B, S1, K, DQuery), input_dtype),
        GatherKeyRope: T.Tensor((B, S1, K, d_rope), input_dtype),
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        GatherKeyIndex: T.Tensor((B, S1, K, DQueryIndex), input_dtype),
        GatherKeyIndexT: T.Tensor((B, S1, DQueryIndex, K), input_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype),
        SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype),
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
        dQueryIndex_ws: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype),
        PReduced: T.Tensor((B, S1, K), calc_dtype),
        ReduceSumY: T.Tensor((B, S1, K), calc_dtype),
        SoftmaxI: T.Tensor((B, S1, K), calc_dtype),
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        dWeight: T.Tensor((B, S1, Nidx1), input_dtype),
        reluGradRes: T.Tensor((B, S1, Nidx1, K), input_dtype),
        LossOut: T.Tensor((B, S1, 16), calc_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bz = cid // s1_tile_num
            m = cid % s1_tile_num
            s1_start = m * s1_block
            s1_end = T.min(s1_start + s1_block, S1)

            with T.Scope("C"):
                T.barrier_all()
                Q_L1 = T.alloc_L1((N1, k_block), input_dtype)
                GK_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C_L0C = T.alloc_L0C((N1, n_block), calc_dtype)
                for s1_row in T.serial(s1_start, s1_end):
                    for n_idx in T.serial(n_num):
                        T.barrier_all()
                        for d_idx in T.serial(d_num_total):
                            if d_idx < d_num_query:
                                T.copy(Query[bz, s1_row, 0:N1, d_idx*k_block:(d_idx+1)*k_block], Q_L1)
                            else:
                                rope_d = d_idx - d_num_query
                                if d_idx == d_num_total - 1:
                                    T.copy(QueryRope[bz, s1_row, 0:N1, rope_d*k_block:rope_d*k_block+d_rope_last], Q_L1)
                                else:
                                    T.copy(QueryRope[bz, s1_row, 0:N1, rope_d*k_block:(rope_d+1)*k_block], Q_L1)
                            T.barrier_all()
                            if d_idx < d_num_query:
                                T.copy(GatherKey[bz, s1_row, n_idx*n_block:(n_idx+1)*n_block, d_idx*k_block:(d_idx+1)*k_block], GK_L1)
                            else:
                                rope_d = d_idx - d_num_query
                                if d_idx == d_num_total - 1:
                                    T.copy(GatherKeyRope[bz, s1_row, n_idx*n_block:(n_idx+1)*n_block, rope_d*k_block:rope_d*k_block+d_rope_last], GK_L1)
                                else:
                                    T.copy(GatherKeyRope[bz, s1_row, n_idx*n_block:(n_idx+1)*n_block, rope_d*k_block:(rope_d+1)*k_block], GK_L1)
                            T.barrier_all()
                            T.gemm_v0(Q_L1, GK_L1, C_L0C, transpose_B=True, init=(d_idx == 0))
                            T.barrier_all()
                        T.copy(C_L0C, bmm1Res[bz, s1_row, 0:N1, n_idx*n_block:(n_idx+1)*n_block])
                        T.barrier_all()

                QI_L1 = T.alloc_L1((Nidx1, k_block), input_dtype)
                GKI_L1 = T.alloc_L1((n_block, k_block), input_dtype)
                C2_L0C = T.alloc_L0C((Nidx1, n_block), calc_dtype)
                for s1_row in T.serial(s1_start, s1_end):
                    for n_idx in T.serial(n_num):
                        T.barrier_all()
                        for k_idx in T.serial(d_num_query_index):
                            T.copy(QueryIndex[bz, s1_row, 0:Nidx1, k_idx*k_block:(k_idx+1)*k_block], QI_L1)
                            T.barrier_all()
                            T.copy(GatherKeyIndex[bz, s1_row, n_idx*n_block:(n_idx+1)*n_block, k_idx*k_block:(k_idx+1)*k_block], GKI_L1)
                            T.barrier_all()
                            T.gemm_v0(QI_L1, GKI_L1, C2_L0C, transpose_B=True, init=(k_idx == 0))
                            T.barrier_all()
                        T.copy(C2_L0C, bmm2Res[bz, s1_row, 0:Nidx1, n_idx*n_block:(n_idx+1)*n_block], enable_relu=True)
                        T.barrier_all()
                T.set_cross_flag("FIX", 1)

            with T.Scope("V"):
                T.wait_cross_flag(1)
                bmm1_ub = T.alloc_ub((N1, n_block), calc_dtype)
                bmm2_ub = T.alloc_ub((Nidx1, n_block), calc_dtype)
                weight_ub = T.alloc_ub(vector_baseg, calc_dtype)
                weight_half_ub = T.alloc_ub(vector_baseg, input_dtype)
                p_reduced_tmp_ub = T.alloc_ub(n_block, calc_dtype)
                rsy_tmp_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                rsy_reduce_ub = T.alloc_ub(n_block, calc_dtype)
                diff_ub = T.alloc_ub(n_block, calc_dtype)
                p_reduced_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                softmax_i_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                dw_accum_ub = T.alloc_ub(vector_baseg, calc_dtype)
                dw_chunk_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                dw_reduce_ub = T.alloc_ub(vector_baseg, calc_dtype)
                rg_ub = T.alloc_ub((vector_baseg, n_block), calc_dtype)
                rg_tmp_ub = T.alloc_ub(n_block, calc_dtype)
                rg_half_ub = T.alloc_ub((vector_baseg, n_block), input_dtype)
                dw_half_ub = T.alloc_ub(vector_baseg, input_dtype)
                sm_max_ub = T.alloc_ub(G, calc_dtype)
                sm_sum_ub = T.alloc_ub(G, calc_dtype)
                rsy_chunk_ub = T.alloc_ub(n_block, calc_dtype)
                rsy_2d_ub = T.alloc_ub((16, n_block), calc_dtype)
                rsy_max_ub = T.alloc_ub(16, calc_dtype)
                rsy_sum_ub = T.alloc_ub(16, calc_dtype)
                rsy_chunk_max_ub = T.alloc_ub(16, calc_dtype)
                rsy_chunk_sum_ub = T.alloc_ub(16, calc_dtype)
                max_pair_ub = T.alloc_ub((2, 16), calc_dtype)
                mask_ub = T.alloc_ub(n_block, calc_dtype)
                ones_ub = T.alloc_ub(n_block, calc_dtype)
                eps_sm_ub = T.alloc_ub(G, calc_dtype)

                for s1_id in T.serial(s1_start, s1_end):
                    T.barrier_all()
                    T.tile.fill(ones_ub, 1.0)
                    T.barrier_all()
                    T.copy(SoftmaxMax[bz, 0, s1_id, 0:G], sm_max_ub)
                    T.barrier_all()
                    T.copy(SoftmaxSum[bz, 0, s1_id, 0:G], sm_sum_ub)
                    T.barrier_all()
                    T.tile.abs(sm_sum_ub, sm_sum_ub)
                    T.barrier_all()
                    T.tile.fill(eps_sm_ub, 1e-8)
                    T.barrier_all()
                    T.tile.max(sm_sum_ub, sm_sum_ub, eps_sm_ub)
                    T.barrier_all()

                    for n_idx in T.serial(n_num):
                        T.barrier_all()
                        T.copy(bmm1Res[bz, s1_id, 0:N1, n_idx*n_block:(n_idx+1)*n_block], bmm1_ub)
                        T.barrier_all()
                        T.tile.fill(p_reduced_tmp_ub, 0)
                        T.barrier_all()
                        for g_id in range(G):
                            T.barrier_all()
                            T.tile.mul(bmm1_ub[g_id, :], bmm1_ub[g_id, :], scale_value)
                            T.barrier_all()
                            T.tile.sub(bmm1_ub[g_id, :], bmm1_ub[g_id, :], sm_max_ub[g_id])
                            T.barrier_all()
                            T.tile.exp(bmm1_ub[g_id, :], bmm1_ub[g_id, :])
                            T.barrier_all()
                            T.tile.div(bmm1_ub[g_id, :], bmm1_ub[g_id, :], sm_sum_ub[g_id])
                            T.barrier_all()
                            T.tile.add(p_reduced_tmp_ub, p_reduced_tmp_ub, bmm1_ub[g_id, :])
                            T.barrier_all()
                        T.tile.mul(p_reduced_tmp_ub, p_reduced_tmp_ub, 1.0 / G)
                        T.barrier_all()
                        T.copy(p_reduced_tmp_ub, PReduced[bz, s1_id, n_idx*n_block:(n_idx+1)*n_block])
                        T.barrier_all()

                        T.copy(bmm2Res[bz, s1_id, 0:Nidx1, n_idx*n_block:(n_idx+1)*n_block], bmm2_ub)
                        T.barrier_all()
                        T.tile.fill(rsy_reduce_ub, 0)
                        T.barrier_all()
                        for g_id in range(g_num):
                            T.barrier_all()
                            T.copy(Weights[bz, s1_id, g_id*vector_baseg:(g_id+1)*vector_baseg], weight_half_ub)
                            T.barrier_all()
                            T.tile.cast(weight_ub, weight_half_ub, "CAST_NONE", vector_baseg)
                            T.barrier_all()
                            T.tile.fill(rsy_tmp_ub, 0)
                            T.barrier_all()
                            for i in range(vector_baseg):
                                T.barrier_all()
                                T.tile.mul(rsy_tmp_ub[i, :], bmm2_ub[g_id*vector_baseg+i, :], weight_ub[i])
                                T.barrier_all()
                            T.barrier_all()
                            for i in range(vector_baseg):
                                T.barrier_all()
                                T.tile.add(rsy_reduce_ub, rsy_reduce_ub, rsy_tmp_ub[i, :])
                                T.barrier_all()
                        T.barrier_all()
                        T.copy(rsy_reduce_ub, ReduceSumY[bz, s1_id, n_idx*n_block:(n_idx+1)*n_block])
                        T.barrier_all()

                    T.tile.fill(rsy_max_ub, -1e30)
                    T.barrier_all()
                    for n_idx in T.serial(n_num):
                        T.barrier_all()
                        T.copy(ReduceSumY[bz, s1_id, n_idx*n_block:(n_idx+1)*n_block], rsy_chunk_ub)
                        T.barrier_all()
                        for i in range(16):
                            T.barrier_all()
                            T.tile.mul(rsy_2d_ub[i, :], rsy_chunk_ub, 1.0)
                            T.barrier_all()
                        T.reduce_max(rsy_2d_ub, rsy_chunk_max_ub, 1)
                        T.barrier_all()
                        T.tile.mul(max_pair_ub[0, :], rsy_max_ub, 1.0)
                        T.barrier_all()
                        T.tile.mul(max_pair_ub[1, :], rsy_chunk_max_ub, 1.0)
                        T.barrier_all()
                        T.reduce_max(max_pair_ub, rsy_max_ub, 0)
                        T.barrier_all()

                    T.tile.fill(rsy_sum_ub, 0)
                    T.barrier_all()
                    for n_idx in T.serial(n_num):
                        T.barrier_all()
                        T.copy(ReduceSumY[bz, s1_id, n_idx*n_block:(n_idx+1)*n_block], rsy_chunk_ub)
                        T.barrier_all()
                        T.tile.sub(rsy_chunk_ub, rsy_chunk_ub, rsy_max_ub[0])
                        T.barrier_all()
                        T.tile.exp(rsy_chunk_ub, rsy_chunk_ub)
                        T.barrier_all()
                        T.copy(rsy_chunk_ub, SoftmaxI[bz, s1_id, n_idx*n_block:(n_idx+1)*n_block])
                        T.barrier_all()
                        for i in range(16):
                            T.barrier_all()
                            T.tile.mul(rsy_2d_ub[i, :], rsy_chunk_ub, 1.0)
                            T.barrier_all()
                        T.reduce_sum(rsy_2d_ub, rsy_chunk_sum_ub, 1)
                        T.barrier_all()
                        T.tile.add(rsy_sum_ub, rsy_sum_ub, rsy_chunk_sum_ub)
                        T.barrier_all()

                    rsy_sum_eps_ub = T.alloc_ub(16, calc_dtype)
                    T.tile.abs(rsy_sum_ub, rsy_sum_ub)
                    T.barrier_all()
                    T.tile.fill(rsy_sum_eps_ub, 1e-8)
                    T.barrier_all()
                    T.tile.max(rsy_sum_ub, rsy_sum_ub, rsy_sum_eps_ub)
                    T.barrier_all()

                    for n_idx in T.serial(n_num):
                        T.barrier_all()
                        T.copy(SoftmaxI[bz, s1_id, n_idx*n_block:(n_idx+1)*n_block], rsy_chunk_ub)
                        T.barrier_all()
                        T.tile.div(rsy_chunk_ub, rsy_chunk_ub, rsy_sum_ub[0])
                        T.barrier_all()
                        T.copy(rsy_chunk_ub, SoftmaxI[bz, s1_id, n_idx*n_block:(n_idx+1)*n_block])
                        T.barrier_all()

                    # Skip dWeight/reluGrad/Loss computation for diagnosis
                    T.tile.fill(dWeight[bz, s1_id, 0:Nidx1], 0.0)
                    T.barrier_all()
                    T.tile.fill(reluGradRes[bz, s1_id, 0:Nidx1, 0:n_block], 0.0)
                    T.barrier_all()
                    T.tile.fill(LossOut[bz, s1_id, 0:16], 0.0)
                    T.barrier_all()

                T.set_cross_flag("V", 2)

            with T.Scope("C"):
                T.wait_cross_flag(2)
                T.set_cross_flag("FIX", 3)

            with T.Scope("V"):
                T.wait_cross_flag(3)
                T.barrier_all()

    return main


# === Run diagnosis ===
B, S1, S2 = 1, 1, 2048
N1, N2, Nidx1 = 128, 1, 64
D, DIndex, DR, K = 512, 128, 64, 2048
G = N1 // N2
scale = 1.0 / math.sqrt(D)

torch.manual_seed(42)
output_dtype = torch.float16
device = torch.device('npu')

q = torch.randn(B, S1, N1, D, dtype=output_dtype, device=device)
k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=device)
q_index = torch.randn(B, S1, Nidx1, DIndex, dtype=output_dtype, device=device)
k_index = torch.randn(B, S2, N2, DIndex, dtype=output_dtype, device=device)
q_rope = torch.randn(B, S1, N1, DR, dtype=output_dtype, device=device)
k_rope = torch.randn(B, S2, N2, DR, dtype=output_dtype, device=device)
weights = torch.randn(B, S1, Nidx1, dtype=output_dtype, device=device)
a, b, kk = -0.05, 0.05, 3.0
scale_w = (b - a) / (2 * kk)
shift_w = (a + b) / 2
weights = weights * scale_w + shift_w
sparse_indices = torch.zeros(B, S1, N2, K).to(torch.int32).npu()
for s1Idx in range(S1):
    s2RealSize = (int)(S2 - S1 + s1Idx + 1)
    if s2RealSize <= 0:
        s2RealSize = S2
    s2RealLen = min(s2RealSize, K)
    sparse_indices[:, s1Idx, 0, 0:s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
    sparse_indices[:, s1Idx, 0, s2RealLen:K] = -1

# Compute softmax with rope
with torch.no_grad():
    q_full = torch.cat([q.float(), q_rope.float()], dim=-1)
    k_full = torch.cat([k.float().squeeze(2), k_rope.float().squeeze(2)], dim=-1)
    bmm1_full = torch.matmul(q_full, k_full.transpose(-1, -2).unsqueeze(1))
    bmm1_scaled = bmm1_full * scale
    sm_max_val = bmm1_scaled.max(dim=-1).values
    sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)
    softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().to(device=device)
    softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().to(device=device)

# Gather
S2_pad = (S2 + 256 - 1) // 256 * 256
k_p = torch.nn.functional.pad(k, (0,0,0,0,0,S2_pad-S2,0,0)) if S2_pad > S2 else k
ki_p = torch.nn.functional.pad(k_index, (0,0,0,0,0,S2_pad-S2,0,0)) if S2_pad > S2 else k_index
gk, gki = _preprocess_gather(k_p, ki_p, sparse_indices, B, S1, S2_pad, N2, 1, K, D, DIndex)
gather_key_rope = k_rope[torch.arange(B, device=device).view(B, 1, 1),
                          sparse_indices[:, :, 0, :].clamp(min=0), 0, :].contiguous()
gather_key_index_t = gki.permute(0, 1, 3, 2).contiguous()

# Python reference
bmm1_noscale = q.float() @ gk.float().transpose(-2, -1)
bmm2_ref = torch.relu(q_index.float() @ gki.float().transpose(-2, -1))
bmm1_scaled_ref = bmm1_noscale * scale
sm_max_sq = softmax_max.float().squeeze(1)
sm_sum_sq = softmax_sum.float().squeeze(1)
P_ref = torch.exp(bmm1_scaled_ref - sm_max_sq.unsqueeze(-1)) / sm_sum_sq.unsqueeze(-1)
P_reduced_ref = P_ref.mean(dim=2)
w = weights.float().unsqueeze(-1)
reduce_sum_y_ref = (bmm2_ref * w).sum(dim=2)
rsy_max_ref = reduce_sum_y_ref.max(dim=-1, keepdim=True).values
rsy_exp_ref = torch.exp(reduce_sum_y_ref - rsy_max_ref)
rsy_sum_ref = rsy_exp_ref.sum(dim=-1, keepdim=True)
softmax_i_ref = rsy_exp_ref / rsy_sum_ref

# Run diag kernel
func = diag_kernel(B, S1, S2_pad, N1, N2, G, Nidx1, 1, D, DIndex, K, DR, "float16", "float", scale)

bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=device)
bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=device)
dqi_ws = torch.zeros(B, S1, Nidx1, DIndex, dtype=torch.float32, device=device)
p_reduced_out = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
rsy_out = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
si_out = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
dqi_out = torch.zeros(B, S1, Nidx1, DIndex, dtype=output_dtype, device=device)
dw_out = torch.zeros(B, S1, Nidx1, dtype=output_dtype, device=device)
rg_out = torch.zeros(B, S1, Nidx1, K, dtype=output_dtype, device=device)
loss_out = torch.zeros(B, S1, 16, dtype=torch.float32, device=device)

torch.npu.synchronize()
results = func(
    q, q_rope, gk, gather_key_rope,
    q_index, gki, gather_key_index_t,
    weights, softmax_max, softmax_sum,
    bmm1_res_ws, bmm2_res_ws, dqi_ws,
    p_reduced_out, rsy_out, si_out,
    dqi_out, dw_out, rg_out, loss_out,
)
torch.npu.synchronize()

print("=== TileLang kernel中间结果（从输出tensor读取）===")
print(f"P_reduced: range=[{p_reduced_out.min():.8f}, {p_reduced_out.max():.8f}], sum={p_reduced_out.sum():.8f}")
print(f"ReduceSumY: range=[{rsy_out.min():.8f}, {rsy_out.max():.8f}]")
print(f"SoftmaxI: range=[{si_out.min():.8f}, {si_out.max():.8f}], sum={si_out.sum():.8f}")

print(f"\n=== Python参考 ===")
print(f"P_reduced: range=[{P_reduced_ref.min():.8f}, {P_reduced_ref.max():.8f}], sum={P_reduced_ref.sum():.8f}")
print(f"ReduceSumY: range=[{reduce_sum_y_ref.min():.8f}, {reduce_sum_y_ref.max():.8f}]")
print(f"SoftmaxI: range=[{softmax_i_ref.min():.8f}, {softmax_i_ref.max():.8f}], sum={softmax_i_ref.sum():.8f}")

print(f"\n=== 逐层对比 ===")
p_diff = (p_reduced_out.cpu() - P_reduced_ref.cpu()).abs()
print(f"|P_tl - P_ref|: max={p_diff.max():.8f}, mean={p_diff.mean():.8f}")
print(f"  P_ref max_abs={P_reduced_ref.cpu().abs().max():.8f}, rel_err={p_diff.max()/(P_reduced_ref.cpu().abs().max()+1e-8):.8f}")

rsy_diff = (rsy_out.cpu() - reduce_sum_y_ref.cpu()).abs()
print(f"|RSY_tl - RSY_ref|: max={rsy_diff.max():.8f}, mean={rsy_diff.mean():.8f}")
print(f"  RSY_ref max_abs={reduce_sum_y_ref.cpu().abs().max():.8f}, rel_err={rsy_diff.max()/(reduce_sum_y_ref.cpu().abs().max()+1e-8):.8f}")

si_diff = (si_out.cpu() - softmax_i_ref.cpu()).abs()
print(f"|SI_tl - SI_ref|: max={si_diff.max():.8f}, mean={si_diff.mean():.8f}")
print(f"  SI_ref max_abs={softmax_i_ref.cpu().abs().max():.8f}, rel_err={si_diff.max()/(softmax_i_ref.cpu().abs().max()+1e-8):.8f}")

# Also check bmm1 and bmm2
print(f"\n=== bmm1/bmm2对比 ===")
bmm1_diff = (bmm1_res_ws.cpu() - bmm1_noscale.cpu()).abs()
print(f"|bmm1_tl - bmm1_ref|: max={bmm1_diff.max():.6f}, mean={bmm1_diff.mean():.6f}")
bmm2_diff = (bmm2_res_ws.cpu() - bmm2_ref.cpu()).abs()
print(f"|bmm2_tl - bmm2_ref|: max={bmm2_diff.max():.6f}, mean={bmm2_diff.mean():.6f}")