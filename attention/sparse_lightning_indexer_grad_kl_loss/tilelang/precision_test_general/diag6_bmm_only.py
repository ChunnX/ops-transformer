#!/usr/bin/env python3
"""诊断 - 直接对比TileLang kernel的bmm1Res与Python参考，定位误差源头
只测试Matmul1 (bmm1) 和 Matmul2 (bmm2) 的结果
"""

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
DQUERY = 512
DQUERY_INDEX = 128

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

def ceil_div(a, b):
    return (a + b - 1) // b

# Minimal kernel: only compute bmm1 and bmm2, output them
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs=pass_configs,
)
def bmm_only_kernel(
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
    d_num_query_index = DQueryIndex // k_block

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, DQuery), input_dtype),
        QueryRope: T.Tensor((B, S1, N1, d_rope), input_dtype),
        GatherKey: T.Tensor((B, S1, K, DQuery), input_dtype),
        GatherKeyRope: T.Tensor((B, S1, K, d_rope), input_dtype),
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        GatherKeyIndex: T.Tensor((B, S1, K, DQueryIndex), input_dtype),
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype),
        bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
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
                                T.copy(QueryRope[bz, s1_row, 0:N1, rope_d*k_block:(rope_d+1)*k_block], Q_L1)
                            T.barrier_all()
                            if d_idx < d_num_query:
                                T.copy(GatherKey[bz, s1_row, n_idx*n_block:(n_idx+1)*n_block, d_idx*k_block:(d_idx+1)*k_block], GK_L1)
                            else:
                                rope_d = d_idx - d_num_query
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

    return main


# === Run ===
B, S1, S2 = 1, 1, 2048
N1, N2, Nidx1 = 128, 1, 64
D, DIndex, DR, K = 512, 128, 64, 2048
G = N1 // N2
scale = 1.0 / math.sqrt(D)
device = torch.device('npu')
dtype = torch.float16

torch.manual_seed(42)
q = torch.randn(B, S1, N1, D, dtype=dtype, device=device)
k = torch.randn(B, S2, N2, D, dtype=dtype, device=device)
q_index = torch.randn(B, S1, Nidx1, DIndex, dtype=dtype, device=device)
k_index = torch.randn(B, S2, N2, DIndex, dtype=dtype, device=device)
q_rope = torch.randn(B, S1, N1, DR, dtype=dtype, device=device)
k_rope = torch.randn(B, S2, N2, DR, dtype=dtype, device=device)

# Random sparse_indices (same as general test)
sparse_indices = torch.zeros(B, S1, N2, K, dtype=torch.int32).npu()
for s1Idx in range(S1):
    s2RealSize = S2 - S1 + s1Idx + 1
    if s2RealSize <= 0:
        s2RealSize = S2
    s2RealLen = min(s2RealSize, K)
    sparse_indices[:, s1Idx, 0, 0:s2RealLen] = torch.randperm(s2RealSize, dtype=torch.int32)[:s2RealLen].npu()
    sparse_indices[:, s1Idx, 0, s2RealLen:K] = -1

# Gather
S2_pad = (S2 + 256 - 1) // 256 * 256
k_p = torch.nn.functional.pad(k, (0,0,0,0,0,S2_pad-S2,0,0)) if S2_pad > S2 else k
ki_p = torch.nn.functional.pad(k_index, (0,0,0,0,0,S2_pad-S2,0,0)) if S2_pad > S2 else k_index
gk, gki = _preprocess_gather(k_p, ki_p, sparse_indices, B, S1, S2_pad, N2, 1, K, D, DIndex)
gk_rope = k_rope[torch.arange(B, device=device).view(B, 1, 1),
                  sparse_indices[:, :, 0, :].clamp(min=0), 0, :].contiguous()

# Pad rope
d_rope_pad = ((DR + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
q_rope_padded = torch.nn.functional.pad(q_rope, (0, d_rope_pad - DR))
gk_rope_padded = torch.nn.functional.pad(gk_rope, (0, d_rope_pad - DR))

# Python reference bmm1 and bmm2
q_full = torch.cat([q.float(), q_rope.float()], dim=-1)
gk_full = torch.cat([gk.float(), gk_rope.float()], dim=-1)
bmm1_ref = torch.matmul(q_full, gk_full.transpose(-2, -1))  # (B, S1, N1, K)
bmm2_ref = torch.relu(torch.matmul(q_index.float(), gki.float().transpose(-2, -1)))  # (B, S1, Nidx1, K)

# Run kernel
func = bmm_only_kernel(B, S1, S2_pad, N1, N2, G, Nidx1, 1, D, DIndex, K, d_rope_pad, "float16", "float", scale)
bmm1_out = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=device)
bmm2_out = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=device)

torch.npu.synchronize()
results = func(q, q_rope_padded, gk, gk_rope_padded, q_index, gki, bmm1_out, bmm2_out)
torch.npu.synchronize()

bmm1_tl = results[0]
bmm2_tl = results[1]

print("=== bmm1对比 ===")
bmm1_diff = (bmm1_tl.cpu() - bmm1_ref.cpu()).abs()
print(f"  max_diff={bmm1_diff.max():.6f}, mean_diff={bmm1_diff.mean():.6f}")
print(f"  ref max_abs={bmm1_ref.cpu().abs().max():.6f}")
print(f"  rel_err={bmm1_diff.max() / (bmm1_ref.cpu().abs().max() + 1e-8):.6f}")

# Check per-G-head
for g_id in range(min(4, G)):
    diff_g = (bmm1_tl[0, 0, g_id, :].cpu() - bmm1_ref[0, 0, g_id, :].cpu()).abs()
    ref_g = bmm1_ref[0, 0, g_id, :].cpu().abs().max()
    print(f"  head[{g_id}]: max_diff={diff_g.max():.6f}, ref_max={ref_g:.6f}, rel_err={diff_g.max()/(ref_g+1e-8):.6f}")

print("\n=== bmm2对比 ===")
bmm2_diff = (bmm2_tl.cpu() - bmm2_ref.cpu()).abs()
print(f"  max_diff={bmm2_diff.max():.6f}, mean_diff={bmm2_diff.mean():.6f}")
print(f"  ref max_abs={bmm2_ref.cpu().abs().max():.6f}")
print(f"  rel_err={bmm2_diff.max() / (bmm2_ref.cpu().abs().max() + 1e-8):.6f}")

# Also check: does bmm1 from official match?
print("\n=== 官方算子的bmm1（通过完整调用推断）===")
print("  (无法直接获取官方bmm1，但可以对比最终输出)")