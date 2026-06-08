#!/usr/bin/env python3
"""最小化测试: 验证T.tile.sub/div的标量buffer元素broadcast行为"""

import torch
import torch_npu
import math
import sys
import os

import tilelang
import tilelang.language as T

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

M = 128
N = 128

# Test 1: T.tile.sub with scalar buffer element
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def test_sub_scalar_buffer(M: int, N: int):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        ScalarBuf: T.Tensor((M,), "float32"),
        Out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                a_ub = T.alloc_ub((M, N), "float32")
                s_ub = T.alloc_ub(M, "float32")
                out_ub = T.alloc_ub((M, N), "float32")
                
                T.copy(A[0:M, 0:N], a_ub)
                T.barrier_all()
                T.copy(ScalarBuf[0:M], s_ub)
                T.barrier_all()
                
                # Test: sub with scalar buffer element
                for i in range(M):
                    T.tile.sub(a_ub[i, :], a_ub[i, :], s_ub[i])
                    T.barrier_all()
                
                T.copy(a_ub, Out[0:M, 0:N])
                T.barrier_all()

    return main


# Test 2: T.tile.div with scalar buffer element
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def test_div_scalar_buffer(M: int, N: int):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        ScalarBuf: T.Tensor((M,), "float32"),
        Out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                a_ub = T.alloc_ub((M, N), "float32")
                s_ub = T.alloc_ub(M, "float32")
                
                T.copy(A[0:M, 0:N], a_ub)
                T.barrier_all()
                T.copy(ScalarBuf[0:M], s_ub)
                T.barrier_all()
                
                for i in range(M):
                    T.tile.div(a_ub[i, :], a_ub[i, :], s_ub[i])
                    T.barrier_all()
                
                T.copy(a_ub, Out[0:M, 0:N])
                T.barrier_all()

    return main


device = torch.device('npu')

# Test sub
print("=== Test T.tile.sub with scalar buffer element ===")
func_sub = test_sub_scalar_buffer(M, N)
a = torch.randn(M, N, dtype=torch.float32, device=device)
scalar = torch.randn(M, dtype=torch.float32, device=device)
out_sub = torch.zeros(M, N, dtype=torch.float32, device=device)
torch.npu.synchronize()
result_sub = func_sub(a, scalar, out_sub)
torch.npu.synchronize()

# Python reference
ref_sub = a.cpu() - scalar.cpu().unsqueeze(-1)
actual_sub = result_sub[0].cpu() if isinstance(result_sub, (list, tuple)) else result_sub.cpu()

diff_sub = (actual_sub - ref_sub).abs()
print(f"  max_diff={diff_sub.max():.8f}, mean_diff={diff_sub.mean():.8f}")
if diff_sub.max() > 1e-4:
    print(f"  FAIL! Scalar buffer element NOT broadcast correctly")
    # Check if only first element was subtracted
    for i in range(min(4, M)):
        row_diff = (actual_sub[i, :] - ref_sub[i, :]).abs()
        print(f"    row {i}: max_diff={row_diff.max():.8f}, first_elem_diff={row_diff[0]:.8f}, last_elem_diff={row_diff[-1]:.8f}")
else:
    print(f"  PASS! Scalar buffer element broadcast correctly")

# Test div
print("\n=== Test T.tile.div with scalar buffer element ===")
func_div = test_div_scalar_buffer(M, N)
a2 = torch.randn(M, N, dtype=torch.float32, device=device) + 10  # positive values
scalar2 = torch.randn(M, dtype=torch.float32, device=device).abs() + 1  # positive values > 1
out_div = torch.zeros(M, N, dtype=torch.float32, device=device)
torch.npu.synchronize()
result_div = func_div(a2, scalar2, out_div)
torch.npu.synchronize()

ref_div = a2.cpu() / scalar2.cpu().unsqueeze(-1)
actual_div = result_div[0].cpu() if isinstance(result_div, (list, tuple)) else result_div.cpu()

diff_div = (actual_div - ref_div).abs()
print(f"  max_diff={diff_div.max():.8f}, mean_diff={diff_div.mean():.8f}")
if diff_div.max() > 1e-4:
    print(f"  FAIL! Scalar buffer element NOT broadcast correctly")
    for i in range(min(4, M)):
        row_diff = (actual_div[i, :] - ref_div[i, :]).abs()
        print(f"    row {i}: max_diff={row_diff.max():.8f}, first_elem_diff={row_diff[0]:.8f}, last_elem_diff={row_diff[-1]:.8f}")
else:
    print(f"  PASS! Scalar buffer element broadcast correctly")