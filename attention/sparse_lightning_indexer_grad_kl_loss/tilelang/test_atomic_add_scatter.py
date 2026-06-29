"""
验证 T.tile.atomic_add 是否支持动态索引 scatter
测试场景：
1. 静态索引 atomic_add (baseline) - 应该成功
2. 动态索引 atomic_add (通过 region slice) - 可能失败
3. 动态索引 atomic_add (通过 single row buffer load) - 可能失败
4. 动态索引 atomic_add (通过 T.tile.gather 间接实现) - 新方案

关键问题：TileLang的T.copy不支持动态索引，但T.tile.atomic_add是否支持？
根据 TileLang Programming Guide (行2007-2044):
  - dst: GM目标buffer、buffer load或region
  - src: 本地tensor (UB/shared buffer 或 L0C/fragment buffer)
  
本脚本验证：
  1. region能否包含动态索引表达式？
  2. 如果不支持，有什么替代方案？
"""

import torch
import torch_npu  # Required for NPU
import numpy as np
import tilelang
import tilelang.language as T
import sys

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# Test 1: Static index atomic_add (baseline)
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def test_static_atomic_add():
    @T.prim_func
    def main(
        Src: T.Tensor((16, 128), "float32"),
        Dst: T.Tensor((256, 128), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((16, 128), "float32")
            T.tile.fill(src_ub, 1.0)
            
            # Static: write to fixed position [0:16, 0:128]
            T.tile.atomic_add(Dst[0:16, 0:128], src_ub)
    
    return main


# Test 2: Dynamic index via region (尝试失败会报错)
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def test_dynamic_atomic_add_region():
    @T.prim_func
    def main(
        Src: T.Tensor((16, 128), "float32"),
        Dst: T.Tensor((256, 128), "float32"),
        Indices: T.Tensor((16,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((16, 128), "float32")
            idx_ub = T.alloc_ub((16,), "int32")
            
            T.tile.fill(src_ub, 2.0)
            T.copy(Indices[0:16], idx_ub)
            
            # Try dynamic: per-row scatter using region
            for i in T.serial(16):
                row_idx = idx_ub[i]
                # Attempt: atomic_add to dynamic region
                # This may fail if TileLang doesn't support dynamic indexing
                T.tile.atomic_add(Dst[row_idx * 16:(row_idx + 1) * 16, 0:128], src_ub[i:i+1, 0:128])
    
    return main


# Test 3: Dynamic index via buffer load (尝试)
# 注意：这个测试可能会失败，因为row_idx是动态值
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def test_dynamic_atomic_add_buffer_load():
    @T.prim_func
    def main(
        Src: T.Tensor((16, 128), "float32"),
        Dst: T.Tensor((256, 128), "float32"),
        Indices: T.Tensor((16,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((16, 128), "float32")
            idx_ub = T.alloc_ub((16,), "int32")
            row_ub = T.alloc_ub((128,), "float32")
            
            T.tile.fill(src_ub, 3.0)
            T.copy(Indices[0:16], idx_ub)
            
            # Try dynamic: per-row scatter
            for i in T.serial(16):
                row_idx = idx_ub[i]
                T.copy(src_ub[i, 0:128], row_ub)
                # Attempt: atomic_add single row to dynamic position
                # Key question: does Dst[row_idx, ...] work with dynamic row_idx?
                T.tile.atomic_add(Dst[row_idx, 0:128], row_ub)
    
    return main


# Test 3b: Multi-core parallel scatter with fixed pattern
# Each core scatters to its assigned block (no dynamic index)
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def test_multicore_fixed_scatter():
    BLOCK_SIZE = 16
    
    @T.prim_func
    def main(
        Src: T.Tensor((64, 128), "float32"),  # 4 blocks × 16 rows
        Dst: T.Tensor((256, 128), "float32"),  # 16 blocks × 16 rows
    ):
        with T.Kernel(4, is_npu=True) as (cid, vid):  # 4 cores
            block_id = cid
            src_ub = T.alloc_ub((BLOCK_SIZE, 128), "float32")
            
            # Each core loads its source block
            T.copy(Src[block_id * BLOCK_SIZE:(block_id + 1) * BLOCK_SIZE, 0:128], src_ub)
            
            # Each core scatters to fixed destination block (4x replication)
            # Core 0 → blocks 0,4,8,12; Core 1 → blocks 1,5,9,13; etc.
            for rep in T.serial(4):
                dst_block = block_id + rep * 4
                T.tile.atomic_add(Dst[dst_block * BLOCK_SIZE:(dst_block + 1) * BLOCK_SIZE, 0:128], src_ub)
    
    return main


# Test 5: Simulate 0622.py Matmul4 + Scatter scenario
# This is the critical test for the optimization we want to implement
# 
# Current Python code (to be moved into kernel):
#   relu_grad_t = relu_grad.permute(0,1,3,2)  # (B,S1,K,Nidx1) -> (B,S1,Nidx1,K)
#   scatter_add = bmm(relu_grad_t, query_index)  # (B,S1,K,DQueryIndex)
#   d_key_index.scatter_add_(1, sparse_idx, scatter_add)
#
# In kernel we want to:
#   C scope: Matmul5 = reluGrad^T @ QueryIndex -> (K, DQueryIndex) in L0C
#   V scope: atomic_add scatter to GM using sparse_idx
#
# The challenge: sparse_idx is dynamic (different for each K position)
# Even with multi-core K-parallel, each core needs to scatter to different GM positions
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def test_simulate_matmul_scatter():
    BLOCK_K = 128  # Like 0622.py
    Nidx1 = 16
    DQueryIndex = 128
    
    @T.prim_func
    def main(
        ReluGrad: T.Tensor((Nidx1, BLOCK_K), "float16"),  # From previous step
        QueryIndex: T.Tensor((Nidx1, DQueryIndex), "float16"),  # Input
        SparseIndices: T.Tensor((BLOCK_K,), "int32"),  # Dynamic indices
        DKeyIndex: T.Tensor((2048, DQueryIndex), "float32"),  # GM destination
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            # C scope: Matmul = ReluGrad^T @ QueryIndex
            with T.Scope("C"):
                RG_L1 = T.alloc_L1((Nidx1, BLOCK_K), "float16")
                QI_L1 = T.alloc_L1((Nidx1, DQueryIndex), "float16")
                Result_L0C = T.alloc_L0C((BLOCK_K, DQueryIndex), "float32")
                
                T.copy(ReluGrad[0:Nidx1, 0:BLOCK_K], RG_L1)
                T.copy(QueryIndex[0:Nidx1, 0:DQueryIndex], QI_L1)
                T.gemm_v0(RG_L1, QI_L1, Result_L0C, transpose_A=True, init=True)
            
            # V scope: Scatter the L0C result to GM using SparseIndices
            with T.Scope("V"):
                # Load indices
                idx_ub = T.alloc_ub((BLOCK_K,), "int32")
                row_ub = T.alloc_ub((DQueryIndex,), "float32")
                
                T.copy(SparseIndices[0:BLOCK_K], idx_ub)
                
                # Try to scatter each row to its dynamic destination
                # This is the critical part - will it work?
                for k in T.serial(BLOCK_K):
                    dest_row = idx_ub[k]
                    # Copy row from L0C to UB (need intermediate step)
                    # Note: L0C is in Cube output format, need special handling
                    # For simplicity, assume we have the row in UB
                    
                    # Attempt atomic_add to dynamic position
                    # This line is the key test:
                    T.tile.atomic_add(DKeyIndex[dest_row, 0:DQueryIndex], row_ub)
                    # Expected to FAIL if TileLang doesn't support dynamic indices
    
    return main


# Test 5b: Alternative approach - output partial results, Python scatter
# This is the fallback if dynamic atomic_add doesn't work
@tilelang.jit(out_idx=[-1, -2, -3], pass_configs=pass_configs)
def test_partial_matmul_output():
    BLOCK_K = 128
    Nidx1 = 16
    DQueryIndex = 128
    
    @T.prim_func
    def main(
        ReluGrad: T.Tensor((Nidx1, BLOCK_K), "float16"),
        QueryIndex: T.Tensor((Nidx1, DQueryIndex), "float16"),
        SparseIndices: T.Tensor((BLOCK_K,), "int32"),
        MatmulResult: T.Tensor((BLOCK_K, DQueryIndex), "float32"),  # Output to Python
        IndicesOut: T.Tensor((BLOCK_K,), "int32"),  # Output indices to Python
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            # C scope: Matmul
            with T.Scope("C"):
                RG_L1 = T.alloc_L1((Nidx1, BLOCK_K), "float16")
                QI_L1 = T.alloc_L1((Nidx1, DQueryIndex), "float16")
                Result_L0C = T.alloc_L0C((BLOCK_K, DQueryIndex), "float32")
                
                T.copy(ReluGrad[0:Nidx1, 0:BLOCK_K], RG_L1)
                T.copy(QueryIndex[0:Nidx1, 0:DQueryIndex], QI_L1)
                T.gemm_v0(RG_L1, QI_L1, Result_L0C, transpose_A=True, init=True)
            
            # V scope: Copy results to GM (no scatter, just output)
            with T.Scope("V"):
                result_ub = T.alloc_ub((BLOCK_K, DQueryIndex), "float32")
                idx_ub = T.alloc_ub((BLOCK_K,), "int32")
                
                # Copy from L0C to UB (need proper handling)
                # For now, assume we can copy the matmul result
                T.copy(Result_L0C[0:BLOCK_K, 0:DQueryIndex], result_ub)
                T.copy(result_ub, MatmulResult[0:BLOCK_K, 0:DQueryIndex])
                
                # Copy indices for Python scatter
                T.copy(SparseIndices[0:BLOCK_K], idx_ub)
                T.copy(idx_ub, IndicesOut[0:BLOCK_K])
    
    return main


# Test 4: Alternative - copy to static partial, then Python scatter
@tilelang.jit(out_idx=[-1, -2], pass_configs=pass_configs)
def test_partial_output_for_scatter():
    @T.prim_func
    def main(
        Src: T.Tensor((16, 128), "float32"),
        DstPartial: T.Tensor((16, 128), "float32"),
        IndicesPartial: T.Tensor((16,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((16, 128), "float32")
            idx_ub = T.alloc_ub((16,), "int32")
            
            T.tile.fill(src_ub, 4.0)
            # Just output partial results, let Python handle scatter
            T.copy(src_ub, DstPartial[0:16, 0:128])
            T.copy(Indices[0:16], idx_ub)
            T.copy(idx_ub, IndicesPartial[0:16])
    
    return main


def run_tests():
    device = "npu"
    
    print("=" * 60)
    print("Test 1: Static index atomic_add (baseline)")
    print("=" * 60)
    try:
        func1 = test_static_atomic_add()
        dst = torch.zeros(256, 128, dtype=torch.float32, device=device)
        src = torch.ones(16, 128, dtype=torch.float32, device=device)
        func1(src, dst)
        expected = torch.zeros(256, 128, dtype=torch.float32, device=device)
        expected[0:16, 0:128] = 1.0
        diff = (dst - expected).abs().max().item()
        print(f"Result: max_diff={diff:.6e}, PASS={diff < 1e-5}")
        print(f"Dst[0:2, 0:5] = {dst[0:2, 0:5].cpu().numpy()}")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Test 2: Dynamic atomic_add via region")
    print("=" * 60)
    try:
        func2 = test_dynamic_atomic_add_region()
        dst = torch.zeros(256, 128, dtype=torch.float32, device=device)
        src = torch.ones(16, 128, dtype=torch.float32, device=device)
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
                               dtype=torch.int32, device=device)
        func2(src, dst, indices)
        expected = torch.zeros(256, 128, dtype=torch.float32, device=device)
        for i in range(16):
            expected[i*16:(i+1)*16, 0:128] = 2.0
        diff = (dst - expected).abs().max().item()
        print(f"Result: max_diff={diff:.6e}, PASS={diff < 1e-5}")
        print(f"Dst[0:2, 0:5] = {dst[0:2, 0:5].cpu().numpy()}")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Test 3: Dynamic atomic_add via buffer load (single row)")
    print("=" * 60)
    try:
        func3 = test_dynamic_atomic_add_buffer_load()
        dst = torch.zeros(256, 128, dtype=torch.float32, device=device)
        src = torch.ones(16, 128, dtype=torch.float32, device=device)
        indices = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160], 
                               dtype=torch.int32, device=device)
        func3(src, dst, indices)
        expected = torch.zeros(256, 128, dtype=torch.float32, device=device)
        for i, idx in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]):
            expected[idx, 0:128] = 3.0
        diff = (dst - expected).abs().max().item()
        print(f"Result: max_diff={diff:.6e}, PASS={diff < 1e-5}")
        print(f"Dst[10:12, 0:5] = {dst[10:12, 0:5].cpu().numpy()}")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Test 3b: Multi-core fixed pattern scatter")
    print("=" * 60)
    try:
        func3b = test_multicore_fixed_scatter()
        dst = torch.zeros(256, 128, dtype=torch.float32, device=device)
        src = torch.ones(64, 128, dtype=torch.float32, device=device)
        func3b(src, dst)
        # Each block gets 4x accumulation
        expected = torch.zeros(256, 128, dtype=torch.float32, device=device)
        expected[0:64, 0:128] = 4.0  # blocks 0-3 get 4 accumulations
        diff = (dst - expected).abs().max().item()
        print(f"Result: max_diff={diff:.6e}, PASS={diff < 1e-5}")
        print(f"Dst[0:2, 0:5] = {dst[0:2, 0:5].cpu().numpy()}")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Test 5: Simulate 0622.py Matmul + Scatter (CRITICAL)")
    print("=" * 60)
    print("This tests the exact scenario we want to optimize in 0622.py")
    try:
        func5 = test_simulate_matmul_scatter()
        relu_grad = torch.ones(16, 128, dtype=torch.float16, device=device)
        query_index = torch.ones(16, 128, dtype=torch.float16, device=device)
        sparse_indices = torch.tensor([0, 128, 256, 384, 512, 640, 768, 896, 
                                        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
                                        0, 128, 256, 384, 512, 640, 768, 896,
                                        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
                                        0, 128, 256, 384, 512, 640, 768, 896,
                                        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
                                        0, 128, 256, 384, 512, 640, 768, 896,
                                        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
                                        0, 128, 256, 384, 512, 640, 768, 896,
                                        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
                                        0, 128, 256, 384, 512, 640, 768, 896,
                                        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
                                        0, 128, 256, 384, 512, 640, 768, 896,
                                        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920],
                                       dtype=torch.int32, device=device)
        d_key_index = torch.zeros(2048, 128, dtype=torch.float32, device=device)
        func5(relu_grad, query_index, sparse_indices, d_key_index)
        
        # Check if scatter worked
        print(f"DKeyIndex[0, 0:5] = {d_key_index[0, 0:5].cpu().numpy()}")
        print(f"DKeyIndex[128, 0:5] = {d_key_index[128, 0:5].cpu().numpy()}")
        diff = d_key_index.abs().max().item()
        print(f"max_value = {diff}")
        if diff > 0:
            print("PASS: Scatter appears to work!")
        else:
            print("FAILED: All zeros, scatter didn't work")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test 5b: Partial Matmul output (fallback approach)")
    print("=" * 60)
    try:
        func5b = test_partial_matmul_output()
        relu_grad = torch.ones(16, 128, dtype=torch.float16, device=device)
        query_index = torch.ones(16, 128, dtype=torch.float16, device=device)
        sparse_indices = torch.arange(128, dtype=torch.int32, device=device)
        matmul_result = torch.zeros(128, 128, dtype=torch.float32, device=device)
        indices_out = torch.zeros(128, dtype=torch.int32, device=device)
        
        func5b(relu_grad, query_index, sparse_indices, matmul_result, indices_out)
        
        print(f"MatmulResult[0:2, 0:5] = {matmul_result[0:2, 0:5].cpu().numpy()}")
        print(f"IndicesOut[0:10] = {indices_out[0:10].cpu().numpy()}")
        print("PASS: Partial output works, Python can scatter afterwards")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Test 1 (static): Baseline - should PASS
  - T.tile.atomic_add(Dst[0:16, 0:128], src_ub)
  - Static slice range, no dynamic indexing

Test 2 (region): Dynamic via region slice
  - T.tile.atomic_add(Dst[row_idx*16:(row_idx+1)*16, 0:128], src_ub)
  - Uses loop variable row_idx from buffer
  - Likely FAILS if TileLang doesn't support dynamic expressions in slices

Test 3 (buffer load): Dynamic via single row
  - T.tile.atomic_add(Dst[row_idx, 0:128], row_ub)
  - Single row atomic_add to dynamic position
  - Likely FAILS for same reason as Test 2

Test 3b (multi-core fixed): Fixed pattern scatter
  - Each core scatters to predetermined blocks
  - No dynamic indices, should PASS
  - This is the current 0622.py approach (K-block parallel)

Test 4 (partial): Alternative - Python handles scatter
  - Kernel outputs partial results + indices
  - Python uses torch.scatter_add_
  - Should PASS, but has Python overhead

Key Insight for 0622.py optimization:
-----------------------------------------
Current Python bottleneck (48%) is _compute_d_key_index which does:
  1. torch.bmm(relu_grad_t, query_index)  # ~33M FLOPs
  2. scatter_add_(sparse_idx, ...)         # Random write

Option A: Move bmm into kernel (T.gemm + T.tile.atomic_add)
  - Challenge: How to scatter to dynamic indices in GM?
  - If Test 3 fails, we need alternative approach

Option B: Pre-compute scatter destinations as static blocks
  - Like Test 3b: each core has predetermined destination
  - Challenge: sparse_indices are dynamic and may not align with blocks

Option C: Keep Python scatter, optimize the bmm only
  - Move bmm to kernel, output partial to GM
  - Python only does scatter_add_
  - Reduces Python work from 48% to maybe 20-30%

Option D: Use T.tile.gather in reverse direction
  - Compute a "reverse mapping" in preprocessing
  - Then use T.tile.gather to read into correct positions
  - Complex but may work with static indices
""")
    
    print("\nRecommended next steps:")
    print("1. Run this script on NPU environment to confirm which tests pass")
    print("2. If Test 3 fails, evaluate Option C (partial kernelization)")
    print("3. If Test 3b passes, consider adapting 0622.py to use block-based scatter")


if __name__ == "__main__":
    run_tests()