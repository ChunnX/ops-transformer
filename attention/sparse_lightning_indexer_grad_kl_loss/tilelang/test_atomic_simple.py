"""Simple test for T.tile.atomic_add static indexing"""

import torch
import tilelang
import tilelang.language as T

print("Starting test...")

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def test_static_atomic():
    @T.prim_func
    def main(
        Src: T.Tensor((16, 128), "float32"),
        Dst: T.Tensor((256, 128), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((16, 128), "float32")
            T.tile.fill(src_ub, 1.0)
            T.tile.atomic_add(Dst[0:16, 0:128], src_ub)
    return main

print("Compiling kernel...")
try:
    func = test_static_atomic()
    print("Kernel compiled successfully")
    
    print("Creating tensors...")
    dst = torch.zeros(256, 128, dtype=torch.float32, device="npu")
    src = torch.ones(16, 128, dtype=torch.float32, device="npu")
    
    print("Running kernel...")
    func(src, dst)
    
    print("Checking result...")
    result = dst[0:2, 0:5].cpu().numpy()
    print(f"Dst[0:2, 0:5] = {result}")
    print(f"Expected: [[1.0, 1.0, ...], [1.0, 1.0, ...]]")
    
    diff = (dst[0:16, 0:128] - 1.0).abs().max().item()
    print(f"max_diff = {diff}")
    
    if diff < 1e-5:
        print("TEST PASSED: Static atomic_add works!")
    else:
        print(f"TEST FAILED: diff={diff} > 1e-5")
        
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()