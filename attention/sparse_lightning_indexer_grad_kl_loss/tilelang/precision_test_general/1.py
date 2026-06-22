import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()

@tilelang.jit(out_idx=[-1])
def test_indirect(N, dtype="float16"):
    @T.prim_func
    def main(
        Src: T.Tensor((N,), dtype),
        Idx: T.Tensor((2,), "int32"),
        Dst: T.Tensor((2,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                idx_ub = T.alloc_ub(1, "int32")
                for i in T.serial(2):
                    T.copy(Idx[i], idx_ub)
                    T.copy(Src[idx_ub[0]], Dst[i])
    return main

N = 128
src = torch.arange(N, dtype=torch.float16).npu()
idx = torch.tensor([10, 20], dtype=torch.int32).npu()
dst = torch.zeros(2, dtype=torch.float16).npu()

func = test_indirect(N)
result = func(src, idx, dst)
print(f"Expected: [{src[10].item()}, {src[20].item()}]")
print(f"Got:      [{result[0].item()}, {result[1].item()}]")