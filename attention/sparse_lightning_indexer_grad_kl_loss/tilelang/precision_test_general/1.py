import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()


@tilelang.jit(out_idx=[-1])
def test_indirect(N, dtype="float16"):
    @T.prim_func
    def main(
        Src: T.Tensor((N,), dtype),  # 源数据
        Idx: T.Tensor((2,), "int32"),  # 索引值
        Dst: T.Tensor((2,), dtype),  # 输出
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                for i in T.serial(2):
                    # 关键测试：用变量做索引
                    idx_val = T.alloc_var("int32")
                    T.copy(Idx[i], idx_val)
                    T.copy(Src[idx_val], Dst[i])

    return main


N = 128
src = torch.arange(N, dtype=torch.float16).npu()
idx = torch.tensor([10, 20], dtype=torch.int32).npu()
dst = torch.zeros(2, dtype=torch.float16).npu()

func = test_indirect(N)
result = func(src, idx, dst)
print(f"Expected: [{src[10].item()}, {src[20].item()}]")
print(f"Got:      [{result[0].item()}, {result[1].item()}]")
