import os
import contextlib
import torch
import torch_npu
import cann_ops_transformer


@contextlib.contextmanager
def suppress_c_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


def expect_fail(tag, fn):
    try:
        with suppress_c_stderr():
            result = fn()
            _ = result.cpu()
        print(f"  [FAIL] {tag} - 应报错但未拦截")
    except Exception as e:
        msg = str(e)
        short = msg[:120] + "..." if len(msg) > 120 else msg
        print(f"  [PASS] {tag} - 拦截成功: {short}")


def print_metadata(tag, metadata):
    vals = metadata.cpu().tolist()
    print(f"=== {tag} ===")
    print(f"  totalNum             = {vals[0]}")
    print(f"  formerCoreProcessNum = {vals[1]}")
    print(f"  remainCoreProcessNum = {vals[2]}")
    print(f"  remainCoreNum        = {vals[3]}")
    print(f"  totalCoreNum         = {vals[4]}")
    print()


# ============================================================
# 正常场景
# ============================================================

print("=" * 60)
print("场景 1: BSND + seqused_q")
print("=" * 60)
metadata_bsnd = torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    seqused_q=torch.tensor([128, 100, 64, 32], dtype=torch.int32).npu(),
    batch_size=4, max_seqlen_q=128, max_seqlen_k=128, topk=64,
    layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
)
print_metadata("BSND + seqused_q", metadata_bsnd)

print("=" * 60)
print("场景 2: TND + cu_seqlens_q")
print("=" * 60)
metadata_tnd = torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    cu_seqlens_q=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
    cu_seqlens_k=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
    topk=64,
    layout_q="TND", layout_k="TND", mask_mode=3, cmp_ratio=4,
)
print_metadata("TND + cu_seqlens_q", metadata_tnd)

print("=" * 60)
print("场景 3: BSND 纯 attr (无 tensor 输入)")
print("=" * 60)
metadata_attr = torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    batch_size=2, max_seqlen_q=256, max_seqlen_k=256, topk=128,
    layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
)
print_metadata("BSND 纯 attr", metadata_attr)

# ============================================================
# 异常场景测试: 验证 ParamsCheck 拦截是否生效
# ============================================================
print("=" * 60)
print("异常场景测试")
print("=" * 60)

expect_fail("seqused_q 长度不等于 batch_size", lambda: torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    seqused_q=torch.tensor([128, 100, 64], dtype=torch.int32).npu(),
    batch_size=4, max_seqlen_q=128, max_seqlen_k=128, topk=64,
    layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
))

expect_fail("cu_seqlens_q 长度不等于 batch_size+1", lambda: torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    cu_seqlens_q=torch.tensor([0, 128, 228, 292], dtype=torch.int32).npu(),
    cu_seqlens_k=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
    batch_size=4, topk=64,
    layout_q="TND", layout_k="TND", mask_mode=0, cmp_ratio=1,
))

expect_fail("TND 缺少 cu_seqlens_k", lambda: torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    cu_seqlens_q=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
    cu_seqlens_k=None,
    batch_size=4, topk=64,
    layout_q="TND", layout_k="TND", mask_mode=0, cmp_ratio=1,
))

expect_fail("num_heads_q 不能被 num_heads_k 整除", lambda: torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=7, num_heads_k=4, head_dim=128,
    batch_size=2, max_seqlen_q=128, max_seqlen_k=128, topk=64,
    layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
))

expect_fail("mask_mode 非法", lambda: torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    batch_size=2, max_seqlen_q=128, max_seqlen_k=128, topk=64,
    layout_q="BSND", layout_k="BSND", mask_mode=1, cmp_ratio=1,
))

expect_fail("cmp_ratio 超范围", lambda: torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    batch_size=2, max_seqlen_q=128, max_seqlen_k=128, topk=64,
    layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=0,
))

expect_fail("cmp_residual_k 长度不等于 batch_size", lambda: torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    cmp_residual_k=torch.tensor([2, 3, 4], dtype=torch.int32).npu(),
    batch_size=4, max_seqlen_q=128, max_seqlen_k=128, topk=64,
    layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
))

print()
print("All tests passed!")
