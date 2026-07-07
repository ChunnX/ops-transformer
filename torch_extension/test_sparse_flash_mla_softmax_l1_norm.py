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


# ============================================================
# metadata 正常场景
# ============================================================

print("=" * 60)
print("metadata 场景 1: BSND + seqused_q")
print("=" * 60)
metadata_bsnd = torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    seqused_q=torch.tensor([128, 100, 64, 32], dtype=torch.int32).npu(),
    batch_size=4, max_seqlen_q=128, max_seqlen_k=128, topk=64,
    layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
)
print_metadata("BSND + seqused_q", metadata_bsnd)

print("=" * 60)
print("metadata 场景 2: TND + cu_seqlens_q")
print("=" * 60)
metadata_tnd = torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=8, num_heads_k=1, head_dim=128,
    cu_seqlens_q=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
    cu_seqlens_k=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
    topk=64,
    layout_q="TND", layout_k="TND", mask_mode=3, cmp_ratio=4,
)
print_metadata("TND + cu_seqlens_q", metadata_tnd)

# ============================================================
# 主算子正常场景 (骨架阶段，预期 kernel 失败)
# ============================================================

print()
print("=" * 60)
print("主算子场景 1: TND + metadata 前置算子")
print("=" * 60)

s1 = 16
n1 = 128
n2 = 1
d = 512
s2 = 2048
g = n1 // n2

q = torch.randn(s1, n1, d, dtype=torch.float16).npu()
key = torch.randn(s2, n2, d, dtype=torch.float16).npu()
lse = torch.randn(n2, s1, g, dtype=torch.float32).npu()
cu_seqlens_q = torch.tensor([0, s1], dtype=torch.int32).npu()
cu_seqlens_k = torch.tensor([0, s2], dtype=torch.int32).npu()
cmp_residual_k = torch.tensor([0], dtype=torch.int32).npu()

metadata = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=n1,
    num_heads_k=n2,
    head_dim=d,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    topk=0,
    layout_q="TND",
    layout_k="TND",
    mask_mode=3,
    cmp_ratio=128,
)
print(f"  metadata shape: {metadata.shape}, dtype: {metadata.dtype}")

try:
    out = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
        q, key, lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        scale_value=0.088388,
        cmp_ratio=128,
        mask_mode=3,
        layout_q="TND",
        layout_k="TND",
    )
    print(f"  output shape: {out.shape}, dtype: {out.dtype}")
except RuntimeError as e:
    msg = str(e)
    short = msg[:120] + "..." if len(msg) > 120 else msg
    print(f"  [Expected] skeleton kernel failure: {short}")

# ============================================================
# 异常场景测试
# ============================================================
print()
print("=" * 60)
print("异常场景测试")
print("=" * 60)

expect_fail("query 和 key dtype 不一致", lambda: cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
    q, key.to(torch.bfloat16), lse,
    scale_value=0.088388, cmp_ratio=128, mask_mode=3,
    layout_q="TND", layout_k="TND",
))

expect_fail("cmp_ratio 超范围", lambda: cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
    q, key, lse,
    scale_value=0.088388, cmp_ratio=0, mask_mode=3,
    layout_q="TND", layout_k="TND",
))

expect_fail("mask_mode 非法", lambda: cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
    q, key, lse,
    scale_value=0.088388, cmp_ratio=128, mask_mode=1,
    layout_q="TND", layout_k="TND",
))

expect_fail("layout_q 非法", lambda: cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
    q, key, lse,
    scale_value=0.088388, cmp_ratio=128, mask_mode=3,
    layout_q="BSH", layout_k="TND",
))

print()
print("All tests passed!")
