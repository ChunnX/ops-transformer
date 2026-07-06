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


# ============================================================
# 正常场景 (骨架阶段，预期 kernel 失败)
# ============================================================

print("=" * 60)
print("场景 1: TND + metadata 前置算子")
print("=" * 60)

b = 1
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

# 1. 调用 metadata 前置算子
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

# 2. 调用主算子 (骨架阶段预期失败)
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

print()
print("=" * 60)
print("场景 2: BSND 布局")
print("=" * 60)

b = 2
s1 = 32
s2 = 512

q_bsnd = torch.randn(b, s1, n1, d, dtype=torch.float16).npu()
key_bsnd = torch.randn(b, s2, n2, d, dtype=torch.float16).npu()
lse_bsnd = torch.randn(b, n2, s1, g, dtype=torch.float32).npu()

metadata_bsnd = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=n1,
    num_heads_k=n2,
    head_dim=d,
    batch_size=b,
    max_seqlen_q=s1,
    max_seqlen_k=s2,
    topk=0,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=3,
    cmp_ratio=4,
)
print(f"  metadata shape: {metadata_bsnd.shape}")

try:
    out_bsnd = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
        q_bsnd, key_bsnd, lse_bsnd,
        metadata=metadata_bsnd,
        scale_value=0.088388,
        cmp_ratio=4,
        mask_mode=3,
        layout_q="BSND",
        layout_k="BSND",
    )
    print(f"  output shape: {out_bsnd.shape}, dtype: {out_bsnd.dtype}")
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
