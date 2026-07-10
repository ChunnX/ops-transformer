import os
import math
import contextlib
import torch
import torch_npu
import cann_ops_transformer


def _golden_softmax_l1_norm(q, k, lse, D=512):
    """Golden: sum(exp(q@k^T/sqrt(D) - lse)) over G, then divide by G."""
    B, Sq, G, _ = q.shape
    Sk = k.shape[1]
    qf = q.float()
    kf = k.float()
    lf = lse.float()
    qk = torch.matmul(qf, kf.permute(0, 2, 3, 1))  # [B, Sq, G, Sk]
    qk_scale = qk * (1.0 / math.sqrt(D))
    p = torch.exp(qk_scale - lf.permute(0, 2, 3, 1))  # [B, Sq, G, Sk]
    softmax_l1 = p.sum(dim=2) / G  # [B, Sq, Sk]
    return softmax_l1


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

# print("=" * 60)
# print("metadata 场景 1: BSND + seqused_q")
# print("=" * 60)
# metadata_bsnd = torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
#     num_heads_q=8, num_heads_k=1, head_dim=128,
#     seqused_q=torch.tensor([128, 100, 64, 32], dtype=torch.int32).npu(),
#     batch_size=4, max_seqlen_q=128, max_seqlen_k=128, topk=64,
#     layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
# )
# print_metadata("BSND + seqused_q", metadata_bsnd)

# print("=" * 60)
# print("metadata 场景 2: TND + cu_seqlens_q")
# print("=" * 60)
# metadata_tnd = torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
#     num_heads_q=8, num_heads_k=1, head_dim=128,
#     cu_seqlens_q=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
#     cu_seqlens_k=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
#     topk=64,
#     layout_q="TND", layout_k="TND", mask_mode=3, cmp_ratio=4,
# )
# print_metadata("TND + cu_seqlens_q", metadata_tnd)

# ============================================================
# 主算子 e2e 场景 (BSND layout)
# ============================================================

print()
print("=" * 60)
print("主算子场景: BSND e2e")
print("=" * 60)

B = 2
Sq = 123
G = 63
D = 512
Sk = 512
n2 = 1

torch.manual_seed(42)
q = torch.rand(B, Sq, G, D, dtype=torch.float16).npu()
key = torch.rand(B, Sk, n2, D, dtype=torch.float16).npu()
lse = torch.rand(B, n2, Sq, G, dtype=torch.float32).npu()

metadata = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q=G,
    num_heads_k=n2,
    head_dim=D,
    batch_size=B,
    max_seqlen_q=Sq,
    max_seqlen_k=Sk,
    topk=0,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
)

out = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
    q, key, lse,
    metadata=metadata,
    scale_value=1.0 / math.sqrt(D),
    cmp_ratio=1,
    mask_mode=0,
    layout_q="BSND",
    layout_k="BSND",
)
torch.npu.synchronize()

out_ref = _golden_softmax_l1_norm(q.cpu(), key.cpu(), lse.cpu(), D)
diff = (out.cpu().float() - out_ref).abs().max().item()
print(f"  output shape: {out.shape}, dtype: {out.dtype}")
for bidx in range(0, out.shape[0]):
    for s1idx in range(0, out.shape[1]):
        print(f"output[{bidx}, {s1idx}, :] value : {out[bidx, s1idx, :]}")
        print(f"outref[{bidx}, {s1idx}, :] value : {out_ref[bidx, s1idx, :]}")
print(f"  max|diff|={diff:.6f}")
torch.testing.assert_close(out.cpu().float(), out_ref, rtol=5e-3, atol=5e-3)
print("  [PASS] e2e BSND")

# ============================================================
# 异常场景测试
# ============================================================
# print()
# print("=" * 60)
# print("异常场景测试")
# print("=" * 60)

# expect_fail("query 和 key dtype 不一致", lambda: cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
#     q, key.to(torch.bfloat16), lse,
#     scale_value=0.088388, cmp_ratio=128, mask_mode=3,
#     layout_q="TND", layout_k="TND",
# ))

# expect_fail("cmp_ratio 超范围", lambda: cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
#     q, key, lse,
#     scale_value=0.088388, cmp_ratio=0, mask_mode=3,
#     layout_q="TND", layout_k="TND",
# ))

# expect_fail("mask_mode 非法", lambda: cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
#     q, key, lse,
#     scale_value=0.088388, cmp_ratio=128, mask_mode=1,
#     layout_q="TND", layout_k="TND",
# ))

# expect_fail("layout_q 非法", lambda: cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
#     q, key, lse,
#     scale_value=0.088388, cmp_ratio=128, mask_mode=3,
#     layout_q="BSH", layout_k="TND",
# ))

print()
print("All tests passed!")
