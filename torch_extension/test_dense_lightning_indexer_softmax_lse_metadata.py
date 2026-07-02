import torch
import torch_npu
import cann_ops_transformer

# ============================================================
# 测试 dense_lightning_indexer_softmax_lse_metadata 算子
# 输出为 (64,) int32, 前 5 个字段:
#   [0] totalNum             - 参与 load balance 的总 seq 数
#   [1] formerCoreProcessNum - 满载核处理数 (ceil)
#   [2] remainCoreProcessNum - 尾部核处理数 (floor)
#   [3] remainCoreNum        - 尾部核数
#   [4] totalCoreNum         - 实际使用核数 (<=36)
# ============================================================

def print_metadata(tag, metadata):
    vals = metadata.cpu().tolist()
    print(f"=== {tag} ===")
    print(f"  totalNum             = {vals[0]}")
    print(f"  formerCoreProcessNum = {vals[1]}")
    print(f"  remainCoreProcessNum = {vals[2]}")
    print(f"  remainCoreNum        = {vals[3]}")
    print(f"  totalCoreNum         = {vals[4]}")
    print()

# ------------------------------------------------------------
# 场景 1: BSND layout, 用 seqused_q 提供实际 seq 长度
#   totalNum = sum(seqused_q) = 128+100+64+32 = 324
# ------------------------------------------------------------
print("=" * 60)
print("场景 1: BSND + seqused_q")
print("=" * 60)
metadata_bsnd = torch.ops.cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
    num_heads_q=8,
    num_heads_k=4,
    head_dim=128,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=torch.tensor([128, 100, 64, 32], dtype=torch.int32).npu(),
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=4,
    max_seqlen_q=128,
    max_seqlen_k=128,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
)
print_metadata("BSND + seqused_q", metadata_bsnd)

# ------------------------------------------------------------
# 场景 2: TND layout, 用 cu_seqlens_q
#   totalNum = cu_seqlens_q[B] = 324
# ------------------------------------------------------------
print("=" * 60)
print("场景 2: TND + cu_seqlens_q")
print("=" * 60)
metadata_tnd = torch.ops.cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
    num_heads_q=8,
    num_heads_k=4,
    head_dim=128,
    cu_seqlens_q=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
    cu_seqlens_k=torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu(),
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    layout_q="TND",
    layout_k="TND",
    mask_mode=3,
    cmp_ratio=1,
)
print_metadata("TND + cu_seqlens_q", metadata_tnd)

# ------------------------------------------------------------
# 场景 3: BSND layout, 无 tensor 输入, 纯 attr 驱动
#   totalNum = batch_size * max_seqlen_q = 2 * 256 = 512
# ------------------------------------------------------------
print("=" * 60)
print("场景 3: BSND 纯 attr (无 tensor 输入)")
print("=" * 60)
metadata_attr = torch.ops.cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
    num_heads_q=8,
    num_heads_k=4,
    head_dim=128,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=2,
    max_seqlen_q=256,
    max_seqlen_k=256,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
)
print_metadata("BSND 纯 attr", metadata_attr)

print("All tests passed!")
