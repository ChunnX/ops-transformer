import torch
import torch_npu
import cann_ops_transformer


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

print()
print("All tests passed!")
