import torch, torch_npu
from cann_ops_transformer.ops import npu_flash_attn, npu_flash_attn_metadata

torch.npu.set_device(0)

B, N1, N2, S1, S2, D = 1, 1, 1, 64, 64, 128
dtype = torch.float16
dev = torch.device("npu:0")

q = torch.randn(B, N1, S1, D, dtype=dtype, device=dev)
k = torch.randn(B, N2, S2, D, dtype=dtype, device=dev)
v = torch.randn(B, N2, S2, D, dtype=dtype, device=dev)

print("[1] metadata ...")
metadata = npu_flash_attn_metadata(
    num_heads_q=N1, num_heads_kv=N2, head_dim=D,
    batch_size=B, max_seqlen_q=S1, max_seqlen_kv=S2,
    layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD")
torch.npu.synchronize()
print("  OK")

print("[2] npu_flash_attn (BNSD, no block_table) ...")
out, lse = npu_flash_attn(q, k, v, metadata=metadata,
                          layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD")
torch.npu.synchronize()
print(f"  OK  out.shape={out.shape}")

print("[3] npu_flash_attn (BNSD + block_table=tensor) ...")
bt = torch.zeros(B, 4, dtype=torch.int32, device=dev)
try:
    out2, lse2 = npu_flash_attn(q, k, v, block_table=bt, metadata=metadata,
                                layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD")
    torch.npu.synchronize()
    print("  NOT REJECTED - 拦截未生效!")
except Exception as e:
    print(f"  REJECTED: {e}")
