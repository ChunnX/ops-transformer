"""
验证 npu_flash_attn 拦截：非 PA 模式下传入 block_table 应报错。
用法: python test_block_table_guard.py
"""

import torch
import torch_npu
from cann_ops_transformer.ops import npu_flash_attn, npu_flash_attn_metadata

torch.npu.set_device(0)

B, N1, N2, S1, S2, D = 2, 8, 1, 64, 128, 128
device = torch.device("npu:0")
dtype = torch.float16

q = torch.randn(B, N1, S1, D, dtype=dtype, device=device)
k = torch.randn(B, N2, S2, D, dtype=dtype, device=device)
v = torch.randn(B, N2, S2, D, dtype=dtype, device=device)
block_table = torch.zeros(B, 4, dtype=torch.int32, device=device)

metadata = npu_flash_attn_metadata(
    num_heads_q=N1, num_heads_kv=N2, head_dim=D,
    batch_size=B, max_seqlen_q=S1, max_seqlen_kv=S2,
    layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD",
)

# --- Case 1: 非 PA + block_table=None (应正常通过) ---
print("[Case 1] BNSD + block_table=None (should PASS)")
try:
    out, lse = npu_flash_attn(q, k, v, metadata=metadata,
                              layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD")
    print("  -> PASS (no error)")
except Exception as e:
    print(f"  -> UNEXPECTED FAIL: {e}")

# --- Case 2: 非 PA + block_table=实际tensor (应被拦截) ---
print("[Case 2] BNSD + block_table=tensor (should be REJECTED)")
try:
    out, lse = npu_flash_attn(q, k, v, block_table=block_table, metadata=metadata,
                              layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD")
    print("  -> NOT REJECTED (拦截未生效!)")
except Exception as e:
    print(f"  -> REJECTED as expected: {e}")

# --- Case 3: PA 模式 + block_table=实际tensor (应正常通过) ---
print("[Case 3] PA_BBND + block_table=tensor (should PASS)")
k_pa = torch.randn(8, 128, N2, D, dtype=dtype, device=device)
v_pa = torch.randn(8, 128, N2, D, dtype=dtype, device=device)
bt_pa = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
seqused_kv = torch.tensor([S2, S2], dtype=torch.int32, device=device)

metadata_pa = npu_flash_attn_metadata(
    num_heads_q=N1, num_heads_kv=N2, head_dim=D,
    batch_size=B, max_seqlen_q=S1, max_seqlen_kv=S2,
    layout_q="BNSD", layout_kv="PA_BBND", layout_out="BNSD",
)
try:
    out, lse = npu_flash_attn(q, k_pa, v_pa, block_table=bt_pa,
                              seqused_kv=seqused_kv, metadata=metadata_pa,
                              layout_q="BNSD", layout_kv="PA_BBND", layout_out="BNSD")
    print("  -> PASS (no error)")
except Exception as e:
    print(f"  -> UNEXPECTED FAIL: {e}")

# --- Case 4: PA 模式 + block_table=None (应被拦截) ---
print("[Case 4] PA_BBND + block_table=None (should be REJECTED)")
try:
    out, lse = npu_flash_attn(q, k_pa, v_pa, metadata=metadata_pa,
                              seqused_kv=seqused_kv,
                              layout_q="BNSD", layout_kv="PA_BBND", layout_out="BNSD")
    print("  -> NOT REJECTED (PA 缺 block_table 未拦截!)")
except Exception as e:
    print(f"  -> REJECTED as expected: {e}")
