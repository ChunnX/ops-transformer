import torch
import torch_npu

torch_npu.npu.set_device(0)

from cann_ops_transformer import pool_key_indexer

# BSND, fp16, pool_size=1, non-causal, no value
B, S1, N1, D = 1, 64, 8, 128
K_SEQ, N2 = 256, 1

query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
pool_key = torch.randn(B, K_SEQ // 1, N2, D, dtype=torch.float16).npu()
weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
pool_tail_k = torch.randint(0, 1, (B,), dtype=torch.int64).npu()

print("=== BSND fp16 pool_size=1 ===")
print(
    f"query {query.shape}, pool_key {pool_key.shape}, weights {weights.shape}, pool_tail_k {pool_tail_k.shape}"
)

indices, values = pool_key_indexer(
    query,
    pool_key,
    weights,
    pool_tail_k,
    topk=128,
    pool_size=1,
    mask_mode=0,
    return_value=False,
)
print(f"indices {indices.shape} {indices.dtype}")
print(f"values  {values.shape} {values.dtype}")
print("PASSED\n")

# TND, fp16, pool_size=16, causal, with value
q_lens = [32, 32]
k_lens = [64, 64]
total_q = sum(q_lens)
total_k_pools = sum(k // 16 for k in k_lens)

query_t = torch.randn(total_q, 8, D, dtype=torch.float16).npu()
pool_key_t = torch.randn(total_k_pools, N2, D, dtype=torch.float16).npu()
weights_t = torch.randn(total_q, 8, dtype=torch.float16).npu()
pool_tail_k_t = torch.randint(0, 16, (2,), dtype=torch.int64).npu()
actual_seq_q = torch.tensor([32, 64], dtype=torch.int64).npu()
# actual_seq_k 为累计 pool 数（TND 下 last 必须等于 pool_key 的 T 维）
actual_seq_k = (
    torch.tensor([k // 16 for k in k_lens], dtype=torch.int64).cumsum(0).npu()
)

print("=== TND fp16 pool_size=16 causal val=True ===")
print(f"query {query_t.shape}, pool_key {pool_key_t.shape}, weights {weights_t.shape}")

indices_t, values_t = pool_key_indexer(
    query_t,
    pool_key_t,
    weights_t,
    pool_tail_k_t,
    actual_seq_q=actual_seq_q,
    actual_seq_k=actual_seq_k,
    layout_q="TND",
    layout_k="TND",
    topk=128,
    pool_size=16,
    mask_mode=3,
    return_value=True,
)
print(f"indices {indices_t.shape} {indices_t.dtype}")
print(f"values  {values_t.shape} {values_t.dtype}")
print("PASSED\n")

print("ALL DONE")
