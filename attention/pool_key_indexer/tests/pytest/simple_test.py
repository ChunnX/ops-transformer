import torch
import torch_npu

torch_npu.npu.set_device(2)

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

# ---------- PA_BBND 场景 ----------
# LIV2 语义: PA key (blockNum, blockSize, N2, D) 块内每行 = 1 个池(key 不做 token 展开);
# actual_seq_k 为每 batch 池数(非前缀和); PA+BSND -> actual_seq_q 必须为 None
B, S1, G, D = 2, 32, 8, 128
block_size, pool = 16, 16
k_pool_lens = [4, 6]  # 每 batch 池数
blocks_per_batch = [(p_ + block_size - 1) // block_size for p_ in k_pool_lens]
block_num = sum(blocks_per_batch)

query_p = torch.randn(B, S1, G, D, dtype=torch.float16).npu()
weights_p = torch.randn(B, S1, G, dtype=torch.float16).npu()
pool_tail_k_p = torch.randint(0, pool, (B,), dtype=torch.int64).npu()
actual_seq_k_p = torch.tensor(k_pool_lens, dtype=torch.int64).npu()

# 每 batch 的池独立连续填块, 物理块随机排布, 尾块补零
pool_key_cpu = torch.randn(sum(k_pool_lens), 1, D, dtype=torch.float16)
blocks = torch.zeros(block_num, block_size, 1, D, dtype=torch.float16)
ps_, bs_ = 0, 0
for b, nb in enumerate(blocks_per_batch):
    seg = pool_key_cpu[ps_ : ps_ + k_pool_lens[b]]
    pad = nb * block_size - k_pool_lens[b]
    if pad:
        seg = torch.cat([seg, torch.zeros(pad, 1, D, dtype=seg.dtype)], dim=0)
    blocks[bs_ : bs_ + nb] = seg.view(nb, block_size, 1, D)
    ps_ += k_pool_lens[b]
    bs_ += nb
perm = torch.randperm(block_num)
pool_key_p = blocks[perm].contiguous().npu()

# block_table: 每 batch 的物理块映射, 空槽 -1
max_blocks = max(blocks_per_batch)
bt = torch.full((B, max_blocks), -1, dtype=torch.int32)
cur = 0
for b, nb in enumerate(blocks_per_batch):
    for i in range(nb):
        bt[b, i] = perm[cur]
        cur += 1
block_table = bt.npu()

print("=== BSND/PA_BBND fp16 pool_size=16 causal val=True ===")
print(
    f"query {query_p.shape}, pool_key {pool_key_p.shape}, block_table {block_table.shape}"
)

indices_p, values_p = pool_key_indexer(
    query_p,
    pool_key_p,
    weights_p,
    pool_tail_k_p,
    actual_seq_k=actual_seq_k_p,
    block_table=block_table,
    layout_q="BSND",
    layout_k="PA_BBND",
    topk=128,
    pool_size=pool,
    mask_mode=3,
    return_value=True,
)
print(f"indices {indices_p.shape} {indices_p.dtype}")
print(f"values  {values_p.shape} {values_p.dtype}")
print("PASSED\n")

# TND query + PA_BBND key, fp16, pool_size=16, non-causal, no value
# PA+TND: actual_seq_q 必须非空(前缀和); 输出为 TND 布局 (T1, topk+ps-1)
q_lens_pa = [16, 16]
total_q_pa = sum(q_lens_pa)
query_tp = torch.randn(total_q_pa, 8, D, dtype=torch.float16).npu()
weights_tp = torch.randn(total_q_pa, 8, dtype=torch.float16).npu()
pool_tail_k_tp = torch.randint(0, pool, (2,), dtype=torch.int64).npu()
actual_seq_q_tp = torch.tensor(q_lens_pa, dtype=torch.int64).cumsum(0).npu()

print("=== TND/PA_BBND fp16 pool_size=16 nocausal noval ===")
print(
    f"query {query_tp.shape}, pool_key {pool_key_p.shape}, block_table {block_table.shape}"
)

indices_tp, values_tp = pool_key_indexer(
    query_tp,
    pool_key_p,
    weights_tp,
    pool_tail_k_tp,
    actual_seq_q=actual_seq_q_tp,
    actual_seq_k=actual_seq_k_p,
    block_table=block_table,
    layout_q="TND",
    layout_k="PA_BBND",
    topk=128,
    pool_size=pool,
    mask_mode=0,
    return_value=False,
)
print(f"indices {indices_tp.shape} {indices_tp.dtype}")
print(f"values  {values_tp.shape} {values_tp.dtype}")
print("PASSED\n")

# 全局同步: aclnn 异步执行, 不等完就退出会把设备侧错误吞掉
torch_npu.npu.synchronize()

# 输出有效性抽查: PA 用例的 indices 不应是未初始化垃圾
assert (indices_p.cpu() >= 0).sum() > 0, "BSND/PA_BBND indices all invalid"
assert (indices_tp.cpu() >= 0).sum() > 0, "TND/PA_BBND indices all invalid"

print("ALL DONE")
