import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, sys, io, contextlib, math
torch_npu.npu.set_device(0)

# Fixed seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters from li_default_a2
batch_size = 1
q_seq = 8192
k_seq = 8192
q_head_num = 64
k_head_num = 1
head_dim = 128
block_size = 1024
block_num = 10591
qk_dtype = torch.bfloat16
weight_dtype = torch.bfloat16

query = torch.randn(q_seq, q_head_num, head_dim, dtype=qk_dtype).npu()
weights = torch.randn(q_seq, q_head_num, dtype=weight_dtype).npu()
key = torch.randn(block_num, block_size, k_head_num, head_dim, dtype=qk_dtype).npu()
block_table = torch.tensor([[i for i in range(math.ceil(k_seq/block_size))]], dtype=torch.int32).npu()
actual_seq_q = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
actual_seq_k = torch.tensor([float(k_seq)], dtype=torch.int32).npu()

r, v = torch_npu.npu_lightning_indexer(
    query, key, weights,
    actual_seq_lengths_query=actual_seq_q,
    actual_seq_lengths_key=actual_seq_k,
    block_table=block_table,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3)
torch.npu.synchronize()

# Save result
np.save('/tmp/sort_result.npy', r.cpu().numpy())
print(f"Sort: shape={r.shape}, has_neg1={(r==-1).any().item()}, min={r.min().item()}, max={r.max().item()}")
valid = r[r != -1]
print(f"  valid_count={len(valid.flatten())}")
print(f"  first 20: {r.flatten()[:20].tolist()}")

# Check some batches
for b in [0, 100, 500, 1000, 5000, 8000, 8191]:
    vals = r.flatten()[b*512:(b+1)*512]
    v_valid = vals[vals != -1]
    print(f"  batch {b}: {len(v_valid)} valid, first 5={v_valid[:5].tolist()}")
