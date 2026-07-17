import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math, sys
torch_npu.npu.set_device(0)

# Small test: batch=1, q_seq=128, k_seq=512
batch_size = 1
q_seq = 128
k_seq = 512
q_head_num = 64
k_head_num = 1
head_dim = 128
block_size = 128
block_num = 100
qk_dtype = torch.bfloat16
weight_dtype = torch.bfloat16

query = torch.randn(q_seq, q_head_num, head_dim, dtype=qk_dtype).npu()
weights = torch.randn(q_seq, q_head_num, dtype=weight_dtype).npu()
key = torch.randn(block_num, block_size, k_head_num, head_dim, dtype=qk_dtype).npu()
block_table = torch.tensor([[i for i in range(math.ceil(k_seq/block_size))]], dtype=torch.int32).npu()
actual_seq_q = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
actual_seq_k = torch.tensor([float(k_seq)], dtype=torch.int32).npu()

print("Running lightning_indexer...", flush=True)
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
print(f"Done! idx shape: {r.shape}", flush=True)
print(f"has -1: {(r == -1).any().item()}")
print(f"min: {r.min().item()} max: {r.max().item()}")
print(f"first 20: {r.flatten()[:20].tolist()}")
