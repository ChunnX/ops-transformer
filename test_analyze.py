import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math
torch_npu.npu.set_device(0)
torch.manual_seed(42)
np.random.seed(42)

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

# Analyze batch 8191 (cuRealAcSeq=8192, full RadixTopK)
batch_idx = 8191
batch_result = r.flatten()[batch_idx*512:(batch_idx+1)*512]
valid = batch_result[batch_result != -1].sort()[0]
print(f"Batch {batch_idx}: {len(valid)} valid indices")
print(f"  Sort result first 20: {valid[:20].tolist()}")
print(f"  Sort result range: [{valid.min().item()}, {valid.max().item()}]")

# Now check with Sort version result
sort_result = np.load('/tmp/sort_result.npy')
sort_batch = sort_result.flatten()[batch_idx*512:(batch_idx+1)*512]
sort_valid = sort_batch[sort_batch != -1]
sort_valid_sorted = np.sort(sort_valid)
print(f"\nSort batch {batch_idx}: {len(sort_valid)} valid indices")
print(f"  Sort result first 20: {sort_valid_sorted[:20].tolist()}")
print(f"  Sort result range: [{sort_valid_sorted.min()}, {sort_valid_sorted.max()}]")

# Check overlap
radix_set = set(valid.tolist())
sort_set = set(sort_valid.tolist())
overlap = len(radix_set & sort_set)
print(f"\nOverlap: {overlap}/{max(len(radix_set), len(sort_set))}")

# Check if bf16 precision is the issue
# The scores are mm1*weight reduced. Let's see how many unique bf16 values there are
# by looking at the index distribution
print(f"\nRadixTopK indices span: {valid.max().item() - valid.min().item()}")
print(f"Sort indices span: {sort_valid_sorted.max() - sort_valid_sorted.min()}")
