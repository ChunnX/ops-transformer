import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math
torch_npu.npu.set_device(0)

q_seq = 1024; k_seq = 1024; q_head_num = 64; k_head_num = 1; head_dim = 128
block_size = 1024; block_num = math.ceil(k_seq / block_size)
qk_dtype = torch.bfloat16; weight_dtype = torch.bfloat16
sparse_count = 128; sparse_mode = 3

torch.manual_seed(42); np.random.seed(42)
query = torch.randn(q_seq, q_head_num, head_dim, dtype=qk_dtype).npu()
weights = torch.randn(q_seq, q_head_num, dtype=weight_dtype).npu()
key = torch.randn(block_num, block_size, k_head_num, head_dim, dtype=qk_dtype).npu()
block_table = torch.tensor([[i for i in range(block_num)]], dtype=torch.int32).npu()
actual_seq_q = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
actual_seq_k = torch.tensor([float(k_seq)], dtype=torch.int32).npu()
r, v = torch_npu.npu_lightning_indexer(query, key, weights, actual_seq_lengths_query=actual_seq_q, actual_seq_lengths_key=actual_seq_k, block_table=block_table, layout_query="TND", layout_key="PA_BSND", sparse_count=sparse_count, sparse_mode=sparse_mode)
torch.npu.synchronize()
np.save('/tmp/sort_small.npy', r.cpu().numpy())
print(f"Sort: shape={r.shape}, valid={len(r[r!=-1])}")
