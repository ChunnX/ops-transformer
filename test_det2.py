import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math
torch_npu.npu.set_device(0)

q_seq = 8192; k_seq = 8192; q_head_num = 64; k_head_num = 1; head_dim = 128
block_size = 1024; block_num = 10591
sparse_count = 512; sparse_mode = 3

np.random.seed(42)
query = torch.tensor(np.random.uniform(-100, 100, (q_seq, q_head_num, head_dim))).to(torch.bfloat16).npu()
weights = torch.tensor(np.random.uniform(-130, 130, (q_seq, q_head_num))).to(torch.bfloat16).npu()
key = torch.tensor(np.random.uniform(-100, 100, (block_num, block_size, k_head_num, head_dim))).to(torch.bfloat16).npu()
bt = torch.tensor([[i for i in range(math.ceil(k_seq/block_size))]], dtype=torch.int32).npu()
asq = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
ask = torch.tensor([float(k_seq)], dtype=torch.int32).npu()

results = []
for run in range(5):
    r, v = torch_npu.npu_lightning_indexer(query, key, weights,
        actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
        block_table=bt, layout_query="TND", layout_key="PA_BSND",
        sparse_count=sparse_count, sparse_mode=sparse_mode)
    torch.npu.synchronize()
    r_np = r.cpu().numpy()
    results.append(r_np)
    valid = len(r_np[r_np != -1])
    if run == 0:
        print(f"Run {run}: valid={valid}")
    else:
        diff = (results[0] != r_np).sum()
        print(f"Run {run}: valid={valid}, diff={diff}")
