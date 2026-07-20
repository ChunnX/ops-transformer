import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math
torch_npu.npu.set_device(0)

q_seq = 1024; k_seq = 1024; q_head_num = 64; k_head_num = 1; head_dim = 128
block_size = 1024; block_num = math.ceil(k_seq / block_size)
qk_dtype = torch.bfloat16; weight_dtype = torch.bfloat16
sparse_count = 128; sparse_mode = 3
query_datarange = [-100, 100]
key_datarange = [-100, 100]
weights_datarange = [-130, 130]

for run in range(3):
    np.random.seed(run * 42 + 1)
    query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (q_seq, q_head_num, head_dim))).to(qk_dtype).npu()
    weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (q_seq, q_head_num))).to(weight_dtype).npu()
    key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (block_num, block_size, k_head_num, head_dim))).to(qk_dtype).npu()
    bt = torch.tensor([[i for i in range(block_num)]], dtype=torch.int32).npu()
    asq = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
    ask = torch.tensor([float(k_seq)], dtype=torch.int32).npu()
    r, v = torch_npu.npu_lightning_indexer(query, key, weights, actual_seq_lengths_query=asq, actual_seq_lengths_key=ask, block_table=bt, layout_query="TND", layout_key="PA_BSND", sparse_count=sparse_count, sparse_mode=sparse_mode)
    torch.npu.synchronize()
    np.save(f'/tmp/radix_small_{run}.npy', r.cpu().numpy())
    valid = len(r[r!=-1])
    print(f"Run {run}: valid={valid}, batch500={r[500,0,:5].tolist()}")
