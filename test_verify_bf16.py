import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math, struct
torch_npu.npu.set_device(0)

# Reproduce golden's data generation
q_seq = 8192; k_seq = 8192; q_head_num = 64; k_head_num = 1; head_dim = 128
block_size = 1024; block_num = 10591
qk_dtype = torch.bfloat16; weight_dtype = torch.bfloat16
sparse_count = 512; sparse_mode = 3
query_datarange = [-100, 100]
key_datarange = [-100, 100]
weights_datarange = [-130, 130]

# Run 3 times with different random seeds (like pytest does)
for run in range(3):
    np.random.seed(run)  # different seed each run

    query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (q_seq, q_head_num, head_dim))).to(qk_dtype).npu()
    weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (q_seq, q_head_num))).to(weight_dtype).npu()
    key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (block_num, block_size, k_head_num, head_dim))).to(qk_dtype).npu()
    bt = torch.tensor([[i for i in range(math.ceil(k_seq/block_size))]], dtype=torch.int32).npu()
    asq = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
    ask = torch.tensor([float(k_seq)], dtype=torch.int32).npu()

    r, v = torch_npu.npu_lightning_indexer(query, key, weights,
        actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
        block_table=bt, layout_query="TND", layout_key="PA_BSND",
        sparse_count=sparse_count, sparse_mode=sparse_mode)
    torch.npu.synchronize()

    # Also run golden on CPU with same data
    # Golden: compute scores, sort -score ascending (select largest), take top-K
    # We need to compute scores the same way golden does
    # But that's complex (bmm, relu, mask, etc.)
    # Instead, let's just analyze the NPU output distribution

    r_np = r.cpu().numpy()
    valid_counts = []
    for b in range(r_np.shape[0]):
        valid = r_np[b, 0][r_np[b, 0] != -1]
        valid_counts.append(len(valid))

    valid_counts = np.array(valid_counts)
    print(f"\n=== Run {run} (np.random.seed={run}) ===")
    print(f"  Total valid: {valid_counts.sum()}/{r_np.shape[0]*sparse_count}")
    print(f"  Batches with 512 valid: {(valid_counts == 512).sum()}")
    print(f"  Batches with <512 valid: {(valid_counts < 512).sum()}")
    print(f"  Batches with 1 valid: {(valid_counts == 1).sum()}")

    # Sample some batches
    for b in [0, 100, 511, 512, 600, 1000, 5000]:
        vals = r_np[b, 0]
        valid = vals[vals != -1]
        print(f"  batch {b}: {len(valid)} valid, first5={valid[:5].tolist()}")

    np.save(f'/tmp/radix_run{run}.npy', r_np)
