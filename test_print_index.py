import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math
torch_npu.npu.set_device(1)

# Match ref paramset exactly
q_seq = 3; k_seq = 3; q_t_size = 1; k_t_size = 1
q_head_num = 16; k_head_num = 1; head_dim = 128
block_size = 1024; block_num = 1
qk_dtype = torch.bfloat16; weight_dtype = torch.bfloat16
sparse_count = 2048; sparse_mode = 3
query_datarange = [-100, 100]; key_datarange = [-100, 100]; weights_datarange = [-130, 130]

np.random.seed(42)
# Golden uses np.random.uniform for all tensors
query_np = np.random.uniform(-100, 100, (q_t_size, q_head_num, head_dim))
weights_np = np.random.uniform(-130, 130, (q_t_size, q_head_num))
key_np = np.random.uniform(-100, 100, (block_num, block_size, k_head_num, head_dim))

query = torch.tensor(query_np).to(qk_dtype).npu()
weights = torch.tensor(weights_np).to(weight_dtype).npu()
key = torch.tensor(key_np).to(qk_dtype).npu()
bt = torch.tensor([[0]], dtype=torch.int32).npu()
asq = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
ask = torch.tensor([float(k_seq)], dtype=torch.int32).npu()

r, v = torch_npu.npu_lightning_indexer(query, key, weights,
    actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
    block_table=bt, layout_query="TND", layout_key="PA_BSND",
    sparse_count=sparse_count, sparse_mode=sparse_mode)
torch.npu.synchronize()

npu_result = r.cpu().numpy()
print(f"NPU shape: {npu_result.shape}, valid: {len(npu_result[npu_result != -1])}")

# Compute golden on CPU
query_f32 = torch.tensor(query_np.astype(np.float32))
key_f32 = torch.tensor(key_np.astype(np.float32))
weights_f32 = torch.tensor(weights_np.astype(np.float32))

g = q_head_num // k_head_num  # 16
for s1Idx in range(q_seq):
    cuRealAcSeq = s1Idx + 1
    q = query_f32[0, :, :]  # (16, 128) - all heads for batch 0
    # Actually golden uses per-s1 query: query_f32[s1Idx]
    # But q_t_size=1, so query_f32 shape is (1, 16, 128)
    # golden does: cur_q = q_bnsd_tensor[b_idx, :, :curr_actualSeq_q, :]
    # For TND with q_t_size=1, all s1 are in one batch
    pass

# Simpler: just compare NPU output with expected
# For sparse_mode=3, cuRealAcSeq = s1Idx+1
# sparse_count=2048 > cuRealAcSeq, so all valid indices are selected
for s1Idx in range(q_seq):
    cuRealAcSeq = s1Idx + 1
    print(f"\n=== s1Idx={s1Idx}, cuRealAcSeq={cuRealAcSeq} ===")
    npu_vals = npu_result[s1Idx, 0]
    valid = npu_vals[npu_vals != -1]
    valid_sorted = sorted(valid.tolist())
    print(f"  NPU: {len(valid)} valid, sorted: {valid_sorted}")
    if cuRealAcSeq <= sparse_count:
        expected = list(range(cuRealAcSeq))
        print(f"  Expected: {expected}")
        print(f"  Match: {valid_sorted == expected}")
