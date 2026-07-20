import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math
torch_npu.npu.set_device(0)

q_seq = 8192; k_seq = 8192; q_head_num = 64; k_head_num = 1; head_dim = 128
block_size = 1024; block_num = 10591
qk_dtype = torch.bfloat16; weight_dtype = torch.bfloat16
sparse_count = 512; sparse_mode = 3

# Simulate golden: compute scores on CPU, sort, take top-K
# Then compare with NPU RadixTopK output
np.random.seed(42)

query_np = np.random.uniform(-100, 100, (q_seq, q_head_num, head_dim)).astype(np.float32)
weights_np = np.random.uniform(-130, 130, (q_seq, q_head_num)).astype(np.float32)
key_np = np.random.uniform(-100, 100, (block_num, block_size, k_head_num, head_dim)).astype(np.float32)

# Run on NPU
query = torch.tensor(query_np).to(qk_dtype).npu()
weights = torch.tensor(weights_np).to(weight_dtype).npu()
key = torch.tensor(key_np).to(qk_dtype).npu()
bt = torch.tensor([[i for i in range(math.ceil(k_seq/block_size))]], dtype=torch.int32).npu()
asq = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
ask = torch.tensor([float(k_seq)], dtype=torch.int32).npu()
r, v = torch_npu.npu_lightning_indexer(query, key, weights,
    actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
    block_table=bt, layout_query="TND", layout_key="PA_BSND",
    sparse_count=sparse_count, sparse_mode=sparse_mode)
torch.npu.synchronize()
npu_result = r.cpu().numpy()

# Compute golden for a few batches
# Golden: qk_bmm = Q @ K^T, relu, weight * qk, mask, sort -score ascending (largest first), take top-K
query_f32 = torch.tensor(query_np)
key_f32 = torch.tensor(key_np)
weights_f32 = torch.tensor(weights_np)

# Build mask for sparse_mode=3 (rightDownCausal)
# cuRealAcSeq = s1Idx + 1 for each s1
# mask[i,j] = True if j < cuRealAcSeq[i] (valid)

g = q_head_num // k_head_num  # 64
match_count = 0
total_count = 0
float_bf16_match = 0
float_bf16_total = 0

for b in [600, 1000, 2000, 5000]:  # sample batches
    s1Idx = b
    cuRealAcSeq = s1Idx + 1

    # Compute score for this s1
    # Q[s1Idx] shape: (q_head_num, head_dim) = (64, 128)
    # K shape: (k_seq, k_head_num, head_dim) = (8192, 1, 128)
    q = query_f32[s1Idx]  # (64, 128)
    k_flat = key_f32[:k_seq, 0, :].reshape(k_seq, head_dim)  # (8192, 128)

    # qk = Q @ K^T: (64, 128) @ (128, 8192) = (64, 8192)
    qk = torch.mm(q, k_flat.t())  # (64, 8192)
    qk_relu = qk.clamp(min=0)  # relu

    # weight * qk: weights (64,) broadcast to (64, 8192)
    w = weights_f32[s1Idx]  # (64,)
    scores = w.unsqueeze(1) * qk_relu  # (64, 8192)

    # Reduce across heads (group reduce): sum over groups
    # g=64, k_head_num=1, so reduce all 64 heads into 1
    score_reduced = scores.sum(dim=0)  # (8192,)

    # Apply mask
    score_masked = score_reduced[:cuRealAcSeq].clone()
    # No mask needed for valid positions, just limit to cuRealAcSeq

    # Golden: sort -score ascending (largest score first), take top-K
    neg_score = -score_masked
    sorted_idx = torch.argsort(neg_score, stable=True)
    golden_topk = sorted_idx[:sparse_count].sort().values  # sort for comparison

    # NPU result
    npu_idx = npu_result[b, 0]
    npu_valid = np.sort(npu_idx[npu_idx != -1])

    # Compare
    golden_set = set(golden_topk.tolist())
    npu_set = set(npu_valid.tolist())
    overlap = len(golden_set & npu_set)

    # Also compute bf16 topK
    score_bf16 = score_masked.to(torch.bfloat16).to(torch.float32)
    neg_bf16 = -score_bf16
    sorted_bf16 = torch.argsort(neg_bf16, stable=True)
    bf16_topk = sorted_bf16[:sparse_count].sort().values
    bf16_set = set(bf16_topk.tolist())

    float_bf16_overlap = len(golden_set & bf16_set)

    print(f"batch {b} (cuRealAcSeq={cuRealAcSeq}):")
    print(f"  Golden vs NPU overlap: {overlap}/{sparse_count}")
    print(f"  Golden vs bf16 overlap: {float_bf16_overlap}/{sparse_count}")
    print(f"  NPU vs bf16 overlap: {len(npu_set & bf16_set)}/{sparse_count}")

    # Check score distribution
    valid_scores = score_masked.numpy()
    print(f"  Score range: [{valid_scores.min():.2f}, {valid_scores.max():.2f}]")
    # Check bf16 collision
    bf16_scores = score_masked.to(torch.bfloat16)
    unique_float = len(torch.unique(score_masked))
    unique_bf16 = len(torch.unique(bf16_scores))
    print(f"  Unique float scores: {unique_float}, unique bf16 scores: {unique_bf16}, collision: {unique_float - unique_bf16}")
