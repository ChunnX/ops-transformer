import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math
torch_npu.npu.set_device(0)

q_seq = 8192; k_seq = 8192; q_head_num = 64; k_head_num = 1; head_dim = 128
block_size = 1024; block_num = 10591
sparse_count = 512; sparse_mode = 3

np.random.seed(42)
query_np = np.random.uniform(-100, 100, (q_seq, q_head_num, head_dim)).astype(np.float32)
weights_np = np.random.uniform(-130, 130, (q_seq, q_head_num)).astype(np.float32)
key_np = np.random.uniform(-100, 100, (block_num, block_size, k_head_num, head_dim)).astype(np.float32)

query = torch.tensor(query_np).to(torch.bfloat16).npu()
weights = torch.tensor(weights_np).to(torch.bfloat16).npu()
key = torch.tensor(key_np).to(torch.bfloat16).npu()
bt = torch.tensor([[i for i in range(math.ceil(k_seq/block_size))]], dtype=torch.int32).npu()
asq = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
ask = torch.tensor([float(k_seq)], dtype=torch.int32).npu()
r, v = torch_npu.npu_lightning_indexer(query, key, weights,
    actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
    block_table=bt, layout_query="TND", layout_key="PA_BSND",
    sparse_count=sparse_count, sparse_mode=sparse_mode)
torch.npu.synchronize()
npu_result = r.cpu().numpy()

query_f32 = torch.tensor(query_np)
key_f32 = torch.tensor(key_np)
weights_f32 = torch.tensor(weights_np)

CLAMP_VAL = 1e8

for b in [600, 1000, 2000, 5000]:
    s1Idx = b
    cuRealAcSeq = s1Idx + 1
    q = query_f32[s1Idx]
    k_flat = key_f32[:k_seq, 0, :].reshape(k_seq, head_dim)
    qk = torch.mm(q, k_flat.t())
    qk_relu = qk.clamp(min=0)
    w = weights_f32[s1Idx]
    scores = w.unsqueeze(1) * qk_relu
    score_reduced = scores.sum(dim=0)
    score_masked = score_reduced[:cuRealAcSeq].clone()

    npu_idx = npu_result[b, 0]
    npu_valid = np.sort(npu_idx[npu_idx != -1])
    npu_set = set(npu_valid.tolist())

    # Golden float topK
    golden_topk = torch.argsort(-score_masked, stable=True)[:sparse_count]
    golden_set = set(golden_topk.tolist())

    # bf16 topK (no clamp)
    bf16_topk = torch.argsort(-score_masked.to(torch.bfloat16).to(torch.float32), stable=True)[:sparse_count]
    bf16_set = set(bf16_topk.tolist())

    # Mins(1e8) + bf16 topK
    score_clamped = score_masked.clone()
    score_clamped[score_clamped > CLAMP_VAL] = CLAMP_VAL
    clamped_bf16_topk = torch.argsort(-score_clamped.to(torch.bfloat16).to(torch.float32), stable=True)[:sparse_count]
    clamped_set = set(clamped_bf16_topk.tolist())

    n_changed = (score_clamped != score_masked).sum().item()
    unique_float = len(torch.unique(score_masked))
    unique_bf16 = len(torch.unique(score_masked.to(torch.bfloat16)))

    print(f"batch {b} (cuRealAcSeq={cuRealAcSeq}):")
    print(f"  Score range: [{score_masked.min():.2f}, {score_masked.max():.2f}]")
    print(f"  Unique: float={unique_float}, bf16={unique_bf16}, collision={unique_float-unique_bf16}")
    print(f"  Mins({CLAMP_VAL:.0g}) changed {n_changed} scores")
    print(f"  Golden vs NPU:      {len(golden_set & npu_set):3d}/{sparse_count}")
    print(f"  Golden vs bf16:     {len(golden_set & bf16_set):3d}/{sparse_count}")
    print(f"  Golden vs Mins_bf16:{len(golden_set & clamped_set):3d}/{sparse_count}")
    print(f"  NPU vs bf16:        {len(npu_set & bf16_set):3d}/{sparse_count}")
    print(f"  NPU vs Mins_bf16:   {len(npu_set & clamped_set):3d}/{sparse_count}")
