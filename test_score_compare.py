import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math, struct
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

# Read NPU GM scores for batch 600 (first 512)
vals = r[600, 0].cpu().tolist()
npu_scores_0_511 = np.array([struct.unpack('f', struct.pack('i', v))[0] for v in vals], dtype=np.float32)

# CPU golden scores for batch 600
query_f32 = torch.tensor(query_np)
key_f32 = torch.tensor(key_np)
weights_f32 = torch.tensor(weights_np)

b = 600
s1Idx = b
cuRealAcSeq = s1Idx + 1  # 601
q = query_f32[s1Idx]  # (64, 128)
k_flat = key_f32[:k_seq, 0, :].reshape(k_seq, head_dim)  # (8192, 128)
qk = torch.mm(q, k_flat.t())  # (64, 8192)
qk_relu = qk.clamp(min=0)
w = weights_f32[s1Idx]  # (64,)
scores = w.unsqueeze(1) * qk_relu  # (64, 8192)
score_reduced = scores.sum(dim=0)  # (8192,)
cpu_scores = score_reduced[:cuRealAcSeq].numpy()  # (601,)

print(f"=== Batch 600 (cuRealAcSeq={cuRealAcSeq}) ===")
print(f"CPU scores: {len(cpu_scores)}, range=[{cpu_scores.min():.4f}, {cpu_scores.max():.4f}]")
print(f"NPU scores (0~511): {len(npu_scores_0_511)}, range=[{npu_scores_0_511.min():.4f}, {npu_scores_0_511.max():.4f}]")

# Compare first 88 (valid in NPU) and 88~511 (invalid in NPU)
n_valid = sum(1 for s in npu_scores_0_511 if s < 1e29)
print(f"\nNPU valid scores (0~511): {n_valid}")
print(f"NPU invalid scores (0~511): {512 - n_valid}")

# Compare CPU vs NPU for first 88
print(f"\n=== First 10 scores comparison ===")
for i in range(10):
    cpu = cpu_scores[i]
    npu = npu_scores_0_511[i]
    diff = abs(cpu - npu)
    print(f"  score[{i}]: CPU={cpu:12.4f}, NPU={npu:12.4f}, diff={diff:.4f}")

# Check if NPU scores match CPU bf16 (since NPU computes in bf16)
cpu_bf16 = torch.tensor(cpu_scores).to(torch.bfloat16).to(torch.float32).numpy()
print(f"\n=== CPU float vs CPU bf16 (first 10) ===")
for i in range(10):
    print(f"  score[{i}]: float={cpu_scores[i]:12.4f}, bf16={cpu_bf16[i]:12.4f}, NPU={npu_scores_0_511[i]:12.4f}")

# Check NPU vs CPU bf16 for first 88
match_bf16 = sum(1 for i in range(min(n_valid, 88)) if abs(npu_scores_0_511[i] - cpu_bf16[i]) < 1.0)
match_float = sum(1 for i in range(min(n_valid, 88)) if abs(npu_scores_0_511[i] - cpu_scores[i]) < 1.0)
print(f"\n=== Match count (first {min(n_valid, 88)} valid scores) ===")
print(f"  NPU vs CPU float: {match_float}/{min(n_valid, 88)}")
print(f"  NPU vs CPU bf16:  {match_bf16}/{min(n_valid, 88)}")

# Also check if NPU scores are in different order or different values
print(f"\n=== NPU valid scores (first 20) ===")
for i in range(20):
    if npu_scores_0_511[i] < 1e29:
        print(f"  NPU[{i}]={npu_scores_0_511[i]:12.4f}, CPU[{i}]={cpu_scores[i]:12.4f}")
