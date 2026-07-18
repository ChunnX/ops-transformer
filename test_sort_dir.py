import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, struct, math
torch_npu.npu.set_device(0); torch.manual_seed(42); np.random.seed(42)
q = torch.randn(8192, 64, 128, dtype=torch.bfloat16).npu()
w = torch.randn(8192, 64, dtype=torch.bfloat16).npu()
k = torch.randn(10591, 1024, 1, 128, dtype=torch.bfloat16).npu()
bt = torch.tensor([[i for i in range(8)]], dtype=torch.int32).npu()
asq = torch.tensor([8192.0], dtype=torch.int32).npu()
ask = torch.tensor([8192.0], dtype=torch.int32).npu()
r, v = torch_npu.npu_lightning_indexer(q, k, w, actual_seq_lengths_query=asq, actual_seq_lengths_key=ask, block_table=bt, layout_query="TND", layout_key="PA_BSND", sparse_count=512, sparse_mode=3)
torch.npu.synchronize()

# batch 5000 scores (first 512 only)
score_vals = r[5000, 0].cpu().tolist()
scores = {}
for i, v in enumerate(score_vals):
    if v == -1: break
    scores[i] = struct.unpack('f', struct.pack('i', v))[0]

sort = np.load('/tmp/sort_result.npy')
radix_old = np.load('/tmp/radixtopk_result.npy')

s_indices = sort[5000,0][sort[5000,0]!=-1].tolist()
r_indices = radix_old[5000,0][radix_old[5000,0]!=-1].tolist()

# Check how many sort indices are in 0~511 range (we have scores for these)
s_in_range = [i for i in s_indices if i < 512]
r_in_range = [i for i in r_indices if i < 512]
print(f"Sort: {len(s_in_range)}/512 indices in 0~511 range")
print(f"Radix: {len(r_in_range)}/512 indices in 0~511 range")

# Check sort scores for indices in range
print(f"\nSort indices in 0~511 (first 20):")
for idx in sorted(s_in_range)[:20]:
    s = scores.get(idx, float('nan'))
    print(f"  idx={idx:4d}  score={s:12.4f}")

# Check how many sort indices have score = -1e30
neg_inf_count = sum(1 for idx in s_in_range if scores.get(idx, 0) < -1e29)
print(f"\nSort indices with score < -1e29: {neg_inf_count}/{len(s_in_range)}")

# Check radix scores for indices in range
print(f"\nRadix indices in 0~511 (first 20):")
for idx in sorted(r_in_range)[:20]:
    s = scores.get(idx, float('nan'))
    print(f"  idx={idx:4d}  score={s:12.4f}")

# Key question: does sort select largest or smallest scores?
# Check the score range of sort vs radix for indices in 0~511
s_scores = [scores.get(idx, 0) for idx in s_in_range]
r_scores = [scores.get(idx, 0) for idx in r_in_range]
print(f"\nSort scores in range: min={min(s_scores):.2f}, max={max(s_scores):.2f}, avg={sum(s_scores)/len(s_scores):.2f}")
print(f"Radix scores in range: min={min(r_scores):.2f}, max={max(r_scores):.2f}, avg={sum(r_scores)/len(r_scores):.2f}")

# Check all 512 scores sorted
all_scores = sorted(scores.values(), reverse=True)
print(f"\nTop 10 scores (all 512): {all_scores[:10]}")
print(f"Bottom 10 scores (all 512): {all_scores[-10:]}")
print(f"Score at rank 512 (cutoff): {all_scores[min(511, len(all_scores)-1)]:.4f}")
