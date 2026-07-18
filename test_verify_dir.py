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

vals = r[5000, 0].cpu().tolist()
print("=== Radix (largest=false) selected scores ===")
radixIdx = [55, 347, 392, 393, 394, 395, 396, 397, 398, 399]
for j in range(10):
    score = struct.unpack('f', struct.pack('i', vals[j]))[0]
    print(f"  radix_idx={radixIdx[j]:5d}  score={score:12.4f}")

print("\n=== Sort selected scores ===")
sortIdx = [3077, 1922, 2851, 3063, 1178]
for j in range(5):
    score = struct.unpack('f', struct.pack('i', vals[10+j]))[0]
    print(f"  sort_idx={sortIdx[j]:5d}  score={score:12.4f}")

radix_scores = [struct.unpack('f', struct.pack('i', vals[j]))[0] for j in range(10)]
sort_scores = [struct.unpack('f', struct.pack('i', vals[10+j]))[0] for j in range(5)]
print(f"\nRadix avg={sum(radix_scores)/10:.2f}, range=[{min(radix_scores):.2f}, {max(radix_scores):.2f}]")
print(f"Sort avg={sum(sort_scores)/5:.2f}, range=[{min(sort_scores):.2f}, {max(sort_scores):.2f}]")
