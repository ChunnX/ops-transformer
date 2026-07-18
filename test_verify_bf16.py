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

print("=== Sort-selected indices scores ===")
for j in range(10):
    score_bits = vals[j*2]
    idx = vals[j*2+1]
    score = struct.unpack('f', struct.pack('i', score_bits))[0]
    print(f"  sort_idx[{j}]={idx:5d}  score={score:12.4f}")

print("\n=== Radix-selected indices scores ===")
for j in range(10):
    score_bits = vals[20+j*2]
    idx = vals[20+j*2+1]
    score = struct.unpack('f', struct.pack('i', score_bits))[0]
    print(f"  radix_idx[{j}]={idx:5d}  score={score:12.4f}")

print("\n=== First 10 scores (index 0~9) ===")
for j in range(10):
    score_bits = vals[40+j]
    score = struct.unpack('f', struct.pack('i', score_bits))[0]
    print(f"  score[{j}]={score:12.4f}")
