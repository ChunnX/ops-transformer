import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math
torch_npu.npu.set_device(0); torch.manual_seed(42); np.random.seed(42)
q = torch.randn(8192, 64, 128, dtype=torch.bfloat16).npu()
w = torch.randn(8192, 64, dtype=torch.bfloat16).npu()
k = torch.randn(10591, 1024, 1, 128, dtype=torch.bfloat16).npu()
bt = torch.tensor([[i for i in range(8)]], dtype=torch.int32).npu()
asq = torch.tensor([8192.0], dtype=torch.int32).npu()
ask = torch.tensor([8192.0], dtype=torch.int32).npu()
r, v = torch_npu.npu_lightning_indexer(q, k, w, actual_seq_lengths_query=asq, actual_seq_lengths_key=ask, block_table=bt, layout_query="TND", layout_key="PA_BSND", sparse_count=512, sparse_mode=3)
torch.npu.synchronize()

# Check overlap for batch 100 (likely a mismatched batch)
sort = np.load('/tmp/sort_result.npy')
radix = r.cpu().numpy()
for b in [100, 500, 1000, 5000]:
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    rr = set(radix[b,0][radix[b,0]!=-1].tolist())
    print(f"Batch {b}: overlap={len(s&rr)}/512, sort_only={len(s-rr)}, radix_only={len(rr-s)}")
    # Check score range
    # The scores are computed inside the kernel, we can't easily access them
    # But we can check the indices
    if len(s-rr) > 0:
        print(f"  sort_only first 5: {sorted(s-rr)[:5]}")
        print(f"  radix_only first 5: {sorted(rr-s)[:5]}")
