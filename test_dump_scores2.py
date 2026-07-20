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
vals = r[600, 0].cpu().tolist()
scores_512_1023 = np.array([struct.unpack('f', struct.pack('i', v))[0] for v in vals], dtype=np.float32)
np.save('/tmp/batch600_scores_512_1023.npy', scores_512_1023)
n_invalid = (scores_512_1023 > 1e29).sum()
print(f"Saved {len(scores_512_1023)} scores (512~1023), {n_invalid} invalid (+1e30)")
print(f"  Valid: {512 - n_invalid}")
# Check boundary
print(f"  score[600] (idx 88 in this array): {scores_512_1023[88]:.4f}")
print(f"  score[601] (idx 89): {scores_512_1023[89]:.4f}")
