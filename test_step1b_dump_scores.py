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
scores = [struct.unpack('f', struct.pack('i', v))[0] for v in vals]
np.save("/tmp/batch600_scores_512_1023.npy", np.array(scores, dtype=np.float32))
print(f"Saved {len(scores)} scores (index 512~1023)")
print(f"  range: [{min(scores):.4f}, {max(scores):.4f}]")
print(f"  first 5 (512~516): {scores[:5]}")
print(f"  scores[88~92] (600~604): {scores[88:93]}")
print(f"  scores[100~105] (612~617): {scores[100:105]}")
