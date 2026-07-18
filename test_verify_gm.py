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

for batch_id in [0, 600]:
    vals = r[batch_id, 0].cpu().tolist()
    scores = [struct.unpack('f', struct.pack('i', vals[i]))[0] for i in range(5)]
    batch_tag = vals[5]
    sort_len = vals[6]
    print(f"=== Batch {batch_id} (tag={batch_tag}, sortLen={sort_len}) ===")
    print(f"  score[0]   = {scores[0]:12.4f}")
    print(f"  score[1]   = {scores[1]:12.4f}")
    print(f"  score[87]  = {scores[2]:12.4f}")
    print(f"  score[88]  = {scores[3]:12.4f}  {'INVALID' if scores[3] > 1e29 else ''}")
    print(f"  score[511] = {scores[4]:12.4f}  {'INVALID' if scores[4] > 1e29 else ''}")
