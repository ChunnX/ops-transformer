import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, math
torch_npu.npu.set_device(0)
q_seq=1024; k_seq=1024; q_head_num=64; k_head_num=1; head_dim=128
block_size=1024; block_num=math.ceil(k_seq/block_size)
for run in range(3):
    np.random.seed(run*42+1)
    query=torch.tensor(np.random.uniform(-100,100,(q_seq,q_head_num,head_dim))).to(torch.bfloat16).npu()
    weights=torch.tensor(np.random.uniform(-130,130,(q_seq,q_head_num))).to(torch.bfloat16).npu()
    key=torch.tensor(np.random.uniform(-100,100,(block_num,block_size,k_head_num,head_dim))).to(torch.bfloat16).npu()
    bt=torch.tensor([[i for i in range(block_num)]],dtype=torch.int32).npu()
    asq=torch.tensor([float(q_seq)],dtype=torch.int32).npu()
    ask=torch.tensor([float(k_seq)],dtype=torch.int32).npu()
    r,v=torch_npu.npu_lightning_indexer(query,key,weights,actual_seq_lengths_query=asq,actual_seq_lengths_key=ask,block_table=bt,layout_query="TND",layout_key="PA_BSND",sparse_count=128,sparse_mode=3)
    torch.npu.synchronize()
    np.save(f'/tmp/sort_small_{run}.npy',r.cpu().numpy())
    print(f"Sort Run {run}: valid={len(r[r!=-1])}")
