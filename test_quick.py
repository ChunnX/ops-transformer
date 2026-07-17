import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu
torch_npu.npu.set_device(1)
print('Device OK', flush=True)
r = torch_npu.npu_lightning_indexer(
    torch.randn(8192, 64, 128, dtype=torch.bfloat16).npu(),
    torch.randn(100, 1024, 1, 128, dtype=torch.bfloat16).npu(),
    torch.randn(8192, 64, dtype=torch.bfloat16).npu(),
    actual_seq_lengths_query=torch.tensor([8192.0], dtype=torch.int32).npu(),
    actual_seq_lengths_key=torch.tensor([8192.0], dtype=torch.int32).npu(),
    block_table=torch.tensor([[i for i in range(8)]], dtype=torch.int32).npu(),
    layout_query='TND',
    layout_key='PA_BSND',
    sparse_count=512,
    sparse_mode=3)
torch.npu.synchronize()
print(f'Done: {r[0].shape}', flush=True)
