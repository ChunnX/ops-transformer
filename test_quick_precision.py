import sys, os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '1'
import torch, torch_npu, numpy as np
torch_npu.npu.set_device(0)
code = open('/home/tjj/run_bench_paramset.py').read()
code = code.replace('npu_lightning_indexer(input_args)', 'op_out')
exec(code)
idx = op_out[0] if isinstance(op_out, (list, tuple)) else op_out
print('shape:', idx.shape)
print('has -1:', (idx == -1).any().item())
print('min:', idx.min().item(), 'max:', idx.max().item())
print('first 10:', idx.flatten()[:10])
