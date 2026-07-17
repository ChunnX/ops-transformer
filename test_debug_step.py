import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, io, contextlib, struct
torch_npu.npu.set_device(0)
torch.manual_seed(42); np.random.seed(42)
code = open('/home/tjj/run_bench_paramset.py').read()
code = code.replace('for i in range(5):', 'for i in range(1):')
code = code.replace('for i in range(3):', 'for i in range(1):')
code = code.replace('# Warmup', 'r = None')
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    exec(code)
if r is not None:
    f = r.flatten()
    b = 512
    vals = f[b*512:(b+1)*512].tolist()
    print(f"batch {b}: float[0]={struct.unpack('f', struct.pack('i', vals[0]))[0]}, float[1]={struct.unpack('f', struct.pack('i', vals[1]))[0]}, float[512]={struct.unpack('f', struct.pack('i', vals[2]))[0]}")
    print(f"  sortLen={vals[3]}, tileNum={vals[4]}")
