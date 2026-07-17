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
    bf = struct.unpack('f', struct.pack('i', vals[0]))[0]
    curTileK = vals[1]
    print(f"boundaryFloat={bf}, curTileK={curTileK}")
    for i in range(8):
        sf = struct.unpack('f', struct.pack('i', vals[2+i*2]))[0]
        gt = vals[3+i*2]
        print(f"  score[{i}]={sf}, gt={gt}")
