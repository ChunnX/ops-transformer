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
    gh = [vals[0], vals[1], vals[2], vals[3]]
    im = vals[4] & 0xFFFF
    am = vals[5] & 0xFFFF
    bins = [gh[3], gh[2]-gh[3], gh[1]-gh[2], gh[0]-gh[1]]
    print(f"round 7: gh={gh}, bins={bins}, involvedMask=0x{im:04x}, andMask=0x{am:04x}")
