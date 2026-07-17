import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, sys, io, contextlib
torch_npu.npu.set_device(0)
code = open('/home/tjj/run_bench_paramset.py').read()
code = code.replace('for i in range(5):', 'for i in range(1):')
code = code.replace('for i in range(3):', 'for i in range(1):')
code = code.replace('# Warmup', 'r = None')
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    exec(code)
if r is not None:
    flat_all = r.flatten()
    # Check batches where coreTopKOffset might be 0
    for b in [512, 513, 1000, 5000, 8000, 8191]:
        vals = flat_all[b*512:b*512+4].tolist()
        print(f"batch {b}: coreTopKOffset={vals[0]}, involvedMask={vals[1]}, remainK={vals[2]}, totalDefIn={vals[3]}")
