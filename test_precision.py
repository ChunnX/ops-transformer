import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, sys, io, contextlib
torch_npu.npu.set_device(1)
code = open('/home/tjj/run_bench_paramset.py').read()
code = code.replace('set_device(0)', 'set_device(1)')
code = code.replace('for i in range(5):', 'for i in range(1):')
code = code.replace('for i in range(3):', 'for i in range(1):')
code = code.replace('# Warmup', 'r = None')
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    exec(code)
sys.stderr.write("=== OUTPUT START ===\n")
if r is not None:
    sys.stderr.write(f"idx shape: {r.shape}\n")
    sys.stderr.write(f"has -1: {(r == -1).any().item()}\n")
    sys.stderr.write(f"min: {r.min().item()} max: {r.max().item()}\n")
    sys.stderr.write(f"first 20: {r.flatten()[:20].tolist()}\n")
    sys.stderr.write("RESULT_OK\n")
else:
    sys.stderr.write("RESULT_FAIL: r is None\n")
