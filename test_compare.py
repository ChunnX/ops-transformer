import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, sys, io, contextlib
torch_npu.npu.set_device(1)
torch.manual_seed(42)
np.random.seed(42)

code = open('/home/tjj/run_bench_paramset.py').read()
code = code.replace('for i in range(5):', 'for i in range(1):')
code = code.replace('for i in range(3):', 'for i in range(1):')
code = code.replace('# Warmup', 'r = None')
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    exec(code)
if r is not None:
    flat = r.flatten()
    # Show stats
    print(f"shape: {r.shape}")
    print(f"has -1: {(r == -1).any().item()}")
    valid = flat[flat != -1]
    print(f"valid count: {len(valid)}")
    print(f"min: {valid.min().item()} max: {valid.max().item()}")
    # Show batch 8191 (cuRealAcSeq=8192, full)
    b8191 = flat[8191*512:8191*512+512]
    valid8191 = b8191[b8191 != -1]
    print(f"batch8191: {len(valid8191)} valid, sorted: {sorted(valid8191[:20].tolist())}")
    # Show batch 5000
    b5000 = flat[5000*512:5000*512+512]
    valid5000 = b5000[b5000 != -1]
    print(f"batch5000: {len(valid5000)} valid, first: {sorted(valid5000[:20].tolist())}")
