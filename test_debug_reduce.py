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
    # Read reduceOutInner[0] and [1] from positions 60,61
    for pos in [60, 61]:
        val = int(f[512*512 + pos].item())
        try:
            fv = struct.unpack('f', struct.pack('i', val))[0]
            print(f"batch 512 pos {pos}: int={val} (0x{val & 0xFFFFFFFF:08x}), float={fv}")
        except:
            print(f"batch 512 pos {pos}: {val}")
