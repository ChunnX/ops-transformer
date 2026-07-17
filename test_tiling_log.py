import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '1'
import torch, torch_npu, io, contextlib
torch_npu.npu.set_device(0)
code = open('/home/tjj/run_bench_paramset.py').read()
code = code.replace('for i in range(5):', 'for i in range(1):')
code = code.replace('for i in range(3):', 'for i in range(1):')
code = code.replace('# Warmup', 'r = None')
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    exec(code)
