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
    # Read scoreB16Ub[0..3] from positions 60-63 in batch 512 output
    for pos in [60, 61, 62, 63]:
        val = int(f[512*512 + pos].item())
        # Interpret as bf16 bits (uint16)
        bf16_bits = val & 0xFFFF
        # Convert bf16 to float
        float_bits = bf16_bits << 16
        fv = struct.unpack('f', struct.pack('I', float_bits))[0]
        print(f"batch 512 pos {pos}: raw={val}, bf16_bits=0x{bf16_bits:04x}, float={fv}")
