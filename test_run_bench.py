import sys, os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '1'
import torch, torch_npu, numpy as np
torch_npu.npu.set_device(0)
code = open('/home/tjj/run_bench_paramset.py').read()
code = code.replace('for i in range(5):', 'for i in range(1):')
code = code.replace('for i in range(3):', 'for i in range(1):')
exec(code)
print("RESULT_OK")
