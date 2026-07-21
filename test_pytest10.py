import os, sys, subprocess
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch_npu
torch_npu.npu.set_device(1)

for i in range(10):
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'test_lightning_indexer_single.py', '-v', '-m', 'ci', '--tb=short'],
        capture_output=True, text=True,
        cwd='/home/tjj/ops-transformer/attention/lightning_indexer/tests/pytest',
        env={**os.environ, 'ASCEND_GLOBAL_LOG_LEVEL': '0', 'ASCEND_SLOG_PRINT_TO_STDOUT': '0'}
    )
    for line in result.stdout.split('\n'):
        if line.startswith('result '):
            print(f"Run {i+1}: {line}")
