import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import sys
sys.path.insert(0, '/home/tjj/ops-transformer/attention/lightning_indexer/tests/pytest')
import lightning_indexer_golden as golden
import torch, torch_npu, numpy as np
torch_npu.npu.set_device(1)
np.random.seed(42)
params = (1, 3, 3, 1, 1, 16, 1, 128, 1024, 1, torch.bfloat16, torch.bfloat16, torch.int32, [3], [3], 'TND', 'PA_BSND', 2048, 3, [-100,100], [-100,100], [-130,130], None)
cpu_result, npu_result, topk_value, sparse_value = golden.li_output_single(params)
print('CPU shape:', cpu_result.shape, 'valid:', len(cpu_result[cpu_result != -1]))
print('NPU shape:', npu_result.shape, 'valid:', len(npu_result[npu_result != -1]))
cpu_sorted, _ = cpu_result.sort()
npu_sorted, _ = npu_result.sort()
# Print first row
print('CPU row 0 first 10:', cpu_sorted.flatten()[:10].tolist())
print('NPU row 0 first 10:', npu_sorted.flatten()[:10].tolist())
print('CPU row 0 last 10:', cpu_sorted.flatten()[-10:].tolist())
print('NPU row 0 last 10:', npu_sorted.flatten()[-10:].tolist())
# Print all rows (only 54 total rows for 16 heads * 3 q_seq + padding)
cpu_flat = cpu_sorted.reshape(-1, 2048)
npu_flat = npu_sorted.reshape(-1, 2048)
for i in range(min(10, cpu_flat.shape[0])):
    cpu_vals = cpu_flat[i][cpu_flat[i] != -1].tolist()
    npu_vals = npu_flat[i][npu_flat[i] != -1].tolist()
    cpu_set = set(cpu_vals)
    npu_set = set(npu_vals)
    only_cpu = sorted(cpu_set - npu_set)[:5]
    only_npu = sorted(npu_set - cpu_set)[:5]
    overlap = len(cpu_set & npu_set)
    print(f'  row {i}: cpu={len(cpu_vals)}, npu={len(npu_vals)}, overlap={overlap}, only_cpu={only_cpu}, only_npu={only_npu}')
