import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import sys
sys.path.insert(0, '/home/tjj/ops-transformer/attention/lightning_indexer/tests/pytest')
import torch, torch_npu
torch_npu.npu.set_device(0)
print("Device OK", flush=True)
import lightning_indexer_golden
print("Golden imported", flush=True)
test_data = (1, 8192, 8192, 8192, 8192, 64, 1, 128, 1024, 10591, torch.bfloat16, torch.bfloat16, torch.int32, [8192.0], [8192.0], "TND", "PA_BSND", 512, 3, [-100,100], [-100,100], [-130,130], False)
print("Starting golden...", flush=True)
cpu_result, npu_result, topk_value, sparse_value = lightning_indexer_golden.li_output_single(test_data)
print(f"Golden done: cpu={cpu_result.shape}, npu={npu_result.shape}", flush=True)
print("Starting check_result...", flush=True)
import result_compare_method
result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result, topk_value, sparse_value, test_data)
print(f"result={result}", flush=True)
print(f"fulfill_percent={fulfill_percent}", flush=True)
