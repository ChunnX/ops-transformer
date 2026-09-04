异常用例：
testcase_name	api_name	tensor_view_shapes	tensor_dtypes	tensor_formats	attributes	output_tensor_indexes	golden_api	input_data_ranges	precision_tolerances	absolute_precision

Prefill_FP8e4m3_FP8e4m3_BF16_Mxfp8_Empty_000005	torch.ops.cann_ops_transformer.quant_flash_attn	((1024, 80, 128), (0, 8, 512, 128), (0, 8, 512, 128), (1024, 80, 2, 2), (0, 8, 512, 2, 2), (0, 8, 8, 128, 2), (1, 0), (1,), (2,), None, (1,), (1,), None, (2048, 2048), (2, 4096))	('float8_e4m3fn', 'float8_e4m3fn', 'float8_e4m3fn', 'float8_e8m0', 'float8_e8m0', 'float8_e8m0', 'int32', 'float32', 'int32', 'int32', 'int32', 'int32', 'float32', 'int8', 'int32')		{'precision_tolerances': ((0.0078125, 0.0001, 0.005, 0.005, 10),), 'absolute_precision': 1e-08, 'p_scale_value': 448.0, 'cu_seqlens_q_values': [0, 1024], 'cu_seqlens_kv_values': None, 'seqused_q_values': [1020], 'seqused_kv_values': [1023], 'attn_mask_shape': (2048, 2048), 'attn_mask_dtype': 'int8', 'quant_mode': 1, 'softmax_scale': 0.125, 'mask_mode': 3, 'win_left': -1, 'win_right': -1, 'max_seqlen_q': -1, 'max_seqlen_kv': 1023, 'layout_q': 'TND', 'layout_q_descale': 'TND', 'layout_kv': 'PA_BNBD', 'layout_out': 'TND', 'return_softmax_lse': 1, 'N_q': 80, 'N_kv': 8, 'D': 128, 'head_dim_v': None, 'batch_size': -1, 'enable_pa': True, 'kv_cache_layout': 'PA_BNBD', 'q_scale_layout': 'TND', 'block_size': 512}			((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))	((0.0078125, 0.0001, 0.005, 0.005, 10),)	1.00E-08

报下面这个错：

2026-09-04 11:44:42,139 [ERROR] [3242072 Prefill_FP8e4m3_FP8e4m3_BF16_Mxfp8_Empty_000005 OnEagerProfiling] [profiling.py:290]: [Prefill_FP8e4m3_FP8e4m3_BF16_Mxfp8_Empty_000005] Error: basic_string::_S_construct null not valid 
Traceback (most recent call last):
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/profiling.py", line 288, in profile_process
    _do_profile(testcase, backend, device_grant_events, device_granted_indices, dev_id, switches, return_struct)
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/profiling.py", line 1204, in _do_profile
    result_nps, perf, eager_det_status = _capture_stdout_npu_memory(
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/profiling.py", line 397, in _capture_stdout_npu_memory
    return fn(*args, **kwargs)
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/profiling.py", line 490, in _execute_eager
    r = resolved(*args, **kwargs)
  File "/home/y00938753/anaconda3/envs/yxh_py39/lib/python3.9/site-packages/torch/_ops.py", line 1158, in __call__
    return self._op(*args, **(kwargs or {}))
  File "/home/y00938753/Ascend0903/cann/python/site-packages/cann_ops_transformer/ops/attention/quant_flash_attn/quant_flash_attn.py", line 479, in quant_flash_attn
    return op_module.quant_flash_attn(
RuntimeError: basic_string::_S_construct null not valid
