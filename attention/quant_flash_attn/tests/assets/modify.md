异常用例1：
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


异常用例2：
testcase_name	api_name	tensor_view_shapes	tensor_dtypes	tensor_formats	attributes	output_tensor_indexes	golden_api	input_data_ranges	precision_tolerances	absolute_precision

Prefill_FP8e4m3_FP8e4m3_BF16_Mxfp8_Empty_000003	torch.ops.cann_ops_transformer.quant_flash_attn	((0, 20, 64), (4096, 2, 64), (4096, 2, 64), (0, 20, 1, 2), (4096, 2, 1, 2), (64, 2, 64, 2), None, (1,), (2,), (2,), None, None, None, None, (2, 4096))	('float8_e4m3fn', 'float8_e4m3fn', 'float8_e4m3fn', 'float8_e8m0', 'float8_e8m0', 'float8_e8m0', 'int32', 'float32', 'int32', 'int32', 'int32', 'int32', 'float32', 'int8', 'int32')		{'precision_tolerances': ((0.0078125, 0.0001, 0.005, 0.005, 10),), 'absolute_precision': 1e-08, 'p_scale_value': 1.0, 'cu_seqlens_q_values': [0, 4096], 'cu_seqlens_kv_values': [0, 4096], 'seqused_q_values': None, 'seqused_kv_values': None, 'attn_mask_dtype': 'int8', 'quant_mode': 1, 'softmax_scale': 0.125, 'mask_mode': 0, 'win_left': -1, 'win_right': -1, 'max_seqlen_q': 4096, 'max_seqlen_kv': -1, 'layout_q': 'TND', 'layout_q_descale': 'TND', 'layout_kv': 'TND', 'layout_out': 'TND', 'return_softmax_lse': 1, 'N_q': 20, 'N_kv': 2, 'D': 64, 'head_dim_v': None, 'batch_size': -1, 'enable_pa': False, 'kv_cache_layout': 'TND', 'q_scale_layout': 'TND', 'block_size': 0}			((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))	((0.0078125, 0.0001, 0.005, 0.005, 10),)	1.00E-08

报下面这个错：

026-09-04 14:19:36,186 [ERROR] [3738692 Prefill_FP8e4m3_FP8e4m3_BF16_Mxfp8_Empty_000003 OnGenGolden] [profiling.py:603]: [Prefill_FP8e4m3_FP8e4m3_BF16_Mxfp8_Empty_000003] Golden generation failure 
Traceback (most recent call last):
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/golden_generation.py", line 139, in generate_golden
    return _run_api_on_cpu(api_name, raw_inputs, testcase, dist, cpu_backend=cpu_backend)
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/golden_generation.py", line 176, in _run_api_on_cpu
    return _exec_and_convert(api_name, args, kwargs, oidx, cpu_backend)
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/golden_generation.py", line 235, in _exec_and_convert
    result = call_api(api_name, overload_index, resolved, args, kwargs)
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/eager_execution.py", line 29, in call_api
    return resolved(*args, **kwargs)
  File "/home/y00938753/anaconda3/envs/yxh/lib/python3.9/site-packages/torch/_ops.py", line 1158, in __call__
    return self._op(*args, **(kwargs or {}))
NotImplementedError: Could not run 'cann_ops_transformer::quant_flash_attn' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'cann_ops_transformer::quant_flash_attn' is only available for these backends: [PrivateUse1, Meta, SparsePrivateUse1, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradMeta, Tracer, AutocastCPU, AutocastMTIA, AutocastXPU, AutocastMPS, AutocastCUDA, AutocastPrivateUse1, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].

PrivateUse1: registered at /dev/null:89 [kernel]
Meta: registered at /dev/null:89 [kernel]
SparsePrivateUse1: registered at ../torch_npu/csrc/aten/VariableFallbackKernel.cpp:270 [backend fallback]
BackendSelect: fallthrough registered at /pytorch/aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
Python: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:194 [backend fallback]
FuncTorchDynamicLayerBackMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:479 [backend fallback]
Functionalize: registered at /pytorch/aten/src/ATen/FunctionalizeFallbackKernel.cpp:349 [backend fallback]
Named: registered at /pytorch/aten/src/ATen/core/NamedRegistrations.cpp:7 [backend fallback]
Conjugate: registered at /pytorch/aten/src/ATen/ConjugateFallback.cpp:17 [backend fallback]
Negative: registered at /pytorch/aten/src/ATen/native/NegateFallback.cpp:18 [backend fallback]
ZeroTensor: registered at /pytorch/aten/src/ATen/ZeroTensorFallback.cpp:86 [backend fallback]
ADInplaceOrView: fallthrough registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:100 [backend fallback]
AutogradOther: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:63 [backend fallback]
AutogradCPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:67 [backend fallback]
AutogradCUDA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:75 [backend fallback]
AutogradXLA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:83 [backend fallback]
AutogradMPS: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:91 [backend fallback]
AutogradXPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:71 [backend fallback]
AutogradHPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:104 [backend fallback]
AutogradLazy: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:87 [backend fallback]
AutogradMTIA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:79 [backend fallback]
AutogradPrivateUse1: registered at ../torch_npu/csrc/aten/VariableFallbackKernel.cpp:233 [backend fallback]
AutogradMeta: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:95 [backend fallback]
Tracer: registered at /pytorch/torch/csrc/autograd/TraceTypeManual.cpp:294 [backend fallback]
AutocastCPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:322 [backend fallback]
AutocastMTIA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:466 [backend fallback]
AutocastXPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:504 [backend fallback]
AutocastMPS: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:209 [backend fallback]
AutocastCUDA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:165 [backend fallback]
AutocastPrivateUse1: fallthrough registered at ../torch_npu/csrc/aten/AutoCastOps.cpp:20 [backend fallback]
FuncTorchBatched: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:731 [backend fallback]
BatchedNestedTensor: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:758 [backend fallback]
FuncTorchVmapMode: fallthrough registered at /pytorch/aten/src/ATen/functorch/VmapModeRegistrations.cpp:27 [backend fallback]
Batched: registered at /pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp:1075 [backend fallback]
VmapMode: fallthrough registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
FuncTorchGradWrapper: registered at /pytorch/aten/src/ATen/functorch/TensorWrapper.cpp:208 [backend fallback]
PythonTLSSnapshot: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:202 [backend fallback]
FuncTorchDynamicLayerFrontMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:475 [backend fallback]
PreDispatch: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:206 [backend fallback]
PythonDispatcher: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:198 [backend fallback]
