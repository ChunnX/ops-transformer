[Decode_FP8e4m3_FP8e4m3_BF16_Mxfp8_redline_8_24_8_4_1024_72_72_TND_Mask_000001] Error: NPU_PREPROCESS_FAILURE: QuantFlashAttn metadata shape mismatch: placeholder=(2, 102400), generated=(2, 299008)
Traceback (most recent call last):
  File "/home/y00938753/ops-test-kit/ttk/core_modules/npu_preprocess.py", line 96, in invoke_npu_preprocess
    result = func(*args, **hook_kwargs)
  File "/home/y00938753/ops-transformer/attention/quant_flash_attn/tests/assets/spec.py", line 106, in npu_preprocess
    return load_impl_module("npu_preprocess").run(*args, **kwargs)
  File "/home/y00938753/ops-transformer/attention/quant_flash_attn/tests/assets/impl/npu_preprocess.py", line 331, in run
    raise ValueError(
ValueError: QuantFlashAttn metadata shape mismatch: placeholder=(2, 102400), generated=(2, 299008)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/profiling.py", line 313, in profile_process
    _do_profile(testcase, backend, device_grant_events, device_granted_indices, dev_id, switches, return_struct)
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/profiling.py", line 1310, in _do_profile
    result_nps, perf, eager_det_status = _capture_peak_memory(
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/profiling.py", line 427, in _capture_peak_memory
    return fn(*args, **kwargs)
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/profiling.py", line 457, in _execute_eager
    preprocess_result = invoke_npu_preprocess(
  File "/home/y00938753/ops-test-kit/ttk/core_modules/npu_preprocess.py", line 100, in invoke_npu_preprocess
    raise RuntimeError(f"NPU_PREPROCESS_FAILURE: {exc}") from exc
RuntimeError: NPU_PREPROCESS_FAILURE: QuantFlashAttn metadata shape mismatch: placeholder=(2, 102400), generated=(2, 299008)