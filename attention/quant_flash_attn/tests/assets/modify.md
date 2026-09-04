现在要支持异常用例的测试
用例：
		Prefill_FP8e4m3_FP8e4m3_BF16_Mxfp8_Exception_000040	(metadata)mxfp8:cu_q[65538,]， 												4096,20,64	BF16		4096,20	FP32		4096,20,64	FLOAT8_E4M3FN	-1,1	4096,2,64	FLOAT8_E4M3FN	-1,1	4096,2,64	FLOAT8_E4M3FN	-1,1	4096,20,1,2	FLOAT8_E8M0	-1,1	4096,2,1,2	FLOAT8_E8M0	-1,1	64,2,64,2	FLOAT8_E8M0	-1,1				1	1	FP32	1,448	0,4096	65538	INT32	0,1	0,4096	65538	INT32	0,1																			0.125	0	-1	-1	4096	-1	TND	TND	TND	TND	TRUE	-1	20	2	64											
期望报错：
Parameter axis B of QuantFlashAttn has incorrect value 65537. Reason: The value of B must be within the range (0, 65536).
实际报错：
ValueError: [INPUTS] cu_seqlens_q (slot 8) shape mismatch: CSV slot (65538,) != attribute value (2,) ([0, 4096]). Check Excel shape vs cu_seqlens/seqused value.

026-09-04 07:04:11,158 [ERROR] [1959573 Prefill_FP8e4m3_FP8e4m3_BF16_Mxfp8_Exception_000040 OnDumpInput] [profiling.py:290]: [Prefill_FP8e4m3_FP8e4m3_BF16_Mxfp8_Exception_000040] Error: Cannot resolve api_name 'torch.ops.cann_ops_transformer.quant_flash_attn': '_OpNamespace' 'cann_ops_transformer' object has no attribute 'quant_flash_attn' 
Traceback (most recent call last):
  File "/home/y00938753/ops-test-kit/ttk/core_modules/framework_api/api_resolver.py", line 58, in resolve_api
    obj = getattr(obj, p)
  File "/home/y00938753/anaconda3/envs/yxh/lib/python3.9/site-packages/torch/_ops.py", line 1267, in __getattr__
    raise AttributeError(
AttributeError: '_OpNamespace' 'cann_ops_transformer' object has no attribute 'quant_flash_attn'