# 本次修改前

load npu output from file: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_0.bin with dtype:<class 'numpy.float16'> real dtype:fp16
[2026/07/15 08:17:49] [INFO]-tools.py:4338 - Touch file succeed: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_0.bin
[2026/07/15 08:17:49] [INFO]-tools.py:1975 - [data_compare_np]ouput flatten start!
[2026/07/15 08:17:49] [INFO]-tools.py:1978 - [data_compare_np]ouput flatten end!
[2026/07/15 08:17:49] [INFO]-tools.py:2092 - [data_compare_np]overflows_count start!
[2026/07/15 08:17:49] [INFO]-tools.py:2096 - [data_compare_np]overflows_count end!
[2026/07/15 08:17:49] [INFO]-tools.py:2099 - split_count:517104.0; max_diff_hd:10.0;
[2026/07/15 08:17:49] [INFO]-tools.py:1064 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1070 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/15 08:17:49] [INFO]-tools.py:1071 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000001 	 9.8281250 	 9.8281250 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000002 	 6.7929688 	 6.7968750 	 0.0039062 	 0.0005750
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000003 	 6.0585938 	 6.0625000 	 0.0039062 	 0.0006447
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000004 	 7.5117188 	 7.5156250 	 0.0039062 	 0.0005200
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000005 	 0.5166016 	 0.5180664 	 0.0014648 	 0.0028355
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000006 	 -9.4453125 	 -9.4531250 	 0.0078125 	 0.0008271
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000007 	 0.8339844 	 0.8339844 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000008 	 0.1735840 	 0.1734619 	 0.0001221 	 0.0007032
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000009 	 9.3437500 	 9.3437500 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000010 	 -8.4218750 	 -8.4296875 	 0.0078125 	 0.0009276
[2026/07/15 08:17:49] [INFO]-tools.py:1080 - ...   	   ...   	   ...   	   ...    	   ...
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517095 	 -1.4199219 	 -1.4199219 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517096 	 -4.5625000 	 -4.5625000 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517097 	 3.4257812 	 3.4257812 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517098 	 1.8593750 	 1.8593750 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517099 	 6.1445312 	 6.1445312 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517100 	 -4.6093750 	 -4.6093750 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517101 	 -3.7734375 	 -3.7734375 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517102 	 -1.0830078 	 -1.0830078 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517103 	 4.4726562 	 4.4726562 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00517104 	 0.7348633 	 0.7348633 	 0.0000000 	 0.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:2261 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:2263 - Rtol   	 Atol   	 PctThd   	 PctRlt   	 Result
[2026/07/15 08:17:49] [INFO]-tools.py:2264 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:2266 - 0.0050    	 0.000025  	 99.50%   	 34.287493%   	 Failed
[2026/07/15 08:17:49] [INFO]-tools.py:2269 - Max-RelativeError is: 2.0. Threshold is: 10.0.
[2026/07/15 08:17:49] [INFO]-tools.py:1128 - Error Line-----------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1130 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/15 08:17:49] [INFO]-tools.py:1131 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000010 	 -0.1816406 	 -0.1788330 	 0.0028076 	 0.0154572
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000011 	 -0.3781738 	 -0.3815918 	 0.0034180 	 0.0089569
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000054 	 0.0353699 	 0.0341797 	 0.0011902 	 0.0336609
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000057 	 6.2968750 	 3.8242188 	 2.4726562 	 0.3925781
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000058 	 9.9062500 	 -3.0097656 	 12.9160156 	 1.3037109
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000059 	 -4.6015625 	 6.3242188 	 10.9257812 	 1.7265625
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000060 	 -0.8344727 	 -2.7734375 	 1.9389648 	 0.6992188
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000061 	 2.2031250 	 -4.3007812 	 6.5039062 	 1.5126953
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000062 	 1.6035156 	 1.4472656 	 0.1562500 	 0.0974121
[2026/07/15 08:17:49] [INFO]-tools.py:1144 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000228 	 -3.4003906 	 -0.9028320 	 2.4975586 	 0.7348633
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000229 	 7.8125000 	 8.1093750 	 0.2968750 	 0.0366211
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000230 	 6.5390625 	 -4.1406250 	 10.6796875 	 1.6328125
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000231 	 -7.4335938 	 6.5390625 	 13.9726562 	 1.8789062
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000232 	 -7.8476562 	 -4.6054688 	 3.2421875 	 0.4130859
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000233 	 2.4726562 	 -4.4101562 	 6.8828125 	 1.5605469
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000234 	 -1.4394531 	 5.8125000 	 7.2519531 	 1.2470703
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000235 	 4.4492188 	 0.9809570 	 3.4682617 	 0.7797852
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000236 	 3.8281250 	 -0.9555664 	 4.7836914 	 1.2500000
[2026/07/15 08:17:49] [INFO]-tools.py:1144 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/15 08:17:49] [INFO]-tools.py:1149 - Max-RE line:---------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1157 - 00004322 	 4.6445312 	 -4.6484375 	 9.2929688 	 2.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1157 - 00016676 	 5.3398438 	 -5.3398438 	 10.6796875 	 2.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1157 - 00017165 	 5.2968750 	 -5.2968750 	 10.5937500 	 2.0000000
[2026/07/15 08:17:49] [INFO]-tools.py:1164 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:4516 - aclnnFusedInferAttentionScoreV5:aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311 End to compare result. Duration:0.15 second.
[2026/07/15 08:17:49] [INFO]-run_aclnn.py:2887 - output-0 compare end 
[2026/07/15 08:17:49] [INFO]-run_aclnn.py:2889 - output-0 result Failed 34.28749342492032 2.0
[2026/07/15 08:17:49] [INFO]-run_aclnn.py:2885 - start compare case the first output-1 
[2026/07/15 08:17:49] [INFO]-tools.py:4142 - Step:------>>>>>> Start to compare result <<<<<<------ 
[2026/07/15 08:17:49] [INFO]-tools.py:4181 - data-compare for output index:1,and its dtype is fp32.
npu out file: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_output_1.bin
load npu output from file: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_output_1.bin with dtype:<class 'numpy.float32'> real dtype:fp32
*******
输出数据类型： fp32 method为: 1
*******
load npu output from file: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_1.bin with dtype:<class 'numpy.float32'> real dtype:fp32
[2026/07/15 08:17:49] [INFO]-tools.py:4338 - Touch file succeed: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_1.bin
[2026/07/15 08:17:49] [INFO]-tools.py:1975 - [data_compare_np]ouput flatten start!
[2026/07/15 08:17:49] [INFO]-tools.py:1978 - [data_compare_np]ouput flatten end!
[2026/07/15 08:17:49] [INFO]-tools.py:2092 - [data_compare_np]overflows_count start!
[2026/07/15 08:17:49] [INFO]-tools.py:2096 - [data_compare_np]overflows_count end!
[2026/07/15 08:17:49] [INFO]-tools.py:2099 - split_count:9072.0; max_diff_hd:10.0;
[2026/07/15 08:17:49] [INFO]-tools.py:1064 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1070 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/15 08:17:49] [INFO]-tools.py:1071 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000001 	 56.0644913 	 56.0641060 	 0.0003853 	 0.0000069
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000002 	 53.1180115 	 42.6205101 	 10.4975014 	 0.1976260
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000003 	 62.7973251 	 53.2025261 	 9.5947990 	 0.1527899
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000004 	 41.1084213 	 37.4092369 	 3.6991844 	 0.0899860
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000005 	 59.9560432 	 59.9524803 	 0.0035629 	 0.0000594
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000006 	 52.4218941 	 52.3514481 	 0.0704460 	 0.0013438
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000007 	 57.4209442 	 57.4208870 	 0.0000572 	 0.0000010
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000008 	 54.1091957 	 44.2214813 	 9.8877144 	 0.1827363
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000009 	 41.5967941 	 38.0671387 	 3.5296555 	 0.0848540
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00000010 	 50.0907707 	 45.8441200 	 4.2466507 	 0.0847791
[2026/07/15 08:17:49] [INFO]-tools.py:1080 - ...   	   ...   	   ...   	   ...    	   ...
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009063 	 57.3644829 	 57.3651390 	 0.0006561 	 0.0000114
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009064 	 59.0962677 	 59.0962639 	 0.0000038 	 0.0000001
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009065 	 61.8104630 	 61.8104477 	 0.0000153 	 0.0000002
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009066 	 51.2623787 	 51.2623825 	 0.0000038 	 0.0000001
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009067 	 49.0081215 	 49.0082169 	 0.0000954 	 0.0000019
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009068 	 53.0105782 	 53.0105896 	 0.0000114 	 0.0000002
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009069 	 50.7572136 	 50.7572250 	 0.0000114 	 0.0000002
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009070 	 49.6615028 	 49.5053291 	 0.1561737 	 0.0031448
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009071 	 46.5672798 	 46.5673218 	 0.0000420 	 0.0000009
[2026/07/15 08:17:49] [INFO]-tools.py:1061 - 00009072 	 54.4735184 	 54.4734802 	 0.0000381 	 0.0000007
[2026/07/15 08:17:49] [INFO]-tools.py:2261 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:2263 - Rtol   	 Atol   	 PctThd   	 PctRlt   	 Result
[2026/07/15 08:17:49] [INFO]-tools.py:2264 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:2266 - 0.0050    	 0.000025  	 99.50%   	 50.529101%   	 Failed
[2026/07/15 08:17:49] [INFO]-tools.py:2269 - Max-RelativeError is: 0.4641224. Threshold is: 10.0.
[2026/07/15 08:17:49] [INFO]-tools.py:1128 - Error Line-----------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1130 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/15 08:17:49] [INFO]-tools.py:1131 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000001 	 53.1180115 	 42.6205101 	 10.4975014 	 0.1976260
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000002 	 62.7973251 	 53.2025261 	 9.5947990 	 0.1527899
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000003 	 41.1084213 	 37.4092369 	 3.6991844 	 0.0899860
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000007 	 54.1091957 	 44.2214813 	 9.8877144 	 0.1827363
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000008 	 41.5967941 	 38.0671387 	 3.5296555 	 0.0848540
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000009 	 50.0907707 	 45.8441200 	 4.2466507 	 0.0847791
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000010 	 57.1765709 	 40.3307037 	 16.8458672 	 0.2946289
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000011 	 50.7231255 	 44.7978859 	 5.9252396 	 0.1168154
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000014 	 54.3032990 	 46.5443268 	 7.7589722 	 0.1428822
[2026/07/15 08:17:49] [INFO]-tools.py:1144 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000140 	 60.9235954 	 43.8391151 	 17.0844803 	 0.2804247
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000141 	 55.2122040 	 50.5976257 	 4.6145782 	 0.0835790
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000142 	 46.9768867 	 45.3266182 	 1.6502686 	 0.0351294
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000144 	 54.4984169 	 44.5438614 	 9.9545555 	 0.1826577
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000146 	 53.6901703 	 52.7025070 	 0.9876633 	 0.0183956
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000147 	 47.8238564 	 46.9777985 	 0.8460579 	 0.0176911
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000148 	 47.2018471 	 44.7153587 	 2.4864883 	 0.0526778
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000149 	 56.4609566 	 62.2775230 	 5.8165665 	 0.0933975
[2026/07/15 08:17:49] [INFO]-tools.py:1138 - 00000150 	 54.0810623 	 50.2681084 	 3.8129539 	 0.0705044
[2026/07/15 08:17:49] [INFO]-tools.py:1144 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/15 08:17:49] [INFO]-tools.py:1149 - Max-RE line:---------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:1157 - 00001880 	 70.3075333 	 37.6762314 	 32.6313019 	 0.4641224
[2026/07/15 08:17:49] [INFO]-tools.py:1164 - ---------------------------------------------------------------------------------------
[2026/07/15 08:17:49] [INFO]-tools.py:4516 - aclnnFusedInferAttentionScoreV5:aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311 End to compare result. Duration:0.01 second.
[2026/07/15 08:17:49] [INFO]-run_aclnn.py:2887 - output-1 compare end 
[2026/07/15 08:17:49] [INFO]-run_aclnn.py:2889 - output-1 result Failed 50.52910052910053 0.4641224145889282
[2026/07/15 08:17:49] [INFO]-run_single.py:3969 - end execute aclnn case: aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311
[get_each_cs_npu_log]current scene not satisfy,skip
[2026/07/15 08:17:49] [WARNING]-run_single.py:0316 - There not exist kernel file[kernel_name:aclnnFusedInferAttentionScoreV5] in kernel_meta.
[2026/07/15 08:17:52] [/home/z00699874/fuzz/aclnn_fuzz_0713/libs/../golden/aclnn_golden/aclnnFusedInferAttentionScoreV5_golden.csv]-run_single.py:1400 - [WARNING]no golden file:
[2026/07/15 08:17:52] [INFO]-run_single.py:1698 - Case report ->:
[2026/07/15 08:17:52] [INFO]-run_single.py:1699 - aclnnFusedInferAttentionScoreV5,aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311,1784103472,34.28749342492032/50.52910052910053,Failed/Failed,None,0.0,no_bound,,2.0/0.4641224,"[[28, 2, 162, 57], [28, 7168, 162, 57], [28, 7168, 162, 57], [1], [2048, 2048], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1, 7168, 162, 57], [1, 7168, 162, 57], [1], [1], [1], [1], [1], [1], [1]]","['fp16', 'fp16', 'fp16', 'fp16', 'bool', 'uint64', 'fp32', 'uint64', 'fp16', 'fp16', 'fp16', 'fp16', 'int32', 'int64', 'int64', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16']","['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND']","[[-5, 5], [-10, 10], [-10, 10], ['null', 'null'], [-10.0, 10.0], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], [-10, 10], [-10, 10], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null']]","{'actualseqlengths': [], 'actualseqlengthskv': [69, 60, 80, 110, 5376, 43, 81, 3, 98, 107, 18, 92, 3840, 15, 125, 4224, 74, 3328, 75, 100, 29, 56, 21, 89, 74, 65, 3328, 1536], 'prefix_act_lens': [5852], 'qstartidx': [], 'kvstartidx': [], 'numheads': 162, 'scalevalue': 0.1324532357065044, 'pretokens': 946, 'nexttokens': -1, 'inputlayout': 'BSND', 'numkeyvalueheads': 0, 'sparsemode': 4, 'innerprecise': 0, 'blocksize': 0, 'antiquant_mode': 0, 'softmax_lse_flag': True, 'k_antiquant_mode': 0, 'v_antiquant_mode': 0, 'query_quant_mode': 0, 'psetype': 0, 'fused_flag': 1, 'mrandomtype': 'Normal', 'mrandom': 0, 'prandom': 1, 'enablegpu': 'True', 'flaglist': [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]}","[[28, 2, 162, 57], [28, 162, 2, 1]]","['fp16', 'fp32']","None","[]",N/A,"2735166.0000","5908.0000",48689016.0000,"16777216",7398,"","",,,,"",,,,,None,,,Pass,,,,
[2026/07/15 08:17:52] [INFO]-run_single.py:4032 - -------------------------第1用例npu_off执行结束,共1条用例-----------------------------------
[2026/07/15 08:17:52] [INFO]-run_single.py:4059 - Case execution(aclnnFusedInferAttentionScoreV5:1/0) have completed. Duration:80.23 second.
[ERROR] 2026-07-15-08:17:52 (PID:698424, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception

[AIV Block 0/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 0/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 0/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=66.072792 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.000000[AIV Block 0/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=66.072792 iniSum[0]=1.000000 iniMax[1]=7.726464
[AIV Block 0/64]   mmRes[0]=-27.683027
[AIV Block 0/64]   mmRes[1]=167.265762
[AIV Block 0/64]   mmRes[2]=25.492270
[AIV Block 0/64]   mmRes[3]=68.230728
[AIV Block 0/64]   mmRes[64]=57.378597 mmRes[128]=32.709888 mmRes[192]=156.699249 mmRes[255]=-231.890488
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VF256-pre] m=1 oriN=256 preMax[0]=66.072792 preSum[0]=1.000000 preMax[1]=7.726464
[AIV Block 0/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-27.683027 167.265762 25.492270 68.230728 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/64] [VF256-post] m=1 mx0=66.072792 sum0=1.000000 srcPost0=-27.683027 srcPost1=167.265762
[AIV Block 0/64] [VEC-Nd-post] s2L=0 mx0=42.800434 sm0=1.000336 mx1=7.726464 sm1=-279.387848
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=0 mx0=42.800434 sm0=1.000336 mx1=7.726464 sm1=-279.387848
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=0 mx0=42.800434 sm0=1.000336 mx1=7.726464 sm1=-279.387848
[AIV Block 0/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 0/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 0/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=42.800434 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.000336[AIV Block 0/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=42.800434 iniSum[0]=1.000336 iniMax[1]=7.726464
[AIV Block 0/64]   mmRes[0]=-73.496201
[AIV Block 0/64]   mmRes[1]=175.400269
[AIV Block 0/64]   mmRes[2]=179.696045
[AIV Block 0/64]   mmRes[3]=-10.499971
[AIV Block 0/64]   mmRes[64]=150.721924 mmRes[128]=169.427368 mmRes[192]=-130.357605 mmRes[255]=75.636108
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=1 mx0=42.800434 sm0=1.000336 mx1=7.726464 sm1=-279.387848
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=1 mx0=42.800434 sm0=1.000336 mx1=7.726464 sm1=-279.387848
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=1 mx0=42.800434 sm0=1.000336 mx1=7.726464 sm1=-279.387848
[AIV Block 0/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 0/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 0/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 0/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=42.800434 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.044567[AIV Block 0/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=42.800434 iniSum[0]=1.044567 iniMax[1]=7.726464
[AIV Block 0/64]   mmRes[0]=-159.652390
[AIV Block 0/64]   mmRes[1]=-20.926689
[AIV Block 0/64]   mmRes[2]=-148.055710
[AIV Block 0/64]   mmRes[3]=-319.478302
[AIV Block 0/64]   mmRes[64]=136.103928 mmRes[128]=97.351318 mmRes[192]=-87.300552 mmRes[255]=64.942345
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=2 mx0=42.800434 sm0=1.044567 mx1=7.726464 sm1=-280.940002
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=2 mx0=42.800434 sm0=1.044567 mx1=7.726464 sm1=-280.940002
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=2 mx0=42.800434 sm0=1.044567 mx1=7.726464 sm1=-280.940002
[AIV Block 0/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-cpy] s2L=0 vo0=-0.696312 vo1=192.753342
[AIV Block 0/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5920 pLC=23 s2R=68
[AIV Block 0/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 0/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 0/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=42.800434 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.058813[AIV Block 0/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=42.800434 iniSum[0]=1.058813 iniMax[1]=7.726464
[AIV Block 0/64]   mmRes[0]=-155.773819
[AIV Block 0/64]   mmRes[1]=-15.460873
[AIV Block 0/64]   mmRes[2]=-64.166557
[AIV Block 0/64]   mmRes[3]=60.987240
[AIV Block 0/64]   mmRes[64]=-278.447418 mmRes[128]=94.691544 mmRes[192]=-22.246895 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=3 mx0=42.800434 sm0=1.058813 mx1=7.726464 sm1=-282.492157
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=3 mx0=42.800434 sm0=1.058813 mx1=7.726464 sm1=-282.492157
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=3 mx0=42.800434 sm0=1.058813 mx1=7.726464 sm1=-282.492157
[AIV Block 0/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-expUb] s2L=1 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 0/64] [VEC2-upd] s2L=1 vo0=-6.553395 vo1=-2.879407
[AIV Block 0/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=4 s2S=4864 s2R=68 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 0/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=61440 computeMode=1 s1Off=0
[AIV Block 0/64] [COPYIN-offset] s2L=4 maskOff=61440 preOff=64 computeMode=1 compressMode=3 s2R=68 hS1R=1 maskS2Size=2048
[AIV Block 0/64] [MASK-gm] s2L=4 gmAtDeltaOff=61440 val=0 gmAtCorrectOff=36 val=1 s1O=0 s2O=36
[AIV Block 0/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=0 mask[1]=0 reread=0 phyAddr=1ff00
[AIV Block 0/64] [MASK-chk1] s2L=4 mask[0]=0 phyAddr=1ff00
[AIV Block 0/64] [MASK-chk2] s2L=4 mask[0]=0
[AIV Block 0/64] [MASK-chk3] s2L=4 mask[0]=0 expPhy=15700
[AIV Block 0/64] [MASK-chk4] s2L=4 mask[0]=0
[AIV Block 0/64] [MASK-chk5] s2L=4 mask[0]=0 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=45.927269 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.047036[AIV Block 0/64] [VEC-Nd-pre] s2L=4 s2R=68 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=45.927269 iniSum[0]=1.047036 iniMax[1]=7.726464
[AIV Block 0/64]   mmRes[0]=-331.737946
[AIV Block 0/64]   mmRes[1]=-134.870743
[AIV Block 0/64]   mmRes[2]=-134.921310
[AIV Block 0/64]   mmRes[3]=188.724365
[AIV Block 0/64]   mmRes[64]=28.180927 mmRes[128]=12.894497 mmRes[192]=-11.563240 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=0 mask[1]=0
[AIV Block 0/64] [VEC-Nd-post] s2L=4 mx0=45.927269 sm0=1.047036 mx1=7.726464 sm1=-284.044312
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=4 mx0=45.927269 sm0=1.047036 mx1=7.726464 sm1=-284.044312
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=4 mx0=45.927269 sm0=1.047036 mx1=7.726464 sm1=-284.044312
[AIV Block 0/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 0/64] [LSE-out] lseOff=0 lse0=44.251823 lse1=44.251823
[AIV Block 0/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-expUb] s2L=2 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 0/64] [VEC2-upd] s2L=2 vo0=-6.734852 vo1=-3.074134
[AIV Block 0/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 0/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 0/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 0/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=43.699638 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.737043[AIV Block 0/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=43.699638 iniSum[0]=1.737043 iniMax[1]=7.726464
[AIV Block 0/64]   mmRes[0]=149.403381
[AIV Block 0/64]   mmRes[1]=21.309612
[AIV Block 0/64]   mmRes[2]=-208.727676
[AIV Block 0/64]   mmRes[3]=-76.581635
[AIV Block 0/64]   mmRes[64]=-93.687088 mmRes[128]=40.739449 mmRes[192]=99.159515 mmRes[255]=38.584782
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VF256-pre] m=1 oriN=256 preMax[0]=43.699638 preSum[0]=1.737043 preMax[1]=7.726464
[AIV Block 0/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=149.403381 21.309612 -208.727676 -76.581635 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/64] [VF256-post] m=1 mx0=43.699638 sum0=1.737043 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC-Nd-post] s2L=0 mx0=43.001041 sm0=1.008049 mx1=7.726464 sm1=-279.454529
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=0 mx0=43.001041 sm0=1.008049 mx1=7.726464 sm1=-279.454529
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=0 mx0=43.001041 sm0=1.008049 mx1=7.726464 sm1=-279.454529
[AIV Block 0/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-expUb] s2L=3 e0=0.043856 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 0/64] [VEC2-upd] s2L=3 vo0=-6.771321 vo1=-3.036339
[AIV Block 0/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 0/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 0/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 0/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=43.001041 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.008049[AIV Block 0/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=43.001041 iniSum[0]=1.008049 iniMax[1]=7.726464
[AIV Block 0/64]   mmRes[0]=-136.520264
[AIV Block 0/64]   mmRes[1]=-91.531883
[AIV Block 0/64]   mmRes[2]=-29.148708
[AIV Block 0/64]   mmRes[3]=125.789452
[AIV Block 0/64]   mmRes[64]=130.689529 mmRes[128]=-116.977104 mmRes[192]=140.817429 mmRes[255]=98.181000
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=1 mx0=43.001041 sm0=1.008049 mx1=7.726464 sm1=-279.454529
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=1 mx0=43.001041 sm0=1.008049 mx1=7.726464 sm1=-279.454529
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=1 mx0=43.001041 sm0=1.008049 mx1=7.726464 sm1=-279.454529
[AIV Block 0/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 0/64] [VEC2-upd] s2L=4 vo0=-3.337402 vo1=-9.476003
[AIV Block 0/64] [VEC2-final] s2L=4 sum=1.088063 max=45.927269
[AIV Block 0/64] [VEC2-out] s2L=4 vo0=-203528.984375 vo1=368180.812500
[AIV Block 0/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 0/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 0/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 0/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 0/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=43.001041 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.311049[AIV Block 0/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=43.001041 iniSum[0]=1.311049 iniMax[1]=7.726464
[AIV Block 0/64]   mmRes[0]=-104.787178
[AIV Block 0/64]   mmRes[1]=112.562508
[AIV Block 0/64]   mmRes[2]=20.049213
[AIV Block 0/64]   mmRes[3]=-27.285889
[AIV Block 0/64]   mmRes[64]=-10.010809 mmRes[128]=99.811615 mmRes[192]=35.328949 mmRes[255]=86.581589
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=2 mx0=43.001041 sm0=1.311049 mx1=7.726464 sm1=-281.006683
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=2 mx0=43.001041 sm0=1.311049 mx1=7.726464 sm1=-281.006683
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=2 mx0=43.001041 sm0=1.311049 mx1=7.726464 sm1=-281.006683
[AIV Block 0/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 1/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 1/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 1/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=55.332569 mx1=4.842289 mx2=7.169627 mx3=7.133998 sum0=1.002568[AIV Block 1/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=55.332569 iniSum[0]=1.002568 iniMax[1]=4.842289
[AIV Block 1/64]   mmRes[0]=-67.043709
[AIV Block 1/64]   mmRes[1]=21.088055
[AIV Block 1/64]   mmRes[2]=106.565117
[AIV Block 1/64]   mmRes[3]=-140.407883
[AIV Block 1/64]   mmRes[64]=133.048660 mmRes[128]=-71.712479 mmRes[192]=80.787483 mmRes[255]=-48.870888
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VF256-pre] m=1 oriN=256 preMax[0]=55.332569 preSum[0]=1.002568 preMax[1]=4.842289
[AIV Block 1/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-67.043709 21.088055 106.565117 -140.407883 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 1/64] [VF256-post] m=1 mx0=55.332569 sum0=1.002568 srcPost0=-67.043709 srcPost1=21.088055
[AIV Block 1/64] [VEC-Nd-post] s2L=0 mx0=37.558483 sm0=1.337246 mx1=4.842289 sm1=931.718018
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=0 mx0=37.558483 sm0=1.337246 mx1=4.842289 sm1=931.718018
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=0 mx0=37.558483 sm0=1.337246 mx1=4.842289 sm1=931.718018
[AIV Block 1/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 1/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 1/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=37.558483 mx1=4.842289 mx2=7.169627 mx3=7.133998 sum0=1.337246[AIV Block 1/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=37.558483 iniSum[0]=1.337246 iniMax[1]=4.842289
[AIV Block 1/64]   mmRes[0]=-190.409866
[AIV Block 1/64]   mmRes[1]=-104.145378
[AIV Block 1/64]   mmRes[2]=42.020390
[AIV Block 1/64]   mmRes[3]=2.515178
[AIV Block 1/64]   mmRes[64]=36.375416 mmRes[128]=68.614700 mmRes[192]=39.580639 mmRes[255]=144.343018
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=1 mx0=37.558483 sm0=1.337246 mx1=4.842289 sm1=931.718018
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=1 mx0=37.558483 sm0=1.337246 mx1=4.842289 sm1=931.718018
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=1 mx0=37.558483 sm0=1.337246 mx1=4.842289 sm1=931.718018
[AIV Block 1/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 1/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 1/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 1/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=37.558483 mx1=4.842289 mx2=7.169627 mx3=7.133998 sum0=2.301245[AIV Block 1/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=37.558483 iniSum[0]=2.301245 iniMax[1]=4.842289
[AIV Block 1/64]   mmRes[0]=32.319355
[AIV Block 1/64]   mmRes[1]=-51.193855
[AIV Block 1/64]   mmRes[2]=-74.102753
[AIV Block 1/64]   mmRes[3]=142.209137
[AIV Block 1/64]   mmRes[64]=153.727051 mmRes[128]=-297.491669 mmRes[192]=237.858322 mmRes[255]=144.378601
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=2 mx0=37.558483 sm0=2.301245 mx1=4.842289 sm1=936.894226
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=2 mx0=37.558483 sm0=2.301245 mx1=4.842289 sm1=936.894226
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=2 mx0=37.558483 sm0=2.301245 mx1=4.842289 sm1=936.894226
[AIV Block 1/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-cpy] s2L=0 vo0=-24.034456 vo1=71.385178
[AIV Block 1/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5920 pLC=23 s2R=68
[AIV Block 1/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 1/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 1/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=42.083000 mx1=4.842289 mx2=7.169627 mx3=7.133998 sum0=1.025706[AIV Block 1/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=42.083000 iniSum[0]=1.025706 iniMax[1]=4.842289
[AIV Block 1/64]   mmRes[0]=-76.562164
[AIV Block 1/64]   mmRes[1]=-41.593208
[AIV Block 1/64]   mmRes[2]=103.800919
[AIV Block 1/64]   mmRes[3]=37.095795
[AIV Block 1/64]   mmRes[64]=-47.820580 mmRes[128]=41.709175 mmRes[192]=56.869980 mmRes[255]=19.118700
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=3 mx0=42.083000 sm0=1.025706 mx1=4.842289 sm1=942.070435
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=3 mx0=42.083000 sm0=1.025706 mx1=4.842289 sm1=942.070435
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=3 mx0=42.083000 sm0=1.025706 mx1=4.842289 sm1=942.070435
[AIV Block 1/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-expUb] s2L=1 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 1/64] [VEC2-upd] s2L=1 vo0=-9.727498 vo1=-6.684968
[AIV Block 1/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=4 s2S=4864 s2R=68 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 1/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=63488 computeMode=1 s1Off=0
[AIV Block 1/64] [COPYIN-offset] s2L=4 maskOff=63488 preOff=2112 computeMode=1 compressMode=3 s2R=68 hS1R=1 maskS2Size=2048
[AIV Block 1/64] [MASK-gm] s2L=4 gmAtDeltaOff=63488 val=0 gmAtCorrectOff=2084 val=1 s1O=1 s2O=36
[AIV Block 1/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=0 mask[1]=0 reread=0 phyAddr=1ff00
[AIV Block 1/64] [MASK-chk1] s2L=4 mask[0]=0 phyAddr=1ff00
[AIV Block 1/64] [MASK-chk2] s2L=4 mask[0]=0
[AIV Block 1/64] [MASK-chk3] s2L=4 mask[0]=0 expPhy=15700
[AIV Block 1/64] [MASK-chk4] s2L=4 mask[0]=0
[AIV Block 1/64] [MASK-chk5] s2L=4 mask[0]=0 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=42.083000 mx1=4.842289 mx2=7.169627 mx3=7.133998 sum0=1.553712[AIV Block 1/64] [VEC-Nd-pre] s2L=4 s2R=68 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=42.083000 iniSum[0]=1.553712 iniMax[1]=4.842289
[AIV Block 1/64]   mmRes[0]=116.672691
[AIV Block 1/64]   mmRes[1]=303.519928
[AIV Block 1/64]   mmRes[2]=-7.666192
[AIV Block 1/64]   mmRes[3]=43.425591
[AIV Block 1/64]   mmRes[64]=176.457581 mmRes[128]=-39.403732 mmRes[192]=-340282346638528859811704183484516925440.000000 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=0 mask[1]=0
[AIV Block 1/64] [VEC-Nd-post] s2L=4 mx0=42.083000 sm0=1.553712 mx1=4.842289 sm1=947.246643
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=4 mx0=42.083000 sm0=1.553712 mx1=4.842289 sm1=947.246643
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=4 mx0=42.083000 sm0=1.553712 mx1=4.842289 sm1=947.246643
[AIV Block 1/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 1/64] [LSE-out] lseOff=1 lse0=49.703888 lse1=49.703888
[AIV Block 1/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-expUb] s2L=2 e0=0.010840 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 1/64] [VEC2-upd] s2L=2 vo0=-9.947219 vo1=-6.713671
[AIV Block 1/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 1/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 1/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 1/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=49.672550 mx1=4.842289 mx2=7.169627 mx3=7.133998 sum0=1.031833[AIV Block 1/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=49.672550 iniSum[0]=1.031833 iniMax[1]=4.842289
[AIV Block 1/64]   mmRes[0]=-201.852905
[AIV Block 1/64]   mmRes[1]=9.176884
[AIV Block 1/64]   mmRes[2]=120.894989
[AIV Block 1/64]   mmRes[3]=-84.604309
[AIV Block 1/64]   mmRes[64]=64.178848 mmRes[128]=-185.633438 mmRes[192]=159.795166 mmRes[255]=-3.200465
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VF256-pre] m=1 oriN=256 preMax[0]=49.672550 preSum[0]=1.031833 preMax[1]=4.842289
[AIV Block 1/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-201.852905 9.176884 120.894989 -84.604309 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 1/64] [VF256-post] m=1 mx0=49.672550 sum0=1.031833 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=-340282346638528859811704183484516925440.000000
[AIV Block 1/64] [VEC-Nd-post] s2L=0 mx0=34.334976 sm0=1.035788 mx1=4.842289 sm1=952.748230
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=0 mx0=34.334976 sm0=1.035788 mx1=4.842289 sm1=952.748230
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=0 mx0=34.334976 sm0=1.035788 mx1=4.842289 sm1=952.748230
[AIV Block 1/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-expUb] s2L=3 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 1/64] [VEC2-upd] s2L=3 vo0=1.607135 vo1=3.988104
[AIV Block 1/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 1/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 1/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 1/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=34.334976 mx1=4.842289 mx2=7.169627 mx3=7.133998 sum0=1.035788[AIV Block 1/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=34.334976 iniSum[0]=1.035788 iniMax[1]=4.842289
[AIV Block 1/64]   mmRes[0]=-149.177658
[AIV Block 1/64]   mmRes[1]=43.244434
[AIV Block 1/64]   mmRes[2]=-41.155014
[AIV Block 1/64]   mmRes[3]=-66.937706
[AIV Block 1/64]   mmRes[64]=-42.070473 mmRes[128]=75.553139 mmRes[192]=-118.742989 mmRes[255]=153.161346
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=1 mx0=34.334976 sm0=1.035788 mx1=4.842289 sm1=952.748230
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=1 mx0=34.334976 sm0=1.035788 mx1=4.842289 sm1=952.748230
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=1 mx0=34.334976 sm0=1.035788 mx1=4.842289 sm1=952.748230
[AIV Block 1/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 1/64] [VEC2-upd] s2L=4 vo0=1.718036 vo1=3.755847
[AIV Block 1/64] [VEC2-final] s2L=4 sum=1.706180 max=42.083000
[AIV Block 1/64] [VEC2-out] s2L=4 vo0=0.093364 vo1=-6104.332031
[AIV Block 1/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 1/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 1/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 1/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 1/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=34.334976 mx1=4.842289 mx2=7.169627 mx3=7.133998 sum0=1.035913[AIV Block 1/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=34.334976 iniSum[0]=1.035913 iniMax[1]=4.842289
[AIV Block 1/64]   mmRes[0]=45.527115
[AIV Block 1/64]   mmRes[1]=-2.512713
[AIV Block 1/64]   mmRes[2]=-167.665756
[AIV Block 1/64]   mmRes[3]=113.721420
[AIV Block 1/64]   mmRes[64]=64.157967 mmRes[128]=-37.721077 mmRes[192]=-7.322091 mmRes[255]=-273.659332
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=2 mx0=34.334976 sm0=1.035913 mx1=4.842289 sm1=957.924438
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=2 mx0=34.334976 sm0=1.035913 mx1=4.842289 sm1=957.924438
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=2 mx0=34.334976 sm0=1.035913 mx1=4.842289 sm1=957.924438
[AIV Block 1/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 2/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 2/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 2/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=55.503086 mx1=8.399918 mx2=4.398744 mx3=3.491217 sum0=1.022325[AIV Block 2/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=55.503086 iniSum[0]=1.022325 iniMax[1]=8.399918
[AIV Block 2/64]   mmRes[0]=-125.169495
[AIV Block 2/64]   mmRes[1]=-64.456192
[AIV Block 2/64]   mmRes[2]=36.473480
[AIV Block 2/64]   mmRes[3]=-77.795570
[AIV Block 2/64]   mmRes[64]=-150.744598 mmRes[128]=55.533291 mmRes[192]=-77.896194 mmRes[255]=227.690186
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VF256-pre] m=1 oriN=256 preMax[0]=55.503086 preSum[0]=1.022325 preMax[1]=8.399918
[AIV Block 2/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-125.169495 -64.456192 36.473480 -77.795570 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 2/64] [VF256-post] m=1 mx0=55.503086 sum0=1.022325 srcPost0=-125.169495 srcPost1=-64.456192
[AIV Block 2/64] [VEC-Nd-post] s2L=0 mx0=40.175499 sm0=1.096752 mx1=8.399918 sm1=-705.396118
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=0 mx0=40.175499 sm0=1.096752 mx1=8.399918 sm1=-705.396118
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=0 mx0=40.175499 sm0=1.096752 mx1=8.399918 sm1=-705.396118
[AIV Block 2/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 2/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 2/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=40.175499 mx1=8.399918 mx2=4.398744 mx3=3.491217 sum0=1.096752[AIV Block 2/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=40.175499 iniSum[0]=1.096752 iniMax[1]=8.399918
[AIV Block 2/64]   mmRes[0]=-16.704733
[AIV Block 2/64]   mmRes[1]=228.097992
[AIV Block 2/64]   mmRes[2]=-132.405487
[AIV Block 2/64]   mmRes[3]=144.993408
[AIV Block 2/64]   mmRes[64]=39.426411 mmRes[128]=98.555397 mmRes[192]=57.149570 mmRes[255]=-69.360847
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=1 mx0=40.175499 sm0=1.096752 mx1=8.399918 sm1=-705.396118
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=1 mx0=40.175499 sm0=1.096752 mx1=8.399918 sm1=-705.396118
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=1 mx0=40.175499 sm0=1.096752 mx1=8.399918 sm1=-705.396118
[AIV Block 2/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 2/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 2/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 2/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=53.956337 mx1=8.399918 mx2=4.398744 mx3=3.491217 sum0=1.000136[AIV Block 2/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=53.956337 iniSum[0]=1.000136 iniMax[1]=8.399918
[AIV Block 2/64]   mmRes[0]=77.366837
[AIV Block 2/64]   mmRes[1]=101.332100
[AIV Block 2/64]   mmRes[2]=-80.005608
[AIV Block 2/64]   mmRes[3]=9.095348
[AIV Block 2/64]   mmRes[64]=-7.408249 mmRes[128]=-80.578339 mmRes[192]=15.068103 mmRes[255]=-203.842422
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=2 mx0=53.956337 sm0=1.000136 mx1=8.399918 sm1=-709.315002
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=2 mx0=53.956337 sm0=1.000136 mx1=8.399918 sm1=-709.315002
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=2 mx0=53.956337 sm0=1.000136 mx1=8.399918 sm1=-709.315002
[AIV Block 2/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-cpy] s2L=0 vo0=23582.773438 vo1=-175380.890625
[AIV Block 2/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5920 pLC=23 s2R=68
[AIV Block 2/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 2/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 2/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=53.956337 mx1=8.399918 mx2=4.398744 mx3=3.491217 sum0=1.000251[AIV Block 2/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=53.956337 iniSum[0]=1.000251 iniMax[1]=8.399918
[AIV Block 2/64]   mmRes[0]=-133.754807
[AIV Block 2/64]   mmRes[1]=0.850524
[AIV Block 2/64]   mmRes[2]=-92.974884
[AIV Block 2/64]   mmRes[3]=-77.640137
[AIV Block 2/64]   mmRes[64]=-115.059944 mmRes[128]=-105.925255 mmRes[192]=188.816910 mmRes[255]=-9.187069
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=3 mx0=53.956337 sm0=1.000251 mx1=8.399918 sm1=-713.233887
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=3 mx0=53.956337 sm0=1.000251 mx1=8.399918 sm1=-713.233887
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=3 mx0=53.956337 sm0=1.000251 mx1=8.399918 sm1=-713.233887
[AIV Block 2/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-expUb] s2L=1 e0=0.000001 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 2/64] [VEC2-upd] s2L=1 vo0=3.910005 vo1=0.099569
[AIV Block 2/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=4 s2S=4864 s2R=68 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 2/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=61440 computeMode=1 s1Off=0
[AIV Block 2/64] [COPYIN-offset] s2L=4 maskOff=61440 preOff=64 computeMode=1 compressMode=3 s2R=68 hS1R=1 maskS2Size=2048
[AIV Block 2/64] [MASK-gm] s2L=4 gmAtDeltaOff=61440 val=0 gmAtCorrectOff=36 val=1 s1O=0 s2O=36
[AIV Block 2/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=0 mask[1]=0 reread=0 phyAddr=1ff00
[AIV Block 2/64] [MASK-chk1] s2L=4 mask[0]=0 phyAddr=1ff00
[AIV Block 2/64] [MASK-chk2] s2L=4 mask[0]=0
[AIV Block 2/64] [MASK-chk3] s2L=4 mask[0]=0 expPhy=15700
[AIV Block 2/64] [MASK-chk4] s2L=4 mask[0]=0
[AIV Block 2/64] [MASK-chk5] s2L=4 mask[0]=0 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=53.956337 mx1=8.399918 mx2=4.398744 mx3=3.491217 sum0=1.071769[AIV Block 2/64] [VEC-Nd-pre] s2L=4 s2R=68 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=53.956337 iniSum[0]=1.071769 iniMax[1]=8.399918
[AIV Block 2/64]   mmRes[0]=-55.296902
[AIV Block 2/64]   mmRes[1]=190.675949
[AIV Block 2/64]   mmRes[2]=80.816757
[AIV Block 2/64]   mmRes[3]=15.105571
[AIV Block 2/64]   mmRes[64]=-160.400131 mmRes[128]=-10.672861 mmRes[192]=1.995819 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=0 mask[1]=0
[AIV Block 2/64] [VEC-Nd-post] s2L=4 mx0=53.956337 sm0=1.071769 mx1=8.399918 sm1=-717.152771
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=4 mx0=53.956337 sm0=1.071769 mx1=8.399918 sm1=-717.152771
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=4 mx0=53.956337 sm0=1.071769 mx1=8.399918 sm1=-717.152771
[AIV Block 2/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 2/64] [LSE-out] lseOff=274 lse0=55.525166 lse1=55.525166
[AIV Block 2/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-expUb] s2L=2 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 2/64] [VEC2-upd] s2L=2 vo0=-8.421042 vo1=5.334634
[AIV Block 2/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 2/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 2/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 2/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=50.285591 mx1=8.399918 mx2=4.398744 mx3=3.491217 sum0=1.702013[AIV Block 2/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=50.285591 iniSum[0]=1.702013 iniMax[1]=8.399918
[AIV Block 2/64]   mmRes[0]=-157.814499
[AIV Block 2/64]   mmRes[1]=-235.191803
[AIV Block 2/64]   mmRes[2]=-128.693069
[AIV Block 2/64]   mmRes[3]=-176.261032
[AIV Block 2/64]   mmRes[64]=-60.175579 mmRes[128]=-65.413895 mmRes[192]=-55.113869 mmRes[255]=128.714905
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VF256-pre] m=1 oriN=256 preMax[0]=50.285591 preSum[0]=1.702013 preMax[1]=8.399918
[AIV Block 2/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-157.814499 -235.191803 -128.693069 -176.261032 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 2/64] [VF256-post] m=1 mx0=50.285591 sum0=1.702013 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC-Nd-post] s2L=0 mx0=43.987648 sm0=1.000023 mx1=8.399918 sm1=-689.720642
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=0 mx0=43.987648 sm0=1.000023 mx1=8.399918 sm1=-689.720642
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=0 mx0=43.987648 sm0=1.000023 mx1=8.399918 sm1=-689.720642
[AIV Block 2/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-expUb] s2L=3 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 2/64] [VEC2-upd] s2L=3 vo0=-8.422127 vo1=5.335546
[AIV Block 2/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 2/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 2/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 2/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=43.987648 mx1=8.399918 mx2=4.398744 mx3=3.491217 sum0=1.000023[AIV Block 2/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=43.987648 iniSum[0]=1.000023 iniMax[1]=8.399918
[AIV Block 2/64]   mmRes[0]=-163.482773
[AIV Block 2/64]   mmRes[1]=-2.749908
[AIV Block 2/64]   mmRes[2]=39.349976
[AIV Block 2/64]   mmRes[3]=118.493744
[AIV Block 2/64]   mmRes[64]=79.225220 mmRes[128]=-6.295664 mmRes[192]=-116.942017 mmRes[255]=180.853607
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=1 mx0=43.987648 sm0=1.000023 mx1=8.399918 sm1=-689.720642
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=1 mx0=43.987648 sm0=1.000023 mx1=8.399918 sm1=-689.720642
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=1 mx0=43.987648 sm0=1.000023 mx1=8.399918 sm1=-689.720642
[AIV Block 2/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 2/64] [VEC2-upd] s2L=4 vo0=-8.777897 vo1=5.286971
[AIV Block 2/64] [VEC2-final] s2L=4 sum=1.071769 max=53.956337
[AIV Block 2/64] [VEC2-out] s2L=4 vo0=1918.252930 vo1=71560.687500
[AIV Block 2/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 2/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 2/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 2/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 2/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=45.325432 mx1=8.399918 mx2=4.398744 mx3=3.491217 sum0=1.268956[AIV Block 2/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=45.325432 iniSum[0]=1.268956 iniMax[1]=8.399918
[AIV Block 2/64]   mmRes[0]=-14.359776
[AIV Block 2/64]   mmRes[1]=-47.104996
[AIV Block 2/64]   mmRes[2]=49.874458
[AIV Block 2/64]   mmRes[3]=-49.418121
[AIV Block 2/64]   mmRes[64]=-75.732895 mmRes[128]=180.980057 mmRes[192]=-36.980381 mmRes[255]=-294.234802
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=2 mx0=45.325432 sm0=1.268956 mx1=8.399918 sm1=-693.639526
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=2 mx0=45.325432 sm0=1.268956 mx1=8.399918 sm1=-693.639526
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=2 mx0=45.325432 sm0=1.268956 mx1=8.399918 sm1=-693.639526
[AIV Block 2/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 3/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 3/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 3/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=44.519169 mx1=9.723917 mx2=9.704822 mx3=2.985296 sum0=1.582161[AIV Block 3/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=44.519169 iniSum[0]=1.582161 iniMax[1]=9.723917
[AIV Block 3/64]   mmRes[0]=34.025860
[AIV Block 3/64]   mmRes[1]=44.390648
[AIV Block 3/64]   mmRes[2]=-206.972824
[AIV Block 3/64]   mmRes[3]=132.023972
[AIV Block 3/64]   mmRes[64]=74.749847 mmRes[128]=66.008720 mmRes[192]=30.640432 mmRes[255]=-44.640411
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VF256-pre] m=1 oriN=256 preMax[0]=44.519169 preSum[0]=1.582161 preMax[1]=9.723917
[AIV Block 3/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=34.025860 44.390648 -206.972824 132.023972 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 3/64] [VF256-post] m=1 mx0=44.519169 sum0=1.582161 srcPost0=34.025860 srcPost1=44.390648
[AIV Block 3/64] [VEC-Nd-post] s2L=0 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1766.074097
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=0 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1766.074097
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=0 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1766.074097
[AIV Block 3/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 3/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 3/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=74.001419 mx1=9.723917 mx2=9.704822 mx3=2.985296 sum0=1.000041[AIV Block 3/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=74.001419 iniSum[0]=1.000041 iniMax[1]=9.723917
[AIV Block 3/64]   mmRes[0]=264.953857
[AIV Block 3/64]   mmRes[1]=-163.873779
[AIV Block 3/64]   mmRes[2]=-42.622395
[AIV Block 3/64]   mmRes[3]=204.392273
[AIV Block 3/64]   mmRes[64]=74.333931 mmRes[128]=91.743149 mmRes[192]=-94.705299 mmRes[255]=-23.242102
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=1 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1766.074097
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=1 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1766.074097
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=1 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1766.074097
[AIV Block 3/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 3/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 3/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 3/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=74.001419 mx1=9.723917 mx2=9.704822 mx3=2.985296 sum0=1.000041[AIV Block 3/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=74.001419 iniSum[0]=1.000041 iniMax[1]=9.723917
[AIV Block 3/64]   mmRes[0]=165.002991
[AIV Block 3/64]   mmRes[1]=-114.294037
[AIV Block 3/64]   mmRes[2]=122.479958
[AIV Block 3/64]   mmRes[3]=-270.235413
[AIV Block 3/64]   mmRes[64]=92.062225 mmRes[128]=106.293793 mmRes[192]=-229.722534 mmRes[255]=-18.126114
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=2 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1775.885620
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=2 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1775.885620
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=2 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1775.885620
[AIV Block 3/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-cpy] s2L=0 vo0=0.001204 vo1=-931.064026
[AIV Block 3/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5920 pLC=23 s2R=68
[AIV Block 3/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 3/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 3/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=74.001419 mx1=9.723917 mx2=9.704822 mx3=2.985296 sum0=1.000041[AIV Block 3/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=74.001419 iniSum[0]=1.000041 iniMax[1]=9.723917
[AIV Block 3/64]   mmRes[0]=-81.187454
[AIV Block 3/64]   mmRes[1]=-71.472954
[AIV Block 3/64]   mmRes[2]=-142.595901
[AIV Block 3/64]   mmRes[3]=-96.165047
[AIV Block 3/64]   mmRes[64]=-172.149200 mmRes[128]=-84.963181 mmRes[192]=117.730278 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=3 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1785.697144
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=3 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1785.697144
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=3 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1785.697144
[AIV Block 3/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-expUb] s2L=1 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 3/64] [VEC2-upd] s2L=1 vo0=7.316458 vo1=-2.043158
[AIV Block 3/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=4 s2S=4864 s2R=68 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 3/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=63488 computeMode=1 s1Off=0
[AIV Block 3/64] [COPYIN-offset] s2L=4 maskOff=63488 preOff=2112 computeMode=1 compressMode=3 s2R=68 hS1R=1 maskS2Size=2048
[AIV Block 3/64] [MASK-gm] s2L=4 gmAtDeltaOff=63488 val=0 gmAtCorrectOff=2084 val=1 s1O=1 s2O=36
[AIV Block 3/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=0 mask[1]=0 reread=0 phyAddr=1ff00
[AIV Block 3/64] [MASK-chk1] s2L=4 mask[0]=0 phyAddr=1ff00
[AIV Block 3/64] [MASK-chk2] s2L=4 mask[0]=0
[AIV Block 3/64] [MASK-chk3] s2L=4 mask[0]=0 expPhy=15700
[AIV Block 3/64] [MASK-chk4] s2L=4 mask[0]=0
[AIV Block 3/64] [MASK-chk5] s2L=4 mask[0]=0 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=74.001419 mx1=9.723917 mx2=9.704822 mx3=2.985296 sum0=1.000041[AIV Block 3/64] [VEC-Nd-pre] s2L=4 s2R=68 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=74.001419 iniSum[0]=1.000041 iniMax[1]=9.723917
[AIV Block 3/64]   mmRes[0]=-66.643448
[AIV Block 3/64]   mmRes[1]=-62.148781
[AIV Block 3/64]   mmRes[2]=187.988342
[AIV Block 3/64]   mmRes[3]=276.241241
[AIV Block 3/64]   mmRes[64]=51.241177 mmRes[128]=-340282346638528859811704183484516925440.000000 mmRes[192]=-340282346638528859811704183484516925440.000000 mmRes[255]=-2.400862
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=0 mask[1]=0
[AIV Block 3/64] [VEC-Nd-post] s2L=4 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1795.508667
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=4 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1795.508667
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=4 mx0=74.001419 sm0=1.000041 mx1=9.723917 sm1=-1795.508667
[AIV Block 3/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 3/64] [LSE-out] lseOff=275 lse0=44.977962 lse1=44.977962
[AIV Block 3/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-expUb] s2L=2 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 3/64] [VEC2-upd] s2L=2 vo0=7.316457 vo1=-2.043157
[AIV Block 3/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 3/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 3/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 3/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=61.717396 mx1=9.723917 mx2=9.704822 mx3=2.985296 sum0=1.000173[AIV Block 3/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=61.717396 iniSum[0]=1.000173 iniMax[1]=9.723917
[AIV Block 3/64]   mmRes[0]=163.337601
[AIV Block 3/64]   mmRes[1]=227.999908
[AIV Block 3/64]   mmRes[2]=-38.281574
[AIV Block 3/64]   mmRes[3]=-83.831711
[AIV Block 3/64]   mmRes[64]=78.404800 mmRes[128]=-49.674755 mmRes[192]=127.848785 mmRes[255]=106.629509
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VF256-pre] m=1 oriN=256 preMax[0]=61.717396 preSum[0]=1.000173 preMax[1]=9.723917
[AIV Block 3/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=163.337601 227.999908 -38.281574 -83.831711 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 3/64] [VF256-post] m=1 mx0=61.717396 sum0=1.000173 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=-340282346638528859811704183484516925440.000000
[AIV Block 3/64] [VEC-Nd-post] s2L=0 mx0=47.751923 sm0=1.005068 mx1=9.723917 sm1=-1760.578979
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=0 mx0=47.751923 sm0=1.005068 mx1=9.723917 sm1=-1760.578979
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=0 mx0=47.751923 sm0=1.005068 mx1=9.723917 sm1=-1760.578979
[AIV Block 3/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-expUb] s2L=3 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 3/64] [VEC2-upd] s2L=3 vo0=7.316457 vo1=-2.043157
[AIV Block 3/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 3/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 3/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 3/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=47.751923 mx1=9.723917 mx2=9.704822 mx3=2.985296 sum0=1.005068[AIV Block 3/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=47.751923 iniSum[0]=1.005068 iniMax[1]=9.723917
[AIV Block 3/64]   mmRes[0]=210.167313
[AIV Block 3/64]   mmRes[1]=103.239075
[AIV Block 3/64]   mmRes[2]=193.046234
[AIV Block 3/64]   mmRes[3]=-47.906284
[AIV Block 3/64]   mmRes[64]=22.739819 mmRes[128]=115.317703 mmRes[192]=291.129059 mmRes[255]=-113.435341
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=1 mx0=47.751923 sm0=1.005068 mx1=9.723917 sm1=-1760.578979
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=1 mx0=47.751923 sm0=1.005068 mx1=9.723917 sm1=-1760.578979
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=1 mx0=47.751923 sm0=1.005068 mx1=9.723917 sm1=-1760.578979
[AIV Block 3/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 3/64] [VEC2-upd] s2L=4 vo0=7.316457 vo1=-2.043157
[AIV Block 3/64] [VEC2-final] s2L=4 sum=1.000481 max=74.001419
[AIV Block 3/64] [VEC2-out] s2L=4 vo0=-2.332478 vo1=-0.253393
[AIV Block 3/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 3/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 3/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 3/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 3/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=47.751923 mx1=9.723917 mx2=9.704822 mx3=2.985296 sum0=1.005237[AIV Block 3/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=47.751923 iniSum[0]=1.005237 iniMax[1]=9.723917
[AIV Block 3/64]   mmRes[0]=429.602295
[AIV Block 3/64]   mmRes[1]=150.630005
[AIV Block 3/64]   mmRes[2]=23.331858
[AIV Block 3/64]   mmRes[3]=175.337067
[AIV Block 3/64]   mmRes[64]=115.814621 mmRes[128]=259.989105 mmRes[192]=-87.225746 mmRes[255]=230.818848
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=2 mx0=47.751923 sm0=1.005237 mx1=9.723917 sm1=-1770.390503
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=2 mx0=47.751923 sm0=1.005237 mx1=9.723917 sm1=-1770.390503
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=2 mx0=47.751923 sm0=1.005237 mx1=9.723917 sm1=-1770.390503
[AIV Block 3/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 4/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 4/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 4/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=40.445896 mx1=7.538249 mx2=8.706434 mx3=2.874609 sum0=3.072227[AIV Block 4/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=40.445896 iniSum[0]=3.072227 iniMax[1]=7.538249
[AIV Block 4/64]   mmRes[0]=-93.664589
[AIV Block 4/64]   mmRes[1]=27.924973
[AIV Block 4/64]   mmRes[2]=48.886803
[AIV Block 4/64]   mmRes[3]=181.801682
[AIV Block 4/64]   mmRes[64]=96.948227 mmRes[128]=-118.057198 mmRes[192]=-224.574966 mmRes[255]=-174.877838
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VF256-pre] m=1 oriN=256 preMax[0]=40.445896 preSum[0]=3.072227 preMax[1]=7.538249
[AIV Block 4/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-93.664589 27.924973 48.886803 181.801682 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 4/64] [VF256-post] m=1 mx0=40.445896 sum0=3.072227 srcPost0=-93.664589 srcPost1=27.924973
[AIV Block 4/64] [VEC-Nd-post] s2L=0 mx0=41.391850 sm0=1.112879 mx1=7.538249 sm1=153.636459
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=0 mx0=41.391850 sm0=1.112879 mx1=7.538249 sm1=153.636459
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=0 mx0=41.391850 sm0=1.112879 mx1=7.538249 sm1=153.636459
[AIV Block 4/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 4/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 4/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=41.391850 mx1=7.538249 mx2=8.706434 mx3=2.874609 sum0=1.112879[AIV Block 4/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=41.391850 iniSum[0]=1.112879 iniMax[1]=7.538249
[AIV Block 4/64]   mmRes[0]=20.470823
[AIV Block 4/64]   mmRes[1]=-165.920227
[AIV Block 4/64]   mmRes[2]=-65.618553
[AIV Block 4/64]   mmRes[3]=21.730583
[AIV Block 4/64]   mmRes[64]=-116.933510 mmRes[128]=99.366310 mmRes[192]=-56.323349 mmRes[255]=-125.929047
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=1 mx0=41.391850 sm0=1.112879 mx1=7.538249 sm1=153.636459
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=1 mx0=41.391850 sm0=1.112879 mx1=7.538249 sm1=153.636459
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=1 mx0=41.391850 sm0=1.112879 mx1=7.538249 sm1=153.636459
[AIV Block 4/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5911 pLC=23 s2R=220
[AIV Block 4/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 4/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 4/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=41.391850 mx1=7.538249 mx2=8.706434 mx3=2.874609 sum0=1.162414[AIV Block 4/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=41.391850 iniSum[0]=1.162414 iniMax[1]=7.538249
[AIV Block 4/64]   mmRes[0]=48.862591
[AIV Block 4/64]   mmRes[1]=-52.251770
[AIV Block 4/64]   mmRes[2]=77.343079
[AIV Block 4/64]   mmRes[3]=124.626068
[AIV Block 4/64]   mmRes[64]=69.574768 mmRes[128]=-18.237240 mmRes[192]=62.597969 mmRes[255]=-203.987793
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=2 mx0=41.391850 sm0=1.162414 mx1=7.538249 sm1=154.489990
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=2 mx0=41.391850 sm0=1.162414 mx1=7.538249 sm1=154.489990
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=2 mx0=41.391850 sm0=1.162414 mx1=7.538249 sm1=154.489990
[AIV Block 4/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-cpy] s2L=0 vo0=-138526.359375 vo1=-3.558614
[AIV Block 4/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5911 pLC=23 s2R=59
[AIV Block 4/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 4/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 4/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=41.391850 mx1=7.538249 mx2=8.706434 mx3=2.874609 sum0=1.214057[AIV Block 4/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=41.391850 iniSum[0]=1.214057 iniMax[1]=7.538249
[AIV Block 4/64]   mmRes[0]=33.649502
[AIV Block 4/64]   mmRes[1]=19.118299
[AIV Block 4/64]   mmRes[2]=-129.651047
[AIV Block 4/64]   mmRes[3]=-107.886353
[AIV Block 4/64]   mmRes[64]=-125.340744 mmRes[128]=-8.734537 mmRes[192]=36.026272 mmRes[255]=-16.679708
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=3 mx0=41.391850 sm0=1.214057 mx1=7.538249 sm1=155.343521
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=3 mx0=41.391850 sm0=1.214057 mx1=7.538249 sm1=155.343521
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=3 mx0=41.391850 sm0=1.214057 mx1=7.538249 sm1=155.343521
[AIV Block 4/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-expUb] s2L=1 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 4/64] [VEC2-upd] s2L=1 vo0=-4.866573 vo1=-7.689322
[AIV Block 4/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=4 s2S=4864 s2R=59 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 4/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5794 deltaC=21 deltaPre=-925 maskOff=43008 computeMode=1 s1Off=0
[AIV Block 4/64] [COPYIN-offset] s2L=4 maskOff=43008 preOff=64 computeMode=1 compressMode=3 s2R=59 hS1R=1 maskS2Size=2048
[AIV Block 4/64] [MASK-gm] s2L=4 gmAtDeltaOff=43008 val=0 gmAtCorrectOff=36 val=1 s1O=0 s2O=36
[AIV Block 4/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=0 mask[1]=0 reread=0 phyAddr=1ff00
[AIV Block 4/64] [MASK-chk1] s2L=4 mask[0]=0 phyAddr=1ff00
[AIV Block 4/64] [MASK-chk2] s2L=4 mask[0]=0
[AIV Block 4/64] [MASK-chk3] s2L=4 mask[0]=0 expPhy=15700
[AIV Block 4/64] [MASK-chk4] s2L=4 mask[0]=0
[AIV Block 4/64] [MASK-chk5] s2L=4 mask[0]=0 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=43.086498 mx1=7.538249 mx2=8.706434 mx3=2.874609 sum0=1.222999[AIV Block 4/64] [VEC-Nd-pre] s2L=4 s2R=59 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=43.086498 iniSum[0]=1.222999 iniMax[1]=7.538249
[AIV Block 4/64]   mmRes[0]=100.587135
[AIV Block 4/64]   mmRes[1]=13.528423
[AIV Block 4/64]   mmRes[2]=196.564133
[AIV Block 4/64]   mmRes[3]=68.044609
[AIV Block 4/64]   mmRes[64]=-340282346638528859811704183484516925440.000000 mmRes[128]=-2.415581 mmRes[192]=8.291304 mmRes[255]=-27.018843
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=0 mask[1]=0
[AIV Block 4/64] [VEC-Nd-post] s2L=4 mx0=43.086498 sm0=1.222999 mx1=7.538249 sm1=156.197052
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=4 mx0=43.086498 sm0=1.222999 mx1=7.538249 sm1=156.197052
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=4 mx0=43.086498 sm0=1.222999 mx1=7.538249 sm1=156.197052
[AIV Block 4/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 4/64] [LSE-out] lseOff=546 lse0=52.738995 lse1=52.738995
[AIV Block 4/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-expUb] s2L=2 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 4/64] [VEC2-upd] s2L=2 vo0=-4.656067 vo1=-7.798247
[AIV Block 4/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 4/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 4/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 4/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=52.201279 mx1=7.538249 mx2=8.706434 mx3=2.874609 sum0=1.712094[AIV Block 4/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=52.201279 iniSum[0]=1.712094 iniMax[1]=7.538249
[AIV Block 4/64]   mmRes[0]=177.786774
[AIV Block 4/64]   mmRes[1]=-75.375687
[AIV Block 4/64]   mmRes[2]=-127.959305
[AIV Block 4/64]   mmRes[3]=93.991081
[AIV Block 4/64]   mmRes[64]=36.104656 mmRes[128]=-206.682037 mmRes[192]=-63.603222 mmRes[255]=6.758065
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VF256-pre] m=1 oriN=256 preMax[0]=52.201279 preSum[0]=1.712094 preMax[1]=7.538249
[AIV Block 4/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=177.786774 -75.375687 -127.959305 93.991081 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 4/64] [VF256-post] m=1 mx0=52.201279 sum0=1.712094 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VEC-Nd-post] s2L=0 mx0=45.602409 sm0=1.001747 mx1=7.538249 sm1=151.796280
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=0 mx0=45.602409 sm0=1.001747 mx1=7.538249 sm1=151.796280
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=0 mx0=45.602409 sm0=1.001747 mx1=7.538249 sm1=151.796280
[AIV Block 4/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-expUb] s2L=3 e0=0.183664 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 4/64] [VEC2-upd] s2L=3 vo0=-4.777345 vo1=-7.553783
[AIV Block 4/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 4/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 4/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 4/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=45.602409 mx1=7.538249 mx2=8.706434 mx3=2.874609 sum0=1.001747[AIV Block 4/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=45.602409 iniSum[0]=1.001747 iniMax[1]=7.538249
[AIV Block 4/64]   mmRes[0]=22.424114
[AIV Block 4/64]   mmRes[1]=-96.684402
[AIV Block 4/64]   mmRes[2]=-20.044884
[AIV Block 4/64]   mmRes[3]=14.714994
[AIV Block 4/64]   mmRes[64]=-44.980110 mmRes[128]=19.091061 mmRes[192]=-130.354736 mmRes[255]=-102.373268
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=1 mx0=45.602409 sm0=1.001747 mx1=7.538249 sm1=151.796280
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=1 mx0=45.602409 sm0=1.001747 mx1=7.538249 sm1=151.796280
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=1 mx0=45.602409 sm0=1.001747 mx1=7.538249 sm1=151.796280
[AIV Block 4/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 4/64] [VEC2-upd] s2L=4 vo0=-3.156631 vo1=4.522867
[AIV Block 4/64] [VEC2-final] s2L=4 sum=1.222999 max=43.086498
[AIV Block 4/64] [VEC2-out] s2L=4 vo0=229.754547 vo1=-6808.877441
[AIV Block 4/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5911 pLC=23 s2R=220
[AIV Block 4/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 4/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 4/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 4/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=45.602409 mx1=7.538249 mx2=8.706434 mx3=2.874609 sum0=1.012717[AIV Block 4/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=45.602409 iniSum[0]=1.012717 iniMax[1]=7.538249
[AIV Block 4/64]   mmRes[0]=117.485893
[AIV Block 4/64]   mmRes[1]=1.255013
[AIV Block 4/64]   mmRes[2]=-19.652077
[AIV Block 4/64]   mmRes[3]=215.153351
[AIV Block 4/64]   mmRes[64]=-114.929337 mmRes[128]=159.025192 mmRes[192]=-22.867235 mmRes[255]=24.193621
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=2 mx0=45.602409 sm0=1.012717 mx1=7.538249 sm1=152.649811
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=2 mx0=45.602409 sm0=1.012717 mx1=7.538249 sm1=152.649811
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=2 mx0=45.602409 sm0=1.012717 mx1=7.538249 sm1=152.649811
[AIV Block 4/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 5/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 5/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 5/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=37.245216 mx1=5.402357 mx2=7.963130 mx3=9.597655 sum0=1.912705[AIV Block 5/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=37.245216 iniSum[0]=1.912705 iniMax[1]=5.402357
[AIV Block 5/64]   mmRes[0]=-15.134365
[AIV Block 5/64]   mmRes[1]=-50.951782
[AIV Block 5/64]   mmRes[2]=123.908264
[AIV Block 5/64]   mmRes[3]=-198.198227
[AIV Block 5/64]   mmRes[64]=-14.840070 mmRes[128]=-60.491108 mmRes[192]=198.530731 mmRes[255]=115.616882
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VF256-pre] m=1 oriN=256 preMax[0]=37.245216 preSum[0]=1.912705 preMax[1]=5.402357
[AIV Block 5/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-15.134365 -50.951782 123.908264 -198.198227 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 5/64] [VF256-post] m=1 mx0=37.245216 sum0=1.912705 srcPost0=-15.134365 srcPost1=-50.951782
[AIV Block 5/64] [VEC-Nd-post] s2L=0 mx0=34.638496 sm0=1.053172 mx1=5.402357 sm1=-1230.445923
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=0 mx0=34.638496 sm0=1.053172 mx1=5.402357 sm1=-1230.445923
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=0 mx0=34.638496 sm0=1.053172 mx1=5.402357 sm1=-1230.445923
[AIV Block 5/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 5/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 5/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=34.638496 mx1=5.402357 mx2=7.963130 mx3=9.597655 sum0=1.053172[AIV Block 5/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=34.638496 iniSum[0]=1.053172 iniMax[1]=5.402357
[AIV Block 5/64]   mmRes[0]=-1.841506
[AIV Block 5/64]   mmRes[1]=40.323288
[AIV Block 5/64]   mmRes[2]=-59.346825
[AIV Block 5/64]   mmRes[3]=160.995560
[AIV Block 5/64]   mmRes[64]=-364.171173 mmRes[128]=6.197699 mmRes[192]=8.743000 mmRes[255]=83.786423
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=1 mx0=34.638496 sm0=1.053172 mx1=5.402357 sm1=-1230.445923
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=1 mx0=34.638496 sm0=1.053172 mx1=5.402357 sm1=-1230.445923
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=1 mx0=34.638496 sm0=1.053172 mx1=5.402357 sm1=-1230.445923
[AIV Block 5/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5911 pLC=23 s2R=220
[AIV Block 5/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 5/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 5/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=37.551838 mx1=5.402357 mx2=7.963130 mx3=9.597655 sum0=1.625240[AIV Block 5/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=37.551838 iniSum[0]=1.625240 iniMax[1]=5.402357
[AIV Block 5/64]   mmRes[0]=-108.545067
[AIV Block 5/64]   mmRes[1]=-137.140320
[AIV Block 5/64]   mmRes[2]=-40.101273
[AIV Block 5/64]   mmRes[3]=-126.606567
[AIV Block 5/64]   mmRes[64]=-162.105286 mmRes[128]=0.593231 mmRes[192]=-157.370224 mmRes[255]=199.699066
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=2 mx0=37.551838 sm0=1.625240 mx1=5.402357 sm1=-1237.281738
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=2 mx0=37.551838 sm0=1.625240 mx1=5.402357 sm1=-1237.281738
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=2 mx0=37.551838 sm0=1.625240 mx1=5.402357 sm1=-1237.281738
[AIV Block 5/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-cpy] s2L=0 vo0=249107.078125 vo1=36934.882812
[AIV Block 5/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5911 pLC=23 s2R=59
[AIV Block 5/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 5/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 5/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=44.944931 mx1=5.402357 mx2=7.963130 mx3=9.597655 sum0=1.268525[AIV Block 5/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=44.944931 iniSum[0]=1.268525 iniMax[1]=5.402357
[AIV Block 5/64]   mmRes[0]=101.405975
[AIV Block 5/64]   mmRes[1]=-21.287367
[AIV Block 5/64]   mmRes[2]=20.731651
[AIV Block 5/64]   mmRes[3]=70.447327
[AIV Block 5/64]   mmRes[64]=401.261383 mmRes[128]=1.293064 mmRes[192]=65.004059 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=3 mx0=44.944931 sm0=1.268525 mx1=5.402357 sm1=-1244.117554
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=3 mx0=44.944931 sm0=1.268525 mx1=5.402357 sm1=-1244.117554
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=3 mx0=44.944931 sm0=1.268525 mx1=5.402357 sm1=-1244.117554
[AIV Block 5/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-expUb] s2L=1 e0=0.054294 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 5/64] [VEC2-upd] s2L=1 vo0=-3.639362 vo1=1.814614
[AIV Block 5/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=4 s2S=4864 s2R=59 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 5/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5794 deltaC=21 deltaPre=-925 maskOff=45056 computeMode=1 s1Off=0
[AIV Block 5/64] [COPYIN-offset] s2L=4 maskOff=45056 preOff=2112 computeMode=1 compressMode=3 s2R=59 hS1R=1 maskS2Size=2048
[AIV Block 5/64] [MASK-gm] s2L=4 gmAtDeltaOff=45056 val=0 gmAtCorrectOff=2084 val=1 s1O=1 s2O=36
[AIV Block 5/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=0 mask[1]=0 reread=0 phyAddr=1ff00
[AIV Block 5/64] [MASK-chk1] s2L=4 mask[0]=0 phyAddr=1ff00
[AIV Block 5/64] [MASK-chk2] s2L=4 mask[0]=0
[AIV Block 5/64] [MASK-chk3] s2L=4 mask[0]=0 expPhy=15700
[AIV Block 5/64] [MASK-chk4] s2L=4 mask[0]=0
[AIV Block 5/64] [MASK-chk5] s2L=4 mask[0]=0 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=65.988426 mx1=5.402357 mx2=7.963130 mx3=9.597655 sum0=1.000000[AIV Block 5/64] [VEC-Nd-pre] s2L=4 s2R=59 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=65.988426 iniSum[0]=1.000000 iniMax[1]=5.402357
[AIV Block 5/64]   mmRes[0]=-305.907990
[AIV Block 5/64]   mmRes[1]=250.200974
[AIV Block 5/64]   mmRes[2]=-36.608837
[AIV Block 5/64]   mmRes[3]=-111.078819
[AIV Block 5/64]   mmRes[64]=-340282346638528859811704183484516925440.000000 mmRes[128]=-340282346638528859811704183484516925440.000000 mmRes[192]=-340282346638528859811704183484516925440.000000 mmRes[255]=26.450787
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=0 mask[1]=0
[AIV Block 5/64] [VEC-Nd-post] s2L=4 mx0=65.988426 sm0=1.000000 mx1=5.402357 sm1=-1250.953369
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=4 mx0=65.988426 sm0=1.000000 mx1=5.402357 sm1=-1250.953369
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=4 mx0=65.988426 sm0=1.000000 mx1=5.402357 sm1=-1250.953369
[AIV Block 5/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 5/64] [LSE-out] lseOff=547 lse0=48.861492 lse1=48.861492
[AIV Block 5/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-expUb] s2L=2 e0=0.000615 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 5/64] [VEC2-upd] s2L=2 vo0=-10.807304 vo1=0.567751
[AIV Block 5/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 5/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 5/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 5/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=48.861351 mx1=5.402357 mx2=7.963130 mx3=9.597655 sum0=1.000140[AIV Block 5/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=48.861351 iniSum[0]=1.000140 iniMax[1]=5.402357
[AIV Block 5/64]   mmRes[0]=-174.462509
[AIV Block 5/64]   mmRes[1]=36.842709
[AIV Block 5/64]   mmRes[2]=25.186804
[AIV Block 5/64]   mmRes[3]=-114.386284
[AIV Block 5/64]   mmRes[64]=-232.508316 mmRes[128]=-242.638168 mmRes[192]=-254.979156 mmRes[255]=51.407951
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VF256-pre] m=1 oriN=256 preMax[0]=48.861351 preSum[0]=1.000140 preMax[1]=5.402357
[AIV Block 5/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-174.462509 36.842709 25.186804 -114.386284 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 5/64] [VF256-post] m=1 mx0=48.861351 sum0=1.000140 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=-340282346638528859811704183484516925440.000000
[AIV Block 5/64] [VEC-Nd-post] s2L=0 mx0=47.941551 sm0=1.000000 mx1=5.402357 sm1=-1250.478394
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=0 mx0=47.941551 sm0=1.000000 mx1=5.402357 sm1=-1250.478394
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=0 mx0=47.941551 sm0=1.000000 mx1=5.402357 sm1=-1250.478394
[AIV Block 5/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-expUb] s2L=3 e0=0.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 5/64] [VEC2-upd] s2L=3 vo0=7.648346 vo1=6.820942
[AIV Block 5/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 5/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 5/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 5/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=47.941551 mx1=5.402357 mx2=7.963130 mx3=9.597655 sum0=1.000000[AIV Block 5/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=47.941551 iniSum[0]=1.000000 iniMax[1]=5.402357
[AIV Block 5/64]   mmRes[0]=270.163208
[AIV Block 5/64]   mmRes[1]=-169.411316
[AIV Block 5/64]   mmRes[2]=-140.990387
[AIV Block 5/64]   mmRes[3]=-123.014648
[AIV Block 5/64]   mmRes[64]=151.643768 mmRes[128]=-75.251762 mmRes[192]=-85.956749 mmRes[255]=76.249466
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=1 mx0=47.941551 sm0=1.000000 mx1=5.402357 sm1=-1250.478394
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=1 mx0=47.941551 sm0=1.000000 mx1=5.402357 sm1=-1250.478394
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=1 mx0=47.941551 sm0=1.000000 mx1=5.402357 sm1=-1250.478394
[AIV Block 5/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 5/64] [VEC2-upd] s2L=4 vo0=-7.714844 vo1=7.710938
[AIV Block 5/64] [VEC2-final] s2L=4 sum=1.000000 max=65.988426
[AIV Block 5/64] [VEC2-out] s2L=4 vo0=93583.429688 vo1=-420401.843750
[AIV Block 5/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5911 pLC=23 s2R=220
[AIV Block 5/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 5/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 5/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 5/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=47.941551 mx1=5.402357 mx2=7.963130 mx3=9.597655 sum0=1.000003[AIV Block 5/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=47.941551 iniSum[0]=1.000003 iniMax[1]=5.402357
[AIV Block 5/64]   mmRes[0]=69.411865
[AIV Block 5/64]   mmRes[1]=-109.635292
[AIV Block 5/64]   mmRes[2]=-11.750810
[AIV Block 5/64]   mmRes[3]=-12.796642
[AIV Block 5/64]   mmRes[64]=48.575470 mmRes[128]=308.040955 mmRes[192]=-11.880104 mmRes[255]=181.709167
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=2 mx0=47.941551 sm0=1.000003 mx1=5.402357 sm1=-1257.314209
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=2 mx0=47.941551 sm0=1.000003 mx1=5.402357 sm1=-1257.314209
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=2 mx0=47.941551 sm0=1.000003 mx1=5.402357 sm1=-1257.314209
[AIV Block 5/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1



# 本次修改后
[2026/07/15 09:29:01] [INFO]-tools.py:4338 - Touch file succeed: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_0.bin
[2026/07/15 09:29:01] [INFO]-tools.py:1975 - [data_compare_np]ouput flatten start!
[2026/07/15 09:29:01] [INFO]-tools.py:1978 - [data_compare_np]ouput flatten end!
[2026/07/15 09:29:01] [INFO]-tools.py:2092 - [data_compare_np]overflows_count start!
[2026/07/15 09:29:01] [INFO]-tools.py:2096 - [data_compare_np]overflows_count end!
[2026/07/15 09:29:01] [INFO]-tools.py:2099 - split_count:517104.0; max_diff_hd:10.0;
[2026/07/15 09:29:01] [INFO]-tools.py:1064 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1070 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/15 09:29:01] [INFO]-tools.py:1071 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000001 	 9.8281250 	 9.8281250 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000002 	 6.7929688 	 6.7929688 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000003 	 6.0585938 	 6.0585938 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000004 	 7.5117188 	 7.5117188 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000005 	 0.5166016 	 0.5166016 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000006 	 -9.4453125 	 -9.4453125 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000007 	 0.8339844 	 0.8339844 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000008 	 0.1735840 	 0.1735840 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000009 	 9.3437500 	 9.3437500 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000010 	 -8.4218750 	 -8.4218750 	 0.0000000 	 0.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1080 - ...   	   ...   	   ...   	   ...    	   ...
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517095 	 -1.4199219 	 0.2871094 	 1.7070312 	 1.2022008
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517096 	 -4.5625000 	 -0.2380371 	 4.3244629 	 0.9478275
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517097 	 3.4257812 	 0.0835571 	 3.3422241 	 0.9756093
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517098 	 1.8593750 	 -0.0808716 	 1.9402466 	 1.0434940
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517099 	 6.1445312 	 -0.0323486 	 6.1768799 	 1.0052646
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517100 	 -4.6093750 	 -0.0509644 	 4.5584106 	 0.9889433
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517101 	 -3.7734375 	 -0.0361633 	 3.7372742 	 0.9904163
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517102 	 -1.0830078 	 -0.1429443 	 0.9400635 	 0.8680117
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517103 	 4.4726562 	 -0.0589294 	 4.5315857 	 1.0131755
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00517104 	 0.7348633 	 -0.3032227 	 1.0380859 	 1.4126246
[2026/07/15 09:29:01] [INFO]-tools.py:2261 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:2263 - Rtol   	 Atol   	 PctThd   	 PctRlt   	 Result
[2026/07/15 09:29:01] [INFO]-tools.py:2264 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:2266 - 0.0050    	 0.000025  	 99.50%   	 14.330773%   	 Failed
[2026/07/15 09:29:01] [INFO]-tools.py:2269 - Max-RelativeError is: 2.0. Threshold is: 10.0.
[2026/07/15 09:29:01] [INFO]-tools.py:1128 - Error Line-----------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1130 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/15 09:29:01] [INFO]-tools.py:1131 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000114 	 -0.9384766 	 -4.5273438 	 3.5888672 	 0.7929688
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000115 	 5.7890625 	 -6.4335938 	 12.2226562 	 1.8994141
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000116 	 1.3105469 	 -1.3603516 	 2.6708984 	 1.9638672
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000117 	 -9.1093750 	 -6.4296875 	 2.6796875 	 0.2941895
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000118 	 -9.4921875 	 6.8671875 	 16.3593750 	 1.7236328
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000119 	 -6.5156250 	 0.3254395 	 6.8410645 	 1.0498047
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000120 	 -4.0039062 	 5.1914062 	 9.1953125 	 1.7714844
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000121 	 8.7734375 	 2.1562500 	 6.6171875 	 0.7543945
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000122 	 5.4765625 	 -5.2968750 	 10.7734375 	 1.9667969
[2026/07/15 09:29:01] [INFO]-tools.py:1144 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000262 	 3.0273438 	 3.2167969 	 0.1894531 	 0.0588989
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000263 	 -0.9184570 	 -0.9072266 	 0.0112305 	 0.0122299
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000264 	 -6.5898438 	 -6.7109375 	 0.1210938 	 0.0180511
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000265 	 -7.4648438 	 -7.7187500 	 0.2539062 	 0.0328979
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000266 	 -3.9511719 	 -4.0898438 	 0.1386719 	 0.0339050
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000268 	 -9.1640625 	 -9.2343750 	 0.0703125 	 0.0076141
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000269 	 4.8398438 	 4.9140625 	 0.0742188 	 0.0151062
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000271 	 -7.6835938 	 -7.8242188 	 0.1406250 	 0.0179749
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000272 	 -5.6640625 	 -5.7500000 	 0.0859375 	 0.0149460
[2026/07/15 09:29:01] [INFO]-tools.py:1144 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/15 09:29:01] [INFO]-tools.py:1149 - Max-RE line:---------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1157 - 00001720 	 5.8906250 	 -5.8867188 	 11.7773438 	 2.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1157 - 00002505 	 -2.9023438 	 2.9023438 	 5.8046875 	 2.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1157 - 00002871 	 -8.0781250 	 8.0781250 	 16.1562500 	 2.0000000
[2026/07/15 09:29:01] [INFO]-tools.py:1164 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:4516 - aclnnFusedInferAttentionScoreV5:aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311 End to compare result. Duration:0.14 second.
[2026/07/15 09:29:01] [INFO]-run_aclnn.py:2887 - output-0 compare end 
[2026/07/15 09:29:01] [INFO]-run_aclnn.py:2889 - output-0 result Failed 14.330772919954207 2.0
[2026/07/15 09:29:01] [INFO]-run_aclnn.py:2885 - start compare case the first output-1 
[2026/07/15 09:29:01] [INFO]-tools.py:4142 - Step:------>>>>>> Start to compare result <<<<<<------ 
[2026/07/15 09:29:01] [INFO]-tools.py:4181 - data-compare for output index:1,and its dtype is fp32.
npu out file: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_output_1.bin
load npu output from file: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_output_1.bin with dtype:<class 'numpy.float32'> real dtype:fp32
*******
输出数据类型： fp32 method为: 1
*******
load npu output from file: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_1.bin with dtype:<class 'numpy.float32'> real dtype:fp32
[2026/07/15 09:29:01] [INFO]-tools.py:4338 - Touch file succeed: /home/z00699874/fuzz/aclnn_fuzz_0713/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_1.bin
[2026/07/15 09:29:01] [INFO]-tools.py:1975 - [data_compare_np]ouput flatten start!
[2026/07/15 09:29:01] [INFO]-tools.py:1978 - [data_compare_np]ouput flatten end!
[2026/07/15 09:29:01] [INFO]-tools.py:2092 - [data_compare_np]overflows_count start!
[2026/07/15 09:29:01] [INFO]-tools.py:2096 - [data_compare_np]overflows_count end!
[2026/07/15 09:29:01] [INFO]-tools.py:2099 - split_count:9072.0; max_diff_hd:10.0;
[2026/07/15 09:29:01] [INFO]-tools.py:1064 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1070 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/15 09:29:01] [INFO]-tools.py:1071 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000001 	 56.0644913 	 56.0644836 	 0.0000076 	 0.0000001
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000002 	 53.1180115 	 53.1179962 	 0.0000153 	 0.0000003
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000003 	 62.7973251 	 62.7973137 	 0.0000114 	 0.0000002
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000004 	 41.1084213 	 41.0734177 	 0.0350037 	 0.0008515
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000005 	 59.9560432 	 50.1899338 	 9.7661095 	 0.1628878
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000006 	 52.4218941 	 41.8214493 	 10.6004448 	 0.2022141
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000007 	 57.4209442 	 57.4208870 	 0.0000572 	 0.0000010
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000008 	 54.1091957 	 45.9963493 	 8.1128464 	 0.1499347
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000009 	 41.5967941 	 41.5762901 	 0.0205040 	 0.0004929
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00000010 	 50.0907707 	 45.9351082 	 4.1556625 	 0.0829626
[2026/07/15 09:29:01] [INFO]-tools.py:1080 - ...   	   ...   	   ...   	   ...    	   ...
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009063 	 57.3644829 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009064 	 59.0962677 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009065 	 61.8104630 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009066 	 51.2623787 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009067 	 49.0081215 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009068 	 53.0105782 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009069 	 50.7572136 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009070 	 49.6615028 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009071 	 46.5672798 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:1061 - 00009072 	 54.4735184 	 inf 	 inf 	 inf
[2026/07/15 09:29:01] [INFO]-tools.py:2261 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:2263 - Rtol   	 Atol   	 PctThd   	 PctRlt   	 Result
[2026/07/15 09:29:01] [INFO]-tools.py:2264 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:2266 - 0.0050    	 0.000025  	 99.50%   	 25.694444%   	 Failed
[2026/07/15 09:29:01] [INFO]-tools.py:2269 - Max-RelativeError is: 1.3507811. Threshold is: 10.0.
[2026/07/15 09:29:01] [INFO]-tools.py:1128 - Error Line-----------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1130 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/15 09:29:01] [INFO]-tools.py:1131 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000004 	 59.9560432 	 50.1899338 	 9.7661095 	 0.1628878
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000005 	 52.4218941 	 41.8214493 	 10.6004448 	 0.2022141
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000007 	 54.1091957 	 45.9963493 	 8.1128464 	 0.1499347
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000009 	 50.0907707 	 45.9351082 	 4.1556625 	 0.0829626
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000010 	 57.1765709 	 41.5749054 	 15.6016655 	 0.2728682
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000013 	 47.9561920 	 44.6419296 	 3.3142624 	 0.0691102
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000014 	 54.3032990 	 46.8821945 	 7.4211044 	 0.1366603
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000018 	 50.5890617 	 49.7431068 	 0.8459549 	 0.0167221
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000020 	 49.2755623 	 42.4202309 	 6.8553314 	 0.1391223
[2026/07/15 09:29:01] [INFO]-tools.py:1144 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000154 	 55.2819557 	 48.7214775 	 6.5604782 	 0.1186730
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000155 	 66.4170303 	 43.2946625 	 23.1223679 	 0.3481391
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000156 	 46.0075035 	 43.8212395 	 2.1862640 	 0.0475197
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000157 	 43.5326004 	 41.9827194 	 1.5498810 	 0.0356028
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000158 	 48.1041985 	 42.6361084 	 5.4680901 	 0.1136718
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000159 	 45.3469810 	 42.7560463 	 2.5909348 	 0.0571358
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000160 	 54.9117661 	 53.4250488 	 1.4867172 	 0.0270747
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000163 	 60.3323555 	 47.7355919 	 12.5967636 	 0.2087895
[2026/07/15 09:29:01] [INFO]-tools.py:1138 - 00000167 	 47.1393394 	 43.8864517 	 3.2528877 	 0.0690058
[2026/07/15 09:29:01] [INFO]-tools.py:1144 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/15 09:29:01] [INFO]-tools.py:1149 - Max-RE line:---------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:1157 - 00004940 	 50.0651894 	 -17.5619221 	 67.6271114 	 1.3507811
[2026/07/15 09:29:01] [INFO]-tools.py:1164 - ---------------------------------------------------------------------------------------
[2026/07/15 09:29:01] [INFO]-tools.py:4516 - aclnnFusedInferAttentionScoreV5:aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311 End to compare result. Duration:0.01 second.
[2026/07/15 09:29:01] [INFO]-run_aclnn.py:2887 - output-1 compare end 
[2026/07/15 09:29:01] [INFO]-run_aclnn.py:2889 - output-1 result Failed 25.694444444444443 1.3507810831069946
[2026/07/15 09:29:01] [INFO]-run_single.py:3969 - end execute aclnn case: aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311
[get_each_cs_npu_log]current scene not satisfy,skip
[2026/07/15 09:29:01] [WARNING]-run_single.py:0316 - There not exist kernel file[kernel_name:aclnnFusedInferAttentionScoreV5] in kernel_meta.
[2026/07/15 09:29:03] [/home/z00699874/fuzz/aclnn_fuzz_0713/libs/../golden/aclnn_golden/aclnnFusedInferAttentionScoreV5_golden.csv]-run_single.py:1400 - [WARNING]no golden file:
[2026/07/15 09:29:03] [INFO]-run_single.py:1698 - Case report ->:
[2026/07/15 09:29:03] [INFO]-run_single.py:1699 - aclnnFusedInferAttentionScoreV5,aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311,1784107743,14.330772919954207/25.694444444444443,Failed/Failed,None,0.0,no_bound,,2.0/1.3507811,"[[28, 2, 162, 57], [28, 7168, 162, 57], [28, 7168, 162, 57], [1], [2048, 2048], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1, 7168, 162, 57], [1, 7168, 162, 57], [1], [1], [1], [1], [1], [1], [1]]","['fp16', 'fp16', 'fp16', 'fp16', 'bool', 'uint64', 'fp32', 'uint64', 'fp16', 'fp16', 'fp16', 'fp16', 'int32', 'int64', 'int64', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16']","['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND']","[[-5, 5], [-10, 10], [-10, 10], ['null', 'null'], [-10.0, 10.0], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], [-10, 10], [-10, 10], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null']]","{'actualseqlengths': [], 'actualseqlengthskv': [69, 60, 80, 110, 5376, 43, 81, 3, 98, 107, 18, 92, 3840, 15, 125, 4224, 74, 3328, 75, 100, 29, 56, 21, 89, 74, 65, 3328, 1536], 'prefix_act_lens': [5852], 'qstartidx': [], 'kvstartidx': [], 'numheads': 162, 'scalevalue': 0.1324532357065044, 'pretokens': 946, 'nexttokens': -1, 'inputlayout': 'BSND', 'numkeyvalueheads': 0, 'sparsemode': 4, 'innerprecise': 0, 'blocksize': 0, 'antiquant_mode': 0, 'softmax_lse_flag': True, 'k_antiquant_mode': 0, 'v_antiquant_mode': 0, 'query_quant_mode': 0, 'psetype': 0, 'fused_flag': 1, 'mrandomtype': 'Normal', 'mrandom': 0, 'prandom': 1, 'enablegpu': 'True', 'flaglist': [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]}","[[28, 2, 162, 57], [28, 162, 2, 1]]","['fp16', 'fp32']","None","[]",N/A,"2722721.0000","5966.0000",48001524.0000,"16777216",7546,"","",,,,"",,,,,None,,,Pass,,,,
[2026/07/15 09:29:03] [INFO]-run_single.py:4032 - -------------------------第1用例npu_off执行结束,共1条用例-----------------------------------


[AIV Block 0/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 0/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 0/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=2.289532 mx1=1.835438 mx2=-0.088237 mx3=0.808601 sum0=-340282346638528859811704183484516925440.000000[AIV Block 0/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=2.289532 iniSum[0]=-340282346638528859811704183484516925440.000000 iniMax[1]=1.835438
[AIV Block 0/64]   mmRes[0]=-27.683027
[AIV Block 0/64]   mmRes[1]=167.265762
[AIV Block 0/64]   mmRes[2]=25.492270
[AIV Block 0/64]   mmRes[3]=68.230728
[AIV Block 0/64]   mmRes[64]=57.378597 mmRes[128]=32.709888 mmRes[192]=156.699249 mmRes[255]=-231.890488
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VF256-pre] m=1 oriN=256 preMax[0]=2.289532 preSum[0]=-340282346638528859811704183484516925440.000000 preMax[1]=1.835438
[AIV Block 0/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-27.683027 167.265762 25.492270 68.230728 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/64] [VF256-post] m=1 mx0=2.289532 sum0=-340282346638528859811704183484516925440.000000 srcPost0=-27.683027 srcPost1=167.265762
[AIV Block 0/64] [VEC-Nd-post] s2L=0 mx0=33.456089 sm0=1.071173 mx1=1.835438 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=0 mx0=33.456089 sm0=1.071173 mx1=1.835438 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=0 mx0=33.456089 sm0=1.071173 mx1=1.835438 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 0/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 0/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=33.456089 mx1=1.835438 mx2=-0.088237 mx3=0.808601 sum0=1.071173[AIV Block 0/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=33.456089 iniSum[0]=1.071173 iniMax[1]=1.835438
[AIV Block 0/64]   mmRes[0]=-73.496201
[AIV Block 0/64]   mmRes[1]=175.400269
[AIV Block 0/64]   mmRes[2]=179.696045
[AIV Block 0/64]   mmRes[3]=-10.499971
[AIV Block 0/64]   mmRes[64]=150.721924 mmRes[128]=169.427368 mmRes[192]=-130.357605 mmRes[255]=75.636108
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=1 mx0=33.456089 sm0=1.071173 mx1=1.835438 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=1 mx0=33.456089 sm0=1.071173 mx1=1.835438 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=1 mx0=33.456089 sm0=1.071173 mx1=1.835438 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 0/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 0/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 0/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=56.064064 mx1=1.835438 mx2=-0.088237 mx3=3.719086 sum0=1.000110[AIV Block 0/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=56.064064 iniSum[0]=1.000110 iniMax[1]=1.835438
[AIV Block 0/64]   mmRes[0]=-159.652390
[AIV Block 0/64]   mmRes[1]=-20.926689
[AIV Block 0/64]   mmRes[2]=-148.055710
[AIV Block 0/64]   mmRes[3]=-319.478302
[AIV Block 0/64]   mmRes[64]=136.103928 mmRes[128]=97.351318 mmRes[192]=-87.300552 mmRes[255]=64.942345
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=2 mx0=56.064064 sm0=1.000110 mx1=1.835438 sm1=-inf
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=2 mx0=56.064064 sm0=1.000110 mx1=1.835438 sm1=-inf
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=2 mx0=56.064064 sm0=1.000110 mx1=1.835438 sm1=-inf
[AIV Block 0/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-cpy] s2L=0 vo0=-340282346638528859811704183484516925440.000000 vo1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5920 pLC=23 s2R=68
[AIV Block 0/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 0/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 0/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=56.064064 mx1=1.835438 mx2=-0.088237 mx3=3.719086 sum0=1.000111[AIV Block 0/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=56.064064 iniSum[0]=1.000111 iniMax[1]=1.835438
[AIV Block 0/64]   mmRes[0]=-155.773819
[AIV Block 0/64]   mmRes[1]=-15.460873
[AIV Block 0/64]   mmRes[2]=-64.166557
[AIV Block 0/64]   mmRes[3]=60.987240
[AIV Block 0/64]   mmRes[64]=-278.447418 mmRes[128]=94.691544 mmRes[192]=-22.246895 mmRes[255]=10.018247
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=3 mx0=56.064064 sm0=1.000111 mx1=1.835438 sm1=-inf
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=3 mx0=56.064064 sm0=1.000111 mx1=1.835438 sm1=-inf
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=3 mx0=56.064064 sm0=1.000111 mx1=1.835438 sm1=-inf
[AIV Block 0/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-expUb] s2L=1 e0=0.000000 e1=1.000000 e2=1.000000 e3=0.054449
[AIV Block 0/64] [VEC2-upd] s2L=1 vo0=2.658583 vo1=-6.480917
[AIV Block 0/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=4 s2S=4864 s2R=68 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 0/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=36 computeMode=1 s1Off=0
[AIV Block 0/64] [COPYIN-offset] s2L=4 maskOff=36 preOff=36 computeMode=1 compressMode=3 s2R=68 hS1R=1 maskS2Size=2048
[AIV Block 0/64] [MASK-gm] s2L=4 gmAtDeltaOff=36 val=1 gmAtCorrectOff=36 val=1 s1O=0 s2O=36
[AIV Block 0/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=1 mask[1]=1 reread=1 phyAddr=1ff00
[AIV Block 0/64] [MASK-chk1] s2L=4 mask[0]=1 phyAddr=1ff00
[AIV Block 0/64] [MASK-chk2] s2L=4 mask[0]=1
[AIV Block 0/64] [MASK-chk3] s2L=4 mask[0]=1 expPhy=15700
[AIV Block 0/64] [MASK-chk4] s2L=4 mask[0]=1
[AIV Block 0/64] [MASK-chk5] s2L=4 mask[0]=1 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 0/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=56.064064 mx1=1.835438 mx2=-0.088237 mx3=3.719086 sum0=1.000419[AIV Block 0/64] [VEC-Nd-pre] s2L=4 s2R=68 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=56.064064 iniSum[0]=1.000419 iniMax[1]=1.835438
[AIV Block 0/64]   mmRes[0]=-331.737946
[AIV Block 0/64]   mmRes[1]=-134.870743
[AIV Block 0/64]   mmRes[2]=-134.921310
[AIV Block 0/64]   mmRes[3]=188.724365
[AIV Block 0/64]   mmRes[64]=28.180927 mmRes[128]=12.894497 mmRes[192]=-340282346638528859811704183484516925440.000000 mmRes[255]=8.601824
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=1 mask[1]=1
[AIV Block 0/64] [VEC-Nd-post] s2L=4 mx0=56.064064 sm0=1.000419 mx1=1.835438 sm1=-inf
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=4 mx0=56.064064 sm0=1.000419 mx1=1.835438 sm1=-inf
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=4 mx0=56.064064 sm0=1.000419 mx1=1.835438 sm1=-inf
[AIV Block 0/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 0/64] [LSE-out] lseOff=0 lse0=0.000000 lse1=0.000000
[AIV Block 0/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-expUb] s2L=2 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 0/64] [VEC2-upd] s2L=2 vo0=9.828462 vo1=6.797957
[AIV Block 0/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 0/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 0/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 0/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=-340282346638528859811704183484516925440.000000 mx1=-340282346638528859811704183484516925440.000000 mx2=-340282346638528859811704183484516925440.000000 mx3=-340282346638528859811704183484516925440.000000 sum0=-0.048351[AIV Block 0/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=-340282346638528859811704183484516925440.000000 iniSum[0]=-0.048351 iniMax[1]=-340282346638528859811704183484516925440.000000
[AIV Block 0/64]   mmRes[0]=149.403381
[AIV Block 0/64]   mmRes[1]=21.309612
[AIV Block 0/64]   mmRes[2]=-208.727676
[AIV Block 0/64]   mmRes[3]=-76.581635
[AIV Block 0/64]   mmRes[64]=-93.687088 mmRes[128]=40.739449 mmRes[192]=99.159515 mmRes[255]=38.584782
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VF256-pre] m=1 oriN=256 preMax[0]=-340282346638528859811704183484516925440.000000 preSum[0]=-0.048351 preMax[1]=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=149.403381 21.309612 -208.727676 -76.581635 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/64] [VF256-post] m=1 mx0=-340282346638528859811704183484516925440.000000 sum0=-0.048351 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=21.309612
[AIV Block 0/64] [VEC-Nd-post] s2L=0 mx0=43.001041 sm0=1.031143 mx1=-340282346638528859811704183484516925440.000000 sm1=3.536326
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=0 mx0=43.001041 sm0=1.031143 mx1=-340282346638528859811704183484516925440.000000 sm1=3.536326
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=0 mx0=43.001041 sm0=1.031143 mx1=-340282346638528859811704183484516925440.000000 sm1=3.536326
[AIV Block 0/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-expUb] s2L=3 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 0/64] [VEC2-upd] s2L=3 vo0=9.828464 vo1=6.797960
[AIV Block 0/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 0/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 0/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 0/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=43.001041 mx1=-340282346638528859811704183484516925440.000000 mx2=-340282346638528859811704183484516925440.000000 mx3=-340282346638528859811704183484516925440.000000 sum0=1.031143[AIV Block 0/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=43.001041 iniSum[0]=1.031143 iniMax[1]=-340282346638528859811704183484516925440.000000
[AIV Block 0/64]   mmRes[0]=-136.520264
[AIV Block 0/64]   mmRes[1]=-91.531883
[AIV Block 0/64]   mmRes[2]=-29.148708
[AIV Block 0/64]   mmRes[3]=125.789452
[AIV Block 0/64]   mmRes[64]=130.689529 mmRes[128]=-116.977104 mmRes[192]=140.817429 mmRes[255]=98.181000
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=1 mx0=43.001041 sm0=1.031143 mx1=-340282346638528859811704183484516925440.000000 sm1=3.536326
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=1 mx0=43.001041 sm0=1.031143 mx1=-340282346638528859811704183484516925440.000000 sm1=3.536326
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=1 mx0=43.001041 sm0=1.031143 mx1=-340282346638528859811704183484516925440.000000 sm1=3.536326
[AIV Block 0/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 0/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 0/64] [VEC2-upd] s2L=4 vo0=9.829417 vo1=6.796734
[AIV Block 0/64] [VEC2-final] s2L=4 sum=1.000419 max=56.064064
[AIV Block 0/64] [VEC2-out] s2L=4 vo0=26020.457031 vo1=67212.117188
[AIV Block 0/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 0/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 0/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 0/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 0/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 0/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 0/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 0/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 0/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 0/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 0/64] [VEC-Nd-allmax] mx0=43.001041 mx1=1.835438 mx2=-0.088237 mx3=3.719086 sum0=1.334132[AIV Block 0/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/64] [VEC-Nd-pre-max] iniMax[0]=43.001041 iniSum[0]=1.334132 iniMax[1]=1.835438
[AIV Block 0/64]   mmRes[0]=-104.787178
[AIV Block 0/64]   mmRes[1]=112.562508
[AIV Block 0/64]   mmRes[2]=20.049213
[AIV Block 0/64]   mmRes[3]=-27.285889
[AIV Block 0/64]   mmRes[64]=-10.010809 mmRes[128]=99.811615 mmRes[192]=35.328949 mmRes[255]=86.581589
[AIV Block 0/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/64] [VEC-Nd-post] s2L=2 mx0=43.001041 sm0=1.334132 mx1=1.835438 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC-Nd-preUpd] s2L=2 mx0=43.001041 sm0=1.334132 mx1=1.835438 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC-Nd-postUpd] s2L=2 mx0=43.001041 sm0=1.334132 mx1=1.835438 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 0/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 1/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 1/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 1/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=1.763063 mx1=3.408844 mx2=1.562604 mx3=-1.980426 sum0=0.719606[AIV Block 1/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=1.763063 iniSum[0]=0.719606 iniMax[1]=3.408844
[AIV Block 1/64]   mmRes[0]=-67.043709
[AIV Block 1/64]   mmRes[1]=21.088055
[AIV Block 1/64]   mmRes[2]=106.565117
[AIV Block 1/64]   mmRes[3]=-140.407883
[AIV Block 1/64]   mmRes[64]=133.048660 mmRes[128]=-71.712479 mmRes[192]=80.787483 mmRes[255]=-48.870888
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VF256-pre] m=1 oriN=256 preMax[0]=1.763063 preSum[0]=0.719606 preMax[1]=3.408844
[AIV Block 1/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-67.043709 21.088055 106.565117 -140.407883 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 1/64] [VF256-post] m=1 mx0=1.763063 sum0=0.719606 srcPost0=-67.043709 srcPost1=21.088055
[AIV Block 1/64] [VEC-Nd-post] s2L=0 mx0=37.558483 sm0=1.713387 mx1=3.408844 sm1=-0.137714
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=0 mx0=37.558483 sm0=1.713387 mx1=3.408844 sm1=-0.137714
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=0 mx0=37.558483 sm0=1.713387 mx1=3.408844 sm1=-0.137714
[AIV Block 1/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 1/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 1/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=37.558483 mx1=3.408844 mx2=1.562604 mx3=-1.980426 sum0=1.713387[AIV Block 1/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=37.558483 iniSum[0]=1.713387 iniMax[1]=3.408844
[AIV Block 1/64]   mmRes[0]=-190.409866
[AIV Block 1/64]   mmRes[1]=-104.145378
[AIV Block 1/64]   mmRes[2]=42.020390
[AIV Block 1/64]   mmRes[3]=2.515178
[AIV Block 1/64]   mmRes[64]=36.375416 mmRes[128]=68.614700 mmRes[192]=39.580639 mmRes[255]=144.343018
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=1 mx0=37.558483 sm0=1.713387 mx1=3.408844 sm1=-0.137714
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=1 mx0=37.558483 sm0=1.713387 mx1=3.408844 sm1=-0.137714
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=1 mx0=37.558483 sm0=1.713387 mx1=3.408844 sm1=-0.137714
[AIV Block 1/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 1/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 1/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 1/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=53.117985 mx1=3.408844 mx2=1.562604 mx3=-1.408276 sum0=1.000002[AIV Block 1/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=53.117985 iniSum[0]=1.000002 iniMax[1]=3.408844
[AIV Block 1/64]   mmRes[0]=32.319355
[AIV Block 1/64]   mmRes[1]=-51.193855
[AIV Block 1/64]   mmRes[2]=-74.102753
[AIV Block 1/64]   mmRes[3]=142.209137
[AIV Block 1/64]   mmRes[64]=153.727051 mmRes[128]=-297.491669 mmRes[192]=237.858322 mmRes[255]=144.378601
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=2 mx0=53.117985 sm0=1.000002 mx1=3.408844 sm1=0.862380
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=2 mx0=53.117985 sm0=1.000002 mx1=3.408844 sm1=0.862380
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=2 mx0=53.117985 sm0=1.000002 mx1=3.408844 sm1=0.862380
[AIV Block 1/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-cpy] s2L=0 vo0=-0.470661 vo1=-0.718047
[AIV Block 1/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5920 pLC=23 s2R=68
[AIV Block 1/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 1/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 1/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=53.117985 mx1=3.408844 mx2=1.562604 mx3=-1.408276 sum0=1.000004[AIV Block 1/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=53.117985 iniSum[0]=1.000004 iniMax[1]=3.408844
[AIV Block 1/64]   mmRes[0]=-76.562164
[AIV Block 1/64]   mmRes[1]=-41.593208
[AIV Block 1/64]   mmRes[2]=103.800919
[AIV Block 1/64]   mmRes[3]=37.095795
[AIV Block 1/64]   mmRes[64]=-47.820580 mmRes[128]=41.709175 mmRes[192]=56.869980 mmRes[255]=19.118700
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=3 mx0=53.117985 sm0=1.000004 mx1=3.408844 sm1=1.862474
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=3 mx0=53.117985 sm0=1.000004 mx1=3.408844 sm1=1.862474
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=3 mx0=53.117985 sm0=1.000004 mx1=3.408844 sm1=1.862474
[AIV Block 1/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-expUb] s2L=1 e0=0.000000 e1=1.000000 e2=1.000000 e3=0.564311
[AIV Block 1/64] [VEC2-upd] s2L=1 vo0=-6.072203 vo1=-8.415761
[AIV Block 1/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=4 s2S=4864 s2R=68 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 1/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=2084 computeMode=1 s1Off=0
[AIV Block 1/64] [COPYIN-offset] s2L=4 maskOff=2084 preOff=2084 computeMode=1 compressMode=3 s2R=68 hS1R=1 maskS2Size=2048
[AIV Block 1/64] [MASK-gm] s2L=4 gmAtDeltaOff=2084 val=1 gmAtCorrectOff=2084 val=1 s1O=1 s2O=36
[AIV Block 1/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=1 mask[1]=1 reread=1 phyAddr=1ff00
[AIV Block 1/64] [MASK-chk1] s2L=4 mask[0]=1 phyAddr=1ff00
[AIV Block 1/64] [MASK-chk2] s2L=4 mask[0]=1
[AIV Block 1/64] [MASK-chk3] s2L=4 mask[0]=1 expPhy=15700
[AIV Block 1/64] [MASK-chk4] s2L=4 mask[0]=1
[AIV Block 1/64] [MASK-chk5] s2L=4 mask[0]=1 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 1/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=53.117985 mx1=3.408844 mx2=1.562604 mx3=-1.408276 sum0=1.000012[AIV Block 1/64] [VEC-Nd-pre] s2L=4 s2R=68 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=53.117985 iniSum[0]=1.000012 iniMax[1]=3.408844
[AIV Block 1/64]   mmRes[0]=116.672691
[AIV Block 1/64]   mmRes[1]=303.519928
[AIV Block 1/64]   mmRes[2]=-7.666192
[AIV Block 1/64]   mmRes[3]=43.425591
[AIV Block 1/64]   mmRes[64]=176.457581 mmRes[128]=-340282346638528859811704183484516925440.000000 mmRes[192]=31.505104 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=1 mask[1]=1
[AIV Block 1/64] [VEC-Nd-post] s2L=4 mx0=53.117985 sm0=1.000012 mx1=3.408844 sm1=2.862568
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=4 mx0=53.117985 sm0=1.000012 mx1=3.408844 sm1=2.862568
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=4 mx0=53.117985 sm0=1.000012 mx1=3.408844 sm1=2.862568
[AIV Block 1/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 1/64] [LSE-out] lseOff=1 lse0=0.000000 lse1=0.000000
[AIV Block 1/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-expUb] s2L=2 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 1/64] [VEC2-upd] s2L=2 vo0=8.617190 vo1=4.753902
[AIV Block 1/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 1/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 1/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 1/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=2.404803 mx1=0.452005 mx2=2.596038 mx3=-1.125673 sum0=0.792412[AIV Block 1/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=2.404803 iniSum[0]=0.792412 iniMax[1]=0.452005
[AIV Block 1/64]   mmRes[0]=-201.852905
[AIV Block 1/64]   mmRes[1]=9.176884
[AIV Block 1/64]   mmRes[2]=120.894989
[AIV Block 1/64]   mmRes[3]=-84.604309
[AIV Block 1/64]   mmRes[64]=64.178848 mmRes[128]=-185.633438 mmRes[192]=159.795166 mmRes[255]=-3.200465
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VF256-pre] m=1 oriN=256 preMax[0]=2.404803 preSum[0]=0.792412 preMax[1]=0.452005
[AIV Block 1/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-201.852905 9.176884 120.894989 -84.604309 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 1/64] [VF256-post] m=1 mx0=2.404803 sum0=0.792412 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=9.176884
[AIV Block 1/64] [VEC-Nd-post] s2L=0 mx0=34.334976 sm0=1.075562 mx1=0.452005 sm1=-1.043822
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=0 mx0=34.334976 sm0=1.075562 mx1=0.452005 sm1=-1.043822
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=0 mx0=34.334976 sm0=1.075562 mx1=0.452005 sm1=-1.043822
[AIV Block 1/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-expUb] s2L=3 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 1/64] [VEC2-upd] s2L=3 vo0=8.617202 vo1=4.753895
[AIV Block 1/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 1/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 1/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 1/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 1/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=34.334976 mx1=0.452005 mx2=2.596038 mx3=-1.125673 sum0=1.075562[AIV Block 1/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=34.334976 iniSum[0]=1.075562 iniMax[1]=0.452005
[AIV Block 1/64]   mmRes[0]=-149.177658
[AIV Block 1/64]   mmRes[1]=43.244434
[AIV Block 1/64]   mmRes[2]=-41.155014
[AIV Block 1/64]   mmRes[3]=-66.937706
[AIV Block 1/64]   mmRes[64]=-42.070473 mmRes[128]=75.553139 mmRes[192]=-118.742989 mmRes[255]=153.161346
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=1 mx0=34.334976 sm0=1.075562 mx1=0.452005 sm1=-1.043822
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=1 mx0=34.334976 sm0=1.075562 mx1=0.452005 sm1=-1.043822
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=1 mx0=34.334976 sm0=1.075562 mx1=0.452005 sm1=-1.043822
[AIV Block 1/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 1/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 1/64] [VEC2-upd] s2L=4 vo0=8.617202 vo1=4.753891
[AIV Block 1/64] [VEC2-final] s2L=4 sum=1.000012 max=53.117985
[AIV Block 1/64] [VEC2-out] s2L=4 vo0=1546.259644 vo1=-4344.839355
[AIV Block 1/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 1/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 1/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 1/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 1/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 1/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 1/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 1/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 1/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 1/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 1/64] [VEC-Nd-allmax] mx0=34.334976 mx1=3.408844 mx2=2.596038 mx3=-1.125673 sum0=1.078904[AIV Block 1/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 1/64] [VEC-Nd-pre-max] iniMax[0]=34.334976 iniSum[0]=1.078904 iniMax[1]=3.408844
[AIV Block 1/64]   mmRes[0]=45.527115
[AIV Block 1/64]   mmRes[1]=-2.512713
[AIV Block 1/64]   mmRes[2]=-167.665756
[AIV Block 1/64]   mmRes[3]=113.721420
[AIV Block 1/64]   mmRes[64]=64.157967 mmRes[128]=-37.721077 mmRes[192]=-7.322091 mmRes[255]=-273.659332
[AIV Block 1/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 1/64] [VEC-Nd-post] s2L=2 mx0=34.334976 sm0=1.078904 mx1=3.408844 sm1=0.945833
[AIV Block 1/64] [VEC-Nd-preUpd] s2L=2 mx0=34.334976 sm0=1.078904 mx1=3.408844 sm1=0.945833
[AIV Block 1/64] [VEC-Nd-postUpd] s2L=2 mx0=34.334976 sm0=1.078904 mx1=3.408844 sm1=0.945833
[AIV Block 1/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 2/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 2/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 2/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=1.486535 mx1=0.314579 mx2=5.804964 mx3=1.135771 sum0=-340282346638528859811704183484516925440.000000[AIV Block 2/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=1.486535 iniSum[0]=-340282346638528859811704183484516925440.000000 iniMax[1]=0.314579
[AIV Block 2/64]   mmRes[0]=-125.169495
[AIV Block 2/64]   mmRes[1]=-64.456192
[AIV Block 2/64]   mmRes[2]=36.473480
[AIV Block 2/64]   mmRes[3]=-77.795570
[AIV Block 2/64]   mmRes[64]=-150.744598 mmRes[128]=55.533291 mmRes[192]=-77.896194 mmRes[255]=227.690186
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VF256-pre] m=1 oriN=256 preMax[0]=1.486535 preSum[0]=-340282346638528859811704183484516925440.000000 preMax[1]=0.314579
[AIV Block 2/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-125.169495 -64.456192 36.473480 -77.795570 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 2/64] [VF256-post] m=1 mx0=1.486535 sum0=-340282346638528859811704183484516925440.000000 srcPost0=-125.169495 srcPost1=-64.456192
[AIV Block 2/64] [VEC-Nd-post] s2L=0 mx0=40.175499 sm0=1.078922 mx1=0.314579 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=0 mx0=40.175499 sm0=1.078922 mx1=0.314579 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=0 mx0=40.175499 sm0=1.078922 mx1=0.314579 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 2/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 2/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=40.175499 mx1=0.314579 mx2=5.804964 mx3=1.135771 sum0=1.078922[AIV Block 2/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=40.175499 iniSum[0]=1.078922 iniMax[1]=0.314579
[AIV Block 2/64]   mmRes[0]=-16.704733
[AIV Block 2/64]   mmRes[1]=228.097992
[AIV Block 2/64]   mmRes[2]=-132.405487
[AIV Block 2/64]   mmRes[3]=144.993408
[AIV Block 2/64]   mmRes[64]=39.426411 mmRes[128]=98.555397 mmRes[192]=57.149570 mmRes[255]=-69.360847
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=1 mx0=40.175499 sm0=1.078922 mx1=0.314579 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=1 mx0=40.175499 sm0=1.078922 mx1=0.314579 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=1 mx0=40.175499 sm0=1.078922 mx1=0.314579 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 2/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 2/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 2/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=53.956337 mx1=0.314579 mx2=5.804964 mx3=1.135771 sum0=1.000140[AIV Block 2/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=53.956337 iniSum[0]=1.000140 iniMax[1]=0.314579
[AIV Block 2/64]   mmRes[0]=77.366837
[AIV Block 2/64]   mmRes[1]=101.332100
[AIV Block 2/64]   mmRes[2]=-80.005608
[AIV Block 2/64]   mmRes[3]=9.095348
[AIV Block 2/64]   mmRes[64]=-7.408249 mmRes[128]=-80.578339 mmRes[192]=15.068103 mmRes[255]=-203.842422
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=2 mx0=53.956337 sm0=1.000140 mx1=0.314579 sm1=-inf
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=2 mx0=53.956337 sm0=1.000140 mx1=0.314579 sm1=-inf
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=2 mx0=53.956337 sm0=1.000140 mx1=0.314579 sm1=-inf
[AIV Block 2/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-cpy] s2L=0 vo0=-340282346638528859811704183484516925440.000000 vo1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5920 pLC=23 s2R=68
[AIV Block 2/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 2/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 2/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=53.956337 mx1=0.314579 mx2=5.804964 mx3=1.135771 sum0=1.000141[AIV Block 2/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=53.956337 iniSum[0]=1.000141 iniMax[1]=0.314579
[AIV Block 2/64]   mmRes[0]=-133.754807
[AIV Block 2/64]   mmRes[1]=0.850524
[AIV Block 2/64]   mmRes[2]=-92.974884
[AIV Block 2/64]   mmRes[3]=-77.640137
[AIV Block 2/64]   mmRes[64]=-115.059944 mmRes[128]=-105.925255 mmRes[192]=188.816910 mmRes[255]=-9.187069
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=3 mx0=53.956337 sm0=1.000141 mx1=0.314579 sm1=-inf
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=3 mx0=53.956337 sm0=1.000141 mx1=0.314579 sm1=-inf
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=3 mx0=53.956337 sm0=1.000141 mx1=0.314579 sm1=-inf
[AIV Block 2/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-expUb] s2L=1 e0=0.000001 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 2/64] [VEC2-upd] s2L=1 vo0=3.570833 vo1=0.450502
[AIV Block 2/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=4 s2S=4864 s2R=68 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 2/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=36 computeMode=1 s1Off=0
[AIV Block 2/64] [COPYIN-offset] s2L=4 maskOff=36 preOff=36 computeMode=1 compressMode=3 s2R=68 hS1R=1 maskS2Size=2048
[AIV Block 2/64] [MASK-gm] s2L=4 gmAtDeltaOff=36 val=1 gmAtCorrectOff=36 val=1 s1O=0 s2O=36
[AIV Block 2/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=1 mask[1]=1 reread=1 phyAddr=1ff00
[AIV Block 2/64] [MASK-chk1] s2L=4 mask[0]=1 phyAddr=1ff00
[AIV Block 2/64] [MASK-chk2] s2L=4 mask[0]=1
[AIV Block 2/64] [MASK-chk3] s2L=4 mask[0]=1 expPhy=15700
[AIV Block 2/64] [MASK-chk4] s2L=4 mask[0]=1
[AIV Block 2/64] [MASK-chk5] s2L=4 mask[0]=1 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 2/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=53.956337 mx1=0.314579 mx2=5.804964 mx3=1.135771 sum0=1.071658[AIV Block 2/64] [VEC-Nd-pre] s2L=4 s2R=68 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=53.956337 iniSum[0]=1.071658 iniMax[1]=0.314579
[AIV Block 2/64]   mmRes[0]=-55.296902
[AIV Block 2/64]   mmRes[1]=190.675949
[AIV Block 2/64]   mmRes[2]=80.816757
[AIV Block 2/64]   mmRes[3]=15.105571
[AIV Block 2/64]   mmRes[64]=-160.400131 mmRes[128]=-340282346638528859811704183484516925440.000000 mmRes[192]=-340282346638528859811704183484516925440.000000 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=1 mask[1]=1
[AIV Block 2/64] [VEC-Nd-post] s2L=4 mx0=53.956337 sm0=1.071658 mx1=0.314579 sm1=-inf
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=4 mx0=53.956337 sm0=1.071658 mx1=0.314579 sm1=-inf
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=4 mx0=53.956337 sm0=1.071658 mx1=0.314579 sm1=-inf
[AIV Block 2/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 2/64] [LSE-out] lseOff=274 lse0=0.000000 lse1=0.000000
[AIV Block 2/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-expUb] s2L=2 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 2/64] [VEC2-upd] s2L=2 vo0=-8.421025 vo1=5.334671
[AIV Block 2/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 2/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 2/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 2/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=-340282346638528859811704183484516925440.000000 mx1=-340282346638528859811704183484516925440.000000 mx2=-340282346638528859811704183484516925440.000000 mx3=-340282346638528859811704183484516925440.000000 sum0=0.626475[AIV Block 2/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=-340282346638528859811704183484516925440.000000 iniSum[0]=0.626475 iniMax[1]=-340282346638528859811704183484516925440.000000
[AIV Block 2/64]   mmRes[0]=-157.814499
[AIV Block 2/64]   mmRes[1]=-235.191803
[AIV Block 2/64]   mmRes[2]=-128.693069
[AIV Block 2/64]   mmRes[3]=-176.261032
[AIV Block 2/64]   mmRes[64]=-60.175579 mmRes[128]=-65.413895 mmRes[192]=-55.113869 mmRes[255]=128.714905
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VF256-pre] m=1 oriN=256 preMax[0]=-340282346638528859811704183484516925440.000000 preSum[0]=0.626475 preMax[1]=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-157.814499 -235.191803 -128.693069 -176.261032 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 2/64] [VF256-post] m=1 mx0=-340282346638528859811704183484516925440.000000 sum0=0.626475 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=-235.191803
[AIV Block 2/64] [VEC-Nd-post] s2L=0 mx0=43.987648 sm0=1.010485 mx1=-340282346638528859811704183484516925440.000000 sm1=-1.995715
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=0 mx0=43.987648 sm0=1.010485 mx1=-340282346638528859811704183484516925440.000000 sm1=-1.995715
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=0 mx0=43.987648 sm0=1.010485 mx1=-340282346638528859811704183484516925440.000000 sm1=-1.995715
[AIV Block 2/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-expUb] s2L=3 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 2/64] [VEC2-upd] s2L=3 vo0=-8.421026 vo1=5.334672
[AIV Block 2/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 2/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 2/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 2/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 2/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=43.987648 mx1=-340282346638528859811704183484516925440.000000 mx2=-340282346638528859811704183484516925440.000000 mx3=-340282346638528859811704183484516925440.000000 sum0=1.010485[AIV Block 2/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=43.987648 iniSum[0]=1.010485 iniMax[1]=-340282346638528859811704183484516925440.000000
[AIV Block 2/64]   mmRes[0]=-163.482773
[AIV Block 2/64]   mmRes[1]=-2.749908
[AIV Block 2/64]   mmRes[2]=39.349976
[AIV Block 2/64]   mmRes[3]=118.493744
[AIV Block 2/64]   mmRes[64]=79.225220 mmRes[128]=-6.295664 mmRes[192]=-116.942017 mmRes[255]=180.853607
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=1 mx0=43.987648 sm0=1.010485 mx1=-340282346638528859811704183484516925440.000000 sm1=-1.995715
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=1 mx0=43.987648 sm0=1.010485 mx1=-340282346638528859811704183484516925440.000000 sm1=-1.995715
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=1 mx0=43.987648 sm0=1.010485 mx1=-340282346638528859811704183484516925440.000000 sm1=-1.995715
[AIV Block 2/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 2/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 2/64] [VEC2-upd] s2L=4 vo0=-8.776796 vo1=5.286097
[AIV Block 2/64] [VEC2-final] s2L=4 sum=1.071658 max=53.956337
[AIV Block 2/64] [VEC2-out] s2L=4 vo0=1918.252930 vo1=71560.695312
[AIV Block 2/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 2/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 2/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 2/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 2/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 2/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 2/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 2/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 2/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 2/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 2/64] [VEC-Nd-allmax] mx0=43.987648 mx1=0.314579 mx2=5.804964 mx3=1.135771 sum0=1.011251[AIV Block 2/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 2/64] [VEC-Nd-pre-max] iniMax[0]=43.987648 iniSum[0]=1.011251 iniMax[1]=0.314579
[AIV Block 2/64]   mmRes[0]=-14.359776
[AIV Block 2/64]   mmRes[1]=-47.104996
[AIV Block 2/64]   mmRes[2]=49.874458
[AIV Block 2/64]   mmRes[3]=-49.418121
[AIV Block 2/64]   mmRes[64]=-75.732895 mmRes[128]=180.980057 mmRes[192]=-36.980381 mmRes[255]=-294.234802
[AIV Block 2/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 2/64] [VEC-Nd-post] s2L=2 mx0=43.987648 sm0=1.011251 mx1=0.314579 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC-Nd-preUpd] s2L=2 mx0=43.987648 sm0=1.011251 mx1=0.314579 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC-Nd-postUpd] s2L=2 mx0=43.987648 sm0=1.011251 mx1=0.314579 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 2/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 3/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 3/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 3/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=-1.723313 mx1=1.123365 mx2=-2.106705 mx3=-1.557590 sum0=-0.708701[AIV Block 3/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=-1.723313 iniSum[0]=-0.708701 iniMax[1]=1.123365
[AIV Block 3/64]   mmRes[0]=34.025860
[AIV Block 3/64]   mmRes[1]=44.390648
[AIV Block 3/64]   mmRes[2]=-206.972824
[AIV Block 3/64]   mmRes[3]=132.023972
[AIV Block 3/64]   mmRes[64]=74.749847 mmRes[128]=66.008720 mmRes[192]=30.640432 mmRes[255]=-44.640411
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VF256-pre] m=1 oriN=256 preMax[0]=-1.723313 preSum[0]=-0.708701 preMax[1]=1.123365
[AIV Block 3/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=34.025860 44.390648 -206.972824 132.023972 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 3/64] [VF256-post] m=1 mx0=-1.723313 sum0=-0.708701 srcPost0=34.025860 srcPost1=44.390648
[AIV Block 3/64] [VEC-Nd-post] s2L=0 mx0=48.183411 sm0=1.000342 mx1=1.123365 sm1=-1.898126
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=0 mx0=48.183411 sm0=1.000342 mx1=1.123365 sm1=-1.898126
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=0 mx0=48.183411 sm0=1.000342 mx1=1.123365 sm1=-1.898126
[AIV Block 3/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 3/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 3/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=48.183411 mx1=1.123365 mx2=-2.106705 mx3=-1.557590 sum0=1.000342[AIV Block 3/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=48.183411 iniSum[0]=1.000342 iniMax[1]=1.123365
[AIV Block 3/64]   mmRes[0]=264.953857
[AIV Block 3/64]   mmRes[1]=-163.873779
[AIV Block 3/64]   mmRes[2]=-42.622395
[AIV Block 3/64]   mmRes[3]=204.392273
[AIV Block 3/64]   mmRes[64]=74.333931 mmRes[128]=91.743149 mmRes[192]=-94.705299 mmRes[255]=-23.242102
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=1 mx0=48.183411 sm0=1.000342 mx1=1.123365 sm1=-1.898126
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=1 mx0=48.183411 sm0=1.000342 mx1=1.123365 sm1=-1.898126
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=1 mx0=48.183411 sm0=1.000342 mx1=1.123365 sm1=-1.898126
[AIV Block 3/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 3/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 3/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 3/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=58.224110 mx1=1.123365 mx2=0.701845 mx3=-0.420678 sum0=1.000338[AIV Block 3/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=58.224110 iniSum[0]=1.000338 iniMax[1]=1.123365
[AIV Block 3/64]   mmRes[0]=165.002991
[AIV Block 3/64]   mmRes[1]=-114.294037
[AIV Block 3/64]   mmRes[2]=122.479958
[AIV Block 3/64]   mmRes[3]=-270.235413
[AIV Block 3/64]   mmRes[64]=92.062225 mmRes[128]=106.293793 mmRes[192]=-229.722534 mmRes[255]=-18.126114
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=2 mx0=58.224110 sm0=1.000338 mx1=1.123365 sm1=-3.696215
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=2 mx0=58.224110 sm0=1.000338 mx1=1.123365 sm1=-3.696215
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=2 mx0=58.224110 sm0=1.000338 mx1=1.123365 sm1=-3.696215
[AIV Block 3/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-cpy] s2L=0 vo0=0.296502 vo1=0.671022
[AIV Block 3/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5920 pLC=23 s2R=68
[AIV Block 3/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 3/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 3/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=58.224110 mx1=1.123365 mx2=0.701845 mx3=-0.420678 sum0=1.000338[AIV Block 3/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=58.224110 iniSum[0]=1.000338 iniMax[1]=1.123365
[AIV Block 3/64]   mmRes[0]=-81.187454
[AIV Block 3/64]   mmRes[1]=-71.472954
[AIV Block 3/64]   mmRes[2]=-142.595901
[AIV Block 3/64]   mmRes[3]=-96.165047
[AIV Block 3/64]   mmRes[64]=-172.149200 mmRes[128]=-84.963181 mmRes[192]=117.730278 mmRes[255]=-3.078491
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=3 mx0=58.224110 sm0=1.000338 mx1=1.123365 sm1=-5.494305
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=3 mx0=58.224110 sm0=1.000338 mx1=1.123365 sm1=-5.494305
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=3 mx0=58.224110 sm0=1.000338 mx1=1.123365 sm1=-5.494305
[AIV Block 3/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-expUb] s2L=1 e0=0.000044 e1=1.000000 e2=0.060292 e3=0.320808
[AIV Block 3/64] [VEC2-upd] s2L=1 vo0=3.701434 vo1=-1.453264
[AIV Block 3/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=4 s2S=4864 s2R=68 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 3/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=2084 computeMode=1 s1Off=0
[AIV Block 3/64] [COPYIN-offset] s2L=4 maskOff=2084 preOff=2084 computeMode=1 compressMode=3 s2R=68 hS1R=1 maskS2Size=2048
[AIV Block 3/64] [MASK-gm] s2L=4 gmAtDeltaOff=2084 val=1 gmAtCorrectOff=2084 val=1 s1O=1 s2O=36
[AIV Block 3/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=1 mask[1]=1 reread=1 phyAddr=1ff00
[AIV Block 3/64] [MASK-chk1] s2L=4 mask[0]=1 phyAddr=1ff00
[AIV Block 3/64] [MASK-chk2] s2L=4 mask[0]=1
[AIV Block 3/64] [MASK-chk3] s2L=4 mask[0]=1 expPhy=15700
[AIV Block 3/64] [MASK-chk4] s2L=4 mask[0]=1
[AIV Block 3/64] [MASK-chk5] s2L=4 mask[0]=1 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 3/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=58.224110 mx1=1.123365 mx2=0.701845 mx3=-0.420678 sum0=1.000411[AIV Block 3/64] [VEC-Nd-pre] s2L=4 s2R=68 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=58.224110 iniSum[0]=1.000411 iniMax[1]=1.123365
[AIV Block 3/64]   mmRes[0]=-66.643448
[AIV Block 3/64]   mmRes[1]=-62.148781
[AIV Block 3/64]   mmRes[2]=187.988342
[AIV Block 3/64]   mmRes[3]=276.241241
[AIV Block 3/64]   mmRes[64]=51.241177 mmRes[128]=14.078957 mmRes[192]=-340282346638528859811704183484516925440.000000 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=1 mask[1]=1
[AIV Block 3/64] [VEC-Nd-post] s2L=4 mx0=58.224110 sm0=1.000411 mx1=1.123365 sm1=-7.292394
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=4 mx0=58.224110 sm0=1.000411 mx1=1.123365 sm1=-7.292394
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=4 mx0=58.224110 sm0=1.000411 mx1=1.123365 sm1=-7.292394
[AIV Block 3/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 3/64] [LSE-out] lseOff=275 lse0=0.000000 lse1=0.000000
[AIV Block 3/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-expUb] s2L=2 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 3/64] [VEC2-upd] s2L=2 vo0=-3.117832 vo1=7.118527
[AIV Block 3/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 3/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 3/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 3/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=-0.694094 mx1=1.133224 mx2=1.581783 mx3=-1.721820 sum0=0.389628[AIV Block 3/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=-0.694094 iniSum[0]=0.389628 iniMax[1]=1.133224
[AIV Block 3/64]   mmRes[0]=163.337601
[AIV Block 3/64]   mmRes[1]=227.999908
[AIV Block 3/64]   mmRes[2]=-38.281574
[AIV Block 3/64]   mmRes[3]=-83.831711
[AIV Block 3/64]   mmRes[64]=78.404800 mmRes[128]=-49.674755 mmRes[192]=127.848785 mmRes[255]=106.629509
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VF256-pre] m=1 oriN=256 preMax[0]=-0.694094 preSum[0]=0.389628 preMax[1]=1.133224
[AIV Block 3/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=163.337601 227.999908 -38.281574 -83.831711 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 3/64] [VF256-post] m=1 mx0=-0.694094 sum0=0.389628 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=227.999908
[AIV Block 3/64] [VEC-Nd-post] s2L=0 mx0=47.751923 sm0=1.005201 mx1=1.133224 sm1=0.788697
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=0 mx0=47.751923 sm0=1.005201 mx1=1.133224 sm1=0.788697
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=0 mx0=47.751923 sm0=1.005201 mx1=1.133224 sm1=0.788697
[AIV Block 3/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-expUb] s2L=3 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 3/64] [VEC2-upd] s2L=3 vo0=-3.117832 vo1=7.118527
[AIV Block 3/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 3/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 3/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 3/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 3/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=47.751923 mx1=1.133224 mx2=1.581783 mx3=-1.721820 sum0=1.005201[AIV Block 3/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=47.751923 iniSum[0]=1.005201 iniMax[1]=1.133224
[AIV Block 3/64]   mmRes[0]=210.167313
[AIV Block 3/64]   mmRes[1]=103.239075
[AIV Block 3/64]   mmRes[2]=193.046234
[AIV Block 3/64]   mmRes[3]=-47.906284
[AIV Block 3/64]   mmRes[64]=22.739819 mmRes[128]=115.317703 mmRes[192]=291.129059 mmRes[255]=-113.435341
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=1 mx0=47.751923 sm0=1.005201 mx1=1.133224 sm1=0.788697
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=1 mx0=47.751923 sm0=1.005201 mx1=1.133224 sm1=0.788697
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=1 mx0=47.751923 sm0=1.005201 mx1=1.133224 sm1=0.788697
[AIV Block 3/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 3/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 3/64] [VEC2-upd] s2L=4 vo0=-3.117318 vo1=7.118335
[AIV Block 3/64] [VEC2-final] s2L=4 sum=1.000411 max=58.224110
[AIV Block 3/64] [VEC2-out] s2L=4 vo0=40642.230469 vo1=-11248.502930
[AIV Block 3/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5920 pLC=23 s2R=220
[AIV Block 3/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[AIV Block 3/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 3/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 3/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 3/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 3/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 3/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 3/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 3/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[AIV Block 3/64] [VEC-Nd-allmax] mx0=48.985416 mx1=1.133224 mx2=1.581783 mx3=-0.420678 sum0=1.292788[AIV Block 3/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 3/64] [VEC-Nd-pre-max] iniMax[0]=48.985416 iniSum[0]=1.292788 iniMax[1]=1.133224
[AIV Block 3/64]   mmRes[0]=429.602295
[AIV Block 3/64]   mmRes[1]=150.630005
[AIV Block 3/64]   mmRes[2]=23.331858
[AIV Block 3/64]   mmRes[3]=175.337067
[AIV Block 3/64]   mmRes[64]=115.814621 mmRes[128]=259.989105 mmRes[192]=-87.225746 mmRes[255]=230.818848
[AIV Block 3/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 3/64] [VEC-Nd-post] s2L=2 mx0=48.985416 sm0=1.292788 mx1=1.133224 sm1=-1.009392
[AIV Block 3/64] [VEC-Nd-preUpd] s2L=2 mx0=48.985416 sm0=1.292788 mx1=1.133224 sm1=-1.009392
[AIV Block 3/64] [VEC-Nd-postUpd] s2L=2 mx0=48.985416 sm0=1.292788 mx1=1.133224 sm1=-1.009392
[AIV Block 3/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 4/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 4/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 4/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=-0.456876 mx1=-1.946375 mx2=1.271606 mx3=0.146540 sum0=-340282346638528859811704183484516925440.000000[AIV Block 4/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=-0.456876 iniSum[0]=-340282346638528859811704183484516925440.000000 iniMax[1]=-1.946375
[AIV Block 4/64]   mmRes[0]=-93.664589
[AIV Block 4/64]   mmRes[1]=27.924973
[AIV Block 4/64]   mmRes[2]=48.886803
[AIV Block 4/64]   mmRes[3]=181.801682
[AIV Block 4/64]   mmRes[64]=96.948227 mmRes[128]=-118.057198 mmRes[192]=-224.574966 mmRes[255]=-174.877838
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VF256-pre] m=1 oriN=256 preMax[0]=-0.456876 preSum[0]=-340282346638528859811704183484516925440.000000 preMax[1]=-1.946375
[AIV Block 4/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-93.664589 27.924973 48.886803 181.801682 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 4/64] [VF256-post] m=1 mx0=-0.456876 sum0=-340282346638528859811704183484516925440.000000 srcPost0=-93.664589 srcPost1=27.924973
[AIV Block 4/64] [VEC-Nd-post] s2L=0 mx0=41.391850 sm0=1.112852 mx1=-1.946375 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=0 mx0=41.391850 sm0=1.112852 mx1=-1.946375 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=0 mx0=41.391850 sm0=1.112852 mx1=-1.946375 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 4/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 4/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=41.391850 mx1=-1.946375 mx2=1.271606 mx3=0.146540 sum0=1.112852[AIV Block 4/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=41.391850 iniSum[0]=1.112852 iniMax[1]=-1.946375
[AIV Block 4/64]   mmRes[0]=20.470823
[AIV Block 4/64]   mmRes[1]=-165.920227
[AIV Block 4/64]   mmRes[2]=-65.618553
[AIV Block 4/64]   mmRes[3]=21.730583
[AIV Block 4/64]   mmRes[64]=-116.933510 mmRes[128]=99.366310 mmRes[192]=-56.323349 mmRes[255]=-125.929047
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=1 mx0=41.391850 sm0=1.112852 mx1=-1.946375 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=1 mx0=41.391850 sm0=1.112852 mx1=-1.946375 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=1 mx0=41.391850 sm0=1.112852 mx1=-1.946375 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5911 pLC=23 s2R=220
[AIV Block 4/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 4/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 4/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=41.391850 mx1=0.118168 mx2=1.271606 mx3=1.790428 sum0=1.190735[AIV Block 4/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=41.391850 iniSum[0]=1.190735 iniMax[1]=0.118168
[AIV Block 4/64]   mmRes[0]=48.862591
[AIV Block 4/64]   mmRes[1]=-52.251770
[AIV Block 4/64]   mmRes[2]=77.343079
[AIV Block 4/64]   mmRes[3]=124.626068
[AIV Block 4/64]   mmRes[64]=69.574768 mmRes[128]=-18.237240 mmRes[192]=62.597969 mmRes[255]=-203.987793
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=2 mx0=41.391850 sm0=1.190735 mx1=0.118168 sm1=-inf
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=2 mx0=41.391850 sm0=1.190735 mx1=0.118168 sm1=-inf
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=2 mx0=41.391850 sm0=1.190735 mx1=0.118168 sm1=-inf
[AIV Block 4/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-cpy] s2L=0 vo0=-340282346638528859811704183484516925440.000000 vo1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5911 pLC=23 s2R=59
[AIV Block 4/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 4/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 4/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=41.391850 mx1=0.118168 mx2=1.271606 mx3=1.790428 sum0=1.241790[AIV Block 4/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=41.391850 iniSum[0]=1.241790 iniMax[1]=0.118168
[AIV Block 4/64]   mmRes[0]=33.649502
[AIV Block 4/64]   mmRes[1]=19.118299
[AIV Block 4/64]   mmRes[2]=-129.651047
[AIV Block 4/64]   mmRes[3]=-107.886353
[AIV Block 4/64]   mmRes[64]=-125.340744 mmRes[128]=-8.734537 mmRes[192]=36.026272 mmRes[255]=-16.679708
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=3 mx0=41.391850 sm0=1.241790 mx1=0.118168 sm1=-inf
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=3 mx0=41.391850 sm0=1.241790 mx1=0.118168 sm1=-inf
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=3 mx0=41.391850 sm0=1.241790 mx1=0.118168 sm1=-inf
[AIV Block 4/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-expUb] s2L=1 e0=1.000000 e1=0.126876 e2=1.000000 e3=0.193227
[AIV Block 4/64] [VEC2-upd] s2L=1 vo0=-4.866683 vo1=-7.689264
[AIV Block 4/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=4 s2S=4864 s2R=59 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 4/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5794 deltaC=21 deltaPre=-925 maskOff=36 computeMode=1 s1Off=0
[AIV Block 4/64] [COPYIN-offset] s2L=4 maskOff=36 preOff=36 computeMode=1 compressMode=3 s2R=59 hS1R=1 maskS2Size=2048
[AIV Block 4/64] [MASK-gm] s2L=4 gmAtDeltaOff=36 val=1 gmAtCorrectOff=36 val=1 s1O=0 s2O=36
[AIV Block 4/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=1 mask[1]=1 reread=1 phyAddr=1ff00
[AIV Block 4/64] [MASK-chk1] s2L=4 mask[0]=1 phyAddr=1ff00
[AIV Block 4/64] [MASK-chk2] s2L=4 mask[0]=1
[AIV Block 4/64] [MASK-chk3] s2L=4 mask[0]=1 expPhy=15700
[AIV Block 4/64] [MASK-chk4] s2L=4 mask[0]=1
[AIV Block 4/64] [MASK-chk5] s2L=4 mask[0]=1 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 4/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=43.086498 mx1=0.118168 mx2=1.271606 mx3=1.790428 sum0=1.340824[AIV Block 4/64] [VEC-Nd-pre] s2L=4 s2R=59 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=43.086498 iniSum[0]=1.340824 iniMax[1]=0.118168
[AIV Block 4/64]   mmRes[0]=100.587135
[AIV Block 4/64]   mmRes[1]=13.528423
[AIV Block 4/64]   mmRes[2]=196.564133
[AIV Block 4/64]   mmRes[3]=68.044609
[AIV Block 4/64]   mmRes[64]=-340282346638528859811704183484516925440.000000 mmRes[128]=-340282346638528859811704183484516925440.000000 mmRes[192]=8.291304 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=1 mask[1]=1
[AIV Block 4/64] [VEC-Nd-post] s2L=4 mx0=43.086498 sm0=1.340824 mx1=0.118168 sm1=-inf
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=4 mx0=43.086498 sm0=1.340824 mx1=0.118168 sm1=-inf
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=4 mx0=43.086498 sm0=1.340824 mx1=0.118168 sm1=-inf
[AIV Block 4/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 4/64] [LSE-out] lseOff=546 lse0=0.000000 lse1=0.000000
[AIV Block 4/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-expUb] s2L=2 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 4/64] [VEC2-upd] s2L=2 vo0=-4.641701 vo1=-7.555933
[AIV Block 4/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 4/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 4/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 4/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=-340282346638528859811704183484516925440.000000 mx1=-340282346638528859811704183484516925440.000000 mx2=-340282346638528859811704183484516925440.000000 mx3=-340282346638528859811704183484516925440.000000 sum0=2.080548[AIV Block 4/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=-340282346638528859811704183484516925440.000000 iniSum[0]=2.080548 iniMax[1]=-340282346638528859811704183484516925440.000000
[AIV Block 4/64]   mmRes[0]=177.786774
[AIV Block 4/64]   mmRes[1]=-75.375687
[AIV Block 4/64]   mmRes[2]=-127.959305
[AIV Block 4/64]   mmRes[3]=93.991081
[AIV Block 4/64]   mmRes[64]=36.104656 mmRes[128]=-206.682037 mmRes[192]=-63.603222 mmRes[255]=6.758065
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VF256-pre] m=1 oriN=256 preMax[0]=-340282346638528859811704183484516925440.000000 preSum[0]=2.080548 preMax[1]=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=177.786774 -75.375687 -127.959305 93.991081 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 4/64] [VF256-post] m=1 mx0=-340282346638528859811704183484516925440.000000 sum0=2.080548 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=-75.375687
[AIV Block 4/64] [VEC-Nd-post] s2L=0 mx0=45.602409 sm0=1.349849 mx1=-340282346638528859811704183484516925440.000000 sm1=-3.906044
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=0 mx0=45.602409 sm0=1.349849 mx1=-340282346638528859811704183484516925440.000000 sm1=-3.906044
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=0 mx0=45.602409 sm0=1.349849 mx1=-340282346638528859811704183484516925440.000000 sm1=-3.906044
[AIV Block 4/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-expUb] s2L=3 e0=0.183664 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 4/64] [VEC2-upd] s2L=3 vo0=-4.762465 vo1=-7.310294
[AIV Block 4/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 4/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 4/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 4/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 4/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=45.602409 mx1=-340282346638528859811704183484516925440.000000 mx2=-340282346638528859811704183484516925440.000000 mx3=-340282346638528859811704183484516925440.000000 sum0=1.349849[AIV Block 4/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=45.602409 iniSum[0]=1.349849 iniMax[1]=-340282346638528859811704183484516925440.000000
[AIV Block 4/64]   mmRes[0]=22.424114
[AIV Block 4/64]   mmRes[1]=-96.684402
[AIV Block 4/64]   mmRes[2]=-20.044884
[AIV Block 4/64]   mmRes[3]=14.714994
[AIV Block 4/64]   mmRes[64]=-44.980110 mmRes[128]=19.091061 mmRes[192]=-130.354736 mmRes[255]=-102.373268
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=1 mx0=45.602409 sm0=1.349849 mx1=-340282346638528859811704183484516925440.000000 sm1=-3.906044
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=1 mx0=45.602409 sm0=1.349849 mx1=-340282346638528859811704183484516925440.000000 sm1=-3.906044
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=1 mx0=45.602409 sm0=1.349849 mx1=-340282346638528859811704183484516925440.000000 sm1=-3.906044
[AIV Block 4/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 4/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 4/64] [VEC2-upd] s2L=4 vo0=-2.381072 vo1=3.643644
[AIV Block 4/64] [VEC2-final] s2L=4 sum=1.340824 max=43.086498
[AIV Block 4/64] [VEC2-out] s2L=4 vo0=14.984156 vo1=-3036.441162
[AIV Block 4/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5911 pLC=23 s2R=220
[AIV Block 4/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 4/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 4/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 4/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 4/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 4/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 4/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 4/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 4/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 4/64] [VEC-Nd-allmax] mx0=45.602409 mx1=0.118168 mx2=1.271606 mx3=1.790428 sum0=1.360807[AIV Block 4/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 4/64] [VEC-Nd-pre-max] iniMax[0]=45.602409 iniSum[0]=1.360807 iniMax[1]=0.118168
[AIV Block 4/64]   mmRes[0]=117.485893
[AIV Block 4/64]   mmRes[1]=1.255013
[AIV Block 4/64]   mmRes[2]=-19.652077
[AIV Block 4/64]   mmRes[3]=215.153351
[AIV Block 4/64]   mmRes[64]=-114.929337 mmRes[128]=159.025192 mmRes[192]=-22.867235 mmRes[255]=24.193621
[AIV Block 4/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 4/64] [VEC-Nd-post] s2L=2 mx0=45.602409 sm0=1.360807 mx1=0.118168 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VEC-Nd-preUpd] s2L=2 mx0=45.602409 sm0=1.360807 mx1=0.118168 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VEC-Nd-postUpd] s2L=2 mx0=45.602409 sm0=1.360807 mx1=0.118168 sm1=-340282346638528859811704183484516925440.000000
[AIV Block 4/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
CANN Version: 1.0.0, TimeStamp: 202607
[AIV Block 5/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=1ff00
[AIV Block 5/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15600
[AIV Block 5/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=-1.029378 mx1=-2.930444 mx2=3.950408 mx3=0.102666 sum0=-3.817230[AIV Block 5/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=-1.029378 iniSum[0]=-3.817230 iniMax[1]=-2.930444
[AIV Block 5/64]   mmRes[0]=-15.134365
[AIV Block 5/64]   mmRes[1]=-50.951782
[AIV Block 5/64]   mmRes[2]=123.908264
[AIV Block 5/64]   mmRes[3]=-198.198227
[AIV Block 5/64]   mmRes[64]=-14.840070 mmRes[128]=-60.491108 mmRes[192]=198.530731 mmRes[255]=115.616882
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VF256-pre] m=1 oriN=256 preMax[0]=-1.029378 preSum[0]=-3.817230 preMax[1]=-2.930444
[AIV Block 5/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-15.134365 -50.951782 123.908264 -198.198227 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 5/64] [VF256-post] m=1 mx0=-1.029378 sum0=-3.817230 srcPost0=-15.134365 srcPost1=-50.951782
[AIV Block 5/64] [VEC-Nd-post] s2L=0 mx0=36.346630 sm0=1.002227 mx1=-2.930444 sm1=0.855387
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=0 mx0=36.346630 sm0=1.002227 mx1=-2.930444 sm1=0.855387
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=0 mx0=36.346630 sm0=1.002227 mx1=-2.930444 sm1=0.855387
[AIV Block 5/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=21f00
[AIV Block 5/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15700
[AIV Block 5/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=36.346630 mx1=-2.930444 mx2=3.950408 mx3=0.102666 sum0=1.002227[AIV Block 5/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=36.346630 iniSum[0]=1.002227 iniMax[1]=-2.930444
[AIV Block 5/64]   mmRes[0]=-1.841506
[AIV Block 5/64]   mmRes[1]=40.323288
[AIV Block 5/64]   mmRes[2]=-59.346825
[AIV Block 5/64]   mmRes[3]=160.995560
[AIV Block 5/64]   mmRes[64]=-364.171173 mmRes[128]=6.197699 mmRes[192]=8.743000 mmRes[255]=83.786423
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=1 mx0=36.346630 sm0=1.002227 mx1=-2.930444 sm1=0.855387
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=1 mx0=36.346630 sm0=1.002227 mx1=-2.930444 sm1=0.855387
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=1 mx0=36.346630 sm0=1.002227 mx1=-2.930444 sm1=0.855387
[AIV Block 5/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5911 pLC=23 s2R=220
[AIV Block 5/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=1ff00
[AIV Block 5/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15800
[AIV Block 5/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=37.551838 mx1=-1.072202 mx2=3.950408 mx3=0.613007 sum0=2.877977[AIV Block 5/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=37.551838 iniSum[0]=2.877977 iniMax[1]=-1.072202
[AIV Block 5/64]   mmRes[0]=-108.545067
[AIV Block 5/64]   mmRes[1]=-137.140320
[AIV Block 5/64]   mmRes[2]=-40.101273
[AIV Block 5/64]   mmRes[3]=-126.606567
[AIV Block 5/64]   mmRes[64]=-162.105286 mmRes[128]=0.593231 mmRes[192]=-157.370224 mmRes[255]=199.699066
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=2 mx0=37.551838 sm0=2.877977 mx1=-1.072202 sm1=-2.301635
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=2 mx0=37.551838 sm0=2.877977 mx1=-1.072202 sm1=-2.301635
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=2 mx0=37.551838 sm0=2.877977 mx1=-1.072202 sm1=-2.301635
[AIV Block 5/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-cpy] s2L=0 vo0=-3.912960 vo1=0.905033
[AIV Block 5/64] [Bmm1Tail-post] s2L=4 s2S=4864 s2E=5911 pLC=23 s2R=59
[AIV Block 5/64] [MASK-entry] s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=3 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=3 mask[0]=ff phyAddr=21f00
[AIV Block 5/64] [MASK-chk2] s2L=3 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=3 mask[0]=ff expPhy=15600
[AIV Block 5/64] [MASK-chk4] s2L=3 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=3 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=44.944931 mx1=-1.072202 mx2=3.950408 mx3=0.613007 sum0=1.008019[AIV Block 5/64] [VEC-Nd-pre] s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=44.944931 iniSum[0]=1.008019 iniMax[1]=-1.072202
[AIV Block 5/64]   mmRes[0]=101.405975
[AIV Block 5/64]   mmRes[1]=-21.287367
[AIV Block 5/64]   mmRes[2]=20.731651
[AIV Block 5/64]   mmRes[3]=70.447327
[AIV Block 5/64]   mmRes[64]=401.261383 mmRes[128]=1.293064 mmRes[192]=65.004059 mmRes[255]=11.097782
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=3 mx0=44.944931 sm0=1.008019 mx1=-1.072202 sm1=-4.736664
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=3 mx0=44.944931 sm0=1.008019 mx1=-1.072202 sm1=-4.736664
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=3 mx0=44.944931 sm0=1.008019 mx1=-1.072202 sm1=-4.736664
[AIV Block 5/64] [VEC2-entry] s2L=1 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-expUb] s2L=1 e0=0.299630 e1=0.155947 e2=1.000000 e3=0.600291
[AIV Block 5/64] [VEC2-upd] s2L=1 vo0=8.886152 vo1=5.675467
[AIV Block 5/64] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=4 s2S=4864 s2R=59 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=4 isPref=0
[AIV Block 5/64] [MASK-offset] s2L=4 s2Off=36 deltaN=5794 deltaC=21 deltaPre=-925 maskOff=2084 computeMode=1 s1Off=0
[AIV Block 5/64] [COPYIN-offset] s2L=4 maskOff=2084 preOff=2084 computeMode=1 compressMode=3 s2R=59 hS1R=1 maskS2Size=2048
[AIV Block 5/64] [MASK-gm] s2L=4 gmAtDeltaOff=2084 val=1 gmAtCorrectOff=2084 val=1 s1O=1 s2O=36
[AIV Block 5/64] [MASK-gqa-nonpref] s2L=4 tM2=0 mask[0]=1 mask[1]=1 reread=1 phyAddr=1ff00
[AIV Block 5/64] [MASK-chk1] s2L=4 mask[0]=1 phyAddr=1ff00
[AIV Block 5/64] [MASK-chk2] s2L=4 mask[0]=1
[AIV Block 5/64] [MASK-chk3] s2L=4 mask[0]=1 expPhy=15700
[AIV Block 5/64] [MASK-chk4] s2L=4 mask[0]=1
[AIV Block 5/64] [MASK-chk5] s2L=4 mask[0]=1 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[AIV Block 5/64] [VEC-Nd-info] s2L=4 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=65.988426 mx1=-1.072202 mx2=3.950408 mx3=0.613007 sum0=1.000003[AIV Block 5/64] [VEC-Nd-pre] s2L=4 s2R=59 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=65.988426 iniSum[0]=1.000003 iniMax[1]=-1.072202
[AIV Block 5/64]   mmRes[0]=-305.907990
[AIV Block 5/64]   mmRes[1]=250.200974
[AIV Block 5/64]   mmRes[2]=-36.608837
[AIV Block 5/64]   mmRes[3]=-111.078819
[AIV Block 5/64]   mmRes[64]=-21.471369 mmRes[128]=0.078575 mmRes[192]=-20.844194 mmRes[255]=-340282346638528859811704183484516925440.000000
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=1 mask[1]=1
[AIV Block 5/64] [VEC-Nd-post] s2L=4 mx0=65.988426 sm0=1.000003 mx1=-1.072202 sm1=-7.171693
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=4 mx0=65.988426 sm0=1.000003 mx1=-1.072202 sm1=-7.171693
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=4 mx0=65.988426 sm0=1.000003 mx1=-1.072202 sm1=-7.171693
[AIV Block 5/64] [SoftmaxCopyOut] isFd=0 s2L=4 s2LL=4
[AIV Block 5/64] [LSE-out] lseOff=547 lse0=0.000000 lse1=0.000000
[AIV Block 5/64] [VEC2-entry] s2L=2 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-expUb] s2L=2 e0=0.000615 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 5/64] [VEC2-upd] s2L=2 vo0=-12.715517 vo1=3.262269
[AIV Block 5/64] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=0 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=0 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=0 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=0 mask[0]=ff phyAddr=21f00
[AIV Block 5/64] [MASK-chk2] s2L=0 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=0 mask[0]=ff expPhy=15800
[AIV Block 5/64] [MASK-chk4] s2L=0 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=0 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 5/64] [VEC-Nd-info] s2L=0 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=-1.831844 mx1=1.219622 mx2=0.255079 mx3=-1.941301 sum0=-1.159922[AIV Block 5/64] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=-1.831844 iniSum[0]=-1.159922 iniMax[1]=1.219622
[AIV Block 5/64]   mmRes[0]=-174.462509
[AIV Block 5/64]   mmRes[1]=36.842709
[AIV Block 5/64]   mmRes[2]=25.186804
[AIV Block 5/64]   mmRes[3]=-114.386284
[AIV Block 5/64]   mmRes[64]=-232.508316 mmRes[128]=-242.638168 mmRes[192]=-254.979156 mmRes[255]=51.407951
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VF256-pre] m=1 oriN=256 preMax[0]=-1.831844 preSum[0]=-1.159922 preMax[1]=1.219622
[AIV Block 5/64] [VF256-srcUb] m=1 oriN=256 src[0..3]=-174.462509 36.842709 25.186804 -114.386284 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 5/64] [VF256-post] m=1 mx0=-1.831844 sum0=-1.159922 srcPost0=-340282346638528859811704183484516925440.000000 srcPost1=36.842709
[AIV Block 5/64] [VEC-Nd-post] s2L=0 mx0=69.954796 sm0=1.000000 mx1=1.219622 sm1=0.326222
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=0 mx0=69.954796 sm0=1.000000 mx1=1.219622 sm1=0.326222
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=0 mx0=69.954796 sm0=1.000000 mx1=1.219622 sm1=0.326222
[AIV Block 5/64] [VEC2-entry] s2L=3 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-expUb] s2L=3 e0=0.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 5/64] [VEC2-upd] s2L=3 vo0=6.166935 vo1=9.390318
[AIV Block 5/64] [Bmm1Tail-post] s2L=2 s2S=4864 s2E=5911 pLC=23 s2R=256
[AIV Block 5/64] [MASK-entry] s2L=1 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=1 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=1 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=1 mask[0]=ff phyAddr=1ff00
[AIV Block 5/64] [MASK-chk2] s2L=1 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=1 mask[0]=ff expPhy=15600
[AIV Block 5/64] [MASK-chk4] s2L=1 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=1 mask[0]=ff maskPhy=1ff00 mmResPhy=c000 maxPhy=15500 sumPhy=14200
[AIV Block 5/64] [VEC-Nd-info] s2L=1 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=69.954796 mx1=1.219622 mx2=0.255079 mx3=-1.941301 sum0=1.000000[AIV Block 5/64] [VEC-Nd-pre] s2L=1 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=69.954796 iniSum[0]=1.000000 iniMax[1]=1.219622
[AIV Block 5/64]   mmRes[0]=270.163208
[AIV Block 5/64]   mmRes[1]=-169.411316
[AIV Block 5/64]   mmRes[2]=-140.990387
[AIV Block 5/64]   mmRes[3]=-123.014648
[AIV Block 5/64]   mmRes[64]=151.643768 mmRes[128]=-75.251762 mmRes[192]=-85.956749 mmRes[255]=76.249466
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=1 mx0=69.954796 sm0=1.000000 mx1=1.219622 sm1=0.326222
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=1 mx0=69.954796 sm0=1.000000 mx1=1.219622 sm1=0.326222
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=1 mx0=69.954796 sm0=1.000000 mx1=1.219622 sm1=0.326222
[AIV Block 5/64] [VEC2-entry] s2L=4 s2LL=4 v2S1=1
[AIV Block 5/64] [VEC2-expUb] s2L=4 e0=1.000000 e1=1.000000 e2=1.000000 e3=1.000000
[AIV Block 5/64] [VEC2-upd] s2L=4 vo0=-7.714834 vo1=7.710914
[AIV Block 5/64] [VEC2-final] s2L=4 sum=1.000003 max=65.988426
[AIV Block 5/64] [VEC2-out] s2L=4 vo0=93583.429688 vo1=-420401.843750
[AIV Block 5/64] [Bmm1Tail-post] s2L=3 s2S=4864 s2E=5911 pLC=23 s2R=220
[AIV Block 5/64] [MASK-entry] s2L=2 s2S=4864 s2R=256 hS1R=1 gS1M=0 isGqa=1 isMla=0 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=60 prefSz=5852 pLC=23
[AIV Block 5/64] [MASK-gqa-nonmerge] s2L=2 isPref=1
[AIV Block 5/64] [MASK-gqa-pref] s2L=2 fill=0xFF mask[0]=ff
[AIV Block 5/64] [MASK-chk1] s2L=2 mask[0]=ff phyAddr=21f00
[AIV Block 5/64] [MASK-chk2] s2L=2 mask[0]=ff
[AIV Block 5/64] [MASK-chk3] s2L=2 mask[0]=ff expPhy=15700
[AIV Block 5/64] [MASK-chk4] s2L=2 mask[0]=ff
[AIV Block 5/64] [MASK-chk5] s2L=2 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15500 sumPhy=14200
[AIV Block 5/64] [VEC-Nd-info] s2L=2 s2LL=4 s2S=4864 s2E=5911 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=60 preT=946 nextT=-1 compM=3
[AIV Block 5/64] [VEC-Nd-allmax] mx0=69.954796 mx1=1.219622 mx2=3.950408 mx3=0.613007 sum0=1.000000[AIV Block 5/64] [VEC-Nd-pre] s2L=2 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 5/64] [VEC-Nd-pre-max] iniMax[0]=69.954796 iniSum[0]=1.000000 iniMax[1]=1.219622
[AIV Block 5/64]   mmRes[0]=69.411865
[AIV Block 5/64]   mmRes[1]=-109.635292
[AIV Block 5/64]   mmRes[2]=-11.750810
[AIV Block 5/64]   mmRes[3]=-12.796642
[AIV Block 5/64]   mmRes[64]=48.575470 mmRes[128]=308.040955 mmRes[192]=-11.880104 mmRes[255]=181.709167
[AIV Block 5/64] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 5/64] [VEC-Nd-post] s2L=2 mx0=69.954796 sm0=1.000000 mx1=1.219622 sm1=-2.108807
[AIV Block 5/64] [VEC-Nd-preUpd] s2L=2 mx0=69.954796 sm0=1.000000 mx1=1.219622 sm1=-2.108807
[AIV Block 5/64] [VEC-Nd-postUpd] s2L=2 mx0=69.954796 sm0=1.000000 mx1=1.219622 sm1=-2.108807
[AIV Block 5/64] [VEC2-entry] s2L=0 s2LL=4 v2S1=1