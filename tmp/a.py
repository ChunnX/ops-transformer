load npu output from file: /home/z00699874/fuzz/aclnn_fuzz_0703/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_0.bin with dtype:<class 'numpy.float16'> real dtype:fp16
[2026/07/14 09:57:20] [INFO]-tools.py:4315 - Touch file succeed: /home/z00699874/fuzz/aclnn_fuzz_0703/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_0.bin
[2026/07/14 09:57:20] [INFO]-tools.py:1976 - [data_compare_np]ouput flatten start!
[2026/07/14 09:57:20] [INFO]-tools.py:1979 - [data_compare_np]ouput flatten end!
[2026/07/14 09:57:20] [INFO]-tools.py:2093 - [data_compare_np]overflows_count start!
[2026/07/14 09:57:20] [INFO]-tools.py:2097 - [data_compare_np]overflows_count end!
[2026/07/14 09:57:20] [INFO]-tools.py:2100 - split_count:517104.0; max_diff_hd:10.0;
[2026/07/14 09:57:20] [INFO]-tools.py:1065 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1071 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/14 09:57:20] [INFO]-tools.py:1072 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000001 	 9.8281250 	 -1.4707031 	 11.2988281 	 1.1496423
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000002 	 6.7929688 	 4.2851562 	 2.5078125 	 0.3691777
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000003 	 6.0585938 	 -0.4968262 	 6.5554199 	 1.0820035
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000004 	 7.5117188 	 -1.1552734 	 8.6669922 	 1.1537962
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000005 	 0.5166016 	 0.6035156 	 0.0869141 	 0.1682420
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000006 	 -9.4453125 	 2.2910156 	 11.7363281 	 1.2425558
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000007 	 0.8339844 	 -5.7734375 	 6.6074219 	 7.9227166
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000008 	 0.1735840 	 1.1328125 	 0.9592285 	 5.5260197
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000009 	 9.3437500 	 -0.4287109 	 9.7724609 	 1.0458821
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000010 	 -8.4218750 	 0.9169922 	 9.3388672 	 1.1088822
[2026/07/14 09:57:20] [INFO]-tools.py:1081 - ...   	   ...   	   ...   	   ...    	   ...
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517095 	 -1.4199219 	 -1.4199219 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517096 	 -4.5625000 	 -4.5625000 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517097 	 3.4257812 	 3.4257812 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517098 	 1.8593750 	 1.8593750 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517099 	 6.1445312 	 6.1445312 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517100 	 -4.6093750 	 -4.6093750 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517101 	 -3.7734375 	 -3.7734375 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517102 	 -1.0830078 	 -1.0830078 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517103 	 4.4726562 	 4.4726562 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00517104 	 0.7348633 	 0.7348633 	 0.0000000 	 0.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:2264 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:2266 - Rtol   	 Atol   	 PctThd   	 PctRlt   	 Result
[2026/07/14 09:57:20] [INFO]-tools.py:2267 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:2269 - 0.0050    	 0.000025  	 99.50%   	 19.047426%   	 Failed
[2026/07/14 09:57:20] [INFO]-tools.py:2272 - Max-RelativeError is: 2.0. Threshold is: 10.0.
[2026/07/14 09:57:20] [INFO]-tools.py:1129 - Error Line-----------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1131 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/14 09:57:20] [INFO]-tools.py:1132 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000000 	 9.8281250 	 -1.4707031 	 11.2988281 	 1.1494141
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000001 	 6.7929688 	 4.2851562 	 2.5078125 	 0.3691406
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000002 	 6.0585938 	 -0.4968262 	 6.5554199 	 1.0820312
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000003 	 7.5117188 	 -1.1552734 	 8.6669922 	 1.1533203
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000004 	 0.5166016 	 0.6035156 	 0.0869141 	 0.1440430
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000005 	 -9.4453125 	 2.2910156 	 11.7363281 	 1.2421875
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000006 	 0.8339844 	 -5.7734375 	 6.6074219 	 1.1445312
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000007 	 0.1735840 	 1.1328125 	 0.9592285 	 0.8466797
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000008 	 9.3437500 	 -0.4287109 	 9.7724609 	 1.0458984
[2026/07/14 09:57:20] [INFO]-tools.py:1145 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000090 	 -5.8164062 	 -1.9130859 	 3.9033203 	 0.6708984
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000091 	 -7.6289062 	 -7.5117188 	 0.1171875 	 0.0153580
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000092 	 6.5703125 	 7.4609375 	 0.8906250 	 0.1193848
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000093 	 3.4824219 	 6.6328125 	 3.1503906 	 0.4748535
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000094 	 -7.8359375 	 -6.9843750 	 0.8515625 	 0.1087036
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000095 	 -5.5898438 	 -1.9902344 	 3.5996094 	 0.6440430
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000096 	 -0.0014372 	 -1.8808594 	 1.8794222 	 0.9995117
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000097 	 9.3906250 	 -5.1718750 	 14.5625000 	 1.5507812
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000098 	 1.2070312 	 -1.3906250 	 2.5976562 	 1.8681641
[2026/07/14 09:57:20] [INFO]-tools.py:1145 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/14 09:57:20] [INFO]-tools.py:1150 - Max-RE line:---------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1158 - 00000145 	 -6.8710938 	 6.8710938 	 13.7421875 	 2.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1158 - 00002048 	 9.3906250 	 -9.3906250 	 18.7812500 	 2.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1158 - 00017942 	 9.7265625 	 -9.7343750 	 19.4609375 	 2.0000000
[2026/07/14 09:57:20] [INFO]-tools.py:1165 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:4493 - aclnnFusedInferAttentionScoreV5:aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311 End to compare result. Duration:0.13 second.
[2026/07/14 09:57:20] [INFO]-run_aclnn.py:2887 - output-0 compare end 
[2026/07/14 09:57:20] [INFO]-run_aclnn.py:2889 - output-0 result Failed 19.04742566292274 2.0
[2026/07/14 09:57:20] [INFO]-run_aclnn.py:2885 - start compare case the first output-1 
[2026/07/14 09:57:20] [INFO]-tools.py:4122 - Step:------>>>>>> Start to compare result <<<<<<------ 
[2026/07/14 09:57:20] [INFO]-tools.py:4161 - data-compare for output index:1,and its dtype is fp32.
npu out file: /home/z00699874/fuzz/aclnn_fuzz_0703/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_output_1.bin
load npu output from file: /home/z00699874/fuzz/aclnn_fuzz_0703/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_output_1.bin with dtype:<class 'numpy.float32'> real dtype:fp32
*******
输出数据类型： fp32 method为: 1
*******
load npu output from file: /home/z00699874/fuzz/aclnn_fuzz_0703/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_1.bin with dtype:<class 'numpy.float32'> real dtype:fp32
[2026/07/14 09:57:20] [INFO]-tools.py:4315 - Touch file succeed: /home/z00699874/fuzz/aclnn_fuzz_0703/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV5/aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311_cpu_output_1.bin
[2026/07/14 09:57:20] [INFO]-tools.py:1976 - [data_compare_np]ouput flatten start!
[2026/07/14 09:57:20] [INFO]-tools.py:1979 - [data_compare_np]ouput flatten end!
[2026/07/14 09:57:20] [INFO]-tools.py:2093 - [data_compare_np]overflows_count start!
[2026/07/14 09:57:20] [INFO]-tools.py:2097 - [data_compare_np]overflows_count end!
[2026/07/14 09:57:20] [INFO]-tools.py:2100 - split_count:9072.0; max_diff_hd:10.0;
[2026/07/14 09:57:20] [INFO]-tools.py:1065 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1071 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/14 09:57:20] [INFO]-tools.py:1072 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000001 	 56.0644913 	 42.7337570 	 13.3307343 	 0.2377750
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000002 	 53.1180115 	 40.2021980 	 12.9158134 	 0.2431532
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000003 	 62.7973251 	 46.5745163 	 16.2228088 	 0.2583360
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000004 	 41.1084213 	 24.0956516 	 17.0127697 	 0.4138512
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000005 	 59.9560432 	 43.7796288 	 16.1764145 	 0.2698046
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000006 	 52.4218941 	 26.1452541 	 26.2766399 	 0.5012532
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000007 	 57.4209442 	 41.6730881 	 15.7478561 	 0.2742528
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000008 	 54.1091957 	 32.6741180 	 21.4350777 	 0.3961448
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000009 	 41.5967941 	 34.7623405 	 6.8344536 	 0.1643024
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00000010 	 50.0907707 	 29.6643028 	 20.4264679 	 0.4077891
[2026/07/14 09:57:20] [INFO]-tools.py:1081 - ...   	   ...   	   ...   	   ...    	   ...
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009063 	 57.3644829 	 57.3651390 	 0.0006561 	 0.0000114
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009064 	 59.0962677 	 59.0962639 	 0.0000038 	 0.0000001
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009065 	 61.8104630 	 61.8104477 	 0.0000153 	 0.0000002
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009066 	 51.2623787 	 51.2623825 	 0.0000038 	 0.0000001
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009067 	 49.0081215 	 49.0082169 	 0.0000954 	 0.0000019
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009068 	 53.0105782 	 53.0105896 	 0.0000114 	 0.0000002
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009069 	 50.7572136 	 50.7572250 	 0.0000114 	 0.0000002
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009070 	 49.6615028 	 49.5053291 	 0.1561737 	 0.0031448
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009071 	 46.5672798 	 46.5673218 	 0.0000420 	 0.0000009
[2026/07/14 09:57:20] [INFO]-tools.py:1062 - 00009072 	 54.4735184 	 54.4734802 	 0.0000381 	 0.0000007
[2026/07/14 09:57:20] [INFO]-tools.py:2264 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:2266 - Rtol   	 Atol   	 PctThd   	 PctRlt   	 Result
[2026/07/14 09:57:20] [INFO]-tools.py:2267 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:2269 - 0.0050    	 0.000025  	 99.50%   	 22.949735%   	 Failed
[2026/07/14 09:57:20] [INFO]-tools.py:2272 - Max-RelativeError is: 1.1256884. Threshold is: 10.0.
[2026/07/14 09:57:20] [INFO]-tools.py:1129 - Error Line-----------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1131 - Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/07/14 09:57:20] [INFO]-tools.py:1132 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000000 	 56.0644913 	 42.7337570 	 13.3307343 	 0.2377750
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000001 	 53.1180115 	 40.2021980 	 12.9158134 	 0.2431532
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000002 	 62.7973251 	 46.5745163 	 16.2228088 	 0.2583360
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000003 	 41.1084213 	 24.0956516 	 17.0127697 	 0.4138512
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000004 	 59.9560432 	 43.7796288 	 16.1764145 	 0.2698046
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000005 	 52.4218941 	 26.1452541 	 26.2766399 	 0.5012531
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000006 	 57.4209442 	 41.6730881 	 15.7478561 	 0.2742528
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000007 	 54.1091957 	 32.6741180 	 21.4350777 	 0.3961448
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000008 	 41.5967941 	 34.7623405 	 6.8344536 	 0.1643024
[2026/07/14 09:57:20] [INFO]-tools.py:1145 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000091 	 57.8622322 	 21.6727371 	 36.1894951 	 0.6254424
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000092 	 49.2469254 	 35.7716408 	 13.4752846 	 0.2736269
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000093 	 66.0163345 	 32.2042389 	 33.8120956 	 0.5121777
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000094 	 53.4213524 	 34.8423538 	 18.5789986 	 0.3477823
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000095 	 58.0437088 	 27.4208508 	 30.6228580 	 0.5275827
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000096 	 50.5128899 	 32.4783936 	 18.0344963 	 0.3570276
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000097 	 46.4246712 	 38.1366158 	 8.2880554 	 0.1785270
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000098 	 57.6470032 	 35.3614922 	 22.2855110 	 0.3865858
[2026/07/14 09:57:20] [INFO]-tools.py:1139 - 00000099 	 45.4961891 	 24.6791878 	 20.8170013 	 0.4575548
[2026/07/14 09:57:20] [INFO]-tools.py:1145 -      ... 	     ... 	     ... 	     ... 	     ...
[2026/07/14 09:57:20] [INFO]-tools.py:1150 - Max-RE line:---------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:1158 - 00001700 	 56.3741760 	 -7.0855837 	 63.4597597 	 1.1256884
[2026/07/14 09:57:20] [INFO]-tools.py:1165 - ---------------------------------------------------------------------------------------
[2026/07/14 09:57:20] [INFO]-tools.py:4493 - aclnnFusedInferAttentionScoreV5:aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311 End to compare result. Duration:0.01 second.
[2026/07/14 09:57:20] [INFO]-run_aclnn.py:2887 - output-1 compare end 
[2026/07/14 09:57:20] [INFO]-run_aclnn.py:2889 - output-1 result Failed 22.94973544973545 1.1256884336471558
[2026/07/14 09:57:20] [INFO]-run_single.py:3969 - end execute aclnn case: aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311
[2026/07/14 09:57:20] [WARNING]-run_single.py:0316 - There not exist kernel file[kernel_name:aclnnFusedInferAttentionScoreV5] in kernel_meta.
[2026/07/14 09:57:22] [/home/z00699874/fuzz/aclnn_fuzz_0703/libs/../golden/aclnn_golden/aclnnFusedInferAttentionScoreV5_golden.csv]-run_single.py:1400 - [WARNING]no golden file:
[2026/07/14 09:57:22] [INFO]-run_single.py:1698 - Case report ->:
[2026/07/14 09:57:22] [INFO]-run_single.py:1699 - aclnnFusedInferAttentionScoreV5,aclnnFusedInferAttentionScoreV5_Decode_FP16_FP16_FP16_L1_fuzz_28_162_162_2_7168_57_57_57_BSND_LSE_000311,1784023042,19.04742566292274/22.94973544973545,Failed/Failed,None,0.0,no_bound,,2.0/1.1256884,"[[28, 2, 162, 57], [28, 7168, 162, 57], [28, 7168, 162, 57], [1], [2048, 2048], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1, 7168, 162, 57], [1, 7168, 162, 57], [1], [1], [1], [1], [1], [1], [1]]","['fp16', 'fp16', 'fp16', 'fp16', 'bool', 'uint64', 'fp32', 'uint64', 'fp16', 'fp16', 'fp16', 'fp16', 'int32', 'int64', 'int64', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16']","['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND']","[[-5, 5], [-10, 10], [-10, 10], ['null', 'null'], [-10.0, 10.0], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], [-10, 10], [-10, 10], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null']]","{'actualseqlengths': [], 'actualseqlengthskv': [69, 60, 80, 110, 5376, 43, 81, 3, 98, 107, 18, 92, 3840, 15, 125, 4224, 74, 3328, 75, 100, 29, 56, 21, 89, 74, 65, 3328, 1536], 'prefix_act_lens': [5852], 'qstartidx': [], 'kvstartidx': [], 'numheads': 162, 'scalevalue': 0.1324532357065044, 'pretokens': 946, 'nexttokens': -1, 'inputlayout': 'BSND', 'numkeyvalueheads': 0, 'sparsemode': 4, 'innerprecise': 0, 'blocksize': 0, 'antiquant_mode': 0, 'softmax_lse_flag': True, 'k_antiquant_mode': 0, 'v_antiquant_mode': 0, 'query_quant_mode': 0, 'psetype': 0, 'fused_flag': 1, 'mrandomtype': 'Normal', 'mrandom': 0, 'prandom': 1, 'enablegpu': 'True', 'flaglist': [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]}","[[28, 2, 162, 57], [28, 162, 2, 1]]","['fp16', 'fp32']","None","[]",N/A,"525435.6250","4544.7998",12232600.0000,"16777216",7398,"","",,,,"",,,,,None,,,Pass,,,,
[2026/07/14 09:57:22] [INFO]-run_single.py:4032 - -------------------------第1用例npu_off执行结束,共1条用例-----------------------------------
[2026/07/14 09:57:22] [INFO]-run_single.py:4059 - Case execution(aclnnFusedInferAttentionScoreV5:1/0) have completed. Duration:92.18 second.




[AIV Block 0/56] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [VEC-Nd-allmax] mx0=41.914352 mx1=0.215630 mx2=0.341691 mx3=0.678539 sum0=1.000000[AIV Block 0/56] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/56] [VEC-Nd-pre-max] iniMax[0]=41.914352 iniSum[0]=1.000000 iniMax[1]=0.215630
[AIV Block 0/56]   mmRes[0]=-27.683027
[AIV Block 0/56]   mmRes[1]=167.265762
[AIV Block 0/56]   mmRes[2]=25.492270
[AIV Block 0/56]   mmRes[3]=68.230728
[AIV Block 0/56]   mmRes[64]=57.378597 mmRes[128]=32.709888 mmRes[192]=156.699249 mmRes[255]=-231.890488
[AIV Block 0/56] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/56] [VF256-pre] m=1 oriN=256 preMax[0]=41.914352 preSum[0]=1.000000 preMax[1]=0.215630
[AIV Block 0/56] [VF256-srcUb] m=1 oriN=256 src[0..3]=-27.683027 167.265762 25.492270 68.230728 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/56] [VF256-post] m=1 mx0=41.914352 sum0=1.000000 srcPost0=-27.683027 srcPost1=167.265762
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.659897 maxPhy=15400
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.659897
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.659897
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.659897 maxPhy=15400
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.659897
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.659897
[AIV Block 0/56] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [VEC-Nd-allmax] mx0=50.631794 mx1=0.215630 mx2=0.341691 mx3=0.678539 sum0=1.000000[AIV Block 0/56] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/56] [VEC-Nd-pre-max] iniMax[0]=50.631794 iniSum[0]=1.000000 iniMax[1]=0.215630
[AIV Block 0/56]   mmRes[0]=149.403381
[AIV Block 0/56]   mmRes[1]=21.309612
[AIV Block 0/56]   mmRes[2]=-208.727676
[AIV Block 0/56]   mmRes[3]=-76.581635
[AIV Block 0/56]   mmRes[64]=-93.687088 mmRes[128]=40.739449 mmRes[192]=99.159515 mmRes[255]=38.584782
[AIV Block 0/56] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/56] [VF256-pre] m=1 oriN=256 preMax[0]=50.631794 preSum[0]=1.000000 preMax[1]=0.215630
[AIV Block 0/56] [VF256-srcUb] m=1 oriN=256 src[0..3]=149.403381 21.309612 -208.727676 -76.581635 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/56] [VF256-post] m=1 mx0=50.631794 sum0=1.000000 srcPost0=149.403381 srcPost1=21.309612
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.011398 maxPhy=15500
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.011398
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.011398
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.011398 maxPhy=15500
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.011398
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.011398
[AIV Block 0/56] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [VEC-Nd-allmax] mx0=25.657991 mx1=0.215630 mx2=0.341691 mx3=0.678539 sum0=1.042041[AIV Block 0/56] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/56] [VEC-Nd-pre-max] iniMax[0]=25.657991 iniSum[0]=1.042041 iniMax[1]=0.215630
[AIV Block 0/56]   mmRes[0]=72.263680
[AIV Block 0/56]   mmRes[1]=59.700699
[AIV Block 0/56]   mmRes[2]=12.855907
[AIV Block 0/56]   mmRes[3]=107.858582
[AIV Block 0/56]   mmRes[64]=74.422112 mmRes[128]=-15.723137 mmRes[192]=-55.743706 mmRes[255]=-31.537855
[AIV Block 0/56] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/56] [VF256-pre] m=1 oriN=256 preMax[0]=25.657991 preSum[0]=1.042041 preMax[1]=0.215630
[AIV Block 0/56] [VF256-srcUb] m=1 oriN=256 src[0..3]=72.263680 59.700699 12.855907 107.858582 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/56] [VF256-post] m=1 mx0=25.657991 sum0=1.042041 srcPost0=72.263680 srcPost1=59.700699
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.859985 maxPhy=15300
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.859985
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.859985
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.859985 maxPhy=15300
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.859985
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.859985
[AIV Block 0/56] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [VEC-Nd-allmax] mx0=42.375225 mx1=0.215630 mx2=0.341691 mx3=0.678539 sum0=1.431225[AIV Block 0/56] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/56] [VEC-Nd-pre-max] iniMax[0]=42.375225 iniSum[0]=1.431225 iniMax[1]=0.215630
[AIV Block 0/56]   mmRes[0]=-38.094933
[AIV Block 0/56]   mmRes[1]=111.754013
[AIV Block 0/56]   mmRes[2]=-170.864624
[AIV Block 0/56]   mmRes[3]=83.054977
[AIV Block 0/56]   mmRes[64]=99.262955 mmRes[128]=184.145721 mmRes[192]=44.176636 mmRes[255]=2.755916
[AIV Block 0/56] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/56] [VF256-pre] m=1 oriN=256 preMax[0]=42.375225 preSum[0]=1.431225 preMax[1]=0.215630
[AIV Block 0/56] [VF256-srcUb] m=1 oriN=256 src[0..3]=-38.094933 111.754013 -170.864624 83.054977 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/56] [VF256-post] m=1 mx0=42.375225 sum0=1.431225 srcPost0=-38.094933 srcPost1=111.754013
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.924088 maxPhy=15400
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.924088
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.924088
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.924088 maxPhy=15400
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.924088
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=68.924088
[AIV Block 0/56] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [VEC-Nd-allmax] mx0=46.574516 mx1=0.215630 mx2=0.341691 mx3=0.678539 sum0=1.000000[AIV Block 0/56] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/56] [VEC-Nd-pre-max] iniMax[0]=46.574516 iniSum[0]=1.000000 iniMax[1]=0.215630
[AIV Block 0/56]   mmRes[0]=-129.829025
[AIV Block 0/56]   mmRes[1]=-252.773346
[AIV Block 0/56]   mmRes[2]=-57.295830
[AIV Block 0/56]   mmRes[3]=63.070858
[AIV Block 0/56]   mmRes[64]=-57.154278 mmRes[128]=36.169815 mmRes[192]=47.088127 mmRes[255]=55.566555
[AIV Block 0/56] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/56] [VF256-pre] m=1 oriN=256 preMax[0]=46.574516 preSum[0]=1.000000 preMax[1]=0.215630
[AIV Block 0/56] [VF256-srcUb] m=1 oriN=256 src[0..3]=-129.829025 -252.773346 -57.295830 63.070858 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/56] [VF256-post] m=1 mx0=46.574516 sum0=1.000000 srcPost0=-129.829025 srcPost1=-252.773346
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.275589 maxPhy=15500
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.275589
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.275589
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.275589 maxPhy=15500
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.275589
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.275589
[AIV Block 0/56] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [VEC-Nd-allmax] mx0=43.779510 mx1=0.215630 mx2=0.341691 mx3=0.678539 sum0=1.000118[AIV Block 0/56] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/56] [VEC-Nd-pre-max] iniMax[0]=43.779510 iniSum[0]=1.000118 iniMax[1]=0.215630
[AIV Block 0/56]   mmRes[0]=92.591110
[AIV Block 0/56]   mmRes[1]=109.796715
[AIV Block 0/56]   mmRes[2]=-36.563492
[AIV Block 0/56]   mmRes[3]=20.013496
[AIV Block 0/56]   mmRes[64]=39.321362 mmRes[128]=-123.912766 mmRes[192]=233.216995 mmRes[255]=-6.457392
[AIV Block 0/56] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/56] [VF256-pre] m=1 oriN=256 preMax[0]=43.779510 preSum[0]=1.000118 preMax[1]=0.215630
[AIV Block 0/56] [VF256-srcUb] m=1 oriN=256 src[0..3]=92.591110 109.796715 -36.563492 20.013496 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/56] [VF256-post] m=1 mx0=43.779510 sum0=1.000118 srcPost0=92.591110 srcPost1=109.796715
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.124176 maxPhy=15300
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.124176
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.124176
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.124176 maxPhy=15300
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.124176
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.124176
[AIV Block 0/56] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [VEC-Nd-allmax] mx0=41.671658 mx1=0.215630 mx2=0.341691 mx3=0.678539 sum0=1.001431[AIV Block 0/56] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=c000
[AIV Block 0/56] [VEC-Nd-pre-max] iniMax[0]=41.671658 iniSum[0]=1.001431 iniMax[1]=0.215630
[AIV Block 0/56]   mmRes[0]=22.045473
[AIV Block 0/56]   mmRes[1]=-44.061096
[AIV Block 0/56]   mmRes[2]=-207.899368
[AIV Block 0/56]   mmRes[3]=18.254990
[AIV Block 0/56]   mmRes[64]=22.002346 mmRes[128]=112.762650 mmRes[192]=-107.273476 mmRes[255]=-198.673950
[AIV Block 0/56] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/56] [VF256-pre] m=1 oriN=256 preMax[0]=41.671658 preSum[0]=1.001431 preMax[1]=0.215630
[AIV Block 0/56] [VF256-srcUb] m=1 oriN=256 src[0..3]=22.045473 -44.061096 -207.899368 18.254990 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/56] [VF256-post] m=1 mx0=41.671658 sum0=1.001431 srcPost0=22.045473 srcPost1=-44.061096
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.188278 maxPhy=15400
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.188278
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.188278
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.188278 maxPhy=15400
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.188278
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.188278
[AIV Block 0/56] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [VEC-Nd-allmax] mx0=34.760902 mx1=0.215630 mx2=0.341691 mx3=0.678539 sum0=1.001441[AIV Block 0/56] [VEC-Nd-pre] s2L=0 s2R=256 hS1R=1 s2S=4864 phyAddr=4000
[AIV Block 0/56] [VEC-Nd-pre-max] iniMax[0]=34.760902 iniSum[0]=1.001441 iniMax[1]=0.215630
[AIV Block 0/56]   mmRes[0]=-144.173950
[AIV Block 0/56]   mmRes[1]=13.546518
[AIV Block 0/56]   mmRes[2]=-25.746313
[AIV Block 0/56]   mmRes[3]=-124.268692
[AIV Block 0/56]   mmRes[64]=193.755142 mmRes[128]=43.734676 mmRes[192]=156.172302 mmRes[255]=117.126297
[AIV Block 0/56] [VEC-Nd-mask] mask[0]=ff mask[1]=ff
[AIV Block 0/56] [VF256-pre] m=1 oriN=256 preMax[0]=34.760902 preSum[0]=1.001441 preMax[1]=0.215630
[AIV Block 0/56] [VF256-srcUb] m=1 oriN=256 src[0..3]=-144.173950 13.546518 -25.746313 -124.268692 mask[0]=ffffffffffffffff scale=0.132453
[AIV Block 0/56] [VF256-post] m=1 mx0=34.760902 sum0=1.001441 srcPost0=-144.173950 srcPost1=13.546518
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.539780 maxPhy=15500
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.539780
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=0 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.539780
[AIV Block 0/56] [VEC-Nd-post] mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.539780 maxPhy=15500
[AIV Block 0/56] [VEC-Nd-preUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.539780
[AIV Block 0/56] [VEC-Nd-postUpd] s2L=1 mx0=-340282346638528859811704183484516925440.000000 sm0=256.000000 mx1=0.215630 sm1=69.539780
[AIV Block 0/56] [Bmm1Tail-post] s2L=0 s2S=4864 s2E=5920 pLC=23 s2R=256
[AIV Block 0/56] [Bmm1Tail-post] s2L=1 s2S=4864 s2E=5920 pLC=23 s2R=256