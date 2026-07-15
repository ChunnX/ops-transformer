# MoeTokenPermute 100-case comparison

All three paths use the same 100 logical cases, input ranges, seed 42, and run=3 on Ascend950DT_9584. Kernel time is TTK `dyn_perf_us`. ACLNN device time is the sum of every profiled device OP's total time divided by the two profiled API calls; the first call is excluded by TTK as warmup. Host API and workspace times are retained in the detailed CSV but are not used for kernel ratios.

## Overall

| path | precision | exec fail | precision fail | perf samples | median us | median vs kernel | p90 vs kernel | faster than kernel |
|---|---|---|---|---|---|---|---|---|
| kernel | 100/100 | 0 | 0 | 100 | 11.465 |  |  | - |
| aclnn V1 | 92/100 | 5 | 3 | 95 | 14.168 | 1.069 | 3.597 | 10/92 |
| aclnn V2 | 87/100 | 5 | 8 | 95 | 154.141 | 12.145 | 24.235 | 1/87 |

Ratios use only precision-PASS cases. A ratio above 1 means the ACLNN device path is slower than the direct kernel. Aggregates span heterogeneous shapes, so the per-case CSV and performance matrix are the authoritative comparison.

## By token dtype

| dtype | cases | V1 pass | V1 median ratio | V2 pass | V2 median ratio |
|---|---|---|---|---|---|
| float16 | 50 | 45 | 1.062 | 45 | 9.364 |
| bfloat16 | 21 | 20 | 1.050 | 18 | 13.866 |
| float32 | 18 | 17 | 1.042 | 14 | 14.163 |
| int8 | 11 | 10 | 13.605 | 10 | 13.562 |

## ACLNN device routes

V1 profiled OP presence: `{'MoeInitRoutingV2': 81, 'MoeInitRoutingV3': 11, 'Slice': 11, 'Cast': 6}`.

V2 profiled OP presence: `{'MoeInitRoutingV3': 95, 'Slice': 95, 'Cast': 33}`.

## Accuracy failures

| case | V1 | V2 |
|---|---|---|
| edge_empty_n_topk4_rank3 | PASS | EXEC_FAIL |
| edge_empty_hidden | 100%,12.5% | 100%,12.5%,100% |
| shape_rank3_tokens | EXEC_FAIL | EXEC_FAIL |
| shape_rank4_tokens_slice | EXEC_FAIL | EXEC_FAIL |
| shape_rank5_tokens_rank1 | EXEC_FAIL | EXEC_FAIL |
| data_i32_limit_values | PASS | 16.601815823605705%,16.601815823605705%,100% |
| data_i64_limit_values | PASS | 16.601815823605705%,16.601815823605705%,100% |
| data_i32_min_single | PASS | 0.0%,0.0%,100% |
| data_i32_max_single | PASS | 0.0%,0.0%,100% |
| gather_rowloop_asymmetric_1702913 | EXEC_FAIL | 100%,24.237938168303373%,100% |
| gather_rowloop_exact_multiple_3403776 | 56.251880264741274%,100% | 24.192220639666065%,24.19095733679302%,100% |
| gather_colsplit_exact_h106432 | 56.263271384546%,100% | PASS |
| shape_rank3_aligned_fp16 | EXEC_FAIL | EXEC_FAIL |
| data_i64_max_single | PASS | 0.0%,0.0%,100% |

## Performance-focused cases

| case | kernel us | V1 us | V1/kernel | V2 us | V2/kernel |
|---|---|---|---|---|---|
| perf_fp16_n128_k8_h7168_full | 12.340 | 14.168 | 1.148 | 157.721 | 12.781 |
| perf_bf16_n512_k8_h7168_full | 30.813 | 31.433 | 1.020 | 192.545 | 6.249 |
| perf_fp16_n1024_k8_h7168_full | 51.325 | 48.525 | 0.945 | 292.450 | 5.698 |
| perf_fp16_n1024_k4_h7168_full | 31.853 | 31.831 | 0.999 | 193.323 | 6.069 |
| perf_fp16_n1024_k4_h7168_crop64 | 17.364 | 23.009 | 1.325 | 173.649 | 10.001 |
| perf_fp16_n1024_k8_h7168_crop64 | 27.706 | 27.988 | 1.010 | 201.444 | 7.271 |
| perf_fp16_n4096_k8_h4096_full | 139.701 | 140.795 | 1.008 | 501.529 | 3.590 |
| perf_fp32_n4096_k1_h4096_full | 48.198 | 49.261 | 1.022 | 216.481 | 4.491 |
| perf_int8_n1280_k8_h4096_full | 35.325 | 188.962 | 5.349 | 189.342 | 5.360 |
| perf_fp16_n1024_k4_h16_full | 19.004 | 19.349 | 1.018 | 156.539 | 8.237 |
| perf_fp16_n1024_k4_h64_full | 19.205 | 19.303 | 1.005 | 156.819 | 8.166 |
| perf_fp16_n1024_k4_h256_full | 19.343 | 19.424 | 1.004 | 157.345 | 8.134 |
| perf_fp16_n1024_k4_h1024_full | 20.185 | 20.430 | 1.012 | 161.666 | 8.009 |
| perf_fp16_n1024_k4_h4096_full | 25.080 | 29.116 | 1.161 | 175.365 | 6.992 |
| perf_fp16_n1024_k4_h16_crop64 | 16.664 | 17.869 | 1.072 | 155.695 | 9.343 |
| perf_fp16_n1024_k4_h256_crop64 | 16.697 | 18.242 | 1.093 | 156.345 | 9.364 |
| perf_fp16_n1024_k4_h1024_crop64 | 17.306 | 18.256 | 1.055 | 159.917 | 9.241 |
| perf_fp16_n1024_k4_h4096_crop64 | 17.107 | 21.444 | 1.253 | 165.855 | 9.695 |

## Detailed data

See `moe_token_permute_100_comparison.csv` for every case, exact precision strings, execution status, device OP breakdown, host API time, workspace-query time, and all ratios.
