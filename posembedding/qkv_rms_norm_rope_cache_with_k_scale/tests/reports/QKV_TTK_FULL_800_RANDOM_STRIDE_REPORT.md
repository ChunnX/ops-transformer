# QkvRmsNormRopeCacheWithKScale TTK 随机 stride 精度报告

生成时间：2026-08-18T14:26:35+00:00

## 结论

- CSV 语义用例：800 条。
- 已执行并去重：800 条；PASS=800，FAIL=0。
- 回归状态：全量完成。
- offset：未配置，仅测试 stride。
- K/V 使用相同的随机行 padding，取值集合为 {8,16,24,32}；scale 独立随机。

## 精度契约

- BF16：单点 `rtol=0.001`，超差点比例不超过 `0.001`。
- FP8：单点 `rtol=0.01`，超差点比例不超过 `0.01`。
- INT8：所有点最大绝对误差 `<=1`。
- Scale FP32：单点 `rtol=0.001`，超差点比例不超过 `0.001`。
- 未命中 cache slot：逐字节不变。

## 已执行场景

| 场景 | 数量 |
|:--|--:|
| M-RoPE TND->TND | 200 |
| RoPE NTD->NTD | 200 |
| RoPE TND->NTD | 200 |
| RoPE TND->TND | 200 |

## 已执行最低命中率

| 指标 | 最低命中率 | 用例 |
|:--|--:|:--|
| `k_cache_untouched` | 100.000000% | `pta_full_rope_nq8_nk1_nv1_t1_ntd_to_ntd_linear__meta_b1_h0_s2026081400_plinear_rk8_rv8_rs32` |
| `k_fp8_close` | 99.888393% | `pta_full_rope_nq56_nk7_nv7_t1_tnd_to_ntd_linear__meta_b1_h0_s2026082050_plinear_rk32_rv32_rs16` |
| `k_int8_abs_le_1` | 100.000000% | `pta_full_mrope_nq8_nk1_nv1_t1_tnd_to_tnd_affine__meta_b1_h0_s2026081475_paffine_rk8_rv8_rs32` |
| `k_scale_cache_untouched` | 100.000000% | `pta_full_rope_nq8_nk1_nv1_t1_ntd_to_ntd_linear__meta_b1_h0_s2026081400_plinear_rk8_rv8_rs32` |
| `k_scale_fp32_close` | 99.975586% | `pta_full_rope_nq32_nk4_nv4_t1024_tnd_to_ntd_reverse__meta_b4_h1_s2026081762_preverse_rk16_rv16_rs24` |
| `q_bf16_close` | 99.971278% | `pta_full_mrope_nq32_nk4_nv4_t17_tnd_to_tnd_random__meta_b1_h0_s2026081779_prandom_rk32_rv32_rs8` |
| `q_fp8_close` | 99.902344% | `pta_full_rope_nq8_nk1_nv1_t1_ntd_to_ntd_linear__meta_b1_h0_s2026081400_plinear_rk8_rv8_rs32` |
| `q_scale_fp32_close` | 99.967192% | `pta_full_rope_nq24_nk3_nv3_t127_tnd_to_ntd_reverse__meta_b16_h1_s2026081655_preverse_rk24_rv24_rs8` |
| `v_cache_untouched` | 100.000000% | `pta_full_rope_nq8_nk1_nv1_t1_ntd_to_ntd_linear__meta_b1_h0_s2026081400_plinear_rk8_rv8_rs32` |
| `v_fp8_close` | 100.000000% | `pta_full_rope_nq8_nk1_nv1_t1_ntd_to_ntd_linear__meta_b1_h0_s2026081400_plinear_rk8_rv8_rs32` |

## 失败项

无。

## CSV stride 分布

| (K pad, V pad, scale pad) | 用例数 |
|:--|--:|
| `(8, 8, 8)` | 60 |
| `(8, 8, 16)` | 46 |
| `(8, 8, 24)` | 54 |
| `(8, 8, 32)` | 53 |
| `(16, 16, 8)` | 65 |
| `(16, 16, 16)` | 53 |
| `(16, 16, 24)` | 59 |
| `(16, 16, 32)` | 33 |
| `(24, 24, 8)` | 50 |
| `(24, 24, 16)` | 41 |
| `(24, 24, 24)` | 54 |
| `(24, 24, 32)` | 55 |
| `(32, 32, 8)` | 42 |
| `(32, 32, 16)` | 54 |
| `(32, 32, 24)` | 36 |
| `(32, 32, 32)` | 45 |

## 原始结果

- `/home/r00957730/ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale/tests/reports/ttk_pta_full_800_random_stride_result.csv`
- `/home/r00957730/ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale/tests/reports/ttk_pta_full_800_random_stride_remaining_116_799.csv`
- `/home/r00957730/ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale/tests/reports/ttk_pta_full_800_random_stride_remaining_749_799_retry.csv`
