# QkvRmsNormRopeCacheWithKScale TTK 随机 stride 精度报告

生成时间：2026-08-18T01:21:40+00:00

## 结论

- CSV 语义用例：800 条。
- 已执行并去重：9 条；PASS=9，FAIL=0。
- 回归状态：阶段性结果，尚未覆盖全部 CSV 用例。
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
| M-RoPE TND->TND | 3 |
| RoPE NTD->NTD | 2 |
| RoPE TND->NTD | 2 |
| RoPE TND->TND | 2 |

## 已执行最低命中率

| 指标 | 最低命中率 | 用例 |
|:--|--:|:--|
| `k_cache_untouched` | 100.000000% | `aclnn_full_mrope_nq8_nk1_nv1_t17_tnd_to_tnd_affine__meta_b4_h0_s2026081479_paffine_rk16_rv16_rs24` |
| `k_fp8_close` | 100.000000% | `aclnn_full_rope_nq8_nk1_nv1_t17_ntd_to_ntd_linear__meta_b16_h0_s2026081404_plinear_rk8_rv8_rs32` |
| `k_int8_abs_le_1` | 100.000000% | `aclnn_full_mrope_nq8_nk1_nv1_t17_tnd_to_tnd_affine__meta_b4_h0_s2026081479_paffine_rk16_rv16_rs24` |
| `k_scale_cache_untouched` | 100.000000% | `aclnn_full_mrope_nq8_nk1_nv1_t17_tnd_to_tnd_affine__meta_b4_h0_s2026081479_paffine_rk16_rv16_rs24` |
| `k_scale_fp32_close` | 99.999380% | `pta_full_mrope_nq64_nk8_nv8_t262143_tnd_to_tnd_random__meta_b16_h0_s2026082199_prandom_rk24_rv24_rs32` |
| `q_bf16_close` | 99.948000% | `pta_full_mrope_nq8_nk1_nv1_t17_tnd_to_tnd_affine__meta_b4_h0_s2026081479_paffine_rk16_rv16_rs24` |
| `q_fp8_close` | 100.000000% | `aclnn_full_rope_nq8_nk1_nv1_t17_ntd_to_ntd_linear__meta_b16_h0_s2026081404_plinear_rk8_rv8_rs32` |
| `q_scale_fp32_close` | 100.000000% | `aclnn_full_rope_nq8_nk1_nv1_t17_ntd_to_ntd_linear__meta_b16_h0_s2026081404_plinear_rk8_rv8_rs32` |
| `v_cache_untouched` | 100.000000% | `aclnn_full_mrope_nq8_nk1_nv1_t17_tnd_to_tnd_affine__meta_b4_h0_s2026081479_paffine_rk16_rv16_rs24` |
| `v_fp8_close` | 100.000000% | `aclnn_full_mrope_nq8_nk1_nv1_t17_tnd_to_tnd_affine__meta_b4_h0_s2026081479_paffine_rk16_rv16_rs24` |

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

- `tests/reports/ttk_aclnn_mrope_stride_probe.csv`
- `tests/reports/ttk_aclnn_stride_alignment_probe.csv`
- `tests/reports/ttk_aclnn_stride_alignment_probe_24.csv`
- `tests/reports/ttk_aclnn_tnd_ntd_stride_probe.csv`
- `tests/reports/ttk_pta_full_800_stride_smoke_pass.csv`
- `tests/reports/ttk_pta_stride_probe.csv`
- `tests/reports/ttk_pta_max_shape_stride_probe.csv`
- `tests/reports/ttk_pta_max_shape_stride_double_per_mille.csv`
