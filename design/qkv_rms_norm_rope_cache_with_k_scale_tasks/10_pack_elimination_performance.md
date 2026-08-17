# QKV KScale 性能变化记录

## 1. 记录方式

本文按时间倒序记录 `QkvRmsNormRopeCacheWithKScale` 在 TND->NTD 路径上的性能变化。每次新增记录只和上一版最新结果比较，避免跨多个历史版本混在同一张表中。

性能口径固定为 msprof `op_summary_*.csv` 中目标算子 `QkvRmsNormRopeCacheWithKScale` 的 `Task Duration(us)`，不是端到端 Python 执行时间。

本文保留各历史版本的原始性能数据和当时结论，不作为当前功能合同。当前 RoPE/M-RoPE 统一要求
`0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`；下列实际采集的三组性能 tuple 均满足该合同。

三组性能 case 固定：

```text
T = 16384
layout_qkv = TND
layout_q_out = NTD
seq_lens = [16384]
slot_mapping = random
cache_stride = d_pad
cache_pad = 8
```

| Case | Nq | Nk | Nv |
|------|----|----|----|
| `n64_kv8_perf_t16384_dpad_tnd_to_ntd` | 64 | 8 | 8 |
| `n16_kv2_perf_t16384_dpad_tnd_to_ntd` | 16 | 2 | 2 |
| `n8_kv1_perf_t16384_dpad_tnd_to_ntd` | 8 | 1 | 1 |

## 2. 2026-07-04 V-first pipeline 与 tokenTilePerAiv cap 扫描

本次在 compact stride 基础上同步 `qkv-kscale-v-first-pipeline` 分支最后一个优化提交，并在 tiling 中增加单 AIV `tokenTilePerAiv` 上界扫描。V cache 的 vec 计算提前到 Q/K vec 前发起，V MTE3 使用独立同步标志，等待 Q/K vec 到安全点后再写出。

本轮已完成 `TOKEN_TILE_PER_AIV_CAP=2～8` 的三组固定性能 case。cap=9/10 按当前结论停止采集：cap=4 已经覆盖低 head 最优点，cap=8 仅让 `Nq=64/Nk=8` 小幅变好，但 `Nq=16/Nk=2` 和 `Nq=8/Nk=1` 明显回退。

| tokenTilePerAiv cap | `n64_kv8` Task Duration(us) | `n16_kv2` Task Duration(us) | `n8_kv1` Task Duration(us) | 说明 |
|---------------------|--------------------------------|--------------------------------|-------------------------------|------|
| 2 | 541.090 | 135.089 | 122.724 | - |
| 3 | 535.215 | 125.172 | 91.702 | - |
| 4 | 534.447 | 120.145 | 75.903 | 默认值；`n16_kv2`、`n8_kv1` 最优，`n64_kv8` 基本持平 |
| 5 | 534.980 | 134.072 | 91.784 | - |
| 6 | 534.360 | 150.615 | 96.156 | - |
| 7 | 533.648 | 149.284 | 95.703 | - |
| 8 | 532.748 | 149.423 | 93.915 | `n64_kv8` 最优，但低 head 场景回退明显 |

选定默认值 `TOKEN_TILE_PER_AIV_CAP=4`。该值相比上一版 compact stride 的变化如下：

| Case | 上一版 compact_stride(us) | 本次 V-first + cap=4(us) | 差值(us) | 变化比例 |
|------|---------------------------|--------------------------|----------|----------|
| `n64_kv8_perf_t16384_dpad_tnd_to_ntd` | 534.076 | 534.447 | 0.371 | 回退 0.07% |
| `n16_kv2_perf_t16384_dpad_tnd_to_ntd` | 150.168 | 120.145 | -30.023 | 提升 19.99% |
| `n8_kv1_perf_t16384_dpad_tnd_to_ntd` | 93.755 | 75.903 | -17.852 | 提升 19.04% |

结论：cap=4 对 `Nq=16/Nk=2` 和 `Nq=8/Nk=1` 分别提升约 20%，`Nq=64/Nk=8` 仅回退 0.07%。考虑典型 `Nq=8*Nk` 场景覆盖和低 head 收益，默认采用 cap=4。

### 2.1 tokenTilePerAiv cap 扫描 op_summary 详细结果

以下结果来自 msprof 导出的 `op_summary_*.csv`。为便于横向对比，在原始列前补充 `Cap` 和 `Case` 两列；其余字段保持 msprof 导出形态。

```csv
Cap,Case,Device_id,Model ID,Task ID,Stream ID,Op Name,OP Type,OP State,Task Type,Task Start Time(us),Task Duration(us),Task Wait Time(us),Block Num,Mix Block Num,HF32 Eligible,Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Context ID,aicore_time(us),aic_total_cycles,aic_mac_time(us),aic_mac_ratio,aic_scalar_time(us),aic_scalar_ratio,aic_mte1_time(us),aic_mte1_ratio,aic_mte2_time(us),aic_mte2_ratio,aic_mte3_time(us),aic_mte3_ratio,aic_fixpipe_time(us),aic_fixpipe_ratio,aic_icache_miss_rate,aiv_time(us),aiv_total_cycles,aiv_vec_time(us),aiv_vec_ratio,aiv_scalar_time(us),aiv_scalar_ratio,aiv_mte2_time(us),aiv_mte2_ratio,aiv_mte3_time(us),aiv_mte3_ratio,aiv_icache_miss_rate,cube_utilization(%)
2,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783165692889574.791	,135.089,1305.379000,32,64,NO,"""16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,134.5,6978660,32.287,0.24,14.382,0.107,8.543,0.064,0.479,0.004,0.001,0.0,26.535,0.197,0.002,134.53,13953538,125.498,0.933,65.997,0.491,73.311,0.545,59.663,0.443,0.001,97.840
2,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783165660803969.207	,541.090,2083.092000,32,64,NO,"""16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,540.51,27829187,105.807,0.196,30.303,0.056,27.305,0.051,0.565,0.001,0.001,0.0,72.88,0.135,0.001,540.54,55652449,456.092,0.844,115.971,0.215,336.662,0.623,351.792,0.651,0.001,97.409
2,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783165714850284.797	,122.724,1502.510000,32,64,NO,"""16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,122.14,6383919,22.021,0.18,14.673,0.12,5.975,0.049,0.538,0.004,0.001,0.0,17.039,0.14,0.002,122.16,12766148,78.868,0.646,65.384,0.535,62.796,0.514,63.934,0.523,0.001,98.520
3,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783165840309845.346	,125.172,1334.863000,32,64,NO,"""16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,124.57,6490176,28.208,0.226,10.938,0.088,7.399,0.059,0.485,0.004,0.001,0.0,26.083,0.209,0.003,124.59,12976709,117.406,0.942,49.865,0.4,58.024,0.466,44.635,0.358,0.001,98.201
3,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783165795845121.229	,535.215,7430.205000,32,64,NO,"""16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,534.71,27707960,105.13,0.197,29.853,0.056,27.13,0.051,0.514,0.001,0.001,0.0,72.78,0.136,0.001,534.74,55409092,453.371,0.848,115.212,0.215,339.636,0.635,345.541,0.646,0.001,98.049
3,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783165861394461.615	,91.702,1430.050000,32,64,NO,"""16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,91.09,4693885,18.345,0.201,11.107,0.122,4.938,0.054,0.48,0.005,0.001,0.0,15.137,0.166,0.003,91.12,9384460,68.392,0.751,49.142,0.539,45.717,0.502,47.056,0.516,0.001,96.944
4,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783165971318379.689	,120.145,1732.395000,32,64,NO,"""16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,119.56,6249050,26.057,0.218,8.311,0.07,6.796,0.057,0.739,0.006,0.001,0.0,26.024,0.218,0.004,119.58,12493217,112.722,0.943,41.943,0.351,46.933,0.392,38.679,0.323,0.002,98.509
4,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783165944469147.751	,534.447,1417.298000,32,64,NO,"""16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,533.8,27744177,104.814,0.196,29.004,0.054,27.048,0.051,0.693,0.001,0.001,0.0,73.075,0.137,0.001,533.83,55482905,451.868,0.846,114.448,0.214,337.095,0.631,347.155,0.65,0.001,98.318
4,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783165992863813.493	,75.903,1481.550000,32,64,NO,"""16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,75.24,3876476,16.258,0.216,8.68,0.115,4.35,0.058,0.487,0.006,0.001,0.0,14.167,0.188,0.004,75.26,7751208,64.797,0.861,40.498,0.538,37.035,0.492,31.859,0.423,0.002,96.726
5,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166124728839.674	,134.072,2127.833000,32,64,NO,"""16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,133.48,6941242,27.068,0.203,8.287,0.062,7.771,0.058,0.658,0.005,0.001,0.0,21.478,0.161,0.004,133.5,13878472,111.425,0.835,36.169,0.271,36.397,0.273,43.153,0.323,0.002,98.054
5,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166097496376.828	,534.980,1417.126000,32,64,NO,"""16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,534.37,27815867,104.655,0.196,29.324,0.055,27.008,0.051,0.557,0.001,0.001,0.0,72.305,0.135,0.001,534.39,55626662,451.326,0.845,114.288,0.214,336.129,0.629,348.532,0.652,0.001,98.474
5,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166156576816.439	,91.784,11686.902000,32,64,NO,"""16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,91.21,4715215,15.066,0.165,7.77,0.085,4.017,0.044,0.512,0.006,0.001,0.0,13.365,0.147,0.005,91.23,9425748,62.193,0.682,36.597,0.401,29.608,0.325,34.996,0.384,0.002,97.297
6,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166265394411.740	,150.615,1739.792000,32,64,NO,"""16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,150.02,7795133,25.887,0.173,6.794,0.045,7.314,0.049,0.702,0.005,0.001,0.0,23.404,0.156,0.005,150.04,15585535,109.379,0.729,33.103,0.221,32.591,0.217,40.712,0.271,0.002,98.022
6,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166238470055.729	,534.360,1595.292000,32,64,NO,"""16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,533.73,27748058,104.785,0.196,29.583,0.055,27.043,0.051,0.681,0.001,0.001,0.0,72.471,0.136,0.001,533.75,55491580,451.676,0.846,114.866,0.215,332.05,0.622,346.476,0.649,0.001,98.348
6,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166285743549.974	,96.156,1397.971000,32,64,NO,"""16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,95.56,4958729,14.161,0.148,6.475,0.068,3.762,0.039,0.613,0.006,0.001,0.0,14.25,0.149,0.006,95.59,9915920,59.746,0.625,33.061,0.346,25.735,0.269,29.721,0.311,0.002,97.670
7,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166397499089.947	,149.284,1395.051000,32,64,NO,"""16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,148.64,7695759,25.176,0.169,6.612,0.044,7.033,0.047,0.579,0.004,0.001,0.0,24.97,0.168,0.006,148.66,15386196,108.134,0.727,30.418,0.205,30.457,0.205,36.296,0.244,0.002,97.635
7,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166370763593.325	,533.648,1458.951000,32,64,NO,"""16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,532.84,27785972,104.468,0.196,28.916,0.054,26.959,0.051,0.696,0.001,0.001,0.0,72.754,0.137,0.001,532.87,55567268,450.44,0.845,114.222,0.214,335.189,0.629,346.815,0.651,0.001,98.614
7,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166431576577.156	,95.703,1342.012000,32,64,NO,"""16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,95.12,4907014,13.675,0.144,6.436,0.068,3.624,0.038,0.465,0.005,0.001,0.0,14.327,0.151,0.007,95.15,9810052,58.995,0.62,30.902,0.325,23.581,0.248,27.973,0.294,0.002,97.109
8,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166546284615.092	,149.423,1408.140000,32,64,NO,"""16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,148.81,7730728,25.09,0.169,6.347,0.043,7.009,0.047,0.588,0.004,0.001,0.0,24.858,0.167,0.006,148.83,15457154,108.07,0.726,30.482,0.205,30.497,0.205,37.861,0.254,0.002,97.987
8,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166520641060.266	,532.748,1781.223000,32,64,NO,"""16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,532.14,27742548,104.494,0.196,30.07,0.057,26.966,0.051,0.503,0.001,0.001,0.0,72.007,0.135,0.001,532.17,55479851,450.547,0.847,114.584,0.215,330.609,0.621,344.114,0.647,0.001,98.626
8,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783166565938156.143	,93.915,1466.732000,32,64,NO,"""16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1""",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"""8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1""",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,93.32,4820060,13.184,0.141,5.721,0.061,3.487,0.037,0.497,0.005,0.001,0.0,14.78,0.158,0.008,93.34,9636791,58.0,0.621,28.865,0.309,21.23,0.227,27.348,0.293,0.002,97.204
```

## 3. 2026-07-03 compact stride vs current cursor

本次在 `current_cursor` 基础上，将 Q/K RoPE NZ UB 写出改为 112 个可用 row 加 1 个 32B gap，即 UB NZ row stride 为 113。同时 tiling 约束 `tokenTilePerAiv * max(Nq,Nk) <= 112`。

该变更主要减少 UB NZ 中无效空洞和 UB2L1 的源侧跳转跨度，写出数据量不变，因此预期收益较小。

| Case | 上一版 current_cursor(us) | 本次 compact_stride(us) | 差值(us) | 变化比例 |
|------|---------------------------|-------------------------|----------|----------|
| `n64_kv8_perf_t16384_dpad_tnd_to_ntd` | 536.753 | 534.076 | 2.677 | 提升 0.50% |
| `n16_kv2_perf_t16384_dpad_tnd_to_ntd` | 151.659 | 150.168 | 1.491 | 提升 0.98% |
| `n8_kv1_perf_t16384_dpad_tnd_to_ntd` | 93.773 | 93.755 | 0.018 | 基本持平，提升 0.02% |

结论：三组性能 case 均无退化，`Nq=64/Nk=8` 和 `Nq=16/Nk=2` 有小幅收益，`Nq=8/Nk=1` 基本持平。收益低于 pack 消除和 `CopyCosSinTile` batch 游标优化，符合其只压缩 UB 临时 NZ 排布、不减少 GM/L1 有效数据量的预期。

## 4. 2026-07-03 current cursor vs no-pack

本次在 no-pack 基础上，将 `CopyCosSinTile` 的 batch 查找改为按 `aivTokenOffset` 单调推进成员游标，避免每个 token 从 batch 0 重新查找。

| Case | 上一版 no-pack(us) | 本次 current_cursor(us) | 差值(us) | 变化比例 |
|------|--------------------|-------------------------|----------|----------|
| `n64_kv8_perf_t16384_dpad_tnd_to_ntd` | 537.417 | 536.753 | 0.664 | 提升 0.12% |
| `n16_kv2_perf_t16384_dpad_tnd_to_ntd` | 156.161 | 151.659 | 4.502 | 提升 2.88% |
| `n8_kv1_perf_t16384_dpad_tnd_to_ntd` | 104.273 | 93.773 | 10.500 | 提升 10.07% |

结论：batch 游标优化对低 head 场景收益更明显，`Nq=8/Nk=1` 的收益最大。

## 5. 2026-07-03 no-pack vs pack baseline

本次比较 Q/K RoPE NZ 后写出阶段的 pack 消除差异。no-pack 版本使用 0/1 交织 mask 直接 store 到 NZ 目标位置，去掉 pack 指令。

| Case | 上一版 pack baseline(us) | 本次 no-pack(us) | 差值(us) | 变化比例 |
|------|--------------------------|------------------|----------|----------|
| `n64_kv8_perf_t16384_dpad_tnd_to_ntd` | 545.543 | 537.417 | 8.126 | 提升 1.49% |
| `n16_kv2_perf_t16384_dpad_tnd_to_ntd` | 162.141 | 156.161 | 5.980 | 提升 3.69% |
| `n8_kv1_perf_t16384_dpad_tnd_to_ntd` | 108.140 | 104.273 | 3.867 | 提升 3.58% |

结论：消除 pack 后三组 case 均有明确收益，低 head 场景提升比例更高。

## 6. 版本说明

| 版本 | Commit | 说明 |
|------|--------|------|
| Compact stride | worktree | 在 current cursor 基础上，Q/K RoPE NZ UB 写出使用 112 可用 row 和 113 row stride。 |
| Current cursor | worktree | 在 no-pack 基础上，`CopyCosSinTile` 直接用成员变量维护 batch 游标。 |
| No-pack | `601b81231 Optimize qkv k scale scatter and add msprof script` | 使用 0/1 交织 mask 直接 store 到 NZ 目标位置，去掉 pack 指令。 |
| Pack baseline | `2c708ddb2 Add qkv kscale q out layout coverage` | 保留原 pack + scatter 写出方式；为了保证同脚本对比，baseline worktree 仅同步当前性能脚本和 case 定义，不修改 kernel 实现。 |

baseline worktree：

```text
/home/s00620103/gitcode/ops-transformer-pack-baseline
```

当前 worktree：

```text
/home/s00620103/gitcode/ops-transformer
```

## 7. 采集方法

环境：

```text
CANN: /usr/local/Ascend/cann-9.1.0/set_env.sh
SOC: ascend950
TORCH_EXTENSIONS_DIR: /home/s00620103/torch_extensions/qkv_rms_norm_rope_cache_with_k_scale
```

性能脚本：

```text
designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_msprof.sh
```

脚本默认执行以下 warmup，降低首个目标算子的波动：

```python
x = torch.zeros((16384, 16384), dtype=torch.float32, device=device)
_ = torch.all(x, dim=0)
torch.npu.synchronize()
```

对每个 case，脚本先完成输入和 cache 的 NPU 初始化，再执行上述 `ReduceAll` warmup，随后在同一条执行路径上立即提交目标算子。这样目标算子在 profiler 中的前一条 NPU 算子固定为 `ReduceAll`；初始化 cache 的 `ZerosLike`/`Cast` 不会夹在 warmup 与目标算子之间。warmup 只用于稳定设备频率，不属于目标算子耗时。

采集命令：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_msprof.sh \
  --output /tmp/qkv_kscale_msprof_compact_stride_20260703

source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_msprof.sh \
  --output /tmp/qkv_kscale_msprof_current_20260703

source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_msprof.sh \
  --output /home/s00620103/msprof/qkv_rms_norm_rope_cache_with_k_scale/20260703_perf_no_pack_warmup_repeat

source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_msprof.sh \
  --output /home/s00620103/msprof/qkv_rms_norm_rope_cache_with_k_scale/20260703_perf_pack_baseline_warmup
```

## 8. op_summary CSV 详细结果

以下结果来自 msprof 导出的 `op_summary_*.csv`。为便于横向对比，在原始列前补充 `Version` 和 `Case` 两列；其余字段保持 CSV 原始字段。结果按版本时间倒序排列。

```csv
Version,Case,Device_id,Model ID,Task ID,Stream ID,Op Name,OP Type,OP State,Task Type,Task Start Time(us),Task Duration(us),Task Wait Time(us),Block Num,Mix Block Num,HF32 Eligible,Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Context ID,aicore_time(us),aic_total_cycles,aic_mac_time(us),aic_mac_ratio,aic_scalar_time(us),aic_scalar_ratio,aic_mte1_time(us),aic_mte1_ratio,aic_mte2_time(us),aic_mte2_ratio,aic_mte3_time(us),aic_mte3_ratio,aic_fixpipe_time(us),aic_fixpipe_ratio,aic_icache_miss_rate,aiv_time(us),aiv_total_cycles,aiv_vec_time(us),aiv_vec_ratio,aiv_scalar_time(us),aiv_scalar_ratio,aiv_mte2_time(us),aiv_mte2_ratio,aiv_mte3_time(us),aiv_mte3_ratio,aiv_icache_miss_rate,cube_utilization(%)
compact_stride,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783064247897624.601,534.076,1508.662000,32,64,NO,"16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,533.45,27680437,104.986,0.197,29.191,0.055,27.093,0.051,0.593,0.001,0.001,0.0,72.541,0.136,0.001,533.48,55354834,457.311,0.857,118.069,0.221,334.413,0.627,418.279,0.784,0.001,98.160
compact_stride,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783064275167068.384,150.168,1425.263000,32,64,NO,"16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,149.57,7744579,25.173,0.168,6.237,0.042,7.033,0.047,0.643,0.004,0.001,0.0,24.118,0.161,0.006,149.6,15484508,108.856,0.728,30.282,0.202,30.89,0.206,36.578,0.245,0.003,97.676
compact_stride,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783064297026130.973,93.755,1684.897000,32,64,NO,"16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,93.13,4823325,13.148,0.141,5.54,0.059,3.477,0.037,0.562,0.006,0.001,0.0,14.436,0.155,0.008,93.16,9642608,57.86,0.621,28.2,0.303,21.38,0.23,25.979,0.279,0.003,97.436
current_cursor,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783049069815754.081,536.753,1428.028000,32,64,NO,"16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,536.11,27866549,104.805,0.195,28.967,0.054,27.046,0.05,0.663,0.001,0.001,0.0,78.753,0.147,0.001,536.14,55727830,467.387,0.872,116.118,0.217,324.438,0.605,409.992,0.765,0.001,98.327
current_cursor,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783049097402571.538,151.659,1601.601000,32,64,NO,"16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,151.09,7826047,25.165,0.167,6.413,0.042,7.03,0.047,0.678,0.004,0.001,0.0,26.112,0.173,0.006,151.11,15647982,110.294,0.73,29.945,0.198,32.082,0.212,36.247,0.24,0.003,97.733
current_cursor,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783049118172975.765,93.773,1352.952000,32,64,NO,"16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,93.18,4812350,13.185,0.142,5.959,0.064,3.487,0.037,0.713,0.008,0.001,0.0,14.907,0.16,0.008,93.21,9620960,58.274,0.625,28.522,0.306,21.117,0.227,25.426,0.273,0.003,97.195
no_pack,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783044615484414.078,537.417,1435.643000,32,64,NO,"16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,536.84,27780249,105.274,0.196,28.983,0.054,27.167,0.051,0.574,0.001,0.001,0.0,79.052,0.147,0.001,536.87,55556272,469.114,0.874,113.166,0.211,322.971,0.602,407.706,0.759,0.001,97.902
no_pack,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783044642027864.895,156.161,1434.569000,32,64,NO,"16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,155.59,8055892,25.175,0.162,6.235,0.04,7.033,0.045,0.769,0.005,0.001,0.0,24.566,0.158,0.006,155.61,16108018,109.788,0.706,40.743,0.262,36.276,0.233,37.709,0.242,0.002,97.703
no_pack,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783044663399346.157,104.273,1563.450000,32,64,NO,"16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,103.65,5348981,13.195,0.127,5.894,0.057,3.49,0.034,0.686,0.007,0.001,0.0,14.148,0.136,0.008,103.68,10693365,58.296,0.562,39.531,0.381,31.035,0.299,26.529,0.256,0.002,97.155
pack,n64_kv8_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783043937890438.199,545.543,2096.864000,32,64,NO,"16384,80,128;128;128;16384,128;16384;1024,8,16,128;1024,8,16,128;1024,8,16,1;2;1;128,128;8",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"64,16384,128;64,16384;1024,8,16,128;1024,8,16,128;1024,8,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,545.04,28245623,105.121,0.193,28.316,0.052,27.127,0.05,0.584,0.001,0.001,0.0,81.67,0.15,0.001,545.06,56487162,499.662,0.917,111.211,0.204,318.335,0.584,394.071,0.723,0.001,98.059
pack,n16_kv2_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783043967037848.448,162.141,2005.157000,32,64,NO,"16384,20,128;128;128;16384,128;16384;1024,2,16,128;1024,2,16,128;1024,2,16,1;2;1;128,128;2",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"16,16384,128;16,16384;1024,2,16,128;1024,2,16,128;1024,2,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,161.53,8403493,25.055,0.155,6.772,0.042,6.999,0.043,0.662,0.004,0.001,0.0,25.078,0.155,0.007,161.55,16802951,117.598,0.728,40.575,0.251,34.446,0.213,35.222,0.218,0.002,98.160
pack,n8_kv1_perf_t16384_dpad_tnd_to_ntd,0,4294967295,11,61,aclnnQkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale_QkvRmsNormRopeCacheWithKScale,QkvRmsNormRopeCacheWithKScale,dynamic,MIX_AIC,1783043988850549.679,108.140,1326.789000,32,64,NO,"16384,10,128;128;128;16384,128;16384;1024,1,16,128;1024,1,16,128;1024,1,16,1;2;1;128,128;1",DT_BF16;FLOAT;FLOAT;FLOAT;INT32;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT;INT32;INT32;DT_BF16;FLOAT,NCL;ND;ND;ND;ND;NCHW;NCHW;NCHW;ND;ND;ND;ND,"8,16384,128;8,16384;1024,1,16,128;1024,1,16,128;1024,1,16,1",DT_FLOAT8_E4M3FN;FLOAT;DT_FLOAT8_E4M3FN;DT_FLOAT8_E4M3FN;FLOAT,NCL;ND;NCHW;NCHW;NCHW,N/A,107.57,5596480,13.089,0.122,5.198,0.048,3.462,0.032,0.515,0.005,0.001,0.0,14.779,0.137,0.008,107.59,11189354,62.538,0.581,38.392,0.357,31.753,0.295,26.123,0.243,0.002,98.015
```

## 9. 功能校验摘要

msprof 脚本运行时会先执行 torch_extension golden 对比，再进入 profiler 导出。上述性能记录对应的所有 case 均返回 `CASE PASS` 和 `SUMMARY cases=1 success=1 failed=0`。

对比项：

```text
q_out
k_cache_touched
v_cache_touched
q_scale
k_scale_cache_touched
```

其中 Q/K FP8 量化输出存在预期内的 1 LSB 级差异，整体 match 约为 99.98%；V cache 和 scale 输出均满足当前脚本阈值。

## 10. M-RoPE 使能后的 RoPE 回归性能（2026-07-29）

本节比较 M-RoPE 功能合入前后的 RoPE 路径，固定 `Nq/Nk/Nv=16/2/2`，只改变 `T` 和 layout；
不使用 `Nq=16,Nk=4,Nv=4` 作为边界场景。每个 case 采集 3 轮，交替 before/after 顺序，统计
`op_summary_*.csv` 中目标算子的 `Task Duration(us)` 中位数。Profiler 各流水指标存在重叠，不能相加。

| Case | Before (us) | After (us) | 变化 |
|------|------------:|-----------:|-----:|
| `n16_kv2_perf_t5379_dpad_tnd` | 45.827 | 45.568 | -0.57% |
| `n16_kv2_perf_t65536_dpad_tnd` | 489.731 | 460.842 | -5.90% |
| `n16_kv2_perf_t65536_dpad_ntd` | 507.201 | 468.753 | -7.58% |

结论：M-RoPE 使能没有造成 RoPE 回归；`T=5379` 基本持平，长序列 TND/NTD 分别有约 5.9%/7.6%
收益。该 A/B 只测 RoPE 路径；后来短期加入的 M-RoPE L2 selector 未改变 RoPE selector，并已按
第 12 节的独立性能结论删除。
原始 profile 位于 `/tmp/qkv_rope_ab_profile_layouts_20260729`，分析脚本为
`/tmp/qkv_rope_ab_analyze_layouts.py`。

## 11. 大输出端到端正确性（非性能数据）

大输出用例 `n16_kv2_mrope_perf_t45000_tnd` 固定 `Nq/Nk/Nv=16/2/2`、`T=45000`。
Q(BF16)+K(INT8)+V(FP8)+K-scale(FP32) 实际写入约 198.10 MiB。q_out、k_cache、v_cache 和
k_scale_cache 均需 100% 对齐；该记录用于验证大输出精度与多 tile，不承担 L2 模板选择语义。

## 12. Fixpipe q_out L2 uncache A/B 与删除结论

在删除前基线 `aaa73a6dc` 上构建同源 ON/OFF 包，唯一 kernel 行为差异为 M-RoPE q_out Fixpipe target 是否设置
`CACHE_MODE_DISABLE`。按 OFF/ON 交替顺序采集 5 轮 `msprof Task Duration`，结果如下：

| 场景 | cacheable 中位数 | uncache 中位数 | uncache 变化 | 成对收益均值及 95% CI |
|------|-----------------:|----------------:|---------------:|-----------------------:|
| `T=5379` 负控，两包均 cacheable | 39.637 us | 39.749 us | 慢 0.283% | -0.648% `[-1.696%, +0.400%]` |
| `T=45000`，输出约 198.10 MiB | 315.972 us | 317.284 us | 慢 0.415% | -0.303% `[-1.020%, +0.414%]` |
| `T=65536`，大输出压力场景 | 486.373 us | 484.970 us | 快 0.288% | +0.306% `[-0.840%, +1.452%]` |

两个目标场景方向相反、幅度都小于 0.5%，且三个场景的成对 95% 置信区间全部跨 0。Fixpipe 分项
同样没有稳定下降，因此差异归类为性能波动，删除 `Q_OUT_L2_UNCACHEABLE` 模板、host L2 阈值计算
及 q_out cache hint。M-RoPE 恢复为单一 compact 模板；`T=45000/65536` 继续作为精度和压力回归。

## 13. ReduceAll 邻接 warmup 后的当前基线（2026-07-31）

本轮使用 transformer 仓的 `qkv_rms_norm_rope_cache_with_k_scale_msprof.sh` 完整执行 8 个性能 case，每个 case 采集一轮；目标算子的前一条 NPU 算子均为 `ReduceAll`。性能口径为 `op_summary_*.csv` 的 `Task Duration(us)`，精度 8/8 通过。

| Case | 上一版本 (us) | 当前版本 (us) | 变化 |
|------|--------------:|--------------:|-----:|
| `n64_kv8_perf_t16384_dpad_tnd_to_ntd` | 474.518 | 473.647 | -0.18% |
| `n16_kv2_perf_t16384_dpad_tnd_to_ntd` | 123.323 | 123.064 | -0.21% |
| `n8_kv1_perf_t16384_dpad_tnd_to_ntd` | 75.195 | 74.974 | -0.29% |
| `n16_kv2_perf_t5379_dpad_tnd` | 46.026 | 45.602 | -0.92% |
| `n16_kv2_mrope_perf_t5379_dpad_tnd` | 39.265 | 40.121 | +2.18% |
| `n16_kv2_perf_t65536_dpad_tnd` | 462.597 | 462.231 | -0.08% |
| `n16_kv2_perf_t65536_dpad_ntd` | 470.933 | 472.359 | +0.30% |
| `n16_kv2_perf_t65536_dpad_tnd_to_ntd` | 471.252 | 467.484 | -0.80% |

当前 profile：`/home/s00620103/tmp/qkv_perf_current_reduceall_adjacent_rope_single_tile_20260731`。
上一版本 profile：`/home/s00620103/tmp/qkv_perf_previous_6de3_reduceall_adjacent_20260731`。
两次性能采集中的所有 case 均通过精度检查；M-RoPE 的单点增幅属于本轮一次采样结果，不能据此判定稳定回归。此前出现的约 `2x` 时延样本对应设备有效频率约半频并伴随设备进程占用，已从版本比较中剔除。

## 14. 随机 mrope_position 的长序列性能基线（2026-08-03）

本轮使用 `ops-transformer 6ca8b3400` 已安装 package 和当前 DiscussionForum Golden。RoPE 与
M-RoPE 均固定 `Nq/Nk/Nv=16/2/2`、TND->TND、`d_pad=8`，按 `T=5379/65536/131072`
成对采集，每个 case 只执行一轮。所有 M-RoPE case 默认按 case seed 均匀随机生成
`mrope_position[T,3]`。目标任务前一行均为同 stream、连续 Task ID 的 `ReduceAll`；性能主值从
`op_summary_*.csv` 精确筛选唯一 `OP Type=QkvRmsNormRopeCacheWithKScale` 的
`Task Duration(us)`。六个性能 case 的精度均通过，完整 NPU Golden 为 `185/185`，section-0
不变量为 `3/3`。

| T | RoPE (us) | M-RoPE (us) | M-RoPE - RoPE | M-RoPE 吞吐（M token/s） |
|---:|----------:|------------:|---------------:|----------------------------:|
| 5,379 | 46.040 | 40.052 | -5.988 us (-13.01%) | 134.30 |
| 65,536 | 461.704 | 517.545 | +55.841 us (+12.09%) | 126.63 |
| 131,072 | 954.321 | 1,102.565 | +148.244 us (+15.53%) | 118.88 |

RoPE 与 M-RoPE 的输出 dtype、Q scale 和 K cache 语义不同，上表差值只描述两个完整场景的实测
总时延，不能解释为单一 M-RoPE 特性的净开销。M-RoPE 从 64K 到 128K 的 token 数增加 2 倍，
task duration 增加 `2.130x`，吞吐下降 `6.12%`。pipe 分项如下；各 pipe 可重叠，不能相加得到
task duration。

| T | AICore (us) | Fixpipe (us) | AIV Vec (us) | AIV MTE2 (us) | AIV MTE3 (us) |
|---:|------------:|-------------:|-------------:|--------------:|--------------:|
| 5,379 | 39.440 | 10.402 | 30.648 | 17.557 | 14.242 |
| 65,536 | 516.960 | 126.848 | 363.249 | 386.911 | 386.755 |
| 131,072 | 1,102.090 | 315.266 | 716.950 | 854.335 | 908.645 |

此前 `T=5379` 的最近数据为 `40.920 us`，此前 `T=65536` 的仿射 position 数量级参考为
`486.326 us`。这两组历史数据都使用固定仿射 position；65K 数据还使用不同 runner 且没有同一
ReduceAll 邻接协议。随机 position 会改变 cos/sin 访问局部性，因此本轮结果不能与它们做严格的
版本回归百分比判断。本节三点数据作为随机 position 合同下的新单轮基线。

原始 profile 与严格解析脚本位于
`/home/s00620103/tmp/qkv_mrope_random_position_perf_20260803`；完整精度日志位于
`/home/s00620103/tmp/qkv_mrope_128k_full_precision_20260803.log`。
