# QBSA MXFP8 STC 用例场景总结

## 1. 用例规模

- 默认 testcase 文件保留 `BASE_01`、`MIX_00~04` 共 6 条原有用例，默认关闭。
- 新增 STC 共 37 条，默认全部启用。
- MXFP8 固定约束为 `D=128`、`S1BaseSize=128`、`S2BaseSize=512`、TND + PA_BNSD。
- 所有 STC 的 `sparse_count` 按 `ceil(max(actual_seq_kv) / sparse_block_size)` 设置。
- `block_num` 至少能够容纳所有 batch 的逻辑 PA page，并额外覆盖大容量 page table 场景。

## 2. 单 Batch 场景

| No. | Case | 主要场景与覆盖目的 |
|---:|---|---|
| 1 | `STC_B1_BS128_MIN_PAGE_EQ_MASK3` | 最小 PA page，S1=S2=128，block_num/sparse_count=1。 |
| 2 | `STC_B1_BS64_STITCH_8_BLOCKS_MASK3` | 8 个 64-token sparse block 拼成一个 S2Base=512 计算块。 |
| 3 | `STC_B1_BS128_TWO_S2_LOOPS_REVERSE` | 两轮 S2 循环，sparse indices 和 block table 均反序，LSE 关闭。 |
| 4 | `STC_B1_BS64_S2_TAIL_MASK3` | S2=513 尾块，9 个 BS64 sparse block，tail 选块。 |
| 5 | `STC_B1_BS128_S1_TAIL_GT_GQA10` | S1=641 尾行，S1>S2，GQA=10。 |
| 6 | `STC_B1_BS64_BOTH_TAIL_GQA16` | S1/S2 均不对齐，BS64，GQA=16，高 head 数。 |
| 7 | `STC_B1_NOMASK_BS128_EQ_TAIL` | mask_mode=0，S1=S2=127，双尾块。 |
| 8 | `STC_B1_NOMASK_BS64_S2_TAIL_SCALE8` | 无 mask、BS64、S2 长尾、自定义 softmax scale、LSE 关闭。 |
| 9 | `STC_B1_BLOCKNUM128_BS128_RANDOM` | block_num=128，随机 sparse indices 和随机 block table。 |
| 10 | `STC_B1_BLOCKNUM512_BS64_REVERSE` | block_num=512，BS64，反序 sparse/page 映射。 |
| 11 | `STC_B1_BLOCKNUM1024_BS128_TAIL` | block_num=1024，S2=2049，17 个 sparse block，tail 选块。 |
| 12 | `STC_B1_BLOCKNUM8192_BS64_GQA10` | block_num=8192 的超大 PA 容量边界。 |
| 13 | `STC_B1_HIGH_RANGE_QK_PSCALE64` | Q/K range=128，PScale=64，量化动态范围压力。 |
| 14 | `STC_B1_ASYMMETRIC_DATA_RANGE` | Q/K/V range=64/2/128，验证非对称输入 scale。 |
| 15 | `STC_B1_CUSTOM_SOFTMAX_HIGH_S1_GT_S2` | S1>S2，softmax_scale=0.25，验证外部 scale 生效。 |
| 16 | `STC_B1_LSE_OFF_MASK3_BS128` | mask_mode=3、LSE 关闭、两轮 S2。 |
| 17 | `STC_B1_LSE_OFF_NOMASK_BS64_TAIL` | 无 mask、无 LSE、BS64 双尾块。 |

## 3. B=2 场景

| No. | Case | 主要场景与覆盖目的 |
|---:|---|---|
| 18 | `STC_B2_UNIFORM_BS128_MASK3` | 等长双 batch 主路径。 |
| 19 | `STC_B2_RAGGED_BS128_S1_TAIL` | Q=[128,257]、KV=[512,385]，batch 内不等长。 |
| 20 | `STC_B2_RAGGED_BS64_RANDOM` | BS64、双 batch 长短反向、随机索引、GQA=16。 |
| 21 | `STC_B2_ZERO_LENGTH_ROWS_MASK3` | 一个 batch 的 Q 长度为0，另一个 batch 的 KV 长度为0。 |
| 22 | `STC_B2_RAGGED_S1_GT_S2_NOMASK` | S1>S2、无 mask、LSE 关闭、GQA=10。 |

## 4. B=4 场景

| No. | Case | 主要场景与覆盖目的 |
|---:|---|---|
| 23 | `STC_B4_UNIFORM_BS64_MASK3` | 四 batch 等长，BS64 拼接主路径。 |
| 24 | `STC_B4_RAGGED_BS128_RANDOM` | Q 递增、KV 递减，随机 sparse indices 和 block table。 |
| 25 | `STC_B4_RAGGED_BS64_BOTH_TAIL` | 四组不同尾块，tail sparse 模式。 |
| 26 | `STC_B4_HIGH_HEAD_BS64_MASK3` | N1=128、N2=8、GQA=16，高 head 压力。 |
| 27 | `STC_B4_NOMASK_LSE_OFF_BS128` | 不等长 Q、等长 KV、无 mask、无 LSE。 |
| 28 | `STC_B4_EXTREME_SCALE_RAGGED_TAIL` | PScale=64、输入 range 最大128、ragged+tail+随机页表。 |

## 5. B=8 场景

| No. | Case | 主要场景与覆盖目的 |
|---:|---|---|
| 29 | `STC_B8_UNIFORM_BS128_MASK3` | 8 batch 等长基础并行场景。 |
| 30 | `STC_B8_RAGGED_BS128_REVERSE` | Q 递增、KV 递减，反序 sparse indices 和 block table。 |
| 31 | `STC_B8_RAGGED_BS64_RANDOM_LARGE_S2` | BS64、S2 最大2049、sparse_count=33、随机映射。 |
| 32 | `STC_B8_NOMASK_BS128_RANDOM_LSE_OFF` | 无 mask、无 LSE、随机映射、低 PScale=0.25。 |

## 6. B=16 与空输入场景

| No. | Case | 主要场景与覆盖目的 |
|---:|---|---|
| 33 | `STC_B16_UNIFORM_TINY_BS64_MASK3` | 16 个小序列 batch，验证大 batch 调度。 |
| 34 | `STC_B16_RAGGED_BS128_RANDOM` | 16 batch 周期性变长，随机 sparse/page 映射。 |
| 35 | `STC_B16_ONE_LARGE_BATCH_BS128` | 15 个短 batch 加1个超长 batch，S2 最大8192。 |
| 36 | `STC_B16_HIGH_HEAD_BS64_NOMASK` | B=16、N1=40、N2=8、无 mask、无 LSE。 |
| 37 | `STC_EMPTY_ACTUAL_SEQ_INPUT` | actualSeqLengths 和 actualSeqLengthsKv 传空 Tensor。 |

## 7. 泛化因子汇总

| 因子 | 覆盖范围 |
|---|---|
| Batch | `B=1/2/4/8/16`；等长、ragged、零长度、一长多短。 |
| Head | `N2=1/2/3/4/5/8`，GQA=`5/8/10/16`，N1 最大128。 |
| Block size | `64/128`；覆盖 BS64 多块拼接到 S2Base=512。 |
| PA block_num | `1/2/4/5/8/16/32/64/128/512/1024/8192`。 |
| sparse_count | `1/2/4/5/7/8/9/13/16/17/33/64`。 |
| S1/S2关系 | `S1<S2`、`S1=S2`、`S1>S2`。 |
| 对齐状态 | S1/S2 对齐、S1 对齐 S2 不对齐、S1 不对齐 S2 对齐、S1/S2 均不对齐。 |
| Mask/LSE | mask_mode=`0/3`，return_softmax_lse=`true/false`。 |
| Sparse模式 | sequential、reverse、random、tail。 |
| Block table模式 | sequential、reverse、random。 |
| PScale | `0.25~64`。 |
| Q/K/V数据范围 | `0.5~128`，并覆盖三者非对称范围。 |
| softmax_scale | 默认 `1/sqrt(128)`，以及 `0.03125/0.044194.../0.125/0.25`。 |
| 边界输入 | 空 actual-seq、部分 batch 零长度、S1/S2 尾块、大 PA 容量。 |

## 8. 设计目的

这批 STC 不是对所有因子做笛卡尔积，而是选择能够触发不同 kernel 路径的典型组合：

1. BS64 和 BS128 分别验证不同 sparse block 拼接与 scale 搬运路径。
2. S1/S2 尾块验证 C1、V1、C2、V2 的有效行列和偏移处理。
3. ragged batch、零长度和一长多短验证 actual sequence、workspace 和多核任务分配。
4. reverse/random/tail 验证 sparse_indices 与 block_table 不同映射关系。
5. 大 block_num 验证 PA 高地址偏移，不通过叠加大量 head 无限制放大内存。
6. 扩大的 PScale 和 Q/K/V range 验证 MXFP8 scale、随路反量化和 P 静态量化的动态范围。
7. mask/LSE 开关验证模板入口、tiling key 和输出分支。
