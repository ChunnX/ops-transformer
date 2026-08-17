# SparseAttentionScorePrefill A2 性能优化报告

更新时间：2026-08-18

## 0. A2 `batchGroupsMax` 结论（2026-08-18）

历史版本曾在 `ElementS=float` 分支强制 `batchGroupsMax=1`，用于规避当时尚未证明的
S/P staging 生命周期。该保护已由提交 `0980f38b9` 移除；当前 A2 kernel 使用统一的容量
公式：

```text
min(8,
    64 / groupSize,
    8192 / (RoundUp(blockSize, 16) * groupSize))
```

其中 `64` 是 A2 QK/PV 的 `L0_TILE_M`，`8192` 是每个 AIV 的 FP32 softmax stage
容量，`8` 只是 q-group 数组的上限。对出版形状 `groupSize=16, blockSize=128`，实际
上限为 `4`，对应 `M=64`，同时正好填满 L0C 单 stage、softmax FS 和 GM S/P stage。
因此不能恢复固定 `1`，也不能直接改成 `8`；A5 的 `8` 是其 M128 数据路径的容量，
不是 A2 的通用上限。

远端验证覆盖了 1/2/4 组批处理：Q=16/32/64 的全量 CPU golden 均通过，最大绝对误差
分别不超过 `2.44e-4`、`1.22e-4`、`1.22e-4`。生产 GQA 热 CSR（Q=16、Q=64）三个
`innerPrecise=0/1/4` 子路径也通过，说明放宽到容量上限后同步和 ring 生命周期没有回归。

## 1. 结论

当前最终矩阵在 Ascend 910B3 物理 Device 7、固定频率 1800 MHz 下采集。目标
B1/Q2048/H16/KVH1/D128/block128/topK16 的 64K/128K/256K 三档几何均值为：

- `innerPrecise=0`：`2010.805 us`，tiling key 20003；
- `innerPrecise=1`：`1971.353 us`，tiling key 20002；
- `innerPrecise=4`：`2014.074 us`，tiling key 20001。

`innerPrecise=0` 对同形 `SparseAttentionScore` 的三档降幅分别为
`25.364%/22.409%/21.749%`，每档均超过 20%；三档几何平均加速为 `1.302x`，
等价几何平均耗时降低 `23.190%`。早期 Device 1 的 3688.554→2648.453 us
优化链保留在第 3 节作为历史拆解，不再作为当前版本的最终性能结论。

## 2. 测试范围

| 项目 | 配置 |
|---|---|
| SoC | Ascend 910B3, A2/arch22 |
| 设备 | 物理 Device 7；应用和 msprof 均直接使用 `npu:7` |
| 频率 | Current Freq = Rated Freq = 1800 MHz |
| Batch | 1 |
| Q Seqlen | 2048 |
| KV Seqlen | 65536 / 131072 / 262144 |
| Q Heads | 16 |
| KV Heads | 1 |
| Head Dim | 128 |
| Block Size | 128 |
| TopK | 16 |
| 输入/输出 dtype | BF16 |
| `innerPrecise` | 0 / 1 / 4 |
| Tiling key | 20003 / 20002 / 20001 |
| BlockDim / MixBlockDim | 20 / 40 |

性能入口为 [run_sparse_attention_score_prefill_perf.py](../tests/run_sparse_attention_score_prefill_perf.py)，只构造输入并调用 prefill 算子，不调用 decode，也不运行 CPU golden。性能结论以 `msprof` 的 kernel Task Duration 为准，Python/NPU Event 计时只用于快速观察。

## 3. 历史优化链路

本节保留早期 Device 1、innerPrecise=4 的逐项 A/B，用于说明各优化来源；最终性能
验收以第 10、11 节 Device 7 的统一 10-launch 矩阵为准。

### 3.1 已验证优化链路

| 版本 | 耗时 (us) | 相对上一版本 | 相对原始基线 | 相对原始加速 | 状态 |
|---|---:|---:|---:|---:|---|
| 原始 A2 基线 | 3688.554 | - | - | 1.000x | 已采集 |
| LSE 搬运合并 | 3671.133 | -0.472% | -0.472% | 1.005x | 已验证 |
| O partial 批量搬运 | 3662.854 | -0.226% | -0.697% | 1.007x | 已验证 |
| FP32 `RowMuls` 行批处理 | 2813.256 | -23.195% | -23.730% | 1.311x | 已验证，主要收益 |
| 消除冗余 `invRowSum` | 2665.973 | -5.235% | -27.723% | 1.384x | 已验证 |
| 每 AIV 独立处理完整 GQA task | 2661.268 | -0.177% | -27.851% | 1.386x | 已验证 |
| LSE 标量广播改为 `Brcb` | 2653.993 | -0.273% | -28.048% | 1.390x | 已验证 |
| scale 标量除法改为 Vector `Div` | **2648.453** | **-0.209%** | **-28.198%** | **1.393x** | 当前最佳已验证版本 |

Phase2 三项新优化相对 `2665.973 us` 版本合计降低 `17.520 us`，收益 `0.657%`。这说明完整 GQA task、`Brcb` 和 Vector `Div` 的方向正确，但固定 task 管理和少量标量操作已经不是主要剩余瓶颈。

### 3.2 独立实验

| 实验 | 平均耗时 (us) | 对比版本 | 变化 | 结论 |
|---|---:|---:|---:|---|
| Phase1 fully-causal softmax 融合 | 2670.243 | 2665.973 | +0.160% | 未产生收益，不纳入当前实现 |

### 3.3 采样口径

早期五个结果来自对应 `OpBasicInfo.csv` 的单次 `Task Duration`。后续 Phase2 实验使用 `msprof --task-time=on`，profile 包含 3 次 warmup 和 1 次测量，表中采用 `op_statistic.csv` 的 4 次平均值。

最新 Vector `Div` profile 的 4 次 Task Duration 为：

```text
2617.672 us, 2666.173 us, 2656.393 us, 2653.573 us
```

平均值为 `2648.453 us`，最小值为 `2617.672 us`，最大值为 `2666.173 us`。只取 warmup 后的最后一次测量时，耗时为 `2653.573 us`；相对 `2665.973 us` 的收益为 `0.465%`。因此，小于 1% 的 Phase2 收益应视为方向性结果，最终门禁建议用完全一致的 launch-count 配置重复采集。

## 4. 历史 Profile 热点

最新 Vector `Div` 版本的四次 `op_summary.csv` 均值如下：

| Pipeline | 平均时间 (us) | 说明 |
|---|---:|---|
| AIV total | 2638.738 | 当前关键路径 |
| AIV scalar | 1951.734 | 约占 AIV 时间 74%，首要剩余热点 |
| AIV vector | 1055.617 | 与 scalar pipeline 存在重叠 |
| AIV MTE2 | 258.655 | 不是主瓶颈 |
| AIV MTE3 | 137.449 | 不是主瓶颈 |
| AIC total | 1103.710 | 明显短于 AIV |
| AIC Cube/MAC | 145.180 | Cube 不是当前瓶颈 |
| AIC scalar | 650.321 | 次要热点 |

上一版详细 profile 还显示 GM 到 UB、UB 到 GM 带宽利用率较低，UB bank conflict 和 Vector resource conflict 均不高。结合当前 AIV scalar 占比，后续优化应优先减少 Phase1/Phase2 的标量控制、循环发射和同步等待，而不是继续做小粒度 DMA 合并。

## 5. 已实施优化

### 5.1 Phase2 O partial 批量搬运

根据 A2 UB 容量动态提高 `splitsPerBuf`，对 D=128、group size=16、topK=16 的目标 shape，一次批量搬运多个 split，减少小 DMA 发射次数。单独收益较小，但为后续 `RowMuls` 批处理和双缓冲提供了连续 UB 布局。

### 5.2 FP32 RowMuls 批处理

将逐行 `GetValue + Duplicate + Mul/Add` 改为按多行执行的 Vector `Mul`，并保持逐 split 的 FP32 `Mul -> Add` 累加顺序。该项将耗时从 `3662.854 us` 降到 `2813.256 us`，是目前最主要的性能收益。

### 5.3 消除冗余 invRowSum

利用 Phase2 合并公式：

```text
O = sum_j(exp(max_j - global_max) / denominator * O_partial_j)
denominator = sum_j(row_sum_j * exp(max_j - global_max))
```

直接使用 combined scale，避免在 O partial 上再次执行逐元素 `1 / rowSum` 归一化。该项降低 `147.283 us`，相对上一版提升 `5.235%`。

### 5.4 每 AIV 独立处理完整 GQA task

Phase2 从两个 AIV 分别处理 8 行，调整为每个线性 AIV 独立处理一个完整的 16 行 GQA task。该方案使统计量搬运命中连续布局，并减少固定 task 开销。实测单项收益约 `0.177%`。

### 5.5 Brcb 和 Vector Div

将 LSE 的逐元素 `GetValue/SetValue/Duplicate` 广播替换为 `Brcb`，再将每行 scale 的标量除法和广播替换为 Vector `Div`。两项累计降低约 `12.815 us`，没有改变 Phase2 的 FP32 计算公式和最终 BF16 cast 点。

## 6. 精度验证

历史 Vector `Div` 优化落地时完成以下 arch22 定向回归；后续当前对象的完整补充结果见本节下半部分：

| 用例 | 关注点 | 结果 |
|---|---|---|
| `test_arch22_bf16_q16_hot_csr_precision` | 最小 hot-CSR、多批 pipeline 边界 | PASS |
| `test_arch22_bf16_q128_hot_csr_precision` | 单 hot row 跨多个 batch | PASS |
| `test_arch22_bf16_q2048_kv133120_precision_and_perf` | 生产规模 Q，首/中/尾 token CPU sample golden | PASS |

这些用例对完整输出执行 NaN、Inf 和全零检查，并对选定 token 执行 CPU 两阶段 golden 比较，容差为 `1.5e-2`。Q=2048 用例比较 token `0/1024/2047`，不是全量 2048 token CPU golden；因此报告中的 PASS 表示当前定向回归通过，不等价于全 shape 全元素 exhaustively verified。

精度入口为 [test_sparse_attention_score_prefill_npu.py](../tests/test_sparse_attention_score_prefill_npu.py)。

2026-08-18 远端物理 Device 7 的补充回归结果：

| 回归集合 | 结果 |
|---|---|
| medium + partial-tail + GQA16 uneven + group-size sweep | `4 passed, 26 subtests passed` |
| D256 四组路径 | `4 passed, 12 subtests passed` |
| blockSize 16/32/.../128 | `1 passed, 24 subtests passed` |
| KV/Q 边界、连续 KV、多 batch | `5 passed, 35 subtests passed` |

补充集合的最大绝对误差为 `9.77e-4`；没有 NaN、Inf 或全零输出。上述测试均使用当前安装对象，且覆盖 `innerPrecise=0/1/4`（按用例配置）。

最终同对象能力回归在物理 Device 7 重跑并保存原始日志：

```text
/opt/rrh/code3/sparse_attention_test/prefill_final_capability_precision_20260818.log
17 passed, 89 subtests passed in 1007.24s
max_abs_diff = 9.77e-4
```

该集合包含 LSE、medium、partial tail、GQA16、groupSize 1..16、D256、H64、
blockSize 16..128、KV=256K、totalQ=1024K、axis-0 stride、blockSize=0 以及
生产 GQA 的 inner0/1/4 全输出和 hot-CSR sample golden。所有 NPU 调用均在毫秒级
完成，长耗时来自 H64/G16 的 CPU golden，没有触发 180 秒单算子 watchdog。

生产 topK16 压力矩阵由 8 个隔离 worker 完成，日志为：

```text
/opt/rrh/code3/sparse_attention_test/prefill_remaining_worker_{0..7}_current_20260818.log
worker cases = 22 + 14 + 16 + 14 + 18 + 22 + 14 + 22 = 142
result = 142 / 142 PASS, max_abs_diff = 0.003906
```

1024K 总 Q 使用 `H=1/topK=1/D=128` 常量边界逐元素精确校验；该结果证明索引
上界可运行，不表示 H64/G16/topK16/D256 的 1024K 笛卡尔组合能放入 HBM。

## 7. 当前版本状态

动态 A2 batch-group 上限由提交 `0980f38b9` 引入；最后一个 kernel 行为提交
`1ff100bdd` 修复 D256/BF16 split reduction buffer 别名。测试提交 `7d10706ee`
和 `762999c4d` 补齐 D256/blockSize/GQA/KV 边界、动态 cap 与超时保护。
远端安装的 kernel 对象 hash 为
`aa352364400927e119f27eadc9dfefc004de225239082106d4ac55f2f10f0bfa`，与本轮 profile 中抓取的对象一致。

`batchGroupsMax=1U` 不再作为 `ElementS=float` 的固定限制。当前有效上限为：

```text
min(8, 64 / groupSize, 8192 / (RoundUp(blockSize, 16) * groupSize))
```

对 `groupSize=16, blockSize=128`，上限为 4；Q=16/32/64 分别覆盖 1/2/4 个 group，全部通过全量 CPU golden。恢复固定 1 会保留正确性但丢失已验证的流水并行度，不应作为正式实现。

本轮没有未验证的 Phase1/Phase2 源码改动混入性能结果；后续实验应重新编译、安装并用对象 hash 与 profile 对齐后再纳入报告。

## 8. 复现命令

以下命令省略具体主机、账号和安装路径，避免将环境凭据写入仓库。执行前将占位符替换为当前 A2 环境配置。

### 8.1 编译安装

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate rrh
source <CANN_TOOLKIT>/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=<CUSTOM_OPP_PATH>
unset ASCEND_RT_VISIBLE_DEVICES
export NPU_DEVICE=npu:7

cd <OPS_TRANSFORMER_ROOT>
bash build.sh --pkg --soc=ascend910b --ops=sparse_attention_score_prefill -j16
./build_out/cann-ops-transformer-custom_linux-aarch64.run \
  --quiet --install-path="${ASCEND_CUSTOM_OPP_PATH}"
```

### 8.2 精度回归

```bash
export CANN_OPS_TRANSFORMER_LAZY_IMPORT=1
cd <OPS_TRANSFORMER_ROOT>/attention/sparse_attention_score_prefill

TEST_FILE=tests/test_sparse_attention_score_prefill_npu.py

# 能力边界与三个 innerPrecise 模式
python -m pytest -q -s "${TEST_FILE}" -k \
  "softmax_lse or single_batch_medium or partial_last_block or \
   gqa16_uneven or group_size_sweep or head_dim_256 or max_heads_d256 or \
   paged_block_size_sweep or kv_seqlen_256k or kv256k_gqa16 or \
   total_q_1024k or axis0_strided or continuous_kv or \
   prod_inner_precise"

# 生产 H64/KVH4/G16/topK16 压力矩阵；建议按测试方法隔离进程执行
python -m pytest -q -s "${TEST_FILE}" -k "test_prod_"
```

### 8.3 性能采集

```bash
cd <OPS_TRANSFORMER_ROOT>/attention/sparse_attention_score_prefill

for INNER in 0 1 4; do
  for KV in 65536 131072 262144; do
    OUT=<PROFILE_OUTPUT_ROOT>/inner${INNER}_kv${KV}
    timeout -k 10s 180s msprof op --application="/root/miniconda3/envs/rrh/bin/python tests/run_sparse_attention_score_prefill_perf.py \
      --batch 1 \
      --q-seqlen 2048 \
      --kv-seqlen ${KV} \
      --q-heads 16 \
      --kv-heads 1 \
      --head-dim 128 \
      --block-size 128 \
      --top-k 16 \
      --inner-precise ${INNER} \
      --warmup 3 \
      --repeat 20 \
      --device npu:7" \
      --aic-metrics=BasicInfo \
      --launch-count=10 \
      --warm-up=5 \
      --kill=on \
      --replay-mode=kernel \
      --output="${OUT}"
  done
done
```

## 9. Profile 归档

| 版本 | 归档目录 |
|---|---|
| 原始基线 | `/opt/rrh/code3/sparse_attention_test/prefill_128k_before_lse_dma_npu1_20260814_0943` |
| LSE DMA | `/opt/rrh/code3/sparse_attention_test/prefill_128k_lse_dma_npu1_20260814_0950` |
| O partial DMA | `/opt/rrh/code3/sparse_attention_test/prefill_128k_opartial_dma_npu1_20260814_1000` |
| RowMuls | `/opt/rrh/code3/sparse_attention_test/prefill_128k_rowmuls_npu1_20260814_1005` |
| no invRowSum | `/opt/rrh/code3/sparse_attention_test/prefill_128k_no_inv_npu1_20260814_1048a` |
| Phase1 fusion 独立实验 | `/opt/rrh/code3/sparse_attention_test/prefill_128k_p1_fused_npu1_20260814_1127` |
| Phase2 full AIV | `/opt/rrh/code3/sparse_attention_test/prefill_128k_phase2_full_aiv_npu1_20260814_1133` |
| Phase2 Brcb | `/opt/rrh/code3/sparse_attention_test/prefill_128k_phase2_brcb_npu1_20260814_1138` |
| Phase2 Vector Div | `/opt/rrh/code3/sparse_attention_test/prefill_128k_phase2_vecdiv_npu1_20260814_1146` |
| 当前 64K inner0 | `/opt/rrh/code3/sparse_attention_test/prefill_current_64k_inner0_npu7_fresh_20260818/OPPROF_20260818024451_PMYNLNAZJVLRRQTK` |
| 当前 128K inner0 | `/opt/rrh/code3/sparse_attention_test/prefill_current_128k_inner0_npu7_fresh_20260818/OPPROF_20260818024304_XXLVCHRPFGHXCXIG` |
| 当前 256K inner0 | `/opt/rrh/code3/sparse_attention_test/prefill_current_256k_inner0_npu7_fresh_20260818/OPPROF_20260818024529_ZHGSOCZWYNOJJZBU` |
| 当前 64K inner1 | `/opt/rrh/code3/sparse_attention_test/prefill_current_64k_inner1_npu7_fresh_20260818_030021/OPPROF_20260818030035_IJVLFCTEUPKJBNSI` |
| 当前 128K inner1 | `/opt/rrh/code3/sparse_attention_test/prefill_current_128k_inner1_npu7_fresh_20260818_030021/OPPROF_20260818030151_TKTXNZPSBCYSSJMH` |
| 当前 256K inner1 | `/opt/rrh/code3/sparse_attention_test/prefill_current_256k_inner1_npu7_fresh_20260818_030021/OPPROF_20260818030225_KSZFWKABMTWSZHWS` |
| 当前 64K inner4 | `/opt/rrh/code3/sparse_attention_test/prefill_current_64k_inner4_npu7_fresh_20260818_030021/OPPROF_20260818030323_SZTXTKFVIIQDVWYH` |
| 当前 128K inner4 | `/opt/rrh/code3/sparse_attention_test/prefill_current_128k_inner4_npu7_fresh_20260818_030021/OPPROF_20260818030356_RSNXBDKHCWWNDVIW` |
| 当前 256K inner4 | `/opt/rrh/code3/sparse_attention_test/prefill_current_256k_inner4_npu7_fresh_20260818_030021/OPPROF_20260818030430_LHTSJBMODFHQAJHK` |
| Score 64K inner0 | `/opt/rrh/code3/sparse_attention_test/score_current_64k_inner0_npu7_fresh_20260818/OPPROF_20260818024752_TUIHYSSZJFEWCHDV` |
| Score 128K inner0 | `/opt/rrh/code3/sparse_attention_test/score_current_128k_inner0_npu7_fresh_20260818/OPPROF_20260818025122_DTMCFSZTDVXSSFRW` |
| Score 256K inner0 | `/opt/rrh/code3/sparse_attention_test/score_current_256k_inner0_npu7_fresh_20260818/OPPROF_20260818024837_NJXSGLYXZVRQTJBP` |

## 10. 2026-08-18 inner0 性能门禁

以下数据均为物理 Device 7、`Current Freq=Rated Freq=1800 MHz`、Block/Mix=20/40，取同一次
`msprof op --application` 采集的 10 次 Task Duration 中位数。Prefill 使用 0708 toolkit，Score
基线使用 0805-2 toolkit；两者 shape、seed 和 topK 构造一致，因此结果是跨 toolkit 的同形 A/B，不能替代同一软件栈的绝对基准。

| KV | Prefill key | Prefill (us) | Score key | Score (us) | Prefill 降幅 | 20% 门槛 (us) | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64K | 20003 | 1987.620 | 20002 | 2663.083 | 25.364% | 2130.467 | PASS |
| 128K | 20003 | 2008.830 | 20002 | 2589.002 | 22.409% | 2071.201 | PASS |
| 256K | 20003 | 2036.261 | 20002 | 2602.212 | 21.749% | 2081.770 | PASS |

Prefill 三档使用同一对象 hash `aa352364400927e119f27eadc9dfefc004de225239082106d4ac55f2f10f0bfa`；
Score 三档对象 hash 为 `27dddbb4c0ccadb3fdba09535093a1acb7596bc207ac622ff6a664fc393732c4`。
三档均满足 `prefill <= 0.8 * score`。Python/NPU Event 仅用于确认进程完成，不作为性能门禁。

三档 speedup 的几何平均为 `1.302x`，对应几何平均耗时降低 `23.190%`。

## 11. 2026-08-18 全精度模式性能矩阵

下表均取每份 `msprof op` 中 10 次 Task Duration 的中位数。九组 Prefill profile
使用相同安装对象，均为 `mix`、Block/Mix=20/40、Device 7、1800/1800 MHz；另行
执行的同 shape 探针确认输出为 BF16 `(2048,16,128)` 且全部 finite。

| KV | inner0 / key20003 (us) | inner1 / key20002 (us) | inner4 / key20001 (us) |
|---:|---:|---:|---:|
| 64K | 1987.620 | **1953.449** | 2001.200 |
| 128K | 2008.830 | **1967.219** | 2004.550 |
| 256K | 2036.261 | **1993.600** | 2036.661 |
| 三点几何均值 | 2010.805 | **1971.353** | 2014.074 |

`innerPrecise=1` 的 BF16 O_partial 减少了 Workspace 搬运，在三档中比 inner4
快 `2.386%/1.862%/2.114%`。收益只有约 2%，说明当前 kernel 仍主要受固定控制、
同步和标量发射限制，而不是单纯受 O_partial 字节数限制。

### 11.1 原始 10-launch Task Duration

以下单位均为 us；顺序不参与中位数计算：

```text
Prefill inner0 64K : 1966.859375, 1968.379395, 1971.199463, 1975.739502, 1987.259766, 1987.979736, 1988.099731, 1989.219727, 1992.719849, 1994.979858
Prefill inner0 128K: 1996.019897, 2001.720093, 2006.740112, 2006.780151, 2008.280151, 2009.380127, 2016.040283, 2017.140381, 2026.880493, 2027.220581
Prefill inner0 256K: 2009.320190, 2010.520264, 2023.380493, 2033.320679, 2036.160767, 2036.360718, 2036.460693, 2040.120850, 2042.000854, 2046.860962
Prefill inner1 64K : 1956.099121, 1957.719116, 1924.758545, 1958.279175, 1950.598999, 1956.259155, 1938.138794, 1950.799072, 1976.939575, 1932.698608
Prefill inner1 128K: 1971.439453, 1949.639038, 1971.899414, 1962.199219, 1966.499390, 1972.079468, 1965.839355, 1967.939331, 1948.739014, 1977.899536
Prefill inner1 256K: 1993.319824, 1979.939575, 2011.060181, 2009.860229, 2007.240112, 2008.600220, 1993.879883, 1992.199829, 1983.679688, 1992.339844
Prefill inner4 64K : 2017.600342, 2001.619995, 2009.380127, 1966.599365, 1999.739990, 2011.940186, 2000.780029, 1990.319824, 2008.600220, 1997.719971
Prefill inner4 128K: 2014.260254, 2003.120117, 2010.040161, 2005.680054, 2018.360352, 1992.059814, 2025.020508, 1995.559937, 1996.319946, 2003.420044
Prefill inner4 256K: 2054.941162, 2023.320435, 2045.020874, 2030.140625, 2021.580444, 2064.421387, 2024.220459, 2038.560791, 2054.000977, 2034.760742
Score inner0 64K   : 2577.771484, 2665.973389, 2661.853271, 2662.813232, 2663.073242, 2655.973145, 2665.453369, 2667.213379, 2665.813232, 2663.093262
Score inner0 128K  : 2599.951904, 2597.211914, 2584.231689, 2582.331543, 2588.171875, 2589.831787, 2592.951904, 2587.951660, 2585.611816, 2590.231689
Score inner0 256K  : 2590.731934, 2604.472168, 2598.251953, 2606.212158, 2604.512207, 2601.612061, 2602.031982, 2603.511963, 2602.392090, 2596.111816
```

## 12. 后续建议

1. 如继续尝试 Phase1 nnz 负载均衡，必须先做 q16/q128/q2048 精度回归，再用同口径 profile 与本报告基线比较；只有实测收益为正才保留。
2. 统一所有性能版本的 `msprof` launch-count 和 warmup 采样口径，至少重复 3 轮，使用 median 作为最终门禁。
3. 对 Phase1 的 core8 长尾做专门 profile；若负载均衡收益不足，继续减少 QK/softmax/PV pipeline 的标量控制和同步等待。
4. Phase2 后续优化优先考虑 task 级 MTE2/Vector overlap；继续做单个 scalar loop 的微优化预计收益有限。
5. 保持 decode 算子不参与 performance runner，避免把另一算子的耗时和同步行为混入 prefill 结果。
6. 继续降低 scalar 耗时，优先减少剩余的 `GetValue`/`SetValue`。
7. 结合实际 UB 账本评估扩大批处理；任何 UB/GM 换性能方案都需单独做精度和 profile A/B。

## 13. 最终验收目标

1. 性能 shape 固定为 B1/Q2048/H16/KVH1/D128/block128/topK16，KV 覆盖
   64K/128K/256K，三个 `innerPrecise` 模式均以 `msprof op` 采集。
2. `innerPrecise=0` 逐档对比独立环境中的同形 `SparseAttentionScore`，要求 Prefill
   Task Duration 至少降低 20%；当前三档全部 PASS。
3. 泛化范围 D128/256、blockSize 0 或 16..128、H<=64、G<=16、总 Q<=1024K、
   KV<=256K 以精度/功能矩阵验收，性能矩阵不宣称覆盖这些上界的笛卡尔积。
4. Decode 算子不进入 Prefill performance runner，也不作为本轮性能基线。
