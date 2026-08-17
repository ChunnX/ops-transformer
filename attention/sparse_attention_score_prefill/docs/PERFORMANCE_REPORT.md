# SparseAttentionScorePrefill A2 性能优化报告

更新时间：2026-08-14

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
分别不超过 `2.44e-4`、`1.22e-4`、`1.22e-4`。生产 GQA 热 CSR（Q=16、Q=128）三个
`innerPrecise=0/1/4` 子路径也通过，说明放宽到容量上限后同步和 ring 生命周期没有回归。

同形 B4/Q6144/KV8192/inner0 的 Score 基线为 `30779.574 us`（key 20002）；当前
Prefill 正确 CSR 的 Python event 约 `23.1 ms`，但该大 workspace 在共享卡上触发了
msprof memory snapshot 失败，因此不能把 event 数字当作正式 Task Duration。B1/Q6144
的有效 msprof profile 为 `5596.772 us`（key 20003），AIV scalar/control 仍是主要
瓶颈。完整 profile 归档和精度日志见本报告后续章节。

## 1. 结论

在 Ascend 910B3 单卡、物理 Device 1、固定频率 1800 MHz 下，目标 128K prefill shape 的已验证实现由 `3688.554 us` 降至 `2648.453 us`，耗时降低 `28.198%`，等价加速 `1.393x`。

最新结果已经超过“相对原始性能基线降低 20%”的目标线 `2950.843 us`。如果把已经包含 `RowMuls` 和去除冗余 `invRowSum` 的 `2665.973 us` 版本视为新的优化起点，则后续 Phase2 微优化累计降低 `17.520 us`，收益为 `0.657%`；以该版本为基线的 20% 目标线是 `2132.779 us`，目前尚未达到。

当前性能瓶颈仍在 AIV，最新 profile 中 AIV 平均执行约 `2638.738 us`，其中 scalar 约 `1951.734 us`，占约 `74%`。AIC 平均执行约 `1103.710 us`，不是关键路径。

## 2. 测试范围

| 项目 | 配置 |
|---|---|
| SoC | Ascend 910B3, A2/arch22 |
| 设备 | 物理 Device 1；通过可见设备映射为进程内 `npu:0` |
| 频率 | Current Freq = Rated Freq = 1800 MHz |
| Batch | 1 |
| Q Seqlen | 2048 |
| KV Seqlen | 131072 |
| Q Heads | 16 |
| KV Heads | 1 |
| Head Dim | 128 |
| Block Size | 128 |
| TopK | 16 |
| 输入/输出 dtype | BF16 |
| `innerPrecise` | 4，O partial 使用 FP32 |
| Tiling key | 20001 |
| BlockDim / MixBlockDim | 20 / 40 |

性能入口为 [run_sparse_attention_score_prefill_perf.py](../tests/run_sparse_attention_score_prefill_perf.py)，只构造输入并调用 prefill 算子，不调用 decode，也不运行 CPU golden。性能结论以 `msprof` 的 kernel Task Duration 为准，Python/NPU Event 计时只用于快速观察。

## 3. 性能结果

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

## 4. Profile 热点

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

最新已 profile 的 Vector `Div` 版本完成以下 arch22 定向回归：

| 用例 | 关注点 | 结果 |
|---|---|---|
| `test_arch22_bf16_q16_hot_csr_precision` | 最小 hot-CSR、多批 pipeline 边界 | PASS |
| `test_arch22_bf16_q128_hot_csr_precision` | 单 hot row 跨多个 batch | PASS |
| `test_arch22_bf16_q2048_kv133120_precision_and_perf` | 生产规模 Q，首/中/尾 token CPU sample golden | PASS |

这些用例对完整输出执行 NaN、Inf 和全零检查，并对选定 token 执行 CPU 两阶段 golden 比较，容差为 `1.5e-2`。Q=2048 用例比较 token `0/1024/2047`，不是全量 2048 token CPU golden；因此报告中的 PASS 表示当前定向回归通过，不等价于全 shape 全元素 exhaustively verified。

精度入口为 [test_sparse_attention_score_prefill_npu.py](../tests/test_sparse_attention_score_prefill_npu.py)。

## 7. 待验证改动

当前工作区还包含 Phase1 基于 CSR nnz 的连续负载均衡实验 `ComputeCoreBounds`。目标 CSR 在固定模 20 分核时，最忙 core 处理约 2048 个 QK pair，普通 core 约 1536 个；静态估算平衡后最大值可降到约 1640。

当前工作区源码因此不再与 `2648.453 us` 对应的 Vector `Div` profile 二进制完全一致。直接重新编译当前工作区会得到“Vector Div + Phase1 nnz 负载均衡”的组合版本，而不是报告中的已验证最佳版本。主要已验证优化已提交在 `cbd2f5927`；后续 Phase2 改动和 Phase1 实验仍处于未提交状态。

该改动尚未完成重新编译、安装、精度回归和 `msprof` 采集，因此：

- 不计入 `2648.453 us` 当前最佳结果；
- 不作为已完成性能收益；
- 在 q16/q128/q2048 精度通过并得到同口径 profile 前，不应提交或合入性能结论。

## 8. 复现命令

以下命令省略具体主机、账号和安装路径，避免将环境凭据写入仓库。执行前将占位符替换为当前 A2 环境配置。

### 8.1 编译安装

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate rrh
source <CANN_TOOLKIT>/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=<CUSTOM_OPP_PATH>
export ASCEND_RT_VISIBLE_DEVICES=1

cd <OPS_TRANSFORMER_ROOT>
bash build.sh --pkg --soc=ascend910b --ops=sparse_attention_score_prefill -j16
./build_out/cann-ops-transformer-custom_linux-aarch64.run \
  --quiet --install-path="${ASCEND_CUSTOM_OPP_PATH}"
```

### 8.2 精度回归

```bash
export CANN_OPS_TRANSFORMER_LAZY_IMPORT=1
cd <OPS_TRANSFORMER_ROOT>/attention/sparse_attention_score_prefill

python -m pytest -q -s \
  tests/test_sparse_attention_score_prefill_npu.py::TestSparseAttentionScorePrefillNPU::test_arch22_bf16_q16_hot_csr_precision \
  tests/test_sparse_attention_score_prefill_npu.py::TestSparseAttentionScorePrefillNPU::test_arch22_bf16_q128_hot_csr_precision \
  tests/test_sparse_attention_score_prefill_npu.py::TestSparseAttentionScorePrefillNPU::test_arch22_bf16_q2048_kv133120_precision_and_perf
```

### 8.3 性能采集

```bash
cd <OPS_TRANSFORMER_ROOT>/attention/sparse_attention_score_prefill
OUT=<PROFILE_OUTPUT_DIR>

msprof --output="${OUT}" \
  --aic-metrics=PipeUtilization \
  --task-time=on \
  python tests/run_sparse_attention_score_prefill_perf.py \
    --batch 1 \
    --q-seqlen 2048 \
    --kv-seqlen 131072 \
    --q-heads 16 \
    --kv-heads 1 \
    --head-dim 128 \
    --block-size 128 \
    --top-k 16 \
    --inner-precise 4 \
    --warmup 3 \
    --repeat 1 \
    --device npu:0
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

## 10. 后续建议

1. 完成 Phase1 nnz 负载均衡的编译、q16/q128/q2048 精度回归和同口径 profile；只有实测收益为正才保留。
2. 统一所有性能版本的 `msprof` launch-count 和 warmup 采样口径，至少重复 3 轮，使用 median 作为最终门禁。
3. 对 Phase1 的 core8 长尾做专门 profile；若负载均衡收益不足，继续减少 QK/softmax/PV pipeline 的标量控制和同步等待。
4. Phase2 后续优化优先考虑 task 级 MTE2/Vector overlap；继续做单个 scalar loop 的微优化预计收益有限。
5. 保持 decode 算子不参与 performance runner，避免把另一算子的耗时和同步行为混入 prefill 结果。
6. 降低scalar耗时，减少GetValue、SetValue使用
7. UB使用率不高，是否可以使用更多UB或者GM换性能
## 11. 目标
1. 给定shape，batch=1，qSeqlen=2048，numHeads=16，kvHeads=1，headSize=128，blockSize=128，topK=16，kvseq=128k进行性能优化，
2. 性能测试使用msprof op采集，测试脚本为[run_sparse_attention_score_prefill_perf.py]
3. 修改后不需要手动同步，会自动同步到环境上
4. 目标性能时2660us，对应decode算子性能数据再/opt/rrh/code3/sparse_attention_test/OPPROF_20260813161936_WWNMVREXAVBYEYUJ目录下，性能要求再提升20%以上
5. 环境信息见\code3\ops-transformer\attention\sparse_attention_score_prefill\docs\env.md
