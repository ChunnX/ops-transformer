# Performance Runbook

本 runbook 面向后续单算子性能采集。完整静态分析见 [../performance-test-flow.md](../performance-test-flow.md)。

## Default Flow

当前 `ops-transformer` 没有统一 `--perf` 入口，默认通过 eager example 加 `msprof op` 采集。

```bash
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16 -O3
./build_out/cann-ops-transformer-custom_linux-aarch64.run
bash build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b
cd build
msprof op --application=./test_aclnn_<op_name> --output=../artifacts/<op_name>/msprof/<tag>
```

如果 `msprof op --application` 在当前 CANN 版本不可用，可按文档使用：

```bash
msprof op ./test_aclnn_<op_name>
```

完整方法见 [../performance-optimization-method.md](../performance-optimization-method.md)。本 runbook 只记录可直接执行的采集步骤。

## Preflight Gate

profiling 前先确认三个 gate：

1. 候选 gate：目标算子在主线 AscendC AI Core 范围，且本轮输入矩阵已固定。
2. 精度 gate：可用 UT、pytest、example golden 或 `scripts/st_accuracy` selected case 已正常通过。
3. package/load gate：运行时加载的是当前 `custom_transformer`，不是 CANN 原始实现或旧 custom 产物。

建议记录：

```bash
pwd
git rev-parse HEAD || true
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
sha256sum build_out/cann-ops-transformer-custom_linux-aarch64.run 2>/dev/null || true
sha256sum /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so 2>/dev/null || true
find /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer -path '*ascend910b*' -type f | rg '<op_name>|kernel|config'
```

如果缺少 `custom_transformer` op_api、目标 kernel 或 config，先修 package/install，不要把 profiling 失败写成算子性能结论。

## Minimum Matrix

每个候选优化至少覆盖：

- 目标大 shape。
- 常见中等 shape。
- 小 shape。
- 边界 shape。
- 相关 dtype。
- 关键 attrs 或分支条件。

Transformer 额外维度：

- attention：batch、seq length、head num、head dim、KV cache、mask、layout、quant。
- GMM/MoE：expert 数、group 分布、token 分布、routing/topk 参数。
- MC2：rank、world size、HCCL、拓扑、通信/计算 overlap。

## Sampling Rule

- 每个 case 至少采集 3 次。
- 默认用 median。
- 记录所有样本，至少记录 baseline/candidate 的样本数组。
- 低于约 1% 的变化默认按噪声处理。
- baseline 和 candidate 应在同一时间窗口采集。若间隔较长或机器状态可能漂移，补跑同窗口 baseline。

## Metrics

优先记录：

- kernel name。
- `Task Duration(us)`。
- `Block Dim`。
- tiling key。
- total device time。
- ArithmeticUtilization 摘要。

计算：

```text
kernel_gain = (baseline_kernel_us - candidate_kernel_us) / baseline_kernel_us * 100%
total_gain = (baseline_total_us - candidate_total_us) / baseline_total_us * 100%
```

如果一个 aclnn 调用包含多个 runtime task，必须说明只看单 kernel 还是看 total device time。

## Launch Count And Case Mapping

使用 batch runner 或 wrapper 作为 profiling application 时，`msprof op --launch-count` 必须覆盖真实 device task 数。`case_count` 只在每个 case 正好启动一个 device task 时成立；组合 ACLNN 调用可能在一个 case 中启动多个 runtime task。

case 映射规则：

- 记录每个 case 的 shape、dtype、format、attrs、预期 kernel name 或 tiling key。
- 如果 batch 内混合多个 kernel name、suffix 或 tiling key，不要只按 CSV 文件名、timestamp 或行顺序映射 case。
- 优先用 `msprof op` 控制台日志中的 kernel 启动顺序、kernel name、launch index 和输入矩阵重建映射。
- 做不到可靠映射时，改成每个 case 单独输出目录，再汇总矩阵。
- ACLNN 组合链需要同步记录目标 kernel time 和 total device time；只看目标 kernel 可能掩盖前后处理回退。

## Package Load Evidence

profiling 前后建议记录：

```bash
echo "$ASCEND_HOME_PATH"
echo "$ASCEND_OPP_PATH"
echo "$ASCEND_CUSTOM_OPP_PATH"
find "$ASCEND_OPP_PATH/vendors" -maxdepth 3 -type f | rg '<op_name>|cust_opapi|transformer'
```

如果使用 custom package：

- 记录 `.run` 包路径。
- 记录安装命令。
- 记录 vendor 目录。
- 记录目标 kernel/config 是否存在。
- 记录 `libcust_opapi.so` hash。
- 如使用 `--opkernel` 快速路径，记录手动同步到 vendor 目录的 binary/config 路径和 hash；否则不能把结果当作运行时新产物。

## Simulator

simulator profiling 适合分析 pipeline，不等于真实硬件性能：

```bash
msprof op simulator --output=$PWD/pipeline_auto --kernel-name "<KernelName>" ./test_aclnn_<op_name>
```

Ascend 950 相关 `cannsim` 路径不作为当前 B3 默认流程。

## After Each Loop

每轮实验后记录：

```text
operator:
hypothesis:
changed files:
build command:
install/sync command:
accuracy command:
profiling command:
matrix:
decision:
raw evidence path:
retention action:
```

如果后续继续建立或更新 `agent_ops/<op>/`，把以上内容写入对应算子的 README/manifest，不要塞回 `AGENTS.md`。

最小矩阵表建议：

```text
case_id | shape | dtype | attrs | kernel | tiling_key | baseline_samples_us | candidate_samples_us | median_gain | total_gain | decision
```
