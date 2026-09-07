# MoeEp cycle / msprof 分析工具

本目录配套 `MoeEpDispatch`、`MoeEpDispatchEpilogue`、`MoeEpCombine`、
`MoeEpCombineEpilogue` 的两种互斥采集方式：

- cycle 模式记录 kernel 内函数边界，保留 `uint64` 原始 cycle，并生成函数统计和逻辑流水图；
- msprof 模式保留硬件 CSV/`visualize_data.bin`，生成 AIV/AIC 的 MTE2、MTE3、Vector、Scalar 等真实流水利用率报告。

本轮仅在 **DispatchEpilogue 与 Combine** 两个 kernel 接入 marker。公共 context ABI、四个
ring 和四算子历史解析映射沿用 flow；Dispatch/CombineEpilogue 即使实际执行，未插点的
两个 raw 也仍为空，这是预期，不应为了填满文件自动修改这两个业务 kernel。

## 构建与 cycle 采集

```bash
bash build.sh --pkg --moe-ep-profiler --module=mc2 --soc=ascend950 -j16
MOE_EP_CYCLE_PROFILER=1 python moe_train_test.py
```

测试入口补丁默认运行 continue 链路 105 轮、前 5 轮作为 warmup、统计最后 50 轮。
每个 rank 在唯一 `run_id` 下调用一次
`ElasticBuffer.enable_cycle_profiler(4096)`，全部 kernel 完成并做唯一一次末尾 NPU 同步后，调用
`dump_cycle_profiler(run_dir, rank)`。循环内不会调用 torch profiler、`prof.step()`、NPU
同步、大 `torch.exp` 或额外 collective，也不会增加额外 kernel launch。

打点的被测区间只执行 `GetSystemCycle` 和本地记录写入；GM ring 写入和 DCCI flush 在
`KERNEL_END` 之后。打点指令本身仍有不可消除的少量 device 开销，因此不能解释为严格零开销。
`KERNEL` 只覆盖 Process 的 BEGIN→END，调用前 Init（包括 Combine 的 rank offsets
读取）不在计时范围。不要把该 duration 等同于完整 kernel launch 耗时。
末尾 profiler Flush 仍会延长真实 launch、改变 continue 时序和 Combine 提前退出收益；
生产性能与稳定性需普通 OFF 构建对照，不能用 profiling 跑通证明 race 已修复。

同 group 只允许一个已 enable profiler 的 `ElasticBuffer`；两个实例均未 enable 时保留
原有共享行为，避免实例间覆盖或释放正在使用的 ring。外部测试脚本只提供参考补丁，
须按当前脚本语义移植；自动解析参考命令已加入 `--frequency-mhz 1000`。

手工解析：

```bash
python mc2/tools/moe_ep_profiler/analyze_cycle.py /path/to/run \
  --warmup 5 --tail-count 50 --frequency-mhz 1000
```

沿用 flow 历史记录的 Ascend 950 环境，`GetSystemCycle()` tick 为 1000 MHz，因此 cycle raw
应显式传 `--frequency-mhz 1000`，即 `cycles / 1000 = us`。该参数优先于所有自动探测；
本轮未上板重新校准，迁移到其他型号或环境后需重新确认 tick 频率。

省略显式参数时，如果本次 msprof 输出不在 run 目录下，可通过
`--frequency-search-root /path/to/this-run-msprof` 指定。本工具按以下顺序自动解析频率并在
`analysis_manifest.json` 中记录来源：

1. 本次 `OpBasicInfo.csv` 的 `Current Freq`；
2. CANN `platform_config*.ini` 的 `[VectorCoreSpec] vec_freq`；
3. A5 fallback 1650 MHz。

主要输出：

- `events.csv`：每个原始 begin/end 事件及绝对 cycle；
- `intervals.csv`：配对区间；
- `function_summary.csv`：warmup 后统计；
- `tail_summary.csv`：后 N 轮统计；
- `functions_rankNNN.trace.json`、`function_summary.svg`：函数级 trace/图；
- `logical_pipeline_rankNNN.trace.json`、`logical_pipeline_rankNNN.svg`：相对
  `KERNEL_BEGIN` 的 p50 逻辑流水；
- `core_status.csv`、`analysis_manifest.json`：ring wrap/drop、ABI、频率和告警。

四个 raw buffer 总会落盘；未插点或从未执行的算子保持全零，解析时会跳过并写入
`unused_raw_files`。一轮内每个 core 最多暂存 32 个 marker，报告中出现 local-drop 时该轮数据不完整。

不同 rank 的 `GetSystemCycle` 不保证同一 epoch。所有 trace 按 rank 独立归零或按本 rank
kernel 起点归一化，不可用来声称跨 rank 的绝对时间对齐。

## msprof 采集

通用入口：

```bash
python mc2/tools/moe_ep_profiler/run_msprof.py \
  --output-root ./moe_ep_msprof_runs --warm-up 0 --launch-count 1 -- \
  python /path/to/moe_train_test.py
```

多 rank 脚本已有内部 warmup/迭代时，保持一次外层 launch，避免把整个多进程入口重复启动。

外部测试脚本补丁还支持一次入口自动重启：

```bash
MOE_EP_RUN_MSPROF=1 OPS_TRANSFORMER_ROOT=/path/to/ops-transformer python moe_train_test.py
```

msprof 子进程会自动关闭 cycle marker，避免两类 instrumentation 叠加。输出保留 8 个硬件
CSV 和 `visualize_data.bin` 到 `raw_msprof/`，并生成 `pipeline_metrics.csv`、
`msprof_summary.csv`、`msprof_pipeline.svg`、`msprof_report.md`。`SendPhase` 的函数边界来自
cycle 数据；msprof 指标是 kernel/core 粒度，报告不会把它伪装成 SendPhase 独占计数。

## 测试

```bash
python -m unittest discover -s mc2/tools/moe_ep_profiler/tests -v
python mc2/tools/moe_ep_profiler/validate_instrumentation.py \
  --ops moe_ep_dispatch_epilogue moe_ep_combine --base-ref HEAD
```

第二条仅为插点前 HEAD 与工作区的只读结构/业务 token 检查；插点已提交时改用真实基线。
它不证明动态分支/循环 marker 预算、流水依赖或 CANN 编译正确；默认只检查本轮两算子。
将来扩展到四 kernel 时显式传全部四个 `--ops`，各路径 marker≤32 仍需人工审阅及 raw drop 检查。

## 接入与分支迁移

重新接入打点、更新 kernel 阶段或切换分支时，参见
[INTEGRATION_AND_MIGRATION.md](INTEGRATION_AND_MIGRATION.md)。
