# MoE EP Device-Side Profiler 接入与迁移指南

本文用于在 `flow` 更新、切换分支或 kernel 重构后，快速恢复
`moe_ep_dispatch`、`moe_ep_dispatch_epilogue`、`moe_ep_combine` 和
`moe_ep_combine_epilogue` 的 device-side cycle 打点。

日常构建、运行和解析命令见 [README.md](README.md)。业务流水及 marker 选点依据见
`mc2/docs/moe_ep_pipeline.md`；该目录可能是本地未跟踪内容，迁移 profiler 时不要覆盖它。

## 1. 当前实现的计时语义

- 仅分析非 Hybrid 模板；Dispatch Hybrid 不插 marker。
- 四个 kernel 当前都按 AIV 物理角色记录。`MTE2`、`MTE3`、`Vector`、`Scalar`、
  `URMA`、`Wait` 是逻辑事件标签，不代表 `GetSystemCycle()` 能直接区分真实硬件流水。
- `Mark()` 只在核内执行 `GetSystemCycle()` 并写本地固定数组，不在每个 marker 处写 GM、
  不做 host-device 同步。
- 每次 kernel invocation、每个核最多保存 32 条 marker。全部 GM ring 写入与 DCCI
  集中在 `KERNEL_END` 之后执行，因此这些批量写回开销不计入 `KERNEL` 区间；中间
  marker 的 `GetSystemCycle()` 和本地数组写入仍有少量不可消除的 device 开销。
- 本次目标 Ascend 950 环境已确认 `GetSystemCycle()` tick 为 1000 MHz，因此按下式换算：

  ```text
  duration_cycles = end_cycle - begin_cycle
  duration_us     = duration_cycles / 1000
  ```

  解析时显式传 `--frequency-mhz 1000`，不要让 AIV `Current Freq` 或 `vec_freq`
  覆盖 system-cycle 的换算频率。迁移到其他型号或环境后必须重新确认 tick 频率。
- 不同 rank 的 cycle epoch 不保证同步。跨 rank 只能比较 duration，或以各 rank 自身
  `KERNEL_BEGIN` 为相对起点，不能直接拼接绝对时间戳。

## 2. 数据链路与必须成组迁移的 ABI

当前方案不经过 tiling，也不是 workspace/address-bridge 方案。完整链路是：

```text
torch extension 分配四个 persistent NPU ring
  -> 将四个地址写入 host MoeCommContext 尾部
  -> 刷新 HCCL 可能复用的 cached device context 尾部
  -> kernel Init 从 device MoeCommContext 取得本算子 ring
  -> 每核 Mark() 写 local records
  -> KERNEL_END 后 Flush() 批量写 GM ring + DCCI
  -> Python 在最终 NPU synchronize 后拷回并原子写 .profbin
  -> analyze_cycle.py 恢复 invocation、配对区间、统计并出图
```

以下内容必须一起移植，否则容易出现 ABI 错配或全零 buffer：

1. Device 侧 context 尾部：
   [`../../common/op_kernel/mc2_moe_context.h`](../../common/op_kernel/mc2_moe_context.h)
2. Host 侧相同布局及 static assertion：
   [`../../common/torch_extension/csrc/elastic_buffer.cpp`](../../common/torch_extension/csrc/elastic_buffer.cpp)
3. 四个 persistent ring 的分配、context 地址注入和 cached-context tail refresh。
4. PyBind 的 enable/dump 接口与 Python 包装：
   [`../../common/torch_extension/elastic_buffer.py`](../../common/torch_extension/elastic_buffer.py)
5. Device profiler 实现：
   [`../../common/op_kernel/moe_ep_profiler.h`](../../common/op_kernel/moe_ep_profiler.h)

注意：HCCL 可能按 tag 复用旧 context。新建/获取 context 后必须保留 profiler tail 的
显式刷新，否则 kernel 可能继续使用旧 ring 地址。

## 3. 构建开关传播链

profiling 包必须同时编译 kernel 和 torch extension。完整链路为：

```text
build.sh --pkg --moe-ep-profiler
  -> 顶层 ENABLE_MOE_EP_PROFILING=ON
  -> cmake/config.cmake 将值传给 prepare.sh
  -> prepare.sh 将 -DENABLE_MOE_EP_PROFILING 传给子 CMake
  -> 四个 op_host/CMakeLists.txt 给 kernel 增加宏

同时：
build.sh 导出 CANN_OPS_MOE_EP_PROFILING_BUILD=1
  -> torch_extension/setup.py 替换 Python build literal
  -> lazy JIT 的 cxx_args 增加 -DENABLE_MOE_EP_PROFILING=1
  -> host/device MoeCommContext ABI 保持一致
```

关键文件：

- [`../../../build.sh`](../../../build.sh)
- [`../../../CMakeLists.txt`](../../../CMakeLists.txt)
- [`../../../cmake/config.cmake`](../../../cmake/config.cmake)
- [`../../../cmake/scripts/prepare.sh`](../../../cmake/scripts/prepare.sh)
- [`../../../torch_extension/setup.py`](../../../torch_extension/setup.py)
- 四个算子的 `op_host/CMakeLists.txt`

只替换已安装环境中的 `elastic_buffer.cpp/.py` 不足以开启 profiling。运行时环境变量也
不能改变已经编译完成的 kernel。必须重新构建并安装 profiling 算子包和对应 wheel，
重新 source vendor 环境，并清理会命中旧产物的 lazy-JIT cache。

推荐构建命令：

```bash
bash build.sh --pkg --moe-ep-profiler --module=mc2 --soc=ascend950 -j16
```

只构建四个目标算子：

```bash
bash build.sh --pkg --moe-ep-profiler --soc=ascend950 \
  --ops=moe_ep_dispatch,moe_ep_dispatch_epilogue,moe_ep_combine,moe_ep_combine_epilogue \
  -j16
```

## 4. Kernel 标准接入模板

每个非 Hybrid kernel 都按以下六个锚点检查：

1. include `moe_ep_profiler.h`；
2. 定义稳定的 stage enum；
3. 类成员加入 `MOE_EP_PROFILE_DECLARE()`；
4. `Init()` 从 `MoeCommContext` 选择本算子的 ring；
5. 在目标函数/阶段前后插入匹配的 BEGIN/END；
6. `KERNEL_END` marker 后调用 `MOE_EP_PROFILE_FLUSH()`。

伪代码：

```cpp
MOE_EP_PROFILE_MARK(MOE_EP_PROFILE_EVENT(
    MoeEpProfileOp::DISPATCH,
    MoeEpProfileEngine::VECTOR,
    STAGE_X,
    MoeEpProfileBoundary::BEGIN));
TargetFunction();
MOE_EP_PROFILE_MARK(MOE_EP_PROFILE_EVENT(
    MoeEpProfileOp::DISPATCH,
    MoeEpProfileEngine::VECTOR,
    STAGE_X,
    MoeEpProfileBoundary::END));

// 需要 payload 时使用 MOE_EP_PROFILE_MARK_PAYLOAD(..., payload)。
MOE_EP_PROFILE_MARK(MOE_EP_PROFILE_EVENT(
    MoeEpProfileOp::DISPATCH,
    MoeEpProfileEngine::SCALAR,
    STAGE_KERNEL,
    MoeEpProfileBoundary::END));
MOE_EP_PROFILE_FLUSH();
```

规则：

- BEGIN/END 必须覆盖同一语义区域，所有提前返回路径也必须成对。
- `Flush()` 不得放入被测区间。
- stage 数值一旦进入 raw 数据就保持稳定；新增 stage 只追加，不复用旧 ID。
- 修改 marker 后计算最坏路径条数，必须不超过 32。超限会设置 local-drop 标志。
- 深层 wait 可在真正轮询函数内部打点，避免把前后无关逻辑算入等待时间。
- 一个混合函数，例如 `SendPhase`，不能仅因外层标为 `MTE2` 就断言它只有 MTE2。
  函数内 MTE2/MTE3/Scalar 的真实并行需单独跑 msprof，或增加更细的逻辑子阶段。

事件 ID 位布局：

```text
[31:28] operator
[27:24] physical role，当前固定 AIV=1
[23:20] logical engine
[19:1]  stage
[0]     BEGIN/END
```

新增/重命名 stage 后同步更新
[`analyze_cycle.py`](analyze_cycle.py) 中的 `DEFAULT_STAGE_NAMES`。临时也可通过
`--event-map` 提供名字；未知 ID 会显示为 `STAGE_N`，不会阻止 raw 解析。

## 5. 四个算子的当前事件

| 算子 | Stage | Engine | 包围的行为 |
|---|---|---|---|
| Dispatch | `KERNEL` | Scalar | 整个非 Hybrid `Process()` |
| Dispatch | `BUFFER_INIT` | Scalar | `BufferInit()` |
| Dispatch | `COUNT_BUILD` | Vector | `CalSendCnt()` 和同步 |
| Dispatch | `COUNT_ISSUE` | URMA | count WQE 组装/下发 |
| Dispatch | `COUNT_WAIT` | Wait | 仅最后一个 AIV 执行的 `GetRecvCount()` 状态轮询 |
| Dispatch | `COUNT_AGGREGATE` | Vector | 仅最后一个 AIV 执行的接收计数聚合、状态清零 |
| Dispatch | `SLOT_PREFIX` | Vector | 非 cached 的 slot prefix |
| Dispatch | `PAYLOAD_PREP` | MTE2 | 当前包围整个 `SendPhase()` 和同步 |
| Dispatch | `PAYLOAD_ISSUE` | URMA | `WriteToRemoteWindow()` |
| DispatchEpilogue | `KERNEL` | Scalar | 整个 `Process()` |
| DispatchEpilogue | `PREFIX` | Scalar | 非 cached prefix sum |
| DispatchEpilogue | `PAYLOAD_WAIT` | Wait | 最后一个 AIV 的状态等待/清零 |
| DispatchEpilogue | `WAIT_BARRIER` | Wait | `SyncAll<true>()` |
| DispatchEpilogue | `COUNT_HITS` | Vector | `CountHits()` |
| DispatchEpilogue | `WINDOW_OUTPUT` | MTE2 | 从 window 搬到输出 |
| Combine | `KERNEL` | Scalar | 整个 `Process()` |
| Combine | `ADDRESS_TABLE` | Scalar | `BuildAddressTable()` |
| Combine | `DATA_WQE` | URMA | token/data WQE 组装 |
| Combine | `FLAG_COMMIT` | URMA | 完成标志和 prepared writes flush |
| CombineEpilogue | `KERNEL` | Scalar | 整个 `Process()` |
| CombineEpilogue | `BUFFER_INIT` | Scalar | `BuffInit()` |
| CombineEpilogue | `COMPLETION_WAIT` | Wait | AIV0 等待并清 completion flags |
| CombineEpilogue | `WAIT_BARRIER` | Wait | 全 AIV 同步 |
| CombineEpilogue | `REDUCE_OUTPUT` | Vector | token reduce 和输出 |

Kernel 文件：

- [`../../moe_ep_dispatch/op_kernel/moe_ep_dispatch.h`](../../moe_ep_dispatch/op_kernel/moe_ep_dispatch.h)
- [`../../moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h`](../../moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h)
- [`../../moe_ep_combine/op_kernel/moe_ep_combine.h`](../../moe_ep_combine/op_kernel/moe_ep_combine.h)
- [`../../moe_ep_combine_epilogue/op_kernel/moe_ep_combine_epilogue.h`](../../moe_ep_combine_epilogue/op_kernel/moe_ep_combine_epilogue.h)

## 6. Raw 文件和容量

每个 rank 固定写四个文件：

```text
<run>/raw/rank_NNN/
  moe_ep_dispatch.profbin
  moe_ep_dispatch_epilogue.profbin
  moe_ep_combine.profbin
  moe_ep_combine_epilogue.profbin
```

即使算子没有执行，也会生成全零文件。`unused zero-header profiler buffer` 对未执行算子
是正常 warning；如果四个目标算子明明执行过却全部为零，再检查编译开关、enable 时机、
wheel/kernel ABI 和 context 地址注入。

格式概要：

```text
Global header: 64 bytes
  magic/version/op/actual cores/max cores/slots/record bytes/core stride/flags

每核：
  Core control: 64 bytes
    write_count/write_pos/ring_wrapped/local_drop/dropped_local/flush_count
  Ring records: slots_per_core * 16 bytes
    event_id:uint32, payload:uint32, cycle:uint64
```

默认最大 128 核、4096 records/core。ring wrap 表示只保留最后一圈，并不一定是错误；
但用于 warmup/tail 的目标窗口必须仍在 ring 中。`local_drop_seen` 必须为 0。

## 7. 测试脚本接入

外部性能入口的参考补丁位于
[`patches/moe_train_test_cycle_profiler.patch`](patches/moe_train_test_cycle_profiler.patch)。
更新测试脚本时按语义移植，不要机械覆盖新版文件。

必须保持：

1. 开启 `is_continue` 和 `is_performance`，执行足够多轮；
2. 在首次 MoE EP op/context 创建之前调用 `enable_cycle_profiler()`；
3. cycle 模式绕过 `torch_npu.profiler`、`prof.step()` 和逐轮同步等干扰逻辑；
4. 所有循环结束后只做一次必要的 NPU synchronize；
5. 同步完成后 dump 四个 raw buffer；
6. 单机可自动解析，多机先各 rank dump，再集中解析；
7. 多机共享同一个 `MOE_EP_PROFILE_RUN_ID` 和可见的 output root；
8. 自动解析必须传 `--frequency-mhz 1000`。

当前参考补丁的自动解析尚未传第 8 项。手工解析已有 raw 无需重跑 NPU：

```bash
python mc2/tools/moe_ep_profiler/analyze_cycle.py /path/to/run \
  --warmup 5 --tail-count 50 --frequency-mhz 1000
```

## 8. 统计口径和图文件

### `function_summary.csv`

1. 在同一 `(rank, header_op, core_id)` lane 内按 ring 顺序扫描；
2. BEGIN/END 按 `(pair_id, invocation)` 用栈配对，不跨核、不中途跨 invocation；
3. 计算 `duration_cycles = end - begin`；
4. 删除 `invocation < warmup`；
5. 按 `(rank, op, role, engine, function)` 聚合。

分组键不含 `core_id` 和 `invocation`，所以一行会混合该 rank 的所有核和所有
post-warmup 轮次。输出 `count/min/p50/mean/p90/max`，p50/p90 使用线性插值；
所有 `*_us` 再除以 `frequency_mhz`。

条件执行、空路径或提前返回也会进入同一跨核样本集，所以跨核 p50 不一定代表活跃核
耗时。判断核间负载不均或条件路径时，应优先查看按核 trace，必要时生成 core summary。

### `tail_summary.csv`

先对每个 `(rank, op, core_id)` 独立选最后 `tail-count` 个 invocation，再使用与
`function_summary.csv` 相同的聚合键。

### 可视化

- `functions_rankNNN.trace.json`：真实区间时间线，`tid=AIV:<core_id>`，可按核查看。
- `logical_pipeline_rankNNN.trace.json/.svg`：所有核、所有入选 invocation 相对
  `KERNEL_BEGIN` 的 p50 逻辑流水，不是逐核图。
- `function_summary.svg`：当前实际使用 `tail_summary` 的 p50，虽然名字没有体现 tail。
- `core_status.csv`：按核记录 ring/write/flush/drop 状态，不是按核阶段耗时统计。

当前没有 `core_summary.csv/.svg`。若需要“每核 p50/p90 阶段耗时图”，应从
`intervals.csv` 按 `(rank, op, core_id, role, engine, function)` 重新聚合。

## 9. Cycle 与 msprof 分工

- Cycle marker：看函数/阶段的逻辑区间、条件等待、各核差异和多轮统计。
- msprof：看真实 MTE2/MTE3/Vector/Scalar 等硬件流水并行。
- 两种采集应分开运行。`run_msprof.py` 会强制关闭 cycle marker，避免 marker 影响硬件计数。

msprof 单次外层采集：

```bash
python mc2/tools/moe_ep_profiler/run_msprof.py \
  --output-root ./moe_ep_msprof_runs --frequency-mhz 1000 -- \
  python /path/to/moe_train_test.py
```

注意：`analyze_msprof.py` 中的 `--frequency-mhz` 用于报告/manifest 的统一记录；msprof
自身已提供的 `time_us` 不会被二次换算。真正执行 cycle-to-us 换算的是
`analyze_cycle.py`。

## 10. 更新代码或切分支后的迁移顺序

1. 记录当前 profiler 基线提交，先解决业务代码冲突，不要边 rebase 边重新设计 stage。
2. 移植共享 `moe_ep_profiler.h`。
3. 同时移植 device/host 两份 `MoeCommContext` 尾部和 static assertions。
4. 移植四 ring 分配、context 注入、cached-context tail refresh、PyBind、Python enable/dump。
5. 移植顶层 option、`build.sh` 参数、`config.cmake -> prepare.sh -> 子 CMake` 传播链。
6. 移植 `setup.py` build literal 和 lazy-JIT C++ 宏，确保 wheel 与 kernel ABI 一致。
7. 给四个 op 的 CMake 恢复 kernel compile definition。
8. 逐个 kernel 按六锚点重新定位 marker；确认 Dispatch Hybrid 仍跳过。
9. 保持旧 stage ID，追加新事件并更新 parser 名字映射。
10. 按新版测试脚本语义移植 enable、末尾 sync/dump 和自动分析；传入 1000 MHz。
11. 编译安装 profiling 包/wheel，重新 source 环境并清理旧 JIT cache。
12. 先跑小规模 continue 用例，再跑完整性能轮次；cycle 与 msprof 分两次采集。

## 11. 验收清单

- [ ] `--moe-ep-profiler` 从顶层传到 prepare 子 CMake，配置日志显示为 ON。
- [ ] 四个 op 的 kernel 编译命令含 `ENABLE_MOE_EP_PROFILING=1`。
- [ ] torch extension 与 kernel 使用相同的 `MoeCommContext` 大小和尾部布局。
- [ ] profiling build 的 Python 接口可用；普通 build 行为不受影响。
- [ ] enable 在首次 MoE EP op 之前；首次 op 后再 enable 会按预期拒绝。
- [ ] continue 用例结束后，已执行算子的 `.profbin` header 非零。
- [ ] 未执行算子的 zero-header 只产生预期 warning。
- [ ] `analysis_manifest.json` 显示频率来源为 CLI override，值为 `1000.0`。
- [ ] `events.csv` 无 header/event op mismatch。
- [ ] `intervals.csv` duration 非负，BEGIN/END 无异常未配对。
- [ ] `core_status.csv` 中 `local_drop_seen=0`，目标统计窗口未被 ring 覆盖。
- [ ] 活跃核 `flush_count` 与预期 kernel invocation 数量一致。
- [ ] `function_summary.csv`、`tail_summary.csv` 的 count 符合核数、轮数和条件执行范围。
- [ ] `functions_rankNNN.trace.json` 中有多个 `AIV:<core_id>` lane。
- [ ] 不把 logical pipeline 的跨核 p50 当成某个真实核的执行时间线。
- [ ] msprof 单独采集，cycle marker 已关闭。
- [ ] 自动化测试通过：

  ```bash
  python -m unittest discover -s mc2/tools/moe_ep_profiler/tests -v
  ```

## 12. 2026-09-04 `flow` 更新摘要

本次 `git pull origin flow` 将 `148cb7712` fast-forward 到 `6055711d6`，实际包含两个
功能修复：

1. `828a4db58`：补齐 `ENABLE_MOE_EP_PROFILING` 从顶层 CMake 到 `prepare.sh`
   再到子 CMake 的传递，修复参数看似开启但四 kernel 未编入 profiling、raw 全部
   zero-header 的问题。
2. `04f493474`：给 `analyze_cycle.py`、`analyze_msprof.py` 和 `run_msprof.py`
   增加最高优先级 `--frequency-mhz`。对 A5 cycle raw 使用
   `--frequency-mhz 1000`，即 `cycles / 1000 = us`。

两个 merge 提交没有额外的冲突解决内容。新增频率参数尚未补自动化测试，当前测试脚本
参考补丁也尚未把 `1000` 传给自动分析入口，这是后续建议补齐的两项。
