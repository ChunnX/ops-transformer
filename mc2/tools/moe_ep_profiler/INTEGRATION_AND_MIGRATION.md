# MoE EP Device-Side Profiler 接入与迁移指南

本文用于在 `flow` 更新、切换分支或 kernel 重构后，快速恢复
`moe_ep_dispatch`、`moe_ep_dispatch_epilogue`、`moe_ep_combine` 和
`moe_ep_combine_epilogue` 的 device-side cycle 打点。

**本轮接入范围仅为 `moe_ep_dispatch_epilogue` 和 `moe_ep_combine`。** 公共 context ABI、
四 ring、四算子历史 stage 映射保留兼容；另外两个 kernel 本轮不插点，空 raw 是预期。
本文保留四算子协议与旧阶段说明以供后续扩展，但不表示本轮四 kernel 都已启用 marker。

日常构建、运行和解析命令见 [README.md](README.md)。业务流水及 marker 选点依据见
`mc2/docs/moe_ep_pipeline.md`；该目录可能是本地未跟踪内容，迁移 profiler 时不要覆盖它。

## 1. 当前实现的计时语义

- 仅分析非 Hybrid 模板；Dispatch Hybrid 不插 marker。
- 已接入 marker 的 kernel 按 AIV 物理角色记录。`MTE2`、`MTE3`、`Vector`、`Scalar`、
  `URMA`、`Wait` 是逻辑事件标签，不代表 `GetSystemCycle()` 能直接区分真实硬件流水。
- `Mark()` 只在核内执行 `GetSystemCycle()` 并写本地固定数组，不在每个 marker 处写 GM、
  不做 host-device 同步。
- **`KERNEL` 仅覆盖 `Process()` 中的 BEGIN→END，不覆盖此前执行的 `Init()`。**
  例如 Combine 在 Init 读取 `recv_rank_offsets`、建立 runtime/UB 的时间不在 KERNEL 内；
  Process 内的常量 flag 初始化则属于 `FLAG_SOURCE_INIT`。不要将这里的 KERNEL duration
  直接当作 torch/msprof 的完整 kernel launch 耗时。
- 每次 kernel invocation、每个核最多保存 32 条 marker。全部 GM ring 写入与 DCCI
  集中在 `KERNEL_END` 之后执行，因此这些批量写回开销不计入 `KERNEL` 区间；中间
  marker 的 `GetSystemCycle()` 和本地数组写入仍有少量不可消除的 device 开销。
- 沿用 flow 历史记录的 Ascend 950 环境：`GetSystemCycle()` tick 为 1000 MHz，按下式换算。
  本轮未上板重新校准；其他环境必须先确认 tick 频率：

  ```text
  duration_cycles = end_cycle - begin_cycle
  duration_us     = duration_cycles / 1000
  ```

  解析时显式传 `--frequency-mhz 1000`，不要让 AIV `Current Freq` 或 `vec_freq`
  覆盖 system-cycle 的换算频率。迁移到其他型号或环境后必须重新确认 tick 频率。
- 不同 rank 的 cycle epoch 不保证同步。跨 rank 只能比较 duration，或以各 rank 自身
  `KERNEL_BEGIN` 为相对起点，不能直接拼接绝对时间戳。
- KERNEL END 后的 profiler Flush 虽不计入区间，仍会延长真实 launch 尾部，可能改变
  下一 kernel 的开始时机、continue 竞争和 Combine 提前退出收益。必须用普通 OFF 构建
  对照生产性能；profiling 构建跑过不能证明原有 race 已修复。

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

同一 group 的 profiling context 只能由一个已 enable profiler 的 `ElasticBuffer` 持有。
不要让另一实例覆盖 ring 地址，或在第一实例仍可发 kernel 时释放 ring。两个实例均未
enable 时仍保留原共享行为；这项生命周期检查属于 host 集成，不能只靠设备 marker 解决。

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
重新 source vendor 环境，并使用下述独立 lazy-JIT cache 验证新产物。

增量构建切回普通模式时，`build.sh` 必须显式传 `-DENABLE_MOE_EP_PROFILING=OFF`，
同时 unset `CANN_OPS_MOE_EP_PROFILING_BUILD`。只在开启时传 ON 会让旧 CMakeCache 保留
kernel ON、而 wheel 已 OFF，产生 context ABI 错配。本轮补回显式 OFF，并提供源码回归测试。

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

安全处理 lazy-JIT cache（不删除全局 cache）：

1. 停止使用旧扩展的测试进程；安装同一构建生成的算子包和 wheel，重新 source 对应 vendor。
2. 确认实际导入的是新 wheel 的包路径及 build literal，避免源码目录遮蔽已安装包。
3. 推荐在启动新的 Python 进程前使用独立目录，例如 Linux：

   ```bash
   MOE_EP_JIT_DIR="$(mktemp -d /tmp/moe_ep_jit.XXXXXX)"
   export TORCH_EXTENSIONS_DIR="$MOE_EP_JIT_DIR"
   MOE_EP_CYCLE_PROFILER=1 python /path/to/moe_train_test.py
   ```

4. 从 verbose JIT 日志确认实际构建目录、扩展名称和编译宏；若需保留原 cache 根，先从日志
   核实仅对应 ElasticBuffer 扩展的绝对目录，再将该目录备份重命名，重新启动进程构建。
   不递归删除整个 `torch_extensions`、用户目录或共享缓存，也不在仍有进程加载时搬动缓存。
5. OFF 对照使用独立的另一缓存目录与普通 wheel/kernel，核对编译命令无 profiling 宏。

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
- 不为测量增加 `SyncAll`、`SyncFunc`、DCCI、Drain、collective 或逐轮设备同步；包围已有
  依赖边即可。不移动业务同步来改变测量区间，GM/DCCI 仅由原有业务或末尾 profiler Flush 执行。
- 不在 token/rank/tile/retry 循环内无界记录。优先包围整个循环；统计极值路径中的实际
  marker 次数，不能用 stage 数或源文件宏出现次数替代每次 invocation 的预算。
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

下表是兼容旧 raw 的完整阶段表。本轮只有 DispatchEpilogue、Combine 行对应实际 marker；
Dispatch、CombineEpilogue 行为 flow 历史映射，未自动移植其业务 kernel。

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
| DispatchEpilogue | `KERNEL` (0) | Scalar | Process 的业务区间，Init 在外，Flush 在 END 后 |
| DispatchEpilogue | `PREFIX` | Scalar | 非 cached prefix sum |
| DispatchEpilogue | `PAYLOAD_WAIT` | Wait | 最后一个 AIV 的状态等待/清零 |
| DispatchEpilogue | `WAIT_BARRIER` | Wait | `SyncAll<true>()` |
| DispatchEpilogue | `COUNT_HITS` | Vector | `CountHits()` |
| DispatchEpilogue | `WINDOW_OUTPUT` | MTE2 | 从 window 搬到输出 |
| DispatchEpilogue | `COUNT_BARRIER` (6) | Wait | CountHits 后已有的全核 barrier |
| DispatchEpilogue | `RANK_ROW_STARTS` (7) | Scalar | 从各核 rank 计数构造当前核行起点及 rank offsets |
| DispatchEpilogue | `CACHED_RANK_OFFSETS` (8) | MTE2 | 所有 cached 核包围复制 helper；仅 AIV0 实际复制，其他核早退形成短区间 |
| DispatchEpilogue | `OUTPUT_SYNC` (9) | Wait | output copy 后已有的 MTE3_S，单独计时 |
| Combine | `KERNEL` (0) | Scalar | Process 的业务区间，不含 Init 和末尾 profiler Flush |
| Combine | `ADDRESS_TABLE` (1，历史) | Scalar | 仅保留旧 raw 的名字；新直接路径不再发此事件 |
| Combine | `DATA_WQE` (2) | URMA | metadata 读取、标量地址计算、WQE/batch、高水位 Drain |
| Combine | `FLAG_COMMIT` (3) | URMA | 本卡 MTE flag、远端 flag/Flush、业务 DCCI 和原诊断记录 |
| Combine | `FLAG_SOURCE_INIT` (4) | MTE3 | 所有核包围初始化 helper；每核准备 UB 常量，仅 AIV0 写持久 window 源 |
| Combine | `LOCAL_METADATA_COPY` (5) | MTE3 | 各核本卡 metadata 分片及 token/weight 搬运 |
| Combine | `LOCAL_BARRIER` (6) | Wait | 本卡搬运后原有唯一一次 SyncAll |
| Combine | `CORE_ASSIGNMENT` (7) | Scalar | upstream 三分支 rank/channel owner 计算 |
| CombineEpilogue | `KERNEL` | Scalar | 整个 `Process()` |
| CombineEpilogue | `BUFFER_INIT` | Scalar | `BuffInit()` |
| CombineEpilogue | `COMPLETION_WAIT` | Wait | AIV0 等待并清 completion flags |
| CombineEpilogue | `WAIT_BARRIER` | Wait | 全 AIV 同步 |
| CombineEpilogue | `REDUCE_OUTPUT` | Vector | token reduce 和输出 |

DispatchEpilogue 的旧 ID 0～5 和其他算子旧 ID 均不重编号。`WINDOW_OUTPUT` 仅包围
`CopyFromWindow*`，其后的显式 MTE3_S 归 `OUTPUT_SYNC`；`PAYLOAD_WAIT` 在真实等待函数
内部，END 位于接收状态清零及已有 MTE3_S 之后。engine 是逻辑标签：`DATA_WQE` 和
`FLAG_COMMIT` 都不是纯网卡时间，不能据此推导 URMA 物理流水利用率。

当前按业务分支审阅的每核每 invocation 上界：

| 路径 | BEGIN/END marker 上界 | 说明 |
|---|---|---|
| DispatchEpilogue 非 cached、最后核、有输出 | 18 | 9 对，包含真实等待和输出完成边 |
| DispatchEpilogue cached、最后核 | 12 | 6 对；无对应工作的核可少于此值 |
| Combine 直接 metadata 路径 | 14 | 7 对；ADDRESS_TABLE 不再发，owner 内循环不逐项打点 |

这些上界只适用于当前插点位置。下一次重构后必须重新审阅早退、空数据、cached、
低 AIV 轮转和循环路径；不要简单沿用本表。

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

即使算子没有执行，也会生成全零文件。`unused zero-header profiler buffer` 对未插点或未执行算子
是正常 warning；若本轮已插点的两算子明明执行过却全部为零，再检查编译开关、enable 时机、
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

本目录参考补丁的自动解析已传第 8 项；它只是对外部脚本的参考 diff，不代表外部工作区
已经修改。手工解析已有 raw 无需重跑 NPU：

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
8. 逐个选中的 kernel 按六锚点重新定位 marker；确认 Dispatch Hybrid 仍跳过。
9. 保持旧 stage ID，追加新事件并更新 parser 名字映射。
10. 按新版测试脚本语义移植 enable、末尾 sync/dump 和自动分析；传入 1000 MHz。
11. 编译安装 profiling 包/wheel，重新 source 环境并按第 3 节使用独立 JIT cache。
12. 先跑小规模 continue 用例，再跑完整性能轮次；cycle 与 msprof 分两次采集。

## 11. 验收清单

- [ ] `--moe-ep-profiler` 从顶层传到 prepare 子 CMake，配置日志显示为 ON。
- [ ] 本轮两目标 kernel 编译命令含 `ENABLE_MOE_EP_PROFILING=1`；公共四 ring ABI 保持一致。
- [ ] torch extension 与 kernel 使用相同的 `MoeCommContext` 大小和尾部布局。
- [ ] profiling build 的 Python 接口可用；普通 build 行为不受影响。
- [ ] 从 profiling 增量切回普通构建时，kernel 显式 OFF 且 wheel build literal/JIT 宏匹配。
- [ ] enable 在首次 MoE EP op 之前；首次 op 后再 enable 会按预期拒绝。
- [ ] continue 用例结束后，已执行且已插点的 DispatchEpilogue/Combine `.profbin` header 非零。
- [ ] 未插点的 Dispatch/CombineEpilogue 及未执行算子的 zero-header 只产生预期 warning。
- [ ] `analysis_manifest.json` 显示频率来源为 CLI override，值为 `1000.0`。
- [ ] `events.csv` 无 header/event op mismatch。
- [ ] `intervals.csv` duration 非负，BEGIN/END 无异常未配对。
- [ ] `core_status.csv` 中 `local_drop_seen=0`，目标统计窗口未被 ring 覆盖。
- [ ] 活跃核 `flush_count` 与预期 kernel invocation 数量一致。
- [ ] `function_summary.csv`、`tail_summary.csv` 的 count 符合核数、轮数和条件执行范围。
- [ ] `functions_rankNNN.trace.json` 中有多个 `AIV:<core_id>` lane。
- [ ] 不把 logical pipeline 的跨核 p50 当成某个真实核的执行时间线。
- [ ] msprof 单独采集，cycle marker 已关闭。
- [ ] 正常 OFF 构建复核 continue 正确性和性能，不将 profiler Flush 改变时序后的跑通当作 race 修复证据。
- [ ] 自动化测试通过：

  ```bash
  python -m unittest discover -s mc2/tools/moe_ep_profiler/tests -v
  ```

## 12. `flow` 历史更新与本轮移植

以下是原 flow 指南记载的 2026-09-04 历史：`148cb7712` 到 `6055711d6` 包含两个功能修复。
本轮是读取本地 flow 提交内容并通过补丁移植，没有执行 `git pull` 或切换业务分支。

1. `828a4db58`：补齐 `ENABLE_MOE_EP_PROFILING` 从顶层 CMake 到 `prepare.sh`
   再到子 CMake 的传递，修复参数看似开启但四 kernel 未编入 profiling、raw 全部
   zero-header 的问题。
2. `04f493474`：给 `analyze_cycle.py`、`analyze_msprof.py` 和 `run_msprof.py`
   增加最高优先级 `--frequency-mhz`。对 A5 cycle raw 使用
   `--frequency-mhz 1000`，即 `cycles / 1000 = us`。

两个 merge 提交没有额外的冲突解决内容。本轮迁移已补频率 override/CLI 转发测试，
并在参考补丁自动分析入口显式传入 `1000`；没有直接修改外部 `moe_train_test.py`。

## 13. 下次重构后的快速生成与只读验证

先记录**已经调通的业务提交**作为基线；不要 checkout 整个 flow kernel，也不要把旧
`BuildAddressTable`/旧 SendPhase 或旧 workspace 布局带回去。仅公共 profiler/工具可复用，
kernel marker 必须依据当前调用图逐点生成：

1. 列出目标函数当前的 Init、Process、cached/非 cached、空路径、early return 和循环。
2. 按第 4 节六锚点接入；保留现有 Init 在 Process 之前的调用顺序。
3. 建立“旧 ID→旧语义→当前锚点”的表。已废弃事件保留 enum 和 parser 名字、不发 marker；
   新语义追加新 ID。若只是内部函数改名但范围未变，可保留原 ID。
4. 每个 stage 在相同控制分支内配对；原有同步算在哪个区间必须写清，不新增同步。
5. 分支枚举每核最坏 marker 数≤32；禁止无界循环内 marker。KERNEL END 后才 Flush。
6. 更新 `DEFAULT_STAGE_NAMES`、本指南阶段表与测试；验证宏传播链、host/device ABI 和共享
   context 生命周期，最后才构建安装和上板。

在仓库根目录执行（无需 torch、CANN 或 NPU）：

```bash
python -m unittest discover -s mc2/tools/moe_ep_profiler/tests -v
python mc2/tools/moe_ep_profiler/validate_instrumentation.py \
  --ops moe_ep_dispatch_epilogue moe_ep_combine --base-ref HEAD
git diff --check
```

`--base-ref` 必须是本轮插点前的业务版本；提交插点后换成记录的旧提交，而不是继续用 HEAD。
默认只检查本轮两 kernel；以后经确认扩展到四 kernel 时显式传入全部四个 `--ops`。
扩展时先按六锚点适配另外两个当前实现，而不是直接覆盖 flow 的旧业务代码。
该工具只读源码，检查真实 enum 与 parser 名字对应、
六锚点、词法 BEGIN/END 数量、局部容量常量、KERNEL END 后 Flush 及编译宏链的源码锚点。
基线模式剥离 profiler include/enum/宏调用后比较业务 token 序列，能提示意外增删的同步
或业务语句；不自动替换或修复代码。

**通过仅代表结构检查通过。** 这不是完整 C++ 解析器，不证明所有路径成对、循环调用次数、
跨核数据依赖、实际编译命令或 ABI 正确；源文件宏次数也不是动态 marker 预算。
应人工复核 diff、查看 OFF/ON 构建日志，并在上板后检查 `local_drop_seen=0`、flush_count、
空/非空路径成对、raw header 以及明确的 1000 MHz CLI 来源。需要证明未新增业务同步时，
同时审阅基线差异，不能只依赖正则扫描或“不报错”的结果。
