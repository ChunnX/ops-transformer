# FullMeshV2 device profiler

该 profiler 只作用于 A5 `fullmesh_v2` 路径。设备侧不使用 `printf`，每个 AIV 核写独立的 64 字节对齐计数槽和固定记录区，kernel 结束后再由 Host 一次性导出，因此不同核不会争用打印缓冲区。

## 输出格式

一个 rank 输出一个 `fullmesh_v2_prof_rank<N>.bin`，小端序，总大小 41536 字节。

| 区域 | 偏移 | 格式 |
|---|---:|---|
| Header | 0 | 64 字节；前 8 个 `uint32_t` 分别为 magic、layout/version、实际核数、单条记录字节数、每核容量、计数槽跨度、数据区偏移、保留字段 |
| Core counts | 64 | 72 个槽，每槽 64 字节；槽首 `uint32_t` 是该核有效记录数 |
| Records | 4672 | 72 核 × 32 条 × 16 字节；每条为 `{uint32 eventId, uint32 payload, uint64 cycle}` |

`payload` 含义随事件变化：kernel begin/end 的高 16 位为 token 发送核数、低 16 位为 CumSum 核数；其他阶段记录该阶段处理的专家数、状态数或有效专家数。

可视化按以下关键阶段拆分：

| 阶段 | 对应代码 |
|---|---|
| `PREPARE_TOKEN_INPUT` / `PREPARE_COUNT_INPUT` | `ExpIdsCopyAndMaskCal`，分别为 token 发送核和计数核准备输入 |
| `TRANSFER_TOKENS` | `AllToAllDispatch`，向远端 window 搬运 token 数据 |
| `SEND_COUNTS` | `CalAndSendCnt*`，计算并发送各专家 token 数 |
| `WAIT_RECV_COUNTS` | `WaitDispatch` 整体；内部继续拆分状态清理和接收计数汇总 |
| `CLEAR_RECV_STATUS` | `WaitDispatchClearStatus`，清除已消费的接收状态 |
| `GATHER_RECV_COUNTS` | `GatherSumRecvCnt`，汇总每个计数核收到的 token 数 |
| `GET_CUMSUM` | `GetCumSum`，计算计数核之间的前缀和 |
| `BUILD_RECV_CUMSUM` | 构造当前核负责范围内的接收计数前缀和 |
| `PUBLISH_RECV_COUNTS` | 输出接收计数、复制到各核 workspace 并发布完成标志 |
| `DRAIN_PIPELINES` | `PipeBarrier<PIPE_ALL>`，等待发送或计数阶段此前下发的流水操作完成 |
| `WAIT_CUMSUM` / `RESET_CUMSUM_FLAG` | 等待计数结果，并清理本轮同步标志 |
| `BUILD_EXPERT_TASKS` | `SetValidExpertInfo`，筛选当前核实际需要处理的专家 |
| `FORMAT_OUTPUT` | `WaitAndFormatOutput`，等待 token 到达并写出专家数据 |
| `CLEAR_LOCAL_FLAGS` | `ClearLocalWindowDataFlags`，清除已消费的数据到达标志 |
| `NO_ACTIVE_TOKENS` | 当前发送核没有活跃 token，跳过实际发送 |
| `NO_ASSIGNED_STATUS` | 当前核没有分到接收状态，跳过专家数据整理 |
| `NO_VALID_EXPERT` | 当前核分到的专家均未收到 token，跳过输出格式化 |

`SEND_TOKENS`、`WAIT_RECV_COUNTS`、`LOCAL_WINDOW_COPY` 是外层阶段，其内部的细分阶段会在时间线上重叠显示。并非所有核都有 `FORMAT_OUTPUT`：未分到状态或所负责专家没有收到 token 的核会提前返回。

## 导出

在 `aclnnMoeDistributeDispatchV2` 完成并同步 stream 后调用：

```cpp
#include "fullmesh_v2_profiler_dump.h"

FullMeshV2ProfilerDump::Dump(expandXAddr, expandXSize, rankId, outputDir);
```

profiler buffer 固定位于 `expand_x` 起始位置。一次测试只保留最后一次 kernel 的记录；请在需要采集的调用结束后立即导出。该调试模式会覆盖 `expand_x` 开头 41536 字节，只能用于不消费 dispatch `expand_x` 的场景，例如 combine 使用 golden `expand_x` 的用例。正式性能验收前，在编译参数中定义 `MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILING=0`，或将 `moe_distribute_dispatch_v2_full_mesh_profiler.h` 中的默认值改为 0。

## 可视化

```bash
python3 fullmesh_v2_visualize.py fullmesh_v2_prof_rank*.bin --output-dir prof_out --freq-mhz 1650
```

输出：

- `fullmesh_v2_events.csv`：每核原始事件；
- `fullmesh_v2_stages.csv`：配对后的阶段耗时；
- `fullmesh_v2_timeline.svg`：按 rank/core 分行的静态甘特图，顶部图例说明各颜色对应的执行阶段；
- `fullmesh_v2_trace.json`：可由 Chrome `chrome://tracing` 或 MindStudio Insight 打开，时间戳和时长单位为微秒；点开事件可查看阶段说明和 payload。

多 rank 默认各自以最早事件对齐到 0；传 `--keep-cycle-origin` 可保留设备 cycle 原点。
