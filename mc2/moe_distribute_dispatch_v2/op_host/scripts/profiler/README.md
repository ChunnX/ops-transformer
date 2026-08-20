# FullMeshV2 device profiler

该 profiler 只作用于 A5 `fullmesh_v2` 路径。设备侧不使用 `printf`，每个 AIV 核写自己的 64 字节对齐计数槽和固定记录区，kernel 完成后再统一写入 `expand_x`，多核之间不会争用同一个输出地址。

## 二进制布局

每个 rank 导出一个 `fullmesh_v2_prof_rank<N>.bin`，小端序，总大小 41,536 字节。

| 区域 | 偏移 | 格式 |
|---|---:|---|
| Header | 0 | 64 字节；magic、layout/version、实际核数、单条记录字节数、每核容量、计数槽跨度和数据区偏移 |
| Core counts | 64 | 72 个槽，每槽 64 字节；槽首 `uint32_t` 为本核有效记录数 |
| Records | 4672 | 72 核 × 32 条 × 16 字节；每条为 `{uint32 eventId, uint32 payload, uint64 cycle}` |

格式版本 4 同时覆盖串行模式和原有并行模式。每核最多 32 条记录，既保留阶段起止时间，也记录发送侧累计指标：

- 实际 token 发送任务数、分配的发送 buffer 数、真正用到的 buffer 数和 buffer 初始化指令耗时；
- 发送位置计算、输入搬运、量化/转换、UB 组包、等待发送 buffer 的累计 cycle；
- 串行模式进入接收阶段时 `TPipe::Reset`、分核和 UB buffer 初始化的累计 cycle；
- 接收状态轮询次数、状态清理、跨核前缀和轮询、串行 compact count 读取、局部前缀和、输出任务构建、格式化轮询次数和实际拷贝批数。

`fullmesh_v2_core_summary.csv` 每个 rank/core 只输出一行。发送侧未被细分指标覆盖的时间写在 `send_other_cycles`，包括发送 buffer 初始化、循环控制和 MTE3 指令下发等；接收轮询阶段未被 buffer 初始化、状态清理和计数汇总覆盖的时间写在 `wait_recv_other_cycles`。

## 导出

profiler buffer 位于 `expand_x` 起始位置。kernel 和 stream 同步完成后，将每个 rank 的 `expand_x` 前 41,536 字节保存为：

```text
fullmesh_v2_prof_rank0.bin
fullmesh_v2_prof_rank1.bin
...
```

一次测试只保留最后一次 kernel 的结果。调试模式会覆盖 `expand_x` 开头 41,536 字节，只能用于后续不消费本次 dispatch `expand_x` 的性能用例。若 `expand_x` 小于 41,536 字节，设备侧自动禁用记录。

正式性能或精度验证前，将 `MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILING` 设为 0。

## 解析与可视化

```bash
python3 mc2/moe_distribute_dispatch_v2/op_host/scripts/profiler/fullmesh_v2_visualize.py \
  fullmesh_v2_prof_rank*.bin --output-dir prof_out --freq-mhz 1000
```

`--freq-mhz` 是设备频率；若当前环境中 1000 cycles 对应 1 us，就传 1000。

输出：

- `fullmesh_v2_core_summary.csv`：每核一行的完整瓶颈摘要；
- `fullmesh_v2_send_summary.csv`：所有 token 发送核，按 `transfer_cycles` 从慢到快排列；
- `fullmesh_v2_stages.csv`：所有已配对阶段；
- `fullmesh_v2_events.csv`：原始记录；
- `fullmesh_v2_timeline.svg`：带图例的每核时间线；
- `fullmesh_v2_trace.json`：Chrome Trace / MindStudio Insight 时间线。

脚本终端只打印最慢 token 发送核的一行，完整结果查看两个 summary CSV。Chrome Trace 不写不兼容的 `displayTimeUnit=us` 字段。
