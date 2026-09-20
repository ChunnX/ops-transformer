# Dispatch per-rank 计数诊断

## 当前结论

2026-09-20 的 8P、BS=4096、K=6、E=256 单跑日志中，各 rank 的
`num_recv_tokens_per_expert` 全部通过，`num_recv_payloads_per_rank` 全部失败。
rank7 的自身 payload 计数为 963，预期 2265；该计数的本地通知路径不经过 URMA。
所以需要先定位 Dispatch 的计数链，不能直接把错误归因于 payload batch 或 SQ 深度。

本次增加的是**诊断开关，不是已经验证的精度修复**。未改变计数算法、分核、
32 Jetty/SQ4096/CQ64、窗口布局、公开接口或 Combine。

## 开启与运行

`mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h` 中：

```cpp
#ifndef MOE_EP_DISPATCH_COUNT_CHECK
#define MOE_EP_DISPATCH_COUNT_CHECK 1
#endif
```

当前为排查此次失败已默认开启，无需手动改宏。若设备编译命令显式传入该宏，须确认其值为
`1`，不要被 `-DMOE_EP_DISPATCH_COUNT_CHECK=0` 覆盖。沿现有构建/安装流程重新生成并安装包含该 kernel 的
算子包；只改 Python 或只重装旧 wheel 不会启用新的设备代码。

先用失败的同一输入执行单次、非 cached 的 Dispatch 正确性检查，不跑性能循环。
保留所有 rank 的 `[MoeEpCountCheck]` 行和最终比较结果；若没有任何该前缀日志，
不能视为检查通过，应先确认诊断宏、实际加载的算子包和设备 printf 日志收集。
此开关不自动停止原测试脚本的循环，需要在测试脚本中选择单次正确性执行。

## 日志含义

| 标记 | 检查/记录内容 | 定位方式 |
|---|---|---|
| `INPUT` | 原始 TopK 中非负但越界的 expert | 先处理输入或参数不一致；本次参考计数跳过该非法项 |
| `PREPARE` | `actual` 为该 prepare 核的向量计数；`published` 为其写入 GM 的计数行；`expected` 为独立标量参考 | actual 错：本核去重/直方图链；actual 正确但 published 错：发布链 |
| `PREPARE_SUMMARY` | 每个非空 prepare 区间的 mismatch 总数，详细 mismatch 每核最多打印 4 个 | 每个非空 prepare 核都应有一条且 mismatches=0；空分片不执行参考扫描 |
| `SOURCE` | 每个发送目标的 `row_sum`、全局原子累加值 `aggregate`、编码进通知的 `send` | 三者应相同；row_sum 正确但 aggregate 错：初始化/原子汇总；aggregate 正确但 send 错：读取链 |
| `NOTIFY` | 接收侧在专家统计复用 UB 之前保存的通知低位 flag、高位 count | `(rank=S, peer=D)` 的 SOURCE.send 应等于 `(rank=D, source=S)` 的 NOTIFY.count，flag 应为 1 |
| `OUTPUT` | 快照 before、专家统计后的原通知 after、GatherMask 提取值 extracted、最终 GM 值 gm | 四者应相同；按首个不一致的相邻阶段定位 |

`SOURCE` 和 `OUTPUT` 是快照日志，不自动判定；需要按上表比较。各 rank/核的打印顺序
不保证一致，请按 rank、peer/source、core 关联，不要按日志邻接顺序关联。
一次只跑一轮，以免缺少迭代号的日志混在一起。
所有 rank 合计的阶段快照数量随 rank 数呈平方增长；大规模运行可能超过设备 printf
容量。应先使用已知失败的 8P 用例定位，日志缺失或截断不能解释为该阶段通过。

标量参考对每个 token 的有效 expert 映射到 rank，同一 token 命中同一 rank 只计一次；
`-1` 等负 expert 不计入。cached 模式也使用原始 TopK 作为参考，要求缓存与当前路由一致。
因此 cached 下 mismatch 还可能来自不匹配的缓存，不能直接判为 histogram 实现错误。

## 开销与验证边界

关闭时，诊断成员、UB 分配、调用、扫描和日志都由预处理移除。
开启时额外占用 `2 * align32(epWorldSize * 4)` 字节 UB：8P 为 64B，1024P 为 8192B。
参考扫描仅由各 prepare 核读取自己的 BS 区间，不按固定 8P 或固定 64 AIV 工作。

诊断会增加 scalar GM 读取、缓存失效、局部管线同步和 printf，可能改变时序并掩盖竞态；
诊断版通过不代表默认版本已经修复。**诊断版耗时不得用于性能结论。**
结束后关闭宏并重新构建，再做精度与性能回归。

本地只能检查源码、宏隔离和参考计数逻辑；真实 AscendC 编译、设备流水、日志完整性及
最终精度仍需要设备验证。本轮没有使用远端 NPU 环境。
