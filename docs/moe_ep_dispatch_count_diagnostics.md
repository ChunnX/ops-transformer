# Dispatch per-rank 计数诊断

## 当前结论

2026-09-20 的 8P、BS=4096、K=6、E=256 单跑日志中，各 rank 的
`num_recv_tokens_per_expert` 全部通过，`num_recv_payloads_per_rank` 全部失败。
rank7 的自身 payload 计数为 963，预期 2265；该计数的本地通知路径不经过 URMA。
所以需要先定位 Dispatch 的计数链，不能直接把错误归因于 payload batch 或 SQ 深度。

本次增加的是**诊断开关，不是已经验证的精度修复**。未改变计数算法、分核、
32 Jetty/SQ4096/CQ64、窗口布局、公开接口或 Combine。

第一轮开启计数诊断的日志中，192 条 `PREPARE_SUMMARY` 全部报告 8 个 peer 不匹配；
已打印的 `PREPARE` 中 actual 与 published 相等。64 条 SOURCE 的三个值一致，
跨 rank 对应的 SOURCE.send 与 NOTIFY.count 一致，OUTPUT 的四个值也一致。
因此已观察到的首个错误在本核计数链，而不是计数通知传输或接收提取。
这不等于 payload 传输已经验证正确，也尚未区分 rank 映射、去重和 histogram。

为避免重复获得同一层结论，新增 `DEDUP`/`DEDUP_SUMMARY`，分别在去重结束、
per-rank 统计结束读取仍存活的 rank 表；使用原始 TopK 独立重建 rank 和首个命中位置。
这仍是定位补丁，不是根因修复。`selected=0` 的 rank_errors 表示 rank 映射不符，
mask_errors 表示首个命中的保留位不符；`selected=1` 的 rank_errors 表示最终
去重 rank 表不符，此时 mask 已被统计流程复用，mask_errors 固定为 0、不是 mask 检查通过。
该阶段的 actual_keep 使用 -1 表示未读取，不应与 expected_keep 比较。
若两阶段均正确但 PREPARE 错，继续检查 histogram/累加链；若 selected=0 正确而
selected=1 错，检查 Select 到统计结束期间 rank 表的变化。cached 下需先保证缓存与路由一致。

## 当前默认：无日志的同步消融版本

加入 DEDUP 检查后，用户报告正确性通过；两份日志合计 384 条 DEDUP_SUMMARY、
192 条 PREPARE_SUMMARY，错误数全部为 0。该检查同时引入 V_S、S_V、DCCI、
标量扫描和打印，因此只能说明故障对时序/代码生成敏感，尚不能断言根因就是缺少某个同步。

用户已报告无诊断的 0/3（A+B）版本通过；当前将默认配置切换为 0/1，只保留
A 点 V_S → S_V，继续关闭全部 COUNT_CHECK 诊断。仅 A 的设备结果尚待验证：

| COUNT_CHECK | COUNT_SYNC_MASK | 用途 |
|---|---|---|
| 0 | 3 | A、B 两处纯同步，用户已报告通过 |
| 0 | 1（默认） | 只在 A：DedupAndSend 后同步，无诊断扫描、DCCI、打印或诊断 UB |
| 0 | 2 | 只在 B：CalSendCntPerRank 后同步 |
| 0 | 0 | 无新增同步、无计数诊断的对照组 |
| 1 | 0 | 恢复上一轮完整诊断版本 |

A 点位于 cached 分支的 slot 写回之前；B 点位于 rank 表的 V_MTE3/GM 写回之前。
同步只发生在 prepare 核的每个实际 group 内，不增加全核屏障，不改变 sender/prepare 分工。
位掩码非法或同时打开两类实验会编译报错，避免混合变量。宏都是设备代码编译配置，
不是 Python 环境变量；每次切换必须重新构建、安装 kernel 包，并保存对应配置和测试结果。

现在用默认 0/1 跑同一输入的非 cached Dispatch 正确性，不跑性能。这个版本正常情况下
**不应出现 `[MoeEpCountCheck]` 日志**，以测试脚本完整的输出/handle 比较为准，而不是只看不报错。
若仍出现计数诊断日志，先核对设备编译选项和实际安装包，不将其当作纯同步实验。
随后在相同输入、构建选项和运行方式下测 0/2，必要时用 0/0 作对照；记录是首轮还是多轮失败。
若纯同步也失败，再分离 DCCI/延时/编译调度影响，不把它解释为数学逻辑已被排除。
关闭诊断还会取消额外 UB 分配，因此 UB 布局变化也是变量，不能把失败简单归结为同步无效。
若纯同步稳定通过，仍需结合生成代码和实际依赖确定最小正式修复，不能仅凭一次通过删减同步。
最终候选需关闭诊断并覆盖 continue、cached、不同 BS/K/P 和 prepare 核数，再测性能。
32 Jetty、SQ4096、CQ64、窗口布局、公开接口和 Combine 均未改变。

## 恢复完整诊断与运行

`mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h` 中：

```cpp
#ifndef MOE_EP_DISPATCH_COUNT_CHECK
#define MOE_EP_DISPATCH_COUNT_CHECK 1
#endif
#ifndef MOE_EP_DISPATCH_COUNT_SYNC_MASK
#define MOE_EP_DISPATCH_COUNT_SYNC_MASK 0
#endif
```

上面是恢复完整诊断的配置，**不是当前默认配置**。需同时将 COUNT_CHECK 设为 1、COUNT_SYNC_MASK
设为 0；若编译命令显式传宏，须检查覆盖关系。沿现有构建/安装流程重新生成并安装包含该 kernel 的
算子包；只改 Python 或只重装旧 wheel 不会启用新的设备代码。

先用失败的同一输入执行单次、非 cached 的 Dispatch 正确性检查，不跑性能循环。
保留所有 rank 的 `[MoeEpCountCheck]` 行和最终比较结果；若没有任何该前缀日志，
不能视为检查通过，应先确认诊断宏、实际加载的算子包和设备 printf 日志收集。
此开关不自动停止原测试脚本的循环，需要在测试脚本中选择单次正确性执行。

## 日志含义

| 标记 | 检查/记录内容 | 定位方式 |
|---|---|---|
| `INPUT` | 原始 TopK 中非负但越界的 expert | 先处理输入或参数不一致；本次参考计数跳过该非法项 |
| `DEDUP` / `DEDUP_SUMMARY` | 每 group、每阶段的 rank 表与保留 mask 检查；每阶段最多 4 条详细错误 | 按上文 selected=0/1 的含义定位；两个阶段和 PREPARE 都应检查 |
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

COUNT_CHECK 关闭时，诊断成员、UB 分配、调用、扫描和日志都由预处理移除；
COUNT_SYNC_MASK 独立控制两处实验同步，其默认值 1 表示仅 A 点同步，并不表示无新增同步。
开启时额外占用 `2 * align32(epWorldSize * 4)` 字节 UB：8P 为 64B，1024P 为 8192B。
参考扫描仅由各 prepare 核读取自己的 BS 区间，不按固定 8P 或固定 64 AIV 工作。

诊断会增加 scalar GM 读取、缓存失效、局部管线同步和 printf，可能改变时序并掩盖竞态；
诊断版通过不代表默认版本已经修复。**诊断版耗时不得用于性能结论。**
结束后关闭宏并重新构建，再做精度与性能回归。

本地只能检查源码、宏隔离和参考计数逻辑；真实 AscendC 编译、设备流水、日志完整性及
最终精度仍需要设备验证。本轮没有使用远端 NPU 环境。

此前 A+B 同步消融补丁静态检查：`git diff --check` 通过；从实际源码抽取配置和 A/B
调用片段，用 MSVC 预处理器检查默认配置及表中 5 种合法组合，确认同步数量、位置和
诊断调用开关；1/3、0/4、0/-1、2/0 四种非法组合均按预期报错。
0/0 下剔除实验块后的 kernel 主体与本轮修改前关闭诊断的主体逐行一致。
这些检查不构成完整 CANN 编译或设备同步正确性证明。
