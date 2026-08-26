# MegaMoe arch35 Dispatch：rank-round-robin wave 说明

本文对应分支 `codex/dispatch-wave-rank-interleave-trace-v3` 的当前实现，重点说明：

- sendmask 能提供哪些路由信息；
- 当前 wave 如何按 expert 组织 Dispatch；
- compact `topkIndex` 如何转成实际搬运任务；
- task 如何分核并发布 GMM1 ready；
- 当前仍保留的性能限制。

相关代码主要位于：

- `mega_moe_send_mask.h`
- `mega_moe_token_dispatch.h`
- `../mega_moe_wave.h`

## 1. sendmask 提供的信息

源卡将路由项展平为：

```text
topkIndex = tokenId * topK + topKSlot
```

`GatherAndSendExpertMasks` 为每个 `(目标 expert, source rank)` 生成一个 slot：

```text
[route bitmap][count]
```

- bitmap 的每一 bit 对应一个 flatten 后的 `topkIndex`；
- count 表示该 expert 从这个 source rank 接收的 token 数；
- 接收端汇总 count，形成 `cumsumInfo[expert][rank]`。

因此 Dispatch 在进入具体搬运前已经知道：

```text
当前 expert 一共有多少 token；
每个 source rank 分别贡献多少 token。
```

具体 `topkIndex` 在 wave 循环前由共享 RoutePlan 读取 bitmap，并通过
`CreateVecIndex + GatherMask` 一次性压紧到 Route Table。

## 2. wave 和 expert segment

`DispatchWaveAndGetEndCursor` 与 GMM1 使用同一套 M-group 预算，通过 `AdvanceWaveCursor` 得到当前 wave 的范围：

```text
waveBegin = (expertIdx, rowOffsetInExpert)
waveEnd   = (expertIdx, rowOffsetInExpert)
```

一个 wave 可以包含：

- 多个完整的小 expert；
- 一个大 expert 的部分 row；
- 大 expert 尾段与后续小 expert。

随后 Dispatch 按 expert 顺序处理 wave 内各 segment。大 expert 跨 wave 时，`rowOffsetInExpert`
就是该 segment 在完整 rank-round-robin 序列中的绝对 row，因此不需要额外保存或推进 Dispatch 游标。

## 3. 共享 compact 索引生成

`PrepareDispatchCounts` 完成后，全部 AIV1 将 `(expert, rank)` pair 按核号 stride 均分。
每个 pair 只扫描一次完整 sendMask，并将 compact topkIndex 写到：

```text
routeTable[rankGlobalRowBegin + compactOrdinal] = topkIndex
```

每核完成自己的 pair 后写独立 cache-line done；core0 汇总所有 done，再发布一条独立
Route Table ready。该同步不复用 Dispatch→GMM1 或 GMM2→Combine 的 flag。

进入当前 expert segment 后，本核从 `cumsumInfo` 取得每个 rank 的 token 数，并解析该
segment 在每个 rank compact 序列中的 `[ordinalBegin, ordinalEnd)`。第一遍 logical-row
扫描生成原序 `{rankId, compactOrdinal}`，并标记本核实际访问的 rank。只有这些 rank 的
slice 会下发到 `validTopkIndexTensor` 的独立 32B 对齐区域，最后只执行一次 `MTE2_S` 等待：

```text
validTopkIndexTensor:
[used-rank0 slice][padding][used-rank1 slice]...[used-rankN slice]
```

随后第二遍按原 task 顺序原地回填：

```text
{rankId, compactOrdinal}
  -> rankUbBase[rankId] + compactOrdinal - ordinalBegin
  -> 读取 topkIndex
  -> 原地写最终 {rankId, topkIndex} queue
```

不均衡时，第一遍的 rank-round cursor 只在某个 rank 耗尽、进入下一个 active-rank
interval 时更新活跃 rank 表；正常 interval 内按本核 logical row 单调推进。该流程不依赖
单-rank 假设，任意 rank token 分布都走相同路径，也不会搬入本核没有访问的 rank slice。

正常路径的 `topkIndexTensor` 只保存最终 queue：

```text
[最终 {rankId, topkIndex} queue]
```

旧的 grouped descriptor、rank prefix 和 descriptor scatter 均已从共享表路径移除；
原先“scatter 后再回填”的三遍任务扫描缩减为“原序 descriptor + 原地回填”两遍。
旧 per-core mask 扫描仍作为编译期/容量保护回退。

## 4. 正向 rank-round-robin 任务顺序

获得各 rank 的 compact list 后，实际任务按以下顺序正向生成：

```text
ordinal 0: r0t0, r1t0, r2t0, ...
ordinal 1: r0t1, r1t1, r2t1, ...
ordinal 2: r0t2, r1t2, r2t2, ...
```

某个 rank 耗尽后直接跳过。

例如两个 expert 的逻辑任务顺序是：

```text
task0 = E0 r0 a0
task1 = E0 r1 b0
task2 = E0 r0 a1
task3 = E0 r1 b1
task4 = E0 r1 b2
task5 = E1 r0 c0
task6 = E1 r2 d0
task7 = E1 r2 d1
```

即先取 E0 每张活跃卡的第一个 token，再取第二个；E0 完成后继续 E1。每个 core 的
本地任务表已经是最终执行顺序，不再遍历完整序列后用 core id 过滤。

## 5. 分核和 workspace row

正向任务序号仍直接成为该 expert 内的 workspace row，逻辑顺序完全不变。
目的 workspace 按全局 row 固定分核：

```text
globalRow = expertGlobalRowBegin + r
owner     = globalRow mod coreCount
```

`expertGlobalRowBegin` 保留不同 expert 之间的起始 phase。同一 core 在专家内的
后续任务始终跨 `coreCount` 行，不再随 round 更换 lane。

以上例、4 个 core 为例：

```text
round0: core0->row0, core1->row1, core2->row2, core3->row3
round1: core0->row4, core1->row5, core2->row6, core3->row7
```

因此逻辑 row 顺序和来源卡均未改变，但同一 core 不再逐轮换卡。28 核、4 个
均衡 rank 时，由于 `28 % 4 == 0`，同一专家内 core0 领取 row 0、28、56、84，
会固定命中同一 source rank。

GMM 只要求同一 expert 的 row 连续，并不要求 expert 内维持原 rank-major 顺序。Combine 使用 `metaInfo` 中的 `{rankId, tokenId, topkSlot}` 恢复原路由，因此这种 expert 内重排不改变语义。

## 6. MTE 搬运和 ready

MTE ring 消费 route queue 时已经确定：

```text
rankId + topkIndex + destinationRow
```

并直接接入已有 MTE2/MTE3 ring。source rank 改变时只切换远端 GM 基址，不结束 ring；只有以下情况才 drain：

- 当前 GMM1 tile 结束，需要发布 tile ready；
- 当前 expert segment 结束。

每个 destination row 只归属一个 core。core 完成某 tile 中自己负责的 row 后，通过 AtomicAdd 累加实际 ready row 数。所有 core 的计数达到该 tile 的目标 row 数后，GMM1 才会启动。

## 7. 保留的 trace

本次改造保留原 wave trace 结构，并继续统计：

- Dispatch 总区间；
- transfer pipeline；
- route/address 准备；
- MTE3 slot reuse wait；
- MTE2 issue 与 ready wait；
- metadata 准备与 ready wait；
- MTE3 issue 与 stream drain wait；
- ready 发布前同步等待。

trace 仅通过既有 `MEGA_MOE_ENABLE_WAVE_TRACE` 宏控制，没有新增打印。

## 8. 当前需通过实测判断的开销

当前版本仍有三项需要通过 trace 量化的成本：

1. 一次性 RoutePlan 在 28/32/36 核上的临界 producer 与汇总等待；
2. 本核所需 rank slice 连续下发、统一等待后的 `RouteTable GM->UB` 净耗时；
3. 一次 logical-row source 映射与一次原地 queue 回填的 RankRoundPlan 耗时。

正常路径不再使用跨-wave mask scan cursor；专家续段由统一 wave cursor 给出的
`rowOffsetInExpert` 和 Route Table compact ordinal 直接恢复。旧 cursor 仅留在隔离回退路径。
