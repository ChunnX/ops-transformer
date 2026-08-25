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

但具体的 `topkIndex` 仍需读取对应 bitmap，再通过 `CreateVecIndex + GatherMask` 压紧得到。

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

## 3. compact 索引生成

对当前 expert，本核先从 `cumsumInfo` 取得每个 rank 的 token 数，再按本核的轮转 row
游标生成任务。逻辑 row 到 source 的 rank-round-robin 映射保持不变，描述符为：

```text
{rankId, compactOrdinal}
```

随后只对本核确实用到的 rank 执行：

```text
读取 mask slice
  -> CreateVecIndex 生成连续 topkIndex
  -> GatherMask 得到该 rank 的 compact topkIndex
  -> 用 compactOrdinal 直接回填本核描述符中的 topkIndex
```

这里不构造全 expert Layer，也不生成、复制或回放全 wave 的 compact list。规划和回填的
逐 row 工作量都是本核任务数，约为 `segmentRows / coreCount`。

`topkIndexTensor` 分成三个不重叠且对齐的区域：

```text
[最终 {rankId, topkIndex} queue][{rankId, ordinal} descriptor][CreateVecIndex 临时区]
```

两个本核任务表分别按 32B 对齐，临时区按 256 个 int32 对齐。这样复用已有 UB，不新增
GM descriptor workspace，也不再使用任意 4B 偏移的 VEC 搬运。

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

正向任务序号仍直接成为该 expert 内的 workspace row，逻辑顺序完全不变。只在
`coreCount` 行构成的分核 round 内轮转 lane。对 expert 内 row `r`：

```text
round = r / coreCount
lane  = r % coreCount
owner = (expertGlobalRowBegin + lane - round) mod coreCount
```

`round` 使用 expert 内绝对 row，因此 expert 跨 wave 时不会重置；`expertGlobalRowBegin`
保留不同 expert 之间的起始 phase。

以上例、4 个 core 为例：

```text
round0: core0->row0, core1->row1, core2->row2, core3->row3
round1: core0->row5, core1->row6, core2->row7, core3->row4
```

因此逻辑 row 顺序和来源卡均未改变，但同一 core 每轮领取的 lane 改变。28 核、4 个
均衡 rank 时，core0 依次领取 row 0、29、58、87，对应来源 rank 0、1、2、3。

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
- MTE3 reuse wait；
- stream drain wait；
- ready 发布前同步等待。

trace 仅通过既有 `MEGA_MOE_ENABLE_WAVE_TRACE` 宏控制，没有新增打印。

## 8. 当前仍需通过实测判断的开销

当前版本仍有两项需要通过 trace 量化的成本：

1. 每个 AIV1 core 独立 compact 自己用到的 source rank，以避免新增跨核 descriptor 和同步；
2. `CreateVecIndex + GatherMask` 的异步指令下发及既有 `MTE2_V/V_S` 同步等待分别占多少周期。

跨 wave 专家续段已经用每 rank 的 mask scan cursor 恢复扫描位置；同 rank descriptor 也按
compact ordinal 用单调 cursor 跨 mask batch 推进，每项最多解析一次。后续应先依据细分 trace
判断瓶颈属于指令下发还是同步等待，再决定是否值得引入共享索引。当前版本已删除全 wave 的
Plan/Gather/Replay、跨 wave rank-round cursor、旧 Layer 和 rank-major transfer stream。
