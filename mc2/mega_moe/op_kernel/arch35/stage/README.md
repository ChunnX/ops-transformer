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

随后 Dispatch 按 expert 顺序处理 wave 内各 segment。大 expert 跨 wave 时，rank-round-robin 游标会延续，不会从 ordinal 0 重新开始调度。

## 3. compact 索引生成

对当前 expert，本核先从 `cumsumInfo` 取得每个 rank 的 token 数。对于当前 chunk 实际覆盖的 ordinal 区间，再逐 rank 执行：

```text
读取 mask slice
  -> CreateVecIndex 生成连续 topkIndex
  -> GatherMask 得到该 rank 的 compact topkIndex
  -> 将所需 ordinal 区间暂存到 topkIndexTensor
```

这里不再构造 Layer，不做 logical row 到 `(rank, rankOrdinal)` 的逆运算，也不再按 rank 启停一次完整搬运流水。

`topkIndexTensor` 在当前 chunk 内分成两个不重叠区域：

```text
[各 rank 的 compact task list][CreateVecIndex 临时区]
```

这样复用已有 UB，不新增 GM descriptor workspace。

## 4. 正向 rank-round-robin 任务顺序

获得各 rank 的 compact list 后，实际任务按以下顺序正向生成：

```text
ordinal 0: r0t0, r1t0, r2t0, ...
ordinal 1: r0t1, r1t1, r2t1, ...
ordinal 2: r0t2, r1t2, r2t2, ...
```

某个 rank 耗尽后直接跳过。

例如两个 expert 的 Gather 存储顺序是：

```text
task0 = E0 r0 a0
task1 = E0 r0 a1
task2 = E0 r1 b0
task3 = E0 r1 b1
task4 = E0 r1 b2
task5 = E1 r0 c0
task6 = E1 r2 d0
task7 = E1 r2 d1
```

实际 Dispatch 顺序为：

```text
task0, task2, task1, task3, task4, task5, task6, task7
```

也就是先取 E0 每张活跃卡的第一个 token，再取第二个；E0 完成后继续 E1。

## 5. 分核和 workspace row

正向任务序号直接成为该 expert 内的 workspace row。core 归属为：

```text
destinationRow % coreCount == coreIdx
```

`destinationRow` 使用所有 expert 的全局连续 row，因此 core phase 不会在 expert 或 wave 边界重置。

以上例、4 个 core 为例：

```text
core0: task0, task4
core1: task2, task5
core2: task1, task6
core3: task3, task7
```

这保证同一轮优先覆盖不同 source rank；当前 expert 的尾部不足以占满一轮时，后续 expert 会自然接续剩余 core phase。

GMM 只要求同一 expert 的 row 连续，并不要求 expert 内维持原 rank-major 顺序。Combine 使用 `metaInfo` 中的 `{rankId, tokenId, topkSlot}` 恢复原路由，因此这种 expert 内重排不改变语义。

## 6. MTE 搬运和 ready

`AppendDispatchTokenToStream` 接收已经确定的：

```text
rankId + topkIndex + destinationRow
```

并直接接入已有 MTE2/MTE3 ring。source rank 改变时只切换远端 GM 基址，不结束 ring；只有以下情况才 drain：

- 当前 GMM1 tile 结束，需要发布 tile ready；
- 当前 chunk 结束，需要复用 compact/index UB；
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

当前版本刻意先实现清晰、可验证的正向调度，仍有两项已知成本：

1. 每个 AIV1 core 都会独立执行 compact gather，以避免新增跨核 descriptor 和同步；
2. 大 expert 跨多个 wave 时，Gather 仍可能从前序 mask batch 开始扫描，再跳到本 wave 所需 ordinal。

这两项应先结合保留的 trace 判断实际占比，再决定是否引入 mask cursor 或共享索引；当前版本不再保留旧 Layer、rank precheck 和 rank-major transfer stream 逻辑。
