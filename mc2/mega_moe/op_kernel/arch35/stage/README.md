# MegaMoe arch35 Dispatch：rank-interleaved wave 说明

本文说明 arch35 wave Dispatch 当前的路由数据链路、目标行布局、Dispatch core 的实际执行方式，以及与“按 wave 逻辑顺序动态填充 core lane”设计之间的差异。

本文对应当前分支 `codex/dispatch-wave-rank-interleave-trace-v3` 的实现，代码重点位于：

- `mega_moe_send_mask.h`
- `mega_moe_token_dispatch.h`
- `../mega_moe_wave.h`

## 1. sendmask 数据链路

源卡首先读取 flatten 后的 `topkIds`。每个 route item 可以表示为：

```text
topkIndex = tokenId * topK + topKSlot
```

`GatherAndSendExpertMasks` 对每一个全局 expert 执行以下操作：

1. `CompareScalar(topkIds, globalExpertId)` 生成该 expert 的命中 bitmap；
2. `GatherMask` 统计当前 route batch 的命中数量；
3. 将 bitmap 写入目标卡对应的 `(localExpertId, sourceRank)` window slot；
4. 在最后一个 batch 的 bitmap 尾部写入该 expert 从该 source rank 收到的累计 route item 数。

因此一个 `(目标 expert, 源 rank)` 的 slot 逻辑上是：

```text
[mask bitmap][count]
```

bitmap 的每一 bit 对应一个 flatten route item，count 是该 expert 从该 source rank 收到的总 route item 数。count 不是每个 bit 的附加字段，而是 slot 尾部独立存储的 4 字节值。

接收端通过 stride 方式读取所有 source rank 的 count，并构造：

```text
cumsumInfo[expert][rank]
```

例如某个 expert 的接收数量为 `[5, 3, 1, 1]`，则前缀和为 `[5, 8, 9, 10]`，原始 rank-major 区间为：

```text
rank0: row 0..4
rank1: row 5..7
rank2: row 8
rank3: row 9
```

## 2. 接收端如何定位 token

Dispatch 对每个 expert/source-rank 读取对应 mask batch：

1. `CreateVecIndex` 生成当前 batch 的连续 flatten route index；
2. `GatherMask` 按 bitmap 压缩出有效 `topkIndex`；
3. `topkIndex / topK` 得到远端 token 行；
4. `topkIndex % topK` 得到该 token 的 top-k 槽位；
5. `sourceRank` 决定从哪一张卡的 quant window 读取数据；
6. token、scale 和 `metaInfo` 写入本卡 Dispatch workspace。

`metaInfo` 至少记录 source rank、token id 和 top-k index，使后续 Combine 可以按原始路由关系回传。

## 3. 当前的 rank-interleaved 目标布局

`BuildDispatchRankInterleaveLayer` 将每个 expert 的目标 row 按“轮次优先、rank 次优先”映射。对于接收数量 `[5, 3, 1, 1]`，目标行布局为：

| Expert row | 来源 |
| ---: | --- |
| 0 | r0t0 |
| 1 | r1t0 |
| 2 | r2t0 |
| 3 | r3t0 |
| 4 | r0t1 |
| 5 | r1t1 |
| 6 | r0t2 |
| 7 | r1t2 |
| 8 | r0t3 |
| 9 | r0t4 |

rank 耗尽后，后续轮次会跳过该 rank。这个布局已经解决了原始 rank-major 数据导致的 GMM 行分布不均问题。

## 4. 当前 Dispatch core 的实际分工

当前 Dispatch core 使用：

```text
destinationRow % coreCount == coreIdx
```

来决定目标 row 的归属。目标 row 使用 expert 的全局 row 前缀，因此 expert 之间、wave 之间不会重置 modulo 相位。

这带来两个结论：

- 当前没有“某个 core 永久绑定某个 source rank”的固定绑定关系；不同 rank 的 row 可以落到不同 core；
- 如果同一个 wave 中前一个 expert 的尾部没有占满所有 lane，后一个 expert 的连续目标 row 可能自然填充空闲 lane。

但是，当前调用顺序仍然是：

```text
expert → source rank → route batch
```

每个 source rank 在 `DispatchRankTokensInterleaved` 中单独建立/结束 transfer stream。因此当前实现是：

```text
目标 workspace 布局：rank-interleaved
core 归属：按目标 row 做静态 round-robin
实际发送遍历：source-rank-major
```

它并不严格按照全局逻辑序列执行：

```text
r0t0 → r1t0 → r2t0 → r3t0 → r0t1 → ...
```

例如 core1 可能同时拥有 row1、row5、row9，但由于先扫描 rank0，再扫描 rank1，实际发送顺序可能是 `r0t4 → r1t0 → r1t1`。这不影响目标 row 的正确性，但会削弱“按轮次尽早收齐 token”的时序效果。

## 5. wave 边界和大专家切分

`DispatchWaveAndGetEndCursor` 按 expert-major 顺序、以 GMM1 M-group 数量规划 wave。

- 如果 expert A 和 expert B 落在同一个 wave，B 的目标 row 会接续 A 的全局 row，空闲 lane 可能自然转去处理 B；
- 如果大 expert A 被切到多个 wave，B 不会跨 wave 填充 A 当前 wave 的空闲 lane；
- 每个后续 wave 当前仍会从 mask batch 0 开始扫描，并跳过当前 wave 之前已经处理过的 ordinal，存在重复 mask 扫描开销。

## 6. 当前实现与理想设计的偏差

理想设计是：

```text
一个 wave 内维护所有 expert/source-rank 的 active stream；
每一轮最多产生 coreCount 个 work item；
某个 rank 耗尽后立即用其他 rank 或下一个 expert 填充空闲 lane；
相邻的同一 source-rank work item 合并成一个发送 run。
```

当前实现的主要偏差为：

1. 只实现了 expert 内的物理 row 交织，没有形成 wave-level 动态 work queue；
2. core 的实际发送遍历仍按 source rank 分段，不能保证全局 rank-interleaved 时序；
3. transfer ring 在 source rank 之间不能连续复用，rank 切换时可能产生额外 drain/wait；
4. 大专家多 wave 场景会重复扫描前序 route batch；
5. 跨 expert 的 lane 填充主要依赖 global row modulo 的隐式效果，而非显式调度。

## 7. 建议的后续优化方向

建议保留当前 sendmask、`CreateVecIndex/GatherMask` 和 rank-interleaved workspace 布局，只调整 Dispatch 调度层。

### 7.1 使用 wave-level active stream

为当前 wave 维护：

```text
expertIdx
每个 source rank 的 nextOrdinal
当前 mask batch 和 match ordinal
目标 expert row
当前 GMM tile
```

当一个 expert 的 active rank 数不足以填满本轮 lane 时，从当前 wave 中下一个 expert 继续取 work item。

### 7.2 使用 run descriptor，而不是逐 token descriptor

建议描述一段等差发送区间：

```text
expert
sourceRank
起始 ordinal
数量
目标 row 起点
目标 row stride
```

同一 source rank、同一 interleave layer、同一 GMM tile 内的 token 可以合并为一个 run，继续复用现有的 MTE2/MTE3 ring，避免为每个 token 分配完整 descriptor。

### 7.3 提升 transfer stream 的复用范围

当前 stream 以 source rank 调用为边界。后续可以提升到 wave/tile 粒度：

```text
同一 tile 内尽可能连续发出多个 rank 的 work item；
只有发生 tile 切换或 ring 资源不足时才 drain；
```

这样才能真正形成跨 rank 的发送流水。

### 7.4 避免多 wave 重复扫描 mask

为每个 expert/source-rank 保存下一次扫描位置：

```text
nextMaskBatch
matchedRouteOrdinal
```

或者在 wave 规划阶段直接求出本 wave 对应的 ordinal 区间，让 `GatherMask` 从对应 batch 开始读取。

### 7.5 保持 flag 按 expert/tile 计数

调度顺序可以改变，但目标 workspace 和 `metaInfo` 位置不能改变。Dispatch ready flag 仍应按：

```text
localExpertId + GMM1 tile index
```

累计实际写入的 row 数，确保 GMM1 只在对应 tile 数据全部到齐后启动。

## 8. 总结

当前版本已经实现：

```text
sendmask
  → count/cumsum
  → CreateVecIndex + GatherMask
  → rank-interleaved expert row
  → token/scale/metaInfo workspace
  → GMM1 tile ready flag
```

当前尚未完全实现的是：

```text
严格按全局 wave 逻辑顺序的动态 Dispatch 调度。
```

因此后续优化重点不是重新设计 sendmask，而是将当前“按 source rank 分段发送”改造成“wave-level active stream + run 合并 + tile 粒度 ring 复用”。
