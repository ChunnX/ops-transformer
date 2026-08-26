# MegaMoe arch35 Dispatch：wave 内 rank-contiguous 实现

本文说明 MTE A8W8 wave 路径当前的 Dispatch 数据布局与调用关系。

## 1. 保留的 wave 流水

`DispatchWaveAndGetInfo` 仍使用与 GMM1 相同的 M-group 预算计算 wave 边界，
并保留下一 wave 的 Dispatch 预取。wave 内继续按专家顺序处理，因为每个专家
拥有独立的 GMM 权重与 Dispatch→GMM1 ready counter。

## 2. 一次性 Route Table

进入 wave 循环前，`BuildDispatchRouteIndexTable` 将每个
`(localExpert, sourceRank)` 的 sendMask 压紧一次。共享 Route Table 的物理顺序为：

```text
expert0: rank0 compact topkIndex, rank1 compact topkIndex, ...
expert1: rank0 compact topkIndex, rank1 compact topkIndex, ...
```

每个 rank 内保持 `topkIndex` 升序。专家和 rank 的起止位置由
`cumsumInfoTensor` 唯一确定。

## 3. wave 专家分段的连续 ownership

`DispatchExpertWaveSegmentRankContiguous` 只处理当前 wave 给出的专家分段
`[segmentRowBegin, segmentRowEnd)`：

1. 将该分段连续、均衡地分给全部 Dispatch core；余数行分给编号较小的 core。
2. 本 core 得到唯一连续区间 `[coreRowBegin, coreRowEnd)`。
3. 使用专家 cumsum 与该区间求交，得到通常一个或两个 source-rank 段。
4. Route Table 和 Dispatch workspace 使用相同的 rank-contiguous row 布局，
   所以本 core 只需一笔 GM→UB 搬运取得自己的精确连续 Route Table slice。
5. 按相交 rank 段调用现有 MTE ring；rank 段内 source GM 视图保持稳定。
6. 本核全部 MTE3 完成后，按与 GMM1 一致的 tile 粒度发布实际完成行数。

例如某专家各 rank 行数为 `[3, 2, 1]`，其物理 row 为：

```text
row: 0     1     2     3     4     5
src: r0t0  r0t1  r0t2  r1t0  r1t1  r2t0
```

若当前 wave 分段为 `[1, 6)`，由两个 core 处理，则：

```text
core0 -> [1, 4): r0t1, r0t2, r1t0
core1 -> [4, 6): r1t1, r2t0
```

两个 core 各自只加载对应的连续 Route Table slice，不会加载未负责的专家行。

## 4. 已移除的机制

正常 wave 路径不再包含以下 rank-round 机制：

- logical-row/lane-round source 映射；
- per-rank scan cursor；
- grouped descriptor；
- packed `{rankId, topkIndex}` route queue；
- 为本核未负责的 rank ordinal 搬运 Route Table slice；
- 逐 logical row 的 source-rank 解析和 GM 视图切换。

通用非 wave/A8W4 路径仍保留原有按 sendMask 扫描的函数；它们不是当前
MTE A8W8 wave 热路径的一部分。
