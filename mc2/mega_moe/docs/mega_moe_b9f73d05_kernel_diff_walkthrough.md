# MegaMoe `b9f73d05a` Kernel Diff 逐段说明

## 1. 阅读范围

本文只解释提交：

```text
b9f73d05a8d8fe04741aecf5c8836080c4a50b13
optimize(mega_moe): schedule waves by token groups
```

比较基线是它的直接父提交：

```text
27455755ec4fb4aac40cb6ff3d8ee962139ad5d3
```

说明：后续整理分支历史时，本文件被折叠进逻辑等价的第一笔 token-group Wave 提交，
所以该提交会获得新的 SHA。该提交随后还补充了中文注释并规范了新增标识符，但执行逻辑、
地址计算和同步关系没有变化。
本文继续保留 `b9f73d05a` 作为原始追溯号，文中的增删统计和历史行号也以原始快照为准。

阅读当前代码时，可按下表把原始 diff 名称映射到规范后的名称：

| 原始 `b9f73d05a` 名称 | 当前名称 |
|---|---|
| `gmm1TileOffset` | `gmm1TileOffsetInExpert` |
| `GetWaveMGroupCount` | `GetMGroupCountForRows` |
| `GetWaveMGroupTarget` | `GetRequiredMGroupCount` |
| `GetWaveChunkTarget` | `GetMGroupsPerWave` |
| `GetWaveSegmentRows` | `GetWaveSegmentRowCount` |
| `SegmentCursor` | `ExpertRowCursor` |
| `expertRow` | `rowOffsetInExpert` |
| `expertChunkOffset` | `expertMGroupOffset` |
| `SliceExpertSegment` | `BuildExpertSegmentState` |
| `IsSameSegmentCursor(lhs, rhs)` | `IsSameExpertRowCursor(currentCursor, targetCursor)` |
| `gmm1NTileCount_` | `gmm1TilesPerMGroup_` |
| `chunksPerWave_` | `mGroupsPerWave_` |
| `plannedChunks / processedChunks` | `plannedMGroupCount / processedMGroupCount` |
| `waveBeginCursor / gmm1EndCursor`（物理核相位） | `waveStartBlockIndex / gmm1EndBlockIndex` |
| `fixedRoleResonance` | `wouldRepeatSameCoreAssignment` |

只覆盖该提交修改的四个 Kernel 文件，不混入后续的持久化 `BlockMmad`、L2 bypass、
Combine-ready sequence、expert-or-wave Combine 等修改。

| 文件 | 新增 | 删除 | 本提交中的职责 |
|---|---:|---:|---|
| `mega_moe_base.h` | 3 | 0 | 给分段 GMM1 增加专家内 tile 偏移 |
| `mega_moe_impl.h` | 12 | 5 | producer/consumer 使用同一个全专家 tile 状态槽 |
| `mega_moe_utils.h` | 47 | 0 | 计算 Wave 的 256-row chunk 数和 segment 行数 |
| `mega_moe_wave.h` | 220 | 79 | 将固定专家批次改成运行时 count 驱动的 chunk Wave |

Kernel 四文件合计 282 行新增、84 行删除。

文中的“新行”均指 `b9f73d05a` 这个历史快照中的行号，不是当前分支 HEAD 的行号。

## 2. 一句话概括

修改前，Host 用平均 token 数预测 `expertsPerBatch`，Kernel 固定执行：

```text
GMM1([batchBegin, batchEnd))
-> GMM2([batchBegin, batchEnd))
-> next expert batch
```

修改后，Kernel 读取真实专家 count，把专家序列看成“专家内 256-row chunk”的有序流，
每次累计固定数量 `Q = chunksPerWave_`：

```text
take Q expert-local chunks
-> GMM1([waveBegin, waveEnd))
-> GMM2(replay the same range)
-> next chunk wave
```

Wave 可以在热点专家中间结束，但不同专家仍使用各自权重、分别发起 GMM problem。
本提交解决的是“冷专家没有合并，导致一个 Wave 的 GMM1 owner 太少”；它没有把不同专家
拼成一次矩阵乘。

## 3. 最重要的边界：本提交的 Combine 仍按完整专家

`b9f73d05a` 只把 GMM1/GMM2 的计算 Wave 改成 256-row chunk 粒度。此时：

- GMM1 可以在专家内部切段；
- GMM2 按同一个 segment 范围重放；
- 只有走到专家尾时才调用 `PublishGmm2Ready(expertIdx)`；
- `combineExpertEnd = waveEnd.expertIdx`，只 Combine 已经完整结束的专家；
- 如果 `waveEnd=(expert=e, row>0)`，专家 `e` 的前半段虽然已完成 GMM2，
  仍要等后续 Wave 完成整个专家后才 Combine。

因此不要把这笔提交理解成后续的 expert-or-wave Combine-ready。`b9f73d05a` 的关系是：

```text
GMM compute boundary = 256-row chunk Wave boundary
Combine boundary     = complete expert boundary
```

## 4. 整体数据流

### 4.1 三类持久游标

`mega_moe_wave.h` 为不同流水角色维护独立游标：

| 游标 | 使用者 | 作用 |
|---|---|---|
| `dispatchCursor` | AIV1 | 只在首次进入专家时做一次 SendCntCal/Dispatch，同时规划 chunk Wave |
| `gmm1Cursor` | AIC/AIV0 | 按真实 count 消费本 Wave 的 GMM1 segment |
| `gmm2Cursor` | AIC/AIV0 | 重放 GMM1 已确定的 `waveEnd` |

三条路径读取相同 count、使用相同 `chunksPerWave_` 和相同 segment 公式，因而应得到同一个
`(expertIdx, expertRow)` Wave 终点。

### 4.2 为什么需要“完整专家状态 + segment 副本”

`UpdateGroupParams` 的原语义是“从上一个完整专家推进到下一个完整专家”，会同时推进权重、
输入、输出和累计 M 偏移。热点专家跨 Wave 后不能再次调用它，否则会重复推进专家状态。

本提交因此采用：

1. `expertRow == 0` 时才调用一次 `UpdateGroupParams`，保留完整专家状态；
2. `SliceExpertSegment` 复制这个状态；
3. 只在副本上平移 M 相关地址并把 M 改成本 segment 行数；
4. 下一 Wave 仍可从同一个完整专家状态切出后续行。

这是整笔修改能够安全切热点专家的核心。

## 5. `mega_moe_base.h`：给 tile 状态地址增加专家内偏移

### B1：`GMMAddrInfo::gmm1TileOffset`

位置：`b9f73d05a: mega_moe_base.h:45-47`

```diff
 struct GMMAddrInfo {
     ...
+    uint32_t gmm1TileOffset = 0;
 };
```

做什么：

- 保存当前 segment 在“完整专家 GMM1 tile 状态区”中的起始 tile 编号；
- 单位是 tile，不是字节，也不是 cache line；
- 后续 `mega_moe_impl.h` 在它后面再加 segment-local `loopIdx`。

为什么需要：

- 每次 segment GMM 都重新建立 scheduler，`loopIdx` 会从 0 开始；
- 如果专家 `e` 的第二个 segment 仍使用
  `e * maxTilesPerExpert + loopIdx`，就会复用第一个 segment 的状态槽；
- AIV 可能读到旧 round tag，错误地认为后半段 GMM1 已完成。

为什么默认值必须是 0：

- 只有 MTE Wave 路由专家的 segment 路径会设置它；
- legacy、layered 和 shared 路径仍按完整 problem 调用；
- 默认 0 保持这些路径的原地址公式不变。

它是 Kernel 内部参数扩展，不改变外部 tensor ABI 或 tiling ABI。

## 6. `mega_moe_impl.h`：五个 tile flag 地址必须一起平移

五段 diff 都把：

```diff
- expertBase + loopIdx
+ expertBase + gmm1TileOffset + loopIdx
```

加入状态地址公式。公共布局变为：

```text
statusSlot =
    expertIdx * maxTilesPerExpert
  + segmentStartChunk * gmm1NTileCount
  + segmentLocalLoopIdx
```

每个 slot 再乘 `INT_CACHELINE`，所以不同 tile 仍各占一条 64B cache line。

### I1：Generic AIC producer 写状态

位置：`mega_moe_impl.h:517-519`，函数 `AicComputeGeneric`。

```diff
- expertIdx * maxTilesPerExpert + loopIdx
+ expertIdx * maxTilesPerExpert
+ gmmAddrInfo.gmm1TileOffset
+ loopIdx
```

这里的 AIC 在 GMM1 L0C→GM Fixpipe 完成后写 round tag。增加偏移后，专家后续 segment
不会覆盖前一 segment 的完成状态。

### I2：Generic AIV interleaved consumer 轮询状态

位置：`mega_moe_impl.h:625-627`，函数 `AivGmm1PostGeneric` 的
`TopkWeightsPrefetch && IsGmm1Interleaved` 分支。

AIV 必须与 I1 计算完全相同的 slot，才能在对应 GMM1 tile 写回后读取 GM 结果和 metaInfo。
只改 producer 或只改 consumer 都会造成永久轮询或提前命中。

### I3：Generic AIV non-interleaved consumer 轮询状态

位置：`mega_moe_impl.h:687-689`，同一函数的
`TopkWeightsPrefetch && !IsGmm1Interleaved` 分支。

这段与 I2 的目的相同，只是 non-interleaved epilogue 的 GM→UB 和 gate/up 处理方式不同。
两个模板分支必须同时适配 segment 起点。

### I4：A8W4 AIC producer 写状态

位置：`mega_moe_impl.h:1100-1102`，函数 `AicComputeA8W4`。

A8W4 使用另一套 BlockMmad/prologue 实现，不能依赖 Generic 路径的改动自动生效。
这里把同一个 `gmm1TileOffset` 加入 A8W4 producer 的状态槽。

### I5：A8W4 AIV consumer 轮询状态

位置：`mega_moe_impl.h:1158-1160`，函数 `AivGmm1PostA8W4`。

它与 I4 构成 A8W4 producer/consumer 配对。至此 Generic interleaved、
Generic non-interleaved 和 A8W4 的两端都使用同一全专家 tile 编号。

### 本文件没有改变什么

- 没有改变 BlockMmad 的 tile owner 算法；
- 没有改变 round tag，仍是 `expertIdx + 1`；
- 没有改变 FIX_S 的先后关系；
- 没有增加新的状态区容量；
- 只是把 segment-local tile 编号平移回原来为完整专家预留的 slot。

## 7. `mega_moe_utils.h`：新增四个纯调度函数

位置：`mega_moe_utils.h:33-78`。这些函数同时可在 Host 和 Device 编译，但本提交的
MTE Wave Kernel 在 `Init` 和游标推进中使用它们。

### U1：`GetWaveMGroupCount`

```cpp
ceil(tokenCount / tileM)
```

对应实现：

```cpp
(tokenCount - 1U) / tileM + 1U
```

语义：

- 一个 M group 就是当前专家贡献的一个最多 256 行 chunk；
- `count=0` 返回 0；
- `count=1..256` 返回 1；
- `count=257..512` 返回 2。

关键点是不能先把多个冷专家的 token 数相加再做一次 ceil。不同专家不能共享权重，
所以两个各 1 行的专家仍贡献两个独立 M group，而不是一个 2 行 GEMM。

### U2：`GetWaveMGroupTarget`

公式：

```text
ownerTarget = coreCount * minOwnersPerCore
mGroupTarget = ceil(ownerTarget / nTileCount)
```

因为一个 M group 会产生 `nTileCount` 个 scheduler owner，所以这个函数反推：
至少需要多少个 M group，才能给每个 AIC 提供指定深度的 owner。

参数为 0 时返回 1，是防御性保底，避免生成 0 大小 Wave；正常 tiling 下这些参数都应非零。

### U3：`GetWaveChunkTarget`

```text
Q1 = ceil(coreCount * gmm1OwnersPerCore / gmm1NTileCount)
Q2 = ceil(coreCount / gmm2NTileCount)
Q  = max(Q1, Q2)
```

含义：

- `Q1` 保证正常 Wave 的 GMM1 producer 深度；
- `Q2` 保证正常 Wave 的 GMM2 至少有 `coreCount` 个 owner；
- 取较大值，使同一个 Wave 同时满足两级 GMM 的最低铺核要求。

这个 `Q` 只依赖 AIC 数、GMM1/GMM2 N tile 数和目标深度，不依赖专家数量，也不依赖
Host 的平均 token 预测。

### U4：`GetWaveSegmentRows`

公式：

```text
remainingRows  = expertRows - expertRow
maxSegmentRows = chunksLeft * tileM
segmentRows    = min(remainingRows, maxSegmentRows)
```

它从当前专家中取“能放入 Wave 剩余 chunk 预算的最大连续行段”：

- 专家剩余行更少：在专家尾结束；
- Wave 预算先耗尽：在专家中间结束；
- 不跨专家拼成一次 GMM；
- 除专家最后一个不足 256 行的尾块外，`expertRow` 始终保持 256 对齐。

原来的 `GetWaveExpertCount` 没有在这笔提交中删除；只是新的 MTE Wave 主路径不再用它来
决定计算 Wave。

## 8. `mega_moe_wave.h`：状态和接口层 diff

### W1：`ExpertLoopState::expertChunkOffset`

位置：`mega_moe_wave.h:85-87`。

```diff
 struct ExpertLoopState {
     ...
+    uint32_t expertChunkOffset = 0;
 };
```

它表示当前 segment 在专家内从第几个 256-row chunk 开始：

```text
expertChunkOffset = expertRow / 256
```

完整专家持久状态保持 0；只有 `SliceExpertSegment` 产生的临时副本设置非零值。

### W2：新增 `SegmentCursor`

位置：`mega_moe_wave.h:109-112`。

```cpp
struct SegmentCursor {
    uint32_t expertIdx;
    uint32_t expertRow;
};
```

旧游标只有专家编号，无法表达“热点专家已经处理到第 768 行”。新游标同时表示专家和专家内行号。
规范形式是：

- 专家中部：`(e, row>0)`；
- 专家结束：立即规范化为 `(e+1, 0)`；
- 全部结束：`(moeExpertPerRank_, 0)`。

### W3：函数声明从 expert range 改成 cursor range

位置：`mega_moe_wave.h:121-139`、`164-169`。

主要接口变化：

```diff
- DispatchExpertsUntil(nextExpert, dispatchEnd)
+ DispatchWaveAndGetEnd(cursor) -> SegmentCursor

- ProcessGmm1Wave(batchBegin, batchEnd)
+ ProcessGmm1Wave(cursor) -> SegmentCursor

- ProcessGmm2Wave(batchBegin, batchEnd)
+ ProcessGmm2Wave(cursor, waveEnd)
```

同时新增 `SliceExpertSegment` 和 `IsSameSegmentCursor`。这组声明变化把调度边界从
`[expertBegin, expertEnd)` 提升为 `[(expert,row), (expert,row))`。

### W4：成员变量从 `expertsPerBatch_` 改成 tile/chunk 参数

位置：`mega_moe_wave.h:226-227`。

```diff
- uint32_t expertsPerBatch_;
+ uint32_t gmm1NTileCount_;
+ uint32_t chunksPerWave_;
```

- `expertsPerBatch_` 是 Host 平均值预测的专家数边界；
- `gmm1NTileCount_` 用于计算 Wave 大小，也用于全专家 GMM1 tile 状态寻址；
- `chunksPerWave_` 是 Kernel 实际执行的固定 Q。

Host tiling 字段 `expertsPerBatch` 仍存在，但这条 Kernel 路径不再用它控制 Wave。

### W5：GMM1 深度常量

位置：`mega_moe_wave.h:332-333`。

```cpp
GMM1_WAVE_MIN_OWNERS_PER_CORE = GMM1_INTERLEAVED ? 4U : 2U;
```

- non-interleaved：一个 scheduler owner 内顺序做 gate/up 两个 weight block，因此目标为每核
  至少 2 个“成对 owner”；
- interleaved：scheduler 遍历完整 gate+up N，每个 owner 是单个 N tile，因此目标为每核
  至少 4 个 owner。

这使两种实现保持相近的物理 GMM1 下发深度，而不是简单地给两种模式使用同一个数字。

### W6：`Init` 在 Kernel 内计算 Q

位置：`mega_moe_wave.h:365-381`。

删除：

```diff
- expertsPerBatch_ = tilingData->expertsPerBatch;
- expertsPerBatch_ = min(expertsPerBatch_, moeExpertPerRank_);
```

新增：

```cpp
gmm1SchedulerN =
    interleaved ? hiddenDim : hiddenDim / 2;
P1 = ceil(gmm1SchedulerN / 256);
P2 = ceil(h / 256);
Q  = GetWaveChunkTarget(blockNum, P1, P2, gmm1Depth);
```

为什么 `gmm1SchedulerN` 要分模式：

- non-interleaved 的 scheduler N 只覆盖 gate/up 的半宽，owner 内再做两块权重；
- interleaved 的 scheduler N 覆盖完整 gate+up 宽度；
- `P1` 必须与真正的 BlockScheduler tile 数一致，否则 Q 会高估或低估铺核深度。

## 9. `mega_moe_wave.h`：Dispatch diff

### W7：`DispatchExpert` 返回 `sendCnt`

位置：`mega_moe_wave.h:1012-1027`。

```diff
- void DispatchExpert(...)
+ uint64_t DispatchExpert(...)
```

函数仍然只由 AIV1 执行：

1. `SendCntCal(expertIdx, sendCnt)`；
2. 用真实 count 更新完整专家状态；
3. 对非零专家执行一次 meta 生成和 Dispatch；
4. 返回 `sendCnt`。

注意：`DispatchWaveAndGetEnd` 在本提交中没有接收这个返回值；它实际通过
`state.problemShape.M` 读取刚更新的专家 count。因此返回值不是 Q 计算输入，也不改变当前语义，
更像是为调用者暴露 count 的接口准备。

### W8：`DispatchExpertsUntil` 改为 `DispatchWaveAndGetEnd`

位置：`mega_moe_wave.h:1030-1067`。

旧逻辑按专家编号处理到固定 `dispatchEnd`。新逻辑：

```text
plannedChunks = 0
while expert remains and plannedChunks < Q:
    if entering expert at row 0:
        DispatchExpert exactly once
    skip zero-count expert
    take the largest segment fitting remaining Q
    advance (expertIdx, expertRow)
return cursor as waveEnd
```

关键语义：

- Dispatch 仍是完整专家粒度，不会把同一个热点专家 Dispatch 多次；
- chunk cursor 只是规划计算 Wave 边界；
- 热点专家跨 Wave 时，下一轮 `expertRow != 0`，所以不会再次 SendCntCal/Dispatch；
- 零专家不消耗 chunk，但游标继续前进，避免死循环。

## 10. `mega_moe_wave.h`：segment 地址 diff

### W9：新增 `SliceExpertSegment`

位置：`mega_moe_wave.h:1474-1511`。

函数先复制完整专家状态，再只平移与 M 行有关的视图：

| 字段 | 增量 | 指向的数据 |
|---|---:|---|
| `expertBeforeCnt` | `expertRow` | metaInfo 的全局 token 行基址 |
| `expertChunkOffset` | `expertRow / 256` | per-chunk flag/tile 状态基址 |
| `M` | 设为 `segmentRows` | 当前 GMM problem 的行数 |
| `IDX_A_OFFSET` | `expertRow * K` | Dispatch 后的 GMM1 输入 |
| `IDX_A_SCALE_OFFSET` | `expertRow * scaleK` | GMM1 输入 scale |
| `IDX_C_OFFSET` | `expertRow * (N/2)` | SwiGLU/量化数据 |
| `IDX_C_SCALE_OFFSET` | `expertRow * scaleN` | SwiGLU scale |
| `IDX_Y2_OFFSET` | `expertRow * K` | GMM2/输出行视图 |
| `IDX_M_OFFSET` | `expertRow` | M 方向累计行偏移 |
| `IDX_GMM1_OFFSET` | `expertRow * N` | GMM1 临时输出 |
| `IDX_GMM2_OFFSET` | `expertRow * K` | GMM2 临时输出 |

故意不平移：

- `IDX_B_OFFSET / IDX_B_SCALE_OFFSET`；
- `IDX_B2_OFFSET / IDX_B2_SCALE_OFFSET`；
- 专家索引本身。

原因是 segment 仍属于同一专家，必须复用同一份 W1/W2 权重。

`IsSameSegmentCursor` 只是同时比较 `expertIdx` 和 `expertRow`，供 GMM2 精确重放到 GMM1
确定的终点。

### W10：`UpdateGlobalBuffer` 平移三类 ready/status 地址

位置：`mega_moe_wave.h:1539-1564`。

改动包括：

1. epilogue vector flag 起点加 `expertChunkOffset`；
2. `gmm1TileOffset = expertChunkOffset * gmm1NTileCount_`；
3. `swigluToGmm2Flag` 加 `expertChunkOffset * INT_CACHELINE`；
4. `dispatchToGmm1Flag` 加 `expertChunkOffset * INT_CACHELINE`。

为什么三类 flag 都要移动：

```text
Dispatch(row chunk S) -> GMM1(S) -> SwiGLU(S) -> GMM2(S)
```

热点专家第二个 segment 的本地 wave index 会重新从 0 开始。如果仍指向专家首槽，就可能：

- GMM1 误读第一段的 Dispatch-ready；
- GMM2 误读第一段的 SwiGLU-ready；
- prefetch epilogue 误读第一段的 tile status。

所有地址都以完整专家预留区为基址，加专家内 chunk 偏移，不改变 workspace 总布局。

### W11：`GroupMatmulWithSwigluQuant` 的 epilogue 地址同步平移

位置：`mega_moe_wave.h:2193-2194`。

`GroupMatmulWithSwigluQuant` 内还会对传入的 `currentEpilogueOp` 调一次
`UpdateGlobalAddr`。这里也必须加 `expertChunkOffset`，否则 W10 设置好的 segment flag
位置可能被这个调用重新覆盖为专家首槽。

这段和 W10 不是重复无效修改，而是保证两个 epilogue 地址更新入口保持一致。

## 11. `mega_moe_wave.h`：GMM diff

### W12：`ProcessGmm1Wave` 从专家 for-loop 改成 chunk while-loop

位置：`mega_moe_wave.h:2258-2301`。

旧实现：

```text
for expert in [batchBegin, batchEnd):
    UpdateGroupParams(expert)
    GroupMatmul(full expert)
```

新实现：

```text
processedChunks = 0
while cursor not at end and processedChunks < Q:
    if expertRow == 0:
        UpdateGroupParams(current expert)
    skip zero expert
    segmentRows = min(expert remainder, remaining Q * 256)
    segmentState = SliceExpertSegment(...)
    GroupMatmul(segment)
    advance cursor and processedChunks
return waveEnd
```

关键点：

- 同一专家在一个 Wave 内的连续 chunk 合成一次 segment GMM，避免每 256 行都重新建 problem；
- Wave 可以在专家中部返回；
- `UpdateGroupParams` 只在专家入口调用，完整专家状态不会被 segment M 污染；
- AIC 做真实 GMM1，AIV0 走配对 epilogue；AIV1 不进入这里的 GMM 分支，主循环改由
  `DispatchWaveAndGetEnd` 取得它的 Wave 终点。

### W13：`ProcessGmm2Wave` 精确重放 GMM1 范围

位置：`mega_moe_wave.h:2305-2347`。

GMM2 不重新按 Q 规划，而是接收 `waveEnd`：

```text
while gmm2Cursor != waveEnd:
    segmentEnd =
        expert tail, or waveEnd.row when wave ends inside this expert
    run GMM2 on [cursor.row, segmentEnd)
    advance cursor
```

这样 GMM1/GMM2 在阶段切换时覆盖完全相同的行范围，不会出现 GMM2 读到尚未做 SwiGLU 的
下一段。

本提交仍只在：

```cpp
cursor.expertRow >= expertRows
```

时调用 `PublishGmm2Ready`。因此热点专家的部分 segment 不会提前放行 Combine。

## 12. `mega_moe_wave.h`：主流水 diff

### W14：初始化四份完整专家状态和多条 cursor

位置：`mega_moe_wave.h:2421-2432`。

四份 `ExpertLoopState` 都增加了 `expertChunkOffset=0` 初始化。新增独立的：

- `waveBegin`；
- `dispatchCursor`；
- `gmm1Cursor`；
- `gmm2Cursor`；
- `preparedWaveEnd`；
- `combineExpertBegin`。

分离状态的原因是 AIV1 会提前 Dispatch 下一 Wave，而 AIC/AIV0 还在计算当前 Wave；
不能让 Dispatch 的预取状态提前推动 GMM 的地址状态。

### W15：不同核角色独立得到同一个 `waveEnd`

位置：`mega_moe_wave.h:2433-2454`。

角色分工：

| 角色 | 如何得到 `waveEnd` |
|---|---|
| AIC | `ProcessGmm1Wave` 的真实 GMM1 cursor |
| AIV0 | `ProcessGmm1Wave` 的配对 epilogue/虚拟 cursor |
| AIV1 | `DispatchWaveAndGetEnd` 的 dispatch cursor |

AIV1 仍保持原来的“Combine 前预取下一 Wave”约束：

- 首轮先规划/Dispatch 当前 Wave；
- 再规划/Dispatch 下一 Wave，保存到 `preparedWaveEnd`；
- 稳态当前 `waveEnd` 直接复用上轮准备好的终点；
- 然后继续预取再下一 Wave。

这样没有改变 Dispatch 与 Combine 共用 UB/事件资源的先后关系。

### W16：GMM2 调用和 owner 相位修复改用 cursor

位置：`mega_moe_wave.h:2457-2467`。

`ProcessGmm2Wave` 从 `(batchBegin,batchEnd)` 改成 `(gmm2Cursor,waveEnd)`。
原有 `fixedRoleResonance` 修复逻辑被保留，只把“是否还有下一 Wave”的判断从
`batchEnd` 改成 `waveEnd.expertIdx`。

该修复处理一种 modulo 相位共振：

```text
GMM1 + GMM2 的 tile 数余数恰好让 startBlockIdx 回到 Wave 入口，
下一 Wave 可能继续固定同一批物理核角色。
```

如果检测到这种情况，就把下一 Wave 的起点恢复到 GMM1 结束相位。这个提交没有更换朴素滚动
owner 算法，只让它继续适配新的 cursor Wave。

### W17：Combine 只推进到完整专家

位置：`mega_moe_wave.h:2469-2475`。

```cpp
combineExpertEnd = waveEnd.expertIdx;
ProcessCombineExperts(combineExpertBegin, combineExpertEnd);
combineExpertBegin = combineExpertEnd;
waveBegin = waveEnd;
```

解释：

- `waveEnd=(e+1,0)`：专家 `e` 已完整结束，可以 Combine；
- `waveEnd=(e,row>0)`：`waveEnd.expertIdx==e`，Combine 只做到 `e` 之前；
- 下一 Wave 完成专家 `e` 后，`combineExpertEnd` 才会越过它。

这正是本提交“计算按 chunk、Combine 按专家”的边界。

## 13. 从一组 count 看完整行为

假设 `Q=3`，运行时专家 count 为：

```text
[1, 1, 1, 1, 1024, 0, 255]
```

形成：

| Wave | GMM segment | Wave 终点 | 本轮可 Combine |
|---|---|---|---|
| 0 | `e0[0,1)+e1[0,1)+e2[0,1)` | `(e3,0)` | `e0,e1,e2` |
| 1 | `e3[0,1)+e4[0,512)` | `(e4,512)` | `e3` |
| 2 | `e4[512,1024)+e6[0,255)` | `(end,0)` | `e4,e6` |

可见：

- 四个 1-row 冷专家不再各自形成低 owner Wave；
- 热点专家 `e4` 可以跨 Wave 做 GMM1/GMM2；
- 但 `e4[0,512)` 在 Wave 1 结束时还不能 Combine，因为 `e4` 尚未完整 ready。

## 14. 这笔提交没有修改的内容

- 没有修改 Host tiling 结构或外部 ABI；
- 没有删除 Host 对 `expertsPerBatch` 的计算，只是这条 Wave Kernel 不再用它分批；
- 没有修改朴素 `startBlockIdx` modulo owner 分配；
- 没有合并不同专家的 GEMM；
- 没有修改 Combine 的 token/rank/topk 映射；
- 没有让热点专家部分结果提前 Combine；
- 没有持久化 `BlockMmad`；
- 没有做权重 L2 bypass；
- 没有引入后续的单调 Combine-ready sequence。

## 15. 四个文件如何串起来

```text
mega_moe_utils.h
  计算 P1/P2 -> Q，并计算每次能从当前专家拿多少行
        |
        v
mega_moe_wave.h
  用 (expert,row) cursor 规划 Dispatch/GMM1/GMM2 Wave
  SliceExpertSegment 平移所有 M 相关数据地址
        |
        +--> mega_moe_base.h
        |      GMMAddrInfo 携带 gmm1TileOffset
        |
        v
mega_moe_impl.h
  producer/consumer 都用 gmm1TileOffset + localLoopIdx
  访问完整专家状态区中的正确 tile flag
```

如果缺少任何一层：

- 没有 `utils`：无法从 H/N/核数得到稳定 Q；
- 没有 `wave`：无法在专家中部保存进度并切正确地址；
- 没有 `base` 字段：segment 起点无法传入底层 GMM 实现；
- 没有 `impl` 五处平移：prefetch tile flag 会别名、提前命中或永久等待。

## 16. 快速复核原始 diff

```bash
git diff \
  27455755ec4fb4aac40cb6ff3d8ee962139ad5d3 \
  b9f73d05a8d8fe04741aecf5c8836080c4a50b13 \
  -- \
  mc2/mega_moe/op_kernel/arch35/mega_moe_base.h \
  mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h \
  mc2/mega_moe/op_kernel/arch35/mega_moe_utils.h \
  mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h
```

只看零上下文 hunk：

```bash
git diff --unified=0 \
  27455755ec4fb4aac40cb6ff3d8ee962139ad5d3 \
  b9f73d05a8d8fe04741aecf5c8836080c4a50b13 \
  -- mc2/mega_moe/op_kernel/arch35/mega_moe_{base,impl,utils,wave}.h
```
