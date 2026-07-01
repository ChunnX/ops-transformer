# Dispatch 去冗余 B 方案设计: 机会式 ready-row 表

> 目标: 在不引入专家间屏障、不预计算 first 专家的前提下,减少同一源 token 被当前接收卡多个本地专家重复跨卡拉取的问题。
>
> 关联代码: `mc2/mega_moe/op_kernel/arch35/mega_moe.h`, `mega_moe_base.h`, `mega_moe_workspace_info.h`, `op_host/op_tiling/arch35/mega_moe_tiling.cpp`。
>
> 本文是对原 A 方案(预计算 first 专家)的调整设计。A 方案仍可作为保守备选,但建议先 microtest 本 B 方案。

---

## 1. 当前问题与约束

当前 dispatch 阶段的核心流程如下:

```text
Process()
    SendMaskCal()             源端按专家生成 mask/count 并发给目标卡
    QuantProcessInRank()      源端把当前卡的 token 量化到 quantTokenScalePtr
  CrossRankSyncInWorldSize()

  for localExpertId in 0..expertPerRank_-1:
      SendCntCal(localExpertId)
      TripleInfoCalAndDispatch(localExpertId)
          GatherMask(...)                  得到当前专家从每张源卡接收哪些 topK 路由项
          CopyGMToGMPerToken(...)          从源卡的 quantTokenScalePtr 拉 token+scale
          AtomicAdd(dispatchToGmm1Flag)    wave 粒度解锁 GMM1
```

冗余发生在 `CopyGMToGMPerToken()`:

```text
同一个源 token,也就是来自同一张源卡 `sourceRank`、同一个原始 token 行号 `sourceTokenIndex` 的数据,如果被当前接收卡的多个本地专家命中,
它在源卡 `quantTokenScalePtr` 中的 token+scale 完全相同,
但当前实现会被每个命中专家各跨卡读取一次。
```

需要注意三条约束:

1. `Process()` 虽然按 `localExpertId` 写成顺序循环,但这是每个核自己的控制流,不是全局串行。
2. `TripleInfoCalAndDispatch()` 内部按 `idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_` 分核,同一源卡的不同 token 分片由不同 AIV 核处理。
3. expert 之间不能插 `SyncAll`。GMM1 依赖 `flagDispatchToGmm1Ptr` 按 256-row wave 提前启动,屏障会破坏 dispatch/GMM1 overlap。

因此,去重方案必须满足:

- 读到 ready 时,本地源行的数据必须已经 MTE3 落地。
- 读到 not-ready 时,不能等待;需要直接 fallback 到当前远端拉行为。
- 多核同时处理同一个 `(sourceRank, sourceTokenIndex)` 时,最多允许重复远端拉,不能读错或写坏表。

---

## 2. B 方案核心思想

B 方案只维护一张 ready-row 表。表的两个维度分别是"源卡号"和"源 token 行号":

```text
readyRow[sourceRank][sourceTokenIndex]
```

含义:

```text
readyRow[sourceRank][sourceTokenIndex] == 0
    表示当前还没有可复用的本地行。

readyRow[sourceRank][sourceTokenIndex] < 0
    表示已经有一个本地行可以复用。
    低 31 位是 dispatchRevData/dispatchRevScale 中的绝对行号 row。
```

编码:

```text
READY_BIT = 0x80000000
EMPTY     = 0

写 ready: encoded = row | READY_BIT
读 row:   row = encoded & 0x7fffffff
判断:     encoded < 0 表示 ready
```

`row | 0x80000000` 的作用是把 int32 最高位当 ready 标志位。`row` 本身远小于 `2^31`,不会占用最高位;置位后按有符号 int32 解释就是负数。这样一个 int32 同时表达状态和行号。

---

## 3. B 方案运行流程

### 3.1 dispatch 前: 清空 ready-row 表

每轮 dispatch 开始前,当前接收卡本地把 `readyRow` 清零。

推荐位置:

```text
Process()
  SendAndQuantBuffInit()
  SendMaskCal()
  ResetFlagList()
  ResetReadyRowTable()       新增: 清零 readyRow
  QuantProcessInRank()
  SyncAll + CrossRankSync
  DispatchBuffInit()
  ... dispatch ...
```

也可以放在 `DispatchBuffInit()` 前后,但必须保证:

```text
所有 AIV 清零 readyRow 完成
  -> SyncAll<true>() 或等价 barrier
  -> 任何 CopyGMToGMPerToken() 读 readyRow
```

表大小:

```text
epWorldSize * bs * sizeof(int32_t)
```

布局建议:

```text
readyRow[srcRank][token]
```

即源卡连续,每张源卡下有 `bs` 个 token 槽位。这样在 `CopyGMToGMPerToken(remoteRankIdx, tokenIndex)` 中地址计算简单:

```cpp
__gm__ int32_t* readyBase = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.readyRowPtr) +
    static_cast<uint64_t>(remoteRankIdx) * m_;
__gm__ int32_t* readyAddr = readyBase + tokenIndex;
```

### 3.2 dispatch 中: 读表决定数据来源

在 `CopyGMToGMPerToken()` 的 Phase 1 中,当前代码已经逐 token 做:

```cpp
int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + i);
int32_t tokenIndex = topkIndex / topK_;
tripleTensor_[i * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
tripleTensor_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
tripleTensor_[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
```

B 方案在这里增加一次 ready-row 读取和源类型记录:

```cpp
int32_t encoded = AscendC::ReadGmByPassDCache(readyAddr);
if (encoded < 0) {
    int32_t srcRow = encoded & 0x7fffffff;
    tripleTensor_[i * INT32_PER_256B].SetValue(SRC_KIND, SRC_LOCAL);
    tripleTensor_[i * INT32_PER_256B].SetValue(SRC_ROW, srcRow);
} else {
    tripleTensor_[i * INT32_PER_256B].SetValue(SRC_KIND, SRC_REMOTE_AND_PUBLISH);
    tripleTensor_[i * INT32_PER_256B].SetValue(SRC_ROW, 0);
}
```

新增 triple 槽位:

```cpp
constexpr uint32_t SRC_KIND = 3U;
constexpr uint32_t SRC_ROW  = 4U;
constexpr int32_t SRC_REMOTE_AND_PUBLISH = 0;
constexpr int32_t SRC_LOCAL = 1;
```

`tripleTensor_` 每 token 占 8 个 int32,当前只使用 0/1/2,因此可复用 3/4,不增加 UB。

### 3.3 MTE2 发射: 远端拉或本地拷

当前 Phase 2 的 MTE2 只会从远端 rank window 读:

```cpp
DataCopy(copyTmpTensors_[buf], remoteRankGlobalTensor[tokenIndex * copyInNum], copyInNum);
```

B 方案根据 `SRC_KIND` 分两类。

#### 远端路径

用于 `SRC_REMOTE_AND_PUBLISH`:

```cpp
uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
DataCopy(copyTmpTensors_[buf], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
```

这是当前行为。

#### 本地路径

用于 `SRC_LOCAL`。从当前接收卡的 `dispatchRevDataPtr/dispatchRevScalePtr` 中读取已经落地的 token+scale,再写到当前专家自己的 row。

注意当前 workspace 中 token 和 scale 是分开紧密存储的:

```text
dispatchRevDataPtr[row * widthA]
dispatchRevScalePtr[row * widthAScale]
```

但 `copyTmpTensors_[buf]` 的格式仍保持当前输入格式:

```text
copyTmp = [token data padded to mxQuantTokenAlignBytes_][scale]
```

因此本地路径需要两段 MTE2:

```cpp
GlobalTensor<ActivationType> localDataGlobal;
GlobalTensor<QuantScaleOutType> localScaleGlobal;
localDataGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(params_.workspaceInfo.dispatchRevDataPtr));
localScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType*>(params_.workspaceInfo.dispatchRevScalePtr));

int32_t srcRow = tripleTensor_[idx * INT32_PER_256B].GetValue(SRC_ROW);
LocalTensor<ActivationType> tokenScaleBuf = copyTmpTensors_[buf];
LocalTensor<QuantScaleOutType> scaleBuf =
    tokenScaleBuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();

DataCopyPad(tokenScaleBuf, localDataGlobal[srcRow * widthA],
    {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
DataCopyPad(scaleBuf, localScaleGlobal[srcRow * widthAScale],
    {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
```

后续 MTE3 写出保持不变:

```cpp
DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScalebuf, ...);
DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], bufScale, ...);
```

也就是说,无论数据来自远端还是本地,最终都要写入当前 expert 自己的 `dispatchRevData/dispatchRevScale` 行。去重只减少远端读,不改变 GMM1 需要的 per-expert row 布局。

### 3.4 远端路径完成后: 发布 ready row

所有 `SRC_REMOTE_AND_PUBLISH` 的 token 在本核 MTE3 写出完成后,发布 ready row:

```cpp
for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
}

SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();// 或独立 event id

for (int32_t i = 0; i < copyNum; ++i) {
    if (tripleTensor_[i * INT32_PER_256B].GetValue(SRC_KIND) == SRC_REMOTE_AND_PUBLISH) {
        int32_t tokenIndex = tripleTensor_[i * INT32_PER_256B].GetValue(TOKEN_ID);
        int32_t row = rowDstOffsetInCore + i;
        AscendC::WriteGmByPassDCache(readyBase + tokenIndex, row | 0x80000000);
    }
}
```

必须保证顺序:

```text
当前 row 的 token/scale MTE3 写入 workspace
  -> MTE3_S 同步
  -> WriteGmByPassDCache(readyRow)
```

消费者只要用 `ReadGmByPassDCache` 读到 ready,就能安全从本地 workspace 读该 row。

---

## 4. 并发语义

B 方案不追求互斥生产者。多个核可能同时处理同一个 `(sourceRank, sourceTokenIndex)`:

```text
核 A 读表: 0 -> 远端拉 -> 写 row=100 ready
核 B 读表: 0 -> 远端拉 -> 写 row=320 ready
核 C 稍后读表: ready row=100 或 320 -> 本地拷
```

这不是错误,因为:

- row=100 和 row=320 都是同一个源 token `(sourceRank, sourceTokenIndex)` 的 token+scale 完整拷贝。
- 后续消费者从任意一个 ready row 读取,数据语义一致。
- 表里的 ready row 只作为"当前接收卡已经有一份可复用 token 数据"的索引,不承担计数或所有权语义。

因此 last-writer-wins 可接受。竞争只会降低去重收益,不会影响正确性。

### 4.1 为什么不能等待

如果读到表为空或未来引入 `IN_PROGRESS`,消费者都不应等待。当前 dispatch 的性能依赖:

```text
dispatch 某个 wave 写满 -> AtomicAdd dispatchToGmm1Flag -> GMM1 可提前启动
```

等待会让局部 token 的依赖变成全局长尾,容易拖住 wave flag,破坏 overlap。B 方案只做两种选择:

```text
ready -> 本地拷
not ready -> 远端拉
```

没有等待路径。

---

## 5. 相对 A 方案的调整原因

A 方案是原文档主方案:源端提前计算 `first[sourceRank][sourceTokenIndex]`,指定最小本地专家为唯一生产者。

B 方案调整的主要原因如下。

### 5.1 避免 dispatch 前置计算

A 方案新增 `SendFirstCal()`。它需要对目标 rank 和本地专家扫描 topk,计算每个 token 的最小命中专家,再跨卡发表。

这段发生在真正 dispatch 之前,会增加启动延迟。更关键的是,现有 `SendMaskCal()` 已经对全局专家做 `CompareScalar/GatherMask` 来生成 mask/count。`SendFirstCal()` 如果独立实现,会重复扫描 topk。

B 方案只需要 reset 一张 `epWorldSize * bs * int32` 表,不需要额外扫描 topk。

### 5.2 适配当前多核无序执行

A 方案选择"最小专家号"当生产者,这是静态规则,不代表运行时最早完成远端拉的核。

当前 dispatch 中,不同专家、源卡和分片由不同 AIV 核处理。可能出现:

```text
le=3 的核先处理到 sourceTokenIndex 对应的 token
le=2(first) 的核后处理到同一个 sourceTokenIndex 对应的 token
```

在 A 方案中,le=3 看到 first=2 但未 ready,只能 fallback 远端拉;le=2 后面再拉并发布 ready。这样固定 first 未必带来最早 ready。

B 方案让"实际最先完成远端拉的人"发布 ready row,更贴合运行时调度。

### 5.3 降低首版实现复杂度

A 方案需要:

- 新增 `SendFirstCal()`。
- 新增 `firstCalTensor_`。
- 源端按目标 rank 发表 first 表。
- 处理 first 哨兵和 ready 复用两种语义。

B 方案需要:

- 新增 ready-row 表。
- dispatch 前清零。
- `CopyGMToGMPerToken()` 读表、分源、本地拷、发布 ready。

B 方案的主要改动集中在当前瓶颈函数 `CopyGMToGMPerToken()`,更适合先做 microtest。

### 5.4 保留 A 方案作为稳健备选

A 方案仍有价值:

- 单写者更干净。
- ready 写次数更少。
- 正确性证明更简单。

如果 B 方案 profiling 显示:

```text
readyWrite 过多
remoteFallback 过多
tableReadyHit 较低
标量 bypass 读写占比明显
```

则可回到 A 方案,用预计算 first 减少多生产者竞争。

---

## 6. A/B 对比

| 维度 | A 方案: 预计算 first | B 方案: 机会式 ready-row |
|------|----------------------|--------------------------|
| 生产者选择 | 源端提前指定最小本地专家 | 运行时谁先完成远端拉谁发布 |
| dispatch 前计算 | 高,需要扫描 topk 计算 first | 低,只需清零表 |
| 写表者 | 单写者 | 多写者良性竞争 |
| 是否需要 CAS | 不需要 | 不需要 |
| 收益稳定性 | 理论上更稳定,但受 fixed first 是否及时影响 | 依赖运行时时序,但更贴近实际先到先得 |
| ready 写次数 | 每个 `(sourceRank, sourceTokenIndex)` 最多一次 | 竞争窗口内可能多次 |
| 实现复杂度 | 中,新增源端计算阶段 | 低到中,集中改 dispatch 读路径 |
| 主要风险 | 前置计算抵消收益 | 多生产者导致收益波动 |
| 推荐定位 | 保守备选 | 优先 microtest |

推荐调整为 B 方案的原因:当前最不确定的是"减少 HCCS 远端读"到底能否覆盖表读写和本地拷贝成本。B 方案去掉 A 方案的前置 first 计算,能更直接地评估 dispatch 读去重本身的收益。

---

## 7. C/D 方案简述

### 7.1 C 方案: CAS/原子占坑

思路:

```text
0 -> CAS 成 IN_PROGRESS -> 远端拉 -> ready row
ready row -> 本地拷
IN_PROGRESS -> 不等待, fallback 远端拉
```

它试图在不预计算 first 的情况下保证单生产者。但当前代码没有现成 GM CAS 使用模式,而且读到 `IN_PROGRESS` 仍不能等待,否则破坏 overlap;如果 fallback,收益接近 B 方案但复杂度更高。

结论:暂不推荐首版实现。除非确认 GM CAS 可用且低成本,并且 B 方案的多生产者竞争被 profiling 证明是主要瓶颈。

### 7.2 D 方案: 专家间屏障

思路是让较小专家先完成,后续专家严格复用它的结果。

这需要在 expert 间加入 rank 内或全核同步,会破坏当前 `flagDispatchToGmm1Ptr` wave-grain 解锁和 dispatch/GMM1 overlap。

结论:不推荐。

---

## 8. 实现改动清单(B 方案)

### 8.1 workspace 布局

B 方案的 ready-row 表只在当前接收卡本地使用,用于当前卡内 AIV 核之间共享"某个源 token 是否已经被拉回本地 workspace"。它不需要被其他 rank 远端访问,因此应放在 `WorkspaceInfo`,不放在 `PeermemInfo`。

在 `WorkspaceInfo` 中新增:

```cpp
GM_ADDR readyRowPtr;
```

建议放在 `tripleInfoPtr` 或 flag 区之后。该表只存 int32 状态,不参与普通跨卡 DataCopy,512B 对齐即可:

```cpp
readyRowPtr = base + workspaceSize;
workspaceSize += Ops::Base::CeilAlign(
    static_cast<int64_t>(tilingData->epWorldSize) * tilingData->bs * SIZE_INT_32,
    ALIGN_512);
```

host tiling 中不需要增加 `leastCclBufferSize`,因为 `readyRowPtr` 不占用 HCCL/peermem 通信窗口。需要同步增加 `WorkspaceInfo::workspaceSize`,并确保 host 侧 workspace 估算与 kernel 侧 `WorkspaceInfo` 构造一致。

```cpp
readyRowSize = align512(epWorldSize * bs * sizeof(int32_t));
workspaceSize += readyRowSize;
```

### 8.2 kernel 成员

不需要新增 peermem/window offset。B 方案只通过当前 rank 本地 workspace 指针读写 ready-row 表。

可选新增或复用:

```cpp
LocalTensor<int32_t> readyRowResetTensor_; // 可复用 resetTensor_,不一定新增
```

接收端本地读写直接用 `params_.workspaceInfo.readyRowPtr`。跨 rank 不需要访问别人的 ready-row 表,因为 ready-row 是当前接收卡本地用于去重的索引。

### 8.3 reset 函数

新增 `ResetReadyRowTable()`:

```cpp
__aicore__ inline void ResetReadyRowTable()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t total = static_cast<int32_t>(worldSize_ * m_);
    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    TilingByCore(total, coreLen, coreOffset, INT32_PER_256B);
    if (coreLen <= 0) {
        return;
    }
    Duplicate(resetTensor_, 0, coreLen);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    GlobalTensor<int32_t> readyRowGm;
    readyRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.readyRowPtr));
    DataCopy(readyRowGm[coreOffset], resetTensor_, coreLen);
}
```

注意 `resetTensor_` 当前大小按 flag reset 需求分配,需确认是否足够容纳 `coreLen`。若不足,需要单独分配 reset buffer 或分段清零。

### 8.4 `CopyGMToGMPerToken()` 改造

签名不需要新增 `localExpertId`,因为 B 方案不依赖当前专家号:

```cpp
CopyGMToGMPerToken(rowDstOffsetInCore, remoteRankIdx, copyStartIdx, rowsNumInCore)
```

Phase 1 新增 ready 读取和 source kind 记录。

Phase 2 的 prime/steady 中,把 MTE2 发射封装为小逻辑:

```cpp
IssueCopyIn(slot, tokenLogicalIdx):
    if SRC_LOCAL:
        从 dispatchRevData/dispatchRevScale 本地读到 copyTmp
    else:
        从 remoteRankGlobalTensor 远端读到 copyTmp
```

Phase 2 drain 后,对所有 `SRC_REMOTE_AND_PUBLISH` 写 ready row。

Phase 3 triple 搬出保持不变。`SRC_KIND/SRC_ROW` 不需要写到 GM triple 中给 GMM2/Combine 使用,但写出去也无害,因为 triple 槽 3/4 当前未使用。

---

## 9. 正确性论证(B 方案)

1. **本地源行一定语义等价**:ready row 指向的是同一个 `(sourceRank, sourceTokenIndex)` 的 token+scale 拷贝。不同 expert 需要的输入 token 数据相同,所以从任意 ready row 复制都是正确的。
2. **不会读半成品**:发布 ready 前必须等待本核 MTE3 写出完成,并执行 `MTE3_S`。消费者通过 bypass 读到 ready 后,对应 row 数据已落 HBM。
3. **并发写不破坏正确性**:多个生产者写不同 row 时,last-writer-wins。所有 row 都是同一 token 数据,后续读任意 row 都正确。
4. **not-ready 不等待**:读到 0 时 fallback 远端拉,等价当前行为。
5. **跨轮无残留**:每轮 dispatch 前 readyRow 全表清零并同步,不会读到上一轮 row。
6. **GMM1 布局不变**:每个命中 expert 仍写自己的 `dispatchRevData/dispatchRevScale` row,`expertRevTokenNums`、`cumsumInfo`、`flagDispatchToGmm1Ptr` 语义不变。

---

## 10. 性能预期与风险

收益来源:

```text
ready hit 次数 * 一次远端 token+scale MTE2 读成本
  - ready 表 bypass 读成本
  - 本地 dispatchRevData/Scale MTE2 读成本
  - ready 写成本
  - reset 成本
```

适合场景:

- `topK / worldSize` 较大。
- 同一源 token 经常命中同一接收卡的多个本地专家。
- 远端 HCCS 读是 dispatch 瓶颈。

可能不划算的场景:

- `topK <= worldSize` 或路由分布很散,重复少。
- 多数重复 token 同时被多个核处理,ready 命中率低,大量 fallback。
- 标量 bypass 读写在 profiling 中占比明显。

建议保留编译期或 tiling key 开关,至少 microtest 阶段支持:

```text
DEDUP_MODE=0 baseline
DEDUP_MODE=1 ready-row opportunistic
```

---

## 11. 验证计划

### 11.1 功能正确性

1. 构造高重复 topK:同一 `(sourceRank, sourceTokenIndex)` 命中当前接收卡的多个本地专家。
2. 对比 baseline 与 B 方案的:
   - `dispatchRevData`
   - `dispatchRevScale`
   - `tripleInfo`
   - 最终 `yOut`
3. 覆盖边界:
   - `copyNum=0`
   - `maxOutputSize_` 截断
   - `worldSize=1`
   - `bs` 非 256 对齐
   - fp8/fp4 activation 路径

### 11.2 并发竞争

构造同一 token 在多个专家里接近同时处理,观察:

```text
多个生产者写 ready row -> 结果仍正确
ready row last-writer-wins -> 后续本地拷贝仍正确
```

### 11.3 时序安全

保留 `MTE3_S` 的正式版本应稳定。可做 debug 对照:

```text
去掉 MTE3_S 或提前写 ready -> 应能在强竞争场景暴露潜在错误
```

这个对照只用于证明同步必要性,不进入正式实现。

### 11.4 性能统计

建议临时增加统计计数:

```text
readyHit          读表 ready,走本地拷贝的 token 数
remoteFallback    读表 empty,走远端拉的 token 数
readyWrite        发布 ready row 的次数
```

关键判断:

```text
readyHit / (readyHit + remoteFallback) 越高,收益越好。
readyWrite 远大于 unique(sourceRank, sourceTokenIndex) 时,说明竞争窗口较大,可考虑回到 A 方案或 CAS 方案。
```

---

## 12. 推荐结论

首版建议采用 B 方案进行 microtest,原因是:

1. 不引入 `SendFirstCal()` 前置计算,能更直接评估 dispatch 读去重收益。
2. 不依赖 GM CAS,并发竞争是良性的。
3. 不改变 GMM1/Combine 看到的数据布局。
4. 改动集中在 `CopyGMToGMPerToken()` 和 workspace 表区,实现闭环较短。

若 B 方案验证后收益不稳定,再考虑 A 方案作为保守实现:用预计算 first 换取单写者和更少 ready 写,但需要重新评估其前置计算成本是否可接受。
