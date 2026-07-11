# MegaMoe 大 bs 前向改造设计

## 1. 文档定位

本文是面向最新主线 `mega_moe.h` 的前向改造设计，不以旧分支代码为最终实现口径。后续开发应以最新主线中的 MegaMoe kernel 为改造对象，按本文描述补齐大 bs 场景下的 UB 分块、host 容量校验和风险验证。

参考文件：

- 待改造主文件：`mc2/mega_moe/op_kernel/arch35/mega_moe.h`
- 常量与 peer window 布局：`mc2/mega_moe/op_kernel/arch35/mega_moe_base.h`
- workspace 布局：`mc2/mega_moe/op_kernel/arch35/mega_moe_workspace_info.h`
- host 侧 tiling / CCL buffer 校验：`mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp`
- 参考实现来源：`mc2/mega_moe/op_kernel/arch35/mega_moe_prefill.h`

本文里的 `prefill.h` 只作为已经验证过一轮思路的参考，不代表需要照搬所有命名和实现细节。若主线代码已发生变化，应以本文的不变量和阶段语义为准重新落地。

## 2. 核心问题

MegaMoe dispatch 会把每个 token 的 `topK` 个 expert 选择展开为一条路由项序列：

```text
routeItemNum = sendTotalNum_ = bs * topK
routeItemIndex = tokenIndex * topK + topkSlot
```

原 `mega_moe.h` 中多个 UB tensor 默认可以一次容纳全量 `routeItemNum`，包括 topk、mask、index、valid index 和部分动态 triple。大 bs 时这些 tensor 的 UB 峰值会超过单个 AIV 的 UB 容量。

需要解决的不是 GM/HBM 里数据放不下，而是每个阶段的 UB live set 太大。改造目标是：

1. 不改变 dispatch 的数学语义和 row-to-token 映射。
2. 不改变 peer window 中 mask 槽位的外部布局。
3. 将每个阶段随 `routeItemNum` 或 `bs * topK` 增长的 UB tensor 改成有界的 route batch / ring buffer triple / reset batch / unpermute token batch。
4. 让后续开发者可以按函数定位实现点。

## 3. 基础语义

### 3.1 发侧和收侧是什么

在 dispatch 阶段，每张 rank 同时扮演两个角色：

- 源 rank，也就是发侧：根据本 rank 的 `topkIds` 判断本 rank 的哪些 token 需要发给哪些 expert，并把 mask 写到目标 rank 的 peer window。
- 目标 rank，也就是收侧：读取其他源 rank 写到本 rank peer window 的 mask/count，确定本 rank 本地 expert 应接收哪些 token，并把这些 token 搬入本 rank 的 `dispatchRevData`。

因此“发侧”和“收侧”不是两类固定设备，而是同一张卡在 dispatch 流水中的两个视角。对某个 token 来说，拥有原始 token 的 rank 是发侧；对某个 expert 来说，expert 所在 rank 是收侧。每张 rank 会先执行发侧逻辑写别人的 window，也会执行收侧逻辑读取别人写给自己的 window。

在代码路径中可对应为：

```text
发侧 / 源 rank：SendMaskCal()
    读取本 rank 的 topkIds
    生成每个目标 expert 的 mask
    写到目标 rank 的 maskRecv 槽

收侧 / 目标 rank：SendCntCal() + TripleInfoCalAndDispatch()
    读取本 rank maskRecv 槽中的 count 和 mask
    建 cumsum，确定 dispatchRevData 行区间
    从源 rank 的 quantTokenScalePtr 读取 token+scale
    写本 rank 的 dispatchRevData / dispatchRevScale
```

后文提到的发侧 route batch，就是 `SendMaskCal()` 生成 mask 时沿本 rank `topkIds` 切分的 route batch；收侧 route batch，就是 `TripleInfoCalAndDispatch()` 读取 mask 并恢复命中 route item 时沿同一条 route item bit-array 切分的 route batch。两者可以使用不同 batch 长度，因为它们都通过全局偏移 `batchStart / 8` 对齐同一条完整 mask。

### 3.2 mask 是什么

对某个全局 expert `g`，mask 是一条 bit 图：

```text
mask_g[i] = (topkIds[i] == g)
```

其中 `i` 是展开后的 `routeItemIndex`。如果 `mask_g[i] = 1`，说明第 `i` 个路由项要发给 expert `g`。

原实现会一次把全量 `topkIds[0 : routeItemNum)` 放入 UB，然后对每个 expert 做一次全量 `CompareScalar`，生成完整 mask。

大 bs 改造后不改变 mask 定义，只改变生产方式。本文把沿 `routeItemIndex` 维度切出的连续批次称为 route batch，避免使用含义过泛的 `chunk`：

```text
route batch c covers [batchStart, batchStart + batchLen)
mask_g_batch[j] = (topkIds[batchStart + j] == g)
```

每个 route batch 的 mask 写到 peer window 中完整 mask 槽的对应偏移 `batchStart / 8`。所有 route batch 写完后，peer window 中拼出的完整 mask 与原实现一次生成的完整 mask 等价。

### 3.3 count 和 cumsum 是什么

mask 回答“哪些路由项命中 expert”，count 回答“有多少个路由项命中 expert”。目标 rank 需要 count 来决定每个源 rank 的 token 应该写到 `dispatchRevData` 的哪些连续行。

例如某个目标 rank 上的 `localExpert` 收到 4 个源 rank 的 count：

```text
srcRank:   0   1   2   3
count:     3   0   5   2
cumsum:    3   3   8   10
```

则每个源 rank 的目标行区间为：

```text
srcRank 0 -> [0, 3)
srcRank 1 -> [3, 3)
srcRank 2 -> [3, 8)
srcRank 3 -> [8, 10)
```

`blockNumPerRank_` 表示对于某一个 `srcRank`，由多少个 AIV 共同处理它发给当前 `localExpert` 的 token。先用 cumsum 得到某个 `srcRank` 的整体行区间，再在这个 `srcRank` 内按 `blockNumPerRank_` 把命中列表切给多个 AIV。

例如 `srcRank 2` 有 5 行，`blockNumPerRank_ = 2`，则可切为：

```text
AIV lane 0 -> 负责 srcRank 2 命中列表中的 [0, 3)，写 dispatchRevData[3, 6)
AIV lane 1 -> 负责 srcRank 2 命中列表中的 [3, 5)，写 dispatchRevData[6, 8)
```

这里 `[0, 3)` / `[3, 5)` 是该 `srcRank` 对当前 expert 的命中列表序号，不是全局 route item 下标。

### 3.4 GatherMask 分块后如何保持顺序

`GatherMask(index, mask)` 可以理解为：输出 `mask` 为 1 的那些 `index`，并保持升序。

原实现：

```text
index = [0, 1, 2, ..., routeItemNum - 1]
validTopkIndex = GatherMask(index, fullMask)
```

分批实现：

```text
for route batch in routeItemNum:
    indexBatch = [batchStart, batchStart + 1, ...]
    validTopkIndexBatch = GatherMask(indexBatch, maskBatch)
```

由于 route batch 按 `routeItemIndex` 升序扫描，且 `GatherMask` 在 batch 内升序输出，把所有 batch 的输出按扫描顺序拼起来，与全量 `GatherMask` 的输出顺序一致。

实现上不需要真的把 `validTopkIndexBatch` 拼成完整列表。每个 route batch 得到临时命中列表后立即派发，下一轮覆盖该 UB。为了知道当前 batch 对应完整命中列表的哪一段，需要维护：

```text
matchedSoFar = 当前 batch 之前已经命中的数量
matchedInBatch = 当前 batch 命中的数量
当前 batch 对应完整命中列表区间 [matchedSoFar, matchedSoFar + matchedInBatch)
```

如果该区间和本 AIV 负责的 `[coreLo, coreHi)` 有交集，就派发交集部分；否则跳过。这样不用保存完整命中列表，也能得到与原实现相同的 row-to-token 映射。

### 3.5 triple 是什么，ring buffer 怎么做

每派发一行，需要记录一组三元信息：

```text
(srcRank, tokenIndex, topkSlot)
```

这组信息称为 triple，后续 combine 阶段用它找回原 token 位置。

不再使用 triple tile 临时分配方案。实际采用 **6-buffer ring buffer + 逐 token 即时写 GM**：

- `tripleTensor_` 在 `DispatchBuffInit()` 中常驻分配 `DISPATCH_BUFFER_NUM × 32B = 192B`（6 条 ring buffer，每槽一条 triple）。
- `CopyGMToGMPerToken()` 的 MTE3 流水线体内，写完 token+scale 后紧接着 `WaitFlag<S_MTE3>` 等待 triple 就绪，然后 `DataCopy(triple)` 把当前槽的 triple 即时写到 GM `tripleInfoPtr`，再 `SetFlag<MTE3_S>` 释放该槽供下一轮复用。
- 不再有单独的 Phase 3 批量 triple 搬出；不再需要 `DISPATCH_TRIPLE_TILE` 常量。

ring buffer 同步模型（每槽 4 个事件）：

| 事件 | Set 者 | Wait 者 | 语义 |
|---|---|---|---|
| `S_MTE3` | S（SetValue 之后） | MTE3（读 triple 之前） | 本槽 triple 已写入 UB |
| `MTE3_S` | MTE3（读完 triple） | S（下一轮 SetValue 之前） | 本槽 triple 可覆盖 |
| `MTE2_MTE3` | MTE2（DataCopy 之后） | MTE3（写 token+scale 之前） | 本槽 token+scale 已搬入 copyTmp |
| `MTE3_MTE2` | MTE3（读完 copyTmp） | MTE2（下一轮 DataCopy 之前） | 本槽 copyTmp 可覆盖 |

详细时序与 drain 处理见 `docs/refactor-notes/inline-triple-write.md`。

## 4. 总体设计

### 4.1 设计原则

1. peer window 外部布局不因分块改变。
2. mask 的完整结果仍落在原 `maskRecv` 槽中，只是分多次写入。
3. count 仍是每个 `(localExpert, srcRank)` 一个总 count，不能变成 per-batch count 依赖。
4. dispatch 行顺序保持不变：srcRank 顺序、expert 顺序、batch 内 routeItemIndex 顺序都必须和原实现等价。
5. UB 分配必须按阶段计算峰值，不能跨阶段简单相加。

### 4.2 当前主线代码的 UB 生命周期

当前 `mega_moe.h` 中，UB 的大阶段复用关系已经存在：

```text
阶段 A: SendAndQuantBuffInit()
    从 UB 地址 0 开始铺 topk / sendMask / reset / quant buffers
    -> SendMaskCal + ResetFlagList + QuantProcessInRank

阶段 B: DispatchBuffInit()
    从 UB 地址 0 开始重新铺 cumsum / gatherMask / index / copyTmp / triple buffers
    -> SendCntCal + TripleInfoCalAndDispatch + GMM1 dispatch

阶段 C: UnpermuteBuffInit()
    从 UB 地址 0 开始重新铺 dataRes / dataResFp32 / topKWeights / scale buffers
    -> Unpermute
```

因此本次大 bs 改造的 UB 预算也必须按阶段分别计算：阶段 A 的发侧 mask/quant/reset 峰值、阶段 B 的收侧 count/mask/index/triple 峰值、阶段 C 的 unpermute 峰值。不要把 `SendAndQuantBuffInit()`、`DispatchBuffInit()`、`UnpermuteBuffInit()` 中的所有 tensor 累加成一个总 UB。

同时，阶段复用有明确边界：

- A -> B：发侧输出必须已经写到 peer window 或 workspace，例如 mask/count 写到 `maskRecv`，量化 token+scale 写到 `quantTokenScalePtr`；当前 `Process()` 在进入 `DispatchBuffInit()` 前已有 `PipeBarrier<PIPE_ALL>()`、`SyncAll<true>()`、`CrossRankSyncInWorldSize()`。
- B -> C：不能在 `TripleInfoCalAndDispatch()` 刚结束就覆盖阶段 B UB。阶段 B 产生的 count/cumsum 和 expert token 数还会影响 GMM1/GMM2 的 shape/offset、`ExpertTokenNumCopyOut()` 和 combine。只有 GMM1/GMM2/combine 完成，并经过 `PipeBarrier<PIPE_ALL>()`、`SyncAll<true>()` 后，才能进入 `UnpermuteBuffInit()` 覆盖 UB。

实现时的原则是：每次进入新的 `*BuffInit()` 前，上一阶段后续仍需要的信息必须已经落到 GM、workspace 或 peer window；不能把跨阶段语义依赖留在 UB 里。

### 4.3 阶段 A：发侧 / 源 rank 分批生成 mask

原 `mega_moe.h` 受影响 tensor：

- `topkIdsTensor_`：全量 `routeItemNum` topk。
- `sendMaskTensor_[2]`：完整 `maskSlotSize_` mask + count 槽。
- `sendGatherOutTensor_`：全量 GatherMask scratch。
- `resetTensor_`：按本核 reset 份额一次性开 UB。

改造方案：

1. 将 `topkIdsTensor_` 改为 route batch 级 topk buffer，建议命名为 `sendRouteBatchTensor_`，注释说明它保存的是 route item batch，不是原始 token batch。
2. 将 `sendMaskTensor_[2]` 从完整槽大小改为 `sendRouteBatchLen / 8 + 32B`。前半段保存当前 route batch 的 bit mask，尾部 32B 用于末块折叠 count。
3. 将 `sendGatherOutTensor_` 改为 route batch 级 scratch，大小 `sendRouteBatchLen * sizeof(int32_t)`。
4. 新增每核 owned expert 的 count 累加器，例如 `sendCntAccumulateTensor_`。每个 route batch 的 `GatherMask` 只产生当前 batch 命中数，总 count 由该累加器跨 batch 累加。
5. `SendMaskCal()` 循环改为外层 route batch、内层 owned expert：

```text
for route batch in routeItemNum:
    load topk batch
    if last batch: padding 填 -1
    for owned expert:
        CompareScalar -> 当前 batch mask
        GatherMask -> 当前 batch count
        countAcc[expert] += batch count
        if last batch: 把 countAcc 写入 mask buffer 尾部
        DataCopyPad -> peer window slot + batchStart / 8
```

改善效果：

- 原 `topkIdsTensor_` 从 `O(routeItemNum)` 降为 `O(sendRouteBatchLen)`。
- 原 `sendGatherOutTensor_` 从 `O(routeItemNum)` 降为 `O(sendRouteBatchLen)`。
- 原 `sendMaskTensor_` 从完整 `maskSlotSize_` 降为 `O(sendRouteBatchLen / 8)`。

实现定位：

- 改 `SendAndQuantBuffInit()` 的发侧 UB 布局和 route batch 大小反推。
- 改 `SendMaskCal()` 的循环结构和 peer window 写入偏移。
- 保持 `QuantProcessInRank()` 计算语义不变，但它的 UB 占用应计入发侧 fixedBytes。

### 4.4 阶段 B：收侧 / 目标 rank 分批读取 mask 并派发 token

原 `mega_moe.h` 受影响 tensor：

- `gatherMaskTensor_` / `gatherMaskInt32Tensor_`：完整读取 `worldSize_ * maskSlotSize_`。
- `topkIndexTensor_`：全量 route item index。
- `validTopkIndexTensor_`：全量命中 route item index。
- `tripleTensor_`：本 AIV 一次派发区间的全部 triple。

改造方案：

1. `SendCntCal()` 不再读取完整 mask 槽，只通过 stride 读取每个 srcRank 的 4B count 字段，生成 cumsum。
2. 新增 count-only UB buffer，例如 `sendCntTensor_`。注意 GM 到 UB 的 4B 小块 `DataCopyPad` 可能按 32B burst 落位，读取下标应按实际落位设计，例如参考实现中是 `rank * 8`。
3. 在 `TripleInfoCalAndDispatch()` 中，按 `recvRouteBatchLen` 读取当前 `(localExpert, srcRank)` 的 mask 切片。
4. `topkIndexTensor_` 改为 route batch 级，每个 batch 调 `CreateVecIndex(batchStart, recvRouteBatchLen)`，这样输出 index 仍是全局 route item index。
5. `validTopkIndexTensor_` 改为 route batch 级临时列表，不拼接完整列表。使用 `matchedSoFar` 将当前 batch 的命中序号映射到完整命中列表序号。
6. 只有当前 batch 命中区间与本 AIV 的 `[coreLo, coreHi)` 有交集时，才调用 `CopyGMToGMPerToken()` 派发交集部分。
7. `tripleTensor_` 在 `DispatchBuffInit()` 中常驻分配为 6-buffer ring buffer（`DISPATCH_BUFFER_NUM × 32B = 192B`），逐 token 即时写 GM，无需 per-call 动态分配。

伪代码：

```text
for localExpert:
    SendCntCal(localExpert) -> cumsum
    for srcRank lane assigned to this AIV:
        [coreLo, coreHi), rowDstOffset = split count by blockNumPerRank_
        matchedSoFar = 0
        for recv route batch while matchedSoFar < coreHi:
            read mask batch
            CreateVecIndex(batchStart, recvRouteBatchLen)
            GatherMask -> validTopkIndexBatch, matchedInBatch
            [lo, hi) = intersect([matchedSoFar, matchedSoFar + matchedInBatch), [coreLo, coreHi))
            if hi > lo:
                subStart = lo - matchedSoFar
                subDst = rowStartIdxInDst + lo
                CopyGMToGMPerToken(subDst, srcRank, subStart, hi - lo)
            matchedSoFar += matchedInBatch
```

改善效果：

- 原 `gatherMaskTensor_` 不再完整常驻 UB。
- 原 `topkIndexTensor_` / `validTopkIndexTensor_` 从 `O(routeItemNum)` 降为 `O(recvRouteBatchLen)`。
- 原 `tripleTensor_` 从 `O(rowsNumInCore)` 降为固定 `DISPATCH_BUFFER_NUM × 32B = 192B` ring buffer。

实现定位：

- 改 `DispatchBuffInit()` 的收侧 UB 布局：新增 triple ring buffer 常驻分配。
- 改 `SendCntCal()` 为 count-only stride 读取。
- 改 `TripleInfoCalAndDispatch()` 为 route batch 循环 + `matchedSoFar`。
- 改 `CopyGMToGMPerToken()` 为 ring buffer 同步模型：Phase 1+prime 合并、steady body 内 triple 即时写 GM、S_MTE3/MTE3_S 握手。

### 4.5 阶段 C：Unpermute weights 按 token batch 流式读取

原 `mega_moe.h` 受影响 tensor：

- `topKWeightsTensor_`：全量 `bs * topK` float weights。
- `tempLocal`：当 weights 是 bf16 时，全量 `bs * topK` bf16 临时区。

改造方案：

1. `Unpermute()` 已按 token 分核。先用 `TilingByCore(m_)` 得到本核 token 段 `[wCoreOffset, wCoreOffset + wCoreLen)`。
2. 不再把本核全段 weights 一次搬入 UB，而是在本核段内部再按 token batch 流式读取。设本轮 token batch 为 `[tokenBatchStart, tokenBatchStart + tokenBatchLen)`。
3. 只从 `params_.probsGmAddr + tokenBatchStart * topK` 搬入当前 token batch 的 weights。
4. `Unpermute()` 处理当前 token batch 内 token 时，读取权重使用相对 token batch 的下标：

```text
localWeightIndex = (tokenIdx - tokenBatchStart) * topK + expId
```

5. 如果 `TopkWeightsType = bfloat16_t`，bf16 `tempLocal` 也只按 `tokenBatchLen * topK` 分配和 cast。
6. 当前 token batch 的 weights 搬入并完成 cast 后，循环处理该 batch 内 token 的 `DataCopy -> Cast/DeQuant -> Muls/Add -> DataCopy`。处理完后复用同一片 weights UB 处理下一个 token batch。

改善效果：

- `topKWeightsTensor_` 从 `O(bs * topK)` 降为 `O(unpermuteTokenBatchLen * topK)`。
- `tempLocal` 从 `O(bs * topK)` 降为 `O(unpermuteTokenBatchLen * topK)`。
- 这是比“只读本核 token 段”更完整的大 bs 方案：即使单核 token 段仍然很大，也不会让 weights 常驻 UB 随 bs 增长。

实现定位：

- 改 `UnpermuteBuffInit()`：只分配固定 token batch 大小的 `topKWeightsTensor_` / `tempLocal`，并计算 `unpermuteTokenBatchLen_`。
- 改 `Unpermute()`：外层遍历本核 token 段内的 token batch；每个 batch 先搬入 weights，再处理该 batch 内 token；`topKWeightsTensor_.GetValue(...)` 使用 batch-local 下标。

### 4.6 reset flag 分批清零

原 `mega_moe.h` 的 `resetTensor_` 按本核 reset 份额一次性分配。大 `maxOutputSize` 时 `flagDispatchToGmm1Ptr` 对应 slot 数增长，会放大该 UB。

改造方案：

1. 将 `resetTensor_` 固定上限为 `DISPATCH_RESET_BATCH` 个 int32，例如 2048 个 int32，即 8KB。
2. `ResetFlagList()` 中本核负责的 `[coreOffset, coreOffset + coreLen)` 按 batch 循环写同一片零 UB。
3. combine quant 路径下的 `ResetRowGroupCompleteFlags()` 也必须按相同原则处理，不能继续假设 `resetTensor_` 足够容纳 `coreLen`。

实现定位：

- 改 `SendAndQuantBuffInit()` 中 reset buffer 分配。
- 改 `ResetFlagList()` 为 batch loop。
- 复核并改造 `ResetRowGroupCompleteFlags()`。

## 5. UB 规划

### 5.1 route batch 命名约定

本文统一使用 route batch 表示沿 `routeItemIndex` 维度切出的连续批次。主线实现建议使用以下命名，避免 `chunk` 这个泛化词和 matmul tile、token batch 混淆：

```text
sendRouteBatchLen_ / sendRouteBatchNum_
recvRouteBatchLen_ / recvRouteBatchNum_
```

prefill 路径当前使用的编译期固定常量：

```text
DISPATCH_RECV_BATCH_LEN = 12288   // 收侧 route batch，最坏 ws=1024 时 UB 占用 75%
UNPERMUTE_BATCH_LEN    = 1024    // unpermute weights token batch，k∈[1,8192] 安全
```

### 5.2 UB 预算来源总表

UB 预算必须按当前代码生命周期分阶段计算。下表列出每个预算来自哪些函数和 tensor 代码，后续实现时应逐项核对，避免新增 tensor 后忘记更新 batch 长度反推公式。

| UB 预算阶段 | 生命周期 / 代码位置 | 必须计入的 UB 来源 | 是否随大 bs 增长 | 预算用途 |
| --- | --- | --- | --- | --- |
| 阶段 A：发侧 / 源 rank mask + quant + reset | `SendAndQuantBuffInit()`，服务 `SendMaskCal()`、`ResetFlagList()`、`QuantProcessInRank()` | `topkIdsTensor_` 替换后的 send route batch topk；`resetTensor_`；`mxTempTensor_`；`xOutTensor1_/2_`；`xInTensor1_/2_`；`sendMaskTensor_[2]`；`sendGatherOutTensor_`；新增 `sendCntAccumulateTensor_` | topk、sendMask、sendGatherOut 随 `sendRouteBatchLen`；reset 随 `maxOutputSize` 但需封顶；量化 buffer 随 `h` | 固定 `sendRouteBatchLen_ = DISPATCH_SEND_BATCH_LEN` |
| 阶段 B：收侧 / 目标 rank count + dispatch | `DispatchBuffInit()`，服务 `SendCntCal()`、`TripleInfoCalAndDispatch()`、`CopyGMToGMPerToken()`、`ExpertTokenNumCopyOut()` | `expertTokenCntTensor_`；`cumsumInfoTensor_`；替代 `gatherMaskTensor_` 的 count-only `sendCntTensor_`；新增 `maskBatchTensor_`；`validTopkIndexTensor_`；`topkIndexTensor_`；`copyTmpTensors_[6]`；`expertTokenNumsOutTensor_`；`tripleTensor_` ring buffer 常驻（`DISPATCH_BUFFER_NUM × 32B`） | mask/index/validIndex 随 `recvRouteBatchLen`；triple ring buffer 固定 192B；copyTmp 固定 `6 × tokenScaleSize`；cumsum 随 `worldSize * expertPerRank` | 反推或固定 `recvRouteBatchLen_` |
| 阶段 B2：combine quant 行组处理 | `ProcessCombine()`，仅 `CombineQuantMode != COMBINE_NO_QUANT` | 局部 `tripleTensor`：`myRowCount * TRIPLE_SIZE * sizeof(int32)`；`CombineRowGroup()` 内 `combineUbTensor`；`floatTemp` | 主要随 `h` 和 row group tile，不随 `bs` 线性增长；row group 个数随 bs 影响 GM counter，不直接放大单次 UB | 不属于本次大 bs route-batch 必改项；仅保留为 combine quant 分支的独立容量审计 |
| 阶段 C：unpermute | `UnpermuteBuffInit()` / `Unpermute()` | `dataResTensor_`；`dataResFp32Tensor_`；`topKWeightsTensor_`；bf16 weights `tempLocal`；combine quant 下 `bf16ScaleTensor_` / `fp32ScaleTensor_` | weights/tempLocal 随 `unpermuteTokenBatchLen`；dataRes/scale 随 `h` | 反推 `unpermuteTokenBatchLen_` |

说明：阶段 A、阶段 B、阶段 C 的 `*BuffInit()` 都从 UB 地址 0 重新铺 tensor，因此不跨这些阶段相加。阶段 B2 的 `ProcessCombine()` 不是本次大 bs route-batch 改造对象；它会在 `TPosition::VECIN` 上动态构造 `LocalTensor`，但单次 UB 主要由 `h` 和 row group tile 决定，不随 bs 线性放大。本文只把它列为 combine quant 分支的独立容量审计项，避免误算到 A/B/C 的大 bs 分块预算里。

### 5.3 发侧 UB 预算

发侧 fixedBytes 至少包含：

- reset buffer 上限；
- 量化相关 `mxTempTensor_`、`xInTensor1_/2_`、`xOutTensor1_/2_`；
- 每核 owned expert 的 count 累加器。

随 send route batch 线性增长的项：

```text
topk batch       4 * sendRouteBatchLen
mask scratch     4 * sendRouteBatchLen
mask buffers     DOUBLE_BUFFER * sendRouteBatchLen / 8
```

**prefill 路径当前方案**：固定编译期常量 `DISPATCH_SEND_BATCH_LEN = 12288`（与收侧同值）。

推导过程（arch35, UB=256KB, 有效 ~242KB）：

```text
最坏情况 k=8192: 固定项 ≈ reset(8KB) + mxTemp(2KB) + 2×xOut(16.5KB) + 2×xIn(32KB) + sendCntAcc(~128B) ≈ 59KB
可用UB ≈ 242KB - 59KB = 183KB
发送侧线性项: batchLen × 8.125B/item → maxBatchLen ≈ 183KB / 8.125 ≈ 22,500
选取 12288 ≈ 54% 最大 → 留有充足余量
```

**动态反推**（主线改造可参考）：从 UB 余量反推 `sendRouteBatchLen`，向下取 256 对齐，并 cap 到 `ceil_align(routeItemNum, 256)`。

### 5.4 收侧 UB 预算

收侧 fixedBytes 至少包含：

- expert count 临时区；
- cumsum 表；
- count-only 读取 buffer；
- `copyTmpTensors_[6]`（每槽 `tokenScaleSize`）；
- expertTokenNumsOut 临时区；
- triple ring buffer 常驻 `DISPATCH_BUFFER_NUM × 32B = 192B`。

随 recv route batch 线性增长的项：

```text
mask batch        recvRouteBatchLen / 8
topk index        4 * recvRouteBatchLen
valid topk index  4 * recvRouteBatchLen
```

收侧 route batch 长度有两种策略：

**固定常量**（prefill 路径当前方案）：`DISPATCH_RECV_BATCH_LEN = 12288`。

推导过程（arch35, UB=256KB, 系统预留 ~8KB → 可用 ~248KB, DISPATCH_UB_SAFETY=6KB → 有效 ~242KB）：

```text
固定项 = expertTokenCnt(32B) + cumsumInfo(ws*epr*4B align 32B) + sendCnt(ws*32B)
       + copyTmp[6] * tokenScaleSize + expertTokenNumsOut(epr*4B align 32B)
       + tripleRingBuffer(192B)

最坏情况: ws=1024, epr=2, k=8192, tokenScaleSize = align32(8192B + 256B) = 8480B
固定项 ≈ 32 + 8K + 32K + 6×8480 + 64 + 192 ≈ 91,904B

随 batchLen 线性项 = maskBatch(batchLen/8) + topkIndex(4*batchLen) + validTopkIndex(4*batchLen)
                   ≈ batchLen × 8.125B

可用UB = 242KB - 91,904B ≈ 156,864B
maxBatchLen = 156,864 / 8.125 ≈ 19,306 → 对齐256 → 19200

选取 12288: (91,904 + 12288×8.125) / 256KB ≈ 191,744/262,144 ≈ 73%
留有 ~60KB 安全余量给未来固定项增长。
```

**动态反推**（主线改造可考虑）：从 UB 余量反推 `recvRouteBatchLen`，向下取 256 对齐，并 cap 到 `ceil_align(routeItemNum, 256)`。

发侧和收侧的 route batch 可以不同。peer window 中的 mask 是完整 bit-array，只要读写都使用全局偏移 `batchStart / 8`，二者不需要取同一个 batch 长度。

### 5.5 combine quant 辅助 UB 审计项

当 `CombineQuantMode != COMBINE_NO_QUANT` 时，可以单独检查 `ProcessCombine()` 的 UB。它不是本次大 bs 场景需要分批改造的对象，因为其单次 UB 峰值不随 `bs` 线性增长；当前代码中这部分不通过 `UnpermuteBuffInit()` 或 `DispatchBuffInit()` 统一分配，而是在函数内动态构造：

```text
ProcessCombine():
    tripleTensor = myRowCount * TRIPLE_SIZE * sizeof(int32)
    offset += triple bytes

CombineRowGroup():
    combineUbTensor = ubTensorSize * sizeof(T)
    floatTemp = (nScale + nScale / 2) * sizeof(float)
```

其中：

```text
myRowCount <= L1_TILE_M_256
TRIPLE_SIZE = 8 int32 = 32B
combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t)
singleTokenBytes = align32(h) * sizeof(bfloat16_t) + quantRowSizeBytes
floatTempSize = nScale + nScale / 2
```

这一路径的单次 UB 峰值主要随 `h` 增长，不随 bs 线性增长；它不会影响阶段 A/B/C 的 route-batch 长度反推。因此：

- 本次大 bs route-batch 改造不需要修改 `ProcessCombine()` / `CombineRowGroup()` 的 UB 切分；
- 若 `h` 很大，仍可单独核对 `tripleTensor + combineUbTensor + floatTemp` 是否超过 UB；
- row group counter 的 GM 空间和 reset 数量会随 bs 增长，因此 `ResetRowGroupCompleteFlags()` 的清零动作仍应纳入 reset batch 设计。

### 5.6 triple ring buffer 与 reset batch 大小

triple 不再使用 tile 临时分配方案，改为 ring buffer 常驻分配：

```text
DISPATCH_BUFFER_NUM = 6
tripleReserveBytes = DISPATCH_BUFFER_NUM * 32B = 192B
```

该 192B 计入收侧 fixedBytes。`CopyGMToGMPerToken()` 内部通过 6 槽 S_MTE3/MTE3_S ring handshake 逐 token 即时写 triple 到 GM，无需 per-call 动态分配，也不再有单独的 Phase 3 批量搬出。

reset batch 同样建议固定上限：

```text
DISPATCH_RESET_BATCH = 2048 int32
resetBatchBytes = 8KB
```

`ResetFlagList()` 和 `ResetRowGroupCompleteFlags()` 都应使用同一片零 UB 分批写 GM。实际每核 reset 长度 `coreLen` 小于 `DISPATCH_RESET_BATCH` 时退化为单批。

### 5.7 Unpermute token batch 大小

Unpermute 阶段应按 token batch 流式读取 topK weights。

**prefill 路径当前方案**：固定编译期常量 `UNPERMUTE_BATCH_LEN = 1024`。

推导过程（arch35, UB=256KB）：

```text
固定项 = dataResBuf(3 × k × sizeof(bf16) align 32B) + dataResFp32Buf(同尺寸 fp32 版)
       + combine-quant 路径: bf16ScaleBuf + fp32ScaleBuf

k=8192 时: dataRes = 3×8192×2 = 48KB, dataResFp32 = 96KB → 固定 ≈ 144KB
k=512 时:  固定 ≈ 9KB

随 batchLen 线性项 = topKWeightsBuf(batchLen × topK × sizeof(float))
                  + tempLocal(batchLen × topK × sizeof(bf16), 仅 bf16 weights 需要)

最坏情况 topK=8, bf16 weights:
  bytesPerToken = 8×4 + 8×2 = 48B/token

可用UB(k=8192, 最紧): 242KB - 144KB = 98KB
maxBatchLen = 98KB / 48B ≈ 2080 → 1024 安全
```

选取 1024：k∈[1,8192]、topK∈[2,8] 范围内 \( 1024 \times topK \times 6B \)（最坏 48KB）在固定项从 9KB~144KB 的最紧情况（k=8192）下仍安全。

**动态反推方案**（主线改造可参考）：阶段 C fixedBytes 至少包括：

- `dataResTensor_`：`UNPERMUTE_LIST_NUM * h * sizeof(bfloat16_t)` 对齐；
- `dataResFp32Tensor_`：`dataResTensor_` 的 fp32 版本；
- combine quant 路径的 `bf16ScaleTensor_` / `fp32ScaleTensor_`；
- 其它固定控制或对齐余量。

随 `unpermuteTokenBatchLen` 线性增长的项：

```text
topKWeightsTensor_    unpermuteTokenBatchLen * topK * sizeof(float)
tempLocal             unpermuteTokenBatchLen * topK * sizeof(bfloat16_t)  // 仅 TopkWeightsType=bf16 时需要
```

因此可按 dtype 分支反推：

```text
availC = ubSize - DISPATCH_UB_SAFETY - fixedBytesC

if TopkWeightsType == float:
    bytesPerTokenWeights = topK * sizeof(float)
else if TopkWeightsType == bfloat16_t:
    bytesPerTokenWeights = topK * (sizeof(float) + sizeof(bfloat16_t))

unpermuteTokenBatchLen = floor(availC / bytesPerTokenWeights)
unpermuteTokenBatchLen = clamp_to_positive_and_align(unpermuteTokenBatchLen, tokenAlign)
unpermuteTokenBatchLen = min(unpermuteTokenBatchLen, coreTokenLen)
```

`tokenAlign` 可先取 1，保证不改变语义；若后续为搬运效率希望按 8/16/32 token 对齐，必须处理尾 batch。若 `availC <= 0` 或反推结果小于 1，应在 host 或 kernel 初始化阶段显式报错，不应兜底到会越 UB 的配置。

## 6. peer window 与 host 校验

mask 槽位不应按 route batch 改大。保持公式：

```text
compareCount  = ceil_align(routeItemNum * sizeof(int32), 256) / sizeof(int32)
maskAlignSize = ceil_align(compareCount / 8, 32)
maskSlotSize  = maskAlignSize + 32
```

host 侧 `leastCclBufferSize` 必须与 `PeermemInfo` 的真实布局一致：

```text
leastCclBufferSize = PEERMEM_DATA_OFFSET
                   + ceil_align(expertPerRank * epWorldSize * maskSlotSize, 512)
                   + ceil_align(bs * quantTokenScaleBytesPerToken, 512)
                   + ceil_align(bs * h * topK * yDtypeSize, 512)
```

不要为了 route batch 添加 per-batch count 或扩大 mask 槽，除非同时重做 host/window 分配协议。当前推荐方案保持一个完整 mask 槽，只改变 kernel 内部填充方式。

## 7. 正确性不变量

1. mask 的最终 peer window 内容必须等价于原实现完整 mask。
2. 每个 `(localExpert, srcRank)` 仍只有一个总 count，且 count 等于完整 mask 中 bit=1 的数量。
3. route batch 扫描顺序必须按 `routeItemIndex` 升序。
4. batch 内 `GatherMask` 输出必须作为全局 route item index 使用，因此 `CreateVecIndex` 需要从 `batchStart` 开始。
5. `matchedSoFar` 必须只累加当前 `(localExpert, srcRank)` 已扫描 route batch 的命中数。
6. `[coreLo, coreHi)` 是当前 `srcRank` 命中列表按 `blockNumPerRank_` 切给本 AIV 的区间，不是全局 route item 区间。
7. triple ring buffer（6 槽 `S_MTE3`/`MTE3_S` handshake）逐 token 即时写 GM，不改变 row-to-token 映射。
8. 末块 padding 不能误命中任何 expert，建议填 `-1`。
9. 所有涉及 `bs * topK * h` 或 GM/window offset 的计算使用 64 位。
10. 每次调整 UB 布局后，必须同步更新 route batch 长度反推公式。

## 8. 实现任务清单

### 8.1 kernel 侧

- 在 `mega_moe_base.h` 增加或确认：
  - `DISPATCH_BUFFER_NUM`（triple ring buffer 槽数，建议 6）；
  - `DISPATCH_RESET_BATCH`；
  - `DISPATCH_UB_SAFETY`。
- 在 `MegaMoe` 成员中新增：
    - 发侧 route batch 长度和 batch 数；
    - 收侧 route batch 长度和 batch 数；
  - reset batch 长度；
  - 发侧 count 累加器 tensor；
  - 收侧 count-only tensor；
    - 收侧 mask batch tensor。
- 改 `SendAndQuantBuffInit()`：发侧 UB 布局和 `sendRouteBatchLen` 反推。
- 改 `SendMaskCal()`：外层 route batch、内层 expert，末块 count 折叠写 peer window。
- 改 `DispatchBuffInit()`：收侧 UB 布局，新增 triple ring buffer 常驻分配（`DISPATCH_BUFFER_NUM × 32B`），`recvRouteBatchLen` 可固定常量或动态反推。
- 改 `SendCntCal()`：只读 count，不读完整 mask。
- 改 `TripleInfoCalAndDispatch()`：route batch 扫描、`matchedSoFar`、sub-slice 派发（不再有 inner triple tile 循环）。
- 改 `CopyGMToGMPerToken()`：Phase 1+prime 合并，steady body 内 triple 即时写 GM，S_MTE3/MTE3_S ring handshake，Phase 3 删除。
- 改 `UnpermuteBuffInit()` / `Unpermute()`：weights 按本核 token 段内的 token batch 流式读取。
- 改 `ResetFlagList()` / `ResetRowGroupCompleteFlags()`：reset 分批。

### 8.2 host 侧

- 改 `mega_moe_tiling.cpp` 的 `leastCclBufferSize`，确保与 `PeermemInfo` 真实布局一致。
- 所有容量公式使用 int64。
- CCL buffer 放不下时 host 侧直接报错，不让 kernel 静默越界。

## 9. 验证计划

### 9.1 功能验证

- 小 bs 单 batch：与改造前结果逐位一致。
- 跨 batch 且尾块非满：覆盖 tail padding、末块 clamp、count 折叠。
- 多 rank：覆盖跨 rank mask 写入、count 读取、远端 token 搬运。
- `maxOutputSize` clamp：验证 row 区间和 flag atomic 正确。
- `TopkWeightsType = float / bfloat16_t`：验证 unpermute weights token batch 流式读取。
- combine quant 开启：验证 rowGroupComplete reset 分批和 dequant 路径。

### 9.2 内存边界验证

- 打印或静态核对每阶段 UB fixedBytes 和 batchBytes。
- 验证 `sendRouteBatchLen` / `recvRouteBatchLen` 下限仍可容纳所有 fixed tensor。
- 验证 `unpermuteTokenBatchLen >= 1`，并覆盖 float/bf16 weights 两种分支的 bytesPerTokenWeights。
- combine quant 开启时，可单独核对 `ProcessCombine()` 中 `tripleTensor + combineUbTensor + floatTemp` 的 UB 峰值；它不参与大 bs route-batch 长度反推。
- 验证 `maskAlignSize`、`maskSlotSize` 和 host 侧公式一致。
- 验证末块 `batchStart / 8 + sliceBytes <= maskAlignSize`。

### 9.3 性能验证

- 对比中小 bs 单 batch 路径性能，确认没有明显回退。
- 统计 batch 数与耗时关系，确认动态 route batch 充分利用 UB。
- 观察高 lane AIV 的前导 batch 扫描成本，评估是否需要后续 per-batch count 优化。

## 10. 方案评价与后续路线

当前方案是合理的一阶段 kernel 局部解：它以不改变 peer window ABI 和 dispatch 行序为前提，解决主路径 UB OOM，并通过 unpermute token batch 进一步解除 weights UB 与 bs 的强耦合。它仍不应被描述为“任意大 bs 最终方案”，因为 peer window 中 `combineSend` 仍是 `O(bs * topK * h)`，`quantTokenScale` 仍是 `O(bs * h)`。

长期若要支持极大 bs，应考虑：

1. host 层 sub-batch，多轮复用 peer window；
2. unpermute weights 进一步改为 token tile 流式读取；
3. combine 回送路径流式化，降低 `combineSend` window 占用；
4. 在不破坏 window ABI 的前提下评估 per-batch count 优化是否值得做。
