# SparseAttentionScorePrefill 算子设计文档

## 1. 算子目标

`sparse_attention_score_prefill` 面向 Prefill 阶段的 Sparse Attention，采用 **KV-Gather-Q** 方案：以 KV block 为主轴分任务，同一个 KV block 只加载一次，然后遍历所有选择该 block 的 Q token，计算该 block 对这些 Q token 的 partial attention。

目标不是实现逐元素标量参考 kernel，而是实现如下数据流：

1. Phase 1 在 Cube 侧完成 `QK + softmax + PV` 的 partial 计算。
2. Phase 1 以 `1 logical KV block × 1 KV head` 为 task 粒度。
3. Q token 通过 host 提供的反向索引 gather，Q 在 GM 中可能非连续，因此 Q 需要逐行搬运到连续的片上临时区后参与 matmul。
4. Phase 1 将每个 `(qToken, kvHead, slotK)` 的 `partial_O / partial_max / partial_sum` 写入 workspace。
5. Phase 2 在 Vector 侧读取每个 Q 的 topK partial，执行 online-softmax combine，写最终 `attention_out`。
6. Phase 1 与 Phase 2 在同一个 kernel 内执行，中间通过 `SyncAll` 做全局同步。

## 2. 输入、输出与辅助张量

### 2.1 主输入

| Tensor | Shape | Type | 说明 |
| --- | --- | --- | --- |
| `query` | `[total_q_tokens, num_q_heads, D]` | bf16/fp16 | TND 格式 Q。`qToken` 是跨 batch 展平后的 token 下标。 |
| `key` | `[num_physical_blocks, blockSize, kvHeads, D]` | bf16/fp16 | Paged K Cache。 |
| `value` | `[num_physical_blocks, blockSize, kvHeads, D]` | bf16/fp16 | Paged V Cache。 |
| `select_idx` | `[total_q_tokens, topK]` | int32 | 每个 Q token 选择的 logical KV block，下标为跨 batch 展平后的 logical KV block id。Phase 2 可用它跳过无效 slot。 |
| `block_table` | `[batch, maxBlocksPerBatch]` | int32 | 每个 batch 内 logical block 到 physical block 的映射。第二维是寻址 stride，必须来自 shape，不能由 `kv_seqlen` 推导。 |
| `actual_seq_lengths` | `[batch]` | int64 | 每个 batch 的 Q seqlen。 |
| `actual_seq_lengths_kv` | `[batch]` | int64 | 每个 batch 的 KV seqlen。 |

### 2.2 Host 侧反向索引输入

这些张量由 host 从 `select_idx` 构建，并作为 GM 输入传入 kernel，不放入 workspace：

| Tensor | Shape | Type | 说明 |
| --- | --- | --- | --- |
| `kv_to_q_select_idx` | `[numKvBlocks, maxQPerBlock]` | int32 | 第 `kvBlockIdx` 个 logical block 对应的 Q token 列表。 |
| `kv_to_q_select_num` | `[numKvBlocks]` | int32 | 每个 logical KV block 被多少个 Q token 选择。 |
| `kv_to_q_slot_idx` | `[numKvBlocks, maxQPerBlock]` | int32 | 对应 Q token 中该 KV block 位于 `select_idx[qToken, :]` 的第几个 slot，范围 `[0, topK)`。 |
| `kv_block_valid_size` | `[numKvBlocks]` | int32 | 每个 logical KV block 的有效 token 数。非尾块通常为 `blockSize`，尾块为剩余长度。 |

反向索引构建约束：

```text
for qToken in [0, total_q_tokens):
  for slotK in [0, topK):
    kvBlockIdx = select_idx[qToken, slotK]
    if kvBlockIdx is valid:
      append qToken to kv_to_q_select_idx[kvBlockIdx]
      append slotK  to kv_to_q_slot_idx[kvBlockIdx]
```

`kv_to_q_select_idx[kvBlockIdx, :]` 中的 Q token 必须属于该 logical KV block 所在 batch。kernel 会根据 `q_seqlen` 和 `kv_seqlen` 重新计算 batch offset，但不负责跨 batch 合法性修正。

### 2.3 输出

| Tensor | Shape | Type | 说明 |
| --- | --- | --- | --- |
| `attention_out` | `[total_q_tokens, num_q_heads, D]` | bf16/fp16 | TND 格式输出。 |

当前 kernel 接口没有独立 `softmax_lse` 输出；如后续需要 LSE，应新增输出或复用额外 workspace/GM 输出，不应隐式写入当前 workspace。

## 3. Tiling 数据定义

### 3.1 基础字段

| 字段 | 含义 |
| --- | --- |
| `batch` | batch 数，来自 `block_table.shape[0]`。 |
| `numHeads` | Q head 数，来自 `query.shape[1]`。 |
| `kvHeads` | KV head 数，优先来自 attr，否则来自 `key.shape[2]`。 |
| `groupSize` | `numHeads / kvHeads`，GQA 中一个 KV head 对应的 Q head 数。 |
| `embeddingSize` | head size D。当前主目标为 D=128。 |
| `blockSize` | KV block token 数，通常为 128。 |
| `topK` | 每个 Q token 选择的 KV block 数。 |
| `totalQTokens` | 所有 batch 的 Q token 总数，来自 `query.shape[0]`。 |
| `numKvBlocks` | 所有 batch 的有效 logical KV block 总数，`sum(ceil(kv_seqlen[b] / blockSize))`。 |
| `maxBlocksPerBatch` | `block_table.shape[1]`，用于 `block_table` 行 stride。 |
| `maxQPerBlock` | host 扫描 `select_idx` 得到的单个 KV block 最大 Q 数。 |
| `totalTaskNumP1` | Phase 1 task 数，`numKvBlocks * kvHeads`。 |
| `totalTaskNumP2` | Phase 2 task 数，`totalQTokens * kvHeads`。 |

### 3.2 blockDim 策略

同一个 kernel 内包含 Cube Phase 1 和 Vector Phase 2，因此 `blockDim` 不能只按照 Phase 1 计算。推荐：

```text
totalTaskP1 = numKvBlocks * kvHeads
totalTaskP2 = totalQTokens * kvHeads
blockDim = min(max(totalTaskP1, totalTaskP2), max(aicNum, aivNum))
blockDim = max(blockDim, 1)
```

Phase 1 使用：

```text
for taskIdx = blockIdx; taskIdx < totalTaskNumP1; taskIdx += blockDim
```

Phase 2 使用：

```text
for taskIdx = blockIdx; taskIdx < totalTaskNumP2; taskIdx += blockDim
```

如果目标平台 AIC/AIV core 数不同，需要保证无效 core 不越界；每个 phase 都必须用自己的 `totalTaskNumPx` 做循环边界。

## 4. Workspace 布局

workspace 只保存 Phase 1 到 Phase 2 的 partial 数据，不保存反向索引表。

### 4.1 总大小

```text
oBytes         = groupSize * D * sizeof(input_dtype)
oBytesAligned  = AlignUp(oBytes, 4)
statBytes      = groupSize * sizeof(float) * 2   # max + sum
partialSize    = oBytesAligned + statBytes
userWorkspace  = totalQTokens * kvHeads * topK * partialSize
workspaceSize  = libapiWorkspaceSize + userWorkspace
```

kernel 入口使用 `GetUserWorkspace(workspace)` 时，tiling 侧申请的 workspace 必须包含 `libapiWorkspaceSize`。

### 4.2 partial 内部布局

每个 partial 对应一个 `(qToken, kvHeadIdx, slotK)`：

```text
partial_base = ((qToken * kvHeads * topK) + (kvHeadIdx * topK) + slotK) * partialSize

partial_base + 0
  O_tmp[groupSize, D]          # input dtype，存未归一化的 sum(exp(score-rowMax) * V)

partial_base + oBytesAligned
  rowMax[groupSize]            # fp32

partial_base + oBytesAligned + groupSize * sizeof(float)
  rowSum[groupSize]            # fp32，sum(exp(score-rowMax))
```

注意：

1. `O_tmp` 不是归一化后的 partial output，而是未归一化分子。
2. `rowMax` 和 `rowSum` 是每个 Q head 一份，所以长度为 `groupSize`。
3. `rowMax/rowSum` 起始地址必须 4 字节对齐。
4. Phase 2 读取 topK partial 时，必须使用相同 `partialSize` 和 `oBytesAligned`。

## 5. Batch offset 与 logical block 编号

kernel 只接收 `q_seqlen` 和 `kv_seqlen`，batch offset 在 kernel 内计算：

```text
batchQOffset[0] = 0
batchKvBlockOffset[0] = 0
for b in [0, batch):
  batchQOffset[b + 1] = batchQOffset[b] + q_seqlen[b]
  batchKvBlockOffset[b + 1] = batchKvBlockOffset[b] + ceil(kv_seqlen[b] / blockSize)
```

`kvBlockIdx` 是跨 batch 展平后的 logical block id：

```text
batchIdx       = first b where kvBlockIdx < batchKvBlockOffset[b + 1]
localBlockIdx  = kvBlockIdx - batchKvBlockOffset[batchIdx]
physicalBlock  = block_table[batchIdx * maxBlocksPerBatch + localBlockIdx]
```

`qToken` 是跨 batch 展平后的 Q token id：

```text
qBatchIdx = first b where qToken < batchQOffset[b + 1]
localQIdx = qToken - batchQOffset[qBatchIdx]
```

## 6. Phase 1：KV-centric partial compute

### 6.1 Task 映射

```text
taskIdx    -> (kvBlockIdx, kvHeadIdx)
kvBlockIdx = taskIdx / kvHeads
kvHeadIdx  = taskIdx % kvHeads
```

每个 task 处理：

```text
1 logical KV block × 1 KV head × all selected Q tokens of this block
```

### 6.2 片上数据流

目标实现应按 tile 化 matmul 实现，而不是逐元素 GM 标量循环：

1. 从 `block_table` 读取 `physicalBlockId`。
2. 将该 physical block 的 `K[blockSize, D]` 和 `V[blockSize, D]` 搬到 L1/片上缓冲。K/V 对同一 `(kvBlockIdx, kvHeadIdx)` 只加载一次。
3. 从 `kv_to_q_select_num[kvBlockIdx]` 得到 `numQTokens`。
4. 按 `qTile` 遍历 Q token 列表。`qTile` 可以先取 1 保证功能正确，后续再扩展为多行 gather。
5. 对每个 `qTile`：
   - 根据 `kv_to_q_select_idx` 读取非连续 Q token id。
   - 对每个 Q token，搬运其 `groupSize` 个 Q head：`query[qToken, kvHeadIdx * groupSize + gh, :]`。
   - Q 在 GM 中可能非连续，必须逐行搬运；搬入后在片上整理成连续矩阵 `Q_tile[M, D]`，其中 `M = qTile * groupSize`。
6. Cube 执行：
   - `S = Q_tile[M, D] × K[D, blockSize]`
   - `S` 乘 `scaleValue`。
7. Vector 对 `S` 按 row 应用 causal mask 和有效长度 mask。
8. Vector 对每 row 计算 `rowMax` 和 `rowSum`。
9. Vector 得到 softmax 权重 `P = exp(S - rowMax)`。
10. Cube 或 Vector/Cube 组合执行 `O_tmp = P[M, blockSize] × V[blockSize, D]`。
11. 将每个 row 对应的 `(O_tmp, rowMax, rowSum)` 写入 workspace。

### 6.3 GQA 映射

一个 KV head 对应 `groupSize` 个 Q head：

```text
qHeadIdx = kvHeadIdx * groupSize + gh
for gh in [0, groupSize):
  query row = query[qToken, qHeadIdx, :]
```

同一个 `(kvBlockIdx, kvHeadIdx)` task 内，K/V 只加载一次，`groupSize` 个 Q head 共享该 K/V block。

### 6.4 Causal mask

每个 Q token 的 causal 边界不同，不能只按 block 统一 mask。

对每个 `qToken`：

```text
qBatchIdx      = FindBatchForQToken(qToken)
localQIdx      = qToken - batchQOffset[qBatchIdx]
qPosition      = kv_seqlen[qBatchIdx] - q_seqlen[qBatchIdx] + localQIdx
localBlockIdx  = kvBlockIdx - batchKvBlockOffset[qBatchIdx]
kvStartPos     = localBlockIdx * blockSize
causalValidLen = 0 if qPosition < kvStartPos
                 else min(validSize, qPosition - kvStartPos + 1)
```

mask 规则：

```text
for col in [0, blockSize):
  if col >= validSize or col >= causalValidLen:
    score[row, col] = -inf
```

当 `causalValidLen == 0` 时，该 partial 是无效 partial：

```text
O_tmp  = 0
rowMax = -inf
rowSum = 0
```

Phase 2 读取后应自然跳过该 partial。

### 6.5 workspace 写出

Phase 1 每个 `(qToken, kvHeadIdx, slotK)` 写一个 partial：

```text
slotK = kv_to_q_slot_idx[kvBlockIdx, qi]
base  = ((qToken * kvHeads * topK) + (kvHeadIdx * topK) + slotK) * partialSize
```

写出顺序：

```text
O_tmp[gh, d] -> base + gh * D * sizeof(dtype) + d * sizeof(dtype)
rowMax[gh]   -> base + oBytesAligned + gh * sizeof(float)
rowSum[gh]   -> base + oBytesAligned + (groupSize + gh) * sizeof(float)
```

O 写 workspace 时按 Q 行逐行写，因为 Q token 是 gather 得到的，不能假设连续。

## 7. Phase 2：Vector-only CombineScale

### 7.1 Task 映射

```text
taskIdx   -> (qToken, kvHeadIdx)
qToken    = taskIdx / kvHeads
kvHeadIdx = taskIdx % kvHeads
```

每个 task 输出该 `qToken` 在一个 KV head group 下的 `groupSize` 个 Q head：

```text
attention_out[qToken, kvHeadIdx * groupSize + gh, :]
```

### 7.2 Combine 公式

对每个 `gh`：

```text
globalMax = max(rowMax[k] for k in topK valid partials)
globalSum = sum(exp(rowMax[k] - globalMax) * rowSum[k])

out[d] = sum(exp(rowMax[k] - globalMax) * O_tmp[k][d]) / globalSum
```

其中 `O_tmp[k]` 是 Phase 1 写入的未归一化分子。

### 7.3 无效 slot 处理

若 `select_idx[qToken, slotK] < 0` 或该 partial 的 `rowSum == 0`，Phase 2 必须跳过该 slot。推荐 Phase 2 使用原始 `select_idx` 判断 slot 是否有效：

```text
kvBlockIdx = select_idx[qToken, slotK]
if kvBlockIdx < 0:
  continue
```

如果上层保证所有 `topK` slot 均有效，可以省略该判断，但该假设必须在算子接口中明确。

### 7.4 输出写回

```text
outAddr = qToken * numHeads * D + (kvHeadIdx * groupSize + gh) * D + d
```

Phase 2 必须全在 Vector 侧完成，不再触发 Cube 计算。

## 8. Kernel 实现边界

目标 kernel 不应停留在以下标量参考形态：

1. 不应在 GM 上对每个元素重复读取 K/V 做点积。
2. 不应对每个 `(qToken, qHead, d, n)` 用标量 C++ 循环模拟 QK 和 PV。
3. 不应在 Phase 1 对同一 KV block 的 K/V 反复从 GM 读取。
4. 不应忽略 `select_idx` 中无效 slot 的处理。

允许的阶段性实现顺序：

1. **功能优先版本**：`qTile = 1`，但 K/V 必须按 block 搬到片上，Q 逐行 gather，workspace/causal/combine 语义正确。
2. **性能版本**：`qTile > 1`，将多个 Q token 和 `groupSize` head 拼成 `M = qTile * groupSize`，用 Cube 做批量 QK/PV。
3. **优化版本**：加入 double buffer、L1/L0 preload、score/softmax UB 复用和 task 负载均衡。

## 9. 当前代码与目标实现的差距

当前 `sparse_attention_score_prefill_kernel_arch35.h` 只能作为 scalar reference，对目标 kernel 来说存在以下差距：

1. Phase 1 没有 `LoadKvBlockToL1` 的真实实现，K/V 没有按 block 复用。
2. Q 没有 gather 到连续片上矩阵，只是在 GM 中逐元素读取。
3. QK 没有用 Cube matmul，而是标量循环计算 dot。
4. PV 没有用 Cube matmul，而是标量循环累加。
5. 声明了 `LoadKvBlockToL1 / LoadQFromGM / ComputeQK / ComputePV` 等 helper，但没有实现，实际逻辑全部内联在标量循环里。
6. Phase 2 没有使用 `select_idx` 跳过无效 slot；如果存在 `-1` slot，会读未写 workspace。
7. 文档原先没有说明 `partial_O` 是未归一化分子，容易导致 CombineScale 实现错误。
8. 文档原先没有说明 `rowMax/rowSum` 的 byte offset 和 4 字节对齐要求。
9. 文档原先没有说明 `blockDim` 必须同时覆盖 Phase 1 和 Phase 2。
10. 文档原先没有说明反向索引是独立 GM 输入，而不是 workspace 的一部分。

## 10. 后续 kernel 修改建议

建议按以下顺序重构 kernel：

1. 保留当前 scalar reference 作为 golden 对照或 `#ifdef DEBUG_REFERENCE` 路径。
2. 实现 Phase 1 helper：
   - `LoadKvBlockToL1(physicalBlockId, kvHeadIdx, validSize)`
   - `GatherQTileFromGM(kvBlockIdx, kvHeadIdx, qiStart, qTile)`
   - `ComputeQKTile()`
   - `ApplyMaskAndSoftmaxTile()`
   - `ComputePVTile()`
   - `WritePartialTileToWorkspace()`
3. Phase 1 先使用 `qTile = 1` 对齐功能，再扩展到 `qTile > 1`。
4. Phase 2 增加 `select_idx` slot 有效性判断，或在接口中强约束 `topK` 全有效。
5. 针对 D=128/blockSize=128 固化 tiling key，再扩展其他 D/blockSize。
