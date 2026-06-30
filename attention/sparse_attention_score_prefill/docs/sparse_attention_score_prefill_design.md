# SparseAttentionScorePrefill 算子设计文档

## 1. 算子目标

`sparse_attention_score_prefill` 面向 Prefill 阶段的 Sparse Attention，采用 **KV-Gather-Q** 方案：以 KV block 为主轴分任务，同一个 KV block 只加载一次，然后遍历所有选择该 block 的 Q token，计算该 block 对这些 Q token 的 partial attention。

目标实现如下数据流：

1. Phase 1 在 Cube 侧完成 `QK + softmax + PV` 的 partial 计算。
2. Phase 1 以 `1 logical KV block × 1 KV head` 为 task 粒度。
3. Q token 通过 host 提供的反向索引 gather，Q 在 GM 中可能非连续，因此 Q 需要逐行搬运到连续的片上临时区后参与 matmul。
4. Phase 1 将每个 `(qToken, kvHead, slotK)` 的 `O_tmp`（按 row 归一化）与 `LSE` 写入 workspace。
5. Phase 2 在 Vector 侧读取每个 Q 的 topK partial，通过 LSE logsumexp 加权 combine，写最终 `attention_out`。
6. Phase 1 与 Phase 2 在同一个 kernel 内执行，中间通过 `SyncAll` 做全局同步。

稀疏选择关系由 host 预构建的反向索引张量表达；block 有效长度由 kernel 根据 `actual_seq_lengths_kv` 与 `blockSize` 在片上计算。

## 2. 输入、输出与辅助张量

### 2.1 算子输入

| Tensor | Shape | Type | 说明 |
| --- | --- | --- | --- |
| `query` | `[total_q_tokens, num_q_heads, D]` | bf16/fp16 | TND 格式 Q。`qToken` 是跨 batch 展平后的 token 下标。 |
| `key` | `[num_physical_blocks, blockSize, kvHeads, D]` | bf16/fp16 | Paged K Cache。 |
| `value` | `[num_physical_blocks, blockSize, kvHeads, D]` | bf16/fp16 | Paged V Cache。 |
| `block_table` | `[batch, maxBlocksPerBatch]` | int32 | 每个 batch 内 logical block 到 physical block 的映射。第二维是寻址 stride，必须来自 shape，不能由 `kv_seqlen` 推导。 |
| `k2q_row_ptr` | `[kvHeads, totalPackedRows + 1]` | int32 | Dense packed-row CSR 行指针；`totalPackedRows = sum(ceil(kv_seqlen/blockSize))`。 |
| `k2q_q_indices` | `[kvHeads, totalQTokens * topK]` | int32 | CSR 列数据：global `qToken` id，按 head 分段存储。 |
| `k2q_slot_indices` | `[kvHeads, totalQTokens * topK]` | int32 | 每条边在 workspace 中的 `slotK`，范围 `[0, topK)`。 |
| `actual_seq_lengths` | `[batch]` | int32 | 每个 batch 的 Q seqlen。 |
| `actual_seq_lengths_kv` | `[batch]` | int32 | 每个 batch 的 KV seqlen。 |


### 2.2 反向索引输入约束

`k2q_row_ptr`、`k2q_q_indices`、`k2q_slot_indices` 由 host 在调用算子前构建（CSR 格式，对齐 CUDA `build_k2q_csr_reference`），并作为 GM 输入传入 kernel，不放入 workspace。

语义约定：

1. Host 侧 forward 选块为 `select_idx[kvHeads, totalQTokens, topK]`（可用 global logical id 表达，构建 CSR 时转为 batch-local q2k）与 `select_num_idx[kvHeads, totalQTokens]`。
2. **`totalPackedRows`** = `sum_b ceil(kv_seqlen[b]/blockSize)`，与 MSA/CUDA `build_k2q_csr` 一致；未选中的块对应 CSR 空行（`row_ptr[r+1]==row_ptr[r]`）。
3. Phase1 task：`taskIdx = packedRow * kvHeads + kvHeadIdx`，其中 `packedRow` 由 `(batch, local_kv_block)` 按 MSA 规则打包。
4. stride 分核遍历 `taskIdx`；`packedRow = taskIdx / kvHeads` 变化时在循环内增量步进 MSA coord（见 §2.4），再经 `block_table` 取 physical block。单测 `decode_packed_row` 用搜索式解码校验同一语义。
5. `csrStart = k2q_row_ptr[kvHeadIdx, packedRow]`，`csrEnd = k2q_row_ptr[kvHeadIdx, packedRow + 1]`，`numQTokens = csrEnd - csrStart`。
6. `qToken = k2q_q_indices[kvHeadIdx, csrStart + qi]`（global flattened q id），`slotK = k2q_slot_indices[...]`。
7. `k2qNnzUpperBound = k2q_q_indices.shape[1]`，通常为 `totalQTokens * topK`。
8. GQA 下不同 `kvHeadIdx` 可有不同 topK 选块。


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
| `topK` | 每个 Q token 选择的 KV block 数（attr）。 |
| `totalQTokens` | 所有 batch 的 Q token 总数，来自 `query.shape[0]`。 |
| `totalPackedRows` / `numKvBlocks` | `k2q_row_ptr.shape[1] - 1` = `sum_b ceil(kv_seqlen[b]/blockSize)`。tiling 字段名为 `numKvBlocks`。 |
| `maxBlocksPerBatch` | `block_table.shape[1]`，用于 `block_table` 行 stride。 |
| `k2qNnzUpperBound` | `k2q_q_indices.shape[1]`，CSR 数据区每 head 的上界，通常 `totalQTokens * topK`。 |
| `totalTaskNumP1` | Phase 1 task 数，`totalPackedRows * kvHeads`。 |
| `totalTaskNumP2` | Phase 2 task 数，`totalQTokens * kvHeads`。 |

### 3.2 blockDim 策略

同一个 kernel 内包含 Cube Phase 1 和 Vector Phase 2，因此 `blockDim` 不能只按照 Phase 1 计算。推荐：

```text
totalTaskP1 = totalPackedRows * kvHeads
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
oBytes         = groupSize * D * sizeof(float)    # O_tmp 存 fp32（PV 累加精度），与 Q/K/V input dtype 无关
oBytesAligned  = oBytes                          # 当前 tiling/kernel 无额外 padding；groupSize*D*4 恒为 4 字节对齐
statBytes      = groupSize * sizeof(float)       # LSE[groupSize]
partialSize    = oBytesAligned + statBytes
userWorkspace  = totalQTokens * kvHeads * topK * partialSize
workspaceSize  = libapiWorkspaceSize + userWorkspace
```

kernel 入口使用 `GetUserWorkspace(workspace)` 时，tiling 侧申请的 workspace 必须包含 `libapiWorkspaceSize`。

Phase 1 启动前应将 workspace 中所有 partial 的 `LSE` 预置为 `-inf`（或保证未写入 slot 可被 Phase 2 识别为无效）。

### 4.2 partial 内部布局

每个 partial 对应一个 `(qToken, kvHeadIdx, slotK)`：

```text
partial_base = ((qToken * kvHeads * topK) + (kvHeadIdx * topK) + slotK) * partialSize

partial_base + 0
  O_tmp[groupSize, D]          # fp32，存按 row 归一化的 partial output（= sum(exp(S-rowMax)*V) / rowSum）

partial_base + oBytesAligned
  LSE[groupSize]               # fp32，LSE[gh] = rowMax[gh] + log(rowSum[gh])
```

注意：

1. `O_tmp` 是 row 内 softmax 归一化后的 partial output，不是未归一化分子。
2. `LSE` 是每个 Q head 一份，长度为 `groupSize`；无效 slot 为 `-inf`。
3. `LSE` 起始地址必须 4 字节对齐。
4. Phase 2 读取 topK partial 时，必须使用相同 `partialSize` 和 `oBytesAligned`。

## 5. Batch offset、packed row 与有效长度

### 5.1 Batch offset

kernel 根据 `actual_seq_lengths` 和 `actual_seq_lengths_kv` 在片上计算 batch offset：

```text
batchQOffset[0] = 0
batchKvBlockOffset[0] = 0
for b in [0, batch):
  batchQOffset[b + 1] = batchQOffset[b] + q_seqlen[b]
  batchKvBlockOffset[b + 1] = batchKvBlockOffset[b] + ceil(kv_seqlen[b] / blockSize)
```

**Phase1** 不直接使用跨 batch 展平的 `kvBlockIdx`，而是通过 §2.4 的 `(packedRow → batchIdx, localBlockIdx)` 定位 logical KV block：

```text
physicalBlock = block_table[batchIdx * maxBlocksPerBatch + localBlockIdx]
```

**Q token** 仍按展平 id 反查所属 batch（与 KV block 的 batch 可不同，causal 用 Q 侧 seqlen）：

```text
qBatchIdx = first b where qToken < batchQOffset[b + 1]
localQIdx = qToken - batchQOffset[qBatchIdx]
```

`batchKvBlockOffset` 仍可用于 host 侧将 global logical block id 转为 batch-local id（构建 q2k/k2q CSR 时）。

### 5.2 Block 有效长度

每个 logical KV block 的有效 token 数在 kernel 内根据 `actual_seq_lengths_kv` 计算：

```text
kvSeqlenBatch = kv_seqlen[batchIdx]
tailRemain    = kvSeqlenBatch - localBlockIdx * blockSize
validSize     = min(blockSize, tailRemain)   # tailRemain > 0 时有效
```

- 非尾块：`validSize = blockSize`
- 尾块：`validSize = kvSeqlenBatch % blockSize`；若整除则 `validSize = blockSize`
- `tailRemain <= 0` 的 block 不应出现在有效 `numKvBlocks` 范围内

## 6. Phase 1：KV-centric partial compute

### 6.1 Task 映射

```text
taskIdx    -> (packedRow, kvHeadIdx)
packedRow  = taskIdx / kvHeads
kvHeadIdx  = taskIdx % kvHeads
(batchIdx, localBlockIdx) from inline MSA coord step when packedRow advances
```

Phase1 分核：

```text
for taskIdx = blockIdx; taskIdx < totalTaskNumP1; taskIdx += blockDim
```

每个 task 处理：

```text
1 logical KV block × 1 KV head × all selected Q tokens of this block
```

### 6.2 片上数据流

目标实现应按 tile 化 matmul 实现，而不是逐元素 GM 标量循环：

1. 从 `block_table` 读取 `physicalBlockId`。
2. 根据 §5.2 计算 `validSize`。
3. 将该 physical block 当前 `kvHeadIdx` 的 `K[validSize, D]`、`V[validSize, D]` 搬到 L1/片上缓冲；每个 `(packedRow, kvHeadIdx)` task 加载一次，matmul 仅用前 `validSize` 行。
4. 从 CSR 得到 `numQTokens`；`packedRow` 变化时 inline 步进得 `(batchIdx, localBlockIdx)` 后查 `block_table`。
5. 按 `qi in [0, numQTokens)` 遍历 Q 列表。
6. 对每个 `qi`：
   - `qToken = k2q_q_indices[kvHeadIdx, csrStart + qi]`。
   - 对每个 Q token，搬运其 `groupSize` 个 Q head：`query[qToken, kvHeadIdx * groupSize + gh, :]`。
   - Q 在 GM 中可能非连续，必须逐行搬运；搬入后在片上整理成连续矩阵 `Q_tile[M, D]`，其中 `M = qTile * groupSize`。
7. Cube 执行：
   - `S = Q_tile[M, D] × K[D, validSize]`
   - `S` 乘 `scaleValue`。
8. Vector 对 `S` 按 row 应用 causal mask 和有效长度 mask。
9. Vector 对每 row 计算 `rowMax`、`rowSum`，并得到 `LSE = rowMax + log(rowSum)`。
10. Vector 得到 softmax 权重 `P = exp(S - rowMax)`。
11. Cube 或 Vector/Cube 组合执行 `O_unnorm = P[M, validSize] × V[validSize, D]`。
12. Vector 将 `O_tmp = O_unnorm / rowSum` 与 `LSE` 写入 workspace。

### 6.3 GQA 映射

一个 KV head 对应 `groupSize` 个 Q head：

```text
qHeadIdx = kvHeadIdx * groupSize + gh
for gh in [0, groupSize):
  query row = query[qToken, qHeadIdx, :]
```

同一个 `(packedRow, kvHeadIdx)` task 内，K/V 只加载一次，`groupSize` 个 Q head 共享该 K/V block。

### 6.4 Causal mask

每个 Q token 的 causal 边界不同，不能只按 block 统一 mask。

对每个 `qToken`（causal 边界由 **Q 所在 batch** 的 seqlen 决定；KV block 位置由 **packedRow 解码得到的 `localBlockIdx`** 决定）：

```text
qBatchIdx      = FindBatchForQToken(qToken)
localQIdx      = qToken - batchQOffset[qBatchIdx]
qPosition      = kv_seqlen[qBatchIdx] - q_seqlen[qBatchIdx] + localQIdx
kvStartPos     = localBlockIdx * blockSize    # localBlockIdx 来自 packedRow 的 MSA coord
causalValidLen = 0 if qPosition < kvStartPos
                 else min(validSize, qPosition - kvStartPos + 1)
```

mask 规则：

```text
for col in [0, blockSize):
  if col >= validSize or col >= causalValidLen:
    score[row, col] = -inf
```

当 `causalValidLen == 0` 时，kernel/golden **跳过写入**该 partial，依赖 workspace 预置 `LSE == -inf`，Phase 2 自动跳过：

```text
# 不写入 workspace；slot 保持 LSE == -inf
```

### 6.5 workspace 写出

Phase 1 每个 `(qToken, kvHeadIdx, slotK)` 写一个 partial：

```text
csrStart = k2q_row_ptr[kvHeadIdx, packedRow]
slotK = k2q_slot_indices[kvHeadIdx, csrStart + qi]
base  = ((qToken * kvHeads * topK) + (kvHeadIdx * topK) + slotK) * partialSize
```

写出顺序：

```text
O_tmp[gh, d] -> base + gh * D * sizeof(float) + d * sizeof(float)
LSE[gh]      -> base + oBytesAligned + gh * sizeof(float)
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

### 7.2 Combine 公式（kernel 与 golden 共用 LSE logsumexp）

对每个 `gh`，在 topK 个有效 partial 上：

```text
LSE_global = logsumexp(LSE[k])   # 稳定实现：max(LSE) + log(sum(exp(LSE[k] - max(LSE))))

weight[k] = exp(LSE[k] - LSE_global)

out[d] = sum(weight[k] * O_tmp[k][d])
```

其中 `O_tmp[k]` 是 Phase 1 写入的 row 归一化 partial，`LSE[k] = rowMax[k] + log(rowSum[k])`。与旧版 online max/sum 公式数学等价：

```text
out = sum(exp(rowMax-rowMax_g)*rowSum*O_norm) / sum(exp(rowMax-rowMax_g)*rowSum)
    = sum(exp(LSE-LSE_g)*O_norm)
```

### 7.3 无效 slot 处理

Phase 2 通过 workspace 内容判断 slot 是否有效：

```text
for slotK in [0, topK):
  base = ((qToken * kvHeads * topK) + (kvHeadIdx * topK) + slotK) * partialSize
  if LSE[gh] == -inf:    # 无效 slot
    continue
  # 参与 combine
```

规则：

1. Phase 1 仅对 CSR 中列出的 `(qToken, slotK)` 写入有效 partial。
2. 未被 Phase 1 写入的 slot 保持 `LSE == -inf`，Phase 2 必须跳过。
3. causal 全 mask 的 partial 不写入 workspace，slot 保持 `LSE == -inf`，Phase 2 自动跳过。
4. 若上层保证每个 Q token 恰好有 `topK` 个有效 block 且全部写入，可省略额外判断，但该假设必须在调用侧明确。

### 7.4 输出写回

```text
outAddr = qToken * numHeads * D + (kvHeadIdx * groupSize + gh) * D + d
```

Phase 2 必须全在 Vector 侧完成，不再触发 Cube 计算。

## 8. CPU Golden 参考实现

`tests/test_sparse_attention_score_prefill_golden.py` 与 kernel **任务划分、CSR 消费、MSA coord 步进、causal 规则** 对齐，数值路径略有不同：

| 项 | Kernel | CPU Golden |
| --- | --- | --- |
| Phase1 分核 | `taskIdx += blockDim` stride | `taskIdx` 顺序 0..totalTaskNumP1-1（等价 coreNum=1） |
| coord | `Init/AdvancePackedRowCoord` inline | `_init/_advance_packed_row_coord` inline |
| Phase1 partial 统计 | workspace 写 `O_tmp(fp32, row-normalized)` + `LSE` | 同 kernel |
| Phase2 combine | LSE logsumexp over topK partials（§7.2） | 同 kernel（§7.2） |
| QK/SM 中间精度 | QK 输出 fp32 S；SM 在 Vector fp32 | fp32 softmax；P cast bf16 供 PV |

Golden 不模拟 Cube tile 搬运；SM/PV 统计量在 fp32 计算，仅 P 矩阵 cast 为 bf16 以匹配 PV 输入 dtype。

另保留 `cpu_golden_prefill_qcentric_bf16` 作 Q-centric 交叉校验。

## 9. Kernel 实现边界

目标 kernel 不应停留在以下标量参考形态：

1. 不应在 GM 上对每个元素重复读取 K/V 做点积。
2. 不应对每个 `(qToken, qHead, d, n)` 用标量 C++ 循环模拟 QK 和 PV。
3. 不应在 Phase 1 对同一 `(packedRow, kvHeadIdx)` task 的 K/V 反复从 GM 读取。
4. Phase 2 必须能跳过无效 partial（`LSE == -inf` 或 causal 全 mask）。

当前 `sparse_attention_score_prefill_kernel_arch35.h` 已具备：

- Phase1/2 按 `totalTaskNumP1/P2` stride 分核。
- Phase1 `BlockMmadQK` / `BlockMmadPV` + `LoadKResident` / `LoadVResident` 按 block×head 复用 K/V。
- MSA packed row coord 在 task 循环内增量步进（无 GM 查表、无 per-task 搜索）。
- CSR 驱动 Q gather + causal mask + workspace 写出。

仍待完善（相对性能目标）：

1. Q 仍逐 token gather，尚未 `qTile > 1` 批量拼 Q。
2. Phase 2 无效 slot 跳过逻辑需与 workspace 清零策略一并验证。
3. D/blockSize 以外 shape 的 tiling key 扩展。
4. NPU 测试主要差在 golden 与 kernel 的 P cast 舍入及 Cube QK 累加精度。

## 10. 后续 kernel 修改建议

建议按以下顺序继续演进：

1. Phase 1 扩展 `qTile > 1`：批量 gather Q，拼 `M = qTile * groupSize` 做 Cube QK/PV。
2. 性能：
- PRE_LAUNCH流水
- double buffer
- L1/L0 preload
- score/softmax UB 复用
- 空 CSR 行 task 负载均衡
- Regbase优化
