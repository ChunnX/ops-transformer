# aclnnGenericLightningIndexer

- **算子**：`GenericLightningIndexer`
- **完整需求**：见同目录 [generic_lightning_indexer_design.md](generic_lightning_indexer_design.md)

## 功能

对已经完成前处理的 Indexer Q/K 做 `Matmul → 激活 → 沿 head 规约 → 掩码 → TopK →（可选 Expand / 追加尾 token）`。

不包含 RoPE / RMSNorm / AvgPool / CSA 压缩。压缩后的 Key 通过 `cmp_ratio` 消费。

## 收编剖面

| 剖面 | cmp_ratio | act | weighted | sparse_mode | expand | append |
|---|---|---|---|---|---|---|
| LightningIndexer / QLI | 1 | relu | yes | 0/3 | no | no |
| LightningIndexerV2 / QLIv2 | r | relu | yes | 0/3 | no | no |
| PoolKeyIndexer | pool_size | relu | yes | 0/3 | yes | yes |
| Qwen CLI | 4 | relu | **no** | **4** | yes | yes |
| MsaIndexScore | Step 2：`select_mode=1` / `key_reduce_mode=1` 本阶段 tiling 拒绝 | | | | | |

`topk` 一律表示**打分轴**选出的个数。旧 PKI 的 token 预算需先除以 `pool_size`。

## 原型（aclnn 两段式）

```c
aclnnStatus aclnnGenericLightningIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *actualSeqQ, const aclTensor *actualSeqK, const aclTensor *cmpResidualK,
    const aclTensor *blockTable, const aclTensor *qDescale, const aclTensor *kDescale,
    const aclTensor *metadata,
    int64_t topk, int64_t cmpRatio, char *layoutQ, char *layoutK, int64_t sparseMode,
    int64_t actType, bool hasWeightedSum, bool expandIndices, bool appendTail,
    int64_t quantMode, bool returnValue, int64_t selectMode, int64_t keyReduceMode,
    float scoreScale, int64_t keyStride0,
    const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnGenericLightningIndexer(void *workspace, uint64_t workspaceSize,
                                         aclOpExecutor *executor, aclrtStream stream);
```

调用示例：`examples/test_aclnn_generic_lightning_indexer.cpp`。

## 约束（Step 1）

- HeadDim=128，Key N=1，Query N∈[1,64]
- `act_type` 仅 0（ReLU）；identity 与 MSA maxpool 预留到 Step 2
- `append_tail=true` 需要 `expand_indices=true`
- `has_weighted_sum=true` 必须传 weights，反之必须为空
