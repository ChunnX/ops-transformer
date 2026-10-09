# lightning_indexer_batch_invariant

## 功能说明

`lightning_indexer_batch_invariant` 是 `LightningIndexerV2BatchInvariant` 的 PyTorch 自定义接口。其计算功能和参数与
`lightning_indexer` 一致，并默认保证确定性和 batch 一致性：相同输入重复执行的输出完全一致，完全相同的单条样本也不因
batch 大小或 batch 内位置变化而改变输出。

```python
torch.ops.cann_ops_transformer.lightning_indexer_batch_invariant(
    q,
    k,
    w,
    topk,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
)
```

## 参数和返回值

参数含义、shape、dtype 和限制与 `lightning_indexer` 在 A2/A3 上一致。返回值为
`(sparse_indices, sparse_values)`，其中 `sparse_indices` 为 INT32，`sparse_values` 为 FLOAT32。

详细约束请参见
[aclnnLightningIndexerV2BatchInvariant](../../../../attention/lightning_indexer_v2_batch_invariant/docs/aclnnLightningIndexerV2BatchInvariant.md)。
