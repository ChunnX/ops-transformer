# 问题
接到一个issue：https://gitcode.com/cann/ops-transformer/issues/2628
说attention下面的sparse_flash_attention、kv_quant_sparse_flash_attention、lightning_indexer、quant_lightning_indexer四个算子下面的README.md说明文档shape的描述混乱
比如attention/sparse_flash_attention/README.md下面
query的描述为：layout_query为BSND时，shape为(B,S1,N1,D)；layout_query为TND时，shape为(T1,N1,D)。
sparseIndices的描述为：layout_query为BSND时，shape为(B, Q_S, KV_N, sparse_size)；layout_query为TND时，shape为(Q_T, KV_N, sparse_size)。
不统一，标识混乱

后面评审后有新规范：
```
shape在资料内的展开写法如下所示(轴名_参数名，如果不同参数同一个轴数值都相等，则只需要体现轴名)：
q BNSD：(batch_size, seqlen_q, nheads_q, headdim_q)
ort_kv TND：(total_orikv, nheads_kv, headdim_kv)
cmp_kv PA_BBND：(block_num_cmpkv, block_size_cmpkv, nheads_kv, headdim_kv)
cmp_sparse_indices BSND：(batch_size, seqlen_q, nheads_kv, topk_cmpkv)
```
这些不一定对应真实算子的shape需求哈，请结合算子op_host或者op_kernel的真实实现代码的意义来改