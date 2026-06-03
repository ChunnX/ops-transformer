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

# 修改检视一

1. shape描述不要用()，要表达是数组，用[]
2. attention/kv_quant_sparse_flash_attention/README.md：

```
attention结构的Q输入，不支持非连续。query由相同数据类型的q_nope和q_rope按D维度拼接得到。layout_query为"BSND"时shape为(batch_size, seqlen_q, nheads_q, headdim_q)，其中headdim_q值仅支持576，即q_nope+q_rope=512+64。layout_query为"TND"时shape为(total_q, nheads_q, headdim_q)。nheads_q值支持1/2/4/8/16/32/48/64/128。
```
我觉得你把headdim_q和nheads_q约束条件分别写在两个layout_query情况后，要评估一下，只有当headdim_q值仅支持576仅支持layout_query为"BSND"；nheads_q值支持1/2/4/8/16/32/48/64/128仅支持layout_query为"TND"时，这种写法合理，你查看代码评估一下

# 修改检视二

1. 经过评估，shape后续还是放弃`轴名_参数名`的命名风格，采取简写的风格，这个仓在/mnt/workspace/gitCode/cann/ops-transformer有备份，你可以借鉴备份的attention下面sparse_flash_attention、kv_quant_sparse_flash_attention、lightning_indexer、quant_lightning_indexer四个算子下面的README.md，来还原原先的命名风格，不过原来的的命名不统一现象要整改掉，比如
```
query的描述为：layout_query为BSND时，shape为(B,S1,N1,D)；layout_query为TND时，shape为(T1,N1,D)。
sparseIndices的描述为：layout_query为BSND时，shape为(B, Q_S, KV_N, sparse_size)；layout_query为TND时，shape为(Q_T, KV_N, sparse_size)。
不统一，标识混乱
```
正常来说，S1就是Q_S，S2就是KV_S，简写风格后面的数字标识：1为Q，2为KV。S1、S2、N1、N2……这种是内部标识，但外部客户不明白，所以需要整改掉，另外在参数表格的上面加上参数维度含义的改动还是很成功的，对简写风格的shape维度做出解释，但表格里就采取简写风格，视觉上更直观。