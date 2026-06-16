# MixedQuantSparseFlashMlaMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：`MixedQuantSparseFlashMlaMetadata`算子旨在生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引，供后续`MixedQuantSparseFlashMla`算子使用。

## 参数说明

| 参数名           |输入/输出/属性|    描述    | 数据类型    |数据格式|
|-------------|------------|------|-----|-----|
|cu_seqlens_q|可选输入|表示不同Batch中Query的有效Sequence Length，维度为B+1，仅layout_q为TND场景需传入|INT32|ND|
|cu_seqlens_ori_kv|可选输入|表示不同Batch中ori_kv的有效Sequence Length，维度为B+1，仅layout_kv为TND场景需传入|INT32|ND|
|cu_seqlens_cmp_kv|可选输入|表示不同Batch中cmp_kv的有效Sequence Length，维度为B+1，仅layout_kv为TND场景需传入|INT32|ND|
|seqused_q|可选输入|表示不同Batch中Query实际参与运算的Sequence Length，维度为B|INT32|ND|
|seqused_ori_kv|可选输入|表示不同Batch中ori_kv实际参与运算的Sequence Length，维度为B|INT32|ND|
|seqused_cmp_kv|可选输入|表示不同Batch中cmp_kv实际参与运算的Sequence Length，维度为B|INT32|ND|
|cmp_residual_kv|可选输入|表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算。cmp_mask_mode=3且cmp_ratio≠1时必须传入，维度为B|INT32|ND|
|ori_topk_length|可选输入|预留参数，当前不生效|INT32|ND|
|cmp_topk_length|可选输入|预留参数，当前不生效|INT32|ND|
|num_heads_q|属性|表示Query的head个数|INT32|-|
|num_heads_kv|属性|表示Key和Value对应的多头数|INT32|-|
|head_dim|属性|表示注意力头的维度|INT32|-|
|quant_mode|属性|表示量化模式，1表示K、V nope为per-token-group量化，scale类型为bfloat16，2表示K、V nope为per-token-group量化，scale类型为float8_e8m0|INT32|-|
|batch_size|可选属性|表示Batch数量，默认值为0|INT32|-|
|max_seqlen_q|可选属性|表示Query的最长Sequence Length，默认值为0|INT32|-|
|max_seqlen_ori_kv|可选属性|表示ori_kv的最长Sequence Length，默认值为0|INT32|-|
|max_seqlen_cmp_kv|可选属性|表示cmp_kv的最长Sequence Length，默认值为0|INT32|-|
|ori_topk|可选属性|表示ori_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景，默认值为0|INT32|-|
|cmp_topk|可选属性|表示cmp_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景，默认值为0|INT32|-|
|rope_head_dim|可选属性|表示rope头的维度，默认值为64|INT32|-|
|cmp_ratio|可选属性|表示对ori_kv的压缩率，取值范围[1, 128]，默认值为1|INT32|-|
|ori_mask_mode|可选属性|表示q和ori_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，4表示sliding window模式，默认值为0|INT32|-|
|cmp_mask_mode|可选属性|表示q和cmp_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，默认值为0|INT32|-|
|ori_win_left|可选属性|表示q和ori_kv计算中q对过去token计算的数量，-1表示无穷大，默认值为-1|INT32|-|
|ori_win_right|可选属性|表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大，默认值为-1|INT32|-|
|layout_q|可选属性|表示Query的排列格式，支持BSND、TND，默认值为BSND|STRING|-|
|layout_kv|可选属性|表示Key的排列格式，支持BSND、TND、PA_BBND，默认值为BSND|STRING|-|
|has_ori_kv|可选属性|用于标识是否含有ori_kv，默认值为true|BOOL|-|
|has_cmp_kv|可选属性|用于标识是否含有cmp_kv，默认值为true|BOOL|-|
|metadata|输出|表示负载均衡结果输出，shape固定为[1024]|INT32|ND|

## 约束说明

-   SparseFlashMlaMetadata算子需要与SparseFlashMla算子配套使用。
-   参数num_heads_kv仅支持1。
-   参数cu_seqlens_q、cu_seqlens_ori_kv及cu_seqlens_cmp_kv维度为B + 1，要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
-   参数seqused_q、seqused_ori_kv、seqused_cmp_kv维度为B，要求其值表示每个Batch中的有效token数。
-   参数cmp_residual_kv需满足cmp_residual_kv[i] < cmp_ratio。
-   ori\_mask\_mode及cmp\_mask\_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/sparse_mode%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E.md)。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_mixed_quant_sparse_flash_mla_metadata.cpp](./examples/test_aclnn_mixed_quant_sparse_flash_mla_metadata.cpp) | 通过[aclnnMixedQuantSparseFlashMlaMetadata.md](./docs/aclnnMixedQuantSparseFlashMlaMetadata.md.md)调用MixedQuantSparseFlashMlaMetadata算子 |
| torch接口 | [mixed_quant_sparse_flash_mla.py](../../torch_extension/cann_ops_transformer/ops/mixed_quant_sparse_flash_mla.py) | 通过[mixed_quant_sparse_flash_mla_metadata](../../torch_extension/cann_ops_transformer/docs/zh/mixed_quant_sparse_flash_mla_metadata.md)接口方式调用MixedQuantSparseFlashMlaMetadata算子 |