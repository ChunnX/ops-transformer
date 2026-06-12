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

## 参数说明

| 参数名           |输入/输出/属性|    描述    | 数据类型    |数据格式|
|-------------|------------|------|-----|-----|
|cu_seqlens_q|可选输入|表示不同Batch中Query的有效Sequence Length，layout_q为TND场景下必传，以该入参的数量作为Batch值，第一个值为额外值并固定为0，shape固定为(B+1)|INT32|ND|
|cu_seqlens_ori_kv|可选输入|表示不同Batch中ori_kv的有效Sequence Length，layout_kv为TND场景下必传，以该入参的数量作为Batch值，第一个值为额外值并固定为0，shape固定为(B+1)|INT32|ND|
|cu_seqlens_cmp_kv|可选输入|表示不同Batch中cmp_kv的有效Sequence Length，layout_kv为TND场景下必传，以该入参的数量作为Batch值，第一个值为额外值并固定为0，shape固定为(B+1)|INT32|ND|
|seqused_q|可选输入|表示不同Batch中Query实际参与运算的Sequence Length，shape固定为(B)|INT32|ND|
|seqused_ori_kv|可选输入|表示不同Batch中ori_kv实际参与运算的Sequence Length，shape固定为(B)|INT32|ND|
|seqused_cmp_kv|可选输入|表示不同Batch中cmp_kv实际参与运算的Sequence Length，shape固定为(B)|INT32|ND|
|cmp_residual_kv|可选输入|表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算。如果cmp_ratio不为1，且mask为3，则必须传入cmp_residual_kv，shape固定为(B)|INT32|ND|
|ori_topk_length|可选输入|预留参数，表示不同q token对应的ori_kv部分关键稀疏token的个数，shape为(B, S1, N2)或(T1, N2)|INT32|ND|
|cmp_topk_length|可选输入|预留参数，表示不同q token对应的cmp_kv部分关键稀疏token的个数，shape为(B, S1, N2)或(T1, N2)|INT32|ND|
|num_heads_q|属性|表示Query的head个数|INT64|-|
|num_heads_kv|属性|表示Key和Value对应的多头数|INT64|-|
|head_dim|属性|表示注意力头的维度|INT64|-|
|quant_mode|属性|表示量化模式，1表示K、V nope为per-token-group量化，scale类型为bfloat16，2表示K、V nope为per-token-group量化，scale类型为float8_e8m0|INT64|-|
|batch_size|属性|表示Batch数量|INT64|-|
|max_seqlen_q|属性|表示Query的最长Sequence Length|INT64|-|
|max_seqlen_ori_kv|属性|表示ori_kv的最长Sequence Length|INT64|-|
|max_seqlen_cmp_kv|属性|表示cmp_kv的最长Sequence Length|INT64|-|
|ori_topk|属性|表示ori_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景|INT64|-|
|cmp_topk|属性|表示cmp_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景|INT64|-|
|rope_head_dim|属性|表示rope头的维度|INT64|-|
|cmp_ratio|属性|表示对ori_kv的压缩率，取值范围[1, 128]|INT64|-|
|ori_mask_mode|属性|表示q和ori_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，4表示sliding window模式|INT64|-|
|cmp_mask_mode|属性|表示q和cmp_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式|INT64|-|
|ori_win_left|属性|表示q和ori_kv计算中q对过去token计算的数量，-1表示无穷大|INT64|-|
|ori_win_right|属性|表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大|INT64|-|
|layout_q|属性|表示Query的排列格式，支持BSND、TND|STRING|-|
|layout_kv|属性|表示Key的排列格式，支持BSND、TND、PA_BBND|STRING|-|
|has_ori_kv|属性|用于标识是否含有ori_kv|BOOL|-|
|has_cmp_kv|属性|用于标识是否含有cmp_kv|BOOL|-|
|metadata|输出|表示负载均衡结果输出，shape固定为[1024]|INT32|ND|

## 约束说明