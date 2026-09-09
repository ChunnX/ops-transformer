# turbo_quant_sparse_flash_attention

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

面向DeepSeek Sparse Attention（DSA）推理场景，对TQ4分页KV cache执行融合反量化与稀疏Attention计算。query包含512维latent和64维rope，输出为512维latent，rope仅参与QK计算。

KV slot包含256字节打包码本索引、128字节与query同型的rope/scale和2字节float16 scale；value传入与key相同的Tensor。码本、scale定义和latent坐标域约定见[《算子说明》](../../../../attention/turbo_quant_sparse_flash_attention/README.md#功能说明)。

## 函数原型

```python
cann_ops_transformer.turbo_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    scale_value=1.0,
    key_quant_mode=3,
    value_quant_mode=3,
    sparse_block_size=1,
    layout_query="TND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
    return_softmax_lse=False,
) -> (attention_out, softmax_max, softmax_sum)
```

## 参数说明

B为batch数，T为query token总数，N为query head数，Bn为物理页数，Bs为每页token数，K为稀疏块索引槽位数，maxBlocks为每个batch的页表容量。`*`之后的参数只能通过关键字传入。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| query | Tensor | 必选 | 包含latent和rope的查询张量。支持T = 0的空Tensor；N为[1,128]内2的幂。 | float16、bfloat16 | [T,N,576] |
| key | Tensor | 必选 | 打包的分页KV cache。不可为空Tensor；仅支持首轴非连续，详见约束说明。 | int8 | PA_BSND：[Bn,Bs,1,386]；PA_BNBD：[Bn,1,Bs,386] |
| value | Tensor | 必选 | 与key共享latent的值张量。传入与key相同的张量。 | 与key一致 | 与key一致 |
| sparse_indices | Tensor | 必选 | batch内局部稀疏块索引。T = 0时支持空Tensor；有效前缀后以-1填充。 | int32 | [T,1,K] |
| key_dequant_scale | Tensor或None | 可选 | key的独立反量化系数。可传None或空Tensor；当前COMBINE模式不读取其数值。默认值为None。 | float32 | 不限制 |
| value_dequant_scale | Tensor或None | 可选 | value的独立反量化系数。可传None或空Tensor；当前COMBINE模式不读取其数值。默认值为None。 | float32 | 不限制 |
| block_table | Tensor | 必选 | KV cache物理页映射表。不可为空Tensor；页号必须有效。 | int32 | [B,maxBlocks] |
| actual_seq_lengths_query | Tensor | 必选 | 各batch的query长度累加和。不可为空Tensor；非负、单调不减，末项等于T。 | int32 | [B] |
| actual_seq_lengths_kv | Tensor | 必选 | 各batch的有效KV长度。不可为空Tensor；允许值为0；每项不超过对应页表行的容量。 | int32 | [B] |
| scale_value | float | 可选 | QK分数的缩放系数。通常按模型设置，如1/sqrt(512)。默认值为1.0。 | - | - |
| key_quant_mode | int | 可选 | key的量化模式。仅支持3（TQ4）。默认值为3。 | - | - |
| value_quant_mode | int | 可选 | value的量化模式。仅支持3（TQ4）。默认值为3。 | - | - |
| sparse_block_size | int | 可选 | 每个稀疏索引对应的token数。仅支持1、2、4、8、16，必须整除Bs。默认值为1。 | - | - |
| layout_query | str | 可选 | query的逻辑布局。仅支持"TND"。默认值为"TND"。 | - | - |
| layout_kv | str | 可选 | key和value的逻辑布局。支持"PA_BSND"、"PA_BNBD"，不接受"PA_BNSD"或"BNBD"别名。默认值为"PA_BSND"。 | - | - |
| sparse_mode | int | 可选 | 稀疏注意力的mask模式。0表示无因果mask，3表示右下对齐因果mask。默认值为3。 | - | - |
| pre_tokens | int | 可选 | 历史token窗口属性。仅支持INT64_MAX。默认值为9223372036854775807。 | - | - |
| next_tokens | int | 可选 | 未来token窗口属性。仅支持INT64_MAX。默认值为9223372036854775807。 | - | - |
| attention_mode | int | 可选 | Attention计算模式。仅支持2（MLA-absorb）；本接口适用于DSA推理。默认值为2。 | - | - |
| quant_scale_repo_mode | int | 可选 | 反量化系数存储模式。仅支持1（COMBINE），scale位于KV slot尾部。默认值为1。 | - | - |
| tile_size | int | 可选 | 量化粒度属性。仅支持128。默认值为128。 | - | - |
| rope_head_dim | int | 可选 | rope分量维度。仅支持64。默认值为64。 | - | - |
| return_softmax_lse | bool | 可选 | 是否输出softmax_max和softmax_sum。True返回两个统计量，False返回两个空Tensor；不直接返回log-sum-exp。默认值为False。 | - | - |

## 返回值说明

返回由以下三个Tensor组成的tuple。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| attention_out | Tensor | 必选 | Attention输出，支持T = 0的空Tensor。 | 与query一致 | [T,N,512] |
| softmax_max | Tensor | 必选 | softmax逐行最大值；return_softmax_lse为False时为空Tensor。 | float32 | True时[1,T,N]，False时[0] |
| softmax_sum | Tensor | 必选 | softmax逐行指数和；return_softmax_lse为False时为空Tensor。 | float32 | True时[1,T,N]，False时[0] |

## 约束说明

- 该接口仅支持推理前向，不提供反向传播。
- 支持单算子调用及ACL graph捕获和回放。支持Meta/FakeTensor推导输出；未提供GE Converter，不声明支持TorchAir图模式。
- 实际执行时全部输入必须位于同一NPU。query和输出为float16或bfloat16，cache内的rope与query同型，scale固定为float16。
- B、Bn和maxBlocks必须大于0，Bs为[16,1024]内16的倍数。KV head数固定为1，slot宽度固定为386字节。
- cache仅支持第0轴非连续、页间padding和合法storage_offset，页内必须连续；多页时stride(0)至少为Bs × 386字节。不支持负stride或重叠页，单元素轴不约束stride，寻址跨度不得超过INT64_MAX。
- sparse_indices中的索引i表示本batch中从i × sparse_block_size开始的稀疏块，不是物理页号；有效索引在每行前缀，无效后缀填-1。
- 调用方保证索引、长度和页表的设备数值合法。actual_seq_lengths_query为非负单调不减的累加和，末项等于T；actual_seq_lengths_kv为非负的逐batch长度，每项不超过对应页表容量；页表中的物理页号必须落在cache范围内。
- query长度为0的batch不产生输出行；KV长度为0的batch输出零attention、max为-2e38、sum为0。整体T = 0时返回空Tensor，其余元数据仍须合法；不支持空的物理KV cache。
- KV非空但稀疏索引全为-1时，attention为0、max为-2e38，sum不保证为0。不能仅用sum判断行有效性。
- ACL graph捕获前需完成首次调用及JIT编译；回放只更新固定shape张量的内容，不改变其地址。

## 确定性计算

默认支持确定性计算。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_transformer

# 单个 KV token：latent 每个 nibble 为 code 8，rope 为零，float16 scale = 1。
query = torch.zeros((1, 8, 576), dtype=torch.float16, device="npu:0")
cache_cpu = torch.zeros((1, 16, 1, 386), dtype=torch.uint8)
cache_cpu[..., :256] = 0x88
cache_cpu[..., 385] = 0x3c
cache = cache_cpu.view(torch.int8).npu()
out, maximum, total = cann_ops_transformer.turbo_quant_sparse_flash_attention(
    query, cache, cache,
    torch.zeros((1, 1, 1), dtype=torch.int32, device=query.device),
    block_table=torch.zeros((1, 1), dtype=torch.int32, device=query.device),
    actual_seq_lengths_query=torch.ones(1, dtype=torch.int32, device=query.device),
    actual_seq_lengths_kv=torch.ones(1, dtype=torch.int32, device=query.device),
    scale_value=1.0 / (512 ** 0.5), return_softmax_lse=True,
)
torch.npu.synchronize()
print(out.shape, maximum.shape, total.shape)
```

旧Python调用入口见[兼容说明](../../../../attention/turbo_quant_sparse_flash_attention/torch_ops_extension/README.md)。
