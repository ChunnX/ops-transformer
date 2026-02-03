# QuantLightningIndexerMetadata

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 推理系列产品</term>   | √  |

## 功能说明

-   API功能：QuantLightningIndexerMetadata是QuantLightningIndexer的前置算子，通过AICPU为QuantLightningIndexer算子生成分核结果，包括每个核需要处理的数据的起始点、结束点等内容，随后，QuantLightningIndexer根据该分核结果进行实际计算。

-   主要计算过程为：
    1. 获取每个`batch`的基本块大小，并计算负载。
    2. 计算所有`batch`的总负载和总的基本块个数。
    3. 为每个核分配负载，并记录分核结果，分核结果包括每个核需要处理的数据的起始点、结束点等内容。

## 函数原型

```
custom.npu_quant_lightning_indexer_metadata(num_heads_q, num_heads_k, head_dim, query_quant_mode,
key_quant_mode, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, batch_size=0, max_seqlen_q=0, max_seqlen_k=0, layout_query='BSND', layout_key='BSND', sparse_count=2048, sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1, cmp_ratio=1, device='npu:0') -> Tensor
```

## 参数说明
>**说明：**<br> 
>
>- 参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- 使用S1和S2分别表示query和key的输入样本序列长度，N1和N2分别表示query和key对应的多头数，k表示最后选取的索引个数。
-   **num\_heads\_q**（`int`）：必选参数，表示`query`对应的多头数。

-   **num\_heads\_k**（`int`）：必选参数，表示`key`对应的多头数。

-   **head\_dim**（`int`）：必选参数，表示hidden层的最小单元尺寸。

-   **query\_quant\_mode**（`int`）：必选参数，用于标识`query`的量化模式，当前支持Per-Token-Head量化模式，当前仅支持传入0。

-   **key\_quant\_mode**（`int`）：必选参数，用于标识输入`key`的量化模式，当前支持Per-Token-Head量化模式，当前仅支持传入0。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入；之后的参数是可选参数，位置无关，不赋值会使用默认值。

-   **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。不能出现负值。

-   **actual\_seq\_lengths\_key**（`Tensor`）：可选参数，表示不同Batch中压缩前原始`key`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和key的shape的S长度相同。该参数中每个Batch的原始有效token数除以压缩率后不超过`key`中的维度S大小且不小于0，支持长度为B的一维tensor。<br>当`layout_kv`为TND或PA_BSND时，该入参必须传入，`layout_kv`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

-   **batch\_size**（`int`）：可选参数，用于表示输入样本批量大小，默认值为0。

-   **max\_seqlen\_q**（`int`）：可选参数，用于表示输入样本批次中最大的`query`长度，默认值为0。

-   **max\_seqlen\_k**（`int`）：可选参数，用于表示输入样本批次中最大的`key`长度，默认值为0。

-   **layout\_query**（`str`）：可选参数，用于标识`query`的数据排布格式，当前支持BSND、TND，默认值"BSND"。

-   **layout\_key**（`str`）：可选参数，用于标识`key`的数据排布格式，当前支持PA_BSND、BSND、TND，默认值"BSND"。在非PageAttention场景下，layout\_key应与layout\_query保持一致。

-   **sparse\_count**（`int`）：可选参数，代表topK阶段需要保留的block数量，支持[1, 2048]，数据类型支持`int32`。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式，支持0/3，数据类型支持`int32`。 sparse\_mode为0时，代表defaultMask模式。sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

-   **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

-   **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

-   **cmp\_ratio**（`int`）：可选参数，用于稀疏计算，表示key的压缩倍数。数据类型支持`int32`，默认值1，支持1/2/4/8/16/32/64/128。

-   **device**（`str`）：可选参数，用于表示设备，默认值为"npu:0"。

## 返回值说明
`Tensor`
-   **metadata**（`Tensor`）：QuantLightningIndexerMetadata算子传入的分核信息，包括每个Cube核上FlashAttention计算任务的Batch、Head以及Q和K分块的索引，以及每个Vector核上FlashDecode的规约任务索引。数据类型支持`int32`，shape大小为[1024]。

## 约束说明
-   该接口支持图模式。

## 调用示例

-   单算子模式调用
    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math
    import custom_ops

    actual_seq_lengths_query = torch.tensor([8192], dtype=torch.int32).npu()
    actual_seq_lengths_key = torch.tensor([8192], dtype=torch.int32).npu()
    num_heads_q = 64
    num_heads_k = 1
    head_dim = 128
    query_quant_mode = 0
    key_quant_mode = 0
    batch_size = 1
    max_seqlen_q = 8192
    max_seqlen_k = 8192
    layout_query = "BSND"
    layout_key = "PA_BSND"
    sparse_count = 2048
    sparse_mode = 3
    pre_tokens = 2**63-1
    next_tokens = 2**63-1
    cmp_ratio = 4
    device = "npu:0"
    
    meta_data = torch.ops.custom.npu_quant_lightning_indexer_metadata (
                                    actual_seq_lengths_query = actual_seq_lengths_query.npu(),
                                    actual_seq_lengths_key = actual_seq_lengths_key.npu(),
                                    num_heads_q = num_heads_q,
                                    num_heads_k = num_heads_k,
                                    head_dim = head_dim,
                                    query_quant_mode = query_quant_mode, 
                                    key_quant_mode = key_quant_mode,
                                    batch_size = batch_size, 
                                    max_seqlen_q = max_seqlen_q,
                                    max_seqlen_k = max_seqlen_k,  
                                    layout_query = layout_query, 
                                    layout_key = layout_key,
                                    sparse_count = sparse_count, 
                                    sparse_mode = sparse_mode, 
                                    pre_tokens = pre_tokens, 
                                    next_tokens = next_tokens, 
                                    cmp_ratio = cmp_ratio,
                                    device = 'npu:0')
    
    ```
-   aclgraph调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math
    import torchair
    import custom_ops
    from torchair.configs.compiler_config import CompilerConfig

    actual_seq_lengths_query = torch.tensor([8192], dtype=torch.int32).npu()
    actual_seq_lengths_key = torch.tensor([8192], dtype=torch.int32).npu()
    num_heads_q = 64
    num_heads_k = 1
    head_dim = 128
    query_quant_mode = 0
    key_quant_mode = 0
    batch_size = 1
    max_seqlen_q = 8192
    max_seqlen_k = 8192
    layout_query = "BSND"
    layout_key = "PA_BSND"
    sparse_count = 2048
    sparse_mode = 3
    pre_tokens = 2**63-1
    next_tokens = 2**63-1
    cmp_ratio = 4
    device = "npu:0"
    
    class QLIMetadataNetwork(nn.Module):
        def __init__(self):
            super(QLIMetadataNetwork, self).__init__()

        def forward(self, num_heads_q, num_heads_k, head_dim, query_quant_mode, key_quant_mode,
                    actual_seq_lengths_query=None, actual_seq_lengths_key=None, batch_size=0, 
                    max_seqlen_q=0, max_seqlen_k=0, layout_query='BSND', layout_key='BSND',
                    sparse_count=2048, sparse_mode=3, pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1, 
                    cmp_ratio=4, device='npu:0'):
            meta_data = torch_npu.npu_quant_lightning_indexer_metadata(
                                    actual_seq_lengths_query = actual_seq_lengths_query,
                                    actual_seq_lengths_key = actual_seq_lengths_key,
                                    num_heads_q = num_heads_q,
                                    num_heads_k = num_heads_k,
                                    head_dim = head_dim,
                                    query_quant_mode = query_quant_mode, 
                                    key_quant_mode = key_quant_mode,
                                    batch_size = batch_size, 
                                    max_seqlen_q = max_seqlen_q,
                                    max_seqlen_k = max_seqlen_k,  
                                    layout_query = layout_query, 
                                    layout_key = layout_key,
                                    sparse_count = sparse_count, 
                                    sparse_mode = sparse_mode, 
                                    pre_tokens = (1<<63)-1, 
                                    next_tokens = (1<<63)-1, 
                                    cmp_ratio = cmp_ratio,
                                    device = 'npu:0')
            return meta_data
    npu_mode = QLIMetadataNetwork().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)
    meta_data = npu_mode(actual_seq_lengths_query = actual_seq_lengths_query.npu(),
                         actual_seq_lengths_key = actual_seq_lengths_key.npu(),
                         num_heads_q = num_heads_q,
                         num_heads_k = num_heads_k,
                         head_dim = head_dim,
                         query_quant_mode = query_quant_mode, 
                         key_quant_mode = key_quant_mode,
                         batch_size = batch_size, 
                         max_seqlen_q = max_seqlen_q,
                         max_seqlen_k = max_seqlen_k,  
                         layout_query = layout_query, 
                         layout_key = layout_key,
                         sparse_count = sparse_count, 
                         sparse_mode = sparse_mode, 
                         pre_tokens = pre_tokens, 
                         next_tokens = next_tokens, 
                         cmp_ratio = cmp_ratio,
                         device = 'npu:0')
    ```