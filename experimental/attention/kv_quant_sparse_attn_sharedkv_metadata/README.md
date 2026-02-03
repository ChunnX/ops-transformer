# KvQuantSparseAttnSharedkvMetadata

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 推理系列产品</term>   | √  |

## 功能说明
- API功能：`KvQuantSparseAttentionSharedKVMetadata`算子旨在生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引，供后续`KvQuantSparseAttentionSharedKV`算子使用。
- KvQuantSparseAttentionSharedKv计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的$KV$。

## 函数原型

```
npu_kv_quant_sparse_attn_sharedkv_metadata(num_heads_q, num_heads_kv, head_dim, kv_quant_mode, *, cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, 
seqused_q=None, seqused_kv=None, batch_size=0, max_seqlen_q=0, max_seqlen_kv=0, ori_topk=0, cmp_topk=0, tile_size=0, rope_head_dim=0, cmp_ratio=-1, ori_mask_mode=4, 
cmp_mask_mode=3, ori_win_left=127, ori_win_right=0, layout_q='BSND', layout_kv='PA_ND', has_ori_kv=True, has_cmp_kv=True, device='npu:0') -> Tensor
```

## 参数说明

> [!NOTE]
>- q、ori_kv、cmp_kv参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Hidden Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- Q_S和S1表示q shape中的S，S2表示ori_kv shape中的S，S3表示cmp_kv shape中的S；Q\_N和N1表示num\_q\_heads，KV\_N和N2表示num\_ori_kv\_heads和num\_cmp_kv\_heads；T1表示q shape中的T，T2表示ori\_kv shape中的T，T3表示cmp\_kv shape中的输入样本序列长度的累加和。
-   **num_heads_q**（`int`）：必选参数，表示`q`的多头数，目前仅支持64。

-   **num_heads_kv**（`int`）：必选参数，表示`ori_kv`的多头数，目前仅支持1。

-   **head_dim**（`int`）：必选参数，表示hidden层最小的单元尺寸，目前仅支持512。

-   **kv_quant_mode**（`int`）：必选参数，表示kv nope的量化模式，仅支持1，表示K、V nope为per-tile量化，量化后的KV数据类型为`float8_e4m3`。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

-   **cu\_seqlens\_q**（`Tensor`）：可选参数，当`layout_query`为TND时，表示不同Batch中`q`的有效token数，维度为B+1，大小为参数中每个元素的值表示目前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。

-   **cu\_seqlens\_ori\_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`ori_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示目前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。**目前layout_kv仅支持PA_ND，故设置此参数无效。**

-   **cu\_seqlens\_cmp\_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`cmp_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示目前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。**目前layout_kv仅支持PA_ND，故设置此参数无效。**

-   **seqused\_q**（`Tensor`）：可选参数，表示不同Batch中`q`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。**目前暂不支持指定该参数。**

-   **seqused\_kv**（`Tensor`）：可选参数，表示不同Batch中`ori_kv`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。

-   **batch\_size**（`int`）：可选参数，表示表示输入样本批量大小。

-   **max\_seqlen\_q**（`int`）：可选参数，表示表示所有batch中`q`的最大有效token数。

-   **max\_seqlen\_kv**（`int`）：可选参数，表示表示所有batch中`ori_kv`的最大有效token数。

-   **ori_topk**（`int`）：可选参数，表示通过QLI算法从`ori_kv`中筛选出的关键稀疏token的个数。**目前暂不支持指定该参数。**

-   **cmp_topk**（`int`）：可选参数，表示通过QLI算法从`cmp_kv`中筛选出的关键稀疏token的个数，目前仅支持512。

-   **tile\_size**（`int`）：可选参数，表示量化粒度，必须能被`rope_head_dim`整除，默认值为None，目前仅支持64。

-   **rope\_head\_dim**（`int`）：可选参数，表示rope的多头数，默认值为0，目前仅支持64。
    
-   **cmp\_ratio**（`int`）：可选参数，表示对`ori_kv`的压缩率，数据范围支持4/128，默认值为None。

-   **ori\_mask\_mode**（`int`）：可选参数，表示`q`和`ori_kv`计算的mask模式，目前仅支持输入默认值4，代表band模式的mask。

-   **cmp\_mask\_mode**（`int`）：可选参数，表示`q`和`cmp_kv`计算的mask模式，目前仅支持输入默认值3，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

-   **ori\_win\_left**（`int`）：可选参数，表示`q`和`ori_kv`计算中`q`对过去token计算的数量，目前仅支持默认值127。

-   **ori\_win\_right**（`int`）：可选参数，表示`q`和`ori_kv`计算中`q`对未来token计算的数量，目前仅支持默认值0。

-   **layout\_q**（`str`）：可选参数，表示输入`q`的数据排布格式，默认值为BSND，目前支持传入BSND和TND。

-   **layout\_kv**（`str`）：可选参数，表示输入`ori_kv`和`cmp_kv`的数据排布格式，目前仅支持传入默认值PA_ND（PageAttention）。

-   **has\_ori\_kv**（`bool`）：可选参数，表示是否传入`ori_kv`，默认值为true。

-   **has\_cmp\_kv**（`bool`）：可选参数，表示是否传入`cmp_kv`，默认值为true。

-   **device**（`str`）：可选参数，用于获取设备信息，当输入tensor均没有传入时，此字段必填。

## 返回值说明

-   **Metadata**（`Tensor`）：每个cube核上FlashAttention计算任务的Batch、Head、以及 Q 和 K 的分块的索引,以及每个vector核上FlashDecode的规约任务索引。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持aclgraph模式。
-   参数q中的D值仅支持512。

## 调用示例
- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import custom_ops
    import numpy as np
    import torch.nn as nn

    layout_q="TND"
    layout_kv="PA_ND"
    B = 1
    S1 = 1
    S2 = 8193
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    cmp_ratio = 4
    ori_mask_mode = 4
    cmp_mask_mode = 3
    ori_win_left = 127
    ori_win_right = 0
    kv_quant_mode = 1
    tile_size = 64
    rope_head_dim = 64
        
    cu_seqlens_q = torch.arange(0, (B + 1) * S1, step=S1).to(torch.int32).npu()
    seqused_kv = torch.tensor([S2]*B).to(torch.int32).npu()

    metadata = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        kv_quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        cmp_topk=K,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=True,
        device = "npu:0")
    ``` 

-   图模式调用
    ```python
    import torch
    import torch_npu
    import torchair
    import custom_ops
    import numpy as np
    import torch.nn as nn
    from torchair.configs.compiler_config import CompilerConfig

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, num_heads_q, num_heads_kv, head_dim, kv_quant_mode, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, 
                    seqused_q, seqused_kv, batch_size, max_seqlen_q, max_seqlen_kv, ori_topk, cmp_topk, tile_size, rope_head_dim, 
                    cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv, has_ori_kv, has_cmp_kv, device):

            metadata = torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata(
                num_heads_q = num_heads_q,
                num_heads_kv = num_heads_kv,
                head_dim = head_dim,
                kv_quant_mode = kv_quant_mode,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_ori_kv = cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv = cu_seqlens_cmp_kv,
                seqused_q = seqused_q,
                seqused_kv = seqused_kv,
                batch_size = batch_size,
                max_seqlen_q = max_seqlen_q,
                max_seqlen_kv = max_seqlen_kv,
                ori_topk = ori_topk,
                cmp_topk = cmp_topk,
                tile_size = tile_size,
                rope_head_dim = rope_head_dim,
                cmp_ratio = cmp_ratio,
                ori_mask_mode = ori_mask_mode,
                cmp_mask_mode = cmp_mask_mode,
                ori_win_left = ori_win_left,
                ori_win_right = ori_win_right,
                layout_q = layout_q,
                layout_kv = layout_kv,
                has_ori_kv = has_ori_kv,
                has_cmp_kv = has_cmp_kv,
                device=device
                )
            return metadata

    npu_mode = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)

    layout_q="TND"
    layout_kv="PA_ND"
    B = 4
    S1 = 1
    S2 = 8193
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    cmp_ratio = 4
    ori_mask_mode = 4
    cmp_mask_mode = 3
    ori_win_left = 127
    ori_win_right = 0
    kv_quant_mode = 1
    tile_size = 64
    rope_head_dim = 64
        
    cu_seqlens_q = torch.arange(0, (B + 1) * S1, step=S1).to(torch.int32).npu()
    seqused_kv = torch.tensor([S2]*B).to(torch.int32).npu()

    metadata = npu_mode(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        kv_quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        ori_topk = 0,
        cmp_topk=K,
        tile_size = 0,
        rope_head_dim = 0,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=True,
        device = "npu:0")
    ```