# sparse\_flash\_mla\_metadata<a name="ZH-CN_TOPIC_0000002309174912"></a>

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明<a name="zh-cn_topic_0000002168254826_section14441124184110"></a>

-   API功能：

    生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引，供后续[sparse\_flash\_mla](./sparse_flash_mla.md)算子使用。

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
cann_ops_transformer.sparse_flash_mla_metadata(num_heads_q, num_heads_kv, head_dim, *, cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None, batch_size=0, max_seqlen_q=0, max_seqlen_ori_kv=0, max_seqlen_cmp_kv=0, ori_topk=0, cmp_topk=0, cmp_ratio=1, ori_mask_mode=0, cmp_mask_mode=0, ori_win_left=-1, ori_win_right=-1, layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=True) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

-   **cu\_seqlens\_q** (`Tensor`)：可选参数，表示不同Batch中Query的有效Sequence Length，维度为\(B+1\)，仅layout\_q为TND场景需传入。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **cu\_seqlens\_ori\_kv** (`Tensor`)：可选参数，表示不同Batch中ori\_kv的有效Sequence Length，维度为\(B+1\)，仅layout\_kv为TND场景需传入。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **cu\_seqlens\_cmp\_kv** (`Tensor`)：可选参数，表示不同Batch中cmp\_kv的有效Sequence Length，维度为\(B+1\)，仅layout\_kv为TND场景需传入。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **seqused\_q** (`Tensor`)：可选参数，表示不同Batch中Query实际参与运算的Sequence Length，维度为\(B\)。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **seqused\_ori\_kv** (`Tensor`)：可选参数，表示不同Batch中ori\_kv实际参与运算的Sequence Length，维度为\(B\)。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **seqused\_cmp\_kv** (`Tensor`)：可选参数，表示不同Batch中cmp\_kv实际参与运算的Sequence Length，维度为\(B\)。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **cmp\_residual\_kv** (`Tensor`)：可选参数，表示不同Batch中cmp\_kv压缩后Sequence Length的余数，配合cmp\_ratio实现cmp\_kv部分的mask和负载计算。cmp\_mask\_mode=3且cmp\_ratio≠1时必须传入，维度为\(B\)。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **ori\_topk\_length** (`Tensor`)：可选参数，预留参数，当前不生效。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **cmp\_topk\_length** (`Tensor`)：可选参数，预留参数，当前不生效。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **num\_heads\_q** (`int`)：必选参数，表示Query的head个数，当前仅支持2/4/8/16/32/64/128。

-   **num\_heads\_kv** (`int`)：必选参数，表示Key和Value对应的多头数，当前仅支持1。

-   **head\_dim** (`int`)：必选参数，表示注意力头的维度，当前仅支持512。

-   **batch\_size** (`int`)：可选参数，表示Batch数量，默认值为0。

-   **max\_seqlen\_q** (`int`)：可选参数，表示Query的最长Sequence Length，默认值为0。

-   **max\_seqlen\_ori\_kv** (`int`)：可选参数，表示ori\_kv的最长Sequence Length，默认值为0。

-   **max\_seqlen\_cmp\_kv** (`int`)：可选参数，表示cmp\_kv的最长Sequence Length，默认值为0。

-   **ori\_topk** (`int`)：可选参数，预留参数，当前不生效，表示ori\_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景，默认值为0。

-   **cmp\_topk** (`int`)：可选参数，表示cmp\_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景，默认值为0，当前仅支持512/1024。

-   **cmp\_ratio** (`int`)：可选参数，表示对cmp\_kv的压缩率，取值范围\[1, 128\]，默认值为1，当前仅支持4/128。

-   **ori\_mask\_mode** (`int`)：可选参数，表示q和ori\_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，4表示sliding window模式，默认值为0，当前仅支持4。

-   **cmp\_mask\_mode** (`int`)：可选参数，表示q和cmp\_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，默认值为0，当前仅支持3。

-   **ori\_win\_left** (`int`)：可选参数，表示q和ori\_kv计算中q对过去token计算的数量，-1表示无穷大，默认值为-1，当前仅支持127。

-   **ori\_win\_right** (`int`)：可选参数，表示q和ori\_kv计算中q对未来token计算的数量，-1表示无穷大，默认值为-1，当前仅支持0。

-   **layout\_q** (`str`)：可选参数，表示Query的排列格式，支持"BSND"、"TND"，默认值为"BSND"。

-   **layout\_kv** (`str`)：可选参数，表示Key的排列格式，支持"BSND"、"TND"、"PA\_BBND"，默认值为"BSND"。

-   **has\_ori\_kv** (`bool`)：可选参数，用于标识是否含有ori\_kv，默认值为True。

-   **has\_cmp\_kv** (`bool`)：可选参数，用于标识是否含有cmp\_kv，默认值为True。

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>

**metadata** (`Tensor`)：每个cube核上FlashAttention计算任务的Batch、Head、以及 Q 和 K 的分块的索引，以及每个vector核上FlashDecode的规约任务索引。shape固定为\[1024\]，数据类型为`int32`，数据格式为$ND$，不支持非连续的Tensor。

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持aclgraph模式。
-   该接口需与[sparse\_flash\_mla](./sparse_flash_mla.md)算子配套使用。
-   B（Batch）表示输入样本批量大小。
-   参数cu\_seqlens\_q、cu\_seqlens\_ori\_kv及cu\_seqlens\_cmp\_kv要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
-   参数seqused\_q、seqused\_ori\_kv、seqused\_cmp\_kv要求其值表示每个Batch中的有效token数。
-   参数cmp\_residual\_kv需满足cmp\_residual\_kv\[i\] < cmp\_ratio。
-   ori\_mask\_mode及cmp\_mask\_mode所表示的mask模式的详细介绍见[sparse\_mode参数说明](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/sparse_mode参数说明.md)。

## 调用示例<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_transformer
    import numpy as np
    import torch.nn as nn

    metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        cu_seqlens_q = torch.tensor([0, 10], dtype=torch.int32).npu(),
        cu_seqlens_ori_kv = None,
        cu_seqlens_cmp_kv = None,
        seqused_q = None,
        seqused_ori_kv = torch.tensor([8192], dtype=torch.int32).npu(),
        seqused_cmp_kv = torch.tensor([64], dtype=torch.int32).npu(),
        cmp_residual_kv = torch.tensor([1], dtype=torch.int32).npu(),
        ori_topk_length = None,
        cmp_topk_length = None,
        num_heads_q = 128,
        num_heads_kv = 1,
        head_dim = 512,
        batch_size = 1,
        max_seqlen_q = 1,
        max_seqlen_ori_kv = 512,
        max_seqlen_cmp_kv = 32,
        ori_topk = 0,
        cmp_topk = 512,
        cmp_ratio = 4,
        ori_mask_mode = 4,
        cmp_mask_mode = 3,
        ori_win_left = 127,
        ori_win_right = 0,
        layout_q = "TND",
        layout_kv = "PA_BBND",
        has_ori_kv = True,
        has_cmp_kv = True
    )
    ```

-   aclgraph模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_transformer
    import numpy as np
    import torch.nn as nn
    from torchair.configs.compiler_config import CompilerConfig

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv,
                    cmp_residual_kv, ori_topk_length, cmp_topk_length, num_heads_q, num_heads_kv, head_dim, batch_size,
                    max_seqlen_q, max_seqlen_ori_kv, max_seqlen_cmp_kv, ori_topk, cmp_topk, cmp_ratio, ori_mask_mode,
                    cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv, has_ori_kv, has_cmp_kv):
            metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_ori_kv = cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv = cu_seqlens_cmp_kv,
                seqused_q = None,
                seqused_ori_kv = seqused_ori_kv,
                seqused_cmp_kv = seqused_cmp_kv,
                cmp_residual_kv = cmp_residual_kv,
                ori_topk_length = ori_topk_length,
                cmp_topk_length = cmp_topk_length,
                num_heads_q = num_heads_q,
                num_heads_kv = num_heads_kv,
                head_dim = head_dim,
                batch_size = batch_size,
                max_seqlen_q = max_seqlen_q,
                max_seqlen_ori_kv = max_seqlen_ori_kv,
                max_seqlen_cmp_kv = max_seqlen_cmp_kv,
                ori_topk = ori_topk,
                cmp_topk = cmp_topk,
                cmp_ratio = cmp_ratio,
                ori_mask_mode = ori_mask_mode,
                cmp_mask_mode = cmp_mask_mode,
                ori_win_left = ori_win_left,
                ori_win_right = ori_win_right,
                layout_q = layout_q,
                layout_kv = layout_kv,
                has_ori_kv = has_ori_kv,
                has_cmp_kv = has_cmp_kv
            )
            return metadata

    npu_mode = Network().to("npu:%s" % 0)
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)

    metadata = npu_mode(
        cu_seqlens_q = None,
        cu_seqlens_ori_kv = None,
        cu_seqlens_cmp_kv = None,
        seqused_q = None,
        seqused_ori_kv = torch.tensor([8192], dtype=torch.int32).npu(),
        seqused_cmp_kv = None,
        cmp_residual_kv = torch.tensor([10], dtype=torch.int32).npu(),
        ori_topk_length = None,
        cmp_topk_length = None,
        num_heads_q = 64,
        num_heads_kv = 1,
        head_dim = 512,
        batch_size = 1,
        max_seqlen_q = 10,
        max_seqlen_ori_kv = 8192,
        max_seqlen_cmp_kv = 8192,
        ori_topk = 0,
        cmp_topk = 0,
        cmp_ratio = 128,
        ori_mask_mode = 4,
        cmp_mask_mode = 3,
        ori_win_left = 127,
        ori_win_right = 0,
        layout_q = "BSND",
        layout_kv = "BSND",
        has_ori_kv = True,
        has_cmp_kv = True
    )
    ```