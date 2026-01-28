# KvQuantSparseAttnSharedkv

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 推理系列产品</term>   | √  |

## 功能说明
- API功能：KVQuantSparseAttentionSharedKV 算子旨在完成以下形式的Attention计算，支持Sliding Window Attention、Compressed Flash Attention及两者混合：

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的$KV$。

## 函数原型

```
torch_npu.npu_kv_quant_sparse_attn_sharedkv(q, *, ori_kv=None, cmp_kv=None, ori_sparse_indices=None, cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None, 
cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, seqused_q=None, seqused_kv=None, sinks=None, metadata=None, kv_quant_mode=0, tile_size=0, rope_head_dim=0, 
softmax_scale=0, cmp_ratio=0, ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=128, ori_win_right=0, layout_q='BSND', layout_kv='PA_ND', return_softmax_lse=False) -> (Tensor, Tensor)
```

## 参数说明

> [!NOTE]  
>- q、ori_kv、cmp_kv参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Hidden Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- Q_S和S1表示q shape中的S，S2表示ori_kv shape中的S，S3表示cmp_kv shape中的S；Q\_N和N1表示num\_q\_heads，KV\_N和N2表示num\_ori_kv\_heads和num\_cmp_kv\_heads；T1表示q shape中的T，T2表示ori\_kv shape中的T，T3表示cmp\_kv shape中的输入样本序列长度的累加和。
-   **q**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为BSND时shape为[B,S1,N1,D]，当`layout_query`为TND时shape为[T1,N1,D]，其中N1支持1~128。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

-   **ori\_kv**（`Tensor`）：可选参数，对应公式中的$\tilde{K}$和$\tilde{V}$的一部分，为原始不经压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`float8_e4m3fn`，`layout_kv`为PA_ND时shape为[block_num1, block_size1, KV_N, D]，其中block_num1为PageAttention时block总数，bloc_size1为一个block的token数，block_size1取值为16的倍数，最大支持1024。`layout_kv`为BSND时shape为[B, S2, KV_N, D]。`layout_kv`为TND时shape为[T2, KV_N, D]。其中KV_N只支持1。

-   **cmp\_kv**（`Tensor`）：可选参数，对应公式中的$\tilde{K}$和$\tilde{V}$的一部分，为经过压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`float8_e4m3fn`，`layout_kv`为PA_ND时shape为[block_num2, block_size, KV_N, D]，其中block_num2为PageAttention时block总数，block_size2为一个block的token数，block_size2取值为16的倍数，最大支持1024。`layout_kv`为BSND时shape为[B, S3, KV_N, D]。`layout_kv`为TND时shape为[T3, KV_N, D]。其中KV_N只支持1。

-   **ori\_sparse\_indices**（`Tensor`）：可选参数，代表离散取oriKvCache的索引，不支持非连续，数据格式支持ND,数据类型支持`int32`。当`layout_query`为BSND时，shape需要传入[B, Q_S, KV_N, K1]，当`layout_query`为TND时，shape需要传入[Q_T, KV_N, K1]，其中K1为对`ori_kv`一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足K1为512~2K。

-   **cmp\_sparse\_indices**（`Tensor`）：可选参数，代表离散取cmpKvCache的索引，不支持非连续，数据格式支持ND,数据类型支持`int32`。当`layout_query`为BSND时，shape需要传入[B, Q_S, KV_N, K2]，当`layout_query`为TND时，shape需要传入[Q_T, KV_N, K2]，其中K2为对`cmp_kv`一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足K2为512~2K。

-   **ori\_block\_table**（`Tensor`）：可选参数，表示PageAttention中oriKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2_max / block_size向上取整。

-   **cmp\_block\_table**（`Tensor`）：可选参数，表示PageAttention中cmpKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S3对应的block数量，即S3_max / block_size向上取整。

-   **cu\_seqlens\_q**（`Tensor`）：可选参数，当`layout_query`为TND时，表示不同Batch中`q`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。

-   **cu\_seqlens\_ori\_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`ori_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。

-   **cu\_seqlens\_cmp\_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`cmp_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。

-   **seqused\_q**（`Tensor`）：可选参数，表示不同Batch中`q`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。

-   **seqused\_kv**（`Tensor`）：可选参数，表示不同Batch中`ori_kv`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。

-   **sinks**（`Tensor`）：可选参数，注意力下沉tensor，数据格式支持ND，数据类型支持`float32`，shape为[N1]。

-   **metadata**（`Tensor`）：可选参数，为aicpu算子（npu_sparse_attn_sharedkv_metadata）的分核结果，数据格式支持ND，数据类型支持`int32`，shape固定为[1024]。

-   **kv\_quant\_mode**（`Tensor`）：可选参数，kv nope的量化模式，仅支持1，表示K、V nope为per-tile量化为float8_e4m3。

-   **tile\_size**（`Tensor`）：可选参数，表示量化粒度，必须能被rope_head_dim整除，默认值为None，当前仅支持64。

-   **rope\_head\_dim**（`Tensor`）：可选参数，为aicpu算子（npu_kv_quant_sparse_attn_sharedkv_metadata）的分核结果，数据格式支持ND，数据类型支持`int32`，默认值为0，当前仅支持64。

-   **softmax\_scale**（`float`）：可选参数，代表缩放系数，作为q与ori_kv和cmp_kv矩阵乘后Muls的scalar值，数据类型支持`float`，默认值为None，None表示softmax_scale值为1/sqrt(D)。
    
-   **cmp\_ratio**（`int`）：可选参数，表示对ori_kv的压缩率，数据类型支持`int`，数据范围支持2/4/8/16/32/64/128，默认值为None。

-   **ori\_mask\_mode**（`int`）：可选参数，表示q和ori_kv计算的mask模式，仅支持输入默认值4，代表band模式的mask，数据类型支持`int`。

-   **cmp\_mask\_mode**（`int`）：可选参数，表示q和cmp_kv计算的mask模式，仅支持输入默认值3，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，数据类型支持`int`。

-   **ori\_win\_left**（`int`）：可选参数，表示q和ori_kv计算中q对过去token计算的数量，数据类型支持`int`，数据范围支持ori_win_left>=0，或ori_win_left=-1，-1表示没有限制，默认值为127。

-   **ori\_win\_right**（`int`）：可选参数，表示q和ori_kv计算中q对未来token计算的数量，数据类型支持`int`，仅支持默认值0。

-   **layout\_q**（`str`）：可选参数，用于标识输入q的数据排布格式，默认值为BSND，支持传入BSND和TND。

-   **layout\_kv**（`str`）：可选参数，用于标识输入`ori_kv`和`cmp_kv`的数据排布格式，默认值为PA_ND，支持传入TND、BSND和PA_ND，其中PA_ND在使能PageAttention时使用。

-   **return\_softmax\_lse**（`bool`）：可选参数，表示是否输出softmax_lse。True表示返回，False表示不返回；默认值为False。

## 返回值说明

-   **attention\_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`和。当layout_q为BSND时shape为[B,S1,N1,D]，当layout_q为TND时shape为[T1,N1,D]。
-   **softmax\_lse**（`Tensor`）：可选输出，输出q乘ori_kv的结果先取max得到softmax_max，query乘key的结果减去softmax_max，再取exp，最后取sum，得到softmax_sum，最后对softmax_sum取log，再加上softmax_max得到的结果。数据类型支持`float`。当layout_q为BSND时shape为[B,N2,S1,N1/N2]，当layout_q为TND时shape为[N2,T1,N1/N2]。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   参数q中的D和ori_kv、cmp_kv的D值仅支持512。
-   参数ori_kv、cmp_kv的数据类型必须保持一致。

## 调用示例
- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math
    import custom_ops

    layout_q="TND"
    layout_kv="PA_ND"
    q_type=torch.bfloat16
    ori_kv_type=torch.float8_e4m3fn
    cmp_kv_type=torch.float8_e4m3fn
    B = 1
    S1 = 1
    T1 = 1
    S2 = 8193
    actS2 = 8193
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    block_size1 = 128
    block_size2 = 128
    softmax_scale = 0.04419417
    cmp_ratio = 4
    ori_mask_mode = 4
    cmp_mask_mode = 3
    ori_win_left = 127
    ori_win_right = 0
    kv_quant_mode = 1
    tile_size = 64
    rope_head_dim = 64

    q = torch.tensor(np.random.uniform(-10, 10, (B*S1, N1, D))).to(q_type).npu()
        
    cu_seqlens_q = torch.arange(0, (B + 1) * S1, step=S1).to(torch.int32).npu()
    seqused_kv = torch.tensor([S2]*B).to(torch.int32).npu()

    cmp_kv_len = actS2 // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - S1 + 1),  K)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(B * S1 * N2)]).reshape(B, S1, N2, K).to(torch.int32).npu()
        
    block_num1 =  math.ceil(actS2/block_size1) * B
    block_table1 = torch.tensor(np.random.permutation(range(block_num1))).to(torch.int32).reshape(B, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (block_num1, block_size1, N2, D))).to(ori_kv_type).npu()

    block_num2 =  math.ceil(cmp_kv_len/block_size2) * B
    block_table2 = torch.tensor(np.random.permutation(range(block_num2))).to(torch.int32).reshape(B, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (block_num2, block_size2, N2, D))).to(cmp_kv_type).npu()
    sinks = torch.rand(N1).to(torch.float32).npu()

    metadata = torch.zeros((2048), dtype=torch.int32)
    metadata[:3] = torch.tensor([1, 64, 128], dtype=torch.int32)  # 3个数：usedCoreNum, mBaseSize, s2BaseSize
    metadata[3:35] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # 32个数 bN2End
    metadata[35:67] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # 32个数 mEnd
    metadata[67:99] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # 32个数 s2End
    metadata = metadata.npu()

    attn_out = torch_npu.npu_kv_quant_sparse_attn_sharedkv(
                                                q=q,
                                                ori_kv=ori_kv,
                                                cmp_kv=cmp_kv,
                                                cmp_sparse_indices=cmp_sparse_indices,
                                                ori_block_table=block_table1,
                                                cmp_block_table=block_table2,
                                                cu_seqlens_q=cu_seqlens_q,
                                                seqused_kv=seqused_kv,
                                                sinks=sinks,
                                                metadata=metadata,
                                                kv_quant_mode=kv_quant_mode,
                                                tile_size=tile_size,
                                                rope_head_dim=rope_head_dim,
                                                softmax_scale=softmax_scale,
                                                cmp_ratio=cmp_ratio,
                                                ori_mask_mode=ori_mask_mode,
                                                cmp_mask_mode=cmp_mask_mode,
                                                ori_win_left=ori_win_left,
                                                ori_win_right=ori_win_right,
                                                layout_q=layout_q,
                                                layout_kv=layout_kv)
    ``` 

-   图模式调用
    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math
    import custom_ops
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, q, ori_kv, cmp_kv, cmp_sparse_indices, ori_block_table, 
            cmp_block_table, cu_seqlens_q, seqused_kv, sinks, metadata, kv_quant_mode, tile_size, rope_head_dim, 
            softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv):
            return torch_npu.npu_kv_quant_sparse_attn_sharedkv(
                                                q=q,
                                                ori_kv=ori_kv,
                                                cmp_kv=cmp_kv,
                                                cmp_sparse_indices=cmp_sparse_indices,
                                                ori_block_table=ori_block_table,
                                                cmp_block_table=cmp_block_table,
                                                cu_seqlens_q=cu_seqlens_q,
                                                seqused_kv=seqused_kv,
                                                sinks=sinks,
                                                metadata=metadata,
                                                kv_quant_mode=kv_quant_mode,
                                                tile_size=tile_size,
                                                rope_head_dim=rope_head_dim,
                                                softmax_scale=softmax_scale,
                                                cmp_ratio=cmp_ratio,
                                                ori_mask_mode=ori_mask_mode,
                                                cmp_mask_mode=cmp_mask_mode,
                                                ori_win_left=ori_win_left,
                                                ori_win_right=ori_win_right,
                                                layout_q=layout_q,
                                                layout_kv=layout_kv)

    layout_q="TND"
    layout_kv="PA_ND"
    q_type=torch.bfloat16
    ori_kv_type=torch.float8_e4m3fn
    cmp_kv_type=torch.float8_e4m3fn
    B = 1
    S1 = 1
    T1 = 1
    S2 = 8193
    actS2 = 8193
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    block_size1 = 128
    block_size2 = 128
    softmax_scale = 0.04419417
    cmp_ratio = 4
    ori_mask_mode = 4
    cmp_mask_mode = 3
    ori_win_left = 127
    ori_win_right = 0
    kv_quant_mode = 1
    tile_size = 64
    rope_head_dim = 64

    q = torch.tensor(np.random.uniform(-10, 10, (B*S1, N1, D))).to(q_type).npu()
        
    cu_seqlens_q = torch.arange(0, (B + 1) * S1, step=S1).to(torch.int32).npu()
    seqused_kv = torch.tensor([S2]*B).to(torch.int32).npu()

    cmp_kv_len = actS2 // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - S1 + 1),  K)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(B * S1 * N2)]).reshape(B, S1, N2, K).to(torch.int32).npu()
        
    block_num1 =  math.ceil(actS2/block_size1) * B
    block_table1 = torch.tensor(np.random.permutation(range(block_num1))).to(torch.int32).reshape(B, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (block_num1, block_size1, N2, D))).to(ori_kv_type).npu()

    block_num2 =  math.ceil(cmp_kv_len/block_size2) * B
    block_table2 = torch.tensor(np.random.permutation(range(block_num2))).to(torch.int32).reshape(B, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (block_num2, block_size2, N2, D))).to(cmp_kv_type).npu()
    sinks = torch.rand(N1).to(torch.float32).npu()

    metadata = torch.zeros((2048), dtype=torch.int32)
    metadata[:3] = torch.tensor([1, 64, 128], dtype=torch.int32)  # 3个数：usedCoreNum, mBaseSize, s2BaseSize
    metadata[3:35] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # 32个数 bN2End
    metadata[35:67] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # 32个数 mEnd
    metadata[67:99] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # 32个数 s2End
    metadata = metadata.npu()

    npu_mode = Network().npu()
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    config.mode = "reduce-overhead"
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)

    attn_out = npu_mode(q,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=block_table1,
                cmp_block_table=block_table2,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                sinks=sinks,
                metadata=metadata,
                kv_quant_mode=kv_quant_mode,
                tile_size=tile_size,
                rope_head_dim=rope_head_dim,
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv)
    print(attn_out)
    ```