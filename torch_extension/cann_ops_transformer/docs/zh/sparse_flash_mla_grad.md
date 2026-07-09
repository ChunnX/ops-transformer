# sparse_flash_mla_grad / sparse_flash_mla_grad_metadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- API功能：

    * `sparse_flash_mla_grad`：计算`SparseFlashMla`训练场景下注意力的反向输出，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。
    * `sparse_flash_mla_grad_metadata`：用于在`sparse_flash_mla_grad`主算子前生成metadata，此接口为后续扩展预留，暂时无实际作用，输出为None。

- 计算公式：

    阶段一：根据不同cmp_ratio场景，对输入ori_kv与cmp_kv进行选择

    * 当cmp_ratio = 1 (SWA)：

    $$
    selectedKv\text{ }=\text{ }orikv
    $$

    * 当cmp_ratio = 4 (SCFA)：

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }Gather \left( cmpkv,topkIndices \left[ i \left]  \left)) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

    * else (CFA):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }cmpkv)
    $$

    阶段二：计算P、dP、dS

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), lse)
    $$

    $$
    dP = dO \text{ }@\text{ } selectedKv^{{T}}
    $$

    $$
    dS = P \times (dP\text{ } -\text{ } SoftmaxGrad(dO, O))
    $$

    阶段三：计算dQ, dKV, dSinks

    $$
    dQ = dS \text{ } @ \text{ } selectedKv \text{ }  \cdot \text{ } scale
    $$

    $$
    dKV = dS^{{T}} \text{ } @ \text{ } Q \text{ } \cdot \text{ } scale + P^{{T}}  @ \text{ } dO
    $$

    $$
    dSinks = ReduceSum(-P \text{ }\times\text{ } dP \text{ }\times\text{ } SimpleSoftmax(sinks, lse), dim=-1)
    $$

## 函数原型

```python
from cann_ops_transformer.ops import sparse_flash_mla_grad_metadata
sparse_flash_mla_grad_metadata(num_heads_q, num_heads_kv, head_dim,
                            cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
                            seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
                            cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None,
                            batch_size=None, max_seqlen_q=None, max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None,
                            ori_topk=None, cmp_topk=None, cmp_ratio=None, ori_mask_mode=0, cmp_mask_mode=0,
                            ori_win_left=-1, ori_win_right=-1,
                            layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=True) -> Tensor
```
```python
from cann_ops_transformer.ops import sparse_flash_mla_grad
sparse_flash_mla_grad(q, dout, attn_out, softmax_lse, ori_kv=None, cmp_kv=None,
                        ori_sparse_indices=None, cmp_sparse_indices=None, cu_seqlens_q=None,
                        cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, seqused_q=None,
                        seqused_ori_kv=None, seqused_cmp_kv=None, cmp_residual_kv=None, 
                        ori_topk_length=None, cmp_topk_length=None, 
                        sinks=None, metadata=None,
                        softmax_scale=None, cmp_ratio=None, ori_mask_mode=0, cmp_mask_mode=0,
                        ori_win_left=-1, ori_win_right=-1,
                        layout_q="BSND", layout_kv="BSND") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

### sparse_flash_mla_grad

- **q**（`Tensor`）：必选参数，对应公式中的$Q$，支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`。`layout_q`为BSND时shape为[B,S1,N1,D]，当`layout_q`为TND时shape为[T1,N1,D]，B：支持泛化；S1：支持泛化；N1：支持1~128；D：512；T1：B × S1。

- **dout**（`Tensor`）：必选参数，注意力正向输出矩阵的梯度，对应公式中的$dO$，支持非连续，数据格式支持ND，数据类型、shape均与q保持一致。

- **attn_out**（`Tensor`）：必选参数，注意力正向输出矩阵，对应公式中的$O$，支持非连续，数据格式支持ND，数据类型、shape均与q保持一致。

- **softmax_lse**（`Tensor`）：必选参数，注意力正向计算的输出lse，支持非连续，数据格式支持ND，数据类型支持`float32`。`layout_q`为BSND时shape为[B,N2,S1,G]，当`layout_q`为TND时shape为[N2,T1,G]，B：与query的B保持一致；N2：1；S1：与query的S1保持一致；G：N1/N2；T1：B × S1。

- **ori_kv**（`Tensor`）：可选参数，对应公式中的$oriKv$，支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`，`layout_kv`为BSND时shape为[B,S2,N2,D]，当`layout_kv`为TND时shape为[T2,N2,D]，B：与query的B保持一致；S2：支持泛化；N2：1；D：512；T2：B × S2；当前不支持传None。

- **cmp_kv**（`Tensor`）：可选参数，对应公式中的$cmpkv$，支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`，`layout_kv`为BSND时shape为[B,S3,N2,D]，当`layout_kv`为TND时shape为[T3,N2,D]，B：与query的B保持一致；S3：支持泛化；N2：1；D：512；T3：B × S3；传None时按照公式中的SWA场景计算。

- **ori_sparse_indices**（`Tensor`）：可选参数，对应oriKv部分的topk（暂不支持），支持非连续，数据格式支持ND，数据类型支持`int32`，`layout_q`为BSND时shape为[B,S1,N2,K1]，当`layout_q`为TND时shape为[T1,N2,K1]，B：与query的B保持一致；S1：与query的S1保持一致；N2：1；K1：支持泛化；T1：B × S1；当前仅支持传None。

- **cmp_sparse_indices**（`Tensor`）：可选参数，对应公式中的$topkIndices$，支持非连续，数据格式支持ND，数据类型支持`int32`，`layout_q`为BSND时shape为[B,S1,N2,K2]，当`layout_q`为TND时shape为[T1,N2,K2]，B：与query的B保持一致；S1：与query的S1保持一致；N2：1；K2：支持泛化；T1：B × S1；若cmp_kv不为None，此时cmp_sparse_indices不为None时按照SCFA场景计算，为None时按照CFA场景计算；若cmp_kv为None，则cmp_sparse_indices只能为None，此时按SWA场景计算。

- **cu_seqlens_q**（`Tensor`）：可选参数，代表每个Batch中，q的有效token数的累加和形式，当`layout_q`为TND时该参数必传，支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[B+1]，累加和与T1保持一致。

- **cu_seqlens_ori_kv**（`Tensor`）：可选参数，代表每个Batch中，ori_kv的有效token数的累加和形式，当`layout_kv`为TND时该参数必传，支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[B+1]，累加和与T2保持一致。

- **cu_seqlens_cmp_kv**（`Tensor`）：可选参数，代表每个Batch中，cmp_kv的有效token数的累加和形式，当`layout_kv`为TND时该参数必传，支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[B+1]，累加和与T3保持一致。

- **seqused_q**（`Tensor`）：可选参数，表示不同batch中query实际参与运算的token数，当前仅支持传None。

- **seqused_ori_kv**（`Tensor`）：可选参数，表示不同batch中ori_kv实际参与运算的token数，当前仅支持传None。

- **seqused_cmp_kv**（`Tensor`）：可选参数，表示不同batch中cmp_kv实际参与运算的token数，当前仅支持传None。

- **cmp_residual_kv**（`Tensor`）：可选参数，表示每个batch 实际ori_s2 // cmpRatio后的余数，数据类型支持`int32`，支持非连续，数据格式支持ND，shape为[B]，当cmp_kv不为空且cmp_mask_mode=3时必须传入。

- **ori_topk_length**（`Tensor`）：可选参数，表示每行query对应的ori_kv实际可选的topk长度，当前仅支持传None。

- **cmp_topk_length**（`Tensor`）：可选参数，表示每行query对应的cmp_kv实际可选的topk长度，当前仅支持传None。

- **sinks**（`Tensor`）：可选参数，表示注意力下沉tensor，数据类型支持`float32`，支持非连续，数据格式支持ND，shape为[N1]，当前不支持传None。

- **metadata**（`Tensor`）：可选参数，表示tiling下沉的aicpu算子输出结果，当前仅支持传None。

- **softmax_scale**（`double`）：可选参数，代表缩放系数，数据类型支持`double`，默认值：1.0 / sqrt(D)。

- **cmp_ratio**（`int`）：可选参数，代表压缩率，数据类型支持`int`，取值范围：1~128，默认值：1。

- **ori_mask_mode**（`int`）：可选参数，表示q和ori_kv计算的mask模式，数据类型支持`int`，当前仅支持模式4（band模式的mask，滑窗范围由ori_win_left、ori_win_right控制，起点为右下角）。

- **cmp_mask_mode**（`int`）：可选参数，表示q和cmp_kv计算的mask模式，数据类型支持`int`，当前仅支持模式3（rightDownCausal模式的mask，对应以右顶点为划分的下三角场景）。

- **ori_win_left**（`int`）：可选参数，表示q和ori_kv计算中q对过去token计算的数量，数据类型支持`int`，当前仅支持取值127。

- **ori_win_right**（`int`）：可选参数，表示q和ori_kv计算中q对未来token计算的数量，数据类型支持`int`，当前仅支持取值0。

- **layout_q**（`str`）：可选参数，表示q的数据排布格式，支持"BSND"、"TND"。

- **layout_kv**（`str`）：可选参数，表示ori_kv、cmp_kv的数据排布格式，支持"BSND"、"TND"，当前必须与layout_q保持一致。

### sparse_flash_mla_grad_metadata

- **num_heads_q**（`int`）：必选参数，表示公式中$Q$的头数（即N1），数据类型支持`int`。

- **num_heads_kv**（`int`）：必选参数，表示公式中$oriKv$或$cmpkv$的头数（即N2），数据类型支持`int`。

- **head_dim**（`int`）：必选参数，表示头的维度（即D），数据类型支持`int`。

- **cu_seqlens_q**（`Tensor`）：可选参数，代表每个Batch中，q的有效token数的累加和形式，当`layout_q`为TND时该参数必传，支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[B+1]，累加和与T1保持一致。

- **cu_seqlens_ori_kv**（`Tensor`）：可选参数，代表每个Batch中，ori_kv的有效token数的累加和形式，当`layout_kv`为TND时该参数必传，支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[B+1]，累加和与T2保持一致。

- **cu_seqlens_cmp_kv**（`Tensor`）：可选参数，代表每个Batch中，cmp_kv的有效token数的累加和形式，当`layout_kv`为TND时该参数必传，支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[B+1]，累加和与T3保持一致。

- **seqused_q**（`Tensor`）：可选参数，表示不同batch中query实际参与运算的token数，当前仅支持传None。

- **seqused_ori_kv**（`Tensor`）：可选参数，表示不同batch中ori_kv实际参与运算的token数，当前仅支持传None。

- **seqused_cmp_kv**（`Tensor`）：可选参数，表示不同batch中cmp_kv实际参与运算的token数，当前仅支持传None。

- **cmp_residual_kv**（`Tensor`）：可选参数，表示每个batch 实际ori_s2 // cmpRatio后的余数，数据类型支持`int32`，支持非连续，数据格式支持ND，shape为[B]，当cmp_kv不为空且cmp_mask_mode=3时必须传入。

- **ori_topk_length**（`Tensor`）：可选参数，表示每行query对应的ori_kv实际可选的topk长度，当前仅支持传None。

- **cmp_topk_length**（`Tensor`）：可选参数，表示每行query对应的cmp_kv实际可选的topk长度，当前仅支持传None。

- **batch_size**（`int`）：可选参数，表示输入样本批量大小（即B），数据类型支持`int`。

- **max_seqlen_q**（`int`）：可选参数，表示TND场景下输入q的最大序列长度，数据类型支持`int`。

- **max_seqlen_ori_kv**（`int`）：可选参数，表示TND场景下输入ori_kv的最大序列长度，数据类型支持`int`。

- **max_seqlen_cmp_kv**（`int`）：可选参数，表示TND场景下输入cmp_kv的最大序列长度，数据类型支持`int`。

- **ori_topk**（`int`）：可选参数，表示ori_kv的topk长度，数据类型支持`int`。

- **cmp_topk**（`int`）：可选参数，表示cmp_kv的topk长度，数据类型支持`int`。

- **cmp_ratio**（`int`）：可选参数，代表压缩率，数据类型支持`int`，取值范围：1~128，默认值：1。

- **ori_mask_mode**（`int`）：可选参数，表示q和ori_kv计算的mask模式，数据类型支持`int`，当前仅支持模式4（band模式的mask，滑窗范围由ori_win_left、ori_win_right控制，起点为右下角）。

- **cmp_mask_mode**（`int`）：可选参数，表示q和cmp_kv计算的mask模式，数据类型支持`int`，当前仅支持模式3（rightDownCausal模式的mask，对应以右顶点为划分的下三角场景）。

- **ori_win_left**（`int`）：可选参数，表示q和ori_kv计算中q对过去token计算的数量，数据类型支持`int`，当前仅支持取值127。

- **ori_win_right**（`int`）：可选参数，表示q和ori_kv计算中q对未来token计算的数量，数据类型支持`int`，当前仅支持取值0。

- **layout_q**（`str`）：可选参数，表示q的数据排布格式，支持"BSND"、"TND"。

- **layout_kv**（`str`）：可选参数，表示ori_kv、cmp_kv的数据排布格式，支持"BSND"、"TND"，当前必须与layout_q保持一致。

- **has_ori_kv**（`bool`）：可选参数，表示是否传入ori_kv。

- **has_cmp_kv**（`bool`）：可选参数，表示是否传入cmp_kv。

## 返回值说明

### sparse_flash_mla_grad

- **dq**（`Tensor`）：对应公式中的$dQ$，支持非连续，数据格式支持ND，数据类型、shape与输入q保持一致。

- **dori_kv**（`Tensor`）：可选输出，表示输入ori_kv的梯度，支持非连续，数据格式支持ND，数据类型、shape与输入ori_kv保持一致。

- **dcmp_kv**（`Tensor`）：可选输出，表示输入cmp_kv的梯度，支持非连续，数据格式支持ND，数据类型、shape与输入cmp_kv保持一致；当cmp_kv为None时，dcmp_kv也为None。

- **dsinks**（`Tensor`）：可选输出，表示输入sinks的梯度，支持非连续，数据格式支持ND，数据类型支持`float32`，shape与输入sinks保持一致。

- **ori_softmax_l1norm**（`Tensor`）：当前该参数不支持，输出为None。

- **cmp_softmax_l1norm**（`Tensor`）：可选输出，表示q与cmp_kv计算得出的softmax的L1Norm结果，公式为reduceG(softmax)/G；在SCFA场景下该输出不为空，其他场景下输出为None。

### sparse_flash_mla_grad_metadata

- **metadata**（`Tensor`）：暂不支持，输出固定为None。

## 约束说明

- 参数q、dout、attn_out、ori_kv、cmp_kv的数据类型必须保持一致。
- 各个场景关于cmp_kv、cmp_sparse_indices的使用说明如下：
    - SWA场景：要求cmp_kv == None && cmp_sparse_indices == None
    - SCFA场景：要求cmp_kv != None && cmp_sparse_indices != None
    - CFA场景：要求cmp_kv != None && cmp_sparse_indices == None

## 调用示例

- 单算子模式调用

    ```python
    import math
    import random
    import numpy as np

    import torch
    import torch_npu
    import cann_ops_transformer

    S1 = 256
    S2 = 1024
    cmp_ratio = 4
    actual_seq_q = [S1]
    actual_seq_ori_kv = [S2]
    actual_seq_cmp_kv = [S2 // cmp_ratio]
    cmp_residual = [i % cmp_ratio for i in actual_seq_ori_kv]

    T1 = sum(actual_seq_q)
    T2 = sum(actual_seq_ori_kv)
    T3 = sum(actual_seq_cmp_kv)

    B = 1
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    scaleValue = 1 / math.sqrt(D)
    dtype = torch.float16
    input_layout = "TND"

    q_shape = tuple((T1, N1, D))
    ori_kv_shape = tuple((T2, N2, D))
    cmp_kv_shape = tuple((T3, N2, D))
    out_shape = tuple((T1, N1, D))
    lse_shape = tuple((N2, T1, N1 // N2))

    cu_seq_qlen = [0] + [sum(actual_seq_q[:x+1]) for x in range(len(actual_seq_q))]
    cu_seq_ori_kvlen = [0] + [sum(actual_seq_ori_kv[:x+1]) for x in range(len(actual_seq_ori_kv))]
    cu_seq_cmp_kvlen = [0] + [sum(actual_seq_cmp_kv[:x+1]) for x in range(len(actual_seq_cmp_kv))]

    q = (torch.rand(q_shape).to(dtype)) * 2
    ori_kv = (torch.rand(ori_kv_shape).to(dtype)) * 2
    cmp_kv = (torch.rand(cmp_kv_shape).to(dtype)) * 2 if cmp_ratio != 1 else None
    dy = (torch.rand(out_shape).to(dtype)) * 2 
    out = (torch.rand(out_shape).to(dtype)) * 2 # 实际使用场景中应使用前向输出attn_out
    sinks = (torch.rand(N1).to(torch.float32)) * (128)
    lse = (torch.rand(lse_shape).to(torch.float32)) # 实际使用场景中应使用前向输出lse

    if cmp_ratio == 4:
        cmp_sparse_indices = torch.ones([T1, N2, K], dtype=torch.int32) * -1
        accum_t = 0
        for i in range(B):
            cur_s1 = actual_seq_q[i]
            cur_ori_s2 = actual_seq_cmp_kv[i] * cmp_ratio + cmp_residual[i]
            delta_s = cur_ori_s2 - cur_s1
            for s1_idx in range(cur_s1):
                threshold = delta_s + s1_idx + 1 if delta_s + s1_idx + 1 >= 0 else 0
                max_cmp_s2 = threshold // cmp_ratio
                cur_s2_idxs = np.arange(max_cmp_s2)
                if max_cmp_s2 < K:
                    cmp_sparse_indices[accum_t, :, :max_cmp_s2] = torch.tensor(np.random.choice(cur_s2_idxs, size=max_cmp_s2, replace=False))
                else:
                    cmp_sparse_indices[accum_t, :, :] = torch.tensor(np.random.choice(cur_s2_idxs, size=K, replace=False))
                accum_t += 1
    else:
        cmp_sparse_indices = None

    cmp_kv_npu = cmp_kv.npu() if cmp_kv != None else None
    cmp_sparse_indices_npu = cmp_sparse_indices.npu() if cmp_sparse_indices != None else None
    cu_seq_qlen = None if input_layout != "TND" else torch.tensor(cu_seq_qlen).to(torch.int32).npu()
    cu_seq_ori_kvlen = None if input_layout != "TND" else torch.tensor(cu_seq_ori_kvlen).to(torch.int32).npu()
    cu_seq_cmp_kvlen = None if input_layout != "TND" else torch.tensor(cu_seq_cmp_kvlen).to(torch.int32).npu()
    cmp_residual_kv = torch.tensor(cmp_residual).to(torch.int32).npu() if cmp_residual != None else None

    metadata = cann_ops_transformer.ops.sparse_flash_mla_grad_metadata(N1, N2, D)
    npu_out = cann_ops_transformer.ops.sparse_flash_mla_grad(
            q.npu(), dy.npu(), out.npu(), lse.npu(),
            ori_kv=ori_kv.npu(), cmp_kv=cmp_kv_npu, 
            cmp_sparse_indices=cmp_sparse_indices_npu,
            cu_seqlens_q=cu_seq_qlen, 
            cu_seqlens_ori_kv=cu_seq_ori_kvlen, 
            cu_seqlens_cmp_kv=cu_seq_cmp_kvlen,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks.npu(),
            softmax_scale=scaleValue,
            cmp_ratio=cmp_ratio,
            layout_q=input_layout,
            layout_kv=input_layout,
            ori_win_left=127,
            ori_win_right=0,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            metadata=metadata
        )
    ```