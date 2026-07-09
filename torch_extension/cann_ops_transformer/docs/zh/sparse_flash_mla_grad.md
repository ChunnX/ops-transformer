# sparse_flash_mla_grad / sparse_flash_mla_grad_metadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
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

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| q | 必选输入 | 对应公式中的$Q$。`layout_q`="BSND" 时 shape 为 `[B, S1, N1, D]`；`layout_q`="TND" 时 shape 为 `[T1, N1, D]`。B：支持泛化；S1：支持泛化；D：512；T1：B × S1。 | FLOAT16、BFLOAT16 | ND |
| dout | 必选输入 | 注意力正向输出矩阵的梯度，对应公式中的$dO$。数据类型和 shape 均与 q 保持一致。 | FLOAT16、BFLOAT16 | ND |
| attn_out | 必选输入 | 注意力正向输出矩阵，对应公式中的$O$。数据类型和 shape 均与 q 保持一致。 | FLOAT16、BFLOAT16 | ND |
| softmax_lse | 必选输入 | 注意力正向计算的输出 lse。`layout_q`="BSND" 时 shape 为 `[B, N2, S1, G]`；`layout_q`="TND" 时 shape 为 `[N2, T1, G]`。B：与 q 的 B 保持一致；N2：1；S1：与 q 的 S1 保持一致；G：N1/N2；T1：B × S1。 | FLOAT32 | ND |
| ori_kv | 可选输入 | 对应公式中的$oriKv$。`layout_kv`="BSND" 时 shape 为 `[B, S2, N2, D]`；`layout_kv`="TND" 时 shape 为 `[T2, N2, D]`。B：与 q 的 B 保持一致；S2：支持泛化；N2：1；D：512；T2：B × S2。 | FLOAT16、BFLOAT16 | ND |
| cmp_kv | 可选输入 | 对应公式中的$cmpkv$。`layout_kv`="BSND" 时 shape 为 `[B, S3, N2, D]`；`layout_kv`="TND" 时 shape 为 `[T3, N2, D]`。B：与 q 的 B 保持一致；S3：支持泛化；N2：1；D：512；T3：B × S3。传 None 时按 SWA 场景计算。 | FLOAT16、BFLOAT16 | ND |
| ori_sparse_indices | 可选输入 | 对应 oriKv 部分的 topk 索引。`layout_q`="BSND" 时 shape 为 `[B, S1, N2, K1]`；`layout_q`="TND" 时 shape 为 `[T1, N2, K1]`。B：与 q 的 B 保持一致；S1：与 q 的 S1 保持一致；N2：1；K1：支持泛化；T1：B × S1。 | INT32 | ND |
| cmp_sparse_indices | 可选输入 | 对应公式中的$topkIndices$。`layout_q`="BSND" 时 shape 为 `[B, S1, N2, K2]`；`layout_q`="TND" 时 shape 为 `[T1, N2, K2]`。B：与 q 的 B 保持一致；S1：与 q 的 S1 保持一致；N2：1；K2：支持泛化；T1：B × S1。若 cmp_kv 不为 None，此时 cmp_sparse_indices 不为 None 时按 SCFA 场景计算，为 None 时按 CFA 场景计算；若 cmp_kv 为 None，则 cmp_sparse_indices 只能为 None，此时按 SWA 场景计算。 | INT32 | ND |
| cu_seqlens_q | 可选输入 | 每个 Batch 中 q 的有效 token 数的累加和形式。`layout_q`="TND" 时必传。shape 为 `[B+1]`，累加和与 T1 保持一致。 | INT32 | ND |
| cu_seqlens_ori_kv | 可选输入 | 每个 Batch 中 ori_kv 的有效 token 数的累加和形式。`layout_kv`="TND" 时必传。shape 为 `[B+1]`，累加和与 T2 保持一致。 | INT32 | ND |
| cu_seqlens_cmp_kv | 可选输入 | 每个 Batch 中 cmp_kv 的有效 token 数的累加和形式。`layout_kv`="TND" 时必传。shape 为 `[B+1]`，累加和与 T3 保持一致。 | INT32 | ND |
| seqused_q | 可选输入 | 表示不同 batch 中 query 实际参与运算的 token 数。shape 为 `[B]`。 | INT32 | ND |
| seqused_ori_kv | 可选输入 | 表示不同 batch 中 ori_kv 实际参与运算的 token 数。shape 为 `[B]`。 | INT32 | ND |
| seqused_cmp_kv | 可选输入 | 表示不同 batch 中 cmp_kv 实际参与运算的 token 数。shape 为 `[B]`。 | INT32 | ND |
| cmp_residual_kv | 可选输入 | 表示每个 batch S2 // cmpRatio 后的余数。shape 为 `[B]`。当 cmp_kv 不为空且 cmp_mask_mode=3 时必须传入。 | INT32 | ND |
| ori_topk_length | 可选输入 | 表示每行 query 对应的 ori_kv 实际可选的 topk 长度。shape 为 `[B, S1, N2]`（BSND）或 `[T1, N2]`（TND）。当 ori_mask_mode=0 且 ori_sparse_indices 不为 None 时必须传入且必须为准确值。 | INT32 | ND |
| cmp_topk_length | 可选输入 | 表示每行 query 对应的 cmp_kv 实际可选的 topk 长度。shape 为 `[B, S1, N2]`（BSND）或 `[T1, N2]`（TND）。当 cmp_mask_mode=0 且 cmp_sparse_indices 不为 None 时必须传入且必须为准确值。 | INT32 | ND |
| sinks | 可选输入 | 注意力下沉 tensor。shape 为 `[N1]`。 | FLOAT32 | ND |
| metadata | 可选输入 | 表示 tiling 下沉的 aicpu 算子输出结果。 | INT32 | ND |
| softmax_scale | 可选属性 | 缩放系数。默认值：1.0 / sqrt(D)。 | FLOAT | - |
| cmp_ratio | 可选属性 | 压缩率，取值范围 1~128。默认值：1。 | INT | - |
| ori_mask_mode | 可选属性 | q 和 ori_kv 计算的 mask 模式。模式 0 为不做 mask 操作；模式 3 为 rightDownCausal；模式 4 为 band（滑窗，起点右下角）。 | INT | - |
| cmp_mask_mode | 可选属性 | q 和 cmp_kv 计算的 mask 模式。模式 0 为不做 mask 操作；模式 3 为 rightDownCausal。 | INT | - |
| ori_win_left | 可选属性 | q 和 ori_kv 计算中 q 对过去 token 计算的数量。当前仅支持取值 127。 | INT | - |
| ori_win_right | 可选属性 | q 和 ori_kv 计算中 q 对未来 token 计算的数量。当前仅支持取值 0。 | INT | - |
| layout_q | 可选属性 | q 的数据排布格式。支持 "BSND"、"TND"。 | STRING | - |
| layout_kv | 可选属性 | ori_kv、cmp_kv 的数据排布格式。支持 "BSND"、"TND"，当前必须与 layout_q 保持一致。 | STRING | - |

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持 sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional、oriTopkLengthOptional、cmpTopkLengthOptional、metadataOptional 字段。

### sparse_flash_mla_grad_metadata

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| num_heads_q | 必选属性 | 表示公式中$Q$的头数（即 N1）。 | INT | - |
| num_heads_kv | 必选属性 | 表示公式中$oriKv$或$cmpkv$的头数（即 N2）。 | INT | - |
| head_dim | 必选属性 | 表示头的维度（即 D）。 | INT | - |
| cu_seqlens_q | 可选输入 | 每个 Batch 中 q 的有效 token 数的累加和形式。`layout_q`="TND" 时必传。shape 为 `[B+1]`，累加和与 T1 保持一致。 | INT32 | ND |
| cu_seqlens_ori_kv | 可选输入 | 每个 Batch 中 ori_kv 的有效 token 数的累加和形式。`layout_kv`="TND" 时必传。shape 为 `[B+1]`，累加和与 T2 保持一致。 | INT32 | ND |
| cu_seqlens_cmp_kv | 可选输入 | 每个 Batch 中 cmp_kv 的有效 token 数的累加和形式。`layout_kv`="TND" 时必传。shape 为 `[B+1]`，累加和与 T3 保持一致。 | INT32 | ND |
| seqused_q | 可选输入 | 表示不同 batch 中 query 实际参与运算的 token 数。shape 为 `[B]`。 | INT32 | ND |
| seqused_ori_kv | 可选输入 | 表示不同 batch 中 ori_kv 实际参与运算的 token 数。shape 为 `[B]`。 | INT32 | ND |
| seqused_cmp_kv | 可选输入 | 表示不同 batch 中 cmp_kv 实际参与运算的 token 数。shape 为 `[B]`。 | INT32 | ND |
| cmp_residual_kv | 可选输入 | 表示每个 batch S2 // cmpRatio 后的余数。shape 为 `[B]`。当 cmp_kv 不为空且 cmp_mask_mode=3 时必须传入。 | INT32 | ND |
| ori_topk_length | 可选输入 | 表示每行 query 对应的 ori_kv 实际可选的 topk 长度。shape 为 `[B, S1, N2]`（BSND）或 `[T1, N2]`（TND）。当 ori_mask_mode=0 且 ori_sparse_indices 不为 None 时必须传入且必须为准确值。 | INT32 | ND |
| cmp_topk_length | 可选输入 | 表示每行 query 对应的 cmp_kv 实际可选的 topk 长度。shape 为 `[B, S1, N2]`（BSND）或 `[T1, N2]`（TND）。当 cmp_mask_mode=0 且 cmp_sparse_indices 不为 None 时必须传入且必须为准确值。 | INT32 | ND |
| batch_size | 可选属性 | 表示输入样本批量大小（即 B）。 | INT | - |
| max_seqlen_q | 可选属性 | TND 场景下输入 q 的最大序列长度。 | INT | - |
| max_seqlen_ori_kv | 可选属性 | TND 场景下输入 ori_kv 的最大序列长度。 | INT | - |
| max_seqlen_cmp_kv | 可选属性 | TND 场景下输入 cmp_kv 的最大序列长度。 | INT | - |
| ori_topk | 可选属性 | ori_kv 的 topk 长度。 | INT | - |
| cmp_topk | 可选属性 | cmp_kv 的 topk 长度。 | INT | - |
| cmp_ratio | 可选属性 | 压缩率，取值范围 1~128。默认值：1。 | INT | - |
| ori_mask_mode | 可选属性 | q 和 ori_kv 计算的 mask 模式。模式 0 为不做 mask 操作；模式 3 为 rightDownCausal；模式 4 为 band（滑窗，起点右下角）。 | INT | - |
| cmp_mask_mode | 可选属性 | q 和 cmp_kv 计算的 mask 模式。模式 0 为不做 mask 操作；模式 3 为 rightDownCausal。 | INT | - |
| ori_win_left | 可选属性 | q 和 ori_kv 计算中 q 对过去 token 计算的数量。当前仅支持取值 127。 | INT | - |
| ori_win_right | 可选属性 | q 和 ori_kv 计算中 q 对未来 token 计算的数量。当前仅支持取值 0。 | INT | - |
| layout_q | 可选属性 | q 的数据排布格式。支持 "BSND"、"TND"。 | STRING | - |
| layout_kv | 可选属性 | ori_kv、cmp_kv 的数据排布格式。支持 "BSND"、"TND"，当前必须与 layout_q 保持一致。 | STRING | - |
| has_ori_kv | 可选属性 | 表示是否传入 ori_kv。 | BOOL | - |
| has_cmp_kv | 可选属性 | 表示是否传入 cmp_kv。 | BOOL | - |
| metadata | 输出 | 暂不支持，输出固定为 None。 | INT32 | ND |

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持 sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional、oriTopkLengthOptional、cmpTopkLengthOptional、metadataOptional 字段。

## 返回值说明

### sparse_flash_mla_grad

- **dq**（`Tensor`）：对应公式中的$dQ$，支持非连续，数据格式支持ND，数据类型、shape与输入q保持一致。

- **dori_kv**（`Tensor`）：可选输出，表示输入ori_kv的梯度，支持非连续，数据格式支持ND，数据类型、shape与输入ori_kv保持一致。

- **dcmp_kv**（`Tensor`）：可选输出，表示输入cmp_kv的梯度，支持非连续，数据格式支持ND，数据类型、shape与输入cmp_kv保持一致；当cmp_kv为None时，dcmp_kv也为None。

- **dsinks**（`Tensor`）：可选输出，表示输入sinks的梯度，支持非连续，数据格式支持ND，数据类型支持`float32`，shape与输入sinks保持一致。

- **ori_softmax_l1norm**（`Tensor`）：可选输出，表示q与ori_kv计算得出的softmax的L1Norm结果，公式为reduceG(softmax)/G；数据类型为`float32`。`layout_q`为BSND时shape为`[B,S1,N2,K1]`，当`layout_q`为TND时shape为`[T1,N2,K1]`。当 ori_sparse_indices 不为 None 时该输出不为空，其他场景下输出为 None。

- **cmp_softmax_l1norm**（`Tensor`）：可选输出，表示q与cmp_kv计算得出的softmax的L1Norm结果，公式为reduceG(softmax)/G；当 cmp_sparse_indices 不为 None 时该输出不为空，其他场景下输出为 None。

### sparse_flash_mla_grad_metadata

- **metadata**（`Tensor`）：暂不支持，输出固定为None。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - oriSoftmaxL1NormOptional 输出不支持（固定为 None）；cmpSoftmaxL1NormOptional 仅在 SCFA 场景（cmp_kv != None 且 cmp_sparse_indices != None）时输出有效。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - oriSoftmaxL1NormOptional 输出支持；cmpSoftmaxL1NormOptional 支持 SCFA 场景及 cmp_mask_mode=0 的稀疏场景。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持 sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional、oriTopkLengthOptional、cmpTopkLengthOptional、metadataOptional 字段。

## 约束说明

- 参数q、dout、attn_out、ori_kv、cmp_kv的数据类型必须保持一致。
- 入参为空的场景处理：q 为空 Tensor 时直接返回。
- 各个场景关于cmp_kv、cmp_sparse_indices的使用说明如下：
    - SWA场景：要求cmp_kv == None && cmp_sparse_indices == None
    - SCFA场景：要求cmp_kv != None && cmp_sparse_indices != None
    - CFA场景：要求cmp_kv != None && cmp_sparse_indices == None
- **确定性计算**：Ascend 950PR/Ascend 950DT 默认非确定性计算，支持通过 `aclrtCtxSetSysParamOpt` 开启确定性计算；Atlas A2/A3 不支持开启确定性计算。
- **Mask 模式支持**：

    | 模式 | 含义 | 备注 |
    | :--- | :--- | :--- |
    | 0 | 不做 mask 操作 | 支持 |
    | 3 | rightDownCausal | 支持 |
    | 4 | band（滑窗，起点右下角） | oriMaskMode 支持 |

- **规格约束**：

    | 规格项 | 规格 | 规格说明 |
    | :--- | :--- | :--- |
    | B | 支持泛化 | - |
    | S1、S2 | 支持泛化 | 支持S1、S2不等长。 |
    | N1 | 1~128 | - |
    | N2 | 1 | 当前仅支持N2=1。 |
    | D | 512 | q、ori_kv、cmp_kv最后一维需保持一致。 |
    | layout_q/kv | BSND / TND，必须一致 | - |
    | cmp_ratio | 1~128 | - |

  - ori_kv/cmp_kv 传 None 的支持情况:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
  - ori_sparse_indices 的支持情况:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
  - seqused 的支持情况:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持（需传 None）。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
  - ori_topk_length / cmp_topk_length 的支持情况:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持（需传 None）。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
  - sinks 传 None 的支持情况:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
  - metadata 的支持情况:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持传 None。
    - <term>Ascend 950PR/Ascend 950DT</term>：必须传。
  - ori_mask_mode 的支持情况:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持模式 4。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持模式 0、3、4。
  - cmp_mask_mode 的支持情况:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持模式 3。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持模式 0、3。

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