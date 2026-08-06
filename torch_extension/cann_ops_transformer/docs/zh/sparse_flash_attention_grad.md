# sparse_flash_attention_grad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->

## 功能说明

- **接口功能**：`sparse_flash_attention_grad` 计算 SparseFlashAttentionGrad（SFAG，MLA OSS Sink）训练场景下注意力的反向输出，封装 `aclnnSparseFlashAttentionGrad`。支持 MLA（Multi-head Latent Attention）的 OSS Sink 增量：输入 `sinks`（FP32 `[N1]`），输出 `d_sinks`（FP32 `[N1]`）。

- **计算公式**：

  $$
  P = Softmax(scale \cdot Q \cdot K^T, softmax\_max, softmax\_sum)
  $$

  $$
  dS = P \times (dO \cdot V^T - SoftmaxGrad(dO, O))
  $$

  $$
  dQ = scale \cdot dS \cdot K
  $$

  $$
  dKV = scale \cdot dS^T \cdot Q + P^T \cdot dO
  $$

  $$
  dSinks[h] = \sum_{b,i} P_{sink}[b,i,h] \cdot (-D_i[b,i,h])
  $$

  其中 `P_sink` 表示 sink 位置对应的 softmax 概率，`D_i` 为注意力差分项。

## 函数原型

```python
cann_ops_transformer.sparse_flash_attention_grad(
    query,
    key,
    value,
    sparse_indices,
    d_out,
    out,
    softmax_max,
    softmax_sum,
    sinks=None,
    scale_value=1.0,
    sparse_block_size=1,
    *,
    query_rope=None,
    key_rope=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    layout_q="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    win_left=2147483647,
    win_right=2147483647,
    attention_mode=0,
    deterministic=False
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **query**（`Tensor`）：必选参数，对应公式中的$Q$。`layout`="TND" 时shape为 `[T1, N1, D+Dr]`。数据格式支持ND，数据类型支持`bfloat16`/`float16`。

- **key**（`Tensor`）：必选参数，对应公式中的$K$。`layout`="TND" 时shape为 `[T2, N2, D+Dr]`。数据格式支持ND，数据类型与query一致。

- **value**（`Tensor`）：必选参数，对应公式中的$V$。`layout`="TND" 时shape为 `[T2, N2, D]`。数据格式支持ND，数据类型与query一致。

- **sparse_indices**（`Tensor`）：必选参数，topk 索引（select block id），不足部分填充 -1。shape为 `[T1, N2, K]`（TND）。数据格式支持ND，数据类型支持`int32`。

- **d_out**（`Tensor`）：必选参数，注意力正向输出矩阵的梯度，对应公式中的$dO$。数据类型和shape与 `out` 保持一致。

- **out**（`Tensor`）：必选参数，注意力正向输出矩阵，对应公式中的$O$。数据类型和shape与 `query` 保持一致。

- **softmax_max**（`Tensor`）：必选参数，前向计算的 softmax max。shape为 `[T1, N1]`（TND）。数据格式支持ND，数据类型支持`float32`。

- **softmax_sum**（`Tensor`）：必选参数，前向计算的 softmax sum。shape为 `[T1, N1]`（TND）。数据格式支持ND，数据类型支持`float32`。

- **sinks**（`Tensor`，可选）：oss-sink 输入。shape为 `[N1]`。数据格式支持ND，数据类型支持`float32`。传 None 时 sink 路径编译期消除，输出 `d_sinks` 为 shape `[0]` 的空 tensor。

- **scale_value**（`float`）：必选参数，query@key 缩放系数，对应公式中的$scale$。默认值：1.0。

- **sparse_block_size**（`int`）：必选参数，block 大小。取值范围 `{1,8,16,32,64}`，默认值：1。

- **query_rope**（`Tensor`，可选）：query 的 rope 部分，仅 `layout`="TND" 时使用。shape为 `[T1, N1, Dr]`。数据类型与query一致。

- **key_rope**（`Tensor`，可选）：key 的 rope 部分，仅 `layout`="TND" 时使用。shape为 `[T2, N2, Dr]`。数据类型与query一致。

- **cu_seqlens_q**（`Tensor`，可选）：每个 batch 中 query 的有效 token 数。shape为 `[B]`。数据类型支持`int32`。

- **cu_seqlens_kv**（`Tensor`，可选）：每个 batch 中 key/value 的有效 token 数。shape为 `[B]`。数据类型支持`int32`。

- **layout_q**（`str`，可选）：query 输入数据排布，支持 "BSND"/"TND"，默认值：`"BSND"`。

- **layout_kv**（`str`，可选）：key/value 输入数据排布，支持 "BSND"/"TND"，默认值：`"BSND"`。约定与 `layout_q` 恒等。

- **sparse_mode**（`int`，可选）：稀疏模式，默认值：3。

- **win_left**（`int`，可选）：左侧窗口 token 数，默认值：2147483647。

- **win_right**（`int`，可选）：右侧窗口 token 数，默认值：2147483647。

- **attention_mode**（`int`，可选）：注意力模式占位参数（暂不参与计算），默认值：0。

- **deterministic**（`bool`，可选）：是否确定性计算，默认值：False。

## 返回值说明

- **dq**（`Tensor`）：query 的梯度，数据类型和shape与 `query` 保持一致。

- **dk**（`Tensor`）：key 的梯度，数据类型和shape与 `key` 保持一致。

- **dv**（`Tensor`）：value 的梯度，数据类型和shape与 `value` 保持一致。

- **dq_rope**（`Tensor`）：可选输出，query_rope 的梯度。`query_rope` 为 None 时返回 shape `[0]` 的空 tensor。

- **dk_rope**（`Tensor`）：可选输出，key_rope 的梯度。`key_rope` 为 None 时返回 shape `[0]` 的空 tensor。

- **d_sinks**（`Tensor`）：可选输出，sinks 的梯度。数据类型支持`float32`，shape与 `sinks` 保持一致；`sinks` 为 None 时返回 shape `[0]` 的空 tensor。

## 约束说明

- 该接口支持训练场景下使用，支持单算子模式。
- query、key、value、d_out、out、query_rope、key_rope 的数据类型必须一致（`bfloat16`/`float16`）。
- softmax_max、softmax_sum、sinks 的数据类型为 `float32`，sparse_indices 的数据类型为 `int32`。
- `layout` 仅支持 "BSND"/"TND"，且 TND 场景需配套传入 `query_rope`/`key_rope`。
- `sparse_block_size` 仅支持 `{1,8,16,32,64}`。

## 确定性计算

支持通过 `deterministic` 参数控制确定性计算，默认关闭。

## 调用示例

- 单算子模式调用

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    B = 1
    S1 = 64
    S2 = 2048
    N1 = 16
    N2 = 1
    D = 512
    Dr = 64
    K = 2048
    scale_value = 1 / math.sqrt(D)
    dtype = torch.bfloat16
    input_layout = "TND"
    T1 = B * S1
    T2 = B * S2

    query = (torch.randn(T1, N1, D + Dr).to(dtype)).npu()
    key = (torch.randn(T2, N2, D + Dr).to(dtype)).npu()
    value = (torch.randn(T2, N2, D).to(dtype)).npu()
    out = (torch.randn(T1, N1, D + Dr).to(dtype)).npu()
    d_out = (torch.randn(T1, N1, D + Dr).to(dtype)).npu()
    softmax_max = torch.randn(T1, N1).to(torch.float32).npu()
    softmax_sum = torch.rand(T1, N1).add(1.0).to(torch.float32).npu()
    sinks = torch.randn(N1).to(torch.float32).npu()
    sparse_indices = torch.randint(0, S2, (T1, N2, K), dtype=torch.int32).npu()
    query_rope = (torch.randn(T1, N1, Dr).to(dtype)).npu()
    key_rope = (torch.randn(T2, N2, Dr).to(dtype)).npu()

    dq, dk, dv, dq_rope, dk_rope, d_sinks = cann_ops_transformer.sparse_flash_attention_grad(
        query, key, value, sparse_indices, d_out, out, softmax_max, softmax_sum,
        sinks, scale_value, 1,
        query_rope=query_rope, key_rope=key_rope,
        cu_seqlens_q=None, cu_seqlens_kv=None,
        layout_q=input_layout, layout_kv=input_layout, sparse_mode=0, deterministic=False,
    )
    ```
