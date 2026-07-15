# npu_apply_rotary_pos_emb_backward

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

- 接口功能：该接口实现旋转位置编码（RoPE）的反向计算。输入正向算子输出`query_embed`和`key_embed`的梯度，结合正向输入`cos`、`sin`、`query`和`key`，分别计算正向输入`query`、`key`、`cos`和`sin`的梯度。
- 计算公式：
  - $grad\_q\_embed$ / $grad\_k\_embed$：来自 $q\_embed$ / $k\_embed$ 输出端的上游梯度。
  - $grad_{q_1}, grad_{q_2}$ / $grad_{k_1}, grad_{k_2}$：$grad\_q\_embed$ / $grad\_k\_embed$ 沿最后维度均分的两半梯度。
  - $c_1, c_2$ / $s_1, s_2$：$cos$ / $sin$ 沿最后维度均分的两半值。
  - $q_1, q_2$ / $k_1, k_2$：$query$ / $key$ 沿最后维度均分的两半输入。
  - $query\_rotate$ / $key\_rotate$：正向旋转构造的中间张量。
  - $grad\_query$ / $grad\_key$：反向计算得到的完整 Query / Key 梯度。
  - $grad\_cos$ / $grad\_sin$：反向计算得到的完整 Cos / Sin 梯度。
  - $*$：逐元素乘法。
  - $\mathrm{chunk}(\cdot, \mathrm{chunks}=2, \mathrm{dim}=-1)$：沿最后一个维度将张量均分为两部分。
  - $\mathrm{cat}(\cdot, \mathrm{dim}=-1)$：沿最后一个维度拼接张量。
  - $\mathrm{sum}(\cdot, \mathrm{dims})$：对指定的广播维度集合执行求和，将梯度形状还原至正向输入 $cos$ / $sin$ 的原始形状；$\mathrm{dims}$ 为正向计算中 $cos$ / $sin$ 相对于 $query$ / $key$ 的广播维度集合。

  中间量计算：

  $$
  \begin{aligned}
  grad_{q_1}, grad_{q_2} &= \mathrm{chunk}(grad\_q\_embed, \mathrm{chunks}=2, \mathrm{dim}=-1) \\
  grad_{k_1}, grad_{k_2} &= \mathrm{chunk}(grad\_k\_embed, \mathrm{chunks}=2, \mathrm{dim}=-1) \\
  c_1, c_2 &= \mathrm{chunk}(\cos, \mathrm{chunks}=2, \mathrm{dim}=-1) \\
  s_1, s_2 &= \mathrm{chunk}(\sin, \mathrm{chunks}=2, \mathrm{dim}=-1) \\
  \text{query\_rotate} &= \mathrm{cat}(-q_2,\ q_1,\ \mathrm{dim}=-1) \\
  \text{key\_rotate} &= \mathrm{cat}(-k_2,\ k_1,\ \mathrm{dim}=-1)
  \end{aligned}
  $$

  输出计算：

  $$
  \begin{aligned}
  grad\_{query} &= \mathrm{cat}\big(c_1 * grad_{q_1} + s_2 * grad_{q_2},\ c_2 * grad_{q_2} - s_1 * grad_{q_1},\ \mathrm{dim}=-1\big) \\
  grad\_{key} &= \mathrm{cat}\big(c_1 * grad_{k_1} + s_2 * grad_{k_2},\ c_2 * grad_{k_2} - s_1 * grad_{k_1},\ \mathrm{dim}=-1\big) \\
  grad\_cos &= \mathrm{sum}(grad\_q\_embed * \text{query} + grad\_k\_embed * \text{key},\ \mathrm{dims}) \\
  grad\_sin &= \mathrm{sum}(grad\_q\_embed * \text{query\_rotate} + grad\_k\_embed * \text{key\_rotate},\ \mathrm{dims})
  \end{aligned}
  $$

## 函数原型

```python
torch.ops.custom.npu_apply_rotary_pos_emb_backward(
    grad_query_embed,
    grad_key_embed,
    cos,
    sin,
    query,
    key,
    *,
    rotary_mode=0,
    layout=1
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| grad_query_embed | Tensor | 必选 | 正向算子输出`query_embed`的梯度，作为待计算输入。要求为3D或4D Tensor，数据格式为ND，最后一维`D`需满足`D ≤ 128`且`D`为16的倍数。| float16、bfloat16、float32 | 4D/3D：[B,S,N,D] / [S,B,N,D] / [T,N,D] |
| grad_key_embed | Tensor | 必选 | 正向算子输出`key_embed`的梯度，作为待计算输入。除`N`维度外，其余维度需与`grad_query_embed`保持一致。 | 与grad_query_embed一致 | 4D/3D，除N维外与grad_query_embed一致 |
| cos | Tensor | 必选 | 正向算子输入`cos`。`N`维度需为1，其余维度需与`grad_query_embed`保持一致。当需要计算grad_cos时，需要设置requires_grad属性为True。 | 与grad_query_embed一致 | 4D/3D，N维为1 |
| sin | Tensor | 必选 | 正向算子输入`sin`。`N`维度需为1，其余维度需与`grad_query_embed`保持一致。当需要计算grad_sin时，需要设置requires_grad属性为True。 | 与grad_query_embed一致 | 4D/3D，N维为1 |
| query | Tensor | 必选 | 正向算子输入`query`，用于计算`grad_cos`和`grad_sin`。维度需与`grad_query_embed`保持一致。 | 与grad_query_embed一致 | 与grad_query_embed一致 |
| key | Tensor | 必选 | 正向算子输入`key`，用于计算`grad_cos`和`grad_sin`。维度需与`grad_key_embed`保持一致。 | 与grad_query_embed一致 | 与grad_key_embed一致 |
| rotary_mode | int | 可选 | 旋转模式，默认值为0，表示half模式。 | int | - |
| layout | int | 可选 | 输入Tensor的布局格式，支持1-BSND、2-SBND、4-TND，默认值为1。 | int | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| grad_query | Tensor | 必选 | 正向计算输入`query`的梯度。 | 与grad_query_embed一致 | 与grad_query_embed一致 |
| grad_key | Tensor | 必选 | 正向计算输入`key`的梯度。 | 与grad_query_embed一致 | 与grad_key_embed一致 |
| grad_cos | Tensor | 必选 | 正向计算输入`cos`的梯度。 | 与grad_query_embed一致 | 与cos一致 |
| grad_sin | Tensor | 必选 | 正向计算输入`sin`的梯度。 | 与grad_query_embed一致 | 与sin一致 |

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式调用。
- 输入Tensor的数据类型仅支持float16、bfloat16、float32，且所有输入数据类型需保持一致。
- `grad_query_embed`、`grad_key_embed`、`cos`、`sin`、`query`、`key`的维度数需为3或4，且彼此兼容：
  - `grad_query_embed`与`grad_key_embed`的`B/S/T`、`D`维度需一致，`N`维度可不同。
  - `cos`、`sin`的`N`维度必须为1，其余维度需与`grad_query_embed`一致。
  - `query`维度需与`grad_query_embed`一致，`key`维度需与`grad_key_embed`一致。
- 若需要计算grad_cos，需设置cos的requires_grad属性为True；若需要计算grad_sin，需设置sin的requires_grad属性为True。当cos或sin任一requires_grad为True时，query和key会参与反向计算。


## 确定性计算

- 默认支持确定性计算。
- 默认支持batch invariance。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import omni_training_custom_ops  
  batch = 2
  seq_len = 16
  head_num_q = 8
  head_num_k = 1
  head_dim = 64  
  grad_query_embed = torch.randn(batch, seq_len, head_num_q, head_dim, dtype=torch.float16).npu()
  grad_key_embed = torch.randn(batch, seq_len, head_num_k, head_dim, dtype=torch.float16).npu()
  cos = torch.randn(batch, seq_len, 1, head_dim, dtype=torch.float16).npu()
  sin = torch.randn(batch, seq_len, 1, head_dim, dtype=torch.float16).npu()
  query = torch.randn(batch, seq_len, head_num_q, head_dim, dtype=torch.float16).npu()
  key = torch.randn(batch, seq_len, head_num_k, head_dim, dtype=torch.float16).npu()
  cos.requires_grad_(True)
  sin.requires_grad_(True)
  grad_query, grad_key, grad_cos, grad_sin = torch.ops.custom.npu_apply_rotary_pos_emb_backward(
      grad_query_embed,
      grad_key_embed,
      cos,
      sin,
      query,
      key,
      rotary_mode=0,
      layout=1
  )
  ```
