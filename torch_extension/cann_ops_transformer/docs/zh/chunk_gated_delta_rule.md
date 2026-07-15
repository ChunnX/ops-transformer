# cann_ops_transformer.chunk_gated_delta_rule

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

    Chunk 门控 Delta Rule 前向计算，封装 `aclnnChunkGatedDeltaRule`。基于 chunk-wise 策略计算门控 DeltaNet 线性注意力。对 query/key 做 L2 归一化后按 chunk 切分，在 chunk 内通过 Delta Rule（带 beta 门控的递归状态更新）累积历史状态，chunk 间通过 gamma（g）门控做指数衰减传递状态；输出注意力结果 `out` 及最终递归状态 `final_state`。

- 计算公式：

    设 $Q$、$K$、$V$ 分别为 query、key、value，$\beta$ 为 beta 门控，$g$ 为 gamma 门控，$S_0$ 为初始状态，$\alpha$ 为缩放系数。对 $Q$、$K$ 做 L2 归一化后按 chunk 大小 $C$ 切分，在第 $i$ 个 chunk 内：

    $$
    K_{\beta} = K \odot \beta,\quad V_{\beta} = V \odot \beta
    $$

    $$
    A_i = (I + L_i)^{-1} - I,\quad \text{其中 } L_i = -\text{tril}((K_{\beta,i} K_i^T) \odot M_i)
    $$

    其中 $M_i$ 为由 $g$ 累积求和构造的衰减掩码。chunk 间递归状态更新：

    $$
    S_i = S_{i-1} \odot e^{g_{i,\text{last}}} + (K_i \odot e^{g_{i,\text{last}} - g_i})^T (V_{\beta,i} - K_{\text{cumdecay},i} S_{i-1})
    $$

    输出：

    $$
    O_i = (Q_i \odot e^{g_i}) S_{i-1} + \text{tril}(Q_i K_i^T \odot M_i) (V_{\beta,i} - K_{\text{cumdecay},i} S_{i-1})
    $$

**说明**
- 算子内部对 query 和 key 做 L2 归一化，无需用户预先归一化。
- `g`（gamma）为可选输入，不传入时等价于无衰减（$g=0$）。

## 函数原型

```python
cann_ops_transformer.chunk_gated_delta_rule(
    query, key, value, beta, initial_state, actual_seq_lengths,
    *,
    g=None,
    scale=1.0,
) -> (Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| query | Tensor | 必选 | query 输入。 | bfloat16 | (T, Nk, Dk) |
| key | Tensor | 必选 | key 输入。 | bfloat16 | (T, Nk, Dk) |
| value | Tensor | 必选 | value 输入。 | bfloat16 | (T, Nv, Dv) |
| beta | Tensor | 必选 | beta 门控。 | bfloat16 | (T, Nv) |
| initial_state | Tensor | 必选 | 初始递归状态。 | bfloat16 | (B, Nv, Dv, Dk) |
| actual_seq_lengths | Tensor | 必选 | 每个 batch 的实际序列长度，T 等于各元素之和。 | int32 | (B,) |
| g | Tensor | 可选 | gamma 门控，不传入时等价于无衰减。默认值为 None。 | float32 | (T, Nv) |
| scale | float | 可选 | 缩放系数，通常为 1/sqrt(Dk)。默认值为 1.0。 | - | - |

## 返回值说明

返回元组 `(out, final_state)`：

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|------|----------|-------------|
| out | Tensor | 注意力输出。 | bfloat16 | (T, Nv, Dv) |
| final_state | Tensor | 最终递归状态。 | bfloat16 | (B, Nv, Dv, Dk) |

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- `query`、`key` 的 shape 必须一致，为 `(T, Nk, Dk)`。
- `value` 的 shape 为 `(T, Nv, Dv)`，`out` 的 shape 为 `(T, Nv, Dv)`，与 value 前三维一致。
- `initial_state` 和 `final_state` 的 shape 必须一致，为 `(B, Nv, Dv, Dk)`。
- `Nk`、`Nv` 不超过 64；`Dv`、`Dk` 不超过 128。
- `Nv` 必须是 `Nk` 的整数倍。
- `actual_seq_lengths` 各元素之和必须等于 `T`。
- 数据类型约束：`query`/`key`/`value`/`beta`/`initial_state`/`out`/`final_state` 为 bfloat16，`actual_seq_lengths` 为 int32，`g` 为 float32。
- 算子输入不支持有 ±inf 和 nan 的情况。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import chunk_gated_delta_rule

    torch_npu.npu.set_device(0)

    B = 2
    seqlen = 200
    nk = 4
    nv = 8
    dk = 32
    dv = 32
    T = B * seqlen

    q = torch.randn(T, nk, dk, dtype=torch.bfloat16, device="npu")
    k = torch.randn(T, nk, dk, dtype=torch.bfloat16, device="npu")
    v = torch.randn(T, nv, dv, dtype=torch.bfloat16, device="npu")
    beta = torch.rand(T, nv, dtype=torch.bfloat16, device="npu")
    initial_state = torch.randn(B, nv, dv, dk, dtype=torch.bfloat16, device="npu")
    actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device="npu")
    g = torch.rand(T, nv, dtype=torch.float32, device="npu") * -1.0
    scale = 1.0 / (dk ** 0.5)

    out, final_state = chunk_gated_delta_rule(
        q, k, v, beta, initial_state, actual_seq_lengths,
        g=g, scale=scale,
    )
    torch_npu.npu.synchronize()
    assert out.shape == (T, nv, dv)
    assert final_state.shape == (B, nv, dv, dk)
    ```

- 图模式调用

    通过 `torch.compile(backend="atc")` 或 `torchair` 自动将算子转换为 GE 图算子，无需额外配置。

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import chunk_gated_delta_rule

    torch_npu.npu.set_device(0)

    B = 2
    seqlen = 200
    nk = 4
    nv = 8
    dk = 32
    dv = 32
    T = B * seqlen

    class ChunkGatedDeltaRuleModel(torch.nn.Module):
        def forward(self, q, k, v, beta, initial_state, actual_seq_lengths, g):
            out, final_state = chunk_gated_delta_rule(
                q, k, v, beta, initial_state, actual_seq_lengths,
                g=g, scale=1.0 / (dk ** 0.5),
            )
            return out, final_state

    model = ChunkGatedDeltaRuleModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    q = torch.randn(T, nk, dk, dtype=torch.bfloat16, device="npu")
    k = torch.randn(T, nk, dk, dtype=torch.bfloat16, device="npu")
    v = torch.randn(T, nv, dv, dtype=torch.bfloat16, device="npu")
    beta = torch.rand(T, nv, dtype=torch.bfloat16, device="npu")
    initial_state = torch.randn(B, nv, dv, dk, dtype=torch.bfloat16, device="npu")
    actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device="npu")
    g = torch.rand(T, nv, dtype=torch.float32, device="npu") * -1.0

    out, final_state = model(q, k, v, beta, initial_state, actual_seq_lengths, g)
    ```
