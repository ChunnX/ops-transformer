# cann_ops_transformer.chunk\_gated\_delta\_rule

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

    Chunk Gated Delta Rule 前向计算，封装 aclnnChunkGatedDeltaRule。基于 chunk-wise 算法计算门控增量规则（Gated Delta Rule），支持变长序列。对 query/key 做 L2 归一化后，按 chunk 分块计算注意力输出与递归状态更新，可选的门控因子 g 控制状态衰减。

- 计算公式：

    在每个 chunk 内，计算局部注意力并更新递归状态 $S$：

    $$
    Q' = Q \cdot \text{scale}, \quad K_\beta = K \odot \beta, \quad V_\beta = V \odot \beta
    $$

    $$
    A = (I + \text{tril}(-(K_\beta K^T) \cdot \text{decay\_mask}))
    $$

    $$
    O_{\text{inter}} = Q' S_{\text{prev}} \cdot e^{g}, \quad O_{\text{chunk}} = Q' K^T \cdot \text{decay\_mask} \cdot (A V_\beta - K_\text{cumdecay} S_{\text{prev}})
    $$

    $$
    O = O_{\text{inter}} + O_{\text{chunk}}, \quad S_{\text{new}} = S_{\text{prev}} \cdot e^{g_{\text{last}}} + (K \cdot e^{g_{\text{last}} - g})^T \cdot V_{\text{new}}
    $$

    其中，$\text{scale}$ 为缩放系数，$\beta$ 为 beta 因子，$g$ 为门控因子，$\text{decay\_mask}$ 由 $g$ 的累加差值指数化得到。

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
| query | Tensor | 必选 | 查询张量。 | bfloat16 | [T, Nk, Dk] |
| key | Tensor | 必选 | 键张量。 | bfloat16 | [T, Nk, Dk] |
| value | Tensor | 必选 | 值张量。 | bfloat16 | [T, Nv, Dv] |
| beta | Tensor | 必选 | beta 因子。 | bfloat16 | [T, Nv] |
| initial_state | Tensor | 必选 | 初始递归状态。 | bfloat16 | [B, Nv, Dv, Dk] |
| actual_seq_lengths | Tensor | 必选 | 各 batch 的实际序列长度。 | int32 | [B] |
| g | Tensor | 可选 | 门控因子，None 表示不使用门控。默认值为 None。 | float32 | [T, Nv] |
| scale | float | 可选 | 缩放系数，通常取 1/sqrt(Dk)。默认值为 1.0。 | - | - |

## 返回值说明

| 输出名 | 类型 | 描述 | 数据类型 | 维度(shape) |
|--------|------|------|----------|-------------|
| out | Tensor | 注意力输出。 | bfloat16 | [T, Nv, Dv] |
| final_state | Tensor | 最终递归状态。 | bfloat16 | [B, Nv, Dv, Dk] |

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- `query`、`key`、`value`、`beta`、`initial_state` 的 dtype 仅支持 bfloat16。
- `actual_seq_lengths` 的 dtype 仅支持 int32。
- `g` 的 dtype 仅支持 float32，为 None 时等价于门控因子全零（不衰减）。
- `query` 和 `key` 的 shape 必须一致（[T, Nk, Dk]）。
- `out` 的 shape 与 `value` 一致，`final_state` 的 shape 与 `initial_state` 一致。
- `actual_seq_lengths` 中各元素之和必须等于 T。
- 算子输入不支持有 ±inf 和 nan 的情况。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import chunk_gated_delta_rule

    torch_npu.npu.set_device(0)

    B = 1
    T = 64
    Nk = 4
    Nv = 4
    Dk = 128
    Dv = 128

    q = torch.rand((T, Nk, Dk), dtype=torch.bfloat16, device="npu")
    k = torch.rand((T, Nk, Dk), dtype=torch.bfloat16, device="npu")
    v = torch.rand((T, Nv, Dv), dtype=torch.bfloat16, device="npu")
    beta = torch.rand((T, Nv), dtype=torch.bfloat16, device="npu")
    g = torch.rand((T, Nv), dtype=torch.float32, device="npu") * -1.0
    initial_state = torch.rand((B, Nv, Dv, Dk), dtype=torch.bfloat16, device="npu")
    actual_seq_lengths = torch.tensor([T], dtype=torch.int32, device="npu")
    scale = 1.0 / (Dk ** 0.5)

    out, final_state = chunk_gated_delta_rule(
        q, k, v, beta, initial_state, actual_seq_lengths,
        g=g,
        scale=scale,
    )
    ```

- 图模式调用

    通过 `torch.compile(backend="atc")` 或 `torchair` 自动将算子转换为 GE 图算子，无需额外配置。

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import chunk_gated_delta_rule

    torch_npu.npu.set_device(0)

    B = 1
    T = 64
    Nk = 4
    Nv = 4
    Dk = 128
    Dv = 128

    class ChunkGatedDeltaRuleModel(torch.nn.Module):
        def forward(self, q, k, v, beta, initial_state, actual_seq_lengths, g):
            out, final_state = chunk_gated_delta_rule(
                q, k, v, beta, initial_state, actual_seq_lengths,
                g=g,
                scale=1.0 / (Dk ** 0.5),
            )
            return out, final_state

    model = ChunkGatedDeltaRuleModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    q = torch.rand((T, Nk, Dk), dtype=torch.bfloat16, device="npu")
    k = torch.rand((T, Nk, Dk), dtype=torch.bfloat16, device="npu")
    v = torch.rand((T, Nv, Dv), dtype=torch.bfloat16, device="npu")
    beta = torch.rand((T, Nv), dtype=torch.bfloat16, device="npu")
    g = torch.rand((T, Nv), dtype=torch.float32, device="npu") * -1.0
    initial_state = torch.rand((B, Nv, Dv, Dk), dtype=torch.bfloat16, device="npu")
    actual_seq_lengths = torch.tensor([T], dtype=torch.int32, device="npu")

    output = model(q, k, v, beta, initial_state, actual_seq_lengths, g)
    ```
