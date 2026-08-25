# block\_attn\_res\_prepare

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- **接口功能**：完成 Attention Residuals 历史残差注意力两阶段计算的第一阶段，一次并行计算全部 $S$ 个目标 slot 的块间注意力，并返回 softmax 加权分子及统计量 $(O,M,L)$，供下一阶段融合当前残差。

- **计算公式**：

    $$
    \operatorname{block\_attn\_res\_prepare}
    \left(V,Q,\operatorname{valid\_blocks};\epsilon\right)
    \longrightarrow (O,M,L).
    $$

    对 token $t$ 的第 $n$ 个历史来源，首先计算 RMS 归一化因子：

    $$
    R_{t,n}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}V_{t,n,d}^{2}+\epsilon}.
    $$

    对每个 slot、token 和有效历史来源计算 logits：

    $$
    Z_{s,t,n}=R_{t,n}^{-1}\sum_{d=0}^{D-1}Q_{s,d}V_{t,n,d}.
    $$

    设 $N_v$ 为 `valid_blocks` 指定的有效历史来源数，则：

    $$
    M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},
    $$

    $$
    E_{s,t,n}=\exp\left(Z_{s,t,n}-M_{s,t}\right),
    $$

    $$
    L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},
    $$

    $$
    O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}.
    $$

## 函数原型

```python
cann_ops_transformer.block_attn_res_prepare(
    block_res,
    valid_blocks,
    pseudo_query,
    *,
    eps=1.0e-6,
) -> Tuple[Tensor, Tensor, Tensor]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| --- | --- | --- | --- | --- | --- |
| block_res | Tensor | 必选 | 历史残差块，对应公式中的 $V$。数据格式为 ND，不支持非连续 Tensor。$T$ 为 token 数，$N$ 为历史来源缓冲区容量，$D$ 为 hidden dimension。 | float32 | [T, N, D] |
| valid_blocks | Tensor | 必选 | 当前有效历史来源数 $N_v$。数据格式为 ND，不支持非连续 Tensor。值小于等于0时按0处理，值大于 $N$ 时按 $N$ 处理。 | int64 | [1] |
| pseudo_query | Tensor | 必选 | 伪查询，对应公式中的 $Q$。数据格式为 ND，不支持非连续 Tensor，最后一维必须与 `block_res` 的最后一维相同。 | float32 | [S, D] |
| eps | float | 可选 | RMS 归一化的数值稳定项，必须为有限正数，默认值为 `1.0e-6`。 | float | - |

## 返回值说明

返回由 `numerator`、`logit_max` 和 `exp_sum` 组成的三元组。

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度（shape） |
| --- | --- | --- | --- | --- |
| numerator | Tensor | softmax 加权分子，对应公式中的 $O$。数据格式为 ND。 | float32 | [S, T, D] |
| logit_max | Tensor | 有效历史来源 logits 的最大值，对应公式中的 $M$。数据格式为 ND。 | float32 | [S, T] |
| exp_sum | Tensor | max-shift 后的指数和，对应公式中的 $L$。数据格式为 ND。 | float32 | [S, T] |

## 约束说明

- 仅支持 Ascend 950PR/Ascend 950DT。
- 支持单算子模式调用，暂不支持 TorchAir 图模式调用。
- `T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- `block_res.shape[2]` 必须等于 `pseudo_query.shape[1]`。
- `block_res`、`pseudo_query` 及三个输出均为 `float32`，`valid_blocks` 为 `int64`。
- 输入不支持非连续 Tensor。
- `T == 0` 或 `S == 0` 时，返回对应 shape 的空输出。

## 确定性计算

- 默认支持确定性计算。

## 调用示例

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    block_res = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 0.0, -1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device="npu",
    )
    valid_blocks = torch.tensor([2], dtype=torch.int64, device="npu")
    pseudo_query = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device="npu",
    )

    # eps 为可选参数；省略时使用默认值 1.0e-6。
    numerator, logit_max, exp_sum = cann_ops_transformer.block_attn_res_prepare(
        block_res,
        valid_blocks,
        pseudo_query,
    )
    torch_npu.npu.synchronize()

    print("numerator:", numerator)
    print("logit_max:", logit_max)
    print("exp_sum:", exp_sum)
    ```

- TorchAir 图模式调用：暂不支持。
