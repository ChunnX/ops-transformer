# block_attn_res_update

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

- **接口功能**：

  `cann_ops_transformer.block_attn_res_update`将当前`partial_delta`累加到FP32类型的`partial_block`，计算当前partial block的RMSNorm score，并与Phase 1产生的online softmax中间状态合并，输出当前层结果`h`和更新后的`partial_blocks`。底层调用`aclnnBlockAttnResUpdate`接口完成计算。

- **计算公式**：

  设`T`表示展平后的token数，`D`表示hidden size。主要计算过程如下：

  1. 更新当前partial block：

     $$
     p[t,d] = partial\_block[t,d] + \operatorname{FP32}(partial\_delta[t,d])
     $$

  2. 计算RMSNorm分母和当前score：

     $$
     r[t] = \sqrt{\frac{1}{D}\sum_{d=0}^{D-1}p[t,d]^2 + epsilon}
     $$

     $$
     score[t] = \frac{\sum_{d=0}^{D-1}p[t,d] \times effective\_query[d]}{r[t]}
     $$

  3. 将当前score与历史online softmax状态合并：

     $$
     m[t] = \max(inter\_max[t], score[t])
     $$

     $$
     alpha[t] = \exp(inter\_max[t] - m[t]), \qquad beta[t] = \exp(score[t] - m[t])
     $$

     $$
     ell[t] = inter\_exp\_sum[t] \times alpha[t] + beta[t]
     $$

     $$
     numerator[t,d] = inter\_numerator[t,d] \times alpha[t] + p[t,d] \times beta[t]
     $$

  4. 生成输出：

     $$
     partial\_blocks[t,d] = p[t,d]
     $$

     $$
     h[t,d] = \operatorname{Cast}_{partial\_delta.dtype}\left(\frac{numerator[t,d]}{ell[t]}\right)
     $$

## 函数原型

```python
cann_ops_transformer.block_attn_res_update(
    partial_block,
    partial_delta,
    inter_numerator,
    inter_max,
    inter_exp_sum,
    effective_query,
    epsilon,
) -> (Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `partial_block` | Tensor | 必选 | 当前已累计的partial block。必须为NPU上的连续Tensor。 | `torch.float32` | `(T, D)` |
| `partial_delta` | Tensor | 必选 | 本次需要累加的partial增量。必须为NPU上的连续Tensor，shape必须与`partial_block`一致。 | `torch.bfloat16` | `(T, D)` |
| `inter_numerator` | Tensor | 必选 | Phase 1输出的历史online softmax加权和。必须为NPU上的连续Tensor，shape必须与`partial_block`一致。 | `torch.float32` | `(T, D)` |
| `inter_max` | Tensor | 必选 | Phase 1输出的历史最大logit。必须为NPU上的连续Tensor，长度必须等于`T`。 | `torch.float32` | `(T,)` |
| `inter_exp_sum` | Tensor | 必选 | Phase 1输出的历史softmax分母累积值。必须为NPU上的连续Tensor，长度必须等于`T`。 | `torch.float32` | `(T,)` |
| `effective_query` | Tensor | 必选 | 算子外部预计算的有效query。必须为NPU上的连续Tensor，长度必须等于`D`。 | `torch.float32` | `(D,)` |
| `epsilon` | float | 必选 | RMSNorm计算中的FP32数值稳定项，必须为有限正数。 | float | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `h` | Tensor | 必选 | 当前层BlockAttnRes输出。数据类型和shape与`partial_delta`一致，为连续Tensor。 | `torch.bfloat16` | `(T, D)` |
| `partial_blocks` | Tensor | 必选 | 更新后的partial block。shape与`partial_block`一致，为连续Tensor。 | `torch.float32` | `(T, D)` |

## 约束说明

- 适用场景：该接口当前用于前向推理场景。
- 调用方式：该接口支持PyTorch eager单算子模式调用，不支持CPU、CUDA和Autograd反向。
- 6个Tensor输入必须位于同一NPU device，且均为连续Tensor。
- 两个输出均为连续Tensor，且位于输入所在的NPU device。
- `T`必须大于0，不支持空Tensor。
- hidden维`D`必须满足`1 <= D <= 8192`。
- `partial_block`、`partial_delta`和`inter_numerator`的shape必须一致，均为`(T, D)`。
- `effective_query`的shape必须为`(D,)`，`inter_max`和`inter_exp_sum`的shape必须为`(T,)`。
- `epsilon`必须为有限正数。
- 数据类型组合必须满足：

  | partial_block | partial_delta | inter_numerator | inter_max | inter_exp_sum | effective_query | h | partial_blocks |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | `torch.float32` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.bfloat16` | `torch.float32` |

- Torch 对外接口保持函数式语义：`partial_block`不会被修改。C++ bridge 会先创建连续副本
  `partial_blocks`，再将该副本作为底层 ACLNN 的原地输入输出参数。

## 确定性计算

当前接口不提供单独的确定性开关，确定性能力继承底层`aclnnBlockAttnResUpdate`实现。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  T = 8
  D = 4096

  partial_block = torch.randn(
      (T, D), dtype=torch.float32, device="npu:0"
  )
  partial_delta = torch.randn(
      (T, D), dtype=torch.bfloat16, device="npu:0"
  )
  effective_query = torch.randn(
      (D,), dtype=torch.float32, device="npu:0"
  )
  inter_max = torch.randn(
      (T,), dtype=torch.float32, device="npu:0"
  )
  inter_exp_sum = torch.rand(
      (T,), dtype=torch.float32, device="npu:0"
  )
  inter_numerator = torch.randn(
      (T, D), dtype=torch.float32, device="npu:0"
  )
  epsilon = 1.0e-5

  h, partial_blocks = cann_ops_transformer.block_attn_res_update(
      partial_block,
      partial_delta,
      inter_numerator,
      inter_max,
      inter_exp_sum,
      effective_query,
      epsilon,
  )

  assert h.is_contiguous()
  assert partial_blocks.is_contiguous()
  ```
