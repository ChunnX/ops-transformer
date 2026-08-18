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

  `cann_ops_transformer.block_attn_res_update`将当前`delta`原地累加到FP32类型的`partial_block`，计算当前partial block的RMSNorm score，并与Phase 1产生的online softmax中间状态合并，返回当前层结果`h`。更新后的partial block直接写回输入`partial_block`。底层调用`aclnnBlockAttnResUpdate`接口完成计算。

- **计算公式**：

  设`T`表示展平后的token数，`D`表示hidden size。主要计算过程如下：

  1. 更新当前partial block：

     $$
     p[t,d] = partial\_block[t,d] + \operatorname{FP32}(delta[t,d])
     $$

  2. 计算RMSNorm分母和当前score：

     $$
     r[t] = \sqrt{\frac{1}{D}\sum_{d=0}^{D-1}p[t,d]^2 + eps}
     $$

     $$
     score[t] = \frac{\sum_{d=0}^{D-1}p[t,d] \times pseudo\_query[d]}{r[t]}
     $$

  3. 将当前score与历史online softmax状态合并：

     $$
     m[t] = \max(logit\_max[t], score[t])
     $$

     $$
     alpha[t] = \exp(logit\_max[t] - m[t]), \qquad beta[t] = \exp(score[t] - m[t])
     $$

     $$
     ell[t] = exp\_sum[t] \times alpha[t] + beta[t]
     $$

     $$
     updated\_numerator[t,d] = numerator[t,d] \times alpha[t] + p[t,d] \times beta[t]
     $$

  4. 生成输出：

     $$
     partial\_block[t,d] \leftarrow p[t,d]
     $$

     $$
     h[t,d] = \operatorname{Cast}_{delta.dtype}\left(\frac{updated\_numerator[t,d]}{ell[t]}\right)
     $$

## 函数原型

```python
cann_ops_transformer.block_attn_res_update(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    eps,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `partial_block` | Tensor | 必选 | 当前已累计的partial block。必须为NPU上的连续Tensor，调用完成后被原地更新。 | `torch.float32` | `(T, D)` |
| `delta` | Tensor | 必选 | 本次需要累加的partial增量。必须为NPU上的连续Tensor，shape必须与`partial_block`一致。 | `torch.bfloat16` | `(T, D)` |
| `pseudo_query` | Tensor | 必选 | 算子外部预计算的伪query。必须为NPU上的连续Tensor，长度必须等于`D`。 | `torch.float32` | `(D,)` |
| `numerator` | Tensor | 必选 | Phase 1输出的历史online softmax加权和。必须为NPU上的连续Tensor，shape必须与`partial_block`一致。 | `torch.float32` | `(T, D)` |
| `logit_max` | Tensor | 必选 | Phase 1输出的历史最大logit。必须为NPU上的连续Tensor，长度必须等于`T`。 | `torch.float32` | `(T,)` |
| `exp_sum` | Tensor | 必选 | Phase 1输出的历史softmax分母累积值。必须为NPU上的连续Tensor，长度必须等于`T`。 | `torch.float32` | `(T,)` |
| `eps` | float | 必选 | RMSNorm计算中的FP32数值稳定项，必须为有限正数。 | float | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `h` | Tensor | 必选 | 当前层BlockAttnRes输出。数据类型和shape与`delta`一致，为连续Tensor。 | `torch.bfloat16` | `(T, D)` |

## 约束说明

- 适用场景：该接口当前用于前向推理场景。
- 调用方式：该接口支持PyTorch eager单算子模式调用，不支持CPU、CUDA和Autograd反向。
- 6个Tensor输入必须位于同一NPU device，且均为连续Tensor。
- 输出`h`为连续Tensor，且位于输入所在的NPU device。
- token维`T`满足`T >= 0`，hidden维`D`满足`0 <= D <= 8192`。
- 当`T == 0`或`D == 0`时支持空Tensor快速返回。所有输入仍须满足正常的rank和shape关系，
  dtype、连续性、同device和`eps`约束也继续生效；ACLNN L0返回shape与`delta`一致的空`h`，
  `partial_block`保持不变，不进入InferShape、tiling和kernel。
- `partial_block`、`delta`和`numerator`的shape必须一致，均为`(T, D)`。
- `pseudo_query`的shape必须为`(D,)`，`logit_max`和`exp_sum`的shape必须为`(T,)`。
- `eps`必须为有限正数。
- 数据类型组合必须满足：

  | partial_block | delta | pseudo_query | numerator | logit_max | exp_sum | h |
  | --- | --- | --- | --- | --- | --- | --- |
  | `torch.float32` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.bfloat16` |

- Torch 对外接口为原地语义：C++ bridge 直接将输入`partial_block`传给底层ACLNN，调用完成后
  `partial_block`保存累加结果，不会额外复制或返回该Tensor；接口仅返回`h`。

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
  partial_block_ptr = partial_block.data_ptr()
  delta = torch.randn(
      (T, D), dtype=torch.bfloat16, device="npu:0"
  )
  expected_partial_block = partial_block + delta.float()
  pseudo_query = torch.randn(
      (D,), dtype=torch.float32, device="npu:0"
  )
  numerator = torch.randn(
      (T, D), dtype=torch.float32, device="npu:0"
  )
  logit_max = torch.randn(
      (T,), dtype=torch.float32, device="npu:0"
  )
  exp_sum = torch.rand(
      (T,), dtype=torch.float32, device="npu:0"
  )
  eps = 1.0e-5

  h = cann_ops_transformer.block_attn_res_update(
      partial_block,
      delta,
      pseudo_query,
      numerator,
      logit_max,
      exp_sum,
      eps,
  )

  assert h.is_contiguous()
  assert partial_block.is_contiguous()
  assert partial_block.data_ptr() == partial_block_ptr
  torch.testing.assert_close(partial_block, expected_partial_block)
  ```
