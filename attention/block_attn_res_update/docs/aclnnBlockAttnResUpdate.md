# aclnnBlockAttnResUpdate

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/block_attn_res_update)

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

- 接口功能：BlockAttnRes Phase 2 将当前 partial block 与本次增量相加，计算当前 partial block 的
  RMSNorm score，并与 Phase 1 产生的 online softmax 中间状态合并，输出当前层结果 `h`。
  更新后的 partial block 原地写回 `partialBlock`。

- 计算公式：

  设 `T` 表示展平后的 token 数，`D` 表示 hidden size：

  $$
  p[t,d] = partialBlock[t,d] + \operatorname{FP32}(delta[t,d])
  $$

  $$
  r[t] = \sqrt{\frac{1}{D}\sum_{d=0}^{D-1}p[t,d]^2 + eps}
  $$

  $$
  score[t] = \frac{\sum_{d=0}^{D-1}p[t,d] \times effectiveQuery[d]}{r[t]}
  $$

  $$
  m[t] = \max(interMax[t], score[t])
  $$

  $$
  alpha[t] = \exp(interMax[t] - m[t]), \qquad beta[t] = \exp(score[t] - m[t])
  $$

  $$
  ell[t] = interExpSum[t] \times alpha[t] + beta[t]
  $$

  $$
  numerator[t,d] = interNumerator[t,d] \times alpha[t] + p[t,d] \times beta[t]
  $$

  $$
  partialBlock[t,d] = p[t,d]
  $$

  $$
  h[t,d] = \operatorname{BF16}\left(\frac{numerator[t,d]}{ell[t]}\right)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)。必须先调用
`aclnnBlockAttnResUpdateGetWorkspaceSize` 获取 workspace 大小和执行器，再调用
`aclnnBlockAttnResUpdate` 执行计算。

```cpp
aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(
    aclTensor       *partialBlock,
    const aclTensor *delta,
    const aclTensor *interNumerator,
    const aclTensor *interMax,
    const aclTensor *interExpSum,
    const aclTensor *effectiveQuery,
    float            eps,
    aclTensor       *h,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnBlockAttnResUpdate(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnBlockAttnResUpdateGetWorkspaceSize

- **参数说明**

  | 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 非连续的Tensor |
  | --- | --- | --- | --- | --- | --- | --- |
  | `partialBlock`（`aclTensor *`） | 输入/输出 | 当前已累计的 partial block。计算结果原地写回该 Tensor。必须连续，`D` 必须在 `[1, 8192]` 内。 | FLOAT | ND | `[T, D]` | × |
  | `delta`（`const aclTensor *`） | 输入 | 本次需要累加的 partial 增量。必须连续，shape 必须与 `partialBlock` 一致。 | BFLOAT16 | ND | `[T, D]` | × |
  | `interNumerator`（`const aclTensor *`） | 输入 | Phase 1 输出的历史 online softmax 加权和。必须连续，shape 必须与 `partialBlock` 一致。 | FLOAT | ND | `[T, D]` | × |
  | `interMax`（`const aclTensor *`） | 输入 | Phase 1 输出的历史最大 logit。必须连续，长度必须等于 `T`。 | FLOAT | ND | `[T]` | × |
  | `interExpSum`（`const aclTensor *`） | 输入 | Phase 1 输出的历史 softmax 分母累积值。必须连续，长度必须等于 `T`。 | FLOAT | ND | `[T]` | × |
  | `effectiveQuery`（`const aclTensor *`） | 输入 | 算子外部预计算的有效 query。必须连续，长度必须等于 `D`。 | FLOAT | ND | `[D]` | × |
  | `eps`（`float`） | 输入 | RMSNorm 数值稳定项。C++ 类型为 `float`（FP32），必须为有限正数。 | - | - | - | - |
  | `h`（`aclTensor *`） | 输出 | 当前层 BlockAttnRes 输出。必须连续，shape 必须与 `partialBlock` 一致。 | BFLOAT16 | ND | `[T, D]` | × |
  | `workspaceSize`（`uint64_t *`） | 输出 | 返回 Device 侧需要申请的 workspace 大小。 | - | - | - | - |
  | `executor`（`aclOpExecutor **`） | 输出 | 返回包含算子计算流程的执行器。 | - | - | - | - |

  `partialBlock` 虽未使用 `partialBlockRef` 命名，但它是引用参数：接口通过非 `const` 指针表达可写性，
  并将同一个 Tensor 同时作为底层算子的输入和第一个输出。调用完成后，其原数据被更新。

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：`delta` 和 `h` 仅支持 BFLOAT16；其余 Tensor 仅支持 FLOAT。
  <!-- end id7 -->

- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  | 返回码 | 错误码 | 描述 |
  | --- | --- | --- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | `partialBlock`、`delta`、`interNumerator`、`interMax`、`interExpSum`、`effectiveQuery`、`h`、`workspaceSize` 或 `executor` 为空。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | 输入或输出的 dtype、ND format、rank 或 shape 关系不满足约束。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `T <= 0`，或者 `D` 不在 `[1, 8192]` 范围内。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | 任一 Tensor 输入或输出为非连续 Tensor。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `eps` 不是有限正数。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | 当前平台不是 Ascend 950PR/Ascend 950DT。 |

## aclnnBlockAttnResUpdate

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  | --- | --- | --- |
  | `workspace` | 输入 | Device 侧 workspace 地址。 |
  | `workspaceSize` | 输入 | Device 侧 workspace 大小，由第一段接口获取。 |
  | `executor` | 输入 | 包含算子计算流程的执行器。 |
  | `stream` | 输入 | ACL stream。 |

- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 仅支持 ND 数据格式。
- 所有 Tensor 输入和输出必须连续。
- `partialBlock`、`delta`、`interNumerator` 和 `h` 的 shape 必须均为 `[T, D]`。
- `interMax` 和 `interExpSum` 的 shape 必须为 `[T]`；`effectiveQuery` 的 shape 必须为 `[D]`。
- `T > 0`，`1 <= D <= 8192`，不支持空 Tensor。
- `partialBlock`、`interNumerator`、`interMax`、`interExpSum`、`effectiveQuery` 为 FLOAT；`delta` 和 `h` 为 BFLOAT16。
- `eps` 必须为有限正数。图原型的默认值为 `1e-6F`，ACLNN C 接口仍要求调用方显式传入。
- `partialBlock` 为原地更新参数。执行过程中输入槽和第一个输出槽使用相同的 Tensor/GM 地址。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考
[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
```
