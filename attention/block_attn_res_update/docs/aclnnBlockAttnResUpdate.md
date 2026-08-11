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

- 接口功能：BlockAttnRes Phase 2 将当前已累计的partial block与本次partial增量相加，计算当前partial block的RMSNorm score，并与Phase 1产生的online softmax中间状态合并，输出当前层结果`h`和更新后的`partialBlocks`。

- 计算公式：

  设`T`表示展平后的token数，`D`表示hidden size。对每个token `t`，首先更新partial block：

  $$
  p[t,d] = partialBlock[t,d] + \operatorname{FP32}(partialDelta[t,d])
  $$

  计算RMSNorm分母和当前partial block的score：

  $$
  r[t] = \sqrt{\frac{1}{D}\sum_{d=0}^{D-1}p[t,d]^2 + epsilon}
  $$

  $$
  score[t] = \frac{\sum_{d=0}^{D-1}p[t,d] \times effectiveQuery[d]}{r[t]}
  $$

  将当前score与历史online softmax状态合并：

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

  输出为：

  $$
  partialBlocks[t,d] = p[t,d]
  $$

  $$
  h[t,d] = \operatorname{Cast}_{partialDelta.dtype}\left(\frac{numerator[t,d]}{ell[t]}\right)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBlockAttnResUpdateGetWorkspaceSize”接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用“aclnnBlockAttnResUpdate”接口执行计算。

```cpp
aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(
  const aclTensor *partialBlock,
  const aclTensor *partialDelta,
  const aclTensor *effectiveQuery,
  const aclTensor *interMax,
  const aclTensor *interExpSum,
  const aclTensor *interNumerator,
  float            epsilon,
  const aclTensor *h,
  const aclTensor *partialBlocks,
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

  <table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
  <col style="width: 190px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 190px">
  <col style="width: 100px">
  <col style="width: 170px">
  <col style="width: 147px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th style="white-space: nowrap">输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th style="white-space: nowrap">维度(shape)</th>
      <th>非连续的Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>partialBlock（const aclTensor *）</td>
      <td>输入</td>
      <td>表示当前已累计的partial block。</td>
      <td>必选参数。必须为连续Tensor。hidden维D必须在[1, 8192]范围内。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2，shape为[T, D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>partialDelta（const aclTensor *）</td>
      <td>输入</td>
      <td>表示本次需要累加的partial增量。</td>
      <td>必选参数。必须为连续Tensor，shape必须与partialBlock一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2，shape为[T, D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>effectiveQuery（const aclTensor *）</td>
      <td>输入</td>
      <td>表示算子外部预计算的有效query。</td>
      <td>必选参数。必须为连续Tensor，长度必须等于D。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1，shape为[D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>interMax（const aclTensor *）</td>
      <td>输入</td>
      <td>表示Phase 1输出的历史最大logit。</td>
      <td>必选参数。必须为连续Tensor，长度必须等于T。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1，shape为[T]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>interExpSum（const aclTensor *）</td>
      <td>输入</td>
      <td>表示Phase 1输出的历史softmax分母累积值。</td>
      <td>必选参数。必须为连续Tensor，长度必须等于T。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1，shape为[T]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>interNumerator（const aclTensor *）</td>
      <td>输入</td>
      <td>表示Phase 1输出的历史online softmax加权和。</td>
      <td>必选参数。必须为连续Tensor，shape必须与partialBlock一致。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2，shape为[T, D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>epsilon（float）</td>
      <td>输入</td>
      <td>表示RMSNorm计算中的数值稳定项。</td>
      <td>必选属性，C++类型为float（FP32）。必须为有限正数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>h（const aclTensor *）</td>
      <td>输出</td>
      <td>表示当前层BlockAttnRes输出。</td>
      <td>必选参数。必须为连续Tensor，数据类型和shape必须与partialDelta一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2，shape为[T, D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>partialBlocks（const aclTensor *）</td>
      <td>输出</td>
      <td>表示更新后的partial block。</td>
      <td>必选参数。必须为连续Tensor，数据类型和shape必须与partialBlock一致。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2，shape为[T, D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t *）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor **）</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：支持FLOAT16和BFLOAT16两种partialDelta/h数据类型组合；其余Tensor输入和partialBlocks仅支持FLOAT。
  <!-- end id7 -->

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必选参数为空指针，例如partialBlock、partialDelta、effectiveQuery、interMax、interExpSum、interNumerator、h、partialBlocks、workspaceSize或executor为空。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>输入或输出的数据类型、数据格式不在支持范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的rank、shape关系不满足约束，或者D不在[1, 8192]范围内。</td>
    </tr>
    <tr>
      <td>任一Tensor输入或输出为非连续Tensor。</td>
    </tr>
    <tr>
      <td>epsilon不是有限正数。</td>
    </tr>
    <tr>
      <td>h与partialDelta的数据类型不一致。</td>
    </tr>
  </tbody>
  </table>

## aclnnBlockAttnResUpdate

- **参数说明**

  <table style="undefined;table-layout: fixed;width: 795px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 505px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockAttnResUpdateGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>acl stream流。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 仅支持ND数据格式，不支持私有格式。
- 所有Tensor输入和输出均必须为连续Tensor。
- hidden维`D`必须满足`1 <= D <= 8192`。
- shape关系必须满足：
  - `partialBlock`、`partialDelta`、`interNumerator`、`h`和`partialBlocks`的shape均为`[T, D]`且相互一致。
  - `effectiveQuery`的shape为`[D]`。
  - `interMax`和`interExpSum`的shape均为`[T]`。
- 数据类型组合必须满足：

  | partialBlock | partialDelta | effectiveQuery | interMax | interExpSum | interNumerator | h | partialBlocks |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | FLOAT | FLOAT16 | FLOAT | FLOAT | FLOAT | FLOAT | FLOAT16 | FLOAT |
  | FLOAT | BFLOAT16 | FLOAT | FLOAT | FLOAT | FLOAT | BFLOAT16 | FLOAT |

- `epsilon`必须为有限正数。
- 当`T=0`且`D`满足取值范围时，支持空Tensor场景；第一段接口返回ACLNN_SUCCESS，workspaceSize为0。
- 当前接口不定义`partialBlock`和`partialBlocks`之间的输入输出别名或原地覆盖语义。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
```
