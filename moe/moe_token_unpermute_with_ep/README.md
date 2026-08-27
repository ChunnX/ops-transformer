# MoeTokenUnpermuteWithEp

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：根据sortedIndices存储的下标位置，去获取permutedTokens中的输入数据与probs相乘，并进行合并累加。
- 计算公式：设EP有效区间为$[s,e)$，$L$为`sortedIndices`的元素个数；fallback模式下取$s=0,e=L$。第$q$个展开位置的中间量定义如下。

$$
x_q =
\begin{cases}
permutedTokens_{sortedIndices_q-s}, & s \le sortedIndices_q < e \\
0, & \text{otherwise}
\end{cases}
$$

当`probsOptional`非空时，设其shape为$(N,K)$，满足$L=N\times K$，则：

$$
out_i = \sum_{j=0}^{K-1}x_{iK+j}\times probsOptional_{i,j}, \quad 0 \le i < N
$$

当`probsOptional`为空时，非fallback模式下$K=numTopk$且$N=L/K$；fallback模式下有效$K=1$且$N=L$：

$$
out_i = \sum_{j=0}^{K-1}x_{iK+j}, \quad 0 \le i < N
$$

## 参数说明

  <table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>permutedTokens</td>
      <td>输入</td>
      <td>表示经过扩展并排序过的tokens，对应公式中的`permutedTokens`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>输入</td>
      <td>表示需要计算的数据在permutedTokens中的位置，对应公式中的`sortedIndices`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>可选输入</td>
      <td>表示输入tokens对应的专家概率，对应公式中的`probs`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numTopk</td>
      <td>属性</td>
      <td>被选中的专家个数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rangeOptional</td>
      <td>属性</td>
      <td>ep切分的有效范围，size为2。</td>
      <td>aclIntArray*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>属性</td>
      <td>目前仅支持false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional</td>
      <td>属性</td>
      <td>目前仅支持nullptr或空数组。</td>
      <td>aclIntArray*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示permutedTokens反重排的输出结果，对应公式中的`out`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

- 非fallback模式下，numTopk取值范围为[1, 512]；fallback模式下忽略numTopk。
- 不支持paddedMode为`True`。
- 当rangeOptional为nullptr、空数组或`[0, 0]`时进入fallback模式，要求permutedTokens第0维等于sortedIndices长度。
- 非fallback模式下，rangeOptional必须满足`0 <= rangeOptional[0] < rangeOptional[1]`，且permutedTokens第0维等于`rangeOptional[1] - rangeOptional[0]`。
- restoreShapeOptional仅支持nullptr或空数组。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_token_unpermute_with_ep](examples/test_aclnn_moe_token_unpermute_with_ep.cpp) | 通过[aclnnMoeTokenUnpermuteWithEp](docs/aclnnMoeTokenUnpermuteWithEp.md)接口方式调用MoeTokenUnpermuteWithEp算子。 |
