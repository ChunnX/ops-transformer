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
- 计算公式：

设numTopk为$K$，原始token数为$N$，隐藏维度为$H$，rangeOptional为$[r_0,r_1)$，且$s_{i,j}=sortedIndices_{iK+j}$，则：

$$
x_{i,j,h}=\begin{cases}
permutedTokens_{s_{i,j}-r_0,h}, & r_0\le s_{i,j}<r_1 \\
0, & \text{其他情况}
\end{cases}
$$

$$
out_{i,h}=\begin{cases}
\sum_{j=0}^{K-1}x_{i,j,h}\times probsOptional_{i,j}, & probsOptional\text{不为空} \\
\sum_{j=0}^{K-1}x_{i,j,h}, & probsOptional\text{为空}
\end{cases}
$$

## 参数说明

  <table style="table-layout: fixed; width: 1576px">
  <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
  </colgroup>
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
      <td>表示输入tokens对应的专家概率，对应公式中的`probsOptional`。</td>
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
      <td>目前仅支持nullptr。</td>
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

- numTopk必须大于等于1；probsOptional非空时，numTopk必须小于等于512。
- 不支持Broadcast。
- 不支持paddedMode为`True`。
- 当rangeOptional为空时，使用默认值{0,0}，输出为全0，不会回退调用其他算子。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_token_unpermute_with_ep](examples/test_aclnn_moe_token_unpermute_with_ep.cpp) | 通过[aclnnMoeTokenUnpermuteWithEp](docs/aclnnMoeTokenUnpermuteWithEp.md)接口方式调用MoeTokenUnpermuteWithEp算子。 |
