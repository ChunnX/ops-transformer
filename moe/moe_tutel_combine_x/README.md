# MoeTutelCombineX

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 算子功能：MoE（Tutel）combine 阶段的 token 侧聚合。按路由表 `indices`、`locations` 从专家输出缓冲
  `yGrad` 中为每个 token 取出至多 K 行，乘以门控权重 `gates` 后沿 top-k 轴求和，得到每个 token 的结果
  `xGrad`。路由无效的项贡献 0：`indices` 小于 0 表示该 token 在这一路没有分到专家，`locations`
  大于等于容量 C 表示该 token 在这一路被丢弃（drop）。
- 计算公式：

  记 E 为专家数、C 为每专家容量、H 为隐藏维、K 为 top-k、S 为 token 数。

  $$
  valid_{k,s} = (indices_{k,s} \ge 0) \wedge (locations_{k,s} < C)
  $$

  $$
  xGrad_{s,:} = \sum_{k=0}^{K-1}
  \begin{cases}
  gates_{k,s} \cdot yGrad_{\,indices_{k,s},\ locations_{k,s},\ :} & valid_{k,s} \\
  0 & \text{其他}
  \end{cases}
  $$

  累加从 0 开始，因此 K 为 0、或某个 token 的全部路由都无效时，`xGrad` 对应行为全 0。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 130px">
  <col style="width: 130px">
  <col style="width: 415px">
  <col style="width: 230px">
  <col style="width: 100px">
  </colgroup>
  <thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>y_grad</td>
    <td>输入</td>
    <td>专家输出缓冲，对应公式中的yGrad。要求是一个3D的Tensor，维度为[E,C,H]，E为专家数，C为每专家容量，H为隐藏维。</td>
    <td>FLOAT32、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gates</td>
    <td>输入</td>
    <td>门控权重，对应公式中的gates。要求是一个2D的Tensor，维度为[K,S]，K为top-k，S为token数。数据类型与y_grad保持一致。</td>
    <td>FLOAT32、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>indices</td>
    <td>输入</td>
    <td>每个token在每一路top-k上命中的专家序号，对应公式中的indices。shape要求与gates一致，维度为[K,S]。取值小于0表示该路无专家，其贡献为0。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>locations</td>
    <td>输入</td>
    <td>每个token在所命中专家内的容量槽位，对应公式中的locations。shape要求与gates一致，维度为[K,S]。取值大于等于C表示该token在这一路被丢弃，其贡献为0。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_grad</td>
    <td>输出</td>
    <td>token侧的聚合结果，对应公式中的xGrad。维度为[S,H]，数据类型与y_grad保持一致。</td>
    <td>FLOAT32、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- `gates`、`indices`、`locations` 三者的shape必须完全一致，均为[K,S]；K取自`gates`的第0维。
- `y_grad`、`gates`、`x_grad` 的数据类型必须完全一致；`indices`、`locations` 恒为INT32。
- 有效性判据只包含 `indices >= 0` 与 `locations < C` 两条。`indices` 取值超出[0, E)、或 `locations`
  取值为负，均不在校验范围内，由调用方保证；违反时行为未定义。
- 输出 `x_grad` 每次调用都会被完整写入，不依赖调用方预先清零。
- 不支持非连续Tensor；`y_grad` 的rank固定为3，`gates`/`indices`/`locations` 的rank固定为2，不支持
  UNKNOWN_RANK（-2）。
- 精度：FLOAT16、BFLOAT16 输入的乘加与跨top-k累加全程在FLOAT32上完成，末尾一次性转回输出数据类型。
- 本算子不提供aclnn单算子接口，仅支持GE图模式调用。
- <term>Ascend 950PR/Ascend 950DT</term> 的实现由本仓提供；<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
  与 <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 的实现由CANN基础算子包提供，本仓不涉及。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------- | ------------------ | -------------------------------------------- |
| 图模式调用 | [test_geir_moe_tutel_combine_x](./examples/test_geir_moe_tutel_combine_x.cpp) | 通过[算子IR](./op_graph/moe_tutel_combine_x_proto.h)构图方式调用MoeTutelCombineX算子。 |
