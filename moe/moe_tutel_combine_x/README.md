# MoeTutelCombineX

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 算子功能：MoE（Tutel）combine阶段的token侧聚合。按路由表`indices`、`locations`从专家输出缓冲
  `yGrad`中为每个token取出至多K行，乘以门控权重`gates`后沿top-k轴求和，得到每个token的结果
  `xGrad`。专家序号或容量槽位落在合法范围外时，该路由贡献0，且不会读取`yGrad`。
- 计算公式：

  记E为专家数、C为每专家容量、H为隐藏维、K为top-k、S为token数。

  $$
  valid_{k,s} = (0 \le indices_{k,s} < E) \wedge (0 \le locations_{k,s} < C)
  $$

  $$
  xGrad_{s,:} = \sum_{k=0}^{K-1}
  \begin{cases}
  gates_{k,s} \cdot yGrad_{\,indices_{k,s},\ locations_{k,s},\ :} & valid_{k,s} \\
  0 & \text{其他}
  \end{cases}
  $$

  按K轴下标从0到K-1依次累加，累加初值为0。因此K为0、E为0、C为0、或某个token的全部
  路由都无效时，`xGrad`对应行为全0。

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
    <td>每个token在每一路top-k上命中的专家序号，对应公式中的indices。shape要求与gates一致，维度为[K,S]。取值不在[0,E)时，该路由贡献为0。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>locations</td>
    <td>输入</td>
    <td>每个token在所命中专家内的容量槽位，对应公式中的locations。shape要求与gates一致，维度为[K,S]。取值不在[0,C)时，该路由贡献为0。</td>
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

### 产品差异说明

- <term>Ascend 950PR/Ascend 950DT</term>：由本仓的arch35实现提供。本文后续的固定rank、四侧安全边界、
  空Tensor、完整写回和FLOAT32累加约束均适用于该产品。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：由CANN基础算子包中的既有实现提供。本次新增的
  Host和设备源仅在包含Ascend950的构建中纳入，不进入纯A2构建；公共Graph IR沿用基础包既有原型，本次不新增
  或修改`op_graph`原型。A2既有实现不保证
  `indices >= E`或`locations < 0`输入的安全性，也未纳入本次空Tensor与精度验收；跨产品调用时不要依赖
  这些边界行为。
- Kernel通过模板参数派发，并把两个8 bit参数编码为既有raw TilingKey `0x2710`（十进制10000）；
  不使用`TILING_KEY_IS`硬编码分支。

## 约束说明

- `gates`、`indices`、`locations` 三者的shape必须完全一致，均为[K,S]；K取自`gates`的第0维。
- `y_grad`、`gates`、`x_grad` 的数据类型必须完全一致；`indices`、`locations` 恒为INT32。
- <term>Ascend 950PR/Ascend 950DT</term>上的有效路由必须同时满足`0 <= indices < E`与
  `0 <= locations < C`。任一条件不满足时贡献为0，kernel不会读取对应的`y_grad`行。
- 输出 `x_grad` 每次调用都会被完整写入，不依赖调用方预先清零。
- K、E或C为0且输出非空时，输出为全0；S或H为0时输出是shape为[S,H]的空Tensor。
- 不支持非连续Tensor；`y_grad` 的rank固定为3，`gates`/`indices`/`locations` 的rank固定为2，不支持
  UNKNOWN_RANK（-2）。所有维度必须非负；元素数、字节数或寻址乘法超出INT64可表示范围时，host在下发
  kernel前返回失败。Ascend950上由S/H派生的DataCopy GM侧stride超过`2^40-1`字节时也在下发前失败。
- 精度：三种浮点类型的乘法和跨top-k累加均按K轴顺序在FLOAT32上完成，末尾只转回一次输出数据类型。
  有效路由上的Inf/NaN按IEEE浮点运算自然传播，例如`0 * Inf`可产生NaN；无效路由不参与乘法。
- 本算子不提供aclnn单算子接口，仅支持GE图模式调用。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------- | ------------------ | -------------------------------------------- |
| 图模式调用 | [test_geir_moe_tutel_combine_x](./examples/test_geir_moe_tutel_combine_x.cpp) | 通过[公共算子IR](../../common/include/op_graph/op_transformer_proto_extend.h)构图方式调用MoeTutelCombineX算子。 |
