# MoeTutelDispatch

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：MoE（Mixture-of-Experts）Tutel路由方案中的dispatch（encode）阶段。按路由表把token矩阵`x`的整行搬运到「专家 × 容量槽」缓冲`y`中：第`s`个token的第`b`条路由若被判定为有效，就把`x`的第`s`行原样写到`y[indices[b][s]][locations[b][s]][:]`。

  `gates`只作零/非零掩码使用，**不参与乘法**——数值缩放在反向算子MoeTutelCombineX中完成。命中槽位的写入是整行bit级拷贝，不做任何算术运算与类型转换，因此NaN、Inf、负零（-0.0）、非规格化数均原样透传。未被任何路由命中的槽位输出0。

- 计算公式：

  ```python
  x:         [S, H]
  gates:     [B, S]
  indices:   [B, S]
  locations: [B, S]
  y:         [B, capacity, H]
  ```

  $$
  y = 0
  $$

  $$
  y[indices_{b,s}][locations_{b,s}][:] = x[s][:]
  \quad \text{当} \quad 0 \le locations_{b,s} < capacity \ \wedge\ 0 \le indices_{b,s} < B \ \wedge\ gates_{b,s} \neq 0
  $$

  其中 $b \in [0, B)$、$s \in [0, S)$；不满足条件的路由不写出任何字节。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 130px">
  <col style="width: 140px">
  <col style="width: 400px">
  <col style="width: 280px">
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
      <td>x</td>
      <td>输入</td>
      <td>待分发的token矩阵，公式中的x。shape为[S, H]。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gates</td>
      <td>输入</td>
      <td>路由权重，公式中的gates。shape为[B, S]。本算子只判断其是否为零，非零数值不影响结果。</td>
      <td>与x一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>每条路由的目标专家号，公式中的indices。shape为[B, S]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>locations</td>
      <td>输入</td>
      <td>每条路由在目标专家内的容量槽号，公式中的locations。shape为[B, S]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>capacity</td>
      <td>属性</td>
      <td>每个专家的容量槽数，公式中的capacity，取值为非负整数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>分发结果，公式中的y。shape为[B, capacity, H]。</td>
      <td>与x一致</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 参数说明中shape使用的变量说明如下：
  - S：token数量，取值为非负整数。
  - H：隐藏维长度，取值为非负整数，不要求32字节对齐。
  - B：路由条数，同时也是y的专家维长度，取值为非负整数。
- shape关系约束：`gates`、`indices`、`locations`三者shape必须完全相同；`x`的第0维必须等于`gates`的第1维；`y`的shape必须为`[gates.shape[0], capacity, x.shape[1]]`。
- 数据类型约束：`x`、`gates`、`y`必须为同一种数据类型；`indices`、`locations`固定为INT32。
- 输入值域约束：`indices`的取值必须落在`[0, B)`，`locations`的取值必须落在`[0, capacity)`。超出范围的路由被跳过，不写出任何字节，也不报错。
- 唯一性约束：所有被判定为有效的`(indices[b][s], locations[b][s])`二元组必须全局唯一。若多条路由指向同一个`(专家, 槽位)`，写入顺序不确定，结果不保证。
- 未命中槽位：未被任何有效路由写入的`y`位置输出0。
- 精度：命中槽位为整行bit级拷贝，输出与输入逐bit严格相等（bit-exact，rtol=atol=0）。
- 不支持非连续Tensor，不支持动态rank（`-2`）。

## 调用说明

本算子**不提供aclnn对外接口**（构建配置为`ACLNNTYPE aclnn_exclude`），仅支持图模式调用。

| 调用方式   | 调用样例                                                                        | 说明                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 图模式调用 | [test_geir_moe_tutel_dispatch](./examples/test_geir_moe_tutel_dispatch.cpp) | 通过[算子IR](../../common/include/op_graph/op_transformer_proto_extend.h)构图方式调用MoeTutelDispatch算子。 |
