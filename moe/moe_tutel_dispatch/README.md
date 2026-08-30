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

- 算子功能：MoE（Mixture-of-Experts）Tutel路由方案中的dispatch（encode）阶段。按路由表把token矩阵`x`的整行搬运到「专家 × 容量槽」缓冲`y`中：第`s`个token的第`k`条路由若被判定为有效，就把`x`的第`s`行原样写到`y[indices[k][s]][locations[k][s]][:]`。

  `gates`只作零/非零掩码使用，**不参与乘法**——数值缩放在反向算子MoeTutelCombineX中完成。命中槽位的写入是整行bit级拷贝，不做任何算术运算与类型转换，因此NaN、Inf、负零（-0.0）、非规格化数均原样透传。

- Ascend 950计算公式：以下公式和输出初始化行为仅描述Ascend 950PR/Ascend 950DT实现，不反向修改Atlas A2既有产品语义。

  ```python
  x:         [S, H]
  gates:     [K, S]
  indices:   [K, S]
  locations: [K, S]
  y:         [E, capacity, H]  # E由调用方的输出描述提供，可与K不同
  ```

  $$
  y = 0
  $$

  $$
  y[indices_{k,s}][locations_{k,s}][:] = x[s][:]
  \quad \text{当} \quad 0 \le locations_{k,s} < capacity \ \wedge\ 0 \le indices_{k,s} < E \ \wedge\ gates_{k,s} \neq 0
  $$

  其中 $k \in [0, K)$、$s \in [0, S)$；不满足条件的路由不写出任何字节。`E`与`K`相互独立，不能由`gates.shape[0]`推导`E`。

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
      <td>路由权重，公式中的gates。shape为[K, S]。本算子只判断其是否为零，非零数值不影响结果。</td>
      <td>与x一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>每条路由的目标专家号，公式中的indices。shape为[K, S]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>locations</td>
      <td>输入</td>
      <td>每条路由在目标专家内的容量槽号，公式中的locations。shape为[K, S]。</td>
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
      <td>分发结果，公式中的y。shape为[E, capacity, H]；Ascend 950图模式下由调用方显式提供该输出描述。</td>
      <td>与x一致</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

公共IR保持既有输入、输出和属性顺序不变；本次只新增Ascend 950实现，不修改Atlas A2/A3注册或Kernel。

### Ascend 950PR/Ascend 950DT

- 参数说明中shape使用的变量说明如下：
  - S：token数量，取值为非负整数。
  - H：隐藏维长度，取值为非负整数，不要求32字节对齐。
  - K：每个token的路由条数，取值为非负整数。
  - E：输出专家数，取值为非负整数。`E`由调用方提供的`y`描述确定，与`K`无相等约束。
- shape关系约束：`gates`、`indices`、`locations`三者shape必须完全相同；`x`的第0维必须等于`gates`的第1维；调用方提供的`y`必须为`[E, capacity, x.shape[1]]`。图侧兼容InferShape回调原样保留caller输出描述，绝不从`K`推导或覆盖`E`；Tiling负责拒绝缺失或不满足关系的输出描述。
- 数据类型约束：`x`、`gates`、`y`必须为同一种数据类型；`indices`、`locations`固定为INT32。
- 输入值域约束：`indices`的取值必须落在`[0, E)`，`locations`的取值必须落在`[0, capacity)`。超出范围的路由被跳过，不写出任何字节，也不报错。
- 唯一性约束：所有被判定为有效的`(indices[k][s], locations[k][s])`二元组必须全局唯一。若多条路由指向同一个`(专家, 槽位)`，写入顺序不确定，结果不保证，但实现仍保证不越界。
- 未命中槽位：未被任何有效路由写入的`y`位置输出正零，包括`E>K`时没有任何路由命中的专家。
- 精度：命中槽位为整行bit级拷贝，输出与输入逐bit严格相等（bit-exact，rtol=atol=0）。
- alias约束：`y`不得与`x`、`gates`、`indices`或`locations`共享存储。
- 支持动态维，不支持动态rank（`-2`）；每个维度和`capacity`均不得大于INT32_MAX。
- `K*S`、`S*H`、`E*capacity`和`E*capacity*H`必须可由signed int64表示。单次`DataCopyPad`长度不得超过2,097,151字节；更长的H由Kernel分块搬运。
- `E`、`K`、`S`、`H`或`capacity`为0的合法空Tensor均受支持。输出为空时直接成功；路由为空但输出非空时仅写正零。

### Atlas A2训练/推理系列产品

Atlas A2沿用既有产品实现及其原有shape、边界和初始化语义。本次arch35兼容实现不修改A2注册或行为；上文Ascend 950的caller输出描述、`E/K`独立、完整非法route过滤、正零初始化、空Tensor和规模合同不能解释为A2新增承诺。

### Atlas A3训练/推理系列产品

本算子当前不支持Atlas A3，本次不新增A3实现或注册。源码树中的`ascend910_93`残留binary配置不能单独建立产品入口：既有OpDef未登记该SoC，安装包标准op-info/Kernel入口也不可用。

## 调用说明

本算子**不提供aclnn对外接口**（构建配置为`ACLNNTYPE aclnn_exclude`），仅支持图模式调用。

| 调用方式   | 调用样例                                                                        | 说明                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 图模式调用 | [test_geir_moe_tutel_dispatch](./examples/test_geir_moe_tutel_dispatch.cpp) | 通过[算子IR](../../common/include/op_graph/op_transformer_proto_extend.h)构图方式调用MoeTutelDispatch算子。 |
