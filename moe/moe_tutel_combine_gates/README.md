# MoeTutelCombineGates

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：Tutel MoE `combine` 的反向算子之一，求门控权重 `gates` 的梯度。按 `indices`、`locations` 给出的路由，从 `y_grad` 中取出对应的一行，与 `x` 中对应 token 的一行沿隐藏维 `H` 做点积，得到一个标量写入 `gates_grad`。路由越界时该位置输出 0。

  与之配套的另外两个算子是 MoeTutelDispatch（正向 dispatch）与 MoeTutelCombineX（combine 反向求 `x` 的梯度），三者共用同一套路由张量语义。

- **计算公式**：

  S：token（sample）数

  H：hidden 维，即被规约的轴

  E：专家数，同时也是路由轴长度

  C：每个专家的 capacity

  x：(S, H)　　y_grad：(E, C, H)　　indices：(E, S)　　locations：(E, S)　　gates_grad：(E, S)

  $$
  valid_{(e,s)} = (0 \le indices_{(e,s)} < E) \land (0 \le locations_{(e,s)} < C)
  $$

  $$
  gatesGrad_{(e,s)} =
  \begin{cases}
  \sum_{h=0}^{H-1} x_{(s,h)} \cdot yGrad_{(indices_{(e,s)},\ locations_{(e,s)},\ h)}, & valid_{(e,s)} \\
  0, & \text{otherwise}
  \end{cases}
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
      <td>x</td>
      <td>输入</td>
      <td><ul><li>公式中的`x`，表示 combine 正向输出的上游梯度。</li><li>shape 为 2 维 (S, H)。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_grad</td>
      <td>输入</td>
      <td><ul><li>公式中的`yGrad`，表示正向被 combine 的 dispatch 结果。</li><li>shape 为 3 维 (E, C, H)，最后一维必须与`x`的第 2 维相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td><ul><li>公式中的`indices`，表示每个 (路由, token) 对应的专家序号。</li><li>shape 为 2 维 (E, S)。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>locations</td>
      <td>输入</td>
      <td><ul><li>公式中的`locations`，表示 token 在该专家 capacity 内的槽位序号。</li><li>shape 与`indices`完全相同。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gates_grad</td>
      <td>输出</td>
      <td><ul><li>公式中的`gatesGrad`，表示门控权重`gates`的梯度。</li><li>shape 与`indices`相同，为 (E, S)；数据类型与`x`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `x`、`y_grad`、`gates_grad` 的数据类型必须相同，不支持混合数据类型。
- 路由轴长度必须等于专家数，即 `indices`、`locations`、`gates_grad` 的 shape 恒为 (E, S)，其中 E 取自 `y_grad` 的第 1 维、S 取自 `x` 的第 1 维；不满足时在 Host Tiling 阶段返回失败。
- `x` 的第 2 维必须等于 `y_grad` 的第 3 维（同一个 H）。
- 路由越界（`indices` 或 `locations` 超出 `[0, E)`、`[0, C)`）时，对应的 `gates_grad` 元素输出 0，且不会访问越界内存。
- 支持空 Tensor：S 为 0 时空进空出；H 为 0 时按空内积处理，`gates_grad` 全部输出 0。
- FLOAT16/BFLOAT16 输入在计算与规约过程中提升为 FLOAT32，仅在写出时做一次舍入回原数据类型。
- 本算子不提供 aclnn 单算子接口，仅支持 Kernel 直调与 GE 图模式调用。

## 调用说明

| 调用方式  | 样例代码                                                                              | 说明                                                                    |
| --------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 图模式接口 | [test_geir_moe_tutel_combine_gates](examples/test_geir_moe_tutel_combine_gates.cpp) | 通过 GE 图模式构建 MoeTutelCombineGates 算子图并执行 RunGraph 验证。 |
