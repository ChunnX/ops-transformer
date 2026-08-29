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

> 本目录新增的Kernel与Host Tiling注册仅面向Ascend 950；表中其他已支持产品继续使用存量实现，本次不覆盖。

- **算子功能**：Tutel MoE`combine`的反向算子之一，求门控权重`gates`的梯度。按`indices`、`locations`给出的路由，从`y_grad`中取出对应的一行，与`x`中对应token的一行沿隐藏维`H`做点积，得到一个标量写入`gates_grad`。路由越界时该位置输出0。

  与之配套的另外两个算子是MoeTutelDispatch（正向dispatch）与MoeTutelCombineX（combine反向求`x`的梯度），三者共用同一套路由张量语义。

- **计算公式**：

  S：token（sample）数

  H：hidden维，即被规约的轴

  E：专家数

  C：每个专家的capacity

  K：每个token的路由数，满足 $1 \le K \le E$

  x：(S, H)　　y_grad：(E, C, H)　　indices：(K, S)　　locations：(K, S)　　gates_grad：(K, S)

  $$
  valid_{(k,s)} = (0 \le indices_{(k,s)} < E) \land (0 \le locations_{(k,s)} < C)
  $$

  $$
  gatesGrad_{(k,s)} =
  \begin{cases}
  \sum_{h=0}^{H-1} x_{(s,h)} \cdot yGrad_{(indices_{(k,s)},\ locations_{(k,s)},\ h)}, & valid_{(k,s)} \\
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
      <td><ul><li>公式中的`x`，表示combine正向输出的上游梯度。</li><li>shape为2维(S, H)。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_grad</td>
      <td>输入</td>
      <td><ul><li>公式中的`yGrad`，表示正向被combine的dispatch结果。</li><li>shape为3维(E, C, H)，最后一维必须与`x`的第2维相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td><ul><li>公式中的`indices`，表示每个(路由, token)对应的专家序号。</li><li>shape为2维(K, S)。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>locations</td>
      <td>输入</td>
      <td><ul><li>公式中的`locations`，表示token在该专家capacity内的槽位序号。</li><li>shape与`indices`完全相同。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gates_grad</td>
      <td>输出</td>
      <td><ul><li>公式中的`gatesGrad`，表示门控权重`gates`的梯度。</li><li>shape与`indices`相同，为(K, S)；数据类型与`x`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `x`、`y_grad`、`gates_grad`的数据类型必须相同，不支持混合数据类型。
- `indices`、`locations`、`gates_grad`的shape必须都为(K, S)，其中S取自`x`的第1维，K满足$1 \le K \le E$。K不要求等于E，Top-1等K<E场景合法。
- `x`的第2维必须等于`y_grad`的第3维（同一个H）。
- 路由越界（`indices`或`locations`超出`[0, E)`、`[0, C)`）时，对应的`gates_grad`元素输出0，且不会访问越界内存。
- E、C、K必须大于0；S可为0，H可为0。S为0时空进空出，H为0时按空内积处理并输出0。
- C与S之间没有大小约束，C>S是合法场景。
- FLOAT16/BFLOAT16输入在计算与规约过程中提升为FLOAT32，仅在写出时做一次舍入回原数据类型。
- 本算子不提供aclnn单算子接口，仅支持Kernel直调与GE图模式调用。

## 调用说明

| 调用方式  | 样例代码                                                                              | 说明                                                                    |
| --------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 图模式接口 | [test_geir_moe_tutel_combine_gates](examples/test_geir_moe_tutel_combine_gates.cpp) | 通过GE图模式构建MoeTutelCombineGates算子图并执行RunGraph验证。 |
