# NsaSelectedAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas 800I A2 推理产品</term>|      ×     |
|<term>A200I A2 Box 异构组件</term>|      ×     |


## 功能说明

- 算子功能：根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。

- 计算公式：

    根据传入的topkIndice对keyIn和value选取数量为selectedBlockCount个大小为selectedBlockSize的数据重排，公式如下：

    $$
    selectedKey = Gather(key, topkIndices[i]),0<=i<selectedBlockCount \\
    selectedValue = Gather(value, topkIndices[i]),0<=i<selectedBlockCount
    $$

    接着，进行注意力机制的反向计算，计算公式为：

    $$
    V=P^TdY
    $$

    $$
    Q=\frac{((dS)*K)}{\sqrt{d}}
    $$

    $$
    K=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$




## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>query</td>
      <td>输入</td>
      <td>公式中的输入Q。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的输入K。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的输入V。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>可选输入</td>
      <td>注意力正向计算的最终输出。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attentionOutGrad</td>
      <td>输入</td>
      <td>公式中的输入dY。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxMax</td>
      <td>输入</td>
      <td>注意力正向计算的中间输出。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxSum</td>
      <td>输入</td>
      <td>注意力正向计算的中间输出。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topkIndices</td>
      <td>输入</td>
      <td>公式中的所选择KV的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>属性</td>
      <td>公式中d开根号的倒数，表示缩放系数。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>输出</td>
      <td>公式中的dQ，表示query的梯度。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dkOut</td>
      <td>输出</td>
      <td>公式中的dK，表示key的梯度。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dvOut</td>
      <td>输出</td>
      <td>公式中的dV，表示value的梯度。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>


## 约束说明

- 该接口与pytorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value、attentionOut、attentionOutGrad的B（batchsize）必须相等。
- 输入key、value的N（numHead）必须一致。
- 输入query、attentionOut、attentionOutGrad的N（numHead）必须一致。
- 输入value、attentionOut、attentionOutGrad的D（HeadDim）必须一致。
- 输入query、key、value、attentionOut、attentionOutGrad的inputLayout必须一致。
- 关于数据shape的约束，以inputLayout的TND举例。其中：
  - T1：取值范围为1\~2M。T1表示query所有batch下S的和。
  - T2：取值范围为1\~2M。T2表示key、value所有batch下S的和。
  - B：取值范围为1\~2M。
  - N1：取值范围为1\~128。表示query的headNum。N1必须为N2的整数倍。
  - N2：取值范围为1\~128。表示key、value的headNum。
  - G：取值范围为1\~32。G = N1 / N2
  - S：取值范围为1\~128K。对于key、value的S 必须大于等于selectedBlockSize * selectedBlockCount, 且必须为selectedBlockSize的整数倍。
  - D：取值范围为192或128，支持K和V的D（HeadDim）不相等。
- 关于softmaxMax与softmaxSum参数shape的约束：\[T1, N1, 8\]。
- 关于topkIndices参数shape的约束：[T1, N2, selectedBlockCount]。

