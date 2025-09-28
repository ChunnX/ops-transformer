# NsaSelectedAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas 800I A2 推理产品</term>|      ×     |
|<term>A200I A2 Box 异构组件</term>|      ×     |



## 功能说明

- 算子功能：训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算。

- 计算公式：
  选择注意力的正向计算公式如下：

  $$
  selected\_key = Gather(key, topk\_indices[i]),0<=i<selected\_block\_count \\
  selected\_value = Gather(value, topk\_indices[i]),0<=i<selected\_block\_count
  $$
  
  $$
  attention\_out = Softmax(Mask(scale * (query @ selected\_key^T), atten\_mask)) @ selected\_value
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
      <td>公式中的输入query。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的输入key。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的输入value。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topkIndices</td>
      <td>输入</td>
      <td>公式中的topk_indices，表示所选数据的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>可选输入</td>
      <td>公式中的atten_mask，表示注意力掩码，取值为1代表该位不参与计算，为0代表该位参与计算。</td>
      <td>BOOL、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>公式中的scale，表示缩放系数。</li>
          <li>默认值为1.0。</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selectedBlockCount</td>
      <td>属性</td>
      <td>公式中的selected_block_count，表示select block的数量。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>输出</td>
      <td>Softmax计算的Max中间结果，用于反向计算。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>输出</td>
      <td>Softmax计算的Sum中间结果，用于反向计算。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>输出</td>
      <td>公式中的attention_out。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value的batchsize必须相等，即要求传入的actualSeqQLenOptional和actualSeqKvLenOptional具有相同的长度。
- 输入query、key、value的D：Head-Dim必须满足（D_q == D_k && D_k >= D_v）。
- 输入query、key、value的数据类型必须一致。
- 输入query、key、value的input_layout必须一致。
- sparseMode目前支持0和2。
- selectedBlockSize支持<=128且满足16的整数倍。
- selectedBlockCount支持<=32。
- inputLayout目前仅支持TND。
- 支持输入query的N和key/value的N不相等，但必须成比例关系，即N_q / N_kv必须是非0整数，称为G（group），且需满足G <= 32。
- 当attenMaskOptional输入为nullptr时，sparseMode参数不生效，固定为全计算。
- 关于数据shape的约束，以inputLayout的TND举例（注：T等于各batch S的长度累加和。当各batch的S相等时，T=B*S）。其中：
  
  - B（Batchsize）：取值范围为1\~1024。
  - N（Head-Num）：取值范围为1\~128。
  - G（Group）：取值范围为1\~32。
  - S（Seq-Length）：取值范围为1\~128K。同时需要满足S_kv >= selectedBlockSize * selectedBlockCount，且S_kv长度为selectedBlockSize的整数倍。
  - D（Head-Dim）：D_qk=192，D_v=128。

