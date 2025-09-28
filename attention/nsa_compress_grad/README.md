# NsaCompressGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas 800I A2 推理产品</term>|      ×     |
|<term>A200I A2 Box 异构组件</term>|      ×     |


## 功能说明

- 算子功能：aclnnNsaCompress算子的反向计算。

- 计算公式：
  选择注意力的正向计算公式如下：

    $$
    \text{dw} = \text{dk\_cmp} \cdot K^\top
    $$

    $$
    \text{dk} = W^\top \cdot \text{dk\_cmp}
    $$


## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
<colgroup>
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
  </tr>
</thead>
<tbody>
  <tr>
    <td>outputGrad</td>
    <td>输入</td>
    <td>正向算子输出的反向梯度，shape支持[T, N, D]。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input</td>
    <td>输入</td>
    <td>待压缩张量，shape支持[T, N, D]。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>输入</td>
    <td>压缩权重，shape为[compressBlockSize, N]，与input满足broadcast关系。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actSeqLenOptional</td>
    <td>输入</td>
    <td>每个Batch对应的S大小，batch序列长度不等时需输入。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>compressBlockSize</td>
    <td>输入</td>
    <td>压缩滑窗大小。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>compressStride</td>
    <td>输入</td>
    <td>两次压缩滑窗间隔大小。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>actSeqLenType</td>
    <td>输入</td>
    <td>序列长度类型，0表示cumsum结果，1表示每个batch序列大小，当前仅支持0。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layoutOptional</td>
    <td>输入</td>
    <td>输入数据排布格式，支持TND。</td>
    <td>String</td>
    <td>-</td>
  </tr>
  <tr>
    <td>inputGrad</td>
    <td>输出</td>
    <td>input的梯度，shape与input保持一致。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightGrad</td>
    <td>输出</td>
    <td>weight的梯度，shape与weight保持一致。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>


## 约束说明

- compressBlockSize和compressStride要是16的整数倍，且compressBlockSize > compressStride

