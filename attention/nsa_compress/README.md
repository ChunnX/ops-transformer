# NsaCompress

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas 800I A2 推理产品</term>|      ×     |
|<term>A200I A2 Box 异构组件</term>|      ×     |


## 功能说明

- 算子功能：训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。

- 计算公式：

    Nsa Compress正向计算公式如下：
$$
\tilde{K}_t^{\text{cmp}} = f_K^{\text{cmp}}(k_{:t}) = \left\{ \varphi(k_{id+1:id+l}) \bigg| 0 \leq i \leq \left\lfloor \frac{t-l}{d} \right\rfloor \right\}
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
    <td>input</td>
    <td>输入</td>
    <td>待压缩张量，shape支持[T, N, D]。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>输入</td>
    <td>压缩权重，shape支持[compressBlockSize, N]，与input满足broadcast关系。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actSeqLenOptional</td>
    <td>输入</td>
    <td>每个Batch对应的S大小。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>layoutOptional</td>
    <td>输入</td>
    <td>输入数据排布格式，支持BSH、SBH、BSND、BNSD、TND，当前仅支持TND。</td>
    <td>String</td>
    <td>-</td>
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
    <td>output</td>
    <td>输出</td>
    <td>压缩后的结果，shape支持[T, N, D]。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>


## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- input和weight需要满足broadcast关系，input.shape[1]=weight.shape[1]，不支持input、weight为空输入。
- actSeqLenType目前仅支持取值0，即actSeqLenOptional需要是前缀和模式。
- actSeqLenOptional目前不支持为空。
- layoutOptional目前仅支持TND，此时input.shape[0]必须等于actSeqLenOptional[-1]。
- input.shape[1]=weight.shape[1]，需要小于等于128。
- input.shape[2]必须是16的倍数，上限256。
- weight.shape[0]=compressBlockSize，必须是16的倍数，上限128。
- compressStride必须是16的整数倍，并且compressBlockSize>=compressStride。

