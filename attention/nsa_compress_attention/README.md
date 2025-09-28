# NsaCompressAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |



## 功能说明

-   **算子功能**：NSA中compress attention以及select topk索引计算。论文：https://arxiv.org/pdf/2502.11089

-   **计算公式**：压缩block大小：$l$，select block大小：$l'$，压缩stride大小：$d$

$$
P_{cmp} = Softmax(query*key^T) \\
$$

$$
attentionOut = Softmax(atten\_mask(scale*query*key^T, atten\_mask)))*value
$$

$$
P_{slc}[j] = \sum_{m=0}^{l'/d-1}\sum_{n=0}^{l/d-1}P_{cmp} [l'/d*j-m-n],
$$

$$
P_{slc'} = \sum_{h=1}^{H}P_{slc}^{h}
$$

$$
P_{slc'} = topk\_mask(P_{slc'})
$$

$$
topkIndices = topk(P_{slc'})
$$

NsaCompressAttention输入query、key、value的数据排布格式支持从多种维度排布解读，可通过inputLayout传入，当前仅支持TND。
- B：表示输入样本批量大小（Batch）
- T：B和S合轴紧密排列的长度
- S：表示输入样本序列长度（Seq-Length）
- H：表示隐藏层的大小（Head-Size）
- N：表示多头数（Head-Num）
- D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）


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
    <td>query</td>
    <td>输入</td>
    <td>公式中的query。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的key。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入</td>
    <td>公式中的value。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>attenMaskOptional</td>
    <td>输入</td>
    <td>公式中的atten_mask。</td>
    <td>BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actualSelSeqKvLenOptional</td>
    <td>输入</td>
    <td>公式中的P_slc'第0维。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>topkMaskOptional</td>
    <td>输入</td>
    <td>公式中的topk_mask。</td>
    <td>BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scaleValue</td>
    <td>输入</td>
    <td>公式中的scale，缩放系数，一般为D^-0.5。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>compressBlockSize</td>
    <td>输入</td>
    <td>公式中的l，压缩滑窗大小。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>compressStride</td>
    <td>输入</td>
    <td>公式中的d，两次压缩滑窗间隔。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>selectBlockSize</td>
    <td>输入</td>
    <td>公式中的l'，选择块大小。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>selectBlockCount</td>
    <td>输入</td>
    <td>公式中topK选择个数。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>输出</td>
    <td>公式中的attentionOut，输出与query形状一致。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>topkIndicesOut</td>
    <td>输出</td>
    <td>公式中的topkIndices。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
</tbody>
</table>


## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- compressBlockSize、compressStride、selectBlockSize必须是16的整数倍，并且满足：compressBlockSize>=compressStride && selectBlockSize>=compressBlockSize && selectBlockSize%compressStride==0
- compressBlockSize：16对齐，支持到128
- compressStride：16对齐，支持到64
- selectBlockSize：16对齐，支持到128
- selectBlockCount：支持[1~32] && selectBlockCount <= min(SelSkv)
- actualSeqQLenOptional, actualCmpSeqKvLenOptional, actualSelSeqKvLenOptional需要是前缀和模式；且TND格式下必须传入。
- 由于UB限制，CmpSkv需要满足以下约束：CmpSkv <= 14000
- SelSkv = CeilDiv(CmpSkv, selectBlockSize // compressStride)
- layoutOptional目前仅支持TND。
- 输入query、key、value的数据类型必须一致。
- 输入query、key、value的batchSize必须相等。
- 输入query、key、value的headDim必须满足：qD == kD && kD >= vD
- 输入query、key、value的inputLayout必须一致。
- 输入query的headNum为N1，输入key和value的headNum为N2，则N1 >= N2 && N1 % N2 == 0
- 设G = N1 / N2，G需要满足以下约束：G < 128 && 128 % G == 0
- attenMask和topkMask的使用需符合论文描述

