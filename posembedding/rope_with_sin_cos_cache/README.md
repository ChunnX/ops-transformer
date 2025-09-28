# RopeWithSinCosCache

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：推理网络为了提升性能，将sin和cos输入通过cache传入，执行旋转位置编码计算。
- 计算公式：

    1、**mrope模式**：positions的shape输入是[3, numTokens]：
    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    $$
    cos0 = cos[0, :, :mropeSection[0]]
    $$

    $$
    cos1 = cos[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
    $$

    $$
    cos2 = cos[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
    $$

    $$
    cos = torch.cat((cos0, cos1, cos2), dim=-1)
    $$

    $$
    sin0 = sin[0, :, :mropeSection[0]]
    $$

    $$
    sin1 = sin[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
    $$

    $$
    sin2 = sin[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
    $$

    $$
    sin= torch.cat((sin0, sin1, sin2), dim=-1)
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    （1）rotate\_half（GPT-NeoX style）计算模式：
    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    （2）rotate\_interleaved（GPT-J style）计算模式：
    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    2、**rope模式**：positions的shape输入是[numTokens]：
    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    （1）rotate\_half（GPT-NeoX style）计算模式：
    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    （2）rotate\_interleaved（GPT-J style）计算模式：
    $$
    x1 = query\_rot[..., ::2]
    $$

    $$
    x2 = query\_rot[..., 1::2]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$
## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
  <col style="width: 213px">
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
      <td>positions</td>
      <td>输入</td>
      <td>Device侧的aclTensor，输入索引。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>queryIn</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示要执行旋转位置编码的第一个张量，公式中的`query`。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keyIn</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示要执行旋转位置编码的第二个张量，公式中的`key`。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cosSinCache</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示参与计算的位置编码张量。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mropeSection</td>
      <td>输入</td>
      <td>mrope模式下用于整合输入的位置编码张量信息，公式中的`mropeSection`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headSize</td>
      <td>输入</td>
      <td>表示每个注意力头维度大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isNeoxStyle</td>
      <td>输入</td>
      <td>true表示rotate\_half（GPT-NeoX style）计算模式，false表示rotate\_interleaved（GPT-J style）计算模式。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>queryOut</td>
      <td>输出</td>
      <td>输出query执行旋转位置编码后的结果。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keyOut</td>
      <td>输出</td>
      <td>输出key执行旋转位置编码后的结果。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- queryIn、keyIn、cosSinCache只支持2维shape输入。
- numQHeads支持范围: 1~32。
- numKHeads支持范围: 1~32。
- headSize支持范围: 16~128。数据类型为BFLOAT16或FLOAT16时为32的倍数，数据类型为FLOAT32时为16的倍数；
- rotaryDim支持范围: 16~128，始终小于等于headSize。数据类型为BFLOAT16或FLOAT16时为32的倍数，数据类型为FLOAT32时为16的倍数；
- 当输入tensor positions中值域超过cosSinCache的0维maxSeqLen，会有越界报错。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rope_with_sin_cos_cache](examples/test_aclnn_rope_with_sin_cos_cache.cpp) | 通过[aclnnRopeWithSinCosCache](docs/aclnnRopeWithSinCosCache.md)接口方式调用RopeWithSinCosCache算子。 |
