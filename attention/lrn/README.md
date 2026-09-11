# LRN

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3训练系列产品</term>|      ×     |
|<term>Atlas A2训练系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：LRN（Local Response Normalization，局部响应归一化）算子对4-D NCHW输入执行跨通道（ACROSS_CHANNELS）归一化。

- 计算公式：

$$
y_{n,c,h,w} = \frac{x_{n,c,h,w}}{\left(\text{bias} + \alpha \sum_{j \in \text{win}(c)} x_{n,j,h,w}^{2}\right)^{\beta}}
$$

其中r为depth_radius，通道窗口win(c)=[max(0, c-r), min(C-1, c+r)]按通道边界裁剪。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>输入张量，4-D NCHW排布。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>depth_radius</td>
      <td>属性</td>
      <td>通道窗口半径r。取值范围[0, 48]，默认值为5。</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
      <td>bias</td>
      <td>属性</td>
      <td>归一化项的加性常数。必须大于0（保证归一化项大于0），默认值为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
     </tr>
     <tr>
      <td>alpha</td>
      <td>属性</td>
      <td>通道窗口平方和的缩放系数。必须大于0，默认值为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
     </tr>
     <tr>
      <td>beta</td>
      <td>属性</td>
      <td>归一化项的指数。默认值为0.5。</td>
      <td>FLOAT</td>
      <td>-</td>
     </tr>
     <tr>
      <td>norm_region</td>
      <td>属性</td>
      <td>归一化区域。仅支持ACROSS_CHANNELS，默认值为ACROSS_CHANNELS。</td>
      <td>STRING</td>
      <td>-</td>
     </tr>
     <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，shape和数据类型与x完全一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
     </tr>
     </tbody>
    </table>

## 约束说明

  - x必须为4-D NCHW排布，y的shape和数据类型与x完全一致。
  - x、y仅支持FLOAT16、FLOAT，不支持BFLOAT16。
  - bias、alpha必须大于0。

  <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
  <col style="width: 125px">
  <col style="width: 182px">
  <col style="width: 602px">
  </colgroup>
  <thead>
  <tr>
    <th>规格项</th>
    <th>规格</th>
    <th>规格说明</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>depth_radius</td>
    <td>0~48</td>
    <td>默认值为5。</td>
  </tr>
  <tr>
    <td>数据类型</td>
    <td>FLOAT16、FLOAT</td>
    <td>x与y数据类型必须一致。</td>
  </tr>
  <tr>
    <td>norm_region</td>
    <td>ACROSS_CHANNELS</td>
    <td>-</td>
  </tr>
  </tbody>
  </table>

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_lrn.cpp](examples/test_geir_lrn.cpp) | 通过GE IR构图方式调用LRN算子。 |
