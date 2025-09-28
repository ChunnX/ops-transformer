# RopeQuantKvcache

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对输入张量的尾轴进行切分，划分为q、k、vOut，对q、k进行旋转位置编码，生成qOut与kOut，之后对kOut与vOut进行量化。

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
      <td>qkv</td>
      <td>输入</td>
      <td>Device侧的aclTensor，需要切分的张量。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>Device侧的aclTensor，用于旋转位置编码的张量。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>Device侧的aclTensor，用于旋转位置编码的张量。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示量化缩放参数的张量。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_offset</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示量化偏移量的张量。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>k_cache</td>
      <td>输入</td>
      <td>用于原地更新的输入。</td>
      <td>INT8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>v_cache</td>
      <td>输入</td>
      <td>用于原地更新的输入。</td>
      <td>INT8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>indice</td>
      <td>输入</td>
      <td>用于更新量化结果的下标</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size_splits</td>
      <td>属性</td>
      <td>用于对qkv进行切分。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>属性</td>
      <td>表示qkv的数据排布方式。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kv_output</td>
      <td>属性</td>
      <td>控制是否输出原本的k、v。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>q</td>
      <td>输出</td>
      <td>切分出的q。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输出</td>
      <td>切分出的k。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>输出</td>
      <td>切分出的v。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_cache</td>
      <td>输出</td>
      <td>量化后的kOut。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_cache</td>
      <td>输出</td>
      <td>量化后的vOut。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- cos、sin的shape与k相同。


