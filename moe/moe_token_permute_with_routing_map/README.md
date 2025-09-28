# aclnnMoeTokenPermuteWithEp

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

算子功能：MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。

计算公式：
- paddedMode`false`时

    $$
    sortedIndicesFirst=argSort(indices)
    $$

    $$
    sortedIndicesOut=argSort(sortedIndices)
    $$

    当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时

    $$
    permuteTokensOut[sortedIndices[i]-range[0]]=tokens[i//topK]
    $$

    $$
    permuteProbsOut[sortedIndices[i]-rangeOptional[0]]=probsOptional[i]
    $$

  - paddedMode为`true`时

    $$
    permuteTokensOut[i]=tokens[indices[i]]
    $$

    $$
    sortedIndicesOut=indices
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"> <colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 800px">
 <col style="width: 800px">
 <col style="width: 200px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>tokens</td>
   <td>输入</td>
   <td>permute中的输入tokens，公式中的`tokens`。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>indices</td>
   <td>输入</td>
   <td>输入tokens对应的专家索引，公式中的`indices`。</td>
   <td>INT32、INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>probsOptional</td>
   <td>输入</td>
   <td>可选输入，输入tokens对应的专家概率，公式中的`probsOptional`。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>rangeOptional</td>
   <td>属性</td>
   <td>ep切分的有效范围。</td>
   <td>aclIntArray</td>
   <td>-</td>
  </tr>
  <tr>
   <td>numOutTokens</td>
   <td>属性</td>
   <td>有效输出token数，在rangeOptional为空时生效。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>paddedMode</td>
   <td>属性</td>
   <td>为true时表示indices已被填充为代表每个专家选中的token索引。</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>permuteTokensOut</td>
   <td>输出</td>
   <td>indices进行扩展并排序过的tokens，公式中的`permuteTokensOut`。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndicesOut</td>
   <td>输出</td>
   <td>排序后的输出结果</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>permuteProbsOut</td>
   <td>输出</td>
   <td>permute之后的输出。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>



## 约束说明

 - tokens_num和experts_num要求小于`16777215`，pad模式为false时routingMap 中 每行为1或true的个数固定且小于`512`。
 
## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_token_permute_with_routing_map_grad.cpp](examples/test_aclnn_moe_token_permute_with_routing_map.cpp) | 通过[aclnnMoeTokenPermuteWithRoutingMapGrad](docs/aclnnmoeTokenPermuteWithRoutingMap.md)接口方式调用MoeTokenPermuteWithRoutingMapGrad算子。 |

