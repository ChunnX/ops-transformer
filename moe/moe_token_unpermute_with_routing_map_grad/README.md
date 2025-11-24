# MoeTokenUnpermuteWithRoutingMapGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- **算子功能**：aclnnMoeTokenUnpermuteWithRoutingMap的反向传播。
- **计算公式**：

  (1) probs非None：
  
  $$
  permutedTokensGrad[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$
  
  $$
  permutedProbsGrad = permutedTokenGrad * permutedTokensOptional
  $$
  
  $$
  probsGradExpertOrder = \sum_{j=0}^{hidden\_size}(permutedProbsGrad_{i,j})
  $$

    - paddedMode为false时
  
  $$
  probsGradOut = masked\_scatter(routingMapOptional.T,probsGradExpertOrder)
  $$
  
  $$
  permutedProbs = probsOptional.T.masked\_select(routingMapOptional.T)
  $$

  $$
  permutedTokensGradOut = permutedProbs.unsqueeze(-1) * permutedTokensGrad
  $$

    - paddedMode为true时
  
  $$
  probsGradOut[permuteTokenId[i], outIndex[i]/capacity] = probsGradExpertOrder[outIndex[i]]
  $$

  $$
  permutedProbs[outIndex[i]] = probsOptional.view(1)[i]
  $$

  $$
  permutedTokensGradOut = permutedProbs * permutedTokensGrad
  $$

    (2) probs为None：
  $$
  permutedTokensGradOut[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  1. hidden_size指unpermutedTokensGrad的第1维大小。
  2. paddedMode等于true时，每个专家固定能够处理capacity个token。输入routingMapOptional的第1维是experts_num，即专家个数，输入outIndex的第0维是experts_num * capacity，根据这两个维度可以算出capacity。
  3. paddedMode等于false时，每个token固定被topK_num个专家处理。输入unpermutedTokensGrad的第0维是tokens_num，即token的个数，输入outIndex的第0维是tokens_num * capacity，根据这两个维度可以算出topK_num。


  
  $$
  dequantX = Dequant(x,weightScaleOptional,activationScaleOptional,biasOptional)
  $$
  
  $$
  q,k,vOut = SplitTensor(dequantX,dim=-1,`sizeSplits`)
  $$
  
  $$
  qOut,kOut = ApplyRotaryPosEmb(q,k,cos,sin)
  $$
  
  $$
  quantK = Quant(kOut,scaleK,offsetKOptional)
  $$
  
  $$
  quantV = Quant(vOut,scaleV,offsetVOptional)
  $$
  
  如果cacheModeOptional为contiguous则：
  
  $$
  kCacheRef[i][indice[i]]=quantK[i]
  $$
  
  $$
  vCacheRef[i][indice[i]]=quantV[i]
  $$
  
  如果cacheModeOptional为page则：
  
  $$
  kCacheRefView=kCacheRef.view(-1,kCacheRef[-2],kCacheRef[-1])
  $$
  
  $$
  vCacheRefView=vCacheRef.view(-1,vCacheRef[-2],vCacheRef[-1])
  $$
  
  $$
  kCacheRefView[indices[i]]=quantK[i]
  $$
  
  $$
  vCacheRefView[indices[i]]=quantV[i]
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
      <td>unpermutedTokensGrad</td>
      <td>输入</td>
      <td>Device侧的aclTensor。计算公式中的unpermutedTokensGrad，代表正向输出unpermutedTokens的梯度。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outIndex</td>
      <td>输入</td>
      <td>Device侧的aclTensor。计算公式中outIndex，代表输出位置索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permuteTokenId</td>
      <td>输入</td>
      <td>Device侧的aclTensor。计算公式中的permuteTokenId，代表输入permutedTokens每个位置对应的Token序号。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>routingMapOptional</td>
      <td>可选输入</td>
      <td>Device侧的aclTensor，可选输入，当输入probsOptional为空指针时不需要此输入，应该传入空指针。计算公式中的routingMapOptional，代表对应位置的Token是否被对应专家处理。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permutedTokensOptional</td>
      <td>可选输入</td>
      <td>Device侧的aclTensor，可选输入，当输入probsOptional为空指针时不需要此输入，应该传入空指针。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>可选输入</td>
      <td>Device侧的aclTensor，可选输入，当不需要时为空指针。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>属性</td>
      <td>host侧的BOOL。true表示开启paddedMode，false表示关闭paddedMode。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional</td>
      <td>属性</td>
      <td>host侧的aclIntArray。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permutedTokensGradOut</td>
      <td>输出</td>
      <td>输入permutedTokens的梯度，要求是一个2D的Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsGradOutOptional</td>
      <td>可选输出</td>
      <td>当不需要时为空指针。输入probs的梯度。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

-   topkNum <= 512。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_token_unpermute_with_routing_map_grad](examples/test_aclnn_moe_token_unpermute_with_routing_map_grad.cpp) | 通过[aclnnMoeTokenUnpermuteWithRoutingMapGrad](docs/aclnnMoeTokenUnpermuteWithRoutingMapGrad.md)接口方式调用MoeTokenUnpermuteWithRoutingMapGrad算子。 |
