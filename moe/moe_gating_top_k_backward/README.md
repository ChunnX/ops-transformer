# aclnnMoeGatingTopKBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：完成MoE（Mixture of Experts）门控Top-K选择的反向梯度计算。该算子是aclnnMoeGatingTopK的反向算子，根据前向算子输出的归一化得分（xNorm）、上游梯度（gradY）和专家索引（expertIdx），计算输入得分矩阵的梯度（gradX）。支持sigmoid模式（normType=1）。
- 计算公式（sigmoid模式，normType=1）：
  
  步骤1：缩放梯度
  
  $$
  gradYScaled_{ip} = routedScalingFactor \cdot gradY_{ip}
  $$
  
  步骤2：正向renorm的反向传播
  
  $$
  wPrime_{ip} = xNorm_{i,\ expertIdx_{ip}}
  $$
  
  $$
  D_i = \sum_{p} wPrime_{ip} + eps
  $$
  
  $$
  w_{ip} = \frac{wPrime_{ip}}{D_i}
  $$
  
  $$
  beta_i = \sum_{p} w_{ip} \cdot gradYScaled_{ip}
  $$
  
  $$
  gradWPrime_{ip} = \frac{gradYScaled_{ip} - beta_i}{D_i}
  $$
  
  步骤3：散射到完整维度
  
  $$
  gradNormX_{ij} = \sum_{p:\ expertIdx_{ip}=j} gradWPrime_{ip}
  $$
  
  步骤4：Sigmoid反向传播
  
  $$
  gradX_{ij} = xNorm_{ij} \cdot (1 - xNorm_{ij}) \cdot gradNormX_{ij}
  $$



## 参数说明
  
<table style="undefined;table-layout: fixed; width: 1550px"><colgroup>

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
    <td>xNorm</td>
    <td>输入</td>
    <td>计算的输入，对应公式中的xNorm。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradY</td>
    <td>输入</td>
    <td>表示前向算子输出yOut的上游梯度，对应公式中的gradY。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>expertIdx</td>
    <td>输入</td>
    <td>表示前向算子输出的top-k专家索引，对应公式中的expertIdx。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>renorm</td>
    <td>输入</td>
    <td>表示前向算子在softmax模式下renorm标记。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>normType</td>
    <td>输入</td>
    <td>表示norm函数类型。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>routedScalingFactor</td>
    <td>输入</td>
    <td>表示计算yOut使用的routed_scaling_factor系数，对应公式中的routedScalingFactor。</td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>eps</td>
    <td>输入</td>
    <td>表示前向计算使用的防除零常数，对应公式中的eps。</td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradX</td>
    <td>输出</td>
    <td>表示前向算子输入参数x的梯度，对应公式中的gradX。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody></table>


## 约束说明

- 确定性计算：
  - aclnnMoeGatingTopKBackward默认确定性实现。
- xNorm和gradX的shape为[M,N]，N的取值范围为[2,2048]
- gradY和expertIdx的shape为[M,K]，K的取值范围为[1,N]

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_gating_top_k_backward](examples/test_aclnn_moe_gating_top_k_backward.cpp) | 通过[aclnnMoeGatingTopKBackward](docs/aclnnMoeGatingTopKBackward.md)接口方式调用MoeGatingTopKBackward算子。 |