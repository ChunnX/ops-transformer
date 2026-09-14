# MhcSandwichNormPostPreonly

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- **算子功能**：MhcSandwichNormPostPreonly 算子将 mHC（Manifold-Constrained Hyper Connection）架构中 post-norm 到 pre-norm 之间的五个阶段小算子深度融合，减少 kernel 头开销和部分 HBM 访存：

  1. **RMSNorm_0**：对 h_out 做 D 维的均方根归一化；
  2. **MHC_Post**：门控+混合机制，将归一化后的 h_out 与多头残差线性组合得到 x2；
  3. **RMSNorm_mid（可选）**：当 gamma_2 非空时，对 x2 [B,S,N,D] 在 N*D 维上做 RMSNorm；
  4. **MHC_Pre（PreOnly）**：reshape + InvRMS + MatMul + sigmoid gate，加权求和得到 h_in；
  5. **RMSNorm_1**：对 h_in 做 D 维的均方根归一化，输出 h_in_prime。

- **计算公式**：

  **阶段 1: RMSNorm_0**

  $$
  x_{1,(b,s,d)} = \frac{h_{out,(b,s,d)}}{\sqrt{\frac{1}{D} \sum_{i=0}^{D-1} (h_{out,(b,s,i)})^2 + \epsilon}} \cdot \gamma_{0,(d)}
  $$

  **阶段 2: MHC_Post**

  $$
  x_{2,(b,s,n,d)} = \mathrm{h\_post}_{(b,s,n)} \cdot x_{1,(b,s,d)} + \sum_{j=0}^{N-1} \left( \mathrm{h\_res}_{(b,s,j,n)} \cdot \mathrm{residual}_{(b,s,j,d)} \right)
  $$

  **阶段 2.5: RMSNorm_mid（可选，gamma_2 非空时）**

  $$
  x_{2,(b,s,n,d)} = \frac{x_{2,(b,s,n,d)}}{\sqrt{\frac{1}{N \times D} \sum_{i=0}^{N \times D-1} (x_{2,(b,s,i)})^2 + \epsilon}} \cdot \gamma_{2,(n \times d)}
  $$

  **阶段 3: MHC_Pre（PreOnly）**

  $$
  \begin{align}
  \mathrm{invRms}_{pre} &= \frac{1}{\sqrt{\frac{1}{N \times D} \sum_{k=0}^{N \times D - 1} (x_{2,(b,s,k)})^2 + \epsilon}} \\
  x_{pre} &= (x_{2,rs} \cdot \phi^{T}) \odot \mathrm{invRms}_{pre} \\
  h_{pre} &= \sigma(\alpha_{0} \cdot x_{pre} + \mathrm{bias}) + \epsilon_{hc} \\
  h_{in} &= \sum_{n=0}^{N-1} h_{pre,(b,s,n)} \cdot x_{2,(b,s,n,:)}
  \end{align}
  $$

  **阶段 4: RMSNorm_1**

  $$
  h_{in\_prime,(b,s,d)} = \frac{h_{in,(b,s,d)}}{\sqrt{\frac{1}{D} \sum_{i=0}^{D-1} (h_{in,(b,s,i)})^2 + \epsilon}} \cdot \gamma_{1,(d)}
  $$

- **x_2_out** 输出阶段 2（或阶段 2.5，若 gamma_2 非空）之后的 x2；**h_in_prime** 输出最终归一化结果。V2 接口额外支持输出 fp32 的 h_in（return_h_in_f32）。

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
        <td>h_out</td>
        <td>输入</td>
        <td>MHC post-norm 前的 hidden layer 输出。维度shape支持[B,S,D]或[T,D]，不支持空Tensor。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>residual</td>
        <td>输入</td>
        <td>多头残差连接。维度shape支持[B,S,n,D]或[T,n,D]，不支持空Tensor。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_post</td>
        <td>输入</td>
        <td>post-norm 系数。维度shape支持[B,S,n]或[T,n]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_res</td>
        <td>输入</td>
        <td>残差加权系数。维度shape支持[B,S,n,n]或[T,n,n]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>输入</td>
        <td>pre-norm 投影矩阵。维度shape支持[n,n*D]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha_0</td>
        <td>输入</td>
        <td>sigmoid 缩放因子。维度shape支持[n]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>sigmoid 偏置。维度shape支持[n]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gamma_0</td>
        <td>输入</td>
        <td>RMSNorm_0 的缩放参数。维度shape支持[D]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gamma_1</td>
        <td>输入</td>
        <td>RMSNorm_1 的缩放参数。维度shape支持[D]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gamma_2</td>
        <td>输入（可选）</td>
        <td>RMSNorm_mid 的缩放参数，为空时跳过该阶段。维度shape支持[n*D]。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>norm_eps</td>
        <td>属性</td>
        <td>RMSNorm 防除零参数，默认值为1e-6。</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hc_eps</td>
        <td>属性</td>
        <td>MHC 门控防除零参数，默认值为1e-6。</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>return_h_in_f32</td>
        <td>属性</td>
        <td>（仅V2接口）是否额外输出fp32的h_in，默认值为false。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>h_in_prime</td>
        <td>输出</td>
        <td>RMSNorm_1 归一化后的输出。维度shape支持[B,S,D]或[T,D]，不支持空Tensor。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x_2_out</td>
        <td>输出</td>
        <td>MHC_Post（或 RMSNorm_mid）之后的 x2。维度shape支持[B,S,n,D]或[T,n,D]，不支持空Tensor。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_in_f32</td>
        <td>输出</td>
        <td>（仅V2接口，return_h_in_f32=true时）fp32 的 h_in。维度shape支持[B,S,D]或[T,D]。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- 输入约束：
  - 所有必选输入/输出参数不能为空，否则返回失败。
  - 所有输入/输出Tensor的数据格式仅支持ND。
  - 支持 BSND 格式（h_out 为 [B,S,D]）和 TND 格式（h_out 为 [T,D]）两种输入，由 h_out 的维度数决定。
- 规格约束：

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 1~256 | B*S或T支持1~256范围以内。|
| n | 4 | n为固定值4。|
| D | 2560/5120 | D支持2560或5120。|

- 典型值：

| 规格项 | 典型值 |
|:--- |:--- |
| T或B*S | 4 |
| n | 4 |
| D | 2560 |

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [aclnnMhcSandwichNormPostPreonly](docs/aclnnMhcSandwichNormPostPreonly.md) | 通过aclnnMhcSandwichNormPostPreonly接口方式调用MhcSandwichNormPostPreonly算子。 |
| torch api调用 | [npu_mhc_sandwich_norm_post_preonly](docs/npu_mhc_sandwich_norm_post_preonly.md) | 通过torch api接口方式调用MhcSandwichNormPostPreonly算子。 |
