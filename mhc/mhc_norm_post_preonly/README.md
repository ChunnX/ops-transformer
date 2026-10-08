# MhcNormPostPreonly

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- **算子功能**：融合MHC Sandwich结构中的RMSNorm_0、MHC_Post、可选RMSNorm_mid、MHC_Pre（PreOnly）和RMSNorm_1，输出下一层输入、MHC_Post中间结果以及可选的FLOAT32计算结果。

- **计算公式**：以下公式按 [B,S,...] 形式表示；[T,...] 形式将 B*S 展平为 T，后续维度保持不变。

  $$
  \begin{aligned}
  h\_out\_fp32_{b,s,d} &= \operatorname{FP32}(h\_out_{b,s,d}), \\
  inv\_rms\_0_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}h\_out\_fp32_{b,s,i}^{2}+norm\_eps\right), \\
  x\_1_{b,s,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(h\_out\_fp32_{b,s,d}\cdot inv\_rms\_0_{b,s}\cdot gamma\_0_d\right), \\
  term1_{b,s,n,d} &= \operatorname{FP32}(h\_post_{b,s,n})\cdot \operatorname{FP32}(x\_1_{b,s,d}), \\
  term2_{b,s,n,d} &= \sum_{j=0}^{N-1}\operatorname{FP32}(h\_res_{b,s,j,n})\cdot \operatorname{FP32}(residual_{b,s,j,d}), \\
  x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(term1_{b,s,n,d}+term2_{b,s,n,d}\right).
  \end{aligned}
  $$

  其中 j 为归约 head，n 为输出 head。

  当 gamma_2 非空时执行可选 RMSNorm_mid：

  $$
  \begin{aligned}
  x\_2\_flat &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
  inv\_rms\_mid_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_flat_{b,s,k}^{2}+norm\_eps\right), \\
  x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(\operatorname{reshape}\left(x\_2\_flat\cdot inv\_rms\_mid\cdot gamma\_2,[B,S,N,D]\right)_{b,s,n,d}\right).
  \end{aligned}
  $$

  gamma_2 为空时跳过 RMSNorm_mid，随后固定保存 $x\_2\_out=x\_2$。MHC_Pre（PreOnly）与 RMSNorm_1 为：

  $$
  \begin{aligned}
  x\_2\_rs &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
  inv\_rms\_pre_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_rs_{b,s,k}^{2}+mhc\_pre\_norm\_eps\right), \\
  proj\_w &= phi^T, \\
  x\_pre &= (x\_2\_rs\cdot proj\_w)\cdot inv\_rms\_pre, \\
  h\_pre_{b,s,n} &= \operatorname{sigmoid}(alpha\_0\cdot x\_pre_{b,s,n}+bias_n)+hc\_eps, \\
  h\_in_{b,s,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(\sum_{n=0}^{N-1}h\_pre_{b,s,n}\cdot \operatorname{FP32}(x\_2_{b,s,n,d})\right), \\
  h\_in\_rms_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}\operatorname{FP32}(h\_in_{b,s,i})^{2}+norm\_eps\right), \\
  h\_in\_f32_{b,s,d} &= \operatorname{FP32}(h\_in_{b,s,d})\cdot h\_in\_rms_{b,s}\cdot \operatorname{FP32}(gamma\_1_d), \\
  h\_in\_prime &= \operatorname{Cast}_{dtype(h\_out)}(h\_in\_f32).
  \end{aligned}
  $$

  norm_eps 用于RMSNorm_0、RMSNorm_mid和RMSNorm_1，mhc_pre_norm_eps 仅用于MHC_Pre归一化，hc_eps 仅用于pre门控，三者均按FLOAT精度参与计算。

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
      <tr><td>h_out</td><td>输入</td><td>Atten/MLP 层的输出。</td><td>BFLOAT16、FLOAT16</td><td>ND</td></tr>
      <tr><td>residual</td><td>输入</td><td>MHC 多路残差输入；dtype 必须与 h_out 一致。</td><td>BFLOAT16、FLOAT16</td><td>ND</td></tr>
      <tr><td>h_post</td><td>输入</td><td>MHC_Post 主分支权重。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>h_res</td><td>可选输入</td><td>mHC 的 hRes 变换矩阵，是做完 Sinkhorn 变换后的双随机矩阵；索引顺序为 [j,n]。接口预留参数，暂不支持传 None。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>phi</td><td>输入</td><td>MHC_Pre 投影矩阵。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>alpha</td><td>输入</td><td>Sigmoid 门控缩放系数。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>bias</td><td>输入</td><td>Sigmoid 门控偏置。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>pre_mix</td><td>可选输入</td><td>传入时，h_in 的加权求和直接使用 pre_mix 作为权重（典型用法：填入上一轮 h_pre 的输出）；不传入时，h_in 使用本轮内部计算的 h_pre 进行加权求和。接口预留参数，当前仅支持传入 None。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>gamma_0</td><td>可选输入</td><td>RMSNorm_0 缩放权重。接口预留参数，当前暂不支持传 None。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>gamma_1</td><td>输入</td><td>RMSNorm_1 缩放权重。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>gamma_2</td><td>可选输入</td><td>RMSNorm_mid 缩放权重。默认值为 None；非空时启用 RMSNorm_mid，为空时跳过。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>h_in_prime</td><td>输出</td><td>RMSNorm_1 FP32 结果转换为输入 dtype 后的下一层输入。</td><td>BFLOAT16、FLOAT16</td><td>ND</td></tr>
      <tr><td>x_2_out</td><td>输出</td><td>MHC_Post 以及可选 RMSNorm_mid 后的中间结果。</td><td>BFLOAT16、FLOAT16</td><td>ND</td></tr>
      <tr><td>h_in_f32</td><td>可选输出</td><td>RMSNorm_1 的 FP32 结果，仅在 return_h_in_f32=true 时返回。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>h_pre</td><td>可选输出</td><td>本轮内部计算的 pre 结果（sigmoid 后的加权系数），可作为下一轮 pre_mix 的输入。接口预留参数，当前暂不支持输出。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>norm_eps</td><td>可选属性</td><td>RMSNorm_0、RMSNorm_mid 和 RMSNorm_1 的数值稳定项，默认值为 1e-6。</td><td>FLOAT</td><td>-</td></tr>
      <tr><td>mhc_pre_norm_eps</td><td>可选属性</td><td>MHC_Pre 内部 RMS 的数值稳定项，默认值为 1e-6。</td><td>FLOAT</td><td>-</td></tr>
      <tr><td>hc_eps</td><td>可选属性</td><td>Sigmoid 门控结果的加性偏移，默认值为 1e-6。</td><td>FLOAT</td><td>-</td></tr>
      <tr><td>return_h_in_f32</td><td>可选属性</td><td>是否额外返回 FP32 的 h_in_f32，默认值为 false。</td><td>BOOL</td><td>-</td></tr>
    </tbody>
  </table>

## 约束说明

- 仅支持 N=4。
- h_out、residual、h_in_prime、x_2_out 的数据类型须一致，仅支持 BFLOAT16、FLOAT16。
- 输入支持 [T, D] 或 [B, S, D] 两种前导维形式，且各输入的前导维形式须保持一致。
- gamma_0 为接口预留参数，当前暂不支持传 None。
- gamma_2 默认值为 None，传入 None 时跳过 RMSNorm_mid。
- pre_mix、h_pre 为接口预留参数，当前暂不支持使用。
- norm_eps、mhc_pre_norm_eps、hc_eps 默认值为 1e-6。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn调用 | [test_aclnn_mhc_norm_post_preonly](examples/test_aclnn_mhc_norm_post_preonly.cpp) | 通过[aclnnMhcNormPostPreonly](docs/aclnnMhcNormPostPreonly.md)接口调用MhcNormPostPreonly算子。 |
| PyTorch调用 | - | 通过[mhc_norm_post_preonly](docs/torchapi_mhc_norm_post_preonly.md)接口调用MhcNormPostPreonly算子。 |
