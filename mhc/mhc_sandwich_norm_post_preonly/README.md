# MhcSandwichNormPostPreonly

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
      <tr><td>h_out</td><td>输入</td><td>MHC_Post的主分支输入。</td><td>BFLOAT16、FLOAT16</td><td>ND</td></tr>
      <tr><td>residual</td><td>输入</td><td>MHC_Post的多路残差输入，数据类型与h_out一致。</td><td>BFLOAT16、FLOAT16</td><td>ND</td></tr>
      <tr><td>h_post</td><td>输入</td><td>MHC_Post的主分支权重。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>h_res</td><td>输入</td><td>MHC_Post的残差混合权重。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>phi</td><td>输入</td><td>MHC_Pre的投影参数。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>alpha_0</td><td>输入</td><td>MHC_Pre的缩放参数。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>bias</td><td>输入</td><td>MHC_Pre的偏置参数。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>gamma_0</td><td>输入</td><td>RMSNorm_0的缩放因子。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>gamma_1</td><td>输入</td><td>RMSNorm_1的缩放因子。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>gamma_2</td><td>可选输入</td><td>RMSNorm_mid的缩放因子，为空时跳过RMSNorm_mid。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>h_in_prime</td><td>输出</td><td>RMSNorm_1结果转换为h_out数据类型后的下一层输入。</td><td>BFLOAT16、FLOAT16</td><td>ND</td></tr>
      <tr><td>x_2_out</td><td>输出</td><td>MHC_Post结果；gamma_2非空时为RMSNorm_mid处理后的结果。</td><td>BFLOAT16、FLOAT16</td><td>ND</td></tr>
      <tr><td>h_in_f32</td><td>可选输出</td><td>RMSNorm_1的FLOAT32计算结果。</td><td>FLOAT32</td><td>ND</td></tr>
      <tr><td>norm_eps</td><td>可选属性</td><td>RMSNorm_0、RMSNorm_mid和RMSNorm_1的防除零参数，默认值为1e-6。</td><td>FLOAT</td><td>-</td></tr>
      <tr><td>mhc_pre_norm_eps</td><td>可选属性</td><td>MHC_Pre内部RMS的防除零参数，默认值为1e-6。</td><td>FLOAT</td><td>-</td></tr>
      <tr><td>hc_eps</td><td>可选属性</td><td>Sigmoid门控结果的加性偏移，默认值为1e-6。</td><td>FLOAT</td><td>-</td></tr>
      <tr><td>return_h_in_f32</td><td>可选属性</td><td>是否返回h_in_f32，默认值为false。</td><td>BOOL</td><td>-</td></tr>
    </tbody>
  </table>

## 约束说明

- 仅支持 Atlas A5 训练系列产品/Atlas 950 SuperPoD 推理系统。
- 支持 `[T,...]` 和 `[B,S,...]` 两种配套布局，当前实现要求 `N=4`、`D=2560/5120`。
- `h_out` 和 `residual` 支持 FLOAT16、BFLOAT16 且类型须一致，其余 Tensor 为 FLOAT。
- 当前仅支持 ND 格式。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn调用 | [test_aclnn_mhc_sandwich_norm_post_preonly](examples/test_aclnn_mhc_sandwich_norm_post_preonly.cpp) | 通过[aclnnMhcSandwichNormPostPreonly](docs/aclnnMhcSandwichNormPostPreonly.md)接口调用MhcSandwichNormPostPreonly算子。 |
| PyTorch调用 | - | 通过[mhc_sandwich_norm_post_preonly](docs/torchapi_mhc_sandwich_norm_post_preonly.md)接口调用MhcSandwichNormPostPreonly算子。 |
