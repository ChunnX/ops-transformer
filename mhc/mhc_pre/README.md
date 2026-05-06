# MhcPre

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- **算子功能**：基于一系列计算得到MHC架构中hidden层的$H^{res}$和$H^{post}$投影矩阵以及Attention或MLP层的输入矩阵$h^{in}$。

- **计算公式**：

$$
\begin{aligned}
\vec{x^{'}_{l}} &=RMSNorm(\vec{x_{l}})\\
H^{pre}_l &= \alpha^{pre}_{l} ·(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
H^{post}_l &= \alpha^{post}_{l} ·(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
H^{res}_l &= \alpha^{res}_{l} ·(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
H^{pre}_l &= \sigma (H^{pre}_{l})\\
H^{post}_l &= 2\sigma (H^{post}_{l})\\
h_{in} &=\vec{x^{'}_{l}}H^{pre}_l
\end{aligned}
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
        <td>x</td>
        <td>输入</td>
        <td>待计算数据，表示网络中mHC层的输入数据。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>输入</td>
        <td>mHC的参数矩阵。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>输入</td>
        <td>mHC的缩放参数。</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>mHC的bias参数。</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gamma</td>
        <td>可选输入</td>
        <td>表示进行RmsNorm计算的缩放因子。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>out_flag</td>
        <td>可选输入</td>
        <td>表示是否输出inv_rms/h_mix/h_pre, 默认为0表示不输出，为1表示全输出。</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>norm_eps</td>
        <td>可选输入</td>
        <td>RmsNorm的防除零参数。</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hc_eps</td>
        <td>可选输入</td>
        <td>h_pre的sigmoid后的eps参数。</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>h_in</td>
        <td>输出</td>
        <td>输出的h_in作为Attention/MLP层的输入。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_post</td>
        <td>输出</td>
        <td>输出的mHC的h_post变换矩阵。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_res</td>
        <td>输出</td>
        <td>输出的mHC的h_res变换矩阵（未做sinkhorn变换）。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>inv_rms</td>
        <td>可选输出</td>
        <td>RmsNorm计算得到的1/r。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_mix</td>
        <td>可选输出</td>
        <td>x与phi矩阵乘的结果。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_pre</td>
        <td>可选输出</td>
        <td>做完sigmoid计算之后的h_pre矩阵。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- n目前支持4、6、8。
- D支持1~16384范围以内，需满足D为16对齐。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mhc_pre](examples/test_aclnn_mhc_pre.cpp) | 通过[aclnnMhcPre](docs/aclnnMhcPre.md)接口方式调用MhcPre算子。 |
