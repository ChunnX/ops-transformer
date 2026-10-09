# MhcPreSplitPostRes

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

- **算子功能**：MhcPreSplitPostRes算子是MhcPre算子的一部分，负责计算mHC（Manifold-Constrained Hyper Connection）架构中x2输入对应的post分支和comb/Res分支，输出h_post和h_res两个变换矩阵（h_res为可选输出：phi行数为n^2+n时输出，为n时仅输出h_post、h_res为0元素tensor）。具体流程为：将输入x展平并转FP32后计算RMSNorm的invRms因子，执行线性投影得到x_hat，然后切分为post分支和comb分支分别计算h_post和h_res

- **计算公式**：

  $$
  X_{flat} = \mathrm{reshape}(x, [B,S,N \cdot D])
  $$

  $$
  \mathrm{inv\_rms} = \left(\frac{1}{N \cdot D}\sum_{k} X_{flat,k}^{2} + \mathrm{norm\_eps}\right)^{-\frac{1}{2}}
  $$

  $$
  X_{hat} = (X_{flat} \cdot \phi^{T}) \odot \mathrm{inv\_rms}
  $$

  $$
  [X_{post}, X_{comb}] = \mathrm{split}(X_{hat}, [N, N^{2}])
  $$

  $$
  h_{post} = 2 \cdot \sigma(\alpha_{0} \cdot X_{post} + b_{post})
  $$

  $$
  h_{res} = \alpha_{1} \cdot \mathrm{reshape}(X_{comb}, [B,S,N,N]) + \mathrm{reshape}(b_{comb}, [N,N])
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
        <td>待计算的数据，表示网络中mHC层的输入数据x2，对应公式中的x。维度shape支持[B,S,n,D]或[T,n,D]，不支持空Tensor。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>输入</td>
        <td>mHC的变换矩阵，对应公式中的phi。维度shape支持[n<sup>2</sup>+n, n*D]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>输入</td>
        <td>mHC的缩放参数，对应公式中的alpha。维度shape支持[2]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>mHC的bias参数，对应公式中的bias。维度shape支持[n<sup>2</sup>+n]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>norm_eps</td>
        <td>属性</td>
        <td>RMSNorm防除零参数，对应公式中的norm_eps，默认值为1e-6。</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>h_post</td>
        <td>输出</td>
        <td>mHC的h_post变换矩阵，对应公式中的h_post。维度shape支持[B,S,n]或[T,n]，不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_res</td>
        <td>输出</td>
        <td>mHC的comb分支输出矩阵，对应公式中的h_res。可选输出：phi行数为n^2+n（完整形态）时维度为[B,S,n,n]或[T,n,n]；phi行数为n（仅post形态）时为0元素tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- 输入约束：
  - 所有输入参数不能为空，否则返回失败。
  - 所有输入/输出Tensor的数据格式仅支持ND。
- 规格约束：

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 1及以上 | B*S或T至少为1，当前实现不设置上限。|
| n | 4 | n为固定值4。|
| D | 白名单{2560,5120,6144,7168} | D仅支持2560/5120/6144/7168四种典型值，其余值拦截。|

- 典型值：

| 规格项 | 典型值 |
|:--- |:--- |
| T或B*S | 4 |
| n | 4 |
| D | 2560/5120/6144/7168 |

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | - | 通过[aclnnMhcPreSplitPostRes](docs/aclnnMhcPreSplitPostRes.md)接口方式调用MhcPreSplitPostRes算子。 |
| PyTorch API | - | 通过[mhc_pre_split_post_res](docs/torchapi_mhc_pre_split_post_res.md)接口方式调用MhcPreSplitPostRes算子。 |
