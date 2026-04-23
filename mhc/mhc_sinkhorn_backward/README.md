# MhcSinkhornBackward

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：MhcSinkhornBackward是MhcSinkhorn的反向算子。mHC（Manifold-Constrained Hyper-Connections）架构中的MhcSinkhorn算子对输入矩阵做sinkhorn变换得到双随机矩阵$\mathbf{H}_{\text{res}}$，输出的双随机矩阵的所有元素≥0、每一行之和为1且每一列之和为1 (具有范数保持、组合封闭性和凸组合几何解释三大特性)。

- 计算公式：

  - 对mHC架构中双随机矩阵$\mathbf{H}_{\text{res}}$矩阵的梯度进行sinkhorn变换的反向计算得到输入$\mathbf{H}'_{\text{res}}$的梯度

  - 前 $\mathbf{num\_iters}-1$ 次迭代:

    $$
    \begin{aligned}
      \mathbf{dot\_prod}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i+1}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i+1}}{\mathbf{sum\_out}_{2i+1}}, \\
      \mathbf{dot\_prod}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i}}{\mathbf{sum\_out}_{2i}}, \\
    \end{aligned}
    $$

  - 最后一次迭代：

    $$
    \begin{aligned}
      \mathbf{dot\_prod}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{1}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{1}}{\mathbf{sum\_out}_{1}}, \\
      \mathbf{dot\_prod}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\  {⋅}\  \mathbf{norm\_out}_{0}),  \\
      \mathbf{grad}_{input} &← ({\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{0}})\  {⋅}\  \mathbf{norm\_out}_{0} \\
    \end{aligned}
    $$

  - 其中：$\mathbf{grad}_\text{curr}$ 为当前梯度（迭代中间变量），$\mathbf{grad}_\text{input}$ 为最终输入梯度（输出），$\mathbf{norm\_out}_\text{k}$为第$k$个归一化方向向量，$\mathbf{sum\_out}_\text{k}$ 为对应的缩放系数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 952px"><colgroup>
<col style="width: 106px">
<col style="width: 87px">
<col style="width: 445px">
<col style="width: 209px">
<col style="width: 105px">
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
    <td>gradOutput</td>
    <td>输入</td>
    <td>Sinkhorn变换输出的H_res的梯度。</td>
    <td>FLOAT32</td>
    <td>ND</td>
</tr>
<tr>
    <td>normOut</td>
    <td>输入</td>
    <td>Sinkhorn变换正向计算保存的中间norm结果。</td>
    <td>FLOAT32</td>
    <td>ND</td>
</tr>
<tr>
    <td>sumOut</td>
    <td>输入</td>
    <td>Sinkhorn变换正向计算保存的中间sum结果。</td>
    <td>FLOAT32</td>
    <td>ND</td>
</tr>
<tr>
    <td>out</td>
    <td>输出</td>
    <td>Sinkhorn变换的输入的H_res的梯度。</td>
    <td>FLOAT32</td>
    <td>ND</td>
</tr>
</tbody>
</table>

## 约束说明

- numIters：取值范围 1~100，超出则报参数无效。
- n：输入矩阵最后两维尺寸，仅支持 4、6、8。
- align_n：固定为 8，是 n 按 FP32 块大小 32 字节对齐后的值。
- 维度数：输入 gradOutput 仅支持 3 维 (T,n,n) 或 4 维 (B,S,n,n)。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mhc_sinkhorn_backward](examples/test_aclnn_mhc_sinkhorn_backward.cpp) | 通过接口方式调用[aclnnMhcSinkhornBackward](docs/aclnnMhcSinkhornBackward.md)算子。 |
