# cann_ops_transformer.mhc_pre_sinkhorn_backward

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：不支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

    mhc_pre_sinkhorn_backward是[mhc_pre_sinkhorn](#mhc_pre_sinkhorn)的反向算子，计算对应的梯度反向传播。

- 计算公式：

    - **Sinkhorn迭代反向**：设当前梯度为 $\mathbf{grad}_{\text{curr}} = \mathbf{gradHRes}$，共进行 $\mathbf{num\_iters}$ 次迭代。

        - **前 $\mathbf{num\_iters} - 1$ 次迭代**：对迭代编号 $i = \mathbf{num\_iters} - 1, \mathbf{num\_iters} - 2, \dots, 1$，依次执行：

          $$
          \begin{aligned}
              \mathbf{dotProd}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{normOut}_{2i+1} \right) \\
              \mathbf{grad}_{\text{curr}} &\gets \frac{\mathbf{grad}_{\text{curr}} - \mathbf{dotProd}_{2i+1}}{\mathbf{sumOut}_{2i+1}}, \\
              \mathbf{dotProd}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{normOut}_{2i} \right), \\
              \mathbf{grad}_{\text{curr}} &\gets \frac{\mathbf{grad}_{\text{curr}} - \mathbf{dotProd}_{2i}}{\mathbf{sumOut}_{2i}}, \\
          \end{aligned}
          $$

        - **最后一次迭代**：

          $$
          \begin{aligned}
              \mathbf{dotProd}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{normOut}_{1} \right), \\
              \mathbf{grad}_{\text{curr}} &\gets \frac{\mathbf{grad}_{\text{curr}} - \mathbf{dotProd}_{1}}{\mathbf{sumOut}_{1}}, \\
              \mathbf{dotProd}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{normOut}_{0} \right), \\
              \mathbf{HResGrad} &\gets \left( \mathbf{grad}_{\text{curr}} - \mathbf{dotProd}_{0} \right) \cdot \mathbf{normOut}_{0}, \\
          \end{aligned}
          $$

    - **输出组合梯度计算**：

        - 正向公式：$\mathbf{HIn} = \sum_{i=1}^{N} \mathbf{x}[B, S, i, :] \cdot \mathbf{hPre}[B, S, i]$

        - 反向公式：

          $$
          \begin{aligned}
          \mathbf{HPreGrad} &= \text{Reduce}\left(\mathbf{gradHin}.\text{unsqueeze}(-2) \odot \mathbf{x}, \text{dim}=-1\right) \quad ([B,S,N]) \\
          \mathbf{xGradVec3} &= \mathbf{gradHin} \times \mathbf{hPre} \quad ([B,S,N,C])
          \end{aligned}
          $$

    - **Sigmoid门控反向（H_pre）**：

      $$
      \begin{aligned}
      \mathbf{s} &= \mathbf{hPre} - \mathbf{hcEps} \\
      \mathbf{HPre2Grad} &= \mathbf{HPreGrad} \odot \mathbf{s} \odot (1 - \mathbf{s}) \\
      \mathbf{HPre1Grad} &= \mathbf{HPre2Grad} \cdot \mathbf{alphaPre} \\
      \mathbf{alphaPreGrad} &= \sum_{b,s,n}^{B,S,N} \left(\mathbf{HPre2Grad} \cdot \mathbf{hPre1}\right) \\
      \mathbf{biasPreGrad} &= \sum_{b,s}^{B,S} \mathbf{HPre2Grad} \quad ([N])
      \end{aligned}
      $$

    - **Sigmoid门控反向（H_post）**：

      $$
      \begin{aligned}
      \mathbf{HPost2Grad} &= \mathbf{gradHPost} \odot \left(\mathbf{hPost} \cdot \left(1 - \frac{\mathbf{hPost}}{2}\right)\right) \\
      \mathbf{HPost1Grad} &= \mathbf{HPost2Grad} \cdot \mathbf{alphaPost} \\
      \mathbf{alphaPostGrad} &= \sum_{b,s,n}^{B,S,N} \left(\mathbf{HPost2Grad} \cdot \mathbf{hPost1}\right) \\
      \mathbf{biasPostGrad} &= \sum_{b,s}^{B,S} \mathbf{HPost2Grad} \quad ([N])
      \end{aligned}
      $$

    - **残差连接反向（H_res）**：

      $$
      \begin{aligned}
      \mathbf{HRes2Grad} &= \mathbf{HResGrad} \cdot \mathbf{alphaRes} \quad ([B,S,N,N]) \\
      \mathbf{alphaResGrad} &= \sum_{b,s,i,j}^{B,S,N,N} \left(\mathbf{HResGrad} \cdot \mathbf{hRes2}\right) \\
      \mathbf{biasResGrad} &= \sum_{b,s}^{B,S} \mathbf{HResGrad} \quad ([N,N]) \\
      \mathbf{HRes1Grad} &= \text{Reshape}(\mathbf{HRes2Grad}) \quad ([B,S,N^2])
      \end{aligned}
      $$

    - **RMS归一化融合反向**：

      $$
      \begin{aligned}
      \mathbf{hMixTmpGrad} &= \text{Concat}(\mathbf{HPre1Grad}, \mathbf{HPost1Grad}, \mathbf{HRes1Grad}) \quad ([B,S,2N+N^2]) \\
      \mathbf{hMixGrad} &= \mathbf{hMixTmpGrad} \cdot \mathbf{invRms} \\
      \mathbf{invRmsGrad} &= \sum_{\text{last\_dim}} \left(\mathbf{hMixTmpGrad} \cdot \mathbf{hMix}\right) \quad ([B,S,1])
      \end{aligned}
      $$

    - **线性投影层反向**：

      $$
      \begin{aligned}
      \mathbf{xRsGrad} &= \mathbf{hMixGrad} @ \mathbf{phi} \quad ([B,S,NC]) \\
      \mathbf{phiGrad} &= \mathbf{G}^T @ \mathbf{X} \quad ([2N+N^2, NC])
      \end{aligned}
      $$

    - **RMS归一化梯度**：

      $$
      \begin{aligned}
      \mathbf{xRsGradInv} &= - \left(\frac{\mathbf{invRmsGrad} \cdot \mathbf{invRms}^3}{N \times C}\right) \cdot \mathbf{xRs} \\
      \mathbf{xRsGrad} &= \mathbf{xGradMm} + \mathbf{xRsGradInv} \\
      \mathbf{xGrad} &= \mathbf{xGradVec3} + \mathbf{xGradVec1}
      \end{aligned}
      $$

## 函数原型

```
cann_ops_transformer.mhc_pre_sinkhorn_backward(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut, hcEps) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>gradHin</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>输出h_in的梯度。对应公式中H_in_grad。</td>
        <td>BFLOAT16</td>
        <td>(B, S, C)</td>
    </tr>
    <tr>
        <td>gradHPost</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>输出h_post的梯度。对应公式中H_post_grad。</td>
        <td>FLOAT32</td>
        <td>(B, S, N)</td>
    </tr>
    <tr>
        <td>gradHRes</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>输出h_res的梯度。对应公式中H_pre_grad。</td>
        <td>FLOAT32</td>
        <td>(B, S, N, N)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向输入x。对应公式中x。</td>
        <td>BFLOAT16</td>
        <td>(B, S, N, C)</td>
    </tr>
    <tr>
        <td>phi</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向参数phi。对应公式中phi。</td>
        <td>FLOAT32</td>
        <td>(2N+N², N*C)</td>
    </tr>
    <tr>
        <td>alpha</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向参数alpha，包含3个元素。对应公式中α。</td>
        <td>FLOAT32</td>
        <td>(3)</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向参数bias，包含2N+N²个元素。对应公式中bias。</td>
        <td>FLOAT32</td>
        <td>(2N+N²)</td>
    </tr>
    <tr>
        <td>hPre</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向保存的中间结果h_pre（sigmoid之后的hPre矩阵）。对应公式中H_pre。</td>
        <td>FLOAT32</td>
        <td>(B, S, N)</td>
    </tr>
    <tr>
        <td>hcBeforeNorm</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向保存的中间结果hc_before_norm（x与phi矩阵乘的结果）。对应公式中H_mix。</td>
        <td>FLOAT32</td>
        <td>(B, S, N*N+2*N)</td>
    </tr>
    <tr>
        <td>invRms</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向保存的中间结果inv_rms。对应公式中inv_rms。</td>
        <td>FLOAT32</td>
        <td>(B, S, 1)</td>
    </tr>
    <tr>
        <td>sumOut</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>Sinkhorn变换正向计算保存的中间sum结果。对应公式中sum_out。</td>
        <td>FLOAT32</td>
        <td>(2*numIters, B, S, N)</td>
    </tr>
    <tr>
        <td>normOut</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>Sinkhorn变换正向计算保存的中间norm结果。对应公式中norm_out。</td>
        <td>FLOAT32</td>
        <td>(2*numIters, B, S, N, N)</td>
    </tr>
    <tr>
        <td>hcEps</td>
        <td>float</td>
        <td>必选</td>
        <td>数值稳定性参数，与正向算子的hcEps保持一致。</td>
        <td>FLOAT</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 输出说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>gradX</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>输入x的梯度。</td>
        <td>BFLOAT16</td>
        <td>(B, S, N, C)</td>
    </tr>
    <tr>
        <td>gradPhi</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>参数phi的梯度。</td>
        <td>FLOAT32</td>
        <td>(2N+N², N*C)</td>
    </tr>
    <tr>
        <td>gradAlpha</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>参数alpha的梯度。</td>
        <td>FLOAT32</td>
        <td>(3)</td>
    </tr>
    <tr>
        <td>gradBias</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>参数bias的梯度。</td>
        <td>FLOAT32</td>
        <td>(2N+N²)</td>
    </tr>
</tbody>
</table>

## 约束说明
- 该接口支持训练、推理场景下使用
- 该接口支持单算子模式调用
- 规格约束：

  | 规格项       | 规格               | 规格说明                                         |
  | :----------- | :----------------- | :----------------------------------------------- |
  | numIters     | 20                 | Sinkhorn迭代次数需与正向算子一致，当前仅支持20。 |
  | N            | 4                  | 目前只支持4。                                    |
  | C            | 可被128整除         | C需大于0且小于100000，且可被128整除。            |

- 参数约束：
    - 所有输入Tensor均不支持空Tensor。
    - 输入参数的shape需与参数说明中一致。
    - hcEps需与正向算子调用时的hcEps保持一致。
    - hPre、hcBeforeNorm、invRms、sumOut、normOut均为正向算子outFlag=True时的输出，需确保正向算子调用时outFlag=True以获取这些中间结果。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用
```python
import torch
import torch_npu
from cann_ops_transformer.ops import mhc_pre_sinkhorn, mhc_pre_sinkhorn_backward

B = 1
S = 1024
N = 4
C = 1280

# 正向计算（outFlag=True，保存中间结果用于反向）
x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu()
phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
alpha = torch.randn(3, dtype=torch.float32).npu()
bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()

hcMult = 4
numIters = 20
hcEps = 1e-6
normEps = 1e-6
outFlag = True

hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut = mhc_pre_sinkhorn(
    x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag
)

# 反向计算
gradHin = torch.randn_like(hin)
gradHPost = torch.randn_like(hPost)
gradHRes = torch.randn(B, S, N, N, dtype=torch.float32).npu()

gradX, gradPhi, gradAlpha, gradBias = mhc_pre_sinkhorn_backward(
    gradHin, gradHPost, gradHRes,
    x, phi, alpha, bias,
    hPre, hcBeforeNorm, invRms,
    sumOut, normOut, hcEps
)
```