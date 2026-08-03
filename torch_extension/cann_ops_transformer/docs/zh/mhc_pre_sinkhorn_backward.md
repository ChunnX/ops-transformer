# mhc_pre_sinkhorn_backward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

  `mhc_pre_sinkhorn_backward`是`mhc_pre_sinkhorn`的反向算子，用于计算MHC（Manifold-Constrained Hyper-Connections）结构中Sinkhorn变换的反向梯度传播。基于前向计算缓存的中间结果，计算输入`x`和参数`phi`、`alpha`、`bias`的梯度。

- **计算公式**：

  $$
  \begin{aligned}
  gradX &= \nabla_{x}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias})) \\
  gradPhi &= \nabla_{\phi}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias})) \\
  gradAlpha &= \nabla_{\alpha}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias})) \\
  gradBias &= \nabla_{\text{bias}}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias}))
  \end{aligned}
  $$

  >**说明：**
  >
  > 输入支持两种维度格式：BSND（4维）和TND（3维）。其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，T表示所有Batch序列长度的累加和（$T = B \times S$），N表示头数（head数量），C表示每个头的隐藏维度大小（headdim）。

## 函数原型

```python
cann_ops_transformer.mhc_pre_sinkhorn_backward(
    grad_hin, grad_h_post, grad_h_res,
    x, phi, alpha, bias,
    h_pre, hc_before_norm, inv_rms,
    sum_out, norm_out, hc_eps
) -> (Tensor, Tensor, Tensor, Tensor)
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
        <td>grad_hin</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向输出h_in的梯度。</td>
        <td>bfloat16、float16</td>
        <td>
            <ul>
                <li>(B, S, C)</li>
                <li>(T, C)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>grad_h_post</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向输出h_post的梯度。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, N)</li>
                <li>(T, N)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>grad_h_res</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向输出h_res的梯度。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, N, N) 或 (B, S, N*N)</li>
                <li>(T, N, N) 或 (T, N*N)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向输入x。对应公式中x。</td>
        <td>bfloat16、float16</td>
        <td>
            <ul>
                <li>(B, S, N, C)</li>
                <li>(T, N, C)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>phi</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向参数phi。对应公式中ψ。</td>
        <td>float32</td>
        <td>(N*N+2*N, N*C)</td>
    </tr>
    <tr>
        <td>alpha</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向参数alpha。对应公式中α。</td>
        <td>float32</td>
        <td>(3)</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向参数bias。对应公式中β。</td>
        <td>float32</td>
        <td>(2N+N²)</td>
    </tr>
    <tr>
        <td>h_pre</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向缓存的h_pre。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, N)</li>
                <li>(T, N)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>hc_before_norm</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向缓存的hc_before_norm。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, 2N+N²)</li>
                <li>(T, 2N+N²)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>inv_rms</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向缓存的inv_rms（RMSNorm的倒数均方根）。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, 1)</li>
                <li>(T, 1)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sum_out</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向缓存的Sinkhorn迭代求和结果。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(2*numIters, B, S, N)</li>
                <li>(2*numIters, T, N)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>norm_out</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向缓存的Sinkhorn迭代归一化结果。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(2*numIters, B, S, N, N)</li>
                <li>(2*numIters, T, N, N)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>hc_eps</td>
        <td>float</td>
        <td>必选</td>
        <td>Sinkhorn变换的eps参数。对应前向的ε<sub>hc</sub>。</td>
        <td>-</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

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
        <td>grad_x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>x的梯度。数据类型和shape与输入x一致。</td>
        <td>bfloat16、float16</td>
        <td>
            <ul>
                <li>(B, S, N, C)</li>
                <li>(T, N, C)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>grad_phi</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>phi的梯度。数据类型和shape与输入phi一致。</td>
        <td>float32</td>
        <td>(N*N+2*N, N*C)</td>
    </tr>
    <tr>
        <td>grad_alpha</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>alpha的梯度。数据类型和shape与输入alpha一致。</td>
        <td>float32</td>
        <td>(3)</td>
    </tr>
    <tr>
        <td>grad_bias</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>bias的梯度。数据类型和shape与输入bias一致。</td>
        <td>float32</td>
        <td>(2N+N²)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口仅支持训练场景下使用。
- 该接口支持单算子模式调用。
- 参数约束：所有输入Tensor不支持空Tensor。
- 维度约束：所有带B/S维度的输入需保持维度格式一致（同为4维BSND或同为3维TND）。
- 规格约束：
  - numIters：Sinkhorn迭代次数，要求为20（由前向算子决定，sum_out和norm_out的dim0为2*numIters=40）。
  - N：目前仅支持4。
  - C：
    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：要求尾轴C 128对齐，取值[1, 100000]。
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：尾轴C仅支持4096、7168。
    <!-- end id8 -->

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import mhc_pre_sinkhorn, mhc_pre_sinkhorn_backward

  B = 1
  S = 1024
  N = 4
  C = 4096

  x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu()
  phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
  alpha = torch.randn(3, dtype=torch.float32).npu()
  bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()

  hcMult = 4
  numIters = 20
  hcEps = 1e-6
  normEps = 1e-6

  # 前向，获取输出和中间缓存
  hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut = mhc_pre_sinkhorn(
      x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag=True
  )

  # 构造梯度
  gradHin = torch.randn_like(hin)
  gradHPost = torch.randn_like(hPost)
  gradHRes = torch.randn_like(hRes)

  # 反向
  gradX, gradPhi, gradAlpha, gradBias = mhc_pre_sinkhorn_backward(
      gradHin, gradHPost, gradHRes,
      x, phi, alpha, bias,
      hPre, hcBeforeNorm, invRms,
      sumOut, normOut, hcEps
  )
  ```

- TND格式调用（B*S合并为T）：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import mhc_pre_sinkhorn, mhc_pre_sinkhorn_backward

  B = 1
  S = 1024
  T = B * S
  N = 4
  C = 4096

  x = torch.randn(T, N, C, dtype=torch.bfloat16).npu()
  phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
  alpha = torch.randn(3, dtype=torch.float32).npu()
  bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()

  hcMult = 4
  numIters = 20
  hcEps = 1e-6
  normEps = 1e-6

  hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut = mhc_pre_sinkhorn(
      x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag=True
  )

  gradHin = torch.randn_like(hin)
  gradHPost = torch.randn_like(hPost)
  gradHRes = torch.randn_like(hRes)

  gradX, gradPhi, gradAlpha, gradBias = mhc_pre_sinkhorn_backward(
      gradHin, gradHPost, gradHRes,
      x, phi, alpha, bias,
      hPre, hcBeforeNorm, invRms,
      sumOut, normOut, hcEps
  )
  ```
