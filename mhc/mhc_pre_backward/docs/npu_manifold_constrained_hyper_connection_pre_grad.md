# MhcPreBackward

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200/300/500 推理产品</term>|      ×     |

## 功能说明

- 接口功能：MhcPreBackward算子是MhcPre的反向算子，首先说明一下正向算子的功能。

- 正向算子功能：基于一系列计算得到MHC架构中hidden层的$H^{res}$和$H^{post}$投影矩阵以及Atten或MLP层的输入矩阵$h^{in}$。

## 计算公式

$$
\begin{aligned}
\vec{x}'_{l} &=\operatorname{RmsNorm}(\vec{x_{l}})\\
\tilde{\mathcal{H}}^{pre}_l &= \alpha^{pre}_{l} ·(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
\tilde{\mathcal{H}}^{post}_l &= \alpha^{post}_{l} ·(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
\tilde{\mathcal{H}}^{res}_l &= \alpha^{res}_{l} ·mat(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l},\\
\\[1em]
\mathcal{H}^{pre}_l &= \sigma (\tilde{\mathcal{H}}^{pre}_{l})\\
\mathcal{H}^{post}_l &= 2\sigma (\tilde{\mathcal{H}}^{post}_{l})\\
\mathcal{H}^{res}_l &= \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}^{res}_{l})  + hc\_eps \\
\\[1em]
h_{in} &=\vec{x^{'}_{l}}\mathcal{H}^{pre}_l
\end{aligned}
$$

其中：
$$
\operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+norm\_eps}
$$

计算公式经过变换：
$$
\begin{aligned}
\varphi_l &: \text{float32} \quad && [nC, n^2 + 2n] \quad &(1) \\
\vec{\mathbf{x}}_l &: \text{bfloat16} \quad && [1, nC] \quad &(2) \\
\alpha_l^{pre}, \alpha_l^{post}, \alpha_l^{res} &: \text{float32} \quad && Scalars \quad &(3) \\
\mathbf{b}_l &: \text{float32} \quad && [1, n^2 + 2n] \quad &(4) \\
[\tilde{\tilde{\mathcal{H}}}^{pre}_l, \tilde{\tilde{\mathcal{H}}}^{post}_l, \tilde{\tilde{\mathcal{H}}}^{res}_l] &: \text{float32} \quad &&= \vec{\mathbf{x}}_l \varphi_l \quad &(5) \\
r &: \text{float32} \quad &&= \|\vec{\mathbf{x}}_l\|_2 / \sqrt{nC} \quad &(6) \\
[\tilde{\mathcal{H}}^{pre}_l, \tilde{\mathcal{H}}^{post}_l, \tilde{\mathcal{H}}^{res}_l] &: \text{float32} \quad &&= 1/r [\alpha_l^{pre}\tilde{\tilde{\mathcal{H}}}^{pre}_l, \alpha_l^{post}\tilde{\tilde{\mathcal{H}}}^{post}_l, \alpha_l^{res}\tilde{\tilde{\mathcal{H}}}^{res}_l] + \mathbf{b}_l \quad &(7) \\
\mathcal{H}^{pre}_l &: \text{float32} &&= \sigma(\tilde{\mathcal{H}}^{pre}_l) \quad &(8) \\
\mathcal{H}^{post}_l &: \text{float32} &&= 2\sigma (\tilde{\mathcal{H}}^{post}_{l}) \quad &(9) \\
\mathcal{H}^{res}_l &: \text{float32} &&= \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}^{res}_{l})  + hc\_eps &(10) \\
\end{aligned}
$$

其中(10)单独做一个算子，不放在mHC_pre算子中。

---

## 函数原型
```
npu_mhc_pre_backward(Tensor x, Tensor phi, Tensor alpha, Tensor h_in_grad, Tensor h_post_grad, Tensor h_comb_before_grad, Tensor inv_rms, Tensor mm_res, Tensor h_pre, Tensor h_post, *, Tensor? gamma=None, float hc_eps=1e-6) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```
### 参数说明
| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
| x | 输入 | 待计算数据，表示网络中mHC层的输入数据 | 必选参数，不能为空Tensor | BFLOAT16 或 FLOAT16 | ND | ($B,S,n,D$)、($S,B,n,D$) 或 ($T,n,D$) | √ |
| phi | 输入 | mHC的参数矩阵 | 必选参数，不能为空Tensor | FLOAT32 | ND | ($n^2+2n, n*D$) | √ |
| gamma | 输入 | RmsNorm的缩放系数 | 必选参数，不能为空Tensor | FLOAT32 | ND | ($n, D$) | √ |
| alpha | 输入 | mHC的缩放参数 | 必选参数，不能为空Tensor | FLOAT32 | - | (3) | - |
| hc_eps | 可选输入 | $H_{pre}$的sigmoid后的eps参数 | 必选参数 | FLOAT32 | - | - | - |
| grad_h_in | 输入 | h_in作为Atten/MLP层的输入。正向输出h_in对应的梯度 | 必选参数 | BFLOAT16 或 FLOAT16  | ND | ($B,S,D$)、($S,B,D$)  或 ($T,D$)  | - |
| grad_h_post | 输入 | 正向输出$H_{post}$的梯度 | 必选参数 | FLOAT32 | ND | ($B,S,D$)、($S,B,D$)或 ($T,D$)  | - |
| grad_h_res | 输入 | 正向输出$H_{res}$的梯度 | 必选参数 | FLOAT32 | ND | ($B,S,n,n$)、($S,B,n,n$) 或 ($T,n,n$) | - |
| inv_rms | 输入 | 正向RmsRorm计算得到的1/r | 必选参数 | FLOAT32 | ND | ($B,S$)、($S,B$) 或 ($T$) | - |
| out_mm_res | 输入 | 正向计算流x@phi的结果 | 必选参数 | FLOAT32 | ND | ($B,S,n^2+2n$)、($S,B,n^2+2n$) 或 ($T,n^2+2n$) | - |
| h_pre | 输入 | 正向做完sigmoid计算之后的h_pre矩阵 | 必选参数 | FLOAT32 | ND | ($B,S,n$)、($S,B,n$) 或 ($T,n$) | - |
| h_post | 输入 | 正向的h_post输出 | 必选参数 | FLOAT32 | ND | ($B,S,n$)、($S,B,n$)或 ($T,n$)  | - |

### 返回值参数：
| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
| grad_x | 输出 | x对应的梯度 | 必选参数 | BFLOAT16 或 FLOAT16 | ND | ($B,S,n,D$)、($S,B,n,D$) 或 ($T,n,D$)  | - |
| grad_phi | 输出 | phi对应的梯度 | 必选参数 | FLOAT32 | ND | ($n^2+2n, n*D$)  | - |
| grad_alpha | 输出 | alpha对应的梯度 | 必选参数 | FLOAT32 | ND | (3)  | - |
| grad_bias | 输出 | bias对应的梯度 | 必选参数 | FLOAT32 | ND | ($n^2+2n$)  | - |
| grad_gamma | 输出 | gamma对应的梯度 | 必选参数 | FLOAT32 | ND | ($n, D$)  | - |

## 约束说明

### 确定性计算

- MhcPreBackward 算子默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束
- 输入约束：
   - 输入Tensor `x`、`phi`、`alpha`、`gamma`、`grad_h_in`、`grad_h_post`、`grad_h_res`、`inv_rms`、`h_mix`、`h_pre`、`h_post` 不能为空，且必须为Device侧Tensor。

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 1~65536 | B*S 或T支持1~65536范围以内。|
| n | 4、6、8 | n值目前支持4, 6, 8。|
| D | 512~16384 | D支持512~16384范围以内。|

### 典型值

| 规格项 | 典型值 |
|:--- |:--- |
| T或 B*S | 1024/2048/4096 |
| n | 4（推荐） |
| D | 2560/4096（推荐） |
| eps | 1e-6（推荐） |

## 调用示例
- 单算子模式调用
```python
  import torch
  import torch_npu
  import numpy as np
  import omni_training_custom_ops

  T=1024
  n=4
  D=2560
  x = torch.randn(B, S, n, D).bfloat16()
  phi = torch.randn(n*D, n*n + 2*n)
  alpha = torch.tensor([1.1, 0.9, 1.05])
  bias = torch.randn(n*n + 2*n) * 0.1
  gamma = torch.randn(n, D)

  x_ = x.detach().clone().requires_grad_(True)
  phi_ = phi.detach().clone().requires_grad_(True)
  alpha_ = alpha.detach().clone().requires_grad_(True)
  bias_ = bias.detach().clone().requires_grad_(True)

  dh_in = torch.randn(B, S, D).bfloat16()
  dh_post = torch.randn(B, S, n)
  dh_res = torch.randn(B, S, n, n)

  # 正向传播
  h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
      x_, phi_, alpha_, bias_, outflag=True
  )

  # 调用npu接口
  dx1, dphi1, da1, db1, dgamma1 = torch.ops.custom.npu_mhc_pre_backward(
      x.npu(), phi.transpose(-1, -2).npu(), alpha.npu(),
      dh_in.npu(), dh_post.npu(), dh_res.npu(),
      inv_rms.npu(), h_mix.npu(), h_pre.npu(), h_post.npu(), gamma=gamma.npu()
  )
```