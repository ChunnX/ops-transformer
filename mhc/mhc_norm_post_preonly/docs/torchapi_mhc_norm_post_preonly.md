# mhc_norm_post_preonly

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- **接口功能**：融合 RMSNorm_0、MHC_Post、可选 RMSNorm_mid、MHC_Pre（仅 pre 分支）和 RMSNorm_1，输出下一层输入 h_in_prime 与 MHC_Post 中间结果 x_2_out。当 return_h_in_f32=True 时，额外返回 RMSNorm_1 的 FP32 结果 h_in_f32。


- **计算公式**：

    以下公式按保留 B、S 前导维的 [B,S,...] 形式表示；[T,...] 形式将 B*S 展平为 T，后续维度保持不变。

    RMSNorm_0：

    $$
    \begin{aligned}
    h\_out\_fp32_{b,s,d} &= \operatorname{FP32}(h\_out_{b,s,d}), \\
    inv\_rms\_0_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}h\_out\_fp32_{b,s,i}^{2}+norm\_eps\right), \\
    x\_1_{b,s,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(h\_out\_fp32_{b,s,d}\cdot inv\_rms\_0_{b,s}\cdot gamma\_0_d\right).
    \end{aligned}
    $$

    MHC_Post：

    $$
    \begin{aligned}
    term1_{b,s,n,d} &= \operatorname{FP32}(h\_post_{b,s,n})\cdot \operatorname{FP32}(x\_1_{b,s,d}), \\
    term2_{b,s,n,d} &= \sum_{j=0}^{N-1}\operatorname{FP32}(h\_res_{b,s,j,n})\cdot \operatorname{FP32}(residual_{b,s,j,d}), \\
    x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(term1_{b,s,n,d}+term2_{b,s,n,d}\right).
    \end{aligned}
    $$

    其中 j 为归约 head，n 为输出 head。

    当 gamma_2 非空时，在 $N\times D$ 维执行 RMSNorm_mid；为空时跳过：

    $$
    \begin{aligned}
    x\_2\_flat &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
    inv\_rms\_mid_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_flat_{b,s,k}^{2}+norm\_eps\right), \\
    x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(\operatorname{reshape}\left(x\_2\_flat\cdot inv\_rms\_mid\cdot gamma\_2,[B,S,N,D]\right)_{b,s,n,d}\right).
    \end{aligned}
    $$

    当 gamma_2 为空时，跳过本阶段，x_2 保持 MHC_Post 的输出；随后固定保存 $x\_2\_out=x\_2$。

    MHC_Pre（PreOnly）：

    $$
    \begin{aligned}
    x\_2\_rs &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
    inv\_rms\_pre_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_rs_{b,s,k}^{2}+mhc\_pre\_norm\_eps\right), \\
    proj\_w &= phi^T, \\
    x\_pre &= (x\_2\_rs\cdot proj\_w)\cdot inv\_rms\_pre, \\
    h\_pre_{b,s,n} &= \operatorname{sigmoid}(alpha\cdot x\_pre_{b,s,n}+bias_n)+hc\_eps, \\
    h\_in_{b,s,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(\sum_{n=0}^{N-1}h\_pre_{b,s,n}\cdot \operatorname{FP32}(x\_2_{b,s,n,d})\right).
    \end{aligned}
    $$

    RMSNorm_1：

    $$
    \begin{aligned}
    h\_in\_rms_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}\operatorname{FP32}(h\_in_{b,s,i})^{2}+norm\_eps\right), \\
    h\_in\_f32_{b,s,d} &= \operatorname{FP32}(h\_in_{b,s,d})\cdot h\_in\_rms_{b,s}\cdot \operatorname{FP32}(gamma\_1_d), \\
    h\_in\_prime &= \operatorname{Cast}_{dtype(h\_out)}(h\_in\_f32).
    \end{aligned}
    $$

    norm_eps 用于 RMSNorm_0、RMSNorm_mid 和 RMSNorm_1，mhc_pre_norm_eps 仅用于 MHC_Pre 归一化，hc_eps 仅用于 pre 门控。三者均按 FLOAT 精度参与计算。

## 函数原型

```python
cann_ops_transformer.mhc_norm_post_preonly(
    h_out,
    residual,
    h_post,
    h_res: Optional[Tensor],
    phi,
    alpha,
    bias,
    gamma_1,
    *,
    pre_mix: Optional[Tensor] = None,
    gamma_0=None,
    gamma_2=None,
    norm_eps=1e-6,
    mhc_pre_norm_eps=1e-6,
    hc_eps=1e-6,
    return_h_in_f32=False,
) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| h_out | Tensor | 必选 | RMSNorm_0 的输入。数据格式为 ND。 | float16、bfloat16 | [T, D] 或 [B, S, D] |
| residual | Tensor | 必选 | MHC 多路残差输入；dtype 必须与 h_out 一致。 | 同 h_out | [T, N, D] 或 [B, S, N, D] |
| h_post | Tensor | 必选 | MHC_Post 主分支权重。 | float32 | [T, N] 或 [B, S, N] |
| h_res | Tensor | 可选 | mHC 的 hRes 变换矩阵，是做完 Sinkhorn 变换后的双随机矩阵；索引顺序为 $[j,n]$。接口预留参数，暂不支持传 None。 | float32 | [T, N, N] 或 [B, S, N, N] |
| phi | Tensor | 必选 | MHC_Pre 投影矩阵。 | float32 | [N, N*D] |
| alpha | Tensor | 必选 | Sigmoid 门控缩放系数。 | float32 | [1] |
| bias | Tensor | 必选 | Sigmoid 门控偏置。 | float32 | [N] |
| gamma_1 | Tensor | 必选 | RMSNorm_1 缩放权重。 | float32 | [D] |
| pre_mix | Tensor | 可选 | 传入时，h_in 的加权求和直接使用 pre_mix 作为权重（典型用法：填入上一轮 h_pre 的输出）；不传入时，h_in 使用本轮内部计算的 h_pre 进行加权求和。不支持非连续。接口预留参数，当前仅支持传入 None。 | float32 | [T, N] 或 [B, S, N] |
| gamma_0 | Tensor | 可选 | RMSNorm_0 缩放权重。当前暂不支持传 None。 | float32 | [D] |
| gamma_2 | Tensor | 可选 | RMSNorm_mid 缩放权重。默认值为 None；非空时启用 RMSNorm_mid，为空时跳过。 | float32 | [N*D] |
| norm_eps | float | 可选 | RMSNorm_0、RMSNorm_mid 和 RMSNorm_1 的数值稳定项，默认值为 1e-6。 | float | - |
| mhc_pre_norm_eps | float | 可选 | MHC_Pre 内部 RMS 的数值稳定项，默认值为 1e-6。 | float | - |
| hc_eps | float | 可选 | Sigmoid 门控结果的加性偏移，默认值为 1e-6。 | float | - |
| return_h_in_f32 | bool | 可选 | 是否额外返回 FP32 的 h_in_f32，默认值为 False。 | bool | - |

## 返回值说明

当 return_h_in_f32=False 时，返回 (h_in_prime, x_2_out)；当其为 True 时，返回 (h_in_prime, x_2_out, h_in_f32)。接口预留可选输出参数 h_pre，当前暂不支持输出。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| h_in_prime | Tensor | 必选 | RMSNorm_1 FP32 结果转换为输入 dtype 后的下一层输入。 | 同 h_out | [T, D] 或 [B, S, D] |
| x_2_out | Tensor | 必选 | MHC_Post 以及可选 RMSNorm_mid 后的中间结果。 | 同 h_out | [T, N, D] 或 [B, S, N, D] |
| h_in_f32 | Tensor | 可选 | RMSNorm_1 的 FP32 结果，仅在 return_h_in_f32=True 时返回。 | float32 | [T, D] 或 [B, S, D] |
| h_pre | Tensor | 可选 | 本轮内部计算的 pre 结果（sigmoid 后的加权系数），可作为下一轮 pre_mix 的输入。不支持非连续。接口预留参数，当前暂不支持输出。 | float32 | [T, N] 或 [B, S, N] |

## 约束说明

- 支持单算子模式、TorchAir 图模式调用。
- 仅支持 N=4。

## 确定性/Batch一致性

默认确定性实现，不支持 Batch 一致性。

## 调用示例

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    b, s, n, d = 1, 4, 4, 2560
    input_dtype = torch.bfloat16
    device = "npu"

    h_out = torch.randn(b, s, d, dtype=input_dtype, device=device)
    residual = torch.randn(b, s, n, d, dtype=input_dtype, device=device)
    h_post = torch.randn(b, s, n, dtype=torch.float32, device=device)
    h_res = torch.randn(b, s, n, n, dtype=torch.float32, device=device)
    phi = torch.randn(n, n * d, dtype=torch.float32, device=device) * 0.01
    alpha = torch.ones(1, dtype=torch.float32, device=device)
    bias = torch.zeros(n, dtype=torch.float32, device=device)
    gamma_0 = torch.ones(d, dtype=torch.float32, device=device)
    gamma_1 = torch.ones(d, dtype=torch.float32, device=device)

    h_in_prime, x_2_out = cann_ops_transformer.mhc_norm_post_preonly(
        h_out,
        residual,
        h_post,
        h_res,
        phi,
        alpha,
        bias,
        gamma_1,
        gamma_0=gamma_0,
        gamma_2=None,
        norm_eps=1e-6,
        mhc_pre_norm_eps=1e-6,
        hc_eps=1e-6,
    )
    torch_npu.npu.synchronize()

    print("h_in_prime:", h_in_prime.shape, h_in_prime.dtype)
    print("x_2_out:", x_2_out.shape, x_2_out.dtype)

    h_in_prime, x_2_out, h_in_f32 = (
        cann_ops_transformer.mhc_norm_post_preonly(
            h_out,
            residual,
            h_post,
            h_res,
            phi,
            alpha,
            bias,
            gamma_1,
            gamma_0=gamma_0,
            gamma_2=None,
            return_h_in_f32=True,
        )
    )
    torch_npu.npu.synchronize()

    print("h_in_f32:", h_in_f32.shape, h_in_f32.dtype)
    ```

- TorchAir 图模式调用：支持。
