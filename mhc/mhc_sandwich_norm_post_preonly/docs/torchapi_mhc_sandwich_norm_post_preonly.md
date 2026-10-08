# mhc_sandwich_norm_post_preonly

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

- **接口功能**：融合 RMSNorm_0、MHC_Post、可选 RMSNorm_mid、MHC_Pre（仅 pre 分支）和 RMSNorm_1，输出下一层输入 `h_in_prime` 与 MHC_Post 中间结果 `x_2_out`。当 `return_h_in_f32=True` 时，额外返回 RMSNorm_1 的 FP32 结果 `h_in_f32`。

- **计算公式**：

    记 token 数为 $T$（BSND 格式下 $T=B\times S$），超连接路数为 $N$，hidden size 为 $D$。

    RMSNorm_0：

    $$
    x_{1,t,d} =
    \operatorname{Cast}_{dtype(h\_out)}
    \left(
    \frac{h\_out_{t,d}}
    {\sqrt{\frac{1}{D}\sum_{i=0}^{D-1}h\_out_{t,i}^{2}+norm\_eps}}
    \cdot gamma\_0_d
    \right)
    $$

    MHC_Post：

    $$
    x^{raw}_{2,t,n,d} =
    h\_post_{t,n}\cdot x_{1,t,d}
    +\sum_{j=0}^{N-1}h\_res_{t,j,n}\cdot residual_{t,j,d}
    $$

    当 `gamma_2` 非空时，在 $N\times D$ 维执行 RMSNorm_mid；为空时跳过：

    $$
    x_{2,t,n,d} =
    \operatorname{Cast}_{dtype(h\_out)}
    \left(
    \frac{x^{raw}_{2,t,n,d}}
    {\sqrt{\frac{1}{ND}\sum_{k=0}^{ND-1}(x^{raw}_{2,t,k})^2+norm\_eps}}
    \cdot gamma\_2_{n,d}
    \right)
    $$

    MHC_Pre（PreOnly）：

    $$
    inv\_rms_{pre,t} =
    \frac{1}
    {\sqrt{\frac{1}{ND}\sum_{k=0}^{ND-1}x_{2,t,k}^{2}+mhc\_pre\_norm\_eps}}
    $$

    $$
    x_{pre,t,n} =
    \left(\sum_{k=0}^{ND-1}x_{2,t,k}\cdot phi_{n,k}\right)
    \cdot inv\_rms_{pre,t}
    $$

    $$
    h_{pre,t,n}=\sigma(alpha\cdot x_{pre,t,n}+bias_n)+hc\_eps
    $$

    $$
    h_{in,t,d}=\sum_{n=0}^{N-1}h_{pre,t,n}\cdot x_{2,t,n,d}
    $$

    RMSNorm_1：

    $$
    h_{in\_f32,t,d}=
    \frac{h_{in,t,d}}
    {\sqrt{\frac{1}{D}\sum_{i=0}^{D-1}h_{in,t,i}^{2}+norm\_eps}}
    \cdot gamma\_1_d
    $$

    $$
    h_{in\_prime}=\operatorname{Cast}_{dtype(h\_out)}(h_{in\_f32})
    $$

## 函数原型

```python
cann_ops_transformer.mhc_sandwich_norm_post_preonly(
    h_out,
    residual,
    h_post,
    h_res,
    phi,
    alpha,
    bias,
    gamma_1,
    *,
    gamma_0=None,
    gamma_2=None,
    norm_eps=1e-6,
    mhc_pre_norm_eps=1e-6,
    hc_eps=1e-6,
    return_h_in_f32=False,
) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| h_out | Tensor | 必选 | RMSNorm_0 的输入。数据格式为 ND。 | float16、bfloat16 | [T, D] 或 [B, S, D] |
| residual | Tensor | 必选 | MHC 多路残差输入；dtype 必须与 `h_out` 一致。 | 同 h_out | [T, N, D] 或 [B, S, N, D] |
| h_post | Tensor | 必选 | MHC_Post 主分支权重。 | float32 | [T, N] 或 [B, S, N] |
| h_res | Tensor | 必选 | MHC_Post 残差混合权重，索引顺序为 $[j,n]$。 | float32 | [T, N, N] 或 [B, S, N, N] |
| phi | Tensor | 必选 | MHC_Pre 投影矩阵。 | float32 | [N, N*D] |
| alpha | Tensor | 必选 | Sigmoid 门控缩放系数。 | float32 | [1] |
| bias | Tensor | 必选 | Sigmoid 门控偏置。 | float32 | [N] |
| gamma_1 | Tensor | 必选 | RMSNorm_1 缩放权重。 | float32 | [D] |
| gamma_0 | Tensor | 可选 | RMSNorm_0 缩放权重。默认值为 `None`，该参数位置为接口预留；当前版本必须传入非空 Tensor。 | float32 | [D] |
| gamma_2 | Tensor | 可选 | RMSNorm_mid 缩放权重。默认值为 `None`；非空时启用 RMSNorm_mid，为空时跳过。 | float32 | [N*D] |
| norm_eps | float | 可选 | RMSNorm_0、RMSNorm_mid 和 RMSNorm_1 的数值稳定项，默认值为 `1e-6`。 | float | - |
| mhc_pre_norm_eps | float | 可选 | MHC_Pre 内部 RMS 的数值稳定项，默认值为 `1e-6`。 | float | - |
| hc_eps | float | 可选 | Sigmoid 门控结果的加性偏移，默认值为 `1e-6`。 | float | - |
| return_h_in_f32 | bool | 可选 | 是否额外返回 FP32 的 `h_in_f32`，默认值为 `False`。 | bool | - |

## 返回值说明

当 `return_h_in_f32=False` 时，返回 `(h_in_prime, x_2_out)`；当其为 `True` 时，返回 `(h_in_prime, x_2_out, h_in_f32)`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| h_in_prime | Tensor | 必选 | RMSNorm_1 FP32 结果转换为输入 dtype 后的下一层输入。 | 同 h_out | [T, D] 或 [B, S, D] |
| x_2_out | Tensor | 必选 | MHC_Post 以及可选 RMSNorm_mid 后的中间结果。 | 同 h_out | [T, N, D] 或 [B, S, N, D] |
| h_in_f32 | Tensor | 可选 | RMSNorm_1 的 FP32 结果，仅在 `return_h_in_f32=True` 时返回。 | float32 | [T, D] 或 [B, S, D] |

## 约束说明

- 该接口支持单算子模式调用，暂不支持 TorchAir 图模式调用。
- `h_out` 仅支持二维 TND 格式或三维 BSND 格式；所有格式相关输入必须采用同一种格式。
- `residual` 的 dtype 必须与 `h_out` 一致；其余 Tensor 输入必须为 float32。
- 当前版本固定支持 $N=4$、$D\in\{2560,5120\}$，token 总数 $T=B\times S$ 的支持范围为 $[1,256]$。
- `gamma_0=None` 为预留语义，当前版本不支持；未传入 `gamma_0` 时接口抛出 `ValueError`。
- `gamma_2=None` 时跳过 RMSNorm_mid；非空时 shape 必须为 $[N\times D]$。
- `norm_eps`、`mhc_pre_norm_eps` 和 `hc_eps` 必须大于或等于 0。
- 输入 Tensor 需为连续内存。输入包含 `NaN`、`Inf`，或中间结果超出数据类型范围时，不保证输出有效。

## 确定性/Batch一致性

- 确定性说明：在相同硬件、相同输入和相同属性条件下，默认支持确定性计算。

- Batch一致性说明：默认Batch一致性实现。

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

    h_in_prime, x_2_out = cann_ops_transformer.mhc_sandwich_norm_post_preonly(
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
        cann_ops_transformer.mhc_sandwich_norm_post_preonly(
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

- TorchAir 图模式调用：暂不支持。

