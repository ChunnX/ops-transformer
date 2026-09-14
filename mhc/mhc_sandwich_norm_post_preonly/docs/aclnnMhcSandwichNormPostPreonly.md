# aclnnMhcSandwichNormPostPreonly

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

MHC Sandwich Norm Post-Preonly 融合算子，将 MHC post-norm + optional RMSNorm\_mid + MHC pre-norm 融合为单个 kernel，减少 host-device 往返和中间 tensor 内存占用。

## 计算公式

**阶段 1: RMSNorm_0**
$$
x_{1,(b,s,d)} = \frac{h_{out,(b,s,d)}}{\sqrt{\frac{1}{D} \sum_{i=0}^{D-1} (h_{out,(b,s,i)})^2 + \epsilon}} \cdot \gamma_{0,(d)}
$$

**阶段 2: MHC_Post**
$$
x_{2,(b,s,n,d)} = \text{h\_post}_{(b,s,n)} \cdot x_{1,(b,s,d)} + \sum_{j=0}^{N-1} \left( \text{h\_res}_{(b,s,j,n)} \cdot \text{residual}_{(b,s,j,d)} \right)
$$

**阶段 2.5: RMSNorm_mid（可选）**
$$
\begin{align}
x_{2,rs} &= \text{reshape}(x_2, [B, S, N \times D]) \\
\text{if $\gamma\_2$ 非空:} \quad x_{2,(b,s,n,d)} &= \frac{x_{2, rs,(b,s,n\times d)}}{\sqrt{\frac{1}{n*D} \sum_{i=0}^{n*D-1} (x_{2,(b,s,i)})^2 + \epsilon}} \cdot \gamma_{2,(n\times d)}
\end{align}
$$

**阶段 3: MHC_Pre (PreOnly)**
$$
\begin{align}
\text{invRms}_{pre} &= \frac{1}{\sqrt{\frac{1}{N \times D} \sum_{k=0}^{N \times D - 1} (x_{2,rs,(b,s,k)})^2 + \epsilon}} \\
x_{pre} &= (x_{2,rs} \cdot \phi^T) \cdot \text{invRms}_{pre} \\
h_{pre} &= \sigma(\alpha_0 \cdot x_{pre} + \text{bias}) + \epsilon_{hc} \\
h_{in} &= \sum_{n=0}^{N-1} h_{pre,(b,s,n)} \cdot x_{2,(b,s,n,:)}
\end{align}
$$

**阶段 4: RMSNorm_1**
$$
h_{in\_prime,(b,s,d)} = \frac{h_{in,(b,s,d)}}{\sqrt{\frac{1}{D} \sum_{i=0}^{D-1} (h_{in,(b,s,i)})^2 + \epsilon}} \cdot \gamma_{1,(d)}
$$

---

## 函数原型

### v1接口
```c
// 第一段接口
aclnnStatus aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize(
    const aclTensor *hOut,
    const aclTensor *residual,
    const aclTensor *hPost,
    const aclTensor *hRes,
    const aclTensor *phi,
    const aclTensor *alpha0,
    const aclTensor *bias,
    const aclTensor *gamma0,
    const aclTensor *gamma1,
    const aclTensor *gamma2,
    double normEps,
    double hcEps,
    const aclTensor *hInPrimeOut,
    const aclTensor *x2OutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)

// 第二段接口
aclnnStatus aclnnMhcSandwichNormPostPreonly(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```
### v2接口
```c
// 第一段接口
aclnnStatus aclnnMhcSandwichNormPostPreonlyV2GetWorkspaceSize(
    const aclTensor *hOut,
    const aclTensor *residual,
    const aclTensor *hPost,
    const aclTensor *hRes,
    const aclTensor *phi,
    const aclTensor *alpha0,
    const aclTensor *bias,
    const aclTensor *gamma0,
    const aclTensor *gamma1,
    const aclTensor *gamma2,
    double normEps,
    double hcEps,
    bool returnHInF32,
    const aclTensor *hInPrimeOut,
    const aclTensor *x2OutOut,
    const aclTensor *hInF32Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)

// 第二段接口
aclnnStatus aclnnMhcSandwichNormPostPreonlyV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

### Shape 格式

支持两种 shape 格式，由 hOut 的维度数决定：

**BSND 格式 (hOut 3D)** — 批次格式，ATK / 实际推理使用

| 参数名 | shape | ndim |
|:---|:---|:---|
| hOut | [B, S, D] | 3 |
| residual | [B, S, N, D] | 4 |
| hPost | [B, S, N] | 3 |
| hRes | [B, S, N, N] | 4 |
| hInPrimeOut (输出) | [B, S, D] | 3 |
| x2OutOut (输出) | [B, S, N, D] | 4 |
| hInF32Out (可选输出，仅v2接口支持) | [B, S, D] | 3 |


**TND 格式 (hOut 2D)** — 扁平格式，T=B\*S 已预合并

| 参数名 | shape | ndim |
|:---|:---|:---|
| hOut | [T, D] | 2 |
| residual | [T, N, D] | 3 |
| hPost | [T, N] | 2 |
| hRes | [T, N, N] | 3 |
| hInPrimeOut (输出) | [T, D] | 2 |
| x2OutOut (输出) | [T, N, D] | 3 |
| hInF32Out (可选输出，仅v2接口支持) | [T, D] | 2 |

**格式无关的输入** (两种格式下 shape 相同):

| 参数名 | shape |
|:---|:---|
| phi | [N, N\*D] |
| alpha0 | [N] (aclnn 对齐到 8 元素) |
| bias | [N] (同 alpha_0) |
| gamma0 | [D] |
| gamma1 | [D] |
| gamma2 | [N\*D] 或 nullptr |

**一致性约束**: 所有格式相关的输入必须使用同一格式。规则：
- `residual.ndim == hOut.ndim + 1`
- `hPost.ndim == hOut.ndim`
- `hRes.ndim == hOut.ndim + 1`

tiling 通过 `hOut.GetDimNum()` 判断格式，并据此解析其余 tensor 的 shape。格式不一致时会读取到错误的维度值。

### 输入 Tensor

| 参数名 | 输入/输出 | 描述 | dtype | shape | 约束 |
|:---|:---|:---|:---|:---|:---|
| hOut | 输入 | MHC post-norm 前的hidden layer输出 | BF16, FP16 | [B,S,D] 或 [T,D] | B\*S 或 T ∈ [1, 256], D ∈ {2560, 5120} |
| residual | 输入 | 多头残差连接 | 与 hOut 一致 | [B,S,N,D] 或 [T,N,D] | N=4 (固定), dtype 与 hOut 联动 |
| hPost | 输入 | post-norm 系数 | FP32 | [B,S,N] 或 [T,N] | ndim 与 hOut 一致 |
| hRes | 输入 | 残差加权系数 | FP32 | [B,S,N,N] 或 [T,N,N] | ndim == hOut.ndim + 1 |
| phi | 输入 | pre-norm 投影矩阵 | FP32 | [N, N\*D] | N\*D 维度求和累积, range 对精度影响大 |
| alpha0 | 输入 | sigmoid 缩放因子 | FP32 | [N] | aclnn 对齐到 8 元素, 实际只用前 N 个 |
| bias | 输入 | sigmoid 偏置 | FP32 | [N] | 同 alpha_0 |
| gamma0 | 输入 | RMSNorm\_0 缩放权重 | FP32 | [D] | |
| gamma1 | 输入 | RMSNorm\_1 缩放权重 | FP32 | [D] | |
| gamma2 | 输入 (可选) | RMSNorm\_mid 缩放权重 | FP32 | [N\*D] 或 nullptr | 非空时启用 RMSNorm\_mid, 空时跳过 |

### 属性

| 参数名 | 类型 | 默认值 | 描述 | 约束 |
|:---|:---|:---|:---|:---|
| normEps | float | 1e-6 | RMSNorm 中 rsqrt(mean\_sq + eps) 的 epsilon | 必须 >= 0 |
| hcEps | float | 1e-6 | sigmoid gate 输出的 epsilon: gate = σ(z) + hcEps | 必须 >= 0 |
| returnHInF32| bool | false | 判断是否将第一个输出cast为float数据类型 | 仅v2接口支持|

### 输出 Tensor

| 参数名 | 输入/输出 | 描述 | dtype | shape |
|:---|:---|:---|:---|:---|
| hInPrimeOut | 输出 | 经 post-norm + gate + pre-norm 后的hidden layer输出 | 与 hOut 一致 | [B,S,D] 或 [T,D] (与 hOut 格式一致) |
| x2OutOut | 输出 | gate 前的中间结果 | 与 hOut 一致 | [B,S,N,D] 或 [T,N,D] (与 residual 格式一致) |
| hInF32Out | 可选输出 | 仅v2接口支持,hInPrimeOut转成float数据类型的输出| float | [B,S,D] 或 [T,D] (与 hOut 格式一致) |

### dtype 联动规则

- hOut, residual, hInPrimeOut, x2OutOut: 同一dtype 组 (BF16 | FP16)，由 OpDef DataType 列表的位置索引联动
- hPost, hRes, phi, alpha0, bias, gamma0, gamma1, gamma2, hInF32Out: 均为 FP32

### 返回值

返回 `aclnnStatus` 状态码，第一段接口主要完成入参校验，异常场景如下：

| 返回值 | 错误码 | 描述 |
|:---|:---|:---|
| ACLNN_SUCCESS | 0 | 成功 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选输入为 nullptr |
| ACLNN_ERR_PARAM_INVALID | 161002 | shape/dtype 不满足约束 |

## 约束说明

### 确定性计算

在相同硬件、相同输入条件下，算子结果确定性一致。

### 公共约束

1. 输入约束：
   - hOut 和 residual 的 dtype 必须一致
   - 支持两种 shape 格式：BSND (hOut 3D) 或 TND (hOut 2D)，所有格式相关输入必须使用同一格式
   - alpha0 和 bias 的元素数需 ≥ N（实际对齐到 8 元素传入）
   - gamma2 传入时 shape 为 [N\*D]，不传入时为 nullptr
   - returnHInF32仅v2接口支持
   - 算子入参输入值域范围为inf/-inf/nan或者输入值域或中间计算结果出现超过float16/bfloat16数据类型范围时，无法确定输出结果有效性。
2. 内存约束：
   - 所有输入 Tensor 必须连续 (contiguous)
   - 输出 Tensor 由调用方分配，shape 需与 InferShape 结果一致


### 规格约束

| 规格项 | 规格 | 规格说明 |
|:---|:---|:---|
| B\*S 或 T | [1, 256] | token 总数范围 |
| N | 4 | head 数量，固定值 |
| D | {2560, 5120} | embedding 维度，仅支持两种取值 |
| norm_eps | >= 0 | eps<0 导致 sqrt(负数)=NaN |
| hc_eps | > 0 | 同 norm_eps |
| dtype (hOut, residual) | BF16, FP16 | 两者必须一致 |
| dtype (其余输入, h_in_f32) | FP32 | 固定 |

### 典型值

| 规格项 | 典型值 |
|:---|:---|
| B | 1, 2, 4, 20, 24|
| S | 1, 2, 4, 8, 16 |
| D | 2560, 5120 |
| norm_eps | 1e-6 |
| hc_eps | 1e-6 |
| dtype | BF16 |

## 实现说明

### 调用链路

```
torch.ops.custom.npu_mhc_sandwich_norm_post_preonly(...)
  → ops_def_registration.cpp (torch schema)
  → csrc/npu_*.cpp (construct_output_tensors + EXEC_NPU_CMD_V1)
  → aclnnMhcSandwichNormPostPreonly (CANN Exc-generated, op_host_aclnn 模式)
  → op_host/_def.cpp (OpDef 声明)
  → op_host/_tiling.cpp (tiling 计算)
  → op_kernel/ (AscendC kernel)
```
```
torch.ops.custom.npu_mhc_sandwich_norm_post_preonly_v2(...)
  → ops_def_registration.cpp (torch schema)
  → csrc/npu_*.cpp (construct_output_tensors_v2 + EXEC_NPU_CMD_V1)
  → aclnnMhcSandwichNormPostPreonlyV2 (CANN Exc-generated, op_host_aclnn 模式)
  → op_host/_def.cpp (OpDef 声明)
  → op_host/_tiling.cpp (tiling 计算)
  → op_kernel/ (AscendC kernel)
```

## 调用示例

### Python (推荐)

```python
import torch
import torch_npu
import omni_custom_ops

# eager 模式
h_in_prime, x_2_out = torch.ops.custom.npu_mhc_sandwich_norm_post_preonly(
    h_out.npu(),       # [B, S, D], bf16
    residual.npu(),    # [B, S, N, D], bf16
    h_post.npu(),      # [B, S, N], fp32
    h_res.npu(),       # [B, S, N, N], fp32
    phi.npu(),         # [N, N*D], fp32
    alpha_0.npu(),     # [N], fp32
    bias.npu(),        # [N], fp32
    gamma_0.npu(),     # [D], fp32
    gamma_1.npu(),     # [D], fp32
    gamma_2.npu(),     # [N*D], fp32; 或 None
    norm_eps=1e-6,
    hc_eps=1e-6,
)

h_in_prime, x_2_out, h_in_f32 = torch.ops.custom.npu_mhc_sandwich_norm_post_preonly_v2(
    h_out.npu(),       # [B, S, D], bf16
    residual.npu(),    # [B, S, N, D], bf16
    h_post.npu(),      # [B, S, N], fp32
    h_res.npu(),       # [B, S, N, N], fp32
    phi.npu(),         # [N, N*D], fp32
    alpha_0.npu(),     # [N], fp32
    bias.npu(),        # [N], fp32
    gamma_0.npu(),     # [D], fp32
    gamma_1.npu(),     # [D], fp32
    gamma_2 = gamma_2.npu(),     # [N*D], fp32; 或 None
    norm_eps=1e-6,
    hc_eps=1e-6,
    return_h_in_f32=True,        # 输出可选参数h_in_f32
)
```
