# torch.ops.custom.npu\_mhc\_sandwich\_norm\_post\_preonly

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

本算子将五个阶段的小算子深度融合，减少头开销和部分HBM访存：

1. **RMSNorm_0**：对 h_out 做 D 维的均方根归一化
2. **MHC_Post**：门控+混合机制，将归一化后的 h_out 与多头残差线性组合
3. **RMSNorm_mid**：对 x2 [B,S,n,D] 的每个 head 独立在 D 维上做 RMSNorm
4. **MHC_Pre（PreOnly）**：reshape + InvRMS + MatMul + sigmoid gate → 加权求和得 h_in
5. **RMSNorm_1**：对 h_in 做 D 维的均方根归一化

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

```python
torch.ops.custom.npu_mhc_sandwich_norm_post_preonly(
    h_out: Tensor,
    residual: Tensor,
    h_post: Tensor,
    h_res: Tensor,
    phi: Tensor,
    alpha_0: Tensor,
    bias: Tensor,
    gamma_0: Tensor,
    gamma_1: Tensor,
    gamma_2: Optional[Tensor] = None,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]
```

``` python
torch.ops.custom.npu_mhc_sandwich_norm_post_preonly_v2(
    h_out,
    residual,
    h_post,
    h_res,
    phi,
    alpha_0,
    bias,
    gamma_0,
    gamma_1,
    *, 
    gamma_2=None,
    norm_eps=1e-6,
    hc_eps=1e-6,
    return_h_in_f32=False,
) -> Tensor, Tensor, Tensor
```

### 参数说明

支持两种 shape 格式，由 h\_out 的维度数决定：

- **BSND 格式 (h\_out 3D)**：h\_out=[B,S,D], residual=[B,S,N,D]（批次格式，ATK / 实际推理使用）
- **TND 格式 (h\_out 2D)**：h\_out=[T,D], residual=[T,N,D]（扁平格式，T=B\*S 已预合并）

格式无关的输入（两种格式下 shape 相同）：phi, alpha\_0, bias, gamma\_0, gamma\_1, gamma\_2

### npu_mhc_sandwich_norm_post_preonly
#### 输入

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) | 约束 |
|:---|:---|:---|:---|:---|:---|
| h_out | 输入 | MHC post-norm 前的hidden layer输出 | BF16, FP16 | [B,S,D] 或 [T,D] | B\*S 或 T ∈ [1, 256], D ∈ {2560, 5120} |
| residual | 输入 | 多头残差连接 | 与 h\_out 一致 | [B,S,N,D] 或 [T,N,D] | N=4, ndim == h\_out.ndim + 1 |
| h_post | 输入 | post-norm 系数 | FP32 | [B,S,N] 或 [T,N] | ndim == h\_out.ndim |
| h_res | 输入 | 残差加权系数 | FP32 | [B,S,N,N] 或 [T,N,N] | ndim == h\_out.ndim + 1 |
| phi | 输入 | pre-norm 投影矩阵 | FP32 | [N, N\*D] | |
| alpha_0 | 输入 | sigmoid 缩放因子 | FP32 | [N] | |
| bias | 输入 | sigmoid 偏置 | FP32 | [N] | |
| gamma_0 | 输入 | RMSNorm\_0 缩放权重 | FP32 | [D] | |
| gamma_1 | 输入 | RMSNorm\_1 缩放权重 | FP32 | [D] | |
| gamma_2 | 输入 (可选) | RMSNorm\_mid 缩放权重 | FP32 | [N\*D] 或 None | 非空时启用 RMSNorm\_mid |
| norm_eps | 输入 | RMSNorm epsilon | float | 标量 | 默认 1e-6, 必须 > 0 |
| hc_eps | 输入 | sigmoid gate epsilon | float | 标量 | 默认 1e-6, 必须 > 0 |

#### 返回值

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|
| h\_in\_prime | 输出 | 经 post-norm + gate + pre-norm 后的hidden layer输出 | 与 h\_out 一致 | [B,S,D] 或 [T,D] |
| x\_2\_out | 输出 | gate 前的中间结果 | 与 h\_out 一致 | [B,S,N,D] 或 [T,N,D] |

### npu_mhc_sandwich_norm_post_preonly_v2

#### 输入
| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) | 约束 |
|:---|:---|:---|:---|:---|:---|
| h_out | 输入 | MHC post-norm 前的hidden layer输出 | BF16, FP16 | [B,S,D] 或 [T,D] | B\*S 或 T ∈ [1, 256], D ∈ {2560, 5120} |
| residual | 输入 | 多头残差连接 | 与 h\_out 一致 | [B,S,N,D] 或 [T,N,D] | N=4, ndim == h\_out.ndim + 1 |
| h_post | 输入 | post-norm 系数 | FP32 | [B,S,N] 或 [T,N] | ndim == h\_out.ndim |
| h_res | 输入 | 残差加权系数 | FP32 | [B,S,N,N] 或 [T,N,N] | ndim == h\_out.ndim + 1 |
| phi | 输入 | pre-norm 投影矩阵 | FP32 | [N, N\*D] | |
| alpha_0 | 输入 | sigmoid 缩放因子 | FP32 | [N] | |
| bias | 输入 | sigmoid 偏置 | FP32 | [N] | |
| gamma_0 | 输入 | RMSNorm\_0 缩放权重 | FP32 | [D] | |
| gamma_1 | 输入 | RMSNorm\_1 缩放权重 | FP32 | [D] | |
| gamma_2 | 输入 (可选) | RMSNorm\_mid 缩放权重 | FP32 | [N\*D] 或 None | 非空时启用 RMSNorm\_mid |
| norm_eps | 输入 | RMSNorm epsilon | float | 标量 | 默认 1e-6, 必须 > 0 |
| hc_eps | 输入 | sigmoid gate epsilon | float | 标量 | 默认 1e-6, 必须 > 0 |
| return_h_in_f32| 输入 | cast float | bool |  标量 | 默认 false，仅v2接口支持 |

#### 返回值

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|
| h\_in\_prime | 输出 | 经 post-norm + gate + pre-norm 后的hidden layer输出 | 与 h\_out 一致 | [B,S,D] 或 [T,D] |
| x\_2\_out | 输出 | gate 前的中间结果 | 与 h\_out 一致 | [B,S,N,D] 或 [T,N,D] |
| h\_in\_f32 | 输出(可选) | 数据与h\_in\_prime一致，数据类型为float，仅v2接口支持| float |[B,S,D] 或 [T,D]|

## 约束说明

### 确定性计算

在相同硬件、相同输入条件下，算子结果确定性一致。

### 公共约束

- 输入约束：
   - h\_out 和 residual 的 dtype 必须一致（BF16 或 FP16）
   - 所有格式相关输入必须使用同一格式（BSND 或 TND），`residual.ndim == h_out.ndim + 1`
   - 所有输入 Tensor 必须连续 (contiguous)
   - norm\_eps 和 hc\_eps 必须 > 0
- 入图约束：
   - 支持 torch.compile 图模式（通过 torchair converter 注册）
   - 图模式下 gamma\_2=None 对应 CANN Optional Input 空传入
- 输入值域范围：
  算子入参输入值域范围为inf/-inf/nan或者输入值域或中间计算结果出现超过float16/bfloat16数据类型范围时，无法确定输出结果有效性。
  
### 规格约束

| 规格项 | 规格 | 规格说明 |
|:---|:---|:---|
| B\*S 或 T | [1, 256] | token 总数范围 |
| N | 4 | head 数量，固定值 |
| D | {2560, 5120} | embedding 维度，仅支持两种取值 |
| norm\_eps | > 0 | eps=0 导致 rsqrt(0)=inf, eps<0 导致 NaN |
| hc\_eps | > 0 | 同 norm\_eps |
| dtype (h\_out, residual) | BF16, FP16 | 两者必须一致 |
| dtype (其余输入) | FP32 | 固定 |

### 典型值

| 规格项 | 典型值 |
|:---|:---|
| B | 1, 2, 4, 20, 24 |
| S | 1, 2, 4, 8, 16|
| D | 2560, 5120 |
| norm\_eps | 1e-6 |
| hc\_eps | 1e-6 |
| dtype | BF16 |

## 调用示例

### 单算子模式调用 (eager)

```python
import torch
import torch_npu
import omni_custom_ops

B, S, N, D = 1, 4, 4, 2560
dtype = torch.bfloat16

# 构造输入
h_out = torch.randn(B, S, D, dtype=dtype).npu()
residual = torch.randn(B, S, N, D, dtype=dtype).npu()
h_post = torch.randn(B, S, N).npu()
h_res = torch.randn(B, S, N, N).npu()
phi = (torch.randn(N, N * D) * 0.1).npu()
alpha_0 = torch.randn(N).npu()
bias = torch.randn(N).npu()
gamma_0 = (torch.randn(D) * 0.3 + 0.5).npu()
gamma_1 = (torch.randn(D) * 0.15 + 0.9).npu()
gamma_2 = (torch.randn(N * D) * 0.06 + 0.08).npu()  # 或 None（跳过 RMSNorm_mid）
return_h_in_f32 = True

# 调用
h_in_prime, x_2_out = torch.ops.custom.npu_mhc_sandwich_norm_post_preonly(
    h_out, residual, h_post, h_res,
    phi, alpha_0, bias, gamma_0, gamma_1, gamma_2,
    norm_eps=1e-6, hc_eps=1e-6,
)

print(f"h_in_prime: {h_in_prime.shape}, dtype={h_in_prime.dtype}")
# h_in_prime: torch.Size([1, 4, 2560]), dtype=torch.bfloat16
print(f"x_2_out: {x_2_out.shape}, dtype={x_2_out.dtype}")
# x_2_out: torch.Size([1, 4, 4, 2560]), dtype=torch.bfloat16

h_in_prime, x_2_out, h_in_f32 = torch.ops.custom.npu_mhc_sandwich_norm_post_preonly_v2(
    h_out, residual, h_post, h_res,
    phi, alpha_0, bias, gamma_0, gamma_1, gamma_2=gamma_2,
    norm_eps=1e-6, hc_eps=1e-6, return_h_in_f32=return_h_in_f32,
)

print(f"h_in_prime: {h_in_prime.shape}, dtype={h_in_prime.dtype}")
# h_in_prime: torch.Size([1, 4, 2560]), dtype=torch.bfloat16
print(f"x_2_out: {x_2_out.shape}, dtype={x_2_out.dtype}")
# x_2_out: torch.Size([1, 4, 4, 2560]), dtype=torch.bfloat16
print(f"h_in_f32: {h_in_f32.shape}, dtype={h_in_f32.dtype}")
# x_2_out: torch.Size([1, 4, 2560]), dtype=torch.float
```

### 图模式调用 (torch.compile)

```python
import torch
import torch_npu
import omni_custom_ops

B, S, N, D = 1, 4, 4, 2560
dtype = torch.bfloat16

# 构造输入
h_out = torch.randn(B, S, D, dtype=dtype).npu()
residual = torch.randn(B, S, N, D, dtype=dtype).npu()
h_post = torch.randn(B, S, N).npu()
h_res = torch.randn(B, S, N, N).npu()
phi = (torch.randn(N, N * D) * 0.1).npu()
alpha_0 = torch.randn(N).npu()
bias = torch.randn(N).npu()
gamma_0 = (torch.randn(D) * 0.3 + 0.5).npu()
gamma_1 = (torch.randn(D) * 0.15 + 0.9).npu()
gamma_2 = (torch.randn(N * D) * 0.06 + 0.08).npu()  # 或 None（跳过 RMSNorm_mid）
return_h_in_f32 = True

class MhcModule(torch.nn.Module):
    def forward(self, h_out, residual, h_post, h_res,
                phi, alpha_0, bias, gamma_0, gamma_1, gamma_2,
                norm_eps, hc_eps):
        return torch.ops.custom.npu_mhc_sandwich_norm_post_preonly(
            h_out, residual, h_post, h_res,
            phi, alpha_0, bias, gamma_0, gamma_1, gamma_2,
            norm_eps=norm_eps, hc_eps=hc_eps,
        )

model = MhcModule()
compiled = torch.compile(model, backend="npugraph_ex", dynamic=False,
                         options={"force_eager": True})

# 调用方式与 eager 相同
h_in_prime, x_2_out = compiled(
    h_out, residual, h_post, h_res,
    phi, alpha_0, bias, gamma_0, gamma_1, gamma_2,
    1e-6, 1e-6,
)

# v2调用方式
class MhcModuleV2(torch.nn.Module):
    def forward(self, h_out, residual, h_post, h_res,
                phi, alpha_0, bias, gamma_0, gamma_1, gamma_2,
                norm_eps, hc_eps, return_h_in_f32):
        return torch.ops.custom.npu_mhc_sandwich_norm_post_preonly_v2(
            h_out, residual, h_post, h_res,
            phi, alpha_0, bias, gamma_0, gamma_1, gamma_2=gamma_2,
            norm_eps=norm_eps, hc_eps=hc_eps, return_h_in_f32=return_h_in_f32
        )

modelV2 = MhcModuleV2()
compiled = torch.compile(modelV2, backend="npugraph_ex", dynamic=False,
                         options={"force_eager": True})

# 调用方式与 eager 相同
h_in_prime, x_2_out, h_in_f32 = compiled(
    h_out, residual, h_post, h_res,
    phi, alpha_0, bias, gamma_0, gamma_1, gamma_2,
    1e-6, 1e-6, return_h_in_f32
)
```
