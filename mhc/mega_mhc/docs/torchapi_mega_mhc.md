# mega\_mhc

## 产品与状态

目标为 Ascend 950PR/950DT；不支持910。
提供 eager/Meta 接口，暂不提供 torchair GE Converter 或反向梯度。

## 功能与公式

设输入类型为 T，`RMS(x,g)=Cast<T>(FP32(x)*rsqrt(mean(FP32(x)^2)+norm_eps)*g)`。

1. `x1=RMS(h_out,gamma_0)`。
2. `z[n,d]=Cast<T>(h_post[n]*FP32(x1[d])+sum_j h_res[j,n]*FP32(residual[j,d]))`。
3. 未提供 gamma_2 时 `x2=z`；否则对完整 `[4D]` 行做 `x2=RMS(z,gamma_2)`。
4. `p=(FP32(x2.flatten()) @ phi.T)*rsqrt(mean(FP32(x2)^2)+mhc_pre_norm_eps)`。
5. 内部权重 `pre=sigmoid(alpha[0]*p[:4]+bias[:4])+hc_eps`（本版不写出到预留 h_pre）；
   `h_post_new=2*sigmoid(alpha[1]*p[4:8]+bias[4:8])`；
   `h_res_new=reshape(alpha[2]*p[8:24]+bias[8:24],[4,4])`。
6. `h_in=Cast<T>(sum_n pre[n]*FP32(x2[n,:]))`，`h_in_prime=RMS(h_in,gamma_1)`。

不执行 Sinkhorn。各阶段 Cast<T> 是接口语义的一部分。

## 函数原型

```python
mega_mhc(h_out, residual, h_post, h_res, phi, alpha, bias, gamma_1, pre_mix=None, gamma_0=None,
         *, gamma_2=None, norm_eps=1e-6, mhc_pre_norm_eps=1e-6, hc_eps=1e-6, compute_mode=0)
```

## 参数与输出

`...` 表示 `[T]` 或 `[B,S]`；所有输入必须在同一 NPU，支持非连续视图，通过 ACLNN 转连续。
N 固定为4，D 在[2560,7168]内且为256的倍数，前导维度必须为正，乘积不超过 UINT32_MAX。
扩展 D 当前为实验范围，仍存在未通过精度阈值的案例，不代表全范围已通过数值验收。

| 参数 | 必选 | shape / dtype / 语义 |
|---|---|---|
| h_out | 是 | `[...,D]`，FP16/BF16 |
| residual | 是 | `[...,4,D]`，同 h_out 类型 |
| h_post | 是 | `[...,4]`，FP32，Post 权重 |
| h_res | 是 | `[...,4,4]`，FP32，索引为 `[来源head,输出head]` |
| phi | 是 | `[24,4D]`，FP32，投影权重 |
| alpha / bias | 是 | `[3]` / `[24]`，FP32 |
| gamma_1 | 是 | `[D]`，FP32，最终 RMS scale |
| pre_mix | 预留可选 | FP32 ND `[T,4]` 或 `[B,S,4]`，不支持非连续；本版仅允许 None，非空时报错 |
| gamma_0 | 预留可选 | 默认 None，表示 RMSNorm_0 权重；本版必须传入非空 FP32 `[D]`，尚不支持跳过 RMSNorm_0 |
| gamma_2 | 否 | `[4D]`，FP32；None 为四算子，否则五算子，不接受空Tensor |
| norm_eps | 否 | 转为FP32后仍有限且大于0 |
| mhc_pre_norm_eps | 否 | Pre 投影归一化使用的独立 epsilon，转为FP32后仍有限且大于0 |
| hc_eps | 否 | 有限非负，转为FP32不溢出 |
| compute_mode | 否 | 0=FP32，1=HF32；不支持2/3BF16 |

返回 `(h_in_prime, h_post_new, h_res_new, x_2_out, h_pre)`；本版 h_pre 固定为 None。
前四个输出 shape 分别同 `(h_out,h_post,h_res,residual)`，dtype 为 `(T,FP32,FP32,T)`，均为连续 ND。
`h_pre` 在原型中为可选输出，ACLNN 调用必须传空指针，非空时报错。
预留接口约定其未来为本轮计算的 pre 权重，FP32 ND `[T,4]` 或 `[B,S,4]`；本版不分配输出数据、不计算、不写出。
CANN-0830 的 L0 启动层内部使用 `[0]` FP32 张量保留 h_pre 参数槽，避免空输出导致后续参数错位；
此占位不向用户返回，也不作为有效输出接受。
未来非空 pre_mix 将替代加权求和使用的权重；本版不启用这一路径。

Torch 使用 `aclnnMegaMhc`，属性顺序为 normEps、mhcPreNormEps、hcEps、computeMode。
调度模板属于 Host 内部决策，不在算子属性或 Torch/ACLNN 参数中公开。
接口已更正，旧调用程序须同步重新编译。

`gamma0Optional` 保留可选指针形式；本次950实现要求其非空。
同一950配置以确定性为目标；与910按误差阈值验收，不承诺逐位相同或跨batch组成一致。

## 示例

安装950算子包并加载其环境后，按仓库 `torch_extension` 构建流程安装 Torch 包。

```python
import torch
import torch_npu
from cann_ops_transformer import mega_mhc

device, dtype = "npu:1", torch.bfloat16
t, d = 17, 2560
x = torch.randn(t, d, device=device, dtype=dtype)
residual = torch.randn(t, 4, d, device=device, dtype=dtype)
h_post = torch.randn(t, 4, device=device)
h_res = torch.randn(t, 4, 4, device=device)
phi = torch.randn(24, 4*d, device=device) / (4*d)**0.5
alpha = torch.ones(3, device=device)
bias = torch.zeros(24, device=device)
gamma_0 = torch.ones(d, device=device)
gamma_1 = torch.ones(d, device=device)
outputs = mega_mhc(x, residual, h_post, h_res, phi, alpha, bias, gamma_1, None, gamma_0,
                   compute_mode=0)
# 五算子：额外传入 gamma_2=torch.ones(4*d, device=device)。
torch.npu.synchronize()
```
