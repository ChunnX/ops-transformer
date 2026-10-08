# aclnnMhcSandwichNormPostPreonly

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_sandwich_norm_post_preonly)

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | × |
<!-- npu="950" id1 -->
| Atlas A5 训练系列产品/Atlas 950 SuperPoD 推理系统 | √ |
<!-- npu="950" end -->

## 功能说明

MHC Sandwich Norm Post-Preonly 融合算子，将 MHC post-norm、可选的 RMSNorm_mid 和 MHC pre-norm 融合为单个 Kernel，减少 Host 与 Device 间的交互次数和中间 Tensor 的内存占用。

## 计算公式

**阶段1：RMSNorm_0**

$$
x_{1,(b,s,d)} =
\frac{h_{out,(b,s,d)}}
{\sqrt{\frac{1}{D}\sum_{i=0}^{D-1}(h_{out,(b,s,i)})^2+\epsilon_{norm}}}
\cdot\gamma_{0,(d)}
$$

**阶段2：MHC_Post**

$$
x_{2,(b,s,n,d)} =
h_{post,(b,s,n)}\cdot x_{1,(b,s,d)}
+\sum_{j=0}^{N-1}
\left(h_{res,(b,s,j,n)}\cdot residual_{(b,s,j,d)}\right)
$$

**阶段2.5：RMSNorm_mid（可选）**

$$
\begin{aligned}
x_{2,rs}&=\operatorname{reshape}(x_2,[B,S,N\times D]) \\
\gamma_2\ne\varnothing:\quad
x_{2,(b,s,n,d)}&=
\frac{x_{2,rs,(b,s,n\times d)}}
{\sqrt{\frac{1}{N\times D}\sum_{i=0}^{N\times D-1}(x_{2,rs,(b,s,i)})^2+\epsilon_{norm}}}
\cdot\gamma_{2,(n\times d)}
\end{aligned}
$$

**阶段3：MHC_Pre（PreOnly）**

$$
\begin{aligned}
\operatorname{invRms}_{pre}
&=\frac{1}
{\sqrt{\frac{1}{N\times D}\sum_{k=0}^{N\times D-1}(x_{2,rs,(b,s,k)})^2
+\epsilon_{mhc\_pre}}} \\
x_{pre}&=(x_{2,rs}\cdot\phi^T)\cdot\operatorname{invRms}_{pre} \\
h_{pre}&=\sigma(\alpha_0\cdot x_{pre}+bias)+\epsilon_{hc} \\
h_{in}&=\sum_{n=0}^{N-1}h_{pre,(b,s,n)}\cdot x_{2,(b,s,n,:)}
\end{aligned}
$$

**阶段4：RMSNorm_1**

$$
h_{in\_prime,(b,s,d)}=
\frac{h_{in,(b,s,d)}}
{\sqrt{\frac{1}{D}\sum_{i=0}^{D-1}(h_{in,(b,s,i)})^2+\epsilon_{norm}}}
\cdot\gamma_{1,(d)}
$$

其中，$\epsilon_{norm}$、$\epsilon_{mhc\_pre}$ 和 $\epsilon_{hc}$ 分别对应接口参数 `normEps`、`mhcPreNormEps` 和 `hcEps`。

## 函数原型

该算子提供[两段式 ACLNN 接口](../../../docs/zh/context/two_phase_api.md)。

```cpp
aclnnStatus aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize(
    const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost,
    const aclTensor *hRes, const aclTensor *phi, const aclTensor *alpha0,
    const aclTensor *bias, const aclTensor *gamma0, const aclTensor *gamma1,
    const aclTensor *gamma2Optional, double normEps, double mhcPreNormEps,
    double hcEps, bool returnHInF32, const aclTensor *hInPrime,
    const aclTensor *x2Out, const aclTensor *hInF32,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnMhcSandwichNormPostPreonly(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

## aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize

<table>
<tr><th>参数名</th><th>输入/输出</th><th>说明</th><th>数据类型</th><th>格式</th><th>Shape</th><th>非连续Tensor</th></tr>
<tr><td>hOut</td><td>输入</td><td>post 输入。</td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>[T,D] 或 [B,S,D]</td><td>支持</td></tr>
<tr><td>residual</td><td>输入</td><td>类型与 hOut 一致。</td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>[T,N,D] 或 [B,S,N,D]</td><td>支持</td></tr>
<tr><td>hPost</td><td>输入</td><td>post 权重。</td><td>FLOAT</td><td>ND</td><td>[T,N] 或 [B,S,N]</td><td>支持</td></tr>
<tr><td>hRes</td><td>输入</td><td>residual 混合权重。</td><td>FLOAT</td><td>ND</td><td>[T,N,N] 或 [B,S,N,N]</td><td>支持</td></tr>
<tr><td>phi</td><td>输入</td><td>pre 投影参数。</td><td>FLOAT</td><td>ND</td><td>[N,N×D]</td><td>支持</td></tr>
<tr><td>alpha0</td><td>输入</td><td>pre 缩放系数。</td><td>FLOAT</td><td>ND</td><td>[1]</td><td>支持</td></tr>
<tr><td>bias</td><td>输入</td><td>pre 偏置。</td><td>FLOAT</td><td>ND</td><td>[N]</td><td>支持</td></tr>
<tr><td>gamma0</td><td>输入</td><td>第一次 RMSNorm 权重。</td><td>FLOAT</td><td>ND</td><td>[D]</td><td>支持</td></tr>
<tr><td>gamma1</td><td>输入</td><td>第二次 RMSNorm 权重。</td><td>FLOAT</td><td>ND</td><td>[D]</td><td>支持</td></tr>
<tr><td>gamma2Optional</td><td>输入</td><td>可选；nullptr 表示跳过 pre RMSNorm。</td><td>FLOAT</td><td>ND</td><td>[D]</td><td>支持</td></tr>
<tr><td>normEps</td><td>输入</td><td>第一次 RMSNorm epsilon，默认 1e-6。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>mhcPreNormEps</td><td>输入</td><td>第二次 RMSNorm epsilon，默认 1e-6。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>hcEps</td><td>输入</td><td>pre RMSNorm epsilon，默认 1e-6。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>returnHInF32</td><td>输入</td><td>是否返回 FLOAT 中间结果。</td><td>BOOL</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>hInPrime</td><td>输出</td><td>第一次 RMSNorm 输出。</td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>[T,D] 或 [B,S,D]</td><td>-</td></tr>
<tr><td>x2Out</td><td>输出</td><td>pre 分支输出。</td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>[T,N,D] 或 [B,S,N,D]</td><td>-</td></tr>
<tr><td>hInF32</td><td>输出</td><td>returnHInF32=true 时返回 FLOAT 类型 h_in；否则传 nullptr。</td><td>FLOAT</td><td>ND</td><td>[T,D] 或 [B,S,D]</td><td>-</td></tr>
<tr><td>workspaceSize</td><td>输出</td><td>workspace 字节数。</td><td>UINT64</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>executor</td><td>输出</td><td>算子执行器。</td><td>aclOpExecutor **</td><td>-</td><td>-</td><td>-</td></tr>
</table>

## aclnnMhcSandwichNormPostPreonly

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| workspace | 输入 | 设备侧 workspace 地址，大小为 0 时可传 nullptr。 |
| workspaceSize | 输入 | 第一阶段返回值。 |
| executor | 输入 | 第一阶段返回的执行器。 |
| stream | 输入 | aclrtStream 流。 |

## 返回值

返回 `aclnnStatus`，`ACLNN_SUCCESS` 表示成功。

## 约束说明

- 仅支持 Atlas A5 训练系列产品/Atlas 950 SuperPoD 推理系统。
- `N=4`，`D` 为 2560 或 5120，`T` 或 `B×S` 范围为 `[1,1048576]`。
- `hOut` 与 `residual` 类型一致；其余 Tensor 为 FLOAT。
- 两种布局不可混用；`gamma0`、`gamma1` 必选，`gamma2Optional` 可为 nullptr。
- 三个 epsilon 参数均须大于等于 0；仅支持 ND。

## 调用示例

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize(
    hOut, residual, hPost, hRes, phi, alpha0, bias, gamma0, gamma1,
    gamma2, 1e-6, 1e-6, 1e-6, true, hInPrime, x2Out, hInF32,
    &workspaceSize, &executor);
// 按 workspaceSize 申请设备内存后执行：
ret = aclnnMhcSandwichNormPostPreonly(workspace, workspaceSize, executor, stream);
```
