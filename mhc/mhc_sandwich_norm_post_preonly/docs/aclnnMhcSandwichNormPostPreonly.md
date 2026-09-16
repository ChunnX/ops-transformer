# aclnnMhcSandwichNormPostPreonly

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_sandwich_norm_post_preonly)

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

- 接口功能：`aclnnMhcSandwichNormPostPreonly` 将MHC Sandwich结构中相邻的归一化、Post混合和Pre门控计算融合为一个算子，输出下一层输入、MHC_Post中间结果以及可选的FLOAT计算结果。

- 计算流程：

1. **RMSNorm_0**：沿隐藏维 $D$ 对 `hOut` 做均方根归一化，并乘以 `gamma0Optional`。
2. **MHC_Post**：使用 `hPost` 和 `hRes`，将归一化后的 `hOut` 与 $N$ 路 `residual` 进行加权混合，得到 `x2Out`。
3. **RMSNorm_mid（可选）**：当 `gamma2Optional` 非空时，将每个 token 的 `x2Out` 展平为 $N\times D$，在完整的 $N\times D$ 维上做均方根归一化；为空时跳过该阶段。
4. **MHC_Pre（PreOnly）**：将 `x2Out` 投影到 $N$ 路门控权重，经 Sigmoid 后对各路结果加权求和。`PreOnly` 表示这里只计算提供给下一层的 pre 分支，不生成其他 MHC 分支输出。
5. **RMSNorm_1**：沿隐藏维 $D$ 对加权求和结果做均方根归一化，得到下一层输入 `hInPrimeOut`。

算子固定输出 `hInPrimeOut` 和中间结果 `x2Out`。当 `returnHInF32` 为 `true` 时，还通过 `hInF32OutOptional` 返回 RMSNorm_1 的 FLOAT 计算结果；为 `false` 时不返回该可选输出。

该融合减少了等价小算子组合产生的 Kernel 启动开销和中间 Tensor 读写。

- 计算公式：

以下公式按保留B、S前导维的`[B,S,...]`形式表示；`[T,...]`形式将`B*S`展平为`T`，后续维度保持不变。

**阶段1：RMSNorm_0**

$$
\begin{aligned}
h\_out\_fp32_{b,s,d} &= \operatorname{FP32}(h\_out_{b,s,d}), \\
inv\_rms\_0_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}h\_out\_fp32_{b,s,i}^{2}+norm\_eps\right), \\
x\_1_{b,s,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(h\_out\_fp32_{b,s,d}\cdot inv\_rms\_0_{b,s}\cdot gamma\_0_d\right).
\end{aligned}
$$

**阶段2：MHC_Post**

$$
\begin{aligned}
term1_{b,s,n,d} &= \operatorname{FP32}(h\_post_{b,s,n})\cdot \operatorname{FP32}(x\_1_{b,s,d}), \\
term2_{b,s,n,d} &= \sum_{j=0}^{N-1}\operatorname{FP32}(h\_res_{b,s,j,n})\cdot \operatorname{FP32}(residual_{b,s,j,d}), \\
x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(term1_{b,s,n,d}+term2_{b,s,n,d}\right).
\end{aligned}
$$

其中`j`为归约head，`n`为输出head。

**阶段2.5：RMSNorm_mid（可选）**

当 `gamma2Optional` 非空时：

$$
\begin{aligned}
x\_2\_flat &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
inv\_rms\_mid_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_flat_{b,s,k}^{2}+norm\_eps\right), \\
x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(\operatorname{reshape}\left(x\_2\_flat\cdot inv\_rms\_mid\cdot gamma\_2,[B,S,N,D]\right)_{b,s,n,d}\right).
\end{aligned}
$$

当 `gamma2Optional` 为空时，跳过本阶段，`x_2` 保持MHC_Post的输出；随后固定保存：

$$
x\_2\_out=x\_2.
$$

**阶段3：MHC_Pre（PreOnly）**

$$
\begin{aligned}
x\_2\_rs &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
inv\_rms\_pre_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_rs_{b,s,k}^{2}+mhc\_pre\_norm\_eps\right), \\
proj\_w &= phi^T, \\
x\_pre &= (x\_2\_rs\cdot proj\_w)\cdot inv\_rms\_pre, \\
h\_pre_{b,s,n} &= \operatorname{sigmoid}(alpha\_0\cdot x\_pre_{b,s,n}+bias_n)+hc\_eps, \\
h\_in_{b,s,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(\sum_{n=0}^{N-1}h\_pre_{b,s,n}\cdot \operatorname{FP32}(x\_2_{b,s,n,d})\right).
\end{aligned}
$$

**阶段4：RMSNorm_1**

$$
\begin{aligned}
h\_in\_rms_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}\operatorname{FP32}(h\_in_{b,s,i})^{2}+norm\_eps\right), \\
h\_in\_f32_{b,s,d} &= \operatorname{FP32}(h\_in_{b,s,d})\cdot h\_in\_rms_{b,s}\cdot \operatorname{FP32}(gamma\_1_d), \\
h\_in\_prime &= \operatorname{Cast}_{dtype(h\_out)}(h\_in\_f32).
\end{aligned}
$$

`norm_eps`、`mhc_pre_norm_eps` 和 `hc_eps` 在接口中分别对应 `normEps`、`mhcPreNormEps` 和 `hcEps`。`norm_eps` 用于RMSNorm_0、RMSNorm_mid和RMSNorm_1，`mhc_pre_norm_eps` 仅用于MHC_Pre归一化，`hc_eps` 仅用于pre门控。三者均按FLOAT精度参与计算。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMhcSandwichNormPostPreonly"接口执行计算。

```Cpp
aclnnStatus aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize(
    const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost,
    const aclTensor *hRes, const aclTensor *phi, const aclTensor *alpha0,
    const aclTensor *bias, const aclTensor *gamma1,
    const aclTensor *gamma0Optional, const aclTensor *gamma2Optional,
    double normEps, double mhcPreNormEps, double hcEps,
    bool returnHInF32, const aclTensor *hInPrimeOut,
    const aclTensor *x2Out, const aclTensor *hInF32OutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnMhcSandwichNormPostPreonly(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

## aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px"><col style="width: 121px"><col style="width: 287px"><col style="width: 387px">
  <col style="width: 187px"><col style="width: 187px"><col style="width: 187px"><col style="width: 146px">
  </colgroup><thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th></tr></thead><tbody>
  <tr><td>hOut（aclTensor*）</td><td>输入</td><td>MHC_Post的主分支输入，对应公式中的hOut。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, D)或(B, S, D)。</li></ul></td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>2-3</td><td>√</td></tr>
  <tr><td>residual（aclTensor*）</td><td>输入</td><td>MHC_Post的多路残差输入。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, N, D)或(B, S, N, D)，数据类型与hOut一致。</li></ul></td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>3-4</td><td>√</td></tr>
  <tr><td>hPost（aclTensor*）</td><td>输入</td><td>MHC_Post的主分支权重。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, N)或(B, S, N)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>2-3</td><td>√</td></tr>
  <tr><td>hRes（aclTensor*）</td><td>输入</td><td>MHC_Post的残差混合权重。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, N, N)或(B, S, N, N)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>3-4</td><td>√</td></tr>
  <tr><td>phi（aclTensor*）</td><td>输入</td><td>MHC_Pre的投影参数。</td><td><ul><li>不支持空Tensor。</li><li>shape为(N, N×D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>2</td><td>√</td></tr>
  <tr><td>alpha0（aclTensor*）</td><td>输入</td><td>MHC_Pre的缩放参数。</td><td><ul><li>不支持空Tensor。</li><li>shape为(1)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>bias（aclTensor*）</td><td>输入</td><td>MHC_Pre的偏置参数。</td><td><ul><li>不支持空Tensor。</li><li>shape为(N)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>gamma1（aclTensor*）</td><td>输入</td><td>RMSNorm_1的缩放因子。</td><td><ul><li>不支持空Tensor。</li><li>shape为(D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>gamma0Optional（aclTensor*）</td><td>可选输入</td><td>RMSNorm_0的缩放因子。</td><td><ul><li>shape为(D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>gamma2Optional（aclTensor*）</td><td>可选输入</td><td>RMSNorm_mid的缩放因子。</td><td><ul><li>可传空指针，表示跳过RMSNorm_mid。</li><li>非空时shape为(N×D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>normEps（double）</td><td>可选输入</td><td>RMSNorm的防除零参数。</td><td>建议值：1e-6。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>mhcPreNormEps（double）</td><td>可选输入</td><td>MHC_Pre内部RMS的防除零参数。</td><td>建议值：1e-6。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>hcEps（double）</td><td>可选输入</td><td>Sigmoid门控结果的加性偏移。</td><td>建议值：1e-6。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>returnHInF32（bool）</td><td>可选输入</td><td>是否返回RMSNorm_1的FLOAT计算结果。</td><td><ul><li>取值为true时，hInF32OutOptional必须为非空指针。</li><li>取值为false时，hInF32OutOptional必须为空指针。</li></ul></td><td>BOOL</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>hInPrimeOut（aclTensor*）</td><td>输出</td><td>下一层输入，对应公式中的hInPrime。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, D)或(B, S, D)，数据类型与hOut一致。</li></ul></td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>2-3</td><td>×</td></tr>
  <tr><td>x2Out（aclTensor*）</td><td>输出</td><td>MHC_Post或可选RMSNorm_mid的输出。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, N, D)或(B, S, N, D)，数据类型与hOut一致。</li></ul></td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>3-4</td><td>×</td></tr>
  <tr><td>hInF32OutOptional（aclTensor*）</td><td>可选输出</td><td>RMSNorm_1的FLOAT计算结果。</td><td><ul><li>returnHInF32为false时必须传空指针。</li><li>returnHInF32为true时不支持空Tensor，shape为(T, D)或(B, S, D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>2-3</td><td>×</td></tr>
  <tr><td>workspaceSize（uint64_t*）</td><td>输出</td><td>返回需要在Device侧申请的workspace大小。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>executor（aclOpExecutor**）</td><td>输出</td><td>返回op执行器，包含了算子计算流程。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## aclnnMhcSandwichNormPostPreonly

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px"><col style="width: 134px"><col style="width: 844px">
  </colgroup><thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr></thead><tbody>
  <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址，workspaceSize为0时可传空指针。</td></tr>
  <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize获取。</td></tr>
  <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
  <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnMhcSandwichNormPostPreonly默认确定性实现。

- Batch一致性说明：
  - 默认非Batch一致性实现，不支持通过aclrtSetSysParamOpt开启Batch一致性。

- 规格约束：
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - N仅支持4，D仅支持2560或5120，T或B×S范围为[1, 1048576]。
    - hOut与residual的数据类型必须一致；其余输入Tensor的数据类型为FLOAT32。
    - [T, ...]与[B, S, ...]两种布局不可混用，仅支持ND格式。
    - gamma1必选；gamma0Optional和gamma2Optional为可选输入。
    - normEps、mhcPreNormEps和hcEps必须大于等于0。
  <!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize(
    hOut, residual, hPost, hRes, phi, alpha0, bias, gamma1, gamma0Optional,
    gamma2, 1e-6, 1e-6, 1e-6, true, hInPrime, x2Out, hInF32,
    &workspaceSize, &executor);
// 按 workspaceSize 申请设备内存后执行：
ret = aclnnMhcSandwichNormPostPreonly(workspace, workspaceSize, executor, stream);
```
