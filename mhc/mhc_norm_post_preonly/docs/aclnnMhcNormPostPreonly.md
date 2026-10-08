# aclnnMhcNormPostPreonly

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_norm_post_preonly)

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

- 接口功能：从算法特征上看，MHC_Post 与 MHC_Pre（PreOnly）存在可跨层融合并统一计算的 Pattern。MhcNormPostPreonly 将以下计算链融合为一个算子，减少中间 Tensor 读写和算子下发开销：

  RMSNorm_0 → MHC_Post → [可选 RMSNorm_mid] → MHC_Pre（PreOnly）→ RMSNorm_1

- 融合收益与性能优化：通过融合 RMSNorm_0、MHC_Post、可选 RMSNorm_mid、仅 pre 分支的 MHC_Pre 和 RMSNorm_1，减少中间张量搬运和 Kernel 调度开销。算子固定输出 hInPrimeOut 和 x2Out；returnHInF32 为 true 时额外返回 RMSNorm_1 的 FLOAT 计算结果 hInF32OutOptional，为 false 时不返回该可选输出。

- 算子支持 [B,S,...] 和已将前导维展平的 [T,...] 两种输入形式，其中 T=B×S。定义：

  - K=N×D，表示 MHC_Pre 的输入投影维度；
  - B 为 batch 数，S 为序列长度，T 为 token 总数，N 为残差 head 数，D 为每个 head 的 hidden size；b/s/n/d 为对应下标；
  - FP32 表示转换为 FLOAT，Cast 表示转换为指定输出类型，rsqrt 表示平方根的倒数，sigmoid 表示 Sigmoid 函数。

- 计算公式：

  以下公式定义输入、中间结果和输出的关系，按保留 B、S 前导维的 [B,S,...] 形式表示；[T,...] 形式将 B×S 展平为 T，后续维度保持不变。

  **1 RMSNorm_0**

  $$
  \begin{aligned}
  h\_out\_fp32_{b,s,d} &= \operatorname{FP32}(h\_out_{b,s,d}), \\
  inv\_rms\_0_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}h\_out\_fp32_{b,s,i}^{2}+norm\_eps\right), \\
  x\_1_{b,s,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(h\_out\_fp32_{b,s,d}\cdot inv\_rms\_0_{b,s}\cdot gamma\_0_d\right).
  \end{aligned}
  $$

  **2 MHC_Post**

  $$
  \begin{aligned}
  term1_{b,s,n,d} &= \operatorname{FP32}(h\_post_{b,s,n})\cdot \operatorname{FP32}(x\_1_{b,s,d}), \\
  term2_{b,s,n,d} &= \sum_{j=0}^{N-1}\operatorname{FP32}(h\_res_{b,s,j,n})\cdot \operatorname{FP32}(residual_{b,s,j,d}), \\
  x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(term1_{b,s,n,d}+term2_{b,s,n,d}\right).
  \end{aligned}
  $$

  其中 j 为归约 head，n 为输出 head。

  **3 可选 RMSNorm_mid**

  当 gamma2Optional 非空时：

  $$
  \begin{aligned}
  x\_2\_flat &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
  inv\_rms\_mid_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_flat_{b,s,k}^{2}+norm\_eps\right), \\
  x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(h\_out)}\left(\operatorname{reshape}\left(x\_2\_flat\cdot inv\_rms\_mid\cdot gamma\_2,[B,S,N,D]\right)_{b,s,n,d}\right).
  \end{aligned}
  $$

  当 gamma2Optional 为空时，跳过该可选计算，x_2 保持 MHC_Post 的输出；随后固定保存：

  $$
  x\_2\_out=x\_2.
  $$

  **4 MHC_Pre（PreOnly）**

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

  **5 RMSNorm_1**

  $$
  \begin{aligned}
  h\_in\_rms_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}\operatorname{FP32}(h\_in_{b,s,i})^{2}+norm\_eps\right), \\
  h\_in\_f32_{b,s,d} &= \operatorname{FP32}(h\_in_{b,s,d})\cdot h\_in\_rms_{b,s}\cdot \operatorname{FP32}(gamma\_1_d), \\
  h\_in\_prime &= \operatorname{Cast}_{dtype(h\_out)}(h\_in\_f32).
  \end{aligned}
  $$

  norm_eps、mhc_pre_norm_eps 和 hc_eps 在接口中分别对应 normEps、mhcPreNormEps 和 hcEps。norm_eps 用于 RMSNorm_0、RMSNorm_mid 和 RMSNorm_1，mhc_pre_norm_eps 仅用于 MHC_Pre 归一化，hc_eps 仅用于 pre 门控。三者均按 FLOAT 精度参与计算。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnMhcNormPostPreonlyGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMhcNormPostPreonly"接口执行计算。

```Cpp
aclnnStatus aclnnMhcNormPostPreonlyGetWorkspaceSize(
    const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost,
    const aclTensor *hResOptional, const aclTensor *phi, const aclTensor *alpha0,
    const aclTensor *bias, const aclTensor *gamma1,
    const aclTensor *gamma0Optional, const aclTensor *gamma2Optional,
    float normEps, float mhcPreNormEps, float hcEps,
    bool returnHInF32, const aclTensor *hInPrimeOut,
    const aclTensor *x2Out, const aclTensor *hInF32OutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnMhcNormPostPreonly(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

## aclnnMhcNormPostPreonlyGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px"><col style="width: 121px"><col style="width: 287px"><col style="width: 387px">
  <col style="width: 187px"><col style="width: 187px"><col style="width: 187px"><col style="width: 146px">
  </colgroup><thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th></tr></thead><tbody>
  <tr><td>hOut（aclTensor*）</td><td>输入</td><td>MHC_Post的主分支输入，对应公式中的hOut。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, D)或(B, S, D)。</li></ul></td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>2-3</td><td>√</td></tr>
  <tr><td>residual（aclTensor*）</td><td>输入</td><td>MHC_Post的多路残差输入。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, N, D)或(B, S, N, D)，数据类型与hOut一致。</li></ul></td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>3-4</td><td>√</td></tr>
  <tr><td>hPost（aclTensor*）</td><td>输入</td><td>MHC_Post的主分支权重。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, N)或(B, S, N)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>2-3</td><td>√</td></tr>
  <tr><td>hResOptional（aclTensor*）</td><td>可选输入</td><td>mHC的hRes变换矩阵，是做完sinkhorn变换后的双随机矩阵。该参数为接口预留，暂不支持传空指针。</td><td><ul><li>shape为(T, N, N)或(B, S, N, N)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>3-4</td><td>√</td></tr>
  <tr><td>phi（aclTensor*）</td><td>输入</td><td>MHC_Pre的投影参数。</td><td><ul><li>不支持空Tensor。</li><li>shape为(N, N×D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>2</td><td>√</td></tr>
  <tr><td>alpha0（aclTensor*）</td><td>输入</td><td>MHC_Pre的缩放参数。</td><td><ul><li>不支持空Tensor。</li><li>shape为(1)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>bias（aclTensor*）</td><td>输入</td><td>MHC_Pre的偏置参数。</td><td><ul><li>不支持空Tensor。</li><li>shape为(N)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>gamma1（aclTensor*）</td><td>输入</td><td>RMSNorm_1的缩放因子。</td><td><ul><li>不支持空Tensor。</li><li>shape为(D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>gamma0Optional（aclTensor*）</td><td>可选输入</td><td>RMSNorm_0的缩放因子。</td><td><ul><li>暂不支持传空指针。</li><li>shape为(D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>gamma2Optional（aclTensor*）</td><td>可选输入</td><td>RMSNorm_mid的缩放因子。</td><td><ul><li>可传空指针，表示跳过RMSNorm_mid。</li><li>非空时shape为(N×D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>1</td><td>√</td></tr>
  <tr><td>normEps（float）</td><td>可选输入</td><td>RMSNorm的防除零参数。</td><td>建议值：1e-6。</td><td>FLOAT</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>mhcPreNormEps（float）</td><td>可选输入</td><td>MHC_Pre内部RMS的防除零参数。</td><td>建议值：1e-6。</td><td>FLOAT</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>hcEps（float）</td><td>可选输入</td><td>Sigmoid门控结果的加性偏移。</td><td>建议值：1e-6。</td><td>FLOAT</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>returnHInF32（bool）</td><td>可选输入</td><td>是否返回RMSNorm_1的FLOAT计算结果。</td><td><ul><li>取值为true时，hInF32OutOptional必须为非空指针。</li><li>取值为false时，hInF32OutOptional必须为空指针。</li></ul></td><td>BOOL</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>hInPrimeOut（aclTensor*）</td><td>输出</td><td>下一层输入，对应公式中的hInPrime。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, D)或(B, S, D)，数据类型与hOut一致。</li></ul></td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>2-3</td><td>×</td></tr>
  <tr><td>x2Out（aclTensor*）</td><td>输出</td><td>MHC_Post或可选RMSNorm_mid的输出。</td><td><ul><li>不支持空Tensor。</li><li>shape为(T, N, D)或(B, S, N, D)，数据类型与hOut一致。</li></ul></td><td>FLOAT16、BFLOAT16</td><td>ND</td><td>3-4</td><td>×</td></tr>
  <tr><td>hInF32OutOptional（aclTensor*）</td><td>可选输出</td><td>RMSNorm_1的FLOAT计算结果。</td><td><ul><li>returnHInF32为false时必须传空指针。</li><li>returnHInF32为true时不支持空Tensor，shape为(T, D)或(B, S, D)。</li></ul></td><td>FLOAT32</td><td>ND</td><td>2-3</td><td>×</td></tr>
  <tr><td>workspaceSize（uint64_t*）</td><td>输出</td><td>返回需要在Device侧申请的workspace大小。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>executor（aclOpExecutor**）</td><td>输出</td><td>返回op执行器，包含了算子计算流程。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## aclnnMhcNormPostPreonly

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px"><col style="width: 134px"><col style="width: 844px">
  </colgroup><thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr></thead><tbody>
  <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址，workspaceSize为0时可传空指针。</td></tr>
  <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMhcNormPostPreonlyGetWorkspaceSize获取。</td></tr>
  <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
  <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 仅支持 N=4。

## 调用示例

调用示例与 [test_aclnn_mhc_norm_post_preonly.cpp](../examples/test_aclnn_mhc_norm_post_preonly.cpp) 保持一致。以下代码中的输入、输出 TensorResource 和 stream 均由该示例完成创建。

```Cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnMhcNormPostPreonlyGetWorkspaceSize(
    hOut.tensor, residual.tensor, hPost.tensor, hResOptional.tensor, phi.tensor, alpha0.tensor, bias.tensor, gamma1.tensor,
    gamma0.tensor, nullptr, 1e-6, 1e-6, 1e-6, false, hInPrimeOut.tensor, x2Out.tensor, nullptr, &workspaceSize,
    &executor);
CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("GetWorkspaceSize failed, error: %d\n", ret); return -1);

void *workspace = nullptr;
if (workspaceSize > 0) {
    CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
}
ret = aclnnMhcNormPostPreonly(workspace, workspaceSize, executor, stream);
CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("Execute failed, error: %d\n", ret); return -1);
CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return -1);
```
