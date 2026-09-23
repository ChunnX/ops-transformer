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
    const aclTensor *hOut,
    const aclTensor *residual,
    const aclTensor *hPost,
    const aclTensor *hResOptional,
    const aclTensor *phi,
    const aclTensor *alpha,
    const aclTensor *bias,
    const aclTensor *gamma1,
    const aclTensor *preMixOptional,
    const aclTensor *gamma0Optional,
    const aclTensor *gamma2Optional,
    float           normEps,
    float           mhcPreNormEps,
    float           hcEps,
    bool            returnHInF32,
    aclTensor      *hInPrimeOut,
    aclTensor      *x2Out,
    aclTensor      *hInF32OutOptional,
    aclTensor      *hPreOptional,
    uint64_t       *workspaceSize,
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnMhcNormPostPreonly(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMhcNormPostPreonlyGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 387px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 146px">
  </colgroup>
  <thead>
  <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>hOut（aclTensor*）</td>
      <td>输入</td>
      <td>Atten/MLP 层的输出。</td>
      <td><ul><li>不支持空指针。</li><li>shape为(T, D)或(B, S, D)。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
  </tr>
  <tr>
      <td>residual（aclTensor*）</td>
      <td>输入</td>
      <td>MHC 多路残差输入；dtype 必须与 hOut 一致。</td>
      <td><ul><li>不支持空指针。</li><li>shape为(T, N, D)或(B, S, N, D)，数据类型与hOut一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
  </tr>
  <tr>
      <td>hPost（aclTensor*）</td>
      <td>输入</td>
      <td>MHC_Post 主分支权重。</td>
      <td><ul><li>不支持空指针。</li><li>shape为(T, N)或(B, S, N)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
  </tr>
  <tr>
      <td>hResOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>mHC 的 hRes 变换矩阵，是做完 Sinkhorn 变换后的双随机矩阵；索引顺序为 [j,n]。接口预留参数，暂不支持传空指针。</td>
      <td><ul><li>shape为(T, N, N)或(B, S, N, N)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
  </tr>
  <tr>
      <td>phi（aclTensor*）</td>
      <td>输入</td>
      <td>MHC_Pre 投影矩阵。</td>
      <td><ul><li>不支持空指针。</li><li>shape为(N, N×D)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
  </tr>
  <tr>
      <td>alpha（aclTensor*）</td>
      <td>输入</td>
      <td>Sigmoid 门控缩放系数。</td>
      <td><ul><li>不支持空指针。</li><li>shape为(1)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
  </tr>
  <tr>
      <td>bias（aclTensor*）</td>
      <td>输入</td>
      <td>Sigmoid 门控偏置。</td>
      <td><ul><li>不支持空指针。</li><li>shape为(N)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
  </tr>
  <tr>
      <td>gamma1（aclTensor*）</td>
      <td>输入</td>
      <td>RMSNorm_1 缩放权重。</td>
      <td><ul><li>不支持空指针。</li><li>shape为(D)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
  </tr>
  <tr>
      <td>preMixOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>传入时，h_in 的加权求和直接使用 preMixOptional 作为权重（典型用法：填入上一轮 hPreOptional 的输出）；不传入时，h_in 使用本轮内部计算的 hPreOptional 进行加权求和。接口预留参数，当前仅支持传入空指针。</td>
      <td><ul><li>当前仅支持传空指针。</li><li>shape为(T, N)或(B, S, N)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
  </tr>
  <tr>
      <td>gamma0Optional（aclTensor*）</td>
      <td>可选输入</td>
      <td>RMSNorm_0 缩放权重。接口预留参数，当前暂不支持传空指针。</td>
      <td><ul><li>该参数为接口预留，暂不支持传空指针。</li><li>shape为(D)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
  </tr>
  <tr>
      <td>gamma2Optional（aclTensor*）</td>
      <td>可选输入</td>
      <td>RMSNorm_mid 缩放权重。默认值为空指针；非空时启用 RMSNorm_mid，为空时跳过。</td>
      <td><ul><li>可传空指针，表示跳过RMSNorm_mid。</li><li>非空时shape为(N×D)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
  </tr>
  <tr>
      <td>normEps（float）</td>
      <td>可选输入</td>
      <td>RMSNorm_0、RMSNorm_mid 和 RMSNorm_1 的数值稳定项，默认值为 1e-6。</td>
      <td>建议值：1e-6。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>mhcPreNormEps（float）</td>
      <td>可选输入</td>
      <td>MHC_Pre 内部 RMS 的数值稳定项，默认值为 1e-6。</td>
      <td>建议值：1e-6。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hcEps（float）</td>
      <td>可选输入</td>
      <td>Sigmoid 门控结果的加性偏移，默认值为 1e-6。</td>
      <td>建议值：1e-6。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>returnHInF32（bool）</td>
      <td>可选输入</td>
      <td>是否额外返回 FP32 的 hInF32OutOptional，默认值为 false。</td>
      <td><ul><li>取值为true时，hInF32OutOptional必须为非空指针。</li><li>取值为false时，hInF32OutOptional必须为空指针。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hInPrimeOut（aclTensor*）</td>
      <td>输出</td>
      <td>RMSNorm_1 FP32 结果转换为输入 dtype 后的下一层输入。</td>
      <td><ul><li>不支持空指针。</li><li>shape为(T, D)或(B, S, D)，数据类型与hOut一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
  </tr>
  <tr>
      <td>x2Out（aclTensor*）</td>
      <td>输出</td>
      <td>MHC_Post 以及可选 RMSNorm_mid 后的中间结果。</td>
      <td><ul><li>不支持空指针。</li><li>shape为(T, N, D)或(B, S, N, D)，数据类型与hOut一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
  </tr>
  <tr>
      <td>hInF32OutOptional（aclTensor*）</td>
      <td>可选输出</td>
      <td>RMSNorm_1 的 FP32 结果，仅在 returnHInF32=true 时返回。</td>
      <td><ul><li>returnHInF32为false时必须传空指针。</li><li>returnHInF32为true时不支持空Tensor，shape为(T, D)或(B, S, D)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
  </tr>
  <tr>
      <td>hPreOptional（aclTensor*）</td>
      <td>可选输出</td>
      <td>本轮内部计算的 pre 结果（sigmoid 后的加权系数），可作为下一轮 preMixOptional 的输入。接口预留参数，当前暂不支持输出。</td>
      <td><ul><li>当前仅支持传空指针。</li><li>shape为(T, N)或(B, S, N)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
  </tr>
  <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
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

### 确定性计算

- 默认确定性实现，不支持 Batch 一致性。

### 公共约束

1. 输入约束：
   - 必选输入均不支持空Tensor，所有输入、输出Tensor必须位于同一NPU；
   - 所有输入、输出Tensor的数据格式仅支持`ACL_FORMAT_ND`；
   - `hOut`、`residual`、`hInPrimeOut`和`x2Out`的数据类型必须一致，仅支持BFLOAT16或FLOAT16；
   - 两种前导维形式不可混用：`hOut`为`[B,S,D]`时，其余带前导维Tensor必须使用`[B,S,...]`；`hOut`为`[T,D]`时，其余带前导维Tensor必须使用`[T,...]`；
   - 当前仅支持`N=4`；
   - `gamma0Optional`为可选输入，当前仅支持传入非空指针；
   - `preMixOptional`为可选输入，当前仅支持传入空指针；
   - `hResOptional`为可选输入，当前仅支持传入非空指针；
   - `gamma2Optional`为可选输入，传入空指针表示关闭RMSNorm_mid，传入非空指针表示开启RMSNorm_mid；
   - `returnHInF32`为true时，`hInF32OutOptional`必须为非空指针；为false时，`hInF32OutOptional`必须为空指针。
2. 内存约束：
   - Workspace内存需在Device侧申请，且大小不得小于第一段接口返回值；
   - 非连续输入Tensor无需提前转为连续，算子内部自动处理；调用者预分配的输出Tensor必须为连续Tensor。

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 正数 | T、B、S必须为正数；实际可用规模还受Device内存和Workspace资源约束。 |
| N | 4 | MHC head数，固定值。 |
| 维度数 | hOut为2/3 | `hOut`支持二维(T,D)或三维(B,S,D)，其他输入、输出的rank与其联动。 |
| normEps、mhcPreNormEps、hcEps | 建议值1e-6 | 三者均按FLOAT精度参与计算，用于对应阶段的防除零处理。 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_norm_post_preonly.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)
#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

struct TensorResource {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
};

int CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype, bool initialize, TensorResource &resource)
{
    const int64_t elementCount = GetShapeSize(shape);
    const size_t elementSize = dtype == ACL_FLOAT ? sizeof(float) : sizeof(aclFloat16);
    const size_t byteSize = elementCount * elementSize;
    aclError ret = aclrtMalloc(&resource.deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return -1);

    if (initialize) {
        if (dtype == ACL_FLOAT) {
            std::vector<float> hostData(elementCount, 1.0f);
            ret = aclrtMemcpy(resource.deviceAddr, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        } else {
            std::vector<aclFloat16> hostData(elementCount, aclFloat16(1.0f));
            ret = aclrtMemcpy(resource.deviceAddr, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        CHECK_RET(ret == ACL_SUCCESS, return -1);
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), resource.deviceAddr);
    CHECK_RET(resource.tensor != nullptr, return -1);
    return 0;
}

void DestroyTensor(TensorResource &resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
    }
    if (resource.deviceAddr != nullptr) {
        aclrtFree(resource.deviceAddr);
    }
}

int main()
{
    constexpr int32_t deviceId = 0;
    constexpr int64_t T = 1;
    constexpr int64_t N = 4;
    constexpr int64_t D = 2560;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtCreateContext(&context, deviceId) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return -1);

    TensorResource hOut, residual, hPost, hResOptional, phi, alpha, bias, gamma1, gamma0;
    TensorResource hInPrimeOut, x2Out;
    CHECK_RET(CreateTensor({T, D}, ACL_FLOAT16, true, hOut) == 0, return -1);
    CHECK_RET(CreateTensor({T, N, D}, ACL_FLOAT16, true, residual) == 0, return -1);
    CHECK_RET(CreateTensor({T, N}, ACL_FLOAT, true, hPost) == 0, return -1);
    CHECK_RET(CreateTensor({T, N, N}, ACL_FLOAT, true, hResOptional) == 0, return -1);
    CHECK_RET(CreateTensor({N, N * D}, ACL_FLOAT, true, phi) == 0, return -1);
    CHECK_RET(CreateTensor({1}, ACL_FLOAT, true, alpha) == 0, return -1);
    CHECK_RET(CreateTensor({N}, ACL_FLOAT, true, bias) == 0, return -1);
    CHECK_RET(CreateTensor({D}, ACL_FLOAT, true, gamma1) == 0, return -1);
    CHECK_RET(CreateTensor({D}, ACL_FLOAT, true, gamma0) == 0, return -1);
    CHECK_RET(CreateTensor({T, D}, ACL_FLOAT16, false, hInPrimeOut) == 0, return -1);
    CHECK_RET(CreateTensor({T, N, D}, ACL_FLOAT16, false, x2Out) == 0, return -1);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnMhcNormPostPreonlyGetWorkspaceSize(
        hOut.tensor, residual.tensor, hPost.tensor, hResOptional.tensor, phi.tensor, alpha.tensor, bias.tensor,
        gamma1.tensor, nullptr, gamma0.tensor, nullptr, 1e-6, 1e-6, 1e-6, false, hInPrimeOut.tensor, x2Out.tensor,
        nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("GetWorkspaceSize failed, error: %d\n", ret); return -1);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    }
    ret = aclnnMhcNormPostPreonly(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("Execute failed, error: %d\n", ret); return -1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return -1);
    LOG_PRINT("MhcNormPostPreonly compute success!\n");

    for (TensorResource *resource :
         {&hOut, &residual, &hPost, &hResOptional, &phi, &alpha, &bias, &gamma1, &gamma0, &hInPrimeOut, &x2Out}) {
        DestroyTensor(*resource);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
