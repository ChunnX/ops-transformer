# aclnnGroupedMatmulActivationQuantWeightNz

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_activation_quant)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：融合GroupedMatmul、activation和quant。当前版本仅支持gelu_tanh激活函数与MX量化的组合，详细解释见计算公式。

- 计算公式：
  - <term>Ascend 950PR/Ascend 950DT</term>：
    <details>
    <summary>激活函数gelu_tanh + MX量化场景：</summary>
    <a id="激活函数gelu_tanh+MX量化场景"></a>

      - **定义**：

        * **⋅** 表示矩阵乘法。
        * **⊙** 表示逐元素乘法。
        * gelu_tanh激活函数的数学语义为：

          $$
          \text{GELU}_{\text{tanh}}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)
          $$

        * 当前kernel底层实现使用gelu_sigmoid函数对gelu_tanh进行近似计算，定义为：

          `GeluTanh(x) = x / (1 + exp(-1.595769121 * (x + 0.044715 * x^3)))`
        * $E$ 表示专家数，$M$ 表示总token数，$K$ 表示输入特征维度，$N$ 表示输出特征维度。
        * $blocksize$ 表示MX量化时共享指数的分组大小，当前仅支持64。

      - **输入**：

        * $X \in \mathbb{F8}^{M \times K}$：激活矩阵（左矩阵），数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2。
        * $weight \in \mathbb{F8}^{E \times K \times N}$：分组weight矩阵，数据类型仅支持FLOAT8_E4M3FN，格式仅支持FRACTAL_NZ，存储shape必须为5维。
        * $xScale \in \mathbb{E8M0}^{M \times ceil(K / 64) \times 2}$：左矩阵量化因子。
        * $weightScale \in \mathbb{E8M0}^{E \times ceil(K / 64) \times N \times 2}$：weight矩阵量化因子。
        * $bias$：MXFP8场景下必须为空，支持nullptr或空tensorlist。
        * $groupList \in \mathbb{N}^{E}$：cumsum或count的分组索引列表。

      - **转置组合**：

        * 当前x仅支持非转置输入。
        * weight支持非转置和转置输入，其中非转置输入时transposeWeight为false，转置输入时transposeWeight为true。
        * 当前不支持x转置输入。

      - **weight矩阵shape**：

        | transposeWeight | weight逻辑shape | weight存储shape |
        | :---: | :---: | :---: |
        | false | $[E,K,N]$ | $[E,ceil(N/32),ceil(K/16),16,32]$ |
        | true | $[E,N,K]$ | $[E,ceil(K/32),ceil(N/16),16,32]$ |

      - **输出**：

        * $Y \in \mathbb{F8}^{M \times N}$：激活并量化后的输出矩阵，数据类型由quantDtype指定，当前支持FLOAT8_E4M3FN、FLOAT8_E5M2。
        * $YScale \in \mathbb{E8M0}^{M \times ceil(N / 64) \times 2}$：输出量化因子，shape必须为3维。

      - **计算过程**

        - 1. 根据groupList[i]确定当前分组的token范围，$i \in [0, Len(groupList))$。

        - 2. 根据分组确定的入参进行GroupedMatmul和反量化计算：

          $C_i = (X_i \cdot weight_i) \odot xScale_{i\ Broadcast} \odot weightScale_{i\ Broadcast}$

        - 3. 执行gelu_tanh激活：

          $S_i = GeluTanh(C_i)$

          注：当前kernel底层实现使用gelu_sigmoid函数近似计算gelu_tanh，`GeluTanh(x) = x / (1 + exp(-1.595769121 * (x + 0.044715 * x^3)))`。

        - 4. 对激活结果进行MX量化，目标数据类型DType由quantDtype指定：

          - 场景1，当scaleAlg为0时，将激活结果$S_i$在N轴按$k=blocksize$分组，一组$k$个数$\{V_j\}_{j=1}^{k}$动态量化为$\{YScale, \{P_j\}_{j=1}^{k}\}$。

            $$
            shared\_exp = floor(log_2(max_j(|V_j|))) - emax
            $$

            $$
            YScale = 2 ^ {shared\_exp}
            $$

            $$
            P_j = cast\_to\_dst\_type(V_j / YScale, roundMode), \space j \space from \space 1 \space to \space blocksize
            $$

            量化后的$P_j$按对应$V_j$的位置组成输出$Y$，$YScale$按对应N轴分组组成输出$YScale$。

            - $emax$：对应数据类型的最大正则数的指数位。

              | DataType | emax |
              | :---: | :---: |
              | FLOAT8_E4M3FN | 8 |
              | FLOAT8_E5M2 | 15 |

          - 场景2，当scaleAlg为1时，只涉及FP8类型。将激活结果$S_i$在N轴按$k=blocksize$分组，每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标FP8类型。如果最后一块不足$k$个元素，缺失值视为0并按完整块处理。

            找到该块中数值的最大绝对值：

            $$
            Amax(D_{fp32}^b)=max(\{|d_j|\}_{j=1}^{k})
            $$

            将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值：

            $$
            S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
            $$

            将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。先从$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$，再为避免量化溢出对指数向上取整：

            $$
            E_{int}^b = \begin{cases}
            E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\
            E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\
            E_{int}^b, & \text{否则}
            \end{cases}
            $$

            $$
            S_{ue8m0}^b = 2 ^ {E_{int}^b}
            $$

            $$
            R_{fp32}^b = \frac{1}{fp32(S_{ue8m0}^b)}
            $$

            对块内每个元素执行量化：

            $$
            d^j = DType(d_{fp32}^j \cdot R_{fp32}^b), \space j \space from \space 1 \space to \space blocksize
            $$

            最终输出的量化结果为$\left(S^b, [d^j]_{j=1}^{k}\right)$，其中$S^b$表示块缩放因子$S_{ue8m0}^b$，$[d^j]_{j=1}^{k}$表示块内量化后的数据。

    </details>

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedMatmulActivationQuantWeightNz”接口执行计算。

```Cpp
aclnnStatus aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
    const aclTensor     *x,
    const aclTensor     *groupList,
    const aclTensor     *xScale,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *bias,
    const char          *activationType,
    int64_t              dequantDtype,
    bool                 transposeWeight,
    int64_t              groupListType,
    const aclIntArray   *tuningConfig,
    int64_t              quantDtype,
    const char          *roundMode,
    int64_t              scaleAlg,
    double               dstTypeMax,
    aclTensor           *y,
    aclTensor           *yScale,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnGroupedMatmulActivationQuantWeightNz(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize

  - **参数说明**
    <table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th style="white-space: nowrap">输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th style="white-space: nowrap">维度(shape)</th>
        <th>非连续的Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>表示左矩阵，对应公式中的X。</td>
        <td>必选参数。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>groupList</td>
        <td>输入</td>
        <td>表示分组信息，对应公式中的groupList。</td>
        <td>必选参数。根据groupListType表示cumsum或count。</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>xScale</td>
        <td>输入</td>
        <td>表示左矩阵的量化因子，对应公式中的xScale。</td>
        <td>可选参数。当前gelu_tanh+MX量化场景下需要传入。</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>表示weight矩阵，对应公式中的weight。</td>
        <td>动态输入，当前仅支持长度为1的tensorlist，存储shape必须为5维。</td>
        <td>FLOAT8_E4M3FN</td>
        <td>FRACTAL_NZ</td>
        <td>5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightScale</td>
        <td>输入</td>
        <td>表示weight矩阵的量化因子，对应公式中的weightScale。</td>
        <td>动态输入，当前仅支持长度为1的tensorlist。</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>表示偏置。</td>
        <td>动态可选输入，MXFP8场景下必须为空，支持nullptr或空tensorlist。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activationType</td>
        <td>输入</td>
        <td>表示激活函数类型。</td>
        <td>必选属性。当前仅支持"gelu_tanh"。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dequantDtype</td>
        <td>输入</td>
        <td>表示中间GroupedMatmul结果的数据类型。</td>
        <td>可选属性，默认值为0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeWeight</td>
        <td>输入</td>
        <td>表示weight是否转置。</td>
        <td>可选属性，默认值为false。当前x仅支持非转置输入，weight支持非转置和转置输入。weight非转置输入时配置为false，weight转置输入时配置为true。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupListType</td>
        <td>输入</td>
        <td>表示groupList的解释方式。</td>
        <td>可选属性，默认值为0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>tuningConfig</td>
        <td>输入</td>
        <td>表示调优参数，预留参数。</td>
        <td>可选属性，默认值为[0]，当前暂不使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantDtype</td>
        <td>输入</td>
        <td>表示输出y的数据类型。</td>
        <td>可选属性，默认值为36，表示FLOAT8_E4M3FN。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>roundMode</td>
        <td>输入</td>
        <td>表示量化舍入模式。</td>
        <td>可选属性，默认值为"rint"，当前仅支持"rint"。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleAlg</td>
        <td>输入</td>
        <td>表示量化因子计算算法。</td>
        <td>可选属性，默认值为0。当前gelu_tanh+MX量化场景下支持0或1。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstTypeMax</td>
        <td>输入</td>
        <td>表示目标量化类型的最大值配置，预留参数。</td>
        <td>可选属性，默认值为0，当前暂不使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>表示激活并量化后的输出矩阵。</td>
        <td>必选参数。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yScale</td>
        <td>输出</td>
        <td>表示输出量化因子。</td>
        <td>必选参数，shape必须为3维。</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
    </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必选参数为空指针，例如x、groupList、weight、weightScale、y、yScale、workspaceSize或executor为空。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>参数不满足约束，例如activationType不是"gelu_tanh"，输入数据类型或数据格式不支持，shape维度不匹配，或属性值不在支持范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulActivationQuantWeightNz

  - **参数说明**

    <table style="undefined;table-layout: fixed;width: 795px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 505px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace地址。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize获取。</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含算子计算流程。</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>acl stream流。</td>
      </tr>
    </tbody>
    </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

  - 确定性计算：
      - aclnnGroupedMatmulActivationQuantWeightNz默认为确定性实现。
  - 当前仅支持<term>Ascend 950PR/Ascend 950DT</term>。
  - 当前仅支持gelu_tanh激活函数与MX量化的组合。
  - gelu_tanh+MX量化场景下需满足以下约束条件：
      - activationType仅支持"gelu_tanh"。
      - x、y仅支持FLOAT8_E4M3FN、FLOAT8_E5M2；weight仅支持FLOAT8_E4M3FN。
      - xScale、weightScale、yScale仅支持FLOAT8_E8M0。
      - weight仅支持FRACTAL_NZ格式，存储shape必须为5维；其他输入输出Tensor格式仅支持ND。
      - yScale的shape必须为3维，当前为[M, ceil(N / 64), 2]。
      - x仅支持非转置输入，weight支持非转置和转置输入，不支持x转置输入。
      - roundMode仅支持"rint"。
      - scaleAlg仅支持0或1。
      - weight、weightScale、bias均为dynamic tensorlist形式；当前weight和weightScale仅支持长度为1的tensorlist，MXFP8场景下bias必须为空，支持nullptr或空tensorlist。
      - tuningConfig和dstTypeMax为预留参数，当前暂不使用。

## 调用示例

```Cpp
#include "aclnnop/aclnn_grouped_matmul_activation_quant_weight_nz.h"

// 省略aclTensor/aclTensorList创建和Device内存申请。

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

auto ret = aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
    x,
    groupList,
    xScale,
    weight,
    weightScale,
    bias,
    "gelu_tanh",
    0,
    false,
    0,
    tuningConfig,
    36,
    "rint",
    0,
    0.0,
    y,
    yScale,
    &workspaceSize,
    &executor);

// 根据workspaceSize申请workspaceAddr。

ret = aclnnGroupedMatmulActivationQuantWeightNz(workspaceAddr, workspaceSize, executor, stream);
```
