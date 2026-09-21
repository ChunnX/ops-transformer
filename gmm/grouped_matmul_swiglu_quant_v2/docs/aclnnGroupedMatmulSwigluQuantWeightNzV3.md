# aclnnGroupedMatmulSwigluQuantWeightNzV3

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_swiglu_quant_v2)

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

- 接口功能：融合GroupedMatmul、MXFP8反量化、SwiGLU和MXFP8量化。该接口为weight NZ特化版本，调用者必须传入FRACTAL_NZ格式的`weight`。当前仅支持Ascend 950PR/Ascend 950DT的单Tensor MXFP8场景。

- 与V2接口的主要区别：增加SwiGLU计算参数`swigluMode`、`clampLimit`、`gluAlpha`和`gluBias`，以及MX量化参数`roundMode`、`scaleAlg`和`dstTypeMax`。当前`swigluMode`仅支持2。

- 计算公式：

  1. 根据 `groupList` 和 `groupListType` 将 `x` 划分为不同的分组，并为每个分组选择对应的权重切片。
  2. 执行 grouped matmul，并结合 `xScale` 和 `weightScale` 完成 MXFP8 反量化，得到待进行 SwiGLU 计算的矩阵 `x_i`：

     $$x_i = (X_i \cdot W_i) \odot xScale_i \odot weightScale_i$$

     其中缩放因子按MX block规则在K方向每32个元素一组广播，每64个元素存储两个E8M0 scale。

  3. 沿最后一维前后分半，对前半部分执行带clamp和alpha的SiLU，对后半部分执行带clamp和bias的线性分支，再逐元素相乘，得到`swigluOut_i`。
  4. 对 `swigluOut_i` 执行 MXFP8 动态量化，输出量化结果 `output` 和 E8M0 格式的 `outputScale`：

     $$output, outputScale = DynamicMxQuant(swigluOut_i)$$

  - `swigluMode`取值和计算公式：

  - `swigluMode=2`：沿最后一维前后分半拆分，前半部分为 `x_glu`，后半部分为 `x_linear`。本接口固定采用前半部分作为激活分支。

    $$x\_glu = x\_glu.clamp(min=None, max=clampLimit)$$

    $$x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)$$

    $$out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)$$

    $$swigluOut_i = out\_glu * (x\_linear + gluBias)$$

  本接口当前固定沿最后一维拆分，具体约束见“约束说明”。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize` 获取计算所需的 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnGroupedMatmulSwigluQuantWeightNzV3` 执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
    const aclTensor *x,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *weightAssistMatrix,
    const aclTensor *bias,
    const aclTensor *xScale,
    const aclTensor *smoothScale,
    const aclTensor *groupList,
    int64_t dequantMode,
    int64_t dequantDtype,
    int64_t quantMode,
    int64_t groupListType,
    const aclIntArray *tuningConfigOptional,
    int64_t swigluMode,
    double clampLimit,
    double gluAlpha,
    double gluBias,
    const char *roundMode,
    int64_t scaleAlg,
    double dstTypeMax,
    aclTensor *output,
    aclTensor *outputScale,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize

该接口完成输入参数校验、输出 shape 校验、算子执行流程构造，并返回执行所需的 workspace 大小和 executor。

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
      <tr><td>x</td><td>输入</td><td>激活矩阵。</td><td>MXFP8 输入；当前仅支持 FLOAT8_E4M3FN。</td><td>FLOAT8_E4M3FN</td><td>ND</td><td>2 维，(M, K)</td><td>√</td></tr>
      <tr><td>weight</td><td>输入</td><td>分组权重矩阵。</td><td>TensorList长度必须为1；非转置和转置的API入口view shape均为(E, K, N)。仅支持下文规定的NZ/ZN stride和storage shape，不支持任意非连续view。</td><td>FLOAT8_E4M3FN</td><td>FRACTAL_NZ</td><td>3维</td><td>参见约束说明</td></tr>
      <tr><td>weightScale</td><td>输入</td><td>权重量化因子。</td><td>TensorList 长度必须为 1，与 weight 的唯一 Tensor 对应；非转置和转置的API入口view shape均为(E, ceil(K / 64), N, 2)，转置属性必须与weight一致。</td><td>FLOAT8_E8M0</td><td>ND</td><td>4 维</td><td>√</td></tr>
      <tr><td>weightAssistMatrix</td><td>可选输入</td><td>权重辅助矩阵。</td><td>当前不支持有效数据；可传nullptr、长度为0的TensorList，或长度为1且唯一元素为nullptr/shape (0)的TensorList。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>bias</td><td>可选输入</td><td>矩阵乘偏置。</td><td>当前不支持，必须传 nullptr。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>xScale</td><td>输入</td><td>激活量化因子。</td><td>按 K 方向每 64 个元素对应两个 E8M0 scale。</td><td>FLOAT8_E8M0</td><td>ND</td><td>3 维，(M, ceil(K / 64), 2)</td><td>√</td></tr>
      <tr><td>smoothScale</td><td>可选输入</td><td>平滑量化因子。</td><td>当前不支持，必须传 nullptr。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>groupList</td><td>输入</td><td>分组信息。</td><td>长度必须为 E；其语义由 groupListType 决定。</td><td>INT64</td><td>ND</td><td>1 维，(E)</td><td>√</td></tr>
      <tr><td>dequantMode</td><td>输入</td><td>反量化模式。</td><td>当前仅支持 2，表示 MX 量化反量化。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>dequantDtype</td><td>输入</td><td>GroupedMatmul 中间结果类型。</td><td>当前仅支持 0，即 DT_FLOAT。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>quantMode</td><td>输入</td><td>输出量化模式。</td><td>当前仅支持 2，表示 MX 量化。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>groupListType</td><td>输入</td><td>分组列表解释方式。</td><td>0 表示 cumsum，1 表示 count。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>tuningConfigOptional</td><td>可选输入</td><td>tiling 调优配置。</td><td>当前不支持，必须传 nullptr 或空数组。</td><td>aclIntArray</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>swigluMode</td><td>输入</td><td>SwiGLU 计算模式。</td><td>当前仅支持 2：沿最后一维前后分半，前半为激活分支，后半为线性分支。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>clampLimit</td><td>输入</td><td>表示SwiGLU中间结果的裁剪上限。</td><td>必选参数。激活分支裁剪上限为clampLimit，线性分支裁剪范围为[-clampLimit, clampLimit]；必须为有限、可由FLOAT表示且大于0的值。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>gluAlpha</td><td>输入</td><td>表示Sigmoid输入的缩放系数。</td><td>必选参数，必须为有限且可由FLOAT表示的值。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>gluBias</td><td>输入</td><td>表示线性分支的偏置。</td><td>必选参数，必须为有限且可由FLOAT表示的值。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>roundMode</td><td>输入</td><td>MX量化舍入模式。</td><td>可选参数，传入nullptr或空字符串时按`"rint"`处理；当前仅支持`"rint"`。</td><td>const char *</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>scaleAlg</td><td>输入</td><td>MX 量化 scale 算法。</td><td>当前接口支持 0（OCP）和 1（cuBLAS）；其他值（包括 2）会报错。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>dstTypeMax</td><td>输入</td><td>MX 量化目标类型最大值。</td><td>当前 MXFP8 输出仅支持 0.0。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>output</td><td>输出</td><td>MXFP8 量化结果。</td><td>调用方创建并传入，当前仅支持 FLOAT8_E4M3FN。</td><td>FLOAT8_E4M3FN</td><td>ND</td><td>2 维，(M, N / 2)</td><td>√</td></tr>
      <tr><td>outputScale</td><td>输出</td><td>MXFP8 输出 scale。</td><td>调用方创建并传入。</td><td>FLOAT8_E8M0</td><td>ND</td><td>3 维，(M, ceil((N / 2) / 64), 2)</td><td>√</td></tr>
      <tr><td>workspaceSize</td><td>输出</td><td>Device 侧 workspace 大小。</td><td>返回值。</td><td>uint64_t *</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>executor</td><td>输出</td><td>算子执行器。</td><td>返回值，传给第二段接口。</td><td>aclOpExecutor **</td><td>-</td><td>-</td><td>-</td></tr>
    </tbody>
    </table>

- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

第一段接口会完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
<col style="width: 220px">
<col style="width: 120px">
<col style="width: 810px">
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
    <td>x、weight、weightScale、xScale、groupList、output、outputScale、workspaceSize或executor为空指针，或者weight或weightScale的TensorList中包含空Tensor。</td>
  </tr>
  <tr>
    <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="7">161002</td>
    <td>输入或输出的数据类型不在支持范围内；当前 MXFP8 weightNZ 场景要求 x、weight 和 output 为 FLOAT8_E4M3FN，xScale、weightScale 和 outputScale 为 FLOAT8_E8M0，groupList 为 INT64。</td>
  </tr>
  <tr>
    <td>输入或输出的参数维度、shape或format不满足约束，例如weight不是三维FRACTAL_NZ格式，或输出shape不匹配。</td>
  </tr>
  <tr>
    <td>当前设备不是 Ascend 950，或者 weight、weightScale 的 TensorList 长度不为 1。</td>
  </tr>
  <tr>
    <td>dequantMode、quantMode、dequantDtype 不满足 2、2、0 的约束。</td>
  </tr>
  <tr>
    <td>weightAssistMatrix、bias、smoothScale 或 tuningConfigOptional 传入了当前场景不支持的内容。</td>
  </tr>
  <tr>
    <td>swigluMode 不等于 2，或 clampLimit、gluAlpha、gluBias 不满足有限性及 float32 可表示性约束；其中 clampLimit 还必须大于 0。</td>
  </tr>
  <tr>
    <td>roundMode、scaleAlg、dstTypeMax 不满足 MXFP8 场景约束。</td>
  </tr>
</tbody>
</table>

## aclnnGroupedMatmulSwigluQuantWeightNzV3

该接口使用第一段接口返回的 executor，在指定 stream 上执行算子计算。

- **参数说明**

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `workspace` | 输入 | Device 侧申请的 workspace 内存地址；`workspaceSize` 大于 0 时不能为空，等于 0 时可传 `nullptr`。 |
| `workspaceSize` | 输入 | Device 侧 workspace 大小，必须使用第一段接口返回的值。 |
| `executor` | 输入 | 第一段接口返回的 op executor，不能为空。 |
| `stream` | 输入 | 指定算子执行的 ACL stream。 |

- **返回值**

返回 `aclnnStatus` 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

当 `executor` 为空，或 `workspaceSize` 大于 0 且 `workspace` 为空时，返回 `ACLNN_ERR_PARAM_NULLPTR`（161001）。

## 约束说明

  - 确定性计算：
      - aclnnGroupedMatmulSwigluQuantWeightNzV3默认为确定性实现。

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 仅支持 MXFP8 weightNZ 场景，不支持 per-token、MXFP4、A8W4、A4W4 或其他量化场景。
    - 仅支持单 Tensor：`weight` 和 `weightScale` 的 TensorList 长度必须均为 1；不支持将每个 expert 拆成多个 Tensor 的多 Tensor调用方式。
    - `weight`必须为三维FRACTAL_NZ格式，支持非转置和转置weight；`x`仅支持非转置输入。
    - MX量化场景下需满足以下约束条件：
        - 数据类型需要满足下表：
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 300px">
          <col style="width: 300px">
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 130px">
          </colgroup>
          <thead>
            <tr>
              <th>MX量化场景</th>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>xScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>MXFP8</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E8M0</td>
            </tr>
          </tbody>
          </table>

        - shape约束需要满足下表：
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 250px">
          <col style="width: 320px">
          <col style="width: 180px">
          <col style="width: 250px">
          <col style="width: 160px">
          </colgroup>
          <thead>
            <tr>
              <th>MX量化场景</th>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>xScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>MXFP8</td>
              <td>(M, K)</td>
              <td>API入口view：(E, K, N)<br>
              非转置storage：(E, ceil(N / 32), ceil(K / 16), 16, 32)<br>
              转置storage：(E, ceil(K / 32), ceil(N / 16), 16, 32)</td>
              <td>API入口view：(E, ceil(K / 64), N, 2)<br>
              非转置source/storage与入口view相同<br>
              转置source/storage：(E, N, ceil(K / 64), 2)</td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
          </tbody>
          </table>

        - 当前接口要求 `weight` 和 `weightScale` 的 TensorList 长度均为 1；每个唯一 Tensor 内包含 E 个 expert，weight 的逻辑首维和 `groupList` 长度必须均为 E。因此“单 Tensor”表示单个 Tensor 承载全部 expert，并不限制 E 必须为 1。
        - `E`是expert数量，取值范围为[1, 1024]。`M`是token数量，必须大于0；`K`是输入特征维度，必须大于0；`N`是SwiGLU拆分前的输出维度。
        - `N` 必须大于 0 且为 64 的整数倍；SwiGLU 输出维度为 `N / 2`。
        - `weightScale` 的最后一维固定为 2，表示每个 MX block 的 scale 数据。
        - x shape为(M, K)。非转置和转置weight的API入口view shape均为(E, K, N)。非转置使用连续stride `(K * N, N, 1)`，storage shape为(E, ceil(N / 32), ceil(K / 16), 16, 32)。转置场景先构造(E, N, K)的source，再交换末两轴；API入口view stride为`(K * N, 1, K)`，storage shape为(E, ceil(K / 32), ceil(N / 16), 16, 32)。
        - 非转置和转置weightScale的API入口view shape均为(E, ceil(K / 64), N, 2)。非转置source/storage与入口view相同，stride为`(ceil(K / 64) * N * 2, N * 2, 2, 1)`；转置场景先构造(E, N, ceil(K / 64), 2)的source/storage，再交换中间两轴形成stride `(N * ceil(K / 64) * 2, 2, ceil(K / 64) * 2, 1)` 的入口view。
        - output shape为(M, N / 2)，outputScale shape为(M, ceil((N / 2) / 64), 2)。
        - x、xScale、weightScale、groupList、output和outputScale支持满足约束的非连续Tensor。weight仅支持上述规定的NZ/ZN view、stride和storage shape，不支持任意非连续view。
        - `dequantMode=2`、`quantMode=2`，且二者必须匹配。
        - `dequantDtype=0`，表示中间计算使用 `DT_FLOAT`。
        - `groupListType=0`时，`groupList[i]`表示累计token数，数值必须非负、单调不递减且不大于`M`；`groupListType=1`时，`groupList[i]`表示当前group的token数，数值必须非负且所有值之和不大于`M`。`groupList`未指定的输出区域不会被更新。
        - `weightAssistMatrix`、`bias`、`smoothScale` 和 `tuningConfigOptional` 当前必须为空或 `nullptr`；其中 `weightAssistMatrix` 允许传长度为0的 TensorList，或长度为1且唯一元素为 `nullptr`/shape (0) 的 TensorList；`tuningConfigOptional` 允许传空数组。
        - `swigluMode` 当前仅支持 `2`。
        - `clampLimit` 必须为有限、可表示为 `float32` 且大于 0 的值；`gluAlpha` 和 `gluBias` 必须为可表示为 `float32` 的有限值。
        - `roundMode` 仅支持 `"rint"`；传入 `nullptr` 或空字符串时等效于 `"rint"`。
        - `scaleAlg=0` 表示 OCP MX 量化，`scaleAlg=1` 表示 cuBLAS MX 量化；本接口不支持 `scaleAlg=2` 或其他值，传入会报错。
        - 对于本接口固定的MXFP8输出，`scaleAlg`只能取0或1；`dstTypeMax`必须为有限值，且只能取0.0。
  <!-- end id7 -->

## 调用示例

示例代码如下，仅展示 V3 ACLNN 两段式接口的调用方式。完整的 Tensor 创建、Device 内存管理和资源释放代码请参考[示例源码](../examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.cpp)。

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

// 示例场景：E=1，M=2048，K=64，N=128。
// x:            [M, K]，FLOAT8_E4M3FN，ND
// weight:       入口view为TensorList{[E, K, N]}，FLOAT8_E4M3FN，FRACTAL_NZ；
//               转置场景由[E, N, K] source交换末两轴得到该入口view。
// weightScale:  入口view为TensorList{[E, ceil(K/64), N, 2]}，FLOAT8_E8M0，ND；
//               转置场景由[E, N, ceil(K/64), 2] source交换中间两轴得到该入口view。
// xScale:       [M, ceil(K/64), 2]，FLOAT8_E8M0，ND
// output:       [M, N/2]，FLOAT8_E4M3FN，ND
// outputScale:  [M, ceil((N/2)/64), 2]，FLOAT8_E8M0，ND

aclTensorList *weight = nullptr;      // 使用 aclCreateTensorList 创建长度为 1 的 TensorList
aclTensorList *weightScale = nullptr; // 使用 aclCreateTensorList 创建长度为 1 的 TensorList
aclTensor *x = nullptr;               // 创建 FLOAT8_E4M3FN、ND、[M, K]
aclTensor *xScale = nullptr;          // 创建 FLOAT8_E8M0、ND、[M, ceil(K/64), 2]
aclTensor *groupList = nullptr;       // 创建 INT64、ND、[E]
aclTensor *output = nullptr;           // 创建 FLOAT8_E4M3FN、ND、[M, N/2]
aclTensor *outputScale = nullptr;      // 创建 FLOAT8_E8M0、ND、[M, ceil((N/2)/64), 2]
aclrtStream stream = nullptr;

// groupListType=0 表示 cumsum；E=1 时 groupList[0]=M。
int64_t dequantMode = 2;
int64_t dequantDtype = 0;  // DT_FLOAT
int64_t quantMode = 2;
int64_t groupListType = 0;

int64_t swigluMode = 2;
double clampLimit = 7.0;
double gluAlpha = 1.702;
double gluBias = 1.0;
const char *roundMode = "rint";
int64_t scaleAlg = 0;
double dstTypeMax = 0.0;

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
    x, weight, weightScale,
    nullptr,              // weightAssistMatrix
    nullptr,              // bias
    xScale,
    nullptr,              // smoothScale
    groupList,
    dequantMode, dequantDtype, quantMode, groupListType,
    nullptr,              // tuningConfigOptional
    swigluMode,
    clampLimit, gluAlpha, gluBias,
    roundMode, scaleAlg, dstTypeMax,
    output, outputScale, &workspaceSize, &executor);

// 根据 workspaceSize 在 Device 侧申请 workspace。
void *workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

ret = aclnnGroupedMatmulSwigluQuantWeightNzV3(workspace, workspaceSize, executor, stream);
// 检查 ret 后同步 stream，并按需读取 output/outputScale。
aclrtSynchronizeStream(stream);
```
