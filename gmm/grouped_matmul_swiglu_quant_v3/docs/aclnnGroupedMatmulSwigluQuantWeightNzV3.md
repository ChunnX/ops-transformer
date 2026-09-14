# aclnnGroupedMatmulSwigluQuantWeightNzV3

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

- 接口功能：融合 GroupedMatmul、MXFP8 反量化、SwiGLU 和 MXFP8 量化。该接口是内部 `GroupedMatmulSwigluQuantV3` 算子的 weightNZ 专用 ACLNN 封装，当前仅支持 Ascend 950（arch35）的单 Tensor MXFP8 weightNZ 场景。

- 与 V2 的主要区别：V3 增加了可配置的 SwiGLU 参数，包括 `activateDim`、`activateLeft`、`swigluMode`、`clampLimit`、`gluAlpha` 和 `gluBias`。这些参数只作用于 SwiGLU 计算阶段，不改变 GroupedMatmul 和 MXFP8 量化的数据格式约束。

- 计算公式：

  1. 根据 `groupList` 和 `groupListType` 将 `x` 划分为不同的分组，并为每个分组选择对应的权重切片。
  2. 执行 grouped matmul，并结合 `xScale` 和 `weightScale` 完成 MXFP8 反量化，得到待进行 SwiGLU 计算的矩阵 `x_i`：

     $$x_i = (X_i \cdot W_i) \odot xScale_i \odot weightScale_i$$

     其中缩放因子按照 MX block 规则在 K 方向对应的 64 元素块上广播。

  3. 按 `activateDim=-1` 和 `activateLeft` 对 `x_i` 拆分，执行 `swigluMode` 对应的 SwiGLU 计算，得到 `swigluOut_i`。
  4. 对 `swigluOut_i` 执行 MXFP8 动态量化，输出量化结果 `output` 和 E8M0 格式的 `outputScale`：

     $$output, outputScale = DynamicMxQuant(swigluOut_i)$$

- `swigluMode` 取值和计算公式：

  - `swigluMode=0`：标准 SwiGLU。沿最后一维将 `x_i` 前后等分；当 `activateLeft=true` 时，`A_i` 为前半部分、`B_i` 为后半部分，当 `activateLeft=false` 时两者互换。

    $$swigluOut_i = Swiglu(x_i) = Swish(A_i) * B_i$$

  - `swigluMode=1`：按奇偶位置交错拆分。`x_glu` 表示 `x_i` 的偶数索引部分，`x_linear` 表示 `x_i` 的奇数索引部分；`activateLeft` 在此模式下不生效。

    $$x\_glu = x\_glu.clamp(min=None, max=clampLimit)$$

    $$x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)$$

    $$out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)$$

    $$swigluOut_i = out\_glu * (x\_linear + gluBias)$$

  - `swigluMode=2`：沿 `activateDim` 轴前后分半拆分，且不是交错分块。`activateLeft=true` 时 `x_glu` 为前半部分，否则为后半部分。

    $$x\_glu = x\_glu.clamp(min=None, max=clampLimit)$$

    $$x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)$$

    $$out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)$$

    $$swigluOut_i = out\_glu * (x\_linear + gluBias)$$

  - `swigluMode=3`：沿 `activateDim` 轴前后分半拆分，且不是交错分块。`activateLeft=true` 时激活前半部分，否则激活后半部分；固定使用 `alpha=1`，不添加 bias。

    $$x\_glu = x\_glu * sigmoid(x\_glu)$$

    $$x\_glu = x\_glu.clamp(min=None, max=clampLimit)$$

    $$x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)$$

    $$swigluOut_i = x\_glu * x\_linear$$

    此模式忽略 `gluAlpha` 和 `gluBias`。

  本接口当前固定 `activateDim=-1`、`axis=-1`，具体约束见“约束说明”。

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
    int64_t activateDim,
    bool activateLeft,
    int64_t swigluMode,
    double clampLimit,
    double gluAlpha,
    double gluBias,
    int64_t axis,
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

### 参数说明

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
      <tr><td>weight</td><td>输入</td><td>分组权重矩阵。</td><td>TensorList 长度必须为 1；非转置逻辑 view 为 (E, K, N)，转置逻辑 view 为 (E, N, K)，storage 均为五维 FRACTAL_NZ。</td><td>FLOAT8_E4M3FN</td><td>FRACTAL_NZ</td><td>view 3 维，storage 5 维</td><td>√</td></tr>
      <tr><td>weightScale</td><td>输入</td><td>权重量化因子。</td><td>TensorList 长度必须为 1，与 weight 的唯一 Tensor 对应；非转置 shape 为 (E, ceil(K / 64), N, 2)，转置 shape 为 (E, N, ceil(K / 64), 2)。</td><td>FLOAT8_E8M0</td><td>ND</td><td>4 维</td><td>√</td></tr>
      <tr><td>weightAssistMatrix</td><td>可选输入</td><td>权重辅助矩阵。</td><td>当前不支持，必须传 nullptr 或空 TensorList。</td><td>-</td><td>-</td><td>-</td><td>√</td></tr>
      <tr><td>bias</td><td>可选输入</td><td>矩阵乘偏置。</td><td>当前不支持，必须传 nullptr。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>xScale</td><td>输入</td><td>激活量化因子。</td><td>按 K 方向每 64 个元素对应两个 E8M0 scale。</td><td>FLOAT8_E8M0</td><td>ND</td><td>3 维，(M, ceil(K / 64), 2)</td><td>√</td></tr>
      <tr><td>smoothScale</td><td>可选输入</td><td>平滑量化因子。</td><td>当前不支持，必须传 nullptr。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>groupList</td><td>输入</td><td>分组信息。</td><td>长度必须为 E；其语义由 groupListType 决定。</td><td>INT64</td><td>ND</td><td>1 维，(E)</td><td>√</td></tr>
      <tr><td>dequantMode</td><td>输入</td><td>反量化模式。</td><td>当前仅支持 2，表示 MX 量化反量化。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>dequantDtype</td><td>输入</td><td>GroupedMatmul 中间结果类型。</td><td>当前仅支持 0，即 DT_FLOAT。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>quantMode</td><td>输入</td><td>输出量化模式。</td><td>当前仅支持 2，表示 MX 量化。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>groupListType</td><td>输入</td><td>分组列表解释方式。</td><td>0 表示 cumsum，1 表示 count。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>tuningConfigOptional</td><td>可选输入</td><td>tiling 调优配置。</td><td>当前不支持，必须传 nullptr 或空数组。</td><td>aclIntArray</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>activateDim</td><td>输入</td><td>SwiGLU 拆分维度。</td><td>当前仅支持 -1，表示最后一维。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>activateLeft</td><td>输入</td><td>激活分支选择。</td><td>在 swigluMode 为 0、2、3 时，true 表示拆分后的左分支作为激活分支，false 表示右分支作为激活分支；mode 1 按奇偶位置拆分，此参数不生效。</td><td>BOOL</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>swigluMode</td><td>输入</td><td>SwiGLU 计算模式。</td><td>支持 0、1、2、3。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>clampLimit</td><td>输入</td><td>用于限定 SwiGLU 中间结果的 clamp 范围：x_glu 的上界为 clampLimit，x_linear 的范围为 [-clampLimit, clampLimit]。</td><td>必须为有限且可用 float32 表示的值；swigluMode 为 1、2、3 时还必须大于 0，mode 0 不使用该参数。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>gluAlpha</td><td>输入</td><td>SwiGLU alpha 参数。</td><td>用于 swigluMode 为 1、2 的 Sigmoid 输入缩放；必须为有限且可用 float32 表示的值，mode 0、3 不使用该参数。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>gluBias</td><td>输入</td><td>SwiGLU bias 参数。</td><td>用于 swigluMode 为 1、2 的线性分支偏置；必须为有限且可用 float32 表示的值，mode 0、3 不使用该参数。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>axis</td><td>输入</td><td>MX 量化轴。</td><td>当前仅支持 -1，沿最后一维量化。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>roundMode</td><td>输入</td><td>MX 量化舍入模式。</td><td>当前仅支持 "rint"；传 nullptr 或空字符串时按默认值处理。</td><td>const char *</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>scaleAlg</td><td>输入</td><td>MX 量化 scale 算法。</td><td>当前接口支持 0（OCP）和 1（cuBLAS）；其他值（包括 2）会报错。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>dstTypeMax</td><td>输入</td><td>MX 量化目标类型最大值。</td><td>当前 MXFP8 输出仅支持 0.0。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>output</td><td>输出</td><td>MXFP8 量化结果。</td><td>调用方创建并传入，当前仅支持 FLOAT8_E4M3FN。</td><td>FLOAT8_E4M3FN</td><td>ND</td><td>2 维，(M, N / 2)</td><td>√</td></tr>
      <tr><td>outputScale</td><td>输出</td><td>MXFP8 输出 scale。</td><td>调用方创建并传入。</td><td>FLOAT8_E8M0</td><td>ND</td><td>3 维，(M, ceil((N / 2) / 64), 2)</td><td>√</td></tr>
      <tr><td>workspaceSize</td><td>输出</td><td>Device 侧 workspace 大小。</td><td>返回值。</td><td>uint64_t *</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>executor</td><td>输出</td><td>算子执行器。</td><td>返回值，传给第二段接口。</td><td>aclOpExecutor **</td><td>-</td><td>-</td><td>-</td></tr>
    </tbody>
    </table>

### 返回值

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
    <td>x、weight、weightScale、xScale、groupList、output、outputScale、workspaceSize 或 executor 为空指针，或者 TensorList 中包含空 Tensor。</td>
  </tr>
  <tr>
    <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="7">161002</td>
    <td>输入或输出的数据类型不在支持范围内；当前 MXFP8 weightNZ 场景要求 x、weight 和 output 为 FLOAT8_E4M3FN，xScale、weightScale 和 outputScale 为 FLOAT8_E8M0，groupList 为 INT64。</td>
  </tr>
  <tr>
    <td>输入或输出的参数维度、shape 或 format 不满足约束，例如 weight 的逻辑 view 不是三维、storage 不是五维 FRACTAL_NZ，或输出 shape 不匹配。</td>
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
    <td>activateDim 不为 -1，swigluMode 不在 [0, 3]，或 clampLimit、gluAlpha、gluBias 不满足有限性及 float32 可表示性约束；其中 swigluMode 为 1、2、3 时 clampLimit 还必须大于 0。</td>
  </tr>
  <tr>
    <td>axis、roundMode、scaleAlg、dstTypeMax 不满足 MXFP8 场景约束。</td>
  </tr>
</tbody>
</table>

## aclnnGroupedMatmulSwigluQuantWeightNzV3

该接口使用第一段接口返回的 executor，在指定 stream 上执行算子计算。

### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `workspace` | 输入 | Device 侧申请的 workspace 内存地址；大小由第一段接口返回的 `workspaceSize` 决定。 |
| `workspaceSize` | 输入 | Device 侧 workspace 大小，必须使用第一段接口返回的值。 |
| `executor` | 输入 | 第一段接口返回的 op executor。 |
| `stream` | 输入 | 指定算子执行的 ACL stream。 |

### 返回值

返回 `aclnnStatus` 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  - 确定性计算：
      - aclnnGroupedMatmulSwigluQuantWeightNzV3默认为确定性实现。

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 仅支持 MXFP8 weightNZ 场景，不支持 per-token、MXFP4、A8W4、A4W4 或其他量化场景。
    - 仅支持单 Tensor：`weight` 和 `weightScale` 的 TensorList 长度必须均为 1；不支持将每个 expert 拆成多个 Tensor 的多 Tensor调用方式。
    - `weight` 的逻辑 view 必须为三维，storage 必须为五维 FRACTAL_NZ；支持非转置和转置 weight，x 仅支持非转置输入。
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
              <td><ul>
              <li>非转置：storage shape 形如{(E, ceil(K / 16), ceil(N / 32), 16, 32)}，逻辑 view shape 形如{(E, K, N)}</li>
              <li>转置：storage shape 形如{(E, ceil(K / 32), ceil(N / 16), 16, 32)}，逻辑 view shape 形如{(E, N, K)}</li></ul></td>
              <td><ul>
              <li>非转置：(E, ceil(K / 64), N, 2)</li>
              <li>转置：(E, N, ceil(K / 64), 2)</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
          </tbody>
          </table>

        - 当前接口要求 `weight` 和 `weightScale` 的 TensorList 长度均为 1，并要求 `groupList` 长度等于 weight TensorList 长度；因此当前可用场景为 `E=1`。weight Tensor 的逻辑首维、`groupList` 长度和唯一 weight Tensor 的分组数必须一致。
        - `M` 是 token 数量，`K` 是输入特征维度，`N` 是 SwiGLU 拆分前的输出维度；当前 `M`、`K` 必须大于 0。
        - `N` 必须大于 0 且为 64 的整数倍；SwiGLU 输出维度为 `N / 2`。
        - `weightScale` 的最后一维固定为 2，表示每个 MX block 的 scale 数据。
        - x shape为(M, K)，非转置 weight 入口 view shape 为(E, K, N)，storage shape 为(E, ceil(K / 16), ceil(N / 32), 16, 32)；转置 weight 的 view shape 为(E, N, K)，storage shape 为(E, ceil(K / 32), ceil(N / 16), 16, 32)。
        - weightScale shape为(E, ceil(K / 64), N, 2)，xScale shape为(M, ceil(K / 64), 2)。
        - output shape为(M, N / 2)，outputScale shape为(M, ceil((N / 2) / 64), 2)。
        - 输入 Tensor 可以是非连续 Tensor；接口内部会对 x、xScale、groupList 和 weightScale 执行必要的 contiguous 处理，并为 weight 创建 FRACTAL_NZ view，但 Tensor 的 dtype、shape 和 format 元数据必须满足上述约束。
        - `dequantMode=2`、`quantMode=2`，且二者必须匹配。
        - `dequantDtype=0`，表示中间计算使用 `DT_FLOAT`。
        - `groupListType=0` 时，`groupList[i]` 表示累计 token 数，最后一个值不能大于 `M`；`groupListType=1` 时，`groupList[i]` 表示当前 group 的 token 数，所有值之和不能大于 `M`。
        - `weightAssistMatrix`、`bias`、`smoothScale` 和 `tuningConfigOptional` 当前必须为空或 `nullptr`；其中 `weightAssistMatrix` 允许传空 TensorList，`tuningConfigOptional` 允许传空数组。
        - `activateDim` 当前仅支持 `-1`。
        - `swigluMode` 当前支持 `[0, 3]`。
        - `clampLimit` 必须为有限且可表示为 `float32` 的值；当 `swigluMode` 为 1、2 或 3 时还必须大于 0，mode 0 不使用该参数。`gluAlpha` 和 `gluBias` 必须为可表示为 `float32` 的有限值。
        - `axis` 仅支持 `-1`，`roundMode` 仅支持 `"rint"`。
        - `scaleAlg=0` 表示 OCP MX 量化，`scaleAlg=1` 表示 cuBLAS MX 量化；本接口不支持 `scaleAlg=2` 或其他值，传入会报错。
        - 对于本接口固定的 MXFP8 输出，`scaleAlg` 只能取 `0` 或 `1`；`dstTypeMax` 只能取 `0.0`。
        - `dstTypeMax` 必须为有限值，当前仅支持 `0.0`。
        - 执行路径由 `K` 和 `swigluMode` 决定：`swigluMode=1` 始终使用 CGMCT；`swigluMode=0`、`2`、`3` 且 `K` 为 128 对齐时使用 Tensor API；这些模式在 `K` 不满足 128 对齐时使用 CGMCT 兼容路径。非转置 weight 对应 NZ 路径，转置 weight 对应 ZN 路径。
  <!-- end id7 -->

## 调用示例

示例代码如下，仅展示 V3 ACLNN 两段式接口的调用方式。完整的 Tensor 创建、Device 内存管理和资源释放代码请参考[示例源码](../examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_v3_nz.cpp)。

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

// 示例场景：E=1，M=2048，K=128，N=128。
// x:            [M, K]，FLOAT8_E4M3FN，ND
// weight:       TensorList{[E, K, N]}，FLOAT8_E4M3FN，FRACTAL_NZ storage
// weightScale:  TensorList{[E, ceil(K/64), N, 2]}，FLOAT8_E8M0，ND
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

int64_t activateDim = -1;
bool activateLeft = true;
int64_t swigluMode = 0;
double clampLimit = 7.0;
double gluAlpha = 1.702;
double gluBias = 1.0;
int64_t axis = -1;
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
    activateDim, activateLeft, swigluMode,
    clampLimit, gluAlpha, gluBias,
    axis, roundMode, scaleAlg, dstTypeMax,
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

### MXFP8 weightNZ ACLNN 示例

仓库只保留一个典型的 ACLNN 两段式调用示例：[test_aclnn_grouped_matmul_swiglu_quant_v3_nz.cpp](../examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_v3_nz.cpp)。示例使用 `E=1`、`M=2048`、`K=128`、`N=128`，x 始终为非转置输入，并通过命令行参数覆盖四种 `swigluMode` 以及 weight 的非转置 NZ、转置 ZN 两种布局。

示例命令行参数依次为：`swigluMode transposeWeight activateLeft dumpPrefix [scaleAlg]`。

- `swigluMode`：取值 `0`、`1`、`2` 或 `3`。
- `transposeWeight`：`0` 表示非转置 NZ weight，`1` 表示转置 ZN weight。
- `activateLeft`：`0` 或 `1`；mode 1 中该参数不生效。
- `dumpPrefix`：可选的输出 dump 前缀；不需要 dump 时传空字符串或省略。
- `scaleAlg`：可选，默认为 `0`，仅支持 `0` 或 `1`。

在 CANN 9.2 环境中编译示例并设置自定义算子包后，可以分别运行以下调用，覆盖四种模式和两种 weight 布局：

```bash
export CANN_ROOT=/home/developer/Ascend/cann-9.2.0/cann-9.2.0
export ASCEND_CUSTOM_OPP_PATH=/path/to/packages/vendors/custom_transformer
export V3_DEMO_BIN=/path/to/test_aclnn_grouped_matmul_swiglu_quant_v3_nz

${V3_DEMO_BIN} 0 0 1 /tmp/gmm_swiglu_v3_mode0_nz 0
${V3_DEMO_BIN} 0 1 1 /tmp/gmm_swiglu_v3_mode0_zn 0
${V3_DEMO_BIN} 1 0 1 /tmp/gmm_swiglu_v3_mode1_nz 0
${V3_DEMO_BIN} 1 1 1 /tmp/gmm_swiglu_v3_mode1_zn 0
${V3_DEMO_BIN} 2 0 1 /tmp/gmm_swiglu_v3_mode2_nz 1
${V3_DEMO_BIN} 2 1 1 /tmp/gmm_swiglu_v3_mode2_zn 1
${V3_DEMO_BIN} 3 0 1 /tmp/gmm_swiglu_v3_mode3_nz 0
${V3_DEMO_BIN} 3 1 1 /tmp/gmm_swiglu_v3_mode3_zn 0
```

示例内部固定使用 `dequantMode=2`、`dequantDtype=0`、`quantMode=2`、`groupListType=0`、`activateDim=-1`、`clampLimit=7.0`、`gluAlpha=1.702`、`gluBias=1.0`、`axis=-1`、`roundMode="rint"` 和 `dstTypeMax=0.0`，并调用 `aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize` 与 `aclnnGroupedMatmulSwigluQuantWeightNzV3` 完成两段式执行。
