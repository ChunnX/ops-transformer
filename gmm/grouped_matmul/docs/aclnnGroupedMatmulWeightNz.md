# aclnnGroupedMatmulWeightNz

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul)

## 产品支持情况
|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |

## 功能说明

  - **接口功能**：实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如$y_i[m_i,n_i]=x_i[m_i,k_i] times weight_i[k_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应shape。输入输出数据类型均为aclTensorList，对应的功能为：

      - k轴分组：$k_i$各不相同，但$m_i/n_i$每组相同，此时$x_i/weight_i$可以在$k_i$上拼接。
      - m轴分组：$k_i$各组相同，$weight_i/y_i$可以在$n_i$上拼接。

    **与[GroupedMatmulV5](aclnnGroupedMatmulV5.md)接口对比新增功能**：

      - 输入的weight的[数据格式]支持AI处理器亲和数据排布格式（FRACTAL_NZ）。
      - 新增参数quantGroupSize，整数型参数，代表分组量化（per-group）的分组大小，不涉及分组量化时，填0。
      - <term>昇腾910_95 AI处理器</term>：暂不支持quantGroupSize参数。

  - **计算公式**：

      <a id="非量化场景"></a>

      - **非量化场景：**

        $$
        y_i=x_i \times weight_i + bias_i
        $$

      <a id="量化场景"></a>

      - **量化场景（无perTokenScaleOptional）：**

        - x为INT8，bias为INT32

          $$
          y_i=(x_i \times weight_i + bias_i) * scale_i + offset_i
          $$

        - x为INT8，bias为BFLOAT16/FLOAT16/FLOAT32，无offset

          $$
          y_i=(x_i \times weight_i) * scale_i + bias_i
          $$

      - **量化场景（有perTokenScaleOptional）：**

        - x为INT8，bias为INT32

          $$
          y_i=(x_i \times weight_i + bias_i) * scale_i * per\_token\_scale_i
          $$

        - x为INT8，bias为BFLOAT16/FLOAT16/FLOAT32

          $$
          y_i=(x_i \times weight_i) * scale_i * per\_token\_scale_i  + bias_i
          $$
       

      - **量化场景 (mx量化，当前无bias无激活层)：**

        $$
        y_i=(x_i \times per\_token\_scale_i) * (weight_i \times scale_i)
        $$

      <a id="反量化场景"></a>

      - **反量化场景：**

        $$
        y_i=(x_i \times weight_i + bias_i) * scale_i
        $$

      <a id="伪量化场景"></a>

      - **伪量化(perchannel、pergroup)场景：**

        $$
        y_i=x_i \times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
        $$

      - **伪量化(mx)场景：**

        x为BFLOAT16/FLOAT16输入，weight为FLOAT32(表示8个FLOAT4_E2M1)/FLOAT4_E2M1输入

        $$
        y_i=x_i \times (weight_i  * antiquant\_scale_i) + bias_i
        $$

        x为FLOAT8_E4M3FN输入，weight为FLOAT32(表示8个FLOAT4_E2M1)/FLOAT4_E2M1输入

        $$
        y_i=(x_i * per\_token\_scale_i) \times (weight_i  * antiquant\_scale_i) + bias_i
        $$

      - **伪量化(K-CG)场景：**

        $$
        y_i=(x_i \times (weight_i * antiquant\_scale_i)) * scale_i * per\_token\_scale_i + bias_i
        $$

        其中antiquant\_scale_i为weight矩阵pergroup量化参数，scale_i为weight矩阵perchannel量化参数，per\_token\_scale_i为
        pertoken量化参数。
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedMatmulWeightNzGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGroupedMatmulWeightNz”接口执行计算。

```c++
aclnnStatus aclnnGroupedMatmulWeightNzGetWorkspaceSize(
    const aclTensorList *x, 
    const aclTensorList *weight, 
    const aclTensorList *biasOptional, 
    const aclTensorList *scaleOptional, 
    const aclTensorList *offsetOptional, 
    const aclTensorList *antiquantScaleOptional, 
    const aclTensorList *antiquantOffsetOptional, 
    const aclTensorList *perTokenScaleOptional, 
    const aclTensor     *groupListOptional, 
    const aclTensorList *activationInputOptional, 
    const aclTensorList *activationQuantScaleOptional, 
    const aclTensorList *activationQuantOffsetOptional, 
    int64_t              splitItem, 
    int64_t              groupType, 
    int64_t              groupListType, 
    int64_t              actType, 
    aclIntArray         *tuningConfigOptional, 
    int64_t              quantGroupSize, 
    aclTensorList       *out, 
    aclTensorList       *activationFeatureOutOptional, 
    aclTensorList       *dynQuantScaleOutOptional, 
    uint64_t            *workspaceSize, 
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnGroupedMatmulWeightNz(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnGroupedMatmulWeightNzGetWorkspaceSize

  - **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px;">
    <colgroup>
    <col style="width: 190px">
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
    <th>输入/输出</th>
    <th>描述</th>
    <th>使用说明</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>公式中的输入<code>x</code>。</td>
    <td>最大支持128个。</td>
    <td>FLOAT16、BFLOAT16、INT8、INT4<sup>1</sup>、INT32<sup>1</sup>、FLOAT8_E4M3FN<sup>2</sup></td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weight</td>
    <td>输入</td>
    <td>公式中的<code>weight</code>。</td>
    <td>最大支持128个。支持昇腾亲和数据排布格式(nz)。</td>
    <td>FLOAT16、BFLOAT16、INT8、INT4、INT32、FLOAT32、FLOAT4_E2M1<sup>2</sup></td>
    <td>ND、FRACTAL_NZ</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>可选输入</td>
    <td>公式中的<code>bias</code>。</td>
    <td>长度与weight相同。</td>
    <td>FLOAT16、FLOAT32、INT32、BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleOptional</td>
    <td>可选输入</td>
    <td>公式中的<code>scale</code>，代表量化参数中的缩放因子。</td>
    <td>一般情况下，长度与weight相同。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>UINT64<sup>1</sup>、BFLOAT16<sup>1</sup>、FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>offsetOptional</td>
    <td>可选输入</td>
    <td>公式中的<code>offset</code>，代表量化参数中的偏移量。</td>
    <td>长度与weight相同。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>antiquantScaleOptional</td>
    <td>可选输入</td>
    <td>公式中的<code>antiquant_scale</code>，代表伪量化参数中的缩放因子。</td>
    <td>长度与weight相同。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT16、BFLOAT16<sup>1</sup>、FLOAT8_E8M0<sup>2</sup></td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>antiquantOffsetOptional</td>
    <td>可选输入</td>
    <td>公式中的<code>antiquant_offset</code>，代表伪量化参数中的偏移量。</td>
    <td>长度与weight相同。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT16、BFLOAT16<sup>1</sup></td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>perTokenScaleOptional</td>
    <td>可选输入</td>
    <td>公式中的<code>per_token_scale</code>，代表量化参数中的由x量化引入的缩放因子。</td>
    <td>仅支持x、weight、out均为单tensor场景。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT32、FLOAT8_E8M0<sup>2</sup></td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListOptional</td>
    <td>可选输入</td>
    <td>代表输入和输出分组轴方向的matmul大小分布。</td>
    <td>根据groupListType输入不同格式数据。注意：当输出TensorList长度为1时，最后一个值约束了输出的有效部分。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>activationInputOptional</td>
    <td>可选输入</td>
    <td>代表激活函数的反向输入。</td>
    <td>当前只支持传入nullptr。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>activationQuantScaleOptional</td>
    <td>可选输入</td>
    <td>-</td>
    <td>当前只支持传入nullptr。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>activationQuantOffsetOptional</td>
    <td>可选输入</td>
    <td>-</td>
    <td>当前只支持传入nullptr。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>splitItem</td>
    <td>输入</td>
    <td>代表输出是否要做tensor切分。</td>
    <td>0/1代表输出为多tensor；2/3代表输出为单tensor。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupType</td>
    <td>输入</td>
    <td>代表需要分组的轴。</td>
    <td>-1：不分组，0：m轴分组，1：n轴分组，2：k轴分组。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListType</td>
    <td>输入</td>
    <td>代表groupList输入的分组方式。</td>
    <td>0: cumsum结果; 1: 每组大小; 2: [groupIdx, groupSize]。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actType</td>
    <td>输入</td>
    <td>代表激活函数类型。</td>
    <td>取值范围为0-5。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tuningConfigOptional</td>
    <td>可选输入</td>
    <td>第一个数代表各个专家处理的token数的预期值，用于优化tiling。</td>
    <td>兼容历史版本，用户如不适用该参数，不传（即为nullptr）即可。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantGroupSize</td>
    <td>输入</td>
    <td>代表分组量化（per-group）的分组大小。</td>
    <td>不涉及分组量化时，填0。<term>昇腾910_95 AI处理器</term>暂不支持。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>out</td>
    <td>输出</td>
    <td>公式中的输出<code>y</code>。</td>
    <td>最大支持128个。</td>
    <td>FLOAT16、BFLOAT16、INT8、FLOAT32、INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>activationFeatureOutOptional</td>
    <td>输出</td>
    <td>激活函数的输入数据。</td>
    <td>当前只支持传入nullptr。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dynQuantScaleOutOptional</td>
    <td>输出</td>
    <td>-</td>
    <td>当前只支持传入nullptr。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
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
    <td>返回op执行器，包含了算子计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>
    
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
        - 上表数据类型列中的角标“1”代表该系列支持的数据类型，角标“2”代表该系列不支持的数据类型。
        - `weight`可使用`aclnnCalculateMatmulWeightSizeV2`及`aclnnTransMatmulWeight`完成ND到NZ转换。当传入INT32时，接口内部将每个INT32识别成8个INT4。
    - <term>Atlas 推理系列产品</term>：
        - 仅支持FLOAT16。`weight`仅支持FRACTAL_NZ格式，且需通过辅助接口转换。
        - `scaleOptional`、`offsetOptional`等量化/非对称量化参数功能暂不支持，需传空指针。
        - `groupType`只支持m轴分组(0)。`actType`只支持0。`tuningConfigOptional`不支持。
    - <term>昇腾910_95 AI处理器</term>：
        - 上表数据类型列中的角标“2”代表该系列支持的数据类型。
        - `x`支持FLOAT16、BFLOAT16、FLOAT8_E4M3FN、INT8。
        - `weight`支持FLOAT16、BFLOAT16、FLOAT4_E2M1、INT4。支持ND和FRACTAL_NZ格式。可使用aclnnNpuFormatCast接口完成输入Format从ND到AI处理器亲和数据排布格式（NZ）的转换。当数据类型为FLOAT4_E2M1时，还需要在aclnnNpuFormatCast调用后，调用aclnnCast接口将FLOAT32表示的FLOAT4_E2M1转换为正确的类型。但当为INT4类型时，需要使用aclnnConvertWeightToInt4Pack接口完成数据格式从ND到NZ和数据类型从INT32到INT4的转换。当传入FLOAT32或者INT32时，接口内部每个FLOAT32/INT32识别成8个FLOAT4_E2M1/INT4。
        - `scaleOptional`仅支持FLOAT32。`offsetOptional`、`antiquantOffsetOptional`暂不支持。
        - `quantGroupSize`暂不支持。

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，若出现以下错误码，则对应原因为：

    <table>
    <thead>
    <tr>
    <th style="width: 250px">返回值</th>
    <th style="width: 130px">错误码</th>
    <th style="width: 850px">描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td rowspan="4"> ACLNN_ERR_PARAM_NULLPTR </td>
    <td rowspan="4"> 161001 </td>
    <td>1.传入参数是必选输入、输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
    <td>2.传入参数weight的元素存在空指针。</td>
    </tr>
    <tr>
    <td>3.传入参数x的元素为空指针，且传出参数out的元素不为空指针。</td>
    </tr>
    <tr>
    <td>4.传入参数x的元素不为空指针，且传出参数out的元素为空指针。</td>
    </tr>
    <tr>
    <td rowspan="6"> ACLNN_ERR_PARAM_INVALID </td>
    <td rowspan="6"> 161002 </td>
    <td>1.x、weight、biasOptional、scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、groupListOptional、out的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
    <td>2.weight的长度大于128；若bias不为空，bias的长度不等于weight的长度。</td>
    </tr>
    <tr>
    <td>3.groupListOptional维度为1。</td>
    </tr>
    <tr>
    <td>4.splitItem为2、3的场景，out长度不等于1。</td>
    </tr>
    <tr>
    <td>5.splitItem为0、1的场景，out长度不等于weight的长度，groupListOptional长度不等于weight的长度。</td>
    </tr>
    <tr>
    <td>6.传入参数tuningConfigOptional的元素为负数，或者大于x的行数m。</td>
    </tr>
    </tbody>
    </table>

## aclnnGroupedMatmulWeightNz

  - **参数说明：**

    |参数名| 输入/输出   |    描述|
    |-------|---------|----------------|
    |workspace|输入|在Device侧申请的workspace内存地址。|
    |workspaceSize|输入|在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulWeightNzGetWorkspaceSize获取。|
    |executor|输入|op执行器，包含了算子计算流程。|
    |stream|输入|指定执行任务的Stream。|

  - **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnGroupedMatmulWeightNz默认确定性实现。
- **公共约束**
  - 如果传入groupListOptional，当groupListType为0时，groupListOptional必须为非负单调非递减数列；当groupListType为1时，groupListOptional必须为非负数列，且长度不能为1；groupListType为2时，groupListOptional的第二列数据必须为非负数列，且长度不能为1。
  - x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647。

<details>
<summary><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></summary>

  - 非量化场景支持的输入类型为：

    - x为FLOAT16、weight为FLOAT16、biasOptional为FLOAT16、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空、activationInputOptional为空、out为FLOAT16。
    - x为BFLOAT16、weight为BFLOAT16、biasOptional为FLOAT32、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空、activationInputOptional为空、out为BFLOAT16。

  - 量化场景支持的输入类型为：

    - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为BFLOAT16、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空或FLOAT32、activationInputOptional为空、out为BFLOAT16。
    - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为FLOAT32、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空或为FLOAT32、activationInputOptional为空、out为FLOAT16。
    - x为INT4、weight为INT4、biasOptional为空、scaleOptional为UINT64、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空或为FLOAT32、activationInputOptional为空、out为FLOAT16或BFLOAT16。

  - 伪量化场景支持的输入类型为：
  
    - 伪量化参数antiquantScaleOptional和antiquantOffsetOptional的shape要满足下表（其中g为matmul组数，G为pergroup数，$G_i$为第i个tensor的pergroup数）：
        | 使用场景 | 子场景 | shape限制 |
        |:---------:|:-------:| :-------|
        | 伪量化perchannel | weight单 | $[E, N]$|
        | 伪量化perchannel | weight多 | $[n_i]$|
        | 伪量化pergroup | weight单 | $[E, G, N]$|
        | 伪量化pergroup | weight多 | $[G_i, N_i]$|
    - x为INT8、weight为INT4、biasOptional为FLOAT32、scaleOptional为UINT64、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为FLOAT32、activationInputOptional为空。此场景支持对称量化和非对称量化：

      - 对称量化场景：

        - 输出out的dtype为BFLOAT16或FLOAT16
        - offsetOptional为空
        - 仅支持count模式（算子不会检查groupListType的值），k要求为quantGroupSize的整数倍，且要求k <= 18432。其中quantGroupSize为k方向上pergroup量化长度，当前支持quantGroupSize=256。
        - scale为pergroup与perchannel离线融合后的结果，shape要求为$[E, quantGroupNum, N]$，其中$quantGroupNum=k \div quantGroupSize$。
        - Bias为计算过程中离线计算的辅助结果，值要求为$8\times weight \times scale$，并在第1维累加，shape要求为$[E, N]$。
        - 要求N为8的整数倍。

      - 非对称量化场景：

        - 输出out的dtype为FLOAT16
        - 仅支持count模式（算子不会检查groupListType的值）。
        - {k, n}要求为{7168, 4096}或者{2048, 7168}。
        - scale为pergroup与perchannel离线融合后的结果，shape要求为$[E, 1, N]$。
        - offsetOptional不为空。非对称量化offsetOptional为计算过程中离线计算辅助结果，即$antiquantOffset \times scale$，shape要求为$[E, 1, N]$，dtype为FLOAT32。
        - Bias为计算过程中离线计算的辅助结果，值要求为$8\times weight \times scale$，并在第1维累加，shape要求为$[E, N]$。
        - 要求N为8的整数倍。

    - 伪量化场景下，若weight的类型为INT8，仅支持perchannel模式；若weight的类型为INT4，对称量化支持perchannel和pergroup两种模式。若为pergroup，pergroup数G或$G_i$必须要能整除对应的$k_i$。若weight为多tensor，定义pergroup长度$s_i = k_i / G_i$，要求所有$s_i(i=1,2,...g)$都相等。非对称量化支持perchannel模式。

    - 伪量化场景下若weight的类型为INT4，则weight中每一组tensor的最后一维大小都应是偶数。$weight_i$的最后一维指weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。并且在pergroup场景下，当weight转置时，要求pergroup长度$s_i$是偶数。

  - 不同groupType支持场景:
    - 量化、伪量化仅支持groupType为-1和0场景。
    - 支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，y，例如单多单表示支持x为单tensor，weight多tensor，y单tensor的场景。

      | groupType | 支持场景 | 场景限制 |
      |:---------:|:---------:| :-------|
      | -1 | 多多多 |1）仅支持splitItem为0/1<br>2）x中tensor要求维度一致，支持2-6维，weight中tensor需为2维，y中tensor维度和x保持一致<br>3）groupListOptional必须传空<br>4）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>5）x不支持转置 |
      | 0 | 单单单 |1）仅支持splitItem为2/3<br>2）weight中tensor需为3维，x，y中tensor需为2维<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总和与x中tensor的第一维相等，当groupListType为2时，第二列数值的总和与x中tensor的第一维相等<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）支持weight转置<br>6）x不支持转置 |
      | 0 | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional，且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总和与x中tensor的第一维相等且长度最大为128，当groupListType为2时，第二列数值的总和与x中tensor的第一维相等且长度最大为128<br>3）x,weight,y中tensor需为2维<br>4）weight中每个tensor的N轴必须相等<br>5）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置 |
      | 0 | 多多单 |1）仅支持splitItem为2/3<br>2）x,weight,y中tensor需为2维<br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应，当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应，且长度最大为128，当groupListType为2时，groupListOptional第二列的数值需与x中tensor的第一维一一对应，且长度最大为128<br>5）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置 |
</details>

<details>
<summary><term>Atlas 推理系列产品</term></summary>

  - 输入输出只支持float16的数据类型，输出y的n轴大小需要是16的倍数。
    
    支持场景中单表示单tensor，多表示多tensor，表示顺序为x、weight、y。例如单多单表示支持x为单tensor、weight多tensor、y单tensor的场景。
    | groupType | 支持场景 | 场景限制 |
    |:---------:|:-------:| :------ |
    | 0 | 单单单 |1）仅支持splitItem为2/3<br>2）weight中tensor需为3维，x，y中tensor需为2维<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总和与x中tensor的第一维相等<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）支持weight转置，不支持x转置 |
</details>

<details>
<summary><term>昇腾910_95 AI处理器</term></summary>

  - 当前仅支持非量化场景和伪量化场景
  - 非量化场景支持的数据类型为：
    - 输入weight矩阵的n轴与k轴需要满足32B对齐
    - 以下入参为空：scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional、actType、activationFeatureOutOptional
    - 不为空的参数支持的数据类型组合要满足下表
      |groupType| x       | weight  | biasOptional | out     |
      |:-------:|:-------:|:-------:| :------      |:------ |
      |-1/0   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
      |-1/0   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|

  - 伪量化场景支持的数据类型为：
    - 以下入参为空：scaleOptional、offsetOptional、antiquantOffsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional、actType、activationFeatureOutOptional
    - 不为空的参数支持的数据类型组合要满足下表
      |groupType| x       |pertokenScale| weight  |antiquantScale|scaleOptional|antiquantOffset| biasOptional | out     |
      |:-------:|:-------:|:-------:     | :-------:      | :------    | :------  |:------   |:------ |:------ |
      |0   |BFLOAT16      |null          |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null | BFLOAT16/FLOAT32/null    | BFLOAT16|
      |0   |FLOAT16       |null          |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null |FLOAT16/null              | FLOAT16|
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null |FLOAT16/null              | FLOAT16|
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null |BFLOAT16/null             | BFLOAT16|
      |0   |INT8          |FLOAT32       |INT4            |FLOAT16     |FLOAT32 |null |FLOAT32/null              | BFLOAT16|
      |0   |INT8          |FLOAT32       |INT4            |FLOAT16     |FLOAT32 |null |FLOAT32/null              | FLOAT16|
      |0   |BFLOAT16      |null          |FLOAT32         |FLOAT8_E8M0 |null    |null | BFLOAT16/FLOAT32/null    | BFLOAT16|
      |0   |FLOAT16       |null          |FLOAT32         |FLOAT8_E8M0 |null    |null |FLOAT16/null              | FLOAT16|
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT32         |FLOAT8_E8M0 |null    |null |FLOAT16/null              | FLOAT16|
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT32         |FLOAT8_E8M0 |null    |null |BFLOAT16/null             | BFLOAT16|
      |0   |INT8          |FLOAT32       |INT32           |FLOAT16     |FLOAT32 |null |FLOAT32/null              | BFLOAT16|
      |0   |INT8          |FLOAT32       |INT32           |FLOAT16     |FLOAT32 |null |FLOAT32/null              | FLOAT16|
    - 伪量化场景下，当x和weight的类型分别为BFLOAT16/FLOAT16和FLOAT4_E2M1/FLOAT32时，或为INT8和INT4/INT32时，仅支持x、weight均不转置, 为FLOAT8_E4M3FN和FLOAT4_E2M1/FLOAT32时仅支持x不转置且weight转置。

  - 不同groupType支持场景:

    - 支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，out，例如单多单表示支持x为单tensor，weight多tensor，out单tensor的场景。

      | groupType | 支持场景 | 场景限制 |
      |:---------:|:-------:| :------ |
      | -1 | 多多多 |1）仅支持splitItem为0/1<br>2）x，out中tensor需为2维， shape分别为（$m_i$, $k_i$）和（$m_i$, $n_i$）；weight中tensor需为2维，shape为（$n_i$, $k_i$）或（$k_i$, $n_i$）；bias中tensor需为1维，shape为（$n_i$）<br>3） groupListOptional必须传空<br>4）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>5）x不支持转置|
      | 0 | 单单单 |1）仅支持splitItem为2/3<br>2）weight中tensor需为3维，shape为（E, N, K）或（E, K, N）；x，out中tensor需为2维，shape分别为（M, K）和（M, N）；bias中tensor需为2维，shape为（E, N）<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）支持x不转置，weight转置、不转置均支持|
      | 0 | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional， 且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总与x中tensor的第一维相等，长度最大为 128<br>3）x，out中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>4）weight中每个tensor的N轴必须相等<br>5）支持weight转置，但weight的tensorList中每tensor是否转置需保持统一<br>6）x不支持转置|
      | 0 | 多多单 |1）仅支持splitItem为2/3<br>2）x，out中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应，当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应，且长度最大为128<br>5）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置|

</details>

## 调用示例
调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor_New(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
  std::vector<T> hostData(size / sizeof(T), 0);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
  int size = shapes.size();
  std::vector<aclTensor*> tensors(size);
  for (int i = 0; i < size; i++) {
    int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, &tensors[i]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *tensor = aclCreateTensorList(tensors.data(), size);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);

  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
  }
  
  // 检查shape维度
  if (shape.size() != 3) {
    LOG_PRINT("Shape must be 3D for NZ format\n");
    return -1;
  }
  
  int64_t E = shape[0];
  int64_t K = shape[1];
  int64_t N = shape[2];
  
  // 检查维度是否能被整除
  if (N % 64 != 0 || K % 16 != 0) {
    LOG_PRINT("N must be divisible by 64 and K by 16 for NZ format\n");
    return -1;
  }
  
  std::vector<int64_t> shapeNz = {E, N/64, K/16, 16, 64};

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                            shapeNz.data(), shapeNz.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorListNz(const std::vector<std::vector<T>> &hostData, 
                          const std::vector<std::vector<int64_t>> &shapes, 
                          void **deviceAddr,
                          aclDataType dataType, 
                          aclTensorList **tensor)
{
  if (hostData.size() != shapes.size()) {
    LOG_PRINT("hostData size %ld does not match shapes size %ld\n", hostData.size(), shapes.size());
    return -1;
  }
  
  int size = shapes.size();
  std::vector<aclTensor*> tensors(size);
  for (int i = 0; i < size; i++) {
    int ret = CreateAclTensorNz<T>(hostData[i], shapes[i], deviceAddr + i, dataType, &tensors[i]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *tensor = aclCreateTensorList(tensors.data(), size);
  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<std::vector<int64_t>> xShape = {{512, 256}};
  std::vector<std::vector<int64_t>> weightShape = {{2, 256, 256}};
  std::vector<std::vector<int64_t>> yShape = {{512, 256}};
  std::vector<int64_t> groupListShape = {2};
  std::vector<int64_t> groupListData = {256, 512};
  
  void* xDeviceAddr[1];
  void* weightDeviceAddr[1];
  void* yDeviceAddr[1];
  void* biasDeviceAddr[1] = {nullptr};  // 声明biasDeviceAddr
  void* groupListDeviceAddr;
  
  aclTensorList* x = nullptr;
  aclTensorList* weight = nullptr;
  aclTensorList* bias = nullptr;
  aclTensor* groupedList = nullptr;
  aclTensorList* scale = nullptr;
  aclTensorList* offset = nullptr;
  aclTensorList* antiquantScale = nullptr;
  aclTensorList* antiquantOffset = nullptr;
  aclTensorList* perTokenScale = nullptr;
  aclTensorList* activationInput = nullptr;
  aclTensorList* activationQuantScale = nullptr;
  aclTensorList* activationQuantOffset = nullptr;
  aclTensorList* out = nullptr;
  aclTensorList* activationFeatureOut = nullptr;
  aclTensorList* dynQuantScaleOut = nullptr;
  
  int64_t splitItem = 3;
  int64_t groupType = 0;
  int64_t groupListType = 0;
  int64_t actType = 0;
  
  // 创建weight数据
  int64_t weightTotalSize = 1;
  for (const auto& dim : weightShape[0]) {
    weightTotalSize *= dim;
  }
  std::vector<std::vector<int8_t>> wHostDataList(1);
  wHostDataList[0].resize(weightTotalSize * sizeof(uint16_t)); // BF16需要2字节
  
  // 创建tuningconfig aclIntArray
  std::vector<int64_t> tuningConfigData = {512};
  aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

  // 创建x aclTensorList
  ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 创建weight aclTensorList - NZ格式
  ret = CreateAclTensorListNz<int8_t>(wHostDataList, weightShape, weightDeviceAddr, aclDataType::ACL_BF16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 创建y aclTensorList
  ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_BF16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 创建group_list aclTensor
  ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. 调用CANN算子库API
  // 调用aclnnGroupedMatmulWeightNz第一段接口
  ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput, activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, tuningConfig, 0, out, activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnGroupedMatmulWeightNz第二段接口
  ret = aclnnGroupedMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  for (int i = 0; i < 1; i++) {
    auto size = GetShapeSize(yShape[i]);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < 20; j++) {
      LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
    }
    LOG_PRINT("......\n");
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensorList(x);
  aclDestroyTensorList(weight);
  if (bias) aclDestroyTensorList(bias);
  aclDestroyTensorList(out);
  if (groupedList) aclDestroyTensor(groupedList);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  for (int i = 0; i < 1; i++) {
    if (xDeviceAddr[i]) aclrtFree(xDeviceAddr[i]);
    if (weightDeviceAddr[i]) aclrtFree(weightDeviceAddr[i]);
    if (biasDeviceAddr[i]) aclrtFree(biasDeviceAddr[i]);
    if (yDeviceAddr[i]) aclrtFree(yDeviceAddr[i]);
  }
  if (groupListDeviceAddr) aclrtFree(groupListDeviceAddr);
  if (workspaceSize > 0 && workspaceAddr) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```