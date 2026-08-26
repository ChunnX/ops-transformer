# aclnnQuantFlashAttnGrad

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

- 接口功能：计算量化注意力（QuantFlashAttention）的反向梯度。该接口为`aclnnQuantFlashAttention`正向算子的配套反向接口，用于计算Query、Key、Value的梯度（dq、dk、dv）以及sink梯度（dsink）。支持UINT8量化数据类型，支持BSND、BNSD两种数据排布格式。

- 计算公式：

    量化注意力反向梯度计算分为以下几个阶段：

    阶段一：反量化输入数据，将量化输入转换到高精度浮点域

    $$
    Q_{fp} = Q_{quant} \times q\_descale
    $$

    $$
    K_{fp} = K_{quant} \times k\_descale
    $$

    $$
    V_{fp} = V_{quant} \times v\_descale
    $$

    $$
    dO_{fp} = dO_{quant} \times do\_descale
    $$

    阶段二：计算Softmax梯度（PreSfmg阶段）

    $$
    dP = dO \times V
    $$

    $$
    dS = dP \times p\_scale
    $$

    $$
    dS_{grad} = (dS - ds\_scale \times softmax(lse)) \times softmaxLse
    $$

    阶段三：计算Key/Value梯度（Cube主计算阶段）

    $$
    dV = dS_{grad}^{T} \times Q
    $$

    $$
    dK = dS_{grad}^{T} \times Q
    $$

    $$
    dQ = dS_{grad} \times K
    $$

    阶段四：后处理（Post阶段），将中间结果转换输出dtype

    $$
    dq = dQ \times softmax\_scale
    $$

    $$
    dk = dK \times softmax\_scale
    $$

    $$
    dv = dV \times softmax\_scale
    $$

    $$
    dsink = reduce(dS_{grad}, dim=S)
    $$

    其中，$q\_descale$、$k\_descale$、$v\_descale$、$do\_descale$为量化缩放因子，$p\_scale$为P矩阵的缩放因子，$ds\_scale$为反量化缩放因子，$softmax\_scale$为缩放系数（建议值为$1/\sqrt{D}$）。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnQuantFlashAttnGradGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnQuantFlashAttnGrad"接口执行计算。
```c++
aclnnStatus aclnnQuantFlashAttnGradGetWorkspaceSize(
    const aclTensor   *q,
    const aclTensor   *k,
    const aclTensor   *v,
    const aclTensor   *dout,
    const aclTensor   *attnOut,
    const aclTensor   *qDescale,
    const aclTensor   *kDescale,
    const aclTensor   *vDescale,
    const aclTensor   *doDescale,
    const aclTensor   *pScale,
    const aclTensor   *dsScale,
    const aclTensor   *softmaxLse,
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensKvOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedKvOptional,
    const aclTensor   *sinksOptional,
    const aclTensor   *attnMaskOptional,
    const aclTensor   *metadataOptional,
    int64_t            quantMode,
    double             softmaxScale,
    int64_t            maskMode,
    int64_t            winLeft,
    int64_t            winRight,
    int64_t            maxSeqlenQ,
    int64_t            maxSeqlenKv,
    char              *layoutQOptional,
    char              *layoutKvOptional,
    const aclTensor   *dq,
    const aclTensor   *dk,
    const aclTensor   *dv,
    const aclTensor   *dsink,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```
```c++
aclnnStatus aclnnQuantFlashAttnGrad(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream);
```


## aclnnQuantFlashAttnGradGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 180px">
            <col style="width: 100px">
            <col style="width: 290px">
            <col style="width: 140px">
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
            <td>q（aclTensor*）</td>
            <td>输入</td>
            <td>attention结构的输入Q，量化数据。</td>
            <td>
            q、k的N轴需满足GQA约束（N1 = N2 × G）。
            </td>
            <td>UINT8</td>
            <td>ND</td>
            <td>BSND: (B,S1,N1,D)<br>BNSD: (B,N1,S1,D)<br>
            B：支持泛化；S1：支持泛化；N1：支持1~128；D：128
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>k（aclTensor*）</td>
            <td>输入</td>
            <td>attention结构的输入K，量化数据。</td>
            <td>
            当前暂不支持空tensor。
            </td>
            <td>UINT8</td>
            <td>ND</td>
            <td>BSND: (B,S2,N2,D)<br>BNSD: (B,N2,S2,D)<br>
            B：与q的B保持一致；S2：支持泛化；N2：与q的N1满足GQA约束；D：128
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>v（aclTensor*）</td>
            <td>输入</td>
            <td>attention结构的输入V，量化数据。</td>
            <td>
            shape需与k保持一致。
            </td>
            <td>UINT8</td>
            <td>ND</td>
            <td>BSND: (B,S2,N2,D)<br>BNSD: (B,N2,S2,D)<br>
            B：与q的B保持一致；S2：与k的S2保持一致；N2：与k的N2保持一致；D：128
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dout（aclTensor*）</td>
            <td>输入</td>
            <td>正向输出attn_out对应的梯度，量化数据。</td>
            <td>
            shape需与q保持一致。
            </td>
            <td>UINT8</td>
            <td>ND</td>
            <td>BSND: (B,S1,N1,D)<br>BNSD: (B,N1,S1,D)<br>
            与q的shape一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>attnOut（aclTensor*）</td>
            <td>输入</td>
            <td>正向计算输出的attn_out。</td>
            <td>
            shape需与q保持一致。
            </td>
            <td>BFLOAT16</td>
            <td>ND</td>
            <td>BSND: (B,S1,N1,D)<br>BNSD: (B,N1,S1,D)<br>
            与q的shape一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>qDescale（aclTensor*）</td>
            <td>输入</td>
            <td>Query的反量化缩放因子。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(1,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>kDescale（aclTensor*）</td>
            <td>输入</td>
            <td>Key的反量化缩放因子。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(1,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>vDescale（aclTensor*）</td>
            <td>输入</td>
            <td>Value的反量化缩放因子。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(1,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>doDescale（aclTensor*）</td>
            <td>输入</td>
            <td>dout的反量化缩放因子。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(1,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>pScale（aclTensor*）</td>
            <td>输入</td>
            <td>P矩阵的量化缩放因子。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(1,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dsScale（aclTensor*）</td>
            <td>输入</td>
            <td>反量化缩放因子。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(1,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxLse（aclTensor*）</td>
            <td>输入</td>
            <td>注意力正向计算的输出softmaxLse。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N1,S1,1)<br>
            B：与q的B保持一致；N1：与q的N1保持一致；S1：与q的S1保持一致
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensQOptional（aclTensor*）</td>
            <td>输入</td>
            <td>每个Batch中，Query的有效token数。</td>
            <td>
            <ul>
                <li>可选项。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T1保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensKvOptional（aclTensor*）</td>
            <td>输入</td>
            <td>每个Batch中，Key的有效token数。</td>
            <td>
            <ul>
                <li>可选项。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T2保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedQOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示不同batch中q实际参与运算的token数。</td>
            <td>
            长度为B。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedKvOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示不同batch中k实际参与运算的token数。</td>
            <td>
            长度为B。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sinksOptional（aclTensor*）</td>
            <td>输入</td>
            <td>sink场景下的输入tensor。</td>
            <td>
            <ul>
                <li>可选项，支持空tensor。</li>
                <li>当前版本不支持，传nullptr即可。</li>
            </ul>
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>attnMaskOptional（aclTensor*）</td>
            <td>输入</td>
            <td>注意力mask，表示q和k计算的mask模式。</td>
            <td>
            <ul>
                <li>可选项，支持空tensor。</li>
                <li>当maskMode为特定模式时需传入。</li>
            </ul>
            </td>
            <td>INT8</td>
            <td>ND</td>
            <td>(2048,2048)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>metadataOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示tiling下沉的aicpu算子输出结果。</td>
            <td>
            可选项，支持空tensor。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(2, max_schedule_size)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>quantMode（int64_t）</td>
            <td>输入</td>
            <td>量化模式。</td>
            <td>
            <ul>
                <li>0：HIFLOAT8量化</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>softmaxScale（double）</td>
            <td>输入</td>
            <td>缩放系数。</td>
            <td>
            建议值：公式中D开根号的倒数。
            </td>
            <td>FLOAT32</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>maskMode（int64_t）</td>
            <td>输入</td>
            <td>表示q和k计算的mask模式。0：No mask。默认值为0。</td>
            <td>
            当前版本仅支持0（不做mask操作），其他模式暂不支持。
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>winLeft（int64_t）</td>
            <td>输入</td>
            <td>滑窗mask的左窗口大小。</td>
            <td>
            maskMode为band模式时有效，-1表示不使用。
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>winRight（int64_t）</td>
            <td>输入</td>
            <td>滑窗mask的右窗口大小。</td>
            <td>
            maskMode为band模式时有效，-1表示不使用。
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>maxSeqlenQ（int64_t）</td>
            <td>输入</td>
            <td>Query的最大序列长度。</td>
            <td>
            -1表示自动推导。
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>maxSeqlenKv（int64_t）</td>
            <td>输入</td>
            <td>Key/Value的最大序列长度。</td>
            <td>
            -1表示自动推导。
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layoutQOptional（char*）</td>
            <td>输入</td>
            <td>表示输入q的数据排布格式。</td>
            <td>
            支持"BSND"、"BNSD"。需与layoutKvOptional一致。
            </td>
            <td>STRING</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layoutKvOptional（char*）</td>
            <td>输入</td>
            <td>表示输入k/v的数据排布格式。</td>
            <td>
            支持"BSND"、"BNSD"。需与layoutQOptional一致。
            </td>
            <td>STRING</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dq（aclTensor*）</td>
            <td>输出</td>
            <td>Query的梯度。</td>
            <td>
            dtype固定为BFLOAT16。shape与q一致。
            </td>
            <td>BFLOAT16</td>
            <td>ND</td>
            <td>BSND: (B,S1,N1,D)<br>BNSD: (B,N1,S1,D)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dk（aclTensor*）</td>
            <td>输出</td>
            <td>Key的梯度。</td>
            <td>
            dtype固定为BFLOAT16。shape与k一致。
            </td>
            <td>BFLOAT16</td>
            <td>ND</td>
            <td>BSND: (B,S2,N2,D)<br>BNSD: (B,N2,S2,D)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dv（aclTensor*）</td>
            <td>输出</td>
            <td>Value的梯度。</td>
            <td>
            dtype固定为BFLOAT16。shape与v一致。
            </td>
            <td>BFLOAT16</td>
            <td>ND</td>
            <td>BSND: (B,S2,N2,D)<br>BNSD: (B,N2,S2,D)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dsink（aclTensor*）</td>
            <td>输出</td>
            <td>sink的梯度。</td>
            <td>
            dtype固定为FLOAT32。
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1,)<br>
            </td>
            <td>√</td>
        </tr>
        </tbody>
    </table>

- **返回值**

   aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

   第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>参数中存在非法的nullptr。</td>
            </tr>
            <tr>
                <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="3">161002</td>
                <td>输入的数据类型不满足支持类型。</td>
            </tr>
            <tr>
                <td>q、k、v、dout、attnOut、qDescale、kDescale、vDescale、doDescale、pScale、dsScale、softmaxLse必选输入未传。</td>
            </tr>
            <tr>
                <td>q、k、v、dout、attnOut的维度不满足约束条件。</td>
            </tr>
        </tbody>
    </table>


## aclnnQuantFlashAttnGrad

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
    </colgroup>
    <thead>
        <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantFlashAttnGradGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream流。</td>
        </tr>
    </tbody>
    </table>

- **返回值**

   aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。


## 约束说明

- 确定性说明：aclnnQuantFlashAttnGrad默认确定性实现。
- 公共约束：
    - 入参为空处理：q为空Tensor时直接返回。
    - UINT8量化场景下，D必须为128。
    - UINT8量化场景下，q、k、v、dout、attnOut的维度必须一致。
    - k和v的shape必须一致。

- Mask

    当前版本仅支持maskMode=0（不做mask操作），其他模式暂不支持。

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 540px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>maskMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>不做mask操作（ALL_MASK）</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
            <td>当前版本不支持</td>
        </tr>
        <tr>
            <td>4</td>
            <td>band模式的mask，对应滑窗场景。</td>
            <td>当前版本不支持</td>
        </tr>
        </tbody>
    </table>

- 规格约束

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>B</td>
            <td>支持泛化</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1、S2</td>
            <td>支持泛化</td>
            <td>支持S1、S2不等长</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1~128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1~128</td>
            <td>需满足N1 = N2 × G（GQA约束）</td>
        </tr>
        <tr>
            <td>D</td>
            <td>128</td>
            <td>UINT8量化场景固定为128</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/BNSD</td>
            <td>layoutQ与layoutKv需保持一致</td>
        </tr>
        <tr>
            <td>quantMode</td>
            <td>0</td>
            <td>0：UINT8量化</td>
        </tr>
        </tbody>
    </table>


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_flash_attn_grad.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateEmptyAclTensor(const std::vector<int64_t> &shape, void **deviceAddr, aclDataType dataType,
                         aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // BSND layout: q=[B,Sq,N,D], k=[B,Skv,N,D], v=[B,Skv,N,D], dout=[B,Sq,N,D]
    // 量化场景 D=128
    int64_t B = 1;
    int64_t Sq = 512;
    int64_t Skv = 512;
    int64_t N = 1;
    int64_t D = 128;

    std::vector<int64_t> qShape = {B, Sq, N, D};
    std::vector<int64_t> kShape = {B, Skv, N, D};
    std::vector<int64_t> vShape = {B, Skv, N, D};
    std::vector<int64_t> doutShape = {B, Sq, N, D};
    std::vector<int64_t> attnOutShape = {B, Sq, N, D};
    std::vector<int64_t> qDescaleShape = {1};
    std::vector<int64_t> kDescaleShape = {1};
    std::vector<int64_t> vDescaleShape = {1};
    std::vector<int64_t> doDescaleShape = {1};
    std::vector<int64_t> pScaleShape = {1};
    std::vector<int64_t> dsScaleShape = {1};
    std::vector<int64_t> softmaxLseShape = {B, N, Sq, 1};
    std::vector<int64_t> metadataShape = {64};
    std::vector<int64_t> dqShape = {B, Sq, N, D};
    std::vector<int64_t> dkShape = {B, Skv, N, D};
    std::vector<int64_t> dvShape = {B, Skv, N, D};
    std::vector<int64_t> dsinkShape = {N};

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *doutDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;
    void *qDescaleDeviceAddr = nullptr;
    void *kDescaleDeviceAddr = nullptr;
    void *vDescaleDeviceAddr = nullptr;
    void *doDescaleDeviceAddr = nullptr;
    void *pScaleDeviceAddr = nullptr;
    void *dsScaleDeviceAddr = nullptr;
    void *softmaxLseDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    void *dqDeviceAddr = nullptr;
    void *dkDeviceAddr = nullptr;
    void *dvDeviceAddr = nullptr;
    void *dsinkDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *dout = nullptr;
    aclTensor *attnOut = nullptr;
    aclTensor *qDescale = nullptr;
    aclTensor *kDescale = nullptr;
    aclTensor *vDescale = nullptr;
    aclTensor *doDescale = nullptr;
    aclTensor *pScale = nullptr;
    aclTensor *dsScale = nullptr;
    aclTensor *softmaxLse = nullptr;
    aclTensor *metadata = nullptr;
    aclTensor *dq = nullptr;
    aclTensor *dk = nullptr;
    aclTensor *dv = nullptr;
    aclTensor *dsink = nullptr;

    // UINT8 量化输入 (uint8_t 承载)
    std::vector<uint8_t> qHostData(GetShapeSize(qShape), 0);
    std::vector<uint8_t> kHostData(GetShapeSize(kShape), 0);
    std::vector<uint8_t> vHostData(GetShapeSize(vShape), 0);
    std::vector<uint8_t> doutHostData(GetShapeSize(doutShape), 0);
    // attn_out: BF16
    std::vector<short> attnOutHostData(GetShapeSize(attnOutShape), 0);
    // descale / scale: FP32
    std::vector<float> qDescaleHostData(GetShapeSize(qDescaleShape), 1.0f);
    std::vector<float> kDescaleHostData(GetShapeSize(kDescaleShape), 1.0f);
    std::vector<float> vDescaleHostData(GetShapeSize(vDescaleShape), 1.0f);
    std::vector<float> doDescaleHostData(GetShapeSize(doDescaleShape), 1.0f);
    std::vector<float> pScaleHostData = {1.0f};
    std::vector<float> dsScaleHostData = {1.0f};
    std::vector<float> softmaxLseHostData(GetShapeSize(softmaxLseShape), 0.0f);
    // metadata: INT32
    std::vector<int32_t> metadataHostData(GetShapeSize(metadataShape), 0);

    // 创建输入 tensor
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_UINT8, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_UINT8, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_UINT8, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doutHostData, doutShape, &doutDeviceAddr, aclDataType::ACL_UINT8, &dout);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_BFLOAT16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qDescaleHostData, qDescaleShape, &qDescaleDeviceAddr, aclDataType::ACL_FLOAT, &qDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kDescaleHostData, kDescaleShape, &kDescaleDeviceAddr, aclDataType::ACL_FLOAT, &kDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vDescaleHostData, vDescaleShape, &vDescaleDeviceAddr, aclDataType::ACL_FLOAT, &vDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doDescaleHostData, doDescaleShape, &doDescaleDeviceAddr, aclDataType::ACL_FLOAT, &doDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pScaleHostData, pScaleShape, &pScaleDeviceAddr, aclDataType::ACL_FLOAT, &pScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dsScaleHostData, dsScaleShape, &dsScaleDeviceAddr, aclDataType::ACL_FLOAT, &dsScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxLse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 tensor (BF16 / FP32)
    ret = CreateEmptyAclTensor<short>(dqShape, &dqDeviceAddr, aclDataType::ACL_BFLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateEmptyAclTensor<short>(dkShape, &dkDeviceAddr, aclDataType::ACL_BFLOAT16, &dk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateEmptyAclTensor<short>(dvShape, &dvDeviceAddr, aclDataType::ACL_BFLOAT16, &dv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateEmptyAclTensor<float>(dsinkShape, &dsinkDeviceAddr, aclDataType::ACL_FLOAT, &dsink);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 设置属性并调用算子
    int64_t quantMode = 0;          // 0: UINT8
    double softmaxScale = 0.088388; // 1/sqrt(128)
    int64_t maskMode = 0;           // 0: ALL_MASK
    int64_t winLeft = -1;
    int64_t winRight = -1;
    int64_t maxSeqlenQ = Sq;
    int64_t maxSeqlenKv = Skv;
    char layoutQ[5] = {'B', 'S', 'N', 'D', 0};
    char layoutK[5] = {'B', 'S', 'N', 'D', 0};

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnQuantFlashAttnGradGetWorkspaceSize(
        q, k, v, dout, attnOut, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, softmaxLse,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, metadata,
        quantMode, softmaxScale, maskMode, winLeft, winRight, maxSeqlenQ, maxSeqlenKv, layoutQ, layoutK,
        dq, dk, dv, dsink, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnQuantFlashAttnGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnQuantFlashAttnGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantFlashAttnGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnQuantFlashAttnGrad succeeded.\n");

    // 释放资源
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(dout);
    aclDestroyTensor(attnOut);
    aclDestroyTensor(qDescale);
    aclDestroyTensor(kDescale);
    aclDestroyTensor(vDescale);
    aclDestroyTensor(doDescale);
    aclDestroyTensor(pScale);
    aclDestroyTensor(dsScale);
    aclDestroyTensor(softmaxLse);
    aclDestroyTensor(metadata);
    aclDestroyTensor(dq);
    aclDestroyTensor(dk);
    aclDestroyTensor(dv);
    aclDestroyTensor(dsink);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(doutDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
    aclrtFree(qDescaleDeviceAddr);
    aclrtFree(kDescaleDeviceAddr);
    aclrtFree(vDescaleDeviceAddr);
    aclrtFree(doDescaleDeviceAddr);
    aclrtFree(pScaleDeviceAddr);
    aclrtFree(dsScaleDeviceAddr);
    aclrtFree(softmaxLseDeviceAddr);
    aclrtFree(metadataDeviceAddr);
    aclrtFree(dqDeviceAddr);
    aclrtFree(dkDeviceAddr);
    aclrtFree(dvDeviceAddr);
    aclrtFree(dsinkDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
