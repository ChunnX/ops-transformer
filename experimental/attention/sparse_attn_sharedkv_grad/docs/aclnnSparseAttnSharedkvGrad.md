# aclnnSparseAttnSharedkvGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |



## 功能说明

-   **接口功能**：计算`SparseAttnSharedkv`训练场景下注意力的反向输出，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。

-   **计算公式**：

    已知正向公式：
    $$
    Y = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于ori_kv、cmp_kv以及其它入参控制的实际参与计算的 $KV$。

    为方便表达，以变量$S$和$P$表示计算公式：

    $$
    S={Q@\tilde{K}^T \cdot \text{softmax\_scale}}
    $$

    $$
    P=Softmax(S)
    $$

    $$
    Y=P@\tilde{V}
    $$

    则注意力的反向计算公式为：

    $$
    dV=P^T@dY
    $$

    $$
    dQ=((dS)@K) \cdot \text{softmax\_scale}
    $$

    $$
    dK=((dS)^T@Q)\cdot \text{softmax\_scale}
    $$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSparseAttnSharedkvGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSparseAttnSharedkvGrad”接口执行计算。
```c++
aclnnStatus aclnnSparseAttnSharedkvGradGetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *oriKvOptional,
    const aclTensor   *cmpKvOptional,
    const aclTensor   *dOutOptional,
    const aclTensor   *outOptional,
    const aclTensor   *lseOptional,
    const aclTensor   *oriSparseIndicesOptional,
    const aclTensor   *cmpSparseIndicesOptional,
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensOriKvOptional,
    const aclTensor   *cuSeqlensCmpKvOptional,
    const aclTensor   *sinksOptional,
    double             scaleValue,
    int64_t            cmpRatio,
    int64_t            oriMaskMode,
    int64_t            cmpMaskMode,
    int64_t            oriWinLeft,
    int64_t            oriWinRight,
    char              *layoutOptional,
    const aclTensor   *dQueryOut,
    const aclTensor   *dOriKvOut,
    const aclTensor   *dCmpKvOut,
    const aclTensor   *dSinksOutOptional,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```
```c++
aclnnStatus aclnnSparseAttnSharedkvGrad(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream);
```

## aclnnSparseAttnSharedkvGradGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">  
            <col style="width: 400px">  
            <col style="width: 212px">  
            <col style="width: 100px">
            <col style="width: 290px">
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
        </tr></thead>
        <tbody>
        <tr>
            <td>query</td>
            <td>输入</td>
            <td>attention结构的输入Q。</td>
            <td>
            query、oriKvOptional、cmpKvOptional、dOutOptional、outOptional、lseOptional、oriSparseIndicesOptional、cmpSparseIndicesOptional的Shape维度保持一致。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            B：支持泛化；S1：支持泛化；N1：支持128、64、32、16、8、4、2、1；D：512；T1：B × S1
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>oriKvOptional</td>
            <td>输入</td>
            <td>attention结构的原始输入K(V)。</td>
            <td>-</td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)<br>
            B：与query的B保持一致；S2：支持泛化；N2：1；D：512；T2：B × S2
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>cmpKvOptional</td>
            <td>输入</td>
            <td>经过Compressor压缩后的K(V)。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S3,N2,D)、(T3,N2,D)<br>
            B：与query的B保持一致；S3 = S2 // cmpRatio；N2：1；D：512；T3：B × S3
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>dOutOptional</td>
            <td>输入</td>
            <td>注意力输出矩阵的梯度。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            B：与query的B保持一致；S1：与query的S1保持一致；N1：与query的N1保持一致；D：512；T1：B × S1
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>outOptional</td>
            <td>输入</td>
            <td>注意力输出矩阵。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            Shape与dOutOptional保持一致
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>lseOptional</td>
            <td>输入</td>
            <td>注意力正向计算的输出lse，计算公式详见正向文档。</td>
            <td>
            -
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G)、(N2,T1,G)<br>
            B：与query的B保持一致；N2：1；S1：与query的S1保持一致；G：N1/N2；T1：B × S1
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>oriSparseIndicesOptional</td>
            <td>输入</td>
            <td>稀疏场景下选择的oriKvOptional中权重较高的注意力索引。</td>
            <td>
            目前暂不支持对ori_kv进行稀疏计算，故设置此参数无效
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K1)、(T1,N2,K1)<br>
            B：与query的B保持一致；S1：与query的S1保持一致；N2：1；K1：支持泛化；T1：B × S1
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>cmpSparseIndicesOptional</td>
            <td>输入</td>
            <td>稀疏场景下选择的cmpKvOptional中权重较高的注意力索引。</td>
            <td>
            -
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K2)、(T1,N2,K2)<br>
            B：与query的B保持一致；S1：与query的S1保持一致；N2：1；K2：支持泛化；T1：B × S1
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>cuSeqlensQOptional</td>
            <td>输入</td>
            <td>每个Batch中，Query的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T1保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>cuSeqlensOriKvOptional</td>
            <td>输入</td>
            <td>每个Batch中，oriKvOptional的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T2保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>cuSeqlensCmpKvOptional</td>
            <td>输入</td>
            <td>每个Batch中，cmpKvOptional的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T3保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>sinksOptional</td>
            <td>输入</td>
            <td>注意力下沉tensor。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1)<br>
            N1：与query的N1保持一致
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>scaleValue</td>
            <td>输入</td>
            <td>缩放系数。</td>
            <td>
            建议值：公式中d开根号的倒数</li>
            </td>
            <td>FLOAT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>cmpRatio</td>
            <td>输入</td>
            <td>表示对oriKvOptional的压缩率。</td>
            <td>
            建议值：1、4、128。
            </td>
            <td>INT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>oriMaskMode</td>
            <td>输入</td>
            <td>表示query和oriKvOptional计算的mask模式。</td>
        <td>
              <ul>
                <li>表示sparse的模式。sparse不同模式的详细说明请参见<a href="#约束说明">约束说明</a>。</li>
                <li>仅支持模式4。</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
            <td>cmpMaskMode</td>
            <td>输入</td>
            <td>表示query和cmpKvOptional计算的mask模式。</td>
        <td>
              <ul>
                <li>表示sparse的模式。sparse不同模式的详细说明请参见<a href="#约束说明">约束说明</a>。</li>
                <li>仅支持模式3。</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
        <td>oriWinLeft</td>
            <td>输入</td>
            <td>表示query和oriKvOptional计算中query对过去token计算的数量。</td>
            <td>
            <ul>
                <li>仅支持取值：127</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
        <td>oriWinRight</td>
            <td>输入</td>
            <td>表示query和oriKvOptional计算中query对未来token计算的数量。</td>
            <td>
            <ul>
                <li>仅支持取值：0</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layoutOptional</td>
            <td>输入</td>
            <td>表示输入数据排布格式。</td>
            <td>
            支持"BSND"、"TND"。
            </td>
            <td>STRING</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dQueryOut</td>
            <td>输出</td>
            <td>表示query的梯度。</td>
            <td>
            与输入query的Shape维度保持一致
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>dOriKvOut</td>
            <td>输出</td>
            <td>表示oriKvOptional的梯度。</td>
            <td>
            与输入oriKvOptional的Shape维度保持一致
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)<br>
            </td>
            <td>-</td>
        </tr>
        <tr>  
            <td>dCmpKvOptional</td>
            <td>输出</td>
            <td>表示cmpKvOptional的梯度。</td>
            <td>
            与输入cmpKvOptional的Shape维度保持一致。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S3,N2,D)、(T3,N2,D)<br>
            </td>
            <td>-</td>
        </tr>
        <tr>  
            <td>dSinksOutOptional</td>
            <td>输出</td>
            <td>表示sinksOptional的梯度。</td>
            <td>
            与输入sinksOptional的Shape维度保持一致。
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1)<br>
            </td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

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
                <td>必选参数或者输出是空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入变量，如query、oriKvOptional、cmpKvOptional……的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>


## aclnnSparseAttnSharedkvGrad

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseAttnSharedkvGradGetWorkspaceSize获取。</td>
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

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明

- 确定性计算：
  - aclnnSparseAttnSharedkvGrad默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。
- 公共约束
    - 入参为空的场景处理：
        - query为空Tensor：直接返回。

- Mask
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>不做mask操作</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask，必须传入完整的attenmask矩阵</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>2</td>
            <td>leftUpCausal模式的mask，需要传入优化后的attenmask矩阵</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>4</td>
            <td>band模式的mask，滑窗范围由oriWinLeft、oriWinRight控制，参数起点为右下角。</td>
            <td>支持</td>
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
            <td>1~256</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1、S2</td>
            <td>1~128K</td>
            <td>S1、S2支持不等长</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1、2、4、8、16、32、64、128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>-</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>