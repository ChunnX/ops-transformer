# aclnnSparseFlashMlaMetadata

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- 接口功能：该算子为AICPU算子，`SparseFlashMlaMetadata`算子为`SparseFlashMla`算子的前序算子，负责根据输入的序列长度信息和注意力配置参数，生成负载均衡的分核元数据（metadata）。该元数据包含每个AICore上FlashAttention计算任务的Batch、Head、Query分块和KV分块的索引，以及每个VectorCore上FlashDecode归约任务的索引信息。

  **该算子不建议单独使用，建议与aclnnSparseFlashMla算子配合使用，形成完整的工作流。**
- 计算公式：

  该算子为AICPU调度算子，不涉及数值计算。核心流程为：解析各Batch的Q/KV序列长度 → 根据mask模式计算每个S1G块的有效S2范围 → 基于开销模型进行负载均衡分核 → 输出分核元数据。

  输出metadata tensor的shape为(1024,)，数据类型为INT32，内部结构如下：

  - FA Metadata区域（AIC_CORE_NUM × 8个INT32），每个AICore的FA阶段任务信息：

    | 索引 | 含义 |
    | :--- | :--- |
    | 0 | core_enable，该核是否启用 |
    | 1 | bn2_start，BN2起始索引 |
    | 2 | m_start，M（S1G）起始索引 |
    | 3 | s2_start，S2起始索引 |
    | 4 | bn2_end，BN2结束索引 |
    | 5 | m_end，M结束索引 |
    | 6 | s2_end，S2结束索引 |
    | 7 | first_fd_data_workspace_idx，第一份FD归约数据的workspace偏移 |

  - FD Metadata区域（AIV_CORE_NUM × 8个INT32），每个AIVCore的FD归约任务信息：

    | 索引 | 含义 |
    | :--- | :--- |
    | 0 | core_enable，该核是否启用 |
    | 1 | bn2_idx，归约任务的BN2索引 |
    | 2 | m_idx，归约任务的M索引 |
    | 3 | workspace_idx，归约数据在workspace中的存放位置 |
    | 4 | workspace_num，S2核间切分份数 |
    | 5 | m_start，M轴起点 |
    | 6 | m_num，M轴行数 |

- 符号说明

  | 符号                | 含义                                                      |
  | ------------------- | --------------------------------------------------------- |
  | B                   | Batch Size                                                |
  | N1/N2               | Query/KV头数                                              |
  | D                   | 每个注意力头的维度                                        |
  | G                   | GQA分组比，G=N1/N2                                        |
  | S1/S2               | Query/KV序列长度                                          |
  | S1G                 | S1×G方向的分块索引                                        |
  | mBaseSize           | M轴基本块大小，等于G                                      |
  | s2BaseSize          | S2轴基本块大小，固定为512                                  |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnSparseFlashMlaMetadataGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnSparseFlashMlaMetadata`执行实际计算。

```c++
aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    int64_t          numHeadsQ,
    int64_t          numHeadsKv,
    int64_t          headDim,
    int64_t          batchSize,
    int64_t          maxSeqlenQ,
    int64_t          maxSeqlenOriKv,
    int64_t          maxSeqlenCmpKv,
    int64_t          oriTopk,
    int64_t          cmpTopk,
    int64_t          cmpRatio,
    int64_t          oriMaskMode,
    int64_t          cmpMaskMode,
    int64_t          oriWinLeft,
    int64_t          oriWinRight,
    const char      *layoutQOptional,
    const char      *layoutKvOptional,
    bool             hasOriKv,
    bool             hasCmpKv,
    const aclTensor *metaData,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnSparseFlashMlaMetadata(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnSparseFlashMlaMetadataGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width: 220px">
    <col style="width: 100px">
    <col style="width: 300px">
    <col style="width: 300px">
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 160px">
    <col style="width: 100px">
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
      <td>cuSeqlensQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中q的有效token数（前缀和形式）。</td>
      <td>layoutQOptional为TND时必须传入。每个元素表示当前batch与之前所有batch的token数总和。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中oriKv的有效token数（前缀和形式）。</td>
      <td>layoutKvOptional为TND时必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKv的有效token数（前缀和形式）。</td>
      <td>layoutKvOptional为TND且存在cmpKv时必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中q实际参与运算的token数。</td>
      <td>当前暂不支持指定该参数。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中oriKv实际参与运算的token数。</td>
      <td>layoutKvOptional为PA_BBND时必须传入；layoutKvOptional为BSND时可选传入，用于指定每个batch的oriKv有效长度；layoutKvOptional为TND时使用cuSeqlensOriKvOptional表达序列边界。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKv实际参与运算的token数。</td>
      <td>可选输入。传入时shape必须为(B,)，作为每个batch的cmp逻辑有效长度，优先于maxSeqlenCmpKv、cuSeqlensCmpKvOptional或PA block table推导；layoutKvOptional为BSND、TND、PA_BBND时均可使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>压缩KV余数，用于按cmp_len * cmpRatio + residual恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>在C4A/C128A、cmpRatio不等于1且cmpMaskMode为3场景必传，layoutKvOptional为BSND、TND、PA_BBND时均可使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>预留输入，当前版本不支持传入非空Tensor。</td>
      <td>必须传入nullptr或空Tensor；传入非空Tensor会返回参数错误。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>预留输入，当前版本不支持传入非空Tensor。</td>
      <td>必须传入nullptr或空Tensor；传入非空Tensor会返回参数错误。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>Query的多头数。</td>
      <td>仅支持1、2、4、8、16、32、64、128，numHeadsQ / numHeadsKv仅支持[1,128]范围内的2的幂。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv（int64_t）</td>
      <td>输入</td>
      <td>KV的多头数。</td>
      <td>仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim（int64_t）</td>
      <td>输入</td>
      <td>每个注意力头的维度。</td>
      <td>仅支持512。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>输入样本批量大小。</td>
      <td>默认值为0。layoutQ为TND时从cuSeqLensQ推断，无需手动指定。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ（int64_t）</td>
      <td>输入</td>
      <td>所有Batch中q的最大有效token数。</td>
      <td>默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenOriKv（int64_t）</td>
      <td>输入</td>
      <td>所有Batch中oriKv的最大有效token数。</td>
      <td>默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenCmpKv（int64_t）</td>
      <td>输入</td>
      <td>所有Batch中cmpKv的最大有效token数。</td>
      <td>默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriTopk（int64_t）</td>
      <td>输入</td>
      <td>从oriKv中筛选的稀疏token个数。</td>
      <td>当前暂不支持，默认值为0，当前仅支持0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpTopk（int64_t）</td>
      <td>输入</td>
      <td>从cmpKv中筛选的稀疏token个数。</td>
      <td>C4A场景下仅支持512或1024，C1A/C128A场景下为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>cmpKv相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>支持1、4、128；仅传入oriKv时不参与压缩KV计算，C4A场景传4，C128A场景传128。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和oriKv计算的mask模式。</td>
      <td>0: No Mask。<br/>3: RightDownCausal模式。<br/>4: Band模式。默认值为4。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和cmpKv计算的mask模式。</td>
      <td>0: No Mask。<br/>3: RightDownCausal模式。默认值为3。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft（int64_t）</td>
      <td>输入</td>
      <td>滑动窗口向左扩展的token数。</td>
      <td>支持-1或非负数，其中-1表示窗口不受限。默认值为127。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight（int64_t）</td>
      <td>输入</td>
      <td>滑动窗口向右扩展的token数。</td>
      <td>支持-1或非负数，其中-1表示窗口不受限。默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional（char*）</td>
      <td>输入</td>
      <td>标识输入q的数据排布格式。</td>
      <td>支持"BSND"和"TND"，默认值为"BSND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional（char*）</td>
      <td>输入</td>
      <td>标识输入KV的数据排布格式。</td>
      <td>支持"PA_BBND"、"BSND"和"TND"，默认值为"BSND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasOriKv（bool）</td>
      <td>输入</td>
      <td>是否传入oriKv。</td>
      <td>默认值为true。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasCmpKv（bool）</td>
      <td>输入</td>
      <td>是否传入cmpKv。</td>
      <td>C1A场景为false，C4A/C128A场景为true。默认值为true。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metaData（aclTensor*）</td>
      <td>输出</td>
      <td>分核元数据输出，供SparseFlashMla算子使用。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
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
  </tbody>
  </table>

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：numHeadsQ/numHeadsKv支持1、2、4、8、16、32、64、128，oriMaskMode仅支持4，cmpMaskMode仅支持3，oriWinLeft仅支持127，oriWinRight仅支持0。
  - <term>Ascend 950PR/Ascend 950DT</term>：numHeadsQ/numHeadsKv不支持1，oriMaskMode仅支持4，cmpMaskMode仅支持3，oriWinLeft仅支持127，oriWinRight仅支持0。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    <table style="undefined;table-layout: fixed;width: 1200px"><colgroup>
    <col style="width: 262px">
    <col style="width: 121px">
    <col style="width: 817px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建aclOpExecutor失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>workspaceSize或executor为空指针；可选输入做连续化处理后为空指针；或添加SparseFlashMlaMetadata AICPU任务失败。</td>
      </tr>
      <tr>
        <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="9">161002</td>
        <td>batchSize或maxSeqlenQ为负数。</td>
      </tr>
      <tr>
        <td>numHeadsQ不在[1,128]范围内，numHeadsKv不为1，numHeadsQ不能被numHeadsKv整除，或numHeadsQ/numHeadsKv不是[1,128]范围内的2的幂。</td>
      </tr>
      <tr>
        <td>headDim不为512。</td>
      </tr>
      <tr>
        <td>oriMaskMode不为4，或cmpMaskMode不为3。</td>
      </tr>
      <tr>
        <td>oriWinLeft不为127，或oriWinRight不为0。</td>
      </tr>
      <tr>
        <td>C1A场景cmpRatio不为1，或cmpRatio与C4A/C128A场景不匹配。</td>
      </tr>
      <tr>
        <td>cmpTopk不为0、512或1024。</td>
      </tr>
      <tr>
        <td>oriTopkLengthOptional或cmpTopkLengthOptional传入非空Tensor。</td>
      </tr>
      <tr>
        <td>layoutQOptional、layoutKvOptional、cuSeqlens、seqused或metaData的shape、数据类型、必选关系不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

  - <term>Ascend 950PR/Ascend 950DT</term>：

    <table style="undefined;table-layout: fixed;width: 1200px"><colgroup>
    <col style="width: 262px">
    <col style="width: 121px">
    <col style="width: 817px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建aclOpExecutor失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>workspaceSize或executor为空指针；可选输入做连续化处理后为空指针；或添加SparseFlashMlaMetadata AICPU任务失败。</td>
      </tr>
      <tr>
        <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="8">161002</td>
        <td>batchSize或maxSeqlenQ为负数。</td>
      </tr>
      <tr>
        <td>numHeadsQ不在[1,128]范围内，numHeadsKv不为1，numHeadsQ不能被numHeadsKv整除，或numHeadsQ/numHeadsKv不在[1,128]范围内。</td>
      </tr>
      <tr>
        <td>headDim不为512。</td>
      </tr>
      <tr>
        <td>oriMaskMode不为0、3、4，或cmpMaskMode不为0、3。</td>
      </tr>
      <tr>
        <td>oriWinLeft或oriWinRight小于-1。</td>
      </tr>
      <tr>
        <td>hasCmpKv为true时，cmpRatio不在[1,128]范围内。</td>
      </tr>
      <tr>
        <td>hasCmpKv为true时，cmpTopk为负数。</td>
      </tr>
      <tr>
        <td>layoutQOptional、layoutKvOptional、cuSeqlens、seqused或metaData的shape、数据类型、必选关系不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

## aclnnSparseFlashMlaMetadata

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseFlashMlaMetadataGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算

  - aclnnSparseFlashMlaMetadata默认采用确定性实现，相同输入多次调用结果一致。

- 使用约束
  - layoutQOptional和layoutKvOptional组合仅支持"BSND"/"BSND"、"TND"/"TND"、"BSND"/"PA_BBND"、"TND"/"PA_BBND"；非PA_BBND场景下layoutQOptional和layoutKvOptional必须一致。
  - layoutQOptional为TND时，`cuSeqlensQOptional`必须传入。
  - layoutKvOptional为PA_BBND时，`sequsedOriKvOptional`必须传入。BSND场景可选传入`sequsedOriKvOptional`覆盖每个batch的oriKv有效长度；TND场景使用`cuSeqlensOriKvOptional`表达oriKv序列边界。
  - layoutKvOptional为TND时，`cuSeqlensOriKvOptional`必须传入；若hasCmpKv为true，`cuSeqlensCmpKvOptional`也必须传入。
  - `sequsedCmpKvOptional`为所有layoutKvOptional下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度。
  - `cmpResidualKvOptional`为`aclnnSparseFlashMlaMetadata`和`aclnnSparseFlashMla`的可选输入，在C4A/C128A、cmpRatio不等于1且cmpMaskMode为3场景必传，用于恢复cmp侧mask使用的压缩前长度。
  - 该算子为AICPU算子，在Host侧CPU上执行，不占用NPU计算资源。


## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_metadata.h"

#include "../../sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_metadata.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  if (size > 0) {
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  } else {
    *deviceAddr = nullptr;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void PrintMetadataSummary(const optiling::detail::SasMetadata& meta)
{
  printf("AIC core0 enable=%u, bn2_end=%u, m_end=%u, s2_end=%u\n",
         meta.faMetadata[0][optiling::FA_CORE_ENABLE_INDEX],
         meta.faMetadata[0][optiling::FA_BN2_END_INDEX],
         meta.faMetadata[0][optiling::FA_M_END_INDEX],
         meta.faMetadata[0][optiling::FA_S2_END_INDEX]);
}

int main()
{
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 5;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int64_t B = 4;
  int64_t S1 = 128;
  int64_t S2 = 8192;
  int64_t N1 = 64;
  int64_t N2 = 1;
  int64_t D = 512;
  int64_t K = 512;
  int64_t s2Act = 4096;
  int64_t cmpRatio = 4;
  int64_t cmpKvLen = s2Act / cmpRatio;
  int64_t oriMaskMode = 4;
  int64_t cmpMaskMode = 3;
  int64_t oriWinLeft = 127;
  int64_t oriWinRight = 0;

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> cuSeqLensQShape = {B + 1};
  std::vector<int64_t> seqUsedOriKvShape = {B};
  std::vector<int64_t> cmpResidualKvShape = {B};
  std::vector<int64_t> metadataShape = {optiling::SMLA_META_SIZE};
  // 对全部 optional 输入调用 Contiguous，optional 输入传 shape 为 {0} 的空 tensor。
  std::vector<int64_t> emptyShape = {0};
  std::vector<int64_t> seqUsedQShape = emptyShape;

  void* cuSeqLensQDeviceAddr = nullptr;
  void* cuSeqLensOriKvDeviceAddr = nullptr;
  void* cuSeqLensCmpKvDeviceAddr = nullptr;
  void* seqUsedQDeviceAddr = nullptr;
  void* seqUsedOriKvDeviceAddr = nullptr;
  void* seqUsedCmpKvDeviceAddr = nullptr;
  void* cmpResidualKvDeviceAddr = nullptr;
  void* oriTopkLengthDeviceAddr = nullptr;
  void* cmpTopkLengthDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;

  aclTensor* cuSeqLensQ = nullptr;
  aclTensor* cuSeqLensOriKv = nullptr;
  aclTensor* cuSeqLensCmpKv = nullptr;
  aclTensor* seqUsedQ = nullptr;
  aclTensor* seqUsedOriKv = nullptr;
  aclTensor* seqUsedCmpKv = nullptr;
  aclTensor* cmpResidualKv = nullptr;
  aclTensor* oriTopkLength = nullptr;
  aclTensor* cmpTopkLength = nullptr;
  aclTensor* metadata = nullptr;

  std::vector<int32_t> cuSeqLensQHostData(B + 1);
  for (int64_t i = 0; i <= B; i++) {
    cuSeqLensQHostData[i] = static_cast<int32_t>(i * S1);
  }
  std::vector<int32_t> emptyHostData;
  std::vector<int32_t> seqUsedOriKvHostData(B, static_cast<int32_t>(s2Act));
  std::vector<int32_t> cmpResidualKvHostData(B, 0);
  std::vector<int32_t> metadataHostData(optiling::SMLA_META_SIZE, 0);

  ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensQShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensOriKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensCmpKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, seqUsedQShape, &seqUsedQDeviceAddr, aclDataType::ACL_INT32, &seqUsedQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqUsedOriKvHostData, seqUsedOriKvShape, &seqUsedOriKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &seqUsedCmpKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpResidualKvHostData, cmpResidualKvShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32, &cmpResidualKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &oriTopkLengthDeviceAddr, aclDataType::ACL_INT32, &oriTopkLength);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cmpTopkLengthDeviceAddr, aclDataType::ACL_INT32, &cmpTopkLength);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  char layoutQ[] = "TND";
  char layoutKv[] = "PA_BBND";

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  ret = aclnnSparseFlashMlaMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv,
      seqUsedQ, seqUsedOriKv, seqUsedCmpKv,
      cmpResidualKv, oriTopkLength, cmpTopkLength,
      N1, N2, D, B, S1, S2, cmpKvLen,
      0, K, cmpRatio,
      oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      true, true,
      metadata,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSparseFlashMlaMetadata(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashMlaMetadata failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  optiling::detail::SasMetadata result {};
  ret = aclrtMemcpy(&result, sizeof(result), metadataDeviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy metadata result failed. ERROR: %d\n", ret); return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintMetadataSummary(result);
  CHECK_RET(result.faMetadata[0][optiling::FA_CORE_ENABLE_INDEX] == 1U,
            LOG_PRINT("metadata validation failed: core0 is not enabled\n"); return 1);
  // 分核可能在 batch 内按行切分，此时 bn2_end 仍为 0，m_end 已推进。
  CHECK_RET(result.faMetadata[0][optiling::FA_BN2_END_INDEX] > 0U ||
                result.faMetadata[0][optiling::FA_M_END_INDEX] > 0U,
            LOG_PRINT("metadata validation failed: core0 has no assigned work\n"); return 1);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(cuSeqLensQ);
  aclDestroyTensor(cuSeqLensOriKv);
  aclDestroyTensor(cuSeqLensCmpKv);
  aclDestroyTensor(seqUsedQ);
  aclDestroyTensor(seqUsedOriKv);
  aclDestroyTensor(metadata);

  // 7. 释放device资源
  if (cuSeqLensQDeviceAddr != nullptr) {
    aclrtFree(cuSeqLensQDeviceAddr);
  }
  if (seqUsedOriKvDeviceAddr != nullptr) {
    aclrtFree(seqUsedOriKvDeviceAddr);
  }
  if (metadataDeviceAddr != nullptr) {
    aclrtFree(metadataDeviceAddr);
  }
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
