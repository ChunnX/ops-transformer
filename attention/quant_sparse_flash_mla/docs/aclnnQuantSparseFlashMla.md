# aclnnQuantSparseFlashMla

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- 接口功能：

  `aclnnQuantSparseFlashMla`算子旨在完成全量化和稀疏场景下的MLA（Multi-head Latent Attention）注意力计算。该接口支持以下五类计算模式：
  - **SWA（Sliding Window Attention）**：仅传入`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时传入`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和TopK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时传入`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。
  - **ORI_SPARSE**：仅传入`ori_kv`和`ori_sparse_indices`，对原始KV窗口做稀疏注意力。
  - **ORI_CMP_SPARSE**：同时传入`ori_kv`、`cmp_kv`、`ori_sparse_indices`和`cmp_sparse_indices`，对原始KV窗口和连续压缩KV段共同做稀疏注意力。
  
  调用时需要使用`aclnnQuantSparseFlashMlaMetadata`生成的任务列表`metadata`。
  **该算子不建议单独使用，建议与aclnnQuantSparseFlashMlaMetadata算子配合使用，形成完整的工作流。**
  
  典型调用流程如下：

  1. 准备`q`、`ori_kv`、`cmp_kv`、序列长度、`block table`、`sinks`等输入。
  2. 调用`aclnnQuantSparseFlashMlaMetadata`生成`metadata`。
  3. 调用`aclnnQuantSparseFlashMla`，将上一步得到的`metadata`传入主算子。

- 计算公式：

  $$
  O = \text{softmax}(Q \cdot \tilde{K}^T \cdot \text{softmax\_scale}) \cdot \tilde{V}
  $$

  其中$\tilde{K} = \tilde{V}$（共享KV），$\tilde{K}$由滑动窗口内的原始KV和因果边界内的压缩KV拼接而成，具体参与计算的KV范围由模板模式和mask参数决定：

  - 滑动窗口部分（oriKv）：对第$i_{S1}$个Query token，其因果对角线位置为$\text{ori\_threshold} = S2_{act} - S1_{act} + i_{S1} + 1$，窗口范围为$[\max(\text{ori\_threshold} - \text{ori\_win\_left} - 1, 0), \text{ori\_threshold} + \text{ori\_win\_right})$。

  - 压缩KV部分（cmpKv）：因果边界阈值为$\text{cmp\_threshold} = \lfloor \frac{\text{ori\_threshold}}{\text{cmp\_ratio}} \rfloor$。HCA场景取$[0, \text{cmp\_threshold})$内的连续压缩KV；CSA场景通过TopK索引从压缩KV中按需收集，仅保留$\text{begin\_idx} < \text{cmp\_threshold}$的块。

  注意力计算采用Online Softmax（Flash Attention V2），S2方向按分块循环，sinks作为每行softmax的初始最大值：

  $$
  \text{row\_max}^{(0)} = \text{sinks}[g], \quad \text{row\_sum}^{(0)} = 1.0, \quad O^{(0)} = 0
  $$

  $$
  S^{(t)} = Q \cdot K_{tile}^{(t)T} \cdot \text{softmax\_scale}
  $$

  $$
  \text{row\_max}^{(t+1)} = \max(\text{row\_max}^{(t)}, \max(S^{(t)}, \text{dim}=-1))
  $$

  $$
  \text{row\_sum}^{(t+1)} = \exp(\text{row\_max}^{(t)} - \text{row\_max}^{(t+1)}) \cdot \text{row\_sum}^{(t)} + \sum \exp(S^{(t)} - \text{row\_max}^{(t+1)})
  $$

  $$
  O^{(t+1)} = \exp(\text{row\_max}^{(t)} - \text{row\_max}^{(t+1)}) \cdot O^{(t)} + \exp(S^{(t)} - \text{row\_max}^{(t+1)}) \cdot V_{tile}^{(t)}
  $$

  $$
  O_{final} = O^{(T_{s2})} / \text{row\_sum}^{(T_{s2})}
  $$

- 符号说明

  | 符号                | 含义                                                      |
  | ------------------- | --------------------------------------------------------- |
  | Q                   | Query输入，形状为[G, D]（单行）                           |
  | K_tile_t            | 第t个S2分块的KV数据，K=V（共享KV）                         |
  | S_t                 | 第t个分块的QK缩放注意力分数                                |
  | P_t                 | 第t个分块的softmax概率                                     |
  | O_t                 | 第t个分块后的累加输出                                      |
  | softmax_scale       | 缩放系数，通常取每个注意力头维度的倒数平方根                |
  | B                   | Batch Size                                                |
  | S1/S1_act           | Query序列长度/实际有效长度                                 |
  | S2/S2_act           | 原始KV序列长度/实际有效长度                                |
  | N1                  | Query头数                                                 |
  | N2                  | KV头数                                                    |
  | G                   | GQA分组比，G=N1/N2                                        |
  | D                   | 每个注意力头的维度                                        |
  | sinks               | 注意力汇点，形状为[N1]                                    |
  | cmp_ratio           | cmpKv的压缩倍率，用于换算cmp侧mask的压缩前KV长度            |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnQuantSparseFlashMlaGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnQuantSparseFlashMla`执行实际计算。

```Cpp
aclnnStatus aclnnQuantSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
    const aclTensor *qDescaleOptional,
    const aclTensor *oriKvDescaleOptional,
    const aclTensor *cmpKvDescaleOptional,
    const aclTensor *oriSparseIndicesOptional,
    const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional,
    const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    int64_t          quantMode,
    double           softmaxScale,
    int64_t          cmpRatio,
    int64_t          oriMaskMode,
    int64_t          cmpMaskMode,
    int64_t          oriWinLeft,
    int64_t          oriWinRight,
    char            *layoutQOptional,
    char            *layoutKvOptional,
    int64_t          topkValueMode,
    bool             returnSoftmaxLse,
    const aclTensor *attnOutOut,
    const aclTensor *softmaxLseOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantSparseFlashMla(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantSparseFlashMlaGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 300px">
    <col style="width: 300px">
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 280px">
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
      <td>q（aclTensor*）</td>
      <td>输入</td>
      <td>Query输入张量。</td>
      <td>不支持空Tensor。N1/N2仅支持1到128；D仅支持512。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(B, S1, N1, D)</li>
          <li>layoutQ为TND时：(T1, N1, D)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>原始KV输入张量，Key与Value共享同一份数据。</td>
      <td>SWA/CSA/HCA/ORI_SPARSE/ORI_CMP_SPARSE场景必须传入。对应公式中K和V的一部分，表示原始不经压缩的量化KV。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutKv为PA_BBND时：(ori_block_num, ori_block_size, N2, D_KV)，ori_block_size支持1到1024</li>
          <li>layoutKv为BSND时：(B, S2, N2, D_KV)</li>
          <li>layoutKv为TND时：(T2, N2, D_KV)</li>
        </ul>
        N2仅支持1，D_KV仅支持512。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>压缩KV输入张量，Key与Value共享同一份数据。</td>
      <td>CSA/HCA/ORI_CMP_SPARSE场景必须传入，SWA/ORI_SPARSE场景不传入。同oriKvOptional。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutKv为PA_BBND时：(cmp_block_num, cmp_block_size, N2, D_KV)，cmp_block_size支持1到1024</li>
          <li>layoutKv为BSND时：(B, S3, N2, D_KV)</li>
          <li>layoutKv为TND时：(T3, N2, D_KV)</li>
        </ul>
        N2仅支持1，D_KV仅支持512。
      </td>
      <td>√</td>
    </tr>
    <tr>
    <td>qDescaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>对应q的量化参数。</td>
      <td>当前必传。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>
        当前shape固定为[1]，对应per-tensor量化。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriKvDescaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>对应oriKvOptional的量化参数。</td>
      <td>当前必传。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>
        当前shape固定为[1]，对应per-tensor量化。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpKvDescaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>对应cmpKvOptional的量化参数。</td>
      <td>当前必传。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>
        当前shape固定为[1]，对应per-tensor量化。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriSparseIndicesOptional（aclTensor*）</td>
      <td>输入</td>
      <td>代表离散取oriKvCache的索引。</td>
      <td>CSA/ORI_CMP_SPARSE场景必须传入，SWA/HCA/ORI_SPARSE场景不传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td
        <ul>
          <li>layoutQ为BSND时：(B, S1, N2, K1)</li>
          <li>layoutQ为TND时：(T1, N2, K1)</li>
        </ul>
        其中K1为对oriKvOptional的TopK稀疏选择数，范围支持大于0。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpSparseIndicesOptional（aclTensor*）</td>
      <td>输入</td>
      <td>代表离散取cmpKvCache的TopK索引。</td>
      <td>CSA/ORI_CMP_SPARSE场景必须传入，SWA/HCA/ORI_SPARSE场景不传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(B, S1, N2, K2)</li>
          <li>layoutQ为TND时：(T1, N2, K2)</li>
        </ul>
        其中K2为对cmpKvOptional的TopK稀疏选择数，范围支持大于0。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriBlockTableOptional（aclTensor*）</td>
      <td>输入</td>
      <td>PageAttention中oriKvCache存储使用的block映射表。</td>
      <td>layoutKv为PA_BBND时必须传入。第二维长度不小于所有batch中最大的S2对应的block数量。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, ori_max_block_num_per_batch)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpBlockTableOptional（aclTensor*）</td>
      <td>输入</td>
      <td>PageAttention中cmpKvCache存储使用的block映射表。</td>
      <td>CSA/HCA/ORI_CMP_SPARSE场景且layoutKv为PA_BBND时必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, cmp_max_block_num_per_batch)</td>
      <td>√</td>
    </tr>
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
      <td>layoutKvOptional为TND且存在cmpKvOptional时必须传入。</td>
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
      <td>可选输入。传入时shape必须为(B,)，作为每个batch的cmp逻辑有效长度，优先于cmpKvOptional shape、cuSeqlensCmpKvOptional或PA block table推导；layoutKvOptional为BSND、TND、PA_BBND时均可使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>压缩KV余数，用于恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>可选输入。传入时shape必须为(B,)，第b个batch按cmp_len * cmpRatio + cmpResidualKvOptional[b]恢复压缩前KV长度；在CSA/HCA/ORI_CMP_SPARSE、cmpRatio不等于1且cmpMaskMode为3场景必传。该参数是主算子和aclnnQuantSparseFlashMlaMetadata的可选入参，layoutKvOptional为BSND、TND、PA_BBND时均可使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>用于标识oriSparseIndicesOptional实际参与计算的长度。</td>
      <td>ORI_SPARSE/ORI_CMP_SPARSE场景下必传；其他场景不支持传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
      <ul>
        <ul>
          <li>layoutQ为BSND时：(B, S1, N2)</li>
          <li>layoutQ为TND时：(T1, N2)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>用于标识cmpSparseIndicesOptional实际参与计算的长度。</td>
      <td>ORI_SPARSE/ORI_CMP_SPARSE/CSA在cmp_mask_mode=0场景下必传；其他场景不支持传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(B, S1, N2)</li>
          <li>layoutQ为TND时：(T1, N2)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>sinksOptional（aclTensor*）</td>
      <td>输入</td>
      <td>注意力汇点tensor，作为每行softmax的初始最大值。</td>
      <td>必须传入。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(N1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>metadataOptional（aclTensor*）</td>
      <td>输入</td>
      <td>AICPU算子aclnnQuantSparseFlashMlaMetadata的分核结果。</td>
      <td>必须传入。由aclnnQuantSparseFlashMlaMetadata算子生成。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>表示量化模式。</td>
      <td>量化模式1表示Q、K、V 为per-token量化，数据类型为HIFLOAT8。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxScale（double）</td>
      <td>输入</td>
      <td>缩放系数，对应公式中的softmaxScale。</td>
      <td>建议值为1/√D，其中D为每个注意力头的维度。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>cmpKv相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>SWA/ORI_SPARSE场景支持1，CSA/HCA/ORI_CMP_SPARSE场景支持1到128。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和oriKv计算的mask模式。</td>
      <td>支持：<br/>0: No Mask。<br/>3: RightDownCausal模式。<br/>4: Band模式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和cmpKv计算的mask模式。</td>
      <td>支持：<br/>0: No Mask。<br/>3: RightDownCausal模式。。SWA/ORI_SPARSE场景下该参数不生效。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft（int64_t）</td>
      <td>输入</td>
      <td>q和oriKv计算中，在因果边界基础上向左多看的token数。</td>
      <td>支持-1或非负数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight（int64_t）</td>
      <td>输入</td>
      <td>q和oriKv计算中，在因果边界基础上向右多看的token数。</td>
      <td>支持-1或非负数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional（char*）</td>
      <td>输入</td>
      <td>标识输入q的数据排布格式。</td>
      <td>支持"BSND"和"TND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional（char*）</td>
      <td>输入</td>
      <td>标识输入oriKvOptional和cmpKvOptional的数据排布格式。</td>
      <td>支持"PA_BBND"、"BSND"和"TND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkValueMode（int64_t）</td>
      <td>输入</td>
      <td>topk索引取值模式。</td>
      <td>当前支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse（bool）</td>
      <td>输入</td>
      <td>是否返回softmaxLse。</td>
      <td>支持true或false。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOutOut（aclTensor*）</td>
      <td>输出</td>
      <td>注意力计算输出。</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>与q的shape一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>softmaxLseOutOptional（aclTensor*）</td>
      <td>输出</td>
      <td>softmax的log-sum-exp结果。</td>
      <td>returnSoftmaxLse为false时返回占位Tensor；returnSoftmaxLse为true时返回softmax的log-sum-exp结果。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layoutQ为BSND时：(B, N2, S1, N1/N2)</li>
          <li>layoutQ为TND时：(N2, T1, N1/N2)</li>
          <li>returnSoftmaxLse为false时：占位Tensor</li>
        </ul>
      </td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入参数是必选输入、输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>输入变量的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>N1不在1到128范围内，或N2不为1。</td>
    </tr>
    <tr>
      <td>D不为512。</td>
    </tr>
    <tr>
      <td>quantMode不为1。</td>
    </tr>
    <tr>
      <td>oriMaskMode不为0/3/4，或cmpMaskMode不为0/3。</td>
    </tr>
    <tr>
      <td>oriWinLeft和oriWinRight不在-1或非负数范围内。</td>
    </tr>
    <tr>
      <td>SWA/ORI_SPARSE场景cmpRatio不为1，或cmpRatio在CSA/HCA/ORI_CMP_SPARSE场景下范围不在1到128内。</td>
    </tr>
    <tr>
      <td>layoutQOptional、layoutKvOptional、topkValueMode、cmpSparseIndicesOptional、metadataOptional、sinksOptional、cuSeqlens或seqused相关参数规格不在支持范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnQuantSparseFlashMla

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantSparseFlashMlaGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算

  - aclnnQuantSparseFlashMla默认采用确定性实现，相同输入多次调用结果一致。

- 使用约束

  - 本算子仅支持推理场景。
  - 已传入Tensor不支持为空。
  - `metadataOptional`参数必须传入，由`aclnnQuantSparseFlashMlaMetadata`算子生成，shape固定为(1024,)。
  - `cmpResidualKvOptional`为主算子和`aclnnQuantSparseFlashMlaMetadata`的可选入参；传入后用于按`cmp_len * cmpRatio + residual`恢复cmp侧mask使用的压缩前长度。
  - `oriMaskMode`支持0/3/4，`cmpMaskMode`支持0/3，`oriWinLeft`和`oriWinRight`支持1到128，Q\_N支持1到128。

- 三种Attention场景输入要求

  | 场景 | oriKvOptional | cmpKvOptional | oriSparseIndicesOptional | cmpSparseIndicesOptional | 说明 |
  | :--- | :----- | :----- | :----------------- | :--- |
  | SWA   | 必须传入 | 不传入 | 不传入 | 不传入 | 仅滑动窗口注意力 |
  | CSA   | 必须传入 | 必须传入 | 不传入 | 必须传入 | 滑动窗口 + TopK稀疏压缩KV |
  | HCA | 必须传入 | 必须传入 | 不传入 | 不传入 | 滑动窗口 + 稠密压缩KV |
  | ORI_SPARSE   | 必须传入 | 不传入 | 必须传入 | 不传入 | 仅TopK稀疏原始KV |
  | ORI_CMP_SPARSE   | 必须传入 | 必须传入 | 必须传入 | 必须传入 | TopK稀疏原始KV + TopK稀疏压缩KV |

- `cmpRatio`约束：SWA/ORI_SPARSE场景仅支持1，CSA/HCA/ORI_CMP_SPARSE场景仅支持1到128。

- Layout约束

  - `layoutQOptional`和`layoutKvOptional`组合仅支持"BSND"/"BSND"、"TND"/"TND"、"BSND"/"PA_BBND"、"TND"/"PA_BBND"；非PA_BBND场景下`layoutQOptional`和`layoutKvOptional`必须一致。
  - 当`layoutQOptional`为TND时，`cuSeqlensQOptional`必须传入。
  - 当`layoutKvOptional`为PA_BBND时，`sequsedOriKvOptional`必须传入，`oriBlockTableOptional`必须传入。BSND场景可选传入`sequsedOriKvOptional`覆盖每个batch的oriKv有效长度；TND场景使用`cuSeqlensOriKvOptional`表达oriKv序列边界。
  - 当`layoutKvOptional`为TND时，`cuSeqlensOriKvOptional`必须传入。
  - 当`layoutKvOptional`为TND且存在`cmpKvOptional`时，`cuSeqlensCmpKvOptional`必须传入。
  - `sequsedCmpKvOptional`为所有layoutKvOptional下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_sparse_flash_mla.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_sparse_flash_mla.h"
#include "aclnnop/aclnn_mixed_quant_sparse_flash_mla_metadata.h"

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

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

uint16_t FloatToBf16(float f)
{
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t roundingBias = 0x7fffu + lsb;
  bits += roundingBias;
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t h)
{
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
  auto size = GetShapeSize(shape);
  std::vector<uint16_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size && i < 10; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Bf16ToFloat(resultData[i]));
  }
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

std::vector<uint16_t> MakeBf16Data(int64_t size, float value)
{
  std::vector<uint16_t> data(static_cast<size_t>(size), FloatToBf16(value));
  return data;
}

std::vector<uint8_t> MakeHif8Data(int64_t size)
{
  std::vector<uint8_t> data(static_cast<size_t>(size), 0x08);
  return data;
}

}  // namespace

int main()
{
  int32_t deviceId = 0;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int64_t B = 1;
  int64_t S1 = 1;
  int64_t S2 = 1024;
  int64_t N1 = 64;
  int64_t N2 = 1;
  int64_t D = 512;
  int64_t K = 512;
  int64_t oriBlockSize = 128;
  int64_t cmpBlockSize = 128;
  int64_t s2Act = 1024;
  int64_t cmpRatio = 4;
  int64_t oriWinLeft = 127;
  int64_t oriWinRight = 0;
  int64_t oriMaskMode = 4;
  int64_t cmpMaskMode = 3;
  int64_t quantMode = 1;
  int64_t ropeHeadDim = 64;
  double softmaxScale = 1.0 / sqrt(static_cast<double>(D));

  int64_t kvD = D;

  int64_t T1 = B * S1;
  int64_t cmpKvLen = s2Act / cmpRatio;
  int64_t oriBlockNum = ((s2Act + oriBlockSize - 1) / oriBlockSize) * B;
  int64_t cmpBlockNum = ((cmpKvLen + cmpBlockSize - 1) / cmpBlockSize) * B;

  std::vector<int64_t> qShape = {T1, N1, D};
  std::vector<int64_t> oriKvShape = {oriBlockNum, oriBlockSize, N2, kvD};
  std::vector<int64_t> cmpKvShape = {cmpBlockNum, cmpBlockSize, N2, kvD};
  std::vector<int64_t> cmpSparseIndicesShape = {T1, N2, K};
  std::vector<int64_t> oriBlockTableShape = {B, (s2Act + oriBlockSize - 1) / oriBlockSize};
  std::vector<int64_t> cmpBlockTableShape = {B, (cmpKvLen + cmpBlockSize - 1) / cmpBlockSize};
  std::vector<int64_t> cuSeqLensQShape = {B + 1};
  std::vector<int64_t> seqUsedOriKvShape = {B};
  std::vector<int64_t> seqUsedCmpKvShape = {B};
  std::vector<int64_t> cmpResidualKvShape = {B};
  std::vector<int64_t> sinksShape = {N1};
  std::vector<int64_t> metadataShape = {1024};
  std::vector<int64_t> descaleShape = {1};
  std::vector<int64_t> attnOutShape = {T1, N1, D};
  std::vector<int64_t> softmaxLseShape = {T1, N1, 1};
  std::vector<int64_t> emptyShape = {0};

  void* qDeviceAddr = nullptr;
  void* oriKvDeviceAddr = nullptr;
  void* cmpKvDeviceAddr = nullptr;
  void* qDescaleDeviceAddr = nullptr;
  void* oriKvDescaleDeviceAddr = nullptr;
  void* cmpKvDescaleDeviceAddr = nullptr;
  void* cmpSparseIndicesDeviceAddr = nullptr;
  void* oriBlockTableDeviceAddr = nullptr;
  void* cmpBlockTableDeviceAddr = nullptr;
  void* cuSeqLensQDeviceAddr = nullptr;
  void* cuSeqLensOriKvDeviceAddr = nullptr;
  void* cuSeqLensCmpKvDeviceAddr = nullptr;
  void* seqUsedQDeviceAddr = nullptr;
  void* seqUsedOriKvDeviceAddr = nullptr;
  void* seqUsedCmpKvDeviceAddr = nullptr;
  void* cmpResidualKvDeviceAddr = nullptr;
  void* sinksDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* attnOutDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* oriKv = nullptr;
  aclTensor* cmpKv = nullptr;
  aclTensor* qDescale = nullptr;
  aclTensor* oriKvDescale = nullptr;
  aclTensor* cmpKvDescale = nullptr;
  aclTensor* cmpSparseIndices = nullptr;
  aclTensor* oriBlockTable = nullptr;
  aclTensor* cmpBlockTable = nullptr;
  aclTensor* cuSeqLensQ = nullptr;
  aclTensor* cuSeqLensOriKv = nullptr;
  aclTensor* cuSeqLensCmpKv = nullptr;
  aclTensor* seqUsedQ = nullptr;
  aclTensor* seqUsedOriKv = nullptr;
  aclTensor* seqUsedCmpKv = nullptr;
  aclTensor* cmpResidualKv = nullptr;
  aclTensor* sinks = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* attnOut = nullptr;
  aclTensor* softmaxLse = nullptr;

  int64_t qSize = GetShapeSize(qShape);
  int64_t oriKvSize = GetShapeSize(oriKvShape);
  int64_t cmpKvSize = GetShapeSize(cmpKvShape);
  int64_t cmpSparseIndicesSize = GetShapeSize(cmpSparseIndicesShape);
  int64_t oriBlockTableSize = GetShapeSize(oriBlockTableShape);
  int64_t cmpBlockTableSize = GetShapeSize(cmpBlockTableShape);
  int64_t attnOutSize = GetShapeSize(attnOutShape);
  int64_t softmaxLseSize = GetShapeSize(softmaxLseShape);

  std::vector<uint8_t> qHostData = MakeHif8Data(qSize);
  std::vector<uint8_t> oriKvHostData = MakeHif8Data(oriKvSize);
  std::vector<uint8_t> cmpKvHostData = MakeHif8Data(cmpKvSize);
  std::vector<float> qDescaleHostData = {1.0f};
  std::vector<float> oriKvDescaleHostData = {1.0f};
  std::vector<float> cmpKvDescaleHostData = {1.0f};
  std::vector<int32_t> cmpSparseIndicesHostData(cmpSparseIndicesSize);
  std::vector<int32_t> oriBlockTableHostData(oriBlockTableSize);
  std::iota(oriBlockTableHostData.begin(), oriBlockTableHostData.end(), 0);
  std::vector<int32_t> cmpBlockTableHostData(cmpBlockTableSize);
  std::iota(cmpBlockTableHostData.begin(), cmpBlockTableHostData.end(), 0);
  std::vector<int32_t> cuSeqLensQHostData(B + 1);
  for (int64_t i = 0; i <= B; i++) {
    cuSeqLensQHostData[i] = static_cast<int32_t>(i * S1);
  }
  std::vector<int32_t> emptyHostData;
  std::vector<int32_t> seqUsedOriKvHostData(B, static_cast<int32_t>(s2Act));
  std::vector<int32_t> seqUsedCmpKvHostData(B, static_cast<int32_t>(cmpKvLen));
  std::vector<int32_t> cmpResidualKvHostData(B);
  for (int64_t i = 0; i < B; i++) {
    cmpResidualKvHostData[i] = seqUsedOriKvHostData[i] % static_cast<int32_t>(cmpRatio);
  }
  std::vector<float> sinksHostData(N1, 1.0f);
  std::vector<int32_t> metadataHostData(1024, 0);
  std::vector<uint16_t> attnOutHostData = MakeBf16Data(attnOutSize, 0.0f);
  std::vector<float> softmaxLseHostData(softmaxLseSize, 0.0f);

  std::mt19937 gen(42);
  for (int64_t t = 0; t < T1; t++) {
    for (int64_t n = 0; n < N2; n++) {
      for (int64_t k = 0; k < K; k++) {
        cmpSparseIndicesHostData[t * N2 * K + n * K + k] = static_cast<int32_t>(gen() % cmpKvLen);
      }
    }
  }

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_HIFLOAT8, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_HIFLOAT8, &oriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_HIFLOAT8, &cmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qDescaleHostData, descaleShape, &qDescaleDeviceAddr, aclDataType::ACL_FLOAT, &qDescale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriKvDescaleHostData, descaleShape, &oriKvDescaleDeviceAddr, aclDataType::ACL_FLOAT, &oriKvDescale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpKvDescaleHostData, descaleShape, &cmpKvDescaleDeviceAddr, aclDataType::ACL_FLOAT, &cmpKvDescale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpSparseIndicesHostData, cmpSparseIndicesShape, &cmpSparseIndicesDeviceAddr,
                        aclDataType::ACL_INT32, &cmpSparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriBlockTableHostData, oriBlockTableShape, &oriBlockTableDeviceAddr, aclDataType::ACL_INT32,
                        &oriBlockTable);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpBlockTableHostData, cmpBlockTableShape, &cmpBlockTableDeviceAddr, aclDataType::ACL_INT32,
                        &cmpBlockTable);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensQShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensOriKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensCmpKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &seqUsedQDeviceAddr, aclDataType::ACL_INT32, &seqUsedQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqUsedOriKvHostData, seqUsedOriKvShape, &seqUsedOriKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqUsedCmpKvHostData, seqUsedCmpKvShape, &seqUsedCmpKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpResidualKvHostData, cmpResidualKvShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32, &cmpResidualKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  char layoutQ[] = "TND";
  char layoutKv[] = "PA_BBND";

  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;

  ret = aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv,
      seqUsedQ, seqUsedOriKv, seqUsedCmpKv,
      cmpResidualKv, nullptr, nullptr,
      N1, N2, D, quantMode, B,
      S1, S2, cmpKvLen,
      0, K, ropeHeadDim,
      cmpRatio, oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      true, true,
      metadata,
      &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMixedQuantSparseFlashMlaMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadata failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnQuantSparseFlashMlaGetWorkspaceSize(
      q, oriKv, cmpKv,
      qDescale, oriKvDescale, cmpKvDescale,
      nullptr, cmpSparseIndices,
      oriBlockTable, cmpBlockTable,
      cuSeqLensQ, nullptr, nullptr,
      nullptr, seqUsedOriKv, seqUsedCmpKv,
      cmpResidualKv, nullptr, nullptr,
      sinks, metadata,
      quantMode,
      softmaxScale, cmpRatio,
      oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      1, false,
      attnOut, softmaxLse,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantSparseFlashMlaGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnQuantSparseFlashMla(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantSparseFlashMla failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  PrintOutResult(attnOutShape, &attnOutDeviceAddr);

  aclDestroyTensor(q);
  aclDestroyTensor(oriKv);
  aclDestroyTensor(cmpKv);
  aclDestroyTensor(qDescale);
  aclDestroyTensor(oriKvDescale);
  aclDestroyTensor(cmpKvDescale);
  aclDestroyTensor(cmpSparseIndices);
  aclDestroyTensor(oriBlockTable);
  aclDestroyTensor(cmpBlockTable);
  aclDestroyTensor(cuSeqLensQ);
  aclDestroyTensor(cuSeqLensOriKv);
  aclDestroyTensor(cuSeqLensCmpKv);
  aclDestroyTensor(seqUsedQ);
  aclDestroyTensor(seqUsedOriKv);
  aclDestroyTensor(seqUsedCmpKv);
  aclDestroyTensor(cmpResidualKv);
  aclDestroyTensor(sinks);
  aclDestroyTensor(metadata);
  aclDestroyTensor(attnOut);
  aclDestroyTensor(softmaxLse);

  aclrtFree(qDeviceAddr);
  aclrtFree(oriKvDeviceAddr);
  aclrtFree(cmpKvDeviceAddr);
  aclrtFree(qDescaleDeviceAddr);
  aclrtFree(oriKvDescaleDeviceAddr);
  aclrtFree(cmpKvDescaleDeviceAddr);
  aclrtFree(cmpSparseIndicesDeviceAddr);
  aclrtFree(oriBlockTableDeviceAddr);
  aclrtFree(cmpBlockTableDeviceAddr);
  if (cuSeqLensQDeviceAddr != nullptr) {
    aclrtFree(cuSeqLensQDeviceAddr);
  }
  if (seqUsedOriKvDeviceAddr != nullptr) {
    aclrtFree(seqUsedOriKvDeviceAddr);
  }
  if (seqUsedCmpKvDeviceAddr != nullptr) {
    aclrtFree(seqUsedCmpKvDeviceAddr);
  }
  if (cmpResidualKvDeviceAddr != nullptr) {
    aclrtFree(cmpResidualKvDeviceAddr);
  }
  aclrtFree(sinksDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(attnOutDeviceAddr);
  aclrtFree(softmaxLseDeviceAddr);
  if (metadataWorkspaceSize > 0) {
    aclrtFree(metadataWorkspaceAddr);
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
