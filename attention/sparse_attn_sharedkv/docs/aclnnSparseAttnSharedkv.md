# aclnnSparseAttnSharedkv

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- 接口功能：`SparseAttnSharedkv`算子实现基于共享KV（Key=Value）的稀疏注意力计算，支持三种模板模式：Sliding Window Attention（SWA）、Compressed Flash Attention（CFA）和Sparse Compressed Flash Attention（SCFA）。该算子适用于大语言模型推理场景，通过滑动窗口和KV压缩机制大幅降低长序列注意力计算的开销。

- 计算公式：

  $$
  O = \text{softmax}(Q \cdot \tilde{K}^T \cdot \text{softmax\_scale}) \cdot \tilde{V}
  $$

  其中$\tilde{K} = \tilde{V}$（共享KV），$\tilde{K}$由滑动窗口内的原始KV和因果边界内的压缩KV拼接而成，具体参与计算的KV范围由模板模式和mask参数决定：

  - 滑动窗口部分（ori_kv）：对第$i_{S1}$个Query token，其因果对角线位置为$\text{ori\_threshold} = S2_{act} - S1_{act} + i_{S1} + 1$，窗口范围为$[\max(\text{ori\_threshold} - \text{ori\_win\_left} - 1, 0), \text{ori\_threshold} + \text{ori\_win\_right})$。

  - 压缩KV部分（cmp_kv）：因果边界阈值为$\text{cmp\_threshold} = \lfloor \frac{\text{ori\_threshold}}{\text{cmp\_ratio}} \rfloor$。CFA模式下取$[0, \text{cmp\_threshold})$内的连续压缩KV；SCFA模式下通过TopK索引从压缩KV中按需收集，仅保留$\text{begin\_idx} < \text{cmp\_threshold}$的块。

  注意力计算采用Online Softmax（Flash Attention V2），S2方向按512分块循环，sinks作为每行softmax的初始最大值：

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
  | $Q$                 | Query输入，形状为$[G, D]$（单行）                         |
  | $K_{tile}^{(t)}$    | 第$t$个S2分块的KV数据，$K=V$（共享KV）                    |
  | $S^{(t)}$           | 第$t$个分块的QK缩放注意力分数                             |
  | $P^{(t)}$           | 第$t$个分块的softmax概率                                  |
  | $O^{(t)}$           | 第$t$个分块后的累加输出                                   |
  | $\text{softmax\_scale}$ | 缩放系数，通常为$1/\sqrt{D}$                          |
  | $B$                 | Batch Size                                                |
  | $S1$/$S1_{act}$     | Query序列长度/实际有效长度                                 |
  | $S2$/$S2_{act}$     | 原始KV序列长度/实际有效长度                                |
  | $N1$                | Query头数                                                 |
  | $N2$                | KV头数                                                    |
  | $G$                 | GQA分组比，$G = N1/N2$                                    |
  | $D$                 | 每个注意力头的维度                                        |
  | $\text{sinks}$      | 注意力汇点，形状为$[N1]$                                  |
  | $\text{cmp\_ratio}$ | 压缩率，原始KV长度与压缩KV长度的比值                      |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnSparseAttnSharedkvGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnSparseAttnSharedkv`执行实际计算。

```c++
aclnnStatus aclnnSparseAttnSharedkvGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional,
    const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional,
    const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensOriKvOptional,
    const aclTensor *cuSeqLensCmpKvOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKvOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    double           softmaxScale,
    int64_t          cmpRatio,
    int64_t          oriMaskMode,
    int64_t          cmpMaskMode,
    int64_t          oriKvStride,
    int64_t          cmpKvStride,
    int64_t          oriWinLeft,
    int64_t          oriWinRight,
    char            *layoutQ,
    char            *layoutKv,
    bool             returnSoftmaxLse,
    const aclTensor *attnOut,
    const aclTensor *softmaxLse,
    uint64_t       **workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnSparseAttnSharedkv(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnSparseAttnSharedkvGetWorkspaceSize

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
      <td>不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layout_q为BSND时：(B, S1, N1, D)</li>
          <li>layout_q为TND时：(T1, N1, D)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>原始KV输入张量，Key与Value共享同一份数据。</td>
      <td>SWA/CFA/SCFA模式必须传入。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layout_kv为PA_ND时：(ori_block_num, ori_block_size, N2, D)</li>
          <li>layout_kv为BSND时：(B, S2, N2, D)</li>
          <li>layout_kv为TND时：(T2, N2, D)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>压缩KV输入张量，Key与Value共享同一份数据。</td>
      <td>CFA/SCFA模式必须传入，SWA模式不传入。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layout_kv为PA_ND时：(cmp_block_num, cmp_block_size, N2, D)</li>
          <li>layout_kv为BSND时：(B, S3, N2, D)</li>
          <li>layout_kv为TND时：(T3, N2, D)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriSparseIndicesOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>代表离散取oriKvCache的索引。</td>
      <td>当前暂不支持，必须传入nullptr。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpSparseIndicesOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>代表离散取cmpKvCache的TopK索引。</td>
      <td>SCFA模式必须传入，SWA/CFA模式不传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
        <ul>
          <li>layout_q为BSND时：(B, S1, N2, K)</li>
          <li>layout_q为TND时：(T1, N2, K)</li>
        </ul>
        其中K为TopK稀疏选择数，仅支持512。
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriBlockTableOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>PageAttention中oriKvCache存储使用的block映射表。</td>
      <td>layout_kv为PA_ND时必须传入。第二维长度不小于所有batch中最大的S2对应的block数量。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, ori_max_block_num_per_batch)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpBlockTableOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>PageAttention中cmpKvCache存储使用的block映射表。</td>
      <td>CFA/SCFA模式且layout_kv为PA_ND时必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, cmp_max_block_num_per_batch)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqLensQOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中q的有效token数（前缀和形式）。</td>
      <td>layout_q为TND时必须传入。每个元素表示当前batch与之前所有batch的token数总和。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqLensOriKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv的有效token数（前缀和形式）。</td>
      <td>layout_kv为TND时必须传入。当前layout_kv仅支持PA_ND，故设置此参数无效。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqLensCmpKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中cmp_kv的有效token数（前缀和形式）。</td>
      <td>layout_kv为TND且存在cmp_kv时必须传入。当前layout_kv仅支持PA_ND，故设置此参数无效。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedQOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中q实际参与运算的token数。</td>
      <td>当前暂不支持指定该参数。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv实际参与运算的token数。</td>
      <td>layout_kv为PA_ND时必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sinksOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>注意力汇点tensor，作为每行softmax的初始最大值。</td>
      <td>必须传入。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(N1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>metadataOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>AICPU算子SparseAttnSharedkvMetadata的分核结果。</td>
      <td>必须传入。由aclnnSparseAttnSharedkvMetadata算子生成。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxScale（double）</td>
      <td>输入</td>
      <td>缩放系数，对应公式中的softmax_scale。</td>
      <td>建议值为$1/\sqrt{D}$。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>对ori_kv的压缩率。</td>
      <td>仅支持4或128。SWA模式传入1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和ori_kv计算的mask模式。</td>
      <td>仅支持4（Band模式）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode（int64_t）</td>
      <td>输入</td>
      <td>q和cmp_kv计算的mask模式。</td>
      <td>仅支持3（RightDownCausal模式）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriKvStride（int64_t）</td>
      <td>输入</td>
      <td>ori_kv block在元素级别的步长。</td>
      <td>默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpKvStride（int64_t）</td>
      <td>输入</td>
      <td>cmp_kv block在元素级别的步长。</td>
      <td>默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft（int64_t）</td>
      <td>输入</td>
      <td>q和ori_kv计算中，在因果边界基础上向左多看的token数。</td>
      <td>仅支持127。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight（int64_t）</td>
      <td>输入</td>
      <td>q和ori_kv计算中，在因果边界基础上向右多看的token数。</td>
      <td>仅支持0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ（char*）</td>
      <td>输入</td>
      <td>标识输入q的数据排布格式。</td>
      <td>支持"BSND"和"TND"，默认值为"BSND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv（char*）</td>
      <td>输入</td>
      <td>标识输入ori_kv和cmp_kv的数据排布格式。</td>
      <td>支持"PA_ND"、"BSND"和"TND"，默认值为"PA_ND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse（bool）</td>
      <td>输入</td>
      <td>是否返回softmax_lse。</td>
      <td>当前仅支持False。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOut（aclTensor*）</td>
      <td>输出</td>
      <td>注意力计算输出。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>与q的shape一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>softmaxLse（aclTensor*）</td>
      <td>输出</td>
      <td>softmax的log-sum-exp结果。</td>
      <td>当前为无效值。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>与q的shape一致（最后一维为1）</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t**）</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>输入变量的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>N1不为64。</td>
    </tr>
    <tr>
      <td>N2不为1。</td>
    </tr>
    <tr>
      <td>D不为512。</td>
    </tr>
    <tr>
      <td>ori_mask_mode不为4，或cmp_mask_mode不为3。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>调用NPU Runtime接口申请内存/创建Tensor失败。</td>
    </tr>
  </tbody>
  </table>

## aclnnSparseAttnSharedkv

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseAttnSharedkvGetWorkspaceSize获取。</td>
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

  - aclnnSparseAttnSharedkv默认采用确定性实现，相同输入多次调用结果一致。

- 规格约束

  | 规格项        | 规格         | 规格说明                                           |
  | :------------ | :----------- | :------------------------------------------------- |
  | N1            | 64           | Query头数仅支持64。                                 |
  | N2            | 1            | KV头数仅支持1（共享KV + GQA）。                     |
  | D             | 512          | 每个注意力头的维度仅支持512。                        |
  | K（TopK）     | 512          | SCFA模式下TopK稀疏选择数仅支持512。                  |
  | cmp_ratio     | 4 / 128      | 压缩率仅支持4或128。                                 |
  | ori_mask_mode | 4            | 原始KV掩码模式仅支持4（Band模式）。                   |
  | cmp_mask_mode | 3            | 压缩KV掩码模式仅支持3（RightDownCausal模式）。        |
  | ori_win_left  | 127          | 滑动窗口向左扩展仅支持127。                           |
  | ori_win_right | 0            | 滑动窗口向右扩展仅支持0。                             |
  | block_size    | 16的倍数，≤1024 | PageAttention的block_size取值范围。               |

- 使用约束

  - 该接口支持推理场景下使用。
  - 该接口支持aclgraph模式。
  - 当前暂不支持返回`softmax_lse`，`return_softmax_lse`仅支持输入False。
  - 当前暂不支持指定`seqused_q`，设置该参数无效。
  - 当前暂不支持对`ori_kv`进行稀疏计算，设置`ori_sparse_indices`无效。
  - 当前所有输入不支持传入空Tensor。
  - `metadata`参数必须传入，由`aclnnSparseAttnSharedkvMetadata`算子生成，shape固定为(1024,)。
  - `sinks`参数必须传入，shape为(N1,)。

- 三种模板模式输入要求

  | 模式 | ori_kv | cmp_kv | cmp_sparse_indices | 说明 |
  | :--- | :----- | :----- | :----------------- | :--- |
  | SWA  | 必须传入 | 不传入 | 不传入 | 仅滑动窗口注意力 |
  | CFA  | 必须传入 | 必须传入 | 不传入 | 滑动窗口 + 均匀压缩KV |
  | SCFA | 必须传入 | 必须传入 | 必须传入 | 滑动窗口 + TopK稀疏压缩KV |

- Layout约束

  - 当`layout_q`为TND时，`cu_seqlens_q`必须传入。
  - 当`layout_kv`为PA_ND时，`seqused_kv`必须传入，`ori_block_table`必须传入。
  - 当`layout_kv`为TND时，`cu_seqlens_ori_kv`必须传入。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_attn_sharedkv.h"
#include "aclnnop/aclnn_sparse_attn_sharedkv_metadata.h"

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
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= dim;
  }
  return size;
}

int main() {
  // 1. 初始化AscendCL环境（略，参考aclrt相关接口）

  // 2. 构造参数
  int64_t B = 4, S1 = 128, S2 = 8192;
  int64_t N1 = 64, N2 = 1, D = 512, K = 512;
  int64_t ori_block_size = 128, cmp_block_size = 128;
  int64_t s2_act = 4096, cmp_ratio = 4;
  int64_t ori_win_left = 127, ori_win_right = 0;
  double softmax_scale = 1.0 / sqrt((double)D);
  int64_t ori_mask_mode = 4, cmp_mask_mode = 3;

  // 3. 准备输入Tensor（略，参考aclCreateTensor接口）
  // q: [T1, N1, D] = [B*S1, N1, D], BFLOAT16
  // ori_kv: [ori_block_num, ori_block_size, N2, D], BFLOAT16
  // cmp_kv: [cmp_block_num, cmp_block_size, N2, D], BFLOAT16
  // cmp_sparse_indices: [T1, N2, K], INT32
  // ori_block_table: [B, ori_max_block_num], INT32
  // cmp_block_table: [B, cmp_max_block_num], INT32
  // cu_seqlens_q: [B+1], INT32
  // seqused_kv: [B], INT32
  // sinks: [N1], FLOAT32

  // 4. 调用aclnnSparseAttnSharedkvMetadata生成分核元数据
  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor *metadataExecutor = nullptr;
  aclnnSparseAttnSharedkvMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv,
      seqUsedQ, seqUsedKv,
      N1, N2, D, B, S1, S2,
      0, K, cmp_ratio,
      ori_mask_mode, cmp_mask_mode,
      ori_win_left, ori_win_right,
      "TND", "PA_ND",
      true, true,
      metadataTensor,
      &metadataWorkspaceSize, &metadataExecutor);
  // 申请workspace并执行（略）

  // 5. 调用aclnnSparseAttnSharedkvGetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnSparseAttnSharedkvGetWorkspaceSize(
      q, oriKv, cmpKv,
      nullptr, cmpSparseIndices,
      oriBlockTable, cmpBlockTable,
      cuSeqLensQ, nullptr, nullptr,
      nullptr, seqUsedKv,
      sinks, metadataTensor,
      softmax_scale, cmp_ratio,
      ori_mask_mode, cmp_mask_mode,
      0, 0,
      ori_win_left, ori_win_right,
      "TND", "PA_ND",
      false,
      attnOut, softmaxLse,
      &workspaceSize, &executor);

  // 6. 申请workspace并执行
  void *workspace = nullptr;
  aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnSparseAttnSharedkv(workspace, workspaceSize, executor, stream);

  // 7. 释放资源（略）
  return 0;
}
```
