# aclnnSparseAttnSharedkv

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
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

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

uint16_t FloatToFp16(float f)
{
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 31) & 0x1u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = (bits >> 13) & 0x3ffu;
  if (exp <= 0) {
    return static_cast<uint16_t>(sign << 15);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>((sign << 15) | 0x7c00u);
  }
  return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | mant);
}

float Fp16ToFloat(uint16_t h)
{
  uint32_t sign = (h >> 15) & 0x1u;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    f = (sign << 31) | (mant << 13);
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
  }
  float result;
  std::memcpy(&result, &f, sizeof(result));
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
    LOG_PRINT("result[%ld] is: %f\n", i, Fp16ToFloat(resultData[i]));
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

std::vector<uint16_t> MakeFp16Data(int64_t size, float value)
{
  std::vector<uint16_t> data(static_cast<size_t>(size), FloatToFp16(value));
  return data;
}

}  // namespace

int main()
{
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
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
  int64_t oriBlockSize = 128;
  int64_t cmpBlockSize = 128;
  int64_t s2Act = 4096;
  int64_t cmpRatio = 4;
  int64_t oriWinLeft = 127;
  int64_t oriWinRight = 0;
  int64_t oriMaskMode = 4;
  int64_t cmpMaskMode = 3;
  double softmaxScale = 1.0 / sqrt(static_cast<double>(D));

  int64_t T1 = B * S1;
  int64_t cmpKvLen = s2Act / cmpRatio;
  int64_t oriBlockNum = ((s2Act + oriBlockSize - 1) / oriBlockSize) * B;
  int64_t cmpBlockNum = ((cmpKvLen + cmpBlockSize - 1) / cmpBlockSize) * B;

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {T1, N1, D};
  std::vector<int64_t> oriKvShape = {oriBlockNum, oriBlockSize, N2, D};
  std::vector<int64_t> cmpKvShape = {cmpBlockNum, cmpBlockSize, N2, D};
  std::vector<int64_t> cmpSparseIndicesShape = {T1, N2, K};
  std::vector<int64_t> oriBlockTableShape = {B, (s2Act + oriBlockSize - 1) / oriBlockSize};
  std::vector<int64_t> cmpBlockTableShape = {B, (cmpKvLen + cmpBlockSize - 1) / cmpBlockSize};
  std::vector<int64_t> cuSeqLensQShape = {B + 1};
  std::vector<int64_t> seqUsedKvShape = {B};
  std::vector<int64_t> sinksShape = {N1};
  std::vector<int64_t> metadataShape = {1024};
  std::vector<int64_t> attnOutShape = {T1, N1, D};
  std::vector<int64_t> softmaxLseShape = {T1, N1, 1};
  // 对全部 5 个输入调用 Contiguous，optional 输入传 shape 为 {0} 的空 tensor。
  std::vector<int64_t> emptyShape = {0};

  void* qDeviceAddr = nullptr;
  void* oriKvDeviceAddr = nullptr;
  void* cmpKvDeviceAddr = nullptr;
  void* cmpSparseIndicesDeviceAddr = nullptr;
  void* oriBlockTableDeviceAddr = nullptr;
  void* cmpBlockTableDeviceAddr = nullptr;
  void* cuSeqLensQDeviceAddr = nullptr;
  void* cuSeqLensOriKvDeviceAddr = nullptr;
  void* cuSeqLensCmpKvDeviceAddr = nullptr;
  void* seqUsedQDeviceAddr = nullptr;
  void* seqUsedKvDeviceAddr = nullptr;
  void* sinksDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* attnOutDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* oriKv = nullptr;
  aclTensor* cmpKv = nullptr;
  aclTensor* cmpSparseIndices = nullptr;
  aclTensor* oriBlockTable = nullptr;
  aclTensor* cmpBlockTable = nullptr;
  aclTensor* cuSeqLensQ = nullptr;
  aclTensor* cuSeqLensOriKv = nullptr;
  aclTensor* cuSeqLensCmpKv = nullptr;
  aclTensor* seqUsedQ = nullptr;
  aclTensor* seqUsedKv = nullptr;
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

  std::vector<uint16_t> qHostData = MakeFp16Data(qSize, 1.0f);
  std::vector<uint16_t> oriKvHostData = MakeFp16Data(oriKvSize, 1.0f);
  std::vector<uint16_t> cmpKvHostData = MakeFp16Data(cmpKvSize, 1.0f);
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
  std::vector<int32_t> seqUsedKvHostData(B, static_cast<int32_t>(s2Act));
  std::vector<float> sinksHostData(N1, 1.0f);
  std::vector<int32_t> metadataHostData(1024, 0);
  std::vector<uint16_t> attnOutHostData = MakeFp16Data(attnOutSize, 0.0f);
  std::vector<float> softmaxLseHostData(softmaxLseSize, 0.0f);

  std::mt19937 gen(42);
  for (int64_t t = 0; t < T1; t++) {
    for (int64_t n = 0; n < N2; n++) {
      for (int64_t k = 0; k < K; k++) {
        cmpSparseIndicesHostData[t * N2 * K + n * K + k] = static_cast<int32_t>(gen() % cmpKvLen);
      }
    }
  }

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_FLOAT16, &oriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_FLOAT16, &cmpKv);
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
  ret = CreateAclTensor(seqUsedKvHostData, seqUsedKvShape, &seqUsedKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT16, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  char layoutQ[] = "TND";
  char layoutKv[] = "PA_ND";

  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  ret = aclnnSparseAttnSharedkvMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv,
      seqUsedQ, seqUsedKv,
      N1, N2, D, B, S1, S2,
      0, K, cmpRatio,
      oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      true, true,
      metadata,
      &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSparseAttnSharedkvMetadataGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSparseAttnSharedkvMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseAttnSharedkvMetadata failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnSparseAttnSharedkvGetWorkspaceSize(
      q, oriKv, cmpKv,
      nullptr, cmpSparseIndices,
      oriBlockTable, cmpBlockTable,
      cuSeqLensQ, nullptr, nullptr,
      nullptr, seqUsedKv,
      sinks, metadata,
      softmaxScale, cmpRatio,
      oriMaskMode, cmpMaskMode,
      0, 0,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      false,
      attnOut, softmaxLse,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseAttnSharedkvGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSparseAttnSharedkv(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseAttnSharedkv failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(attnOutShape, &attnOutDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(oriKv);
  aclDestroyTensor(cmpKv);
  aclDestroyTensor(cmpSparseIndices);
  aclDestroyTensor(oriBlockTable);
  aclDestroyTensor(cmpBlockTable);
  aclDestroyTensor(cuSeqLensQ);
  aclDestroyTensor(cuSeqLensOriKv);
  aclDestroyTensor(cuSeqLensCmpKv);
  aclDestroyTensor(seqUsedQ);
  aclDestroyTensor(seqUsedKv);
  aclDestroyTensor(sinks);
  aclDestroyTensor(metadata);
  aclDestroyTensor(attnOut);
  aclDestroyTensor(softmaxLse);
  
  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(oriKvDeviceAddr);
  aclrtFree(cmpKvDeviceAddr);
  aclrtFree(cmpSparseIndicesDeviceAddr);
  aclrtFree(oriBlockTableDeviceAddr);
  aclrtFree(cmpBlockTableDeviceAddr);
  if (cuSeqLensQDeviceAddr != nullptr) {
    aclrtFree(cuSeqLensQDeviceAddr);
  }
  if (seqUsedKvDeviceAddr != nullptr) {
    aclrtFree(seqUsedKvDeviceAddr);
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
