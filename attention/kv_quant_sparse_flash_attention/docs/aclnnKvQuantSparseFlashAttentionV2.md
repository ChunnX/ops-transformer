# aclnnKvQuantSparseFlashAttentionV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/kv_quant_sparse_flash_attention)

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

- 接口功能：kv_quant_sparse_flash_attention（QSFA）在sparse_flash_attention的基础上支持了[Per-Token-Head-Tile-128量化]输入，是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。V2版本新增可学习的Attention Sink前反向，提升模型表达能力。

- 计算公式：

    $$
    Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V})
    $$
- 增加 **sink** 后计算公式：
    $$
    S=Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T
    $$
    $$
    m= max(sink, max(\frac{S}{\sqrt{d_k}}))
    $$
    $$
    Attention = \frac{e^{S - m} @\text{Dequant}(\tilde{V}^{INT8},{Scale_V})}{\sum e^{S-m} + e^{sink - m}}
    $$

其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`LightningIndexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度，$\text{Dequant}(\cdot,\cdot)$为反量化函数。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnKvQuantSparseFlashAttentionV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnKvQuantSparseFlashAttentionV2”接口执行计算。

```Cpp
aclnnStatus aclnnKvQuantSparseFlashAttentionV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *keyDequantScaleOptional,
    const aclTensor *valueDequantScaleOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *sinksOptional,
    double           scaleValue,
    int64_t          keyQuantMode,
    int64_t          valueQuantMode,
    int64_t          sparseBlockSize,
    char             *layoutQuery,
    char             *layoutKv,
    int64_t          sparseMode,
    int64_t          preTokens,
    int64_t          nextTokens,
    int64_t          attentionMode,
    int64_t          quantScaleRepoMode,
    int64_t          tileSize,
    int64_t          ropeHeadDim,
    const aclTensor  *attentionOut,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnKvQuantSparseFlashAttentionV2(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    const aclrtStream stream)
```

## aclnnKvQuantSparseFlashAttentionV2GetWorkspaceSize

- **参数说明：**

  > [!NOTE]  
  >
  >- query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
  >- Q_S和KV_S分别表示query和key/value的Sequence Length，Q_N和KV_N分别表示query和key/value的Head Num，Q_D和KV_D分别表示query和key/value的Head Dim，Q_T和KV_T分别表示query和key/value的Total Tokens，sparse_size表示一次离散选取的block数, block_num和block_size分别表示PageAttention场景下的block总数和每个block的token数。
  >


  <table style="undefined;table-layout: fixed; width: 1680px"><colgroup>
  <col style="width: 250px">
  <col style="width: 90px">
  <col style="width: 320px">
  <col style="width: 500px">
  <col style="width: 180px">
  <col style="width: 80px">
  <col style="width: 260px">
  <col style="width: 120px">
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
      <td>query（aclTensor*）</td>
      <td>输入</td>
      <td>attention结构的Query输入，query由相同数据类型的q_nope和q_rope按D维度拼接得到。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>不支持空tensor。</li>
                <li>Q_D值仅支持576，即q_nope+q_rope=512+64；Q_N值支持1/2/4/8/16/32/48/64/128。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
     <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>layout_query为BSND时，shape为(B, Q_S, Q_N, Q_D)。</li>
                <li>layout_query为TND时，shape为(Q_T, Q_N, Q_D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key（aclTensor*）</td>
      <td>输入</td>
      <td>attention结构的Key输入，由k_nope、query相同数据类型的k_rope和float32的量化参数按D维度拼接得到。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的token数，block_size取值为16的整数倍，最大支持到1024。</li>
                <li>KV_N仅支持1，KV_D值仅支持656，即nope+rope*2+dequant_scale*4=512+64*2+4*4。</li>
          </ul>
      </td>
      <td>FLOAT8_E4M3、INT8、HIFLOAT8</td>
      <td>ND</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>layout_kv为PA_BSND时，shape为(block_num, block_size, KV_N, KV_D)。</li>
                <li>layout_kv为BSND时，shape为(B, KV_S, KV_N, KV_D)。</li>
                <li>layout_kv为TND时，shape为(KV_T, KV_N, KV_D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>value（aclTensor*）</td>
      <td>输入</td>
      <td>attention结构的Value输入。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT8_E4M3、INT8、HIFLOAT8</td>
      <td>ND</td>
      <td>shape与key的shape一致。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseIndices（aclTensor*）</td>
      <td>输入</td>
      <td>离散取kvCache的索引。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>不支持空tensor。</li>
                <li>sparse_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse_size大于0。</li>
                <li>当key和value的数据类型为hifloat8时，sparse_size仅支持2048。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>layout_query为BSND时，shape为(B, Q_S, KV_N, sparse_size)。</li>
                <li>layout_query为TND时，shape为(Q_T, KV_N, sparse_size)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyDequantScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>预留参数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valueDequantScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>预留参数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示PageAttention中kvCache存储使用的block映射表。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>不支持空tensor。</li>
                <li>第二维长度需大于0且不小于所有batch中最大的KV_S对应的block数量，即KV_S_max / block_size向上取整。</li>
                <li>layout_kv为PA_BSND时，blockTable必传。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, KV_S_max/block_size)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQueryOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中query的有效token数。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和query shape的Q_S长度相同。</li>
                <li>每个Batch的有效token数不超过query中的维度S大小且不小于0。</li>
                <li>layout_query为TND时，该入参必须传入，且以该入参元素的数量作为B值，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中key和value的有效token数。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和key shape的KV_S长度相同。</li>
                <li>每个Batch的有效token数不超过key/value中的维度S大小且不小于0。</li>
                <li>当layout_kv为TND或PA_BSND时，该入参必须传入。</li>
                <li>layout_kv为TND时，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sinksOptional（aclTensor*）</td>
      <td>输入</td>
      <td>用于初始化softmax max/sum，保证数值稳定性。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(Q_N,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleValue（double）</td>
      <td>输入</td>
      <td>代表缩放系数，公式中d<sub>k</sub>开根号的倒数，作为query和key矩阵乘后Muls的scalar值。</td>
      <td>-</td>
      <td>FLOAT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
    <tr>
      <td>keyQuantMode（int64_t）</td>
      <td>输入</td>
      <td>代表key的量化模式。</td>
      <td>仅支持传入2，表示per_tile量化模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valueQuantMode（int64_t）</td>
      <td>输入</td>
      <td>代表value·的量化模式。</td>
      <td>仅支持传入2，表示per_tile量化模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseBlockSize（int64_t）</td>
      <td>输入</td>
      <td>代表sparse阶段的block大小。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>sparse_block_size为1时，为Token-wise稀疏化场景，将每个token视为独立单元，在计算重要性分数时，评估每个查询token与每个键值token之间的独立关联程度。</li>
                <li>sparse_block_size为大于1小于等于128时，为Block-wise稀疏化场景，将token序列划分为固定大小的连续块，以块为单位进行重要性评估，块内token共享相同的稀疏化决策。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQuery（char*）</td>
      <td>输入</td>
      <td>标识输入query的数据排布格式。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>默认值BSND。</li>
                <li>支持传入BSND和TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv（char*）</td>
      <td>输入</td>
      <td>标识输入key的数据排布格式。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>默认值BSND。</li>
                <li>支持传入TND、BSND和PA_BSND，其中PA_BSND在开启PageAttention时使用。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode（int64_t）</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>sparse_mode为0时，代表全部计算。</li>
                <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens（int64_t）</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens（int64_t）</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionMode（int64_t）</td>
      <td>输入</td>
      <td>表示attention的模式。</td>
      <td>仅支持默认值2，表示MLA-absorb模式，即QK的D包含rope和nope两部分，且KV是同一份。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantScaleRepoMode（int64_t）</td>
      <td>输入</td>
      <td>表示量化参数的存放模式。</td>
      <td>仅支持默认值1，表示combine模式，即量化参数和数据混合存放。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSize（int64_t）</td>
      <td>输入</td>
      <td>表示per_tile时每个参数对应的数据块大小。</td>
      <td>仅在per_tile时有效，仅支持128。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ropeHeadDim（int64_t）</td>
      <td>输入</td>
      <td>表示MLA架构下的rope_head_dim大小。</td>
      <td>仅在attention_mode为2时有效，仅支持64。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut（aclTensor*）</td>
      <td>输出</td>
      <td>代表公式中的输出Attention。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul style="padding-left: 0;list-style-position: inside;">
                <li>layout_query为BSND时，shape为(B, Q_S, Q_N, Q_out_D)。</li>
                <li>layout_query为TND时shape为(Q_T, Q_N, Q_out_D)。</li>
                <li>其中Q_out_D = Q_D - rope_head_dim。</li>
          </ul>
      </td>
      <td>x</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：
  
    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 280px">
    <col style="width: 110px">
    <col style="width: 1000px">
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
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>query、key、value、sparseIndices、keyDequantScaleOptional、valueDequantScaleOptional、blockTableOptional、actualSeqLengthsQueryOptional、actualSeqLengthsKvOptional、sinksOptional、scaleValue、keyQuantMode、valueQuantMode、sparseBlockSize、layoutQuery、layoutKv、sparseMode、preTokens、nextTokens、attentionMode、quantScaleRepoMode、tileSize、ropeHeadDim的数据类型和数据格式不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnKvQuantSparseFlashAttentionV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 150px">
  <col style="width: 110px">
  <col style="width: 800px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnKvQuantSparseFlashAttentionV2GetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- Q_N支持情况：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - Q_N不支持48。
- sparse_block_size支持情况：
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 仅支持sparse_block_size为1。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 支持[1,16]，且要求是2的幂次方，在PageAttention场景下要求sparse_block_size整除block_size。
- sinks支持情况：
  - 仅支持<term>Ascend 950PR/Ascend 950DT</term>，非PageAttention场景。
- 参数key、value数据类型要求：
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 支持float8_e4m3、int8、hifloat8数据类型。
    - 其中key为hifloat8时，sparse_block_count仅支持2048。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 仅支持int8数据类型。
- 非PageAttention场景layout_query和layout_kv取值需要保持一致。
- 支持确定性计算。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_kv_quant_sparse_flash_attention_v2.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<short> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %d\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
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
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
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

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> qShape = {1, 64, 576};                // T1, N1, D
    std::vector<int64_t> kShape = {2048, 1, 656};              // T2, N2, D
    std::vector<int64_t> vShape = {2048, 1, 656};              // T2, N2, D
    std::vector<int64_t> sparseIndicesShape = {1, 1, 2048};    // T1, N2, K
    std::vector<int64_t> outShape = {1, 64, 512};             // T1, N1, D
    std::vector<int64_t> actSeqQLenshape = {1};               // B
    std::vector<int64_t> actSeqKvLenshape = {1};           // B
    std::vector<int64_t> sinksShape = {64};                   // N1

    void* qDeviceAddr = nullptr;
    void* kDeviceAddr = nullptr;
    void* vDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* actSeqQLenDeviceAddr = nullptr;
    void* actSeqKvLenDeviceAddr = nullptr;
    void* sinksDeviceAddr = nullptr;

    aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* sparseIndices = nullptr;
    aclTensor* out = nullptr;
    aclTensor* actSeqQLen = nullptr;
    aclTensor* actSeqKvLen = nullptr;
    aclTensor* sinks = nullptr;

    std::vector<op::fp16_t> qHostData(1 * 64 * 576, 1.0);
    std::vector<int8_t> kHostData(2048 * 1 * 656, 1.0);
    std::vector<int8_t> vHostData(2048 * 1 * 656, 1.0);
    std::vector<int32_t> sparseIndicesHostData(2048);
    std::iota(sparseIndicesHostData.begin(), sparseIndicesHostData.end(), 0);
    std::vector<op::fp16_t> outHostData(1 * 64 * 512, 1.0);
    std::vector<int32_t> actSeqQLenHostData(1, 1);
    std::vector<int32_t> actSeqKvLenHostData(1, 2048);
    std::vector<float> sinksHostData(64, 1.0);

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_INT8, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_INT8, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape,
                          &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqQLenHostData, actSeqQLenshape,
                          &actSeqQLenDeviceAddr, aclDataType::ACL_INT32, &actSeqQLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqKvLenHostData, actSeqKvLenshape,
                          &actSeqKvLenDeviceAddr, aclDataType::ACL_INT32, &actSeqKvLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double scaleValue = 0.0416666666666667;
    int64_t keyQuantMode = 2;
    int64_t valueQuantMode = 2;
    int64_t sparseBlockSize = 1;
    int64_t sparseMode = 0;
    int64_t attentionMode = 2;
    int64_t quantScaleRepoMode = 1;
    int64_t tileSize = 128;
    int64_t ropeHeadDim = 64;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    char layoutQuery[5] = {'T', 'N', 'D', 0};
    char layoutKey[5] = {'T', 'N', 'D', 0};

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnKvQuantSparseFlashAttentionV2第一段接口
    ret = aclnnKvQuantSparseFlashAttentionV2GetWorkspaceSize(q, k, v, sparseIndices, nullptr, nullptr, nullptr, actSeqQLen,
                                                      actSeqKvLen, sinks, scaleValue, keyQuantMode, valueQuantMode,
                                                      sparseBlockSize, layoutQuery, layoutKey, sparseMode, preTokens,
                                                      nextTokens, attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim,
                                                      out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnKvQuantSparseFlashAttentionV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnKvQuantSparseFlashAttentionV2第二段接口
    ret = aclnnKvQuantSparseFlashAttentionV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvQuantSparseFlashAttentionV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(sparseIndices);
    aclDestroyTensor(out);
    aclDestroyTensor(actSeqQLen);
    aclDestroyTensor(actSeqKvLen);
    aclDestroyTensor(sinks);

    // 6. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(sparseIndicesDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(actSeqQLenDeviceAddr);
    aclrtFree(actSeqKvLenDeviceAddr);
    aclrtFree(sinksDeviceAddr);
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
