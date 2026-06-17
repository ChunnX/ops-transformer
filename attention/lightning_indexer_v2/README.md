# LightningIndexerV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：LightningIndexerV2基于一系列操作得到每一个token对应的top-k个位置。

- 计算公式：

  $$
  Top-k \left\{  \left[ 1 \left] \mathop{{}}\nolimits_{{1 \times \text{ }g}}\text{@} \left[  \left( W\text{@} \left[ 1 \left] \mathop{{}}\nolimits_{{1\text{ } \times \text{ }S\mathop{{}}\nolimits_{{k}}}} \left) \text{ } \odot \text{ }ReLU \left( Q\mathop{{}}\nolimits_{{index}}\text{@}K\mathop{{}}\nolimits_{{T}}^{{index}} \left)  \left]  \right\} \right. \right. \right. \right. \right. \right. \right. \right. \right. \right.
  $$

- 主要计算过程为：

  1. 将某个token对应的输入参数`q`（$Q_{index}\in\R^{g\times d}$）乘以给定上下文`k`（$K_{index}\in\R^{S_{k}\times d}$），得到相关性。
  2. 通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
  3. 将其与权重系数`w`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$sparseIndices$，并输出对应的$sparseValues$，作为Attention的输入。

## 参数说明

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnLightningIndexerV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLightningIndexerV2”接口执行计算。

```Cpp
aclnnStatus aclnnLightningIndexerV2GetWorkspaceSize(
    const aclTensor     *q,
    const aclTensor     *k,
    const aclTensor     *w,
    const aclTensor     *cuSeqlensQOptional,
    const aclTensor     *cuSeqlensKOptional,
    const aclTensor     *sequsedQOptional,
    const aclTensor     *sequsedKOptional,
    const aclTensor     *cmpResidualKOptional,
    const aclTensor     *blockTableOptional,
    const aclTensor     *outputIdxOffsetOptional,
    const aclTensor     *metadataOptional,
    int64_t              topk,
    int64_t              maxSeqlenQ,
    char                *layoutQ,
    char                *layoutK,
    int64_t              maskMode,
    int64_t              cmpRatio,
    bool                 returnValue,
    const aclTensor     *sparseIndices,
    const aclTensor     *sparseValues,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnLightningIndexerV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnLightningIndexerV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 240px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 233px">
  <col style="width: 119px">
  <col style="width: 215px">
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
    <td>公式中的输入Q。</td>
    <td>不支持空tensor。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    <td><ul><li>layout_q为BSND时，shape为(B,S1,N1,D)。</li><li>layout_q为TND时，shape为(T1,N1,D)。</li><li>不支持空tensor。</li></ul></td>
    <td>x</td>
    </tr>
    <tr>
    <td>k（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td><ul><li>不支持空tensor。</li><li>block_num为PageAttention时block总数，block_size为一个block的token数。</li></ul></td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    <td><ul><li>layout_k为PA_BBND时，shape为(block_num, block_size, N2, D)。</li><li>layout_k为BSND时，shape为(B, S2, N2, D)。</li><li>layout_k为TND时，shape为(T2, N2, D)。</li></ul>
    </td>
    <td>✓</td>
    </tr>
    <tr>
    <td>w（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的输入W。</td>
    <td>不支持空tensor。</td>
    <td>FLOAT</td>
    <td>ND</td>
    <td><ul><li>layout_q为BSND时，shape为(B,S1,N1)。</li><li>layout_q为TND时，shape为(T1,N1)。</li></ul></td>
    <td>x</td>
    </tr>
    <tr>
    <td>cuSeqlensQOptional（aclTensor*）</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中q的有效token数的累加和</td>
    <td><ul><li>shape约束为长度为B+1的一维tensor。</li><li>不能出现负值。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B+1,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>cuSeqlensKOptional（aclTensor*）</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中k的有效token数的累加和。</td>
    <td><ul><li>shape约束为长度为B+1的一维tensor。</li><li>不能出现负值。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B+1,)</td>
    <td>x</td>
    </tr>
  <tr>
    <td>sequsedQOptional（aclTensor*）</td>
    <td>输入</td>
    <td>不同Batch中q的真实使用长度。</td>
    <td><ul><li>shape约束为长度为B的一维tensor。</li><li>不能出现负值。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
  <tr>
    <td>sequsedKOptional（aclTensor*）</td>
    <td>输入</td>
    <td>不同Batch中k的真实使用长度。</td>
    <td><ul><li>shape约束为长度为B的一维tensor。</li><li>不能出现负值。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
  <tr>
    <td>cmpResidualKOptional（aclTensor*）</td>
    <td>输入</td>
    <td>表示k压缩前token数量除以cmpRatio的余数。</td>
    <td><ul><li>不支持空tensor。</li><li>需要在maskMode等于3，cmpRatio不等于1的场景下使用。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>blockTableOptional（aclTensor*）</td>
    <td>输入</td>
    <td>表示PageAttention中KV存储使用的block映射表。</td>
    <td><ul><li>不支持空tensor。</li><li>PageAttention场景下，block_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大的序列长度对应的block数量）</li><li>shape约束大小为[1-1024]</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>shape支持(B,S2_max/block_size)</td>
    <td>x</td>
    </tr>
  <tr>
    <td>outputIdxOffsetOptional（aclTensor*）</td>
    <td>输入</td>
    <td>表示topK结果输出索引所需要加上的偏移。</td>
    <td><ul><li>值必须大于0。</li><li>加上偏移后，topK index不能超过int32最大值</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
  <tr>
  <td>metadataOptional（aclTensor*）</td>
  <td>输入</td>
  <td>QuantLightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。</td>
  <td><ul><li>不支持空tensor。</li><li>block_size取值为16的整数倍，最大支持到1024。</li></ul></td>
  <td>INT32</td>
  <td>ND</td>
  <td>shape支持(B,S2_max/block_size)</td>
  <td>x</td>
  </tr>
  <tr>
  <td>topk（int64_t）</td>
  <td>输入</td>
  <td>topK阶段需要保留的block数量。</td>
  <td><ul><li>当前支持[1, 8192]。</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>maxSeqlenQ（int64_t）</td>
  <td>输入</td>
  <td>q的最大序列长度。</td>
  <td><ul><li>当前支持[-1]或大于等于[0]。</li><li>-1表示任意可能长度。</li><li>建议值是-1。</li></ul>
  </td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>layoutQ（char*）</td>
  <td>输入</td>
  <td>用于标识输入q的数据排布格式。</td>
  <td><ul><li>当前支持BSND、TND。</li><li>建议值是BSND。</li></ul></td>
  <td>STRING</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>layoutK（char*）</td>
  <td>输入</td>
  <td>用于标识输入k的数据排布格式。</td>
  <td><ul><li>当前支持PA_BBND、BSND、TND。</li><li>建议值是BSND。</li></ul></td>
  <td>STRING</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>maskMode（int64_t）</td>
  <td>输入</td>
  <td>表示mask的模式。</td>
  <td><ul><li>mask_mode为0时，代表defaultMask模式。</li><li>mask_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li><li>建议值是0。</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>cmpRatio（int64_t）</td>
  <td>输入</td>
  <td>用于稀疏计算，表示k的压缩倍数。</td>
  <td><ul><li>支持1/4/128。</li><li>建议值是1。</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>returnValue（bool）</td>
  <td>输入</td>
  <td>代表是否需要返回Indices对应的Values值。</td>
  <td><ul><li>0代表不返回，1代表返回值。</li><li>建议值是0。</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>sparseIndices（aclTensor*）</td>
  <td>输出</td>
  <td>公式中的Indices输出。</td>
  <td>不支持空tensor。</td>
  <td>INT32</td>
  <td>ND</td>
  <td><ul><li>layout_query为"BSND"时输出shape为[B, S1, N2, topk]。</li><li>layout_query为"TND"时输出shape为[T1, N2, topk]。</li></ul></td>
  <td>x</td>
  </tr>
  <tr>
  <td>sparseValues（aclTensor*）</td>
  <td>输出</td>
  <td>公式中的Indices对应的Values输出。</td>
  <td>不支持空tensor。</td>
  <td>INT32</td>
  <td>ND</td>
  <td><ul><li>layout_query为"BSND"时输出shape为[B, S1, N2, topk]。</li><li>layout_query为"TND"时输出shape为[T1, N2, topk]。</li></ul></td>
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

  - q、k、w、q_descale、k_descale参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
  - 使用S1和S2分别表示q和k的输入样本序列长度，N1和N2分别表示q和k对应的多头数，k表示最后选取的索引个数。参数q中的D和参数k中的D值相等为128。T1和T2分别表示q和k的输入样本序列长度的累加和。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
      </tr>
      <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>q、k、w、cuSeqlensQ、cuSeqlensK、sequsedQ、sequsedK、cmpResidualK、blockTable、outputIdxOffset、metadata、sparseIndices、sparseValues的数据类型和数据格式不在支持的范围内。</td>
      </tr>
      </tbody>
    </table>

## aclnnLightningIndexerV2

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantLightningIndexerV2GetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnLightningIndexerV2默认确定性实现。
- 参数q的N支持1~64，k的N支持1。
- headdim支持128。
- pa_kv_cache支持0轴非连续；pa_block_size支持1~1024，满足block大小32B对齐。
- 参数q、k的数据类型应保持一致。
- sparse_indices无效部分填-1；sparse_values无效部分填-inf。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>:
  - topk取值范围当前仅支持[1, 2048]，以及3072、4096、5120、6144、7168、8192。
  - 当前不支持sequsedQOptional、outputIdxOffsetOptional、maxSeqlenQ功能，不建议传入这些参数。
  - 当layout_k为PA_BBND时，必须传入sequsedKOptional；当layout_k不为PA_BBND时，不支持sequsedKOptional功能，不建议传入该参数。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_lightning_indexer_v2.cpp">test_aclnn_lightning_indexer_v2
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnLightningIndexerV2.md">aclnnLightningIndexerV2
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
