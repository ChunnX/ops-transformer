# aclnnGenericBlockSparseAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- **接口功能**：GenericBlockSparseAttention稀疏注意力计算，支持沿着S轴任意粒度的块稀疏模式，通过sparseBlockIdx指定每个Q块选择的KV块，sparseBlockCount指定每个Q块保留的KV块数量，实现高效的稀疏注意力计算。须先调用aclnnGenericBlockSparseAttentionMetadata生成metadata。

- **计算公式**：稀疏块大小：$blockShapeX \times blockShapeY$

  $$
  attentionOut = Softmax(scaleValue \cdot query \cdot key_{sparse}^{T} + atten\_mask) \cdot value_{sparse}
  $$

  GenericBlockSparseAttention输入query、key、value的数据排布格式支持从多种维度排布解读，可通过layoutQ和layoutKv传入。
  - B：表示输入样本批量大小（Batch）
  - T：B和S合轴紧密排列的长度（Total tokens）
  - S：表示输入样本序列长度（Seq-Length）
  - H：表示隐藏层的大小（Head-Size）
  - N：表示多头数（Head-Num）
  - D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）

  当前支持的布局：
  - layoutQ: "TND"
  - layoutKv: "PAGED_BBND"

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnGenericBlockSparseAttentionGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnGenericBlockSparseAttention"接口执行计算。

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *sparseBlockIdx,
  const aclTensor   *sparseBlockCount,
  const aclTensor   *metadataOptional,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *qDequantScaleOptional,
  const aclTensor   *kDequantScaleOptional,
  const aclTensor   *vDequantScaleOptional,
  const aclTensor   *pQuantScaleOptional,
  const aclTensor   *cuSeqLengthsQOptional,
  const aclTensor   *cuSeqLengthsKvOptional,
  const aclTensor   *sequsedQOptional,
  const aclTensor   *sequsedKvOptional,
  const aclTensor   *blockTableOptional,
  const aclIntArray *blockShape,
  int64_t            isPackedGQA,
  char              *layoutQ,
  char              *layoutKv,
  double             scaleValue,
  int64_t            maskType,
  int64_t            quantType,
  double             dstTypeMax,
  int64_t            softmaxPrecision,
  int64_t            winLeft,
  int64_t            winRight,
  int64_t            returnSoftmaxlse,
  aclTensor         *attentionOut,
  aclTensor         *softmaxLseOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnGenericBlockSparseAttentionGetWorkspaceSize

- **参数说明**

<table style="undefined;table-layout: fixed; width: 1550px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 200px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
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
      <td>query</td>
      <td>输入</td>
      <td>公式中的query。</td>
      <td>支持的shape为：
    <ul>
          <li>TND: [totalQTokens, headNum, headDim]。</li>
          <li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li>
          <li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li>
        </ul>
      </td>
      <td>FLOAT16/BFLOAT16/FLOAT8_E4M3FN/FLOAT4_E2M1FN/HIFLOAT8</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的key。</td>
      <td>
        作为原始Key输入时，支持的shape为：
    <ul>
          <li>TND: [totalKTokens, numKeyValueHeads, headDim]。</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li>
        </ul>
        作为Key Cache输入时，支持的shape为：
    <ul>
          <li>PAGED_BBND: [numBlocks, blockSize, numKeyValueHeads, headDim]。</li>
          <li>PAGED_BNBD: [numBlocks, numKeyValueHeads, blockSize, headDim]。</li>
        </ul>
        当前落地为PAGED_BBND时，dim0非连续约束见约束说明；其余layout当前不支持。
      </td>
      <td>FLOAT16/BFLOAT16/FLOAT8_E4M3FN/FLOAT4_E2M1FN/HIFLOAT8</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的value。</td>
      <td>
        作为原始Value输入时，支持的shape为：
    <ul>
          <li>TND: [totalVTokens, numKeyValueHeads, headDim]。</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li>
        </ul>
        作为Value Cache输入时，支持的shape为：
    <ul>
          <li>PAGED_BBND: [numBlocks, blockSize, numKeyValueHeads, headDim]。</li>
          <li>PAGED_BNBD: [numBlocks, numKeyValueHeads, blockSize, headDim]。</li>
        </ul>
        其中blockSize为cache的页的大小，支持范围[16, 512]，需要满足16对齐。当前落地为PAGED_BBND时，dim0非连续约束见约束说明；其余layout当前不支持。
      </td>
      <td>FLOAT16/BFLOAT16/FLOAT8_E4M3FN/FLOAT4_E2M1FN/HIFLOAT8</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引数组，指定每个Q块选择的KV块索引。</td>
      <td>
        存储每个Q块选择的KV块索引，支持的shape随query布局变化：
        <ul>
          <li>query为TND布局时：</li>
          <ul>
            <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[headNum, totalQBlocks, maxSparseBlockCount]</li>
            <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]</li>
          </ul>
          <li>query为BNSD/BSND布局时：</li>
          <ul>
            <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), maxSparseBlockCount]</li>
            <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[batch, numKeyValueHeads, ceilDiv(maxQSeqLength, blockShapeX), maxSparseBlockCount]</li>
          </ul>
        </ul>
        其中 totalQBlocks = Σ ceilDiv(qStorageLen_i, blockShapeX)，i 为 batch 索引；qStorageLen_i 为 cuSeqLengthsQOptional 前缀和差分得到的<strong>存储长度</strong>（与seqused双长度语义见约束说明）。
<br>maxSparseBlockCount为sparseBlockCount tensor中所有元素的最大值，即所有Q块选择的KV块数量的最大值。传入值须 >= 该最大值，当前上限为64。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量。</td>
      <td>
        存储每个Q块实际选择的KV块数量，支持的shape随query布局变化：
        <ul>
          <li>query为TND布局时：</li>
          <ul>
            <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[headNum, totalQBlocks]</li>
            <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[numKeyValueHeads, totalQBlocks]</li>
          </ul>
          <li>query为BNSD/BSND布局时：</li>
          <ul>
            <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX)]</li>
            <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[batch, numKeyValueHeads, ceilDiv(maxQSeqLength, blockShapeX)]</li>
          </ul>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2/3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输入</td>
      <td>稀疏attention的分核信息。</td>
      <td>
        由aclnnGenericBlockSparseAttentionMetadata生成；INT32一维[1024]；须与当前输入/属性配套，不可跨case复用。目标为可选；当前版本必须传入。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>输入</td>
      <td>公式中的atten_mask。</td>
      <td>atten_mask会与稀疏pattern叠加。目标shape为BOOL [2048,2048]（随maskType变化，见「掩码说明」）。当前不支持，必须传入nullptr。</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qDequantScaleOptional</td>
      <td>输入</td>
      <td>query的反量化缩放因子。</td>
      <td>详情见「量化相关说明」。当前仅quantType=0/5场景可传nullptr；quantType=1~4当前不支持，须传入nullptr。
      </td>
      <td>FLOAT32/FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>kDequantScaleOptional</td>
      <td>输入</td>
      <td>key的反量化缩放因子。</td>
      <td>详情见「量化相关说明」。当前仅quantType=0/5场景可传nullptr；quantType=1~4当前不支持，须传入nullptr。
      </td>
      <td>FLOAT32/FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>vDequantScaleOptional</td>
      <td>输入</td>
      <td>value的反量化缩放因子。</td>
      <td>详情见「量化相关说明」。当前仅quantType=0/5场景可传nullptr；quantType=1~4当前不支持，须传入nullptr。
      </td>
      <td>FLOAT32/FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>pQuantScaleOptional</td>
      <td>输入</td>
      <td>非mx量化模式下，online-softmax的结果P矩阵所需的量化系数。</td>
      <td>详情见「量化相关说明」。当前须传入nullptr。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>x</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的query序列长度，以前缀和形式存储。</td>
      <td>可选输入，用于变长序列场景：
        <ul>
          <li>当layoutQ为"TND"时：该项输入必须配置</li>
          <li>当layoutQ为"BNSD""BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；<br>如不配置该项输入(传入nullptr)，算子内会按照query的shape中的S进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的key/value序列长度，以前缀和形式存储。</td>
      <td>可选输入，用于变长序列场景：
        <ul>
          <li>当layoutKv为"TND"/"PAGED_BBND"/"PAGED_BNBD"时：该项输入必须配置</li>
          <li>当layoutKv为"BNSD""BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；<br>如不配置该项输入(传入nullptr)，算子内会按照key/value的shape中的S进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入</td>
      <td>各batch中query的实际序列长度。</td>
      <td>
        可选输入，shape为(B,)。可与cuSeqLengthsQOptional同时传入（layoutQ为TND时cu仍必传）；各batch元素须≤对应cu存储长度；须与Metadata侧一致。双长度语义见约束说明。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各batch中kv的实际序列长度。</td>
      <td>
        可选输入，shape为(B,)。可与cuSeqLengthsKvOptional同时传入（layoutKv为TND/PAGED时cu仍必传）；各batch元素须≤对应cu存储长度；须与Metadata侧一致。双长度语义见约束说明。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>Block表用于PagedAttention。</td>
      <td>如配置此输入，则表示使用PagedAttention，layout、KV shape均需对应配置（见「Paged Attention相关说明」）。目标可为nullptr（原始KV）；当前必须传入，且layoutKv须为PAGED_BBND。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>输入</td>
      <td>代表稀疏块形状数组。</td>
      <td>含两个元素[blockShapeX, blockShapeY]。<br>blockShapeX支持任意值，不可超过int64表示范围。<br>blockShapeY支持按16对齐的任意值，不可超过int64表示范围。<br>开启量化时的额外约束见「量化相关说明」。<br>当前仅支持blockShape=[1,128]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isPackedGQA</td>
      <td>输入</td>
      <td>代表进行块状稀疏时，同一个group内的qHead是否共享同样的稀疏pattern<br>（注：不同batch之间不会共享同样的稀疏pattern，该入参仅区分head维度的共享情况）。</td>
      <td>若取值为0，则代表同一个group内的qHead不共享同样的稀疏pattern；<br>若取值为1，则代表同一个group内的qHead共享同样的稀疏pattern。<br>当前仅支持1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>输入</td>
      <td>代表输入query的数据排布格式。</td>
      <td>目标支持"TND""BNSD""BSND"，详见「layout对应关系说明」。当前仅支持"TND"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>输入</td>
      <td>代表输入key、value的数据排布格式。</td>
      <td>目标支持"TND""BNSD""BSND""PAGED_BBND""PAGED_BNBD"，详见「layout对应关系说明」。当前仅支持"PAGED_BBND"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>输入</td>
      <td>公式中的scale，代表缩放系数。</td>
      <td>一般设置为D^-0.5；传0时算子内按1/√D处理。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>输入</td>
      <td>表示attention计算中的掩码类型。</td>
      <td>
        取值0~5，含义见「掩码说明」。<br>
        当前仅支持1（causal mask，attenMaskOptional须为nullptr，winLeft/winRight为-1）。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>输入</td>
      <td>代表采用的量化手段。</td>
      <td>
        取值0~5，完整配置见「量化相关说明」：
        <ul>
          <li>0：非量化</li>
          <li>1：FLOAT8静态per-group（当前不支持）</li>
          <li>2：FLOAT8动态micro scaling（当前不支持）</li>
          <li>3：FLOAT4动态OCP（当前不支持）</li>
          <li>4：FLOAT4动态CX（当前不支持）</li>
          <li>5：FLOAT8全量化（算子内将P cast为FP8，不传量化系数；Ascend 950支持）</li>
        </ul>
        当前可用：0；Ascend 950上可选5。取值1~4传入将校验失败。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax</td>
      <td>输入</td>
      <td>MXFP4 CX量化时,传入的自定义量化量程。</td>
      <td>
        目标：仅当quantType=3或4时，可设0.0或[6.0, 12.0]。0.0表示Amax为量化结果dtype最大值；[6.0,12.0]表示使用传入值。<br>
        当前只支持传入0.0。
      </td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>输入</td>
      <td>Softmax计算采取的精度级别。</td>
      <td>
        控制online softmax阶段以及rescale阶段运算使用的数据类型。取值0或1：
        <ul>
          <li>0：online softmax和rescale全部采取fp32，适合追求计算精度的场景。</li>
          <li>1：混合精度；online softmax采取fp16/bf16（与attentionOut相同），rescale采取fp32，online softmax阶段可能数值溢出。</li>
        </ul>
        芯片约束见「约束说明」（Ascend 950仅1；Atlas A2/A3上FP16可0/1，BF16仅0；FP8路径仅1）。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>输入</td>
      <td>滑窗attention场景下，滑窗需要向前包含多少个token。</td>
      <td>用于滑窗attention；不使能时必须为-1，需与maskType配合，见「掩码说明」。当前只支持传入-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>输入</td>
      <td>滑窗attention场景下，滑窗需要向后包含多少个token。</td>
      <td>用于滑窗attention；不使能时必须为-1，需与maskType配合，见「掩码说明」。当前只支持传入-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxlse</td>
      <td>输入</td>
      <td>是否使能softmaxLse输出的标志位。</td>
      <td>
    当前只支持传0或1
    <ul>
          <li>0：表示不输出softmaxLse</li>
      <li>1：表示输出softmaxLse，相比不输出softmaxLse可能存在性能损失</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>输出</td>
      <td>公式中的attentionOut。</td>
      <td>数据类型和shape与query保持一致。</td>
      <td>FLOAT16/BFLOAT16（Q/K/V 为 FP8 时由 attentionOut tensor 指定 FP16/BF16）</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>输出</td>
      <td>Softmax计算的log-sum-exp中间结果。</td>
      <td>支持的shape随着query的shape改变：
    <ul>
          <li>query为"TND": [totalQTokens, headNum, 1]。</li>
          <li>query为"BNSD": [batch, headNum, maxQSeqLength, 1]。</li>
          <li>query为"BSND": [batch, maxQSeqLength, headNum, 1]。</li>
        </ul>
      </td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
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
      <td>返回op执行器，包含算子计算流程。</td>
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

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>query/key/value/sparseBlockIdx/sparseBlockCount/attentionOut 等必选指针为空；returnSoftmaxlse=1 时 softmaxLse 为空。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>isPackedGQA、layout、maskType、blockShape、softmaxPrecision、quantType 等与约束不匹配。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>metadata 为空或 Contiguous/InferShape 失败（如 layout 不支持、缺少 blockTable 等）。</td>
    </tr>
  </tbody>
  </table>

## aclnnGenericBlockSparseAttention

- **参数说明**

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGenericBlockSparseAttentionGetWorkspaceSize获取。</td>
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

  返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## layout对应关系说明

<table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>layoutQ</th>
      <th>layoutKv</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">TND</td>
      <td>TND</td>
      <td>用于原始KV输入，每个batch的seqlen拼接在一起的场景。</td>
    </tr>
    <tr>
      <td>PAGED_BBND</td>
      <td>用于paged kv cache输入，数据按[numBlocks, blockSize, numKeyValueHeads, headDim]排布。</td>
    </tr>
    <tr>
      <td>PAGED_BNBD</td>
      <td>用于paged kv cache输入，数据按[numBlocks, numKeyValueHeads, blockSize, headDim]排布。</td>
    </tr>
    <tr>
      <td rowspan="3">BSND</td>
      <td>BSND</td>
      <td>用于原始KV输入。</td>
    </tr>
    <tr>
      <td>PAGED_BBND</td>
      <td>用于paged kv cache输入，数据按[numBlocks, blockSize, numKeyValueHeads, headDim]排布。</td>
    </tr>
    <tr>
      <td>PAGED_BNBD</td>
      <td>用于paged kv cache输入，数据按[numBlocks, numKeyValueHeads, blockSize, headDim]排布。</td>
    </tr>
    <tr>
      <td rowspan="3">BNSD</td>
      <td>BNSD</td>
      <td>用于原始KV输入。</td>
    </tr>
    <tr>
      <td>PAGED_BBND</td>
      <td>用于paged kv cache输入，数据按[numBlocks, blockSize, numKeyValueHeads, headDim]排布。</td>
    </tr>
    <tr>
      <td>PAGED_BNBD</td>
      <td>用于paged kv cache输入，数据按[numBlocks, numKeyValueHeads, blockSize, headDim]排布。</td>
    </tr>
  </tbody>
  </table>

当前支持组合：layoutQ="TND" + layoutKv="PAGED_BBND"。其余layoutQ/layoutKv组合当前不支持。

## Paged Attention相关说明

<table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
        <th>blockTable</th>
        <th>kvLayout</th>
        <th>Key/Value</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td rowspan="2">非空，shape为[batch, maxNumBlocksPerBatch]，代表使能paged cache</td>
      <td>PAGED_BBND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads, headDim]</td>
    </tr>
    <tr>
      <td>PAGED_BNBD</td>
      <td>[numBlocks, numKeyValueHeads, blockSize, headDim]</td>
    </tr>
    <tr>
      <td rowspan="3">空，代表不使能paged cache，算子接收原始KV输入</td>
      <td>TND</td>
      <td>[totalKTokens, numKeyValueHeads, headDim]</td>
    </tr>
    <tr>
      <td>BSND</td>
      <td>[batch, maxKvSeqLength, numKeyValueHeads, headDim]</td>
    </tr>
    <tr>
      <td>BNSD</td>
      <td>[batch, numKeyValueHeads, maxKvSeqLength, headDim]</td>
    </tr>
  </tbody>
  </table>

当前必须传入非空blockTable，且layoutKv为PAGED_BBND。blockTable为nullptr（原始KV）及PAGED_BNBD当前不支持。

## 量化相关说明

下列量化配置按Ascend 950（A5）代际描述；当前可用范围见「约束说明」。

<table style="undefined;table-layout: fixed;width: 2000px"><colgroup>
  <col style="width: 100px">
  <col style="width: 30px">
  <col style="width: 60px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
        <th>quantType</th>
        <th>QKV的数据类型</th>
        <th>对称/非对称</th>
        <th>P量化动态/静态</th>
        <th>量化粒度</th>
        <th>量化参数shape</th>
        <th>量化参数dType</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td>0</td>
      <td>非量化，QKV直接作为输入进行计算</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>
        qDequantScaleOptional(不传)<br>
        kDequantScaleOptional(不传)<br>
        vDequantScaleOptional(不传)<br>
        pQuantScaleOptional(不传)
      </td>
      <td>-</td>
    </tr>
    <tr>
      <td>1</td>
      <td rowspan="2">FLOAT8_E4M3</td>
      <td rowspan="4">对称</td>
      <td>静态</td>
      <td>perGroup，QKV均沿S维度分组，group大小和稀疏块尺寸必须相同；<br>特别的，当KV为paged cache时，blockSize需要为blockShapeY的整数倍</td>
      <td>
        qDequantScaleOptional(必选)：
        <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeX), headNum, 1]。</li>
          <li>BNSD: [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), 1]。</li>
          <li>BSND: [batch, ceilDiv(maxQSeqLength, blockShapeX), headNum, 1]。</li>
        </ul>
        kDequantScaleOptional(必选)：
        <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeY), kvHeadNum, 1]。</li>
          <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, blockShapeY), 1]。</li>
          <li>BSND: [batch, ceilDiv(maxKvSeqLength, blockShapeY), kvHeadNum, 1]。</li>
          <li>PAGED_BBND: [batch, ceilDiv(blockSize, blockShapeY), kvHeadNum, 1]。</li>
          <li>PAGED_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, blockShapeY), 1]。</li>
        </ul>
        vDequantScaleOptional(必选)：
        <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeY), kvHeadNum, 1]。</li>
          <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, blockShapeY), 1]。</li>
          <li>BSND: [batch, ceilDiv(maxKvSeqLength, blockShapeY), kvHeadNum, 1]。</li>
          <li>PAGED_BBND: [batch, ceilDiv(blockSize, blockShapeY), kvHeadNum, 1]。</li>
          <li>PAGED_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, blockShapeY), 1]。</li>
        </ul>
        pQuantScaleOptional(可选)：
        <ul>
          <li>输入时，仅包含单一元素，用于用户控制P的静态量化系数: [1]。</li>
          <li>nullptr: 算子默认P的静态量化系数为448.0。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
    </tr>
    <tr>
      <td>2</td>
      <td>动态</td>
      <td rowspan="3">micro scaling，QKV沿着矩阵乘累加轴，按固定大小32进行分组；<br>特别的，当KV为paged cache时，blockSize需要为64的整数倍</td>
      <td rowspan="3">
        qDequantScaleOptional(必选)：
        <ul>
          <li>TND: [totalQTokens, headNum, ceilDiv(headDim, 64), 2]。</li>
          <li>BNSD: [batch, headNum, maxQSeqLength, ceilDiv(headDim, 64), 2]。</li>
          <li>BSND: [batch, maxQSeqLength, headNum, ceilDiv(headDim, 64), 2]。</li>
        </ul>
        kDequantScaleOptional(必选)：
        <ul>
          <li>TND: [totalKTokens, kvHeadNum, ceilDiv(headDim, 64), 2]。</li>
          <li>BNSD: [batch, kvHeadNum, maxKvSeqLength, ceilDiv(headDim, 64), 2]。</li>
          <li>BSND: [batch, maxKvSeqLength, kvHeadNum, ceilDiv(headDim, 64), 2]。</li>
          <li>PAGED_BBND: [batch, blockSize, kvHeadNum, ceilDiv(headDim, 64), 2]。</li>
          <li>PAGED_BNBD: [batch, kvHeadNum, blockSize, ceilDiv(headDim, 64), 2]。</li>
        </ul>
        vDequantScaleOptional(必选)：
        <ul>
          <li>TND: [batch*ceilDiv(kvSeqLength, 64), kvHeadNum, headDim, 2]。</li>
          <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, 64), headDim, 2]。</li>
          <li>BSND: [batch, ceilDiv(maxKvSeqLength, 64), kvHeadNum, headDim, 2]。</li>
          <li>PAGED_BBND: [batch, ceilDiv(blockSize, 64), kvHeadNum, headDim, 2]。</li>
          <li>PAGED_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, 64), headDim, 2]。</li>
        </ul>
      </td>
      <td rowspan="3">FLOAT8_E4M3</td>
    </tr>
    <tr>
      <td>3</td>
      <td rowspan="2">FLOAT4_E2M1</td>
      <td>动态OCP</td>
    </tr>
    <tr>
      <td>4</td>
      <td>动态CX</td>
    </tr>
    <tr>
      <td>5</td>
      <td>FLOAT8_E4M3</td>
      <td>对称</td>
      <td>静态</td>
      <td>不传入量化系数，而是在算子内直接将P cast成fp8</td>
      <td>
        qDequantScaleOptional(不传)<br>
        kDequantScaleOptional(不传)<br>
        vDequantScaleOptional(不传)<br>
        pQuantScaleOptional(不传)
      </td>
      <td>FLOAT32</td>
    </tr>
  </tbody>
  </table>

当前可用quantType=0；Ascend 950上可选quantType=5。quantType=1~4当前不支持。

## 掩码说明

<table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
<col style="width: 144px">
  <col style="width: 500px">
<col style="width: 144px">
  </colgroup>
  <thead>
    <tr>
        <th>maskType</th>
        <th>含义</th>
        <th>attentionMaskOptional</th>
        <th>winLeft/winRight</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td>0</td>
      <td>不加mask</td>
      <td>不传</td>
      <td>-1/-1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>causal mask</td>
      <td>不传 attenMaskOptional（内置 causal）；规划 BOOL [2048,2048] 下三角</td>
      <td>-1/-1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>window mask</td>
      <td>规划：BOOL [2048,2048] 下三角，与 winLeft/winRight 配合</td>
      <td>实际window包括的向前/向后看的token数</td>
    </tr>
    <tr>
      <td>3~5</td>
      <td>各类特化mask</td>
      <td>后续补充mask描述</td>
      <td>-1/-1</td>
    </tr>
  </tbody>
  </table>

当前仅支持maskType=1（算子内置causal，attenMaskOptional须为nullptr，winLeft/winRight为-1）。maskType为0/2/3~5当前不支持。

## 约束说明

- 确定性计算：aclnnGenericBlockSparseAttention默认确定性实现。
- 调用前须先执行aclnnGenericBlockSparseAttentionMetadata生成metadataOptional（INT32 [1024]），再调用本接口；metadata须与当前输入/属性配套，不可跨case复用。
- layoutQ当前仅支持"TND"。
- layoutKv当前仅支持"PAGED_BBND"。
- isPackedGQA当前仅支持1。
- maskType当前仅支持1（算子内置causal）；attenMaskOptional须传入nullptr；winLeft和winRight当前只支持-1。
- blockShape当前仅支持[1,128]；query/key/value的headDim(D)当前仅支持128；KV页blockSize当前仅支持128，且须等于blockShapeY。
- TND + isPackedGQA=1时：sparseBlockIdx为[N2, totalQBlocks, topK]，sparseBlockCount为[N2, totalQBlocks]；totalQBlocks按cu存储长度分块；maxSparseBlockCount（即topK）须≥sparseBlockCount中所有元素的最大值，且当前上限为64。
- metadataOptional、blockTableOptional、cuSeqLengthsQOptional、cuSeqLengthsKvOptional当前必须传入；blockTableOptional为nullptr（原始KV）当前不支持。
- cuSeqLengthsQOptional在layoutQ为"TND"时必选；cuSeqLengthsKvOptional在layoutKv为"PAGED_BBND"时必选。
- sequsedQOptional/sequsedKvOptional可选，可与cu前缀和同时传入；不传（nullptr）时按cu差分得到的存储长度处理。传入时：分核/任务空间按各batch实际有效长度（seqused）累加；sparseBlockIdx/sparseBlockCount分块与QKV寻址仍按cu存储长度计算；各batch的seqused元素须≤对应cu存储长度，且须与Metadata侧完全一致。
- quantType当前支持0（非量化）；Ascend 950上可选5（FLOAT8_E4M3FN全量化，attentionOut为FLOAT16/BFLOAT16）。quantType为1~4当前不支持。
- qDequantScaleOptional、kDequantScaleOptional、vDequantScaleOptional、pQuantScaleOptional当前须传入nullptr。
- dstTypeMax当前只支持0.0。
- 输入query、key、value的数据类型必须一致；当前支持FLOAT16、BFLOAT16；Ascend 950上另支持FLOAT8_E4M3FN（须quantType=5）。
- returnSoftmaxlse仅支持0或1，分别表示不开启/开启softmaxLse输出；quantType=5时returnSoftmaxlse当前只支持0。
- softmaxPrecision：Ascend 950仅支持1；Atlas A2/A3上FLOAT16可配置0或1，BFLOAT16仅支持0；FLOAT8路径仅支持1。
- scaleValue传0时算子内按$1/\sqrt{D}$处理。
- 输入query的headNum为N1，输入key和value的headNum为N2，则N1 >= N2 && N1 % N2 == 0；groupSize=N1/N2当前须≤128。
- PAGED_BBND下key/value仅dim0（物理页轴）可非连续；页内blockSize×N2×D须连续；stride0 ≥ blockSize×N2×D且按N2×D对齐。
- layout、Paged、量化、掩码等完整参数语义见对应章节。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

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
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
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

  // Regular path smoke case: TND query + PAGED_BBND KV
  int64_t B = 1;
  int64_t S1 = 4;
  int64_t S2 = 256;
  int64_t N1 = 4;
  int64_t N2 = 1;
  int64_t D = 128;
  int64_t topK = 2;
  int64_t blockSize = 128;
  int64_t blockShapeX = 1;
  double scaleValue = 1.0 / sqrt(static_cast<double>(D));

  int64_t T = B * S1;
  int64_t maxBlocks = (S2 + blockSize - 1) / blockSize;
  int64_t totalQBlocks = T;  // blockShapeX == 1

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {T, N1, D};
  std::vector<int64_t> kvShape = {maxBlocks, blockSize, N2, D};
  std::vector<int64_t> sparseIdxShape = {N2, totalQBlocks, topK};
  std::vector<int64_t> sparseCountShape = {N2, totalQBlocks};
  std::vector<int64_t> blockTableShape = {B, maxBlocks};
  std::vector<int64_t> cuSeqQShape = {B + 1};
  std::vector<int64_t> cuSeqKvShape = {B + 1};
  std::vector<int64_t> metadataShape = {1024};
  std::vector<int64_t> attnOutShape = {T, N1, D};
  std::vector<int64_t> softmaxLseShape = {T, N1, 1};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* sparseIdxDeviceAddr = nullptr;
  void* sparseCountDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* cuSeqQDeviceAddr = nullptr;
  void* cuSeqKvDeviceAddr = nullptr;
  void* blockTableDeviceAddr = nullptr;
  void* attnOutDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* sparseIdx = nullptr;
  aclTensor* sparseCount = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* cuSeqQ = nullptr;
  aclTensor* cuSeqKv = nullptr;
  aclTensor* blockTable = nullptr;
  aclTensor* attnOut = nullptr;
  aclTensor* softmaxLse = nullptr;

  int64_t qSize = GetShapeSize(qShape);
  int64_t kvSize = GetShapeSize(kvShape);
  int64_t sparseIdxSize = GetShapeSize(sparseIdxShape);
  int64_t sparseCountSize = GetShapeSize(sparseCountShape);
  int64_t blockTableSize = GetShapeSize(blockTableShape);
  int64_t attnOutSize = GetShapeSize(attnOutShape);
  int64_t softmaxLseSize = GetShapeSize(softmaxLseShape);

  std::vector<uint16_t> qHostData = MakeFp16Data(qSize, 1.0f);
  std::vector<uint16_t> kHostData = MakeFp16Data(kvSize, 1.0f);
  std::vector<uint16_t> vHostData = MakeFp16Data(kvSize, 1.0f);
  std::vector<int32_t> sparseIdxHostData(sparseIdxSize, -1);
  std::vector<int32_t> sparseCountHostData(sparseCountSize, 0);
  std::vector<int32_t> blockTableHostData(blockTableSize);
  std::iota(blockTableHostData.begin(), blockTableHostData.end(), 0);
  std::vector<int64_t> cuSeqQHostData = {0, S1};
  std::vector<int64_t> cuSeqKvHostData = {0, S2};
  std::vector<int32_t> metadataHostData(1024, 0);
  std::vector<uint16_t> attnOutHostData = MakeFp16Data(attnOutSize, 0.0f);
  std::vector<float> softmaxLseHostData(softmaxLseSize, 0.0f);

  // Causal-like sparse window: keep up to topK trailing KV blocks per Q token.
  int64_t history = S2 - S1;
  for (int64_t qBlock = 0; qBlock < totalQBlocks; qBlock++) {
    int64_t visible = history + qBlock + 1;
    int64_t lastKvBlock = std::min(maxBlocks - 1, (visible - 1) / blockSize);
    int64_t count = std::min(topK, lastKvBlock + 1);
    int64_t start = lastKvBlock - count + 1;
    for (int64_t i = 0; i < count; i++) {
      sparseIdxHostData[qBlock * topK + i] = static_cast<int32_t>(start + i);
    }
    sparseCountHostData[qBlock] = static_cast<int32_t>(count);
  }

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, kvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseIdxHostData, sparseIdxShape, &sparseIdxDeviceAddr, aclDataType::ACL_INT32, &sparseIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseCountHostData, sparseCountShape, &sparseCountDeviceAddr, aclDataType::ACL_INT32,
                        &sparseCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqQHostData, cuSeqQShape, &cuSeqQDeviceAddr, aclDataType::ACL_INT64, &cuSeqQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqKvHostData, cuSeqKvShape, &cuSeqKvDeviceAddr, aclDataType::ACL_INT64, &cuSeqKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(blockTableHostData, blockTableShape, &blockTableDeviceAddr, aclDataType::ACL_INT32,
                        &blockTable);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT16, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT,
                        &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const int64_t blockShapeData[] = {blockShapeX, blockSize};
  aclIntArray* blockShape = aclCreateIntArray(blockShapeData, 2);
  CHECK_RET(blockShape != nullptr, LOG_PRINT("aclCreateIntArray failed\n"); return -1);

  char layoutQ[] = "TND";
  char layoutKv[] = "PAGED_BBND";

  // 3. 先调用 Metadata，再调用主算子
  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;
  ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
      sparseIdx, sparseCount, cuSeqQ, cuSeqKv, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, 1, layoutQ, layoutKv, 1,
      0, 1, -1, -1, metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnGenericBlockSparseAttentionMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor,
                                                 stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionMetadata failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret);
            return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnGenericBlockSparseAttentionGetWorkspaceSize(
      q, k, v, sparseIdx, sparseCount, metadata, nullptr, nullptr, nullptr, nullptr, nullptr, cuSeqQ, cuSeqKv, nullptr,
      nullptr, blockTable, blockShape, 1, layoutQ, layoutKv, scaleValue, 1, 0, 0.0, 1, -1, -1, 1, attnOut, softmaxLse,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnGenericBlockSparseAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnGenericBlockSparseAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttention failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(attnOutShape, &attnOutDeviceAddr);

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyIntArray(blockShape);
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(sparseIdx);
  aclDestroyTensor(sparseCount);
  aclDestroyTensor(metadata);
  aclDestroyTensor(cuSeqQ);
  aclDestroyTensor(cuSeqKv);
  aclDestroyTensor(blockTable);
  aclDestroyTensor(attnOut);
  aclDestroyTensor(softmaxLse);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(sparseIdxDeviceAddr);
  aclrtFree(sparseCountDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(cuSeqQDeviceAddr);
  aclrtFree(cuSeqKvDeviceAddr);
  aclrtFree(blockTableDeviceAddr);
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
