# BSASelectBlockMask

## 产品支持情况

| 产品                                    | 是否支持 |
| :-------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>  |    √     |
| <term>Atlas A3训练系列产品</term>      |    √     |
| <term>Atlas A3推理系列产品</term>      |    √     |
| <term>Atlas A2训练系列产品</term>      |    √     |
| <term>Atlas A2推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2推理产品</term> |    ×     |
| <term>Atlas推理系列产品</term>         |    ×     |
| <term>Atlas训练系列产品</term>         |    ×     |

## 功能说明

* 接口功能​：aclnnBSASelectBlockMask是BSA（BlockSparseAttention）的前置算子，负责根据Query和Key的内容动态生成blockSparseMask，使BSA的调用链从"手动提供掩码"变为"根据Q/K内容自适应选择稀疏模式"。

* 计算公式​：

  设blockShape = [blockShapeX, blockShapeY]，压缩后块数：

  $$
  Xblocks = \lceil Sq / blockShapeX \rceil,\quad Yblocks = \lceil Skv / blockShapeY \rceil
  $$

  **Step1：均值池化压缩 (Mean Pooling Compression)**

  当actualBlockLenQuery / actualBlockLenKey为null时（完整压缩）：

  $$
  q\_compressed[b, n, x, d] = \frac{1}{blockShapeX} \sum_{i=0}^{blockShapeX-1} query[b, n, x \cdot blockShapeX + i, d]
  $$

  $$
  k\_compressed[b, n, y, d] = \frac{1}{blockShapeY} \sum_{j=0}^{blockShapeY-1} key[b, n, y \cdot blockShapeY + j, d]
  $$

  当actualBlockLenQuery / actualBlockLenKey非null时（部分压缩），仅对每个block内前actualBlockLen个token取均值：

  $$
  q\_compressed[b, n, x, d] = \frac{1}{actualBlockLenQ[b,x]} \sum_{i=0}^{actualBlockLenQ[b,x]-1} query[b, n, x \cdot blockShapeX + i, d]
  $$

  $$
  k\_compressed[b, n, y, d] = \frac{1}{actualBlockLenK[b,y]} \sum_{j=0}^{actualBlockLenK[b,y]-1} key[b, n, y \cdot blockShapeY + j, d]
  $$

  **Step2a：QK Matmul**

  $$
  score[b, n, x, y] = scale \cdot \sum_{d=0}^{D-1} q\_compressed[b, n, x, d] \cdot k\_compressed[b, n, y, d]
  $$

  **Step2b：Softmax**

  $$
  attn\_score[b, n, x, y] = softmax(score[b, n, x, :]) = \frac{\exp(score[b, n, x, y] - m_{final})}{l_{final}}
  $$

  **Step3：TopK选择生成索引**

  $$
  topk\_value = \text{round}(sparsity \times Xblocks \times Yblocks)
  $$

  $$
  \mathcal{indices}= \text{TopK}\left(attn\_score[b, n, x, y],\; topK\_value\right)
  $$

  其中indices为attn\_score[b, n, x, y] 中topk\_value个最大值对应的索引集合。

  **Step4：生成BlockSparseMask**
  $$
  blockSparseMaskOut[b, n, x, y] =
  \begin{cases}
  1 & (b, n, x, y) \in \mathcal{indices} \\
  0 & (b, n, x, y) \notin \mathcal{indices}
  \end{cases}
  $$

BSASelectBlockMask输入query、key的数据排布格式支持从多种维度排布解读，可通过queryLayout和keyLayout传入。为了方便理解后续支持的具体排布格式（如BNSD、TND等），此处先对排布格式中各缩写字母所代表的维度含义进行统一说明：

* B：表示输入样本批量大小（Batch）
* T：B和S合轴紧密排列的长度（Total tokens）
* S：表示输入样本序列长度（Seq-Length）
* H：表示隐藏层的大小（Head-Size）
* N：表示多头数（Head-Num）
* D：表示隐藏层最小的单元尺寸，需满足D = H / N（Head-Dim）

当前支持的布局：

* queryLayout: "TND" "BNSD"
* keyLayout: "TND" "BNSD"

* **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
    <td>query（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的query矩阵，即公式中的query。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的key矩阵，即公式中的key。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalKTokens, numKeyHeads, headDim]。</li><li>BNSD: [batch, numKeyHeads, maxKSeqLength, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
        <tr>
    <td>blockShape（aclIntArray*）</td>
    <td>输入</td>
    <td>稀疏块形状数组。指定每个稀疏块的二维尺寸（行数和列数）。</td>
    <td><li>当配置此输入时的元素要求：<ul><li>必须包含至少两个元素 [blockShapeX, blockShapeY]。</li><li>blockShapeX: Q方向块大小，必须为64的倍数且大于0。</li><li>blockShapeY: KV方向块大小，必须为64的倍数且大于0。</li></ul></li><li>如不配置（传nullptr），算子将默认blockShapeX = 128，blockShapeY = 128。</li></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>postBlockShape（aclIntArray*）</td>
    <td>输入</td>
    <td>预留参数，用于Softmax后二次压缩。</td>
    <td>当前不支持，必须传入nullptr。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualSeqLengths（aclIntArray*）</td>
    <td>输入</td>
    <td>每个batch的query的实际序列长度。<br>用于描述变长序列场景下（即含有Padding填充数据的场景），每个Batch中实际有效的query token数量。</td>
    <td><li>变长序列场景（当queryLayout为 "TND" 时）：该项输入必须配置。</li><li>定长/变长场景（当queryLayout为 "BNSD" 时）：<ul><li>如配置该项，算子会按指定的有效长度处理，忽略Padding部分的数据，提升性能；</li><li>如不配置（传nullptr），算子将默认把query shape中的S维度作为有效长度进行全量处理。</li></ul></li></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualSeqLengthsK（aclIntArray*）</td>
    <td>输入</td>
    <td>key的实际序列长度数组。<br>用于描述变长序列场景下（即含有Padding填充数据的场景），每个Batch中实际有效的key token数量。</td>
    <td><li>变长序列场景（当keyLayout为 "TND" 时）：该项输入必须配置。</li><li>>定长/变长场景（当keyLayout为 "BNSD" 时）：<ul><li>如配置该项，算子会按指定的有效长度处理，忽略Padding部分的数据，提升性能；</li><li>如不配置（传nullptr），算子将默认把key shape中的S维度作为有效长度进行全量处理。</li></ul></li></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualBlockLenQuery（aclIntArray*）</td>
    <td>输入</td>
    <td>每个query block内实际压缩的有效seq长度。<br>用于部分压缩场景（如末尾不完整块或仅压缩有效token）。</td>
    <td><li>可选输入：<ul><li>BNSD场景：shape为 [B, Xblocks]。</li><li>TND场景：shape为 [TotalBlockNum_Q]（各batch的Xblocks堆叠）。</li><li>每个元素取值范围 [0, blockShapeX]。</li><li>当actualBlockLen = 0时：对应block的q_cmp填0向量，不会被topK选中。</li><li>当actualBlockLen > 0时：仅对前actualBlockLen个token取均值。</li></ul></li>
    <li>如不配置（传nullptr）：对query进行完整压缩（使用完整blockShapeX长度）。</li></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualBlockLenKey（aclIntArray*）</td>
    <td>输入</td>
    <td>每个key block内实际压缩的有效seq长度。<br>用于部分压缩场景（如末尾不完整块或仅压缩有效token）。</td>
    <td><li>可选输入：<ul><li>BNSD场景：shape为 [B, Yblocks]。</li><li>TND场景：shape为 [TotalBlockNum_KV]（各batch的Yblocks堆叠）。</li><li>每个元素取值范围 [0, blockShapeY]。</li><li>当actualBlockLen = 0时：对应block的k_cmp填0向量， 不会被topK选中。</li><li>当actualBlockLen > 0时：仅对前actualBlockLen个token取均值。</li></ul></li>
    <li>如不配置（传nullptr）：对key进行完整压缩（使用完整blockShapeY长度）。</li></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>queryLayout（char*）</td>
    <td>输入</td>
    <td>query的数据排布格式。指示输入张量在内存中的具体排布。</td>
    <td>当前仅支持 "TND"、"BNSD"，queryLayout与keyLayout需要保持一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>keyLayout（char*）</td>
    <td>输入</td>
    <td>key的数据排布格式。指示输入张量在内存中的具体排布。</td>
    <td>当前仅支持 "TND"、"BNSD"，queryLayout与keyLayout需要保持一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyHeads（int64_t）</td>
    <td>输入</td>
    <td>key的注意力头数。</td>
    <td>当前仅支持MHA，numKeyHeads必须等于numHeads。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue（double）</td>
    <td>输入</td>
    <td>缩放系数，即公式中的scale。用于注意力分数的归一化处理。</td>
    <td>一般设置为1 / sqrt(D)。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparsity（double）</td>
    <td>输入</td>
    <td>稀疏度保留比例。指定公式中attn_score中需要保留的块位置占全部块位置的比例。</td>
    <td>取值范围 (0.0, 1.0)。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>blockSparseMaskOut（aclTensor*）</td>
    <td>输出</td>
    <td>块状稀疏掩码输出，表示根据Q/K内容自适应生成的稀疏pattern。可直接作为BSA算子的blockSparseMask输入。</td>
    <td>不支持空Tensor。<br>shape为 [B, N, Xblocks, Yblocks]：<ul><li>Xblocks = ceilDiv(maxQSeqlen, blockShapeX)。</li><li>Yblocks = ceilDiv(maxKSeqlen, blockShapeY)。</li><li>值为1表示该block参与注意力计算，值为0表示不参与。</li></ul></td>
    <td>INT8</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
  </tbody>
  </table>

## 约束说明

* actualSeqLengths在queryLayout为 "TND" 时必选；actualSeqLengthsK在keyLayout为 "TND" 时必选。
* 根据算子支持的输入Layout，query张量Shape中对应的head维度大小记为N1，key张量Shape中对应的head维度大小记为N2。必须满足N1 = N2（仅支持MHA）。
* headDim = 128。
* blockShapeX和blockShapeY必须为64的倍数。
* query和key压缩后，query和key对应的Xblocks和Yblocks需满足Xblocks * Yblocks > 1。
* query和key的数据类型必须一致，仅支持FLOAT16和BFLOAT16。
* blockSparseMaskOut数据类型为INT8（二值：0或1）。
* postBlockShape当前不支持，必须传入nullptr。
* actualBlockLenQuery / actualBlockLenKey若非null，每个元素取值范围 [0, blockShapeX] / [0, blockShapeY]；为null时完整压缩。

## 调用说明

| 调用方式           | 调用样例                                                                                                          | 说明                                                                                                                  |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_bsa_select_block_mask](examples/test_aclnn_bsa_select_block_mask.cpp)                     | 只支持MHA和格式为TND\BNSD的场景，通过[aclnnBSASelectBlockMask](docs/aclnnBSASelectBlockMask.md)接口方式调用BSASelectBlockMask算子。                   |
