# BSASelectBlockMask

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品</term>|      √     |
|<term>Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas A2 推理系列产品</term>|      ×     |
|<term> Atlas 200I/500 A2 推理产品</term>                                         |    ×    |
|<term> Atlas 推理系列产品</term>                                                 |    ×    |
|<term> Atlas 训练系列产品</term>                                                 |    ×    |

## 功能说明

* ​接口功能​：aclnnBSASelectBlockMask 是 BSA（BlockSparseAttention）的前置算子，负责根据 Query 和 Key 的内容动态生成 blockSparseMask，使 BSA 的调用链从“手动提供掩码”变为“根据 Q/K 内容自适应选择稀疏模式”。

* ​计算公式​：

  设 blockShapeOptional = [blockShapeX, blockShapeY]，压缩后块数：

  $$
  Xblocks = \lceil Sq / blockShapeX \rceil,\quad Yblocks = \lceil Skv / blockShapeY \rceil
  $$

  **Step1：均值池化压缩 (Mean Pooling Compression)**

  当 actualBlockLenQueryOptional / actualBlockLenKeyOptional 为 null 时（完整压缩）：

  $$
  q\_compressed[b, n, x, d] = \frac{1}{blockShapeX} \sum_{i=0}^{blockShapeX-1} query[b, n, x \cdot blockShapeX + i, d]
  $$

  $$
  k\_compressed[b, n, y, d] = \frac{1}{blockShapeY} \sum_{j=0}^{blockShapeY-1} key[b, n, y \cdot blockShapeY + j, d]
  $$

  当 actualBlockLenQueryOptional / actualBlockLenKeyOptional 非 null 时（部分压缩），仅对每个 block 内前 actualBlockLen 个 token 取均值：

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
  attn\_score[b, n, x, y] = softmax(score[b, n, x, :])[y] = \frac{\exp(score[b, n, x, y] - m_{final})}{l_{final}}
  $$

  **Step3：TopK 选择生成索引**

  $$
  topk\_value = \text{round}(sparsity \times Xblocks \times Yblocks)
  $$

  $$
  \mathcal{indices}= \text{TopK}\left(attn\_score[b, n, x, y],\; topK\_value\right)
  $$

  其中 indices 为 attn\_score[b, n, x, y] 中 topk\_value 个最大值对应的索引集合。

  **Step4：生成 BlockSparseMask**
  $$
  blockSparseMaskOut[b, n, x, y] =
  \begin{cases}
  1 & (b, n, x, y) \in \mathcal{indices} \\
  0 & (b, n, x, y) \notin \mathcal{indices}
  \end{cases}
  $$

BSASelectBlockMask 输入 query、key 的数据排布格式支持从多种维度排布解读，可通过 queryLayout 和 keyLayout 传入。为了方便理解后续支持的具体排布格式（如 BNSD、TND 等），此处先对排布格式中各缩写字母所代表的维度含义进行统一说明：

* B：表示输入样本批量大小（Batch）
* T：B 和 S 合轴紧密排列的长度（Total tokens）
* S：表示输入样本序列长度（Seq-Length）
* H：表示隐藏层的大小（Head-Size）
* N：表示多头数（Head-Num）
* D：表示隐藏层最小的单元尺寸，需满足 D = H / N（Head-Dim）

当前支持的布局：

* queryLayout: "TND" "BNSD"
* keyLayout: "TND" "BNSD"

## 参数说明

<table style="undefined;table-layout: fixed; width: 1550px">
<colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
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
    </tr>
</thead>
<tbody>
    <tr>
    <td>query（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的查询向量，即公式中的 query。</td>
    <td>不支持空 Tensor。<br>支持的 shape 为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的键向量，即公式中的 key。</td>
    <td>不支持空 Tensor。<br>支持的 shape 为：<ul><li>TND: [totalKTokens, numKeyHeads, headDim]。</li><li>BNSD: [batch, numKeyHeads, maxKSeqLength, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td rowspan="3">blockShapeOptional（aclIntArray*）</td>
    <td rowspan="3">输入</td>
    <td rowspan="3">稀疏块形状数组。指定每个稀疏块的二维尺寸（行数和列数）。</td>
    <td><ul><li>如配置此输入，算子会从中获取稀疏块尺寸；如不配置此输入，算子将默认稀疏块尺寸为 [128, 128]。</li></ul></td>
    <td rowspan="3">INT64</td>
    <td rowspan="3">-</td>
    <td rowspan="3">1</td>
    <td rowspan="3">-</td>
    </tr>
    <tr>
    <td>当配置此输入时的元素要求：<ul><li>必须包含至少两个元素 [blockShapeX, blockShapeY]。</li><li>blockShapeX: Q 方向块大小，必须为 64 的倍数且大于 0。</li><li>blockShapeY: KV 方向块大小，必须为 64 的倍数且大于 0。</li></ul></td>
    </tr>
    <tr>
    <td>如不配置（传 nullptr），算子将默认 blockShapeX = 128，blockShapeY = 128（128 为 64 的倍数，满足约束）。</td>
    </tr>
    <tr>
    <td>postBlockShapeOptional（aclIntArray*）</td>
    <td>输入</td>
    <td>预留参数，用于 Softmax 后二次压缩。</td>
    <td>当前不支持，必须传入 nullptr。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td rowspan="2">actualSeqLengthsOptional（aclIntArray*）</td>
    <td rowspan="2">输入</td>
    <td rowspan="2">query 的实际序列长度数组。<br>用于描述变长序列场景下（即含有 Padding 填充数据的场景），每个 Batch 中实际有效的 query token 数量。</td>
    <td>变长序列场景（当 queryLayout 为 “TND” 时）：该项输入必须配置。因为 TND 格式为一维连续排布，算子需要依赖该数组来准确切分界定各个序列的真实边界。</td>
    <td rowspan="2">INT64</td>
    <td rowspan="2">-</td>
    <td rowspan="2">1</td>
    <td rowspan="2">-</td>
    </tr>
    <tr>
    <td>定长/变长场景（当 queryLayout 为 “BNSD” 时）：<ul><li>如配置该项，算子会按指定的有效长度处理，忽略 Padding 部分的数据，提升性能；</li><li>如不配置（传 nullptr），算子将默认把 query shape 中的 S 维度作为有效长度进行全量处理。</li></ul></td>
    </tr>
    <tr>
    <td rowspan="2">actualSeqLengthsKOptional（aclIntArray*）</td>
    <td rowspan="2">输入</td>
    <td rowspan="2">key 的实际序列长度数组。<br>用于描述变长序列场景下（即含有 Padding 填充数据的场景），每个 Batch 中实际有效的 key token 数量。</td>
    <td>变长序列场景（当 keyLayout 为 “TND” 时）：该项输入必须配置。因为 TND 格式为一维连续排布，算子需要依赖该数组来准确切分界定各个序列的真实边界。</td>
    <td rowspan="2">INT64</td>
    <td rowspan="2">-</td>
    <td rowspan="2">1</td>
    <td rowspan="2">-</td>
    </tr>
    <tr>
    <td>定长/变长场景（当 keyLayout 为 “BNSD” 时）：<ul><li>如配置该项，算子会按指定的有效长度处理，忽略 Padding 部分的数据，提升性能；</li><li>如不配置（传 nullptr），算子将默认把 key shape 中的 S 维度作为有效长度进行全量处理。</li></ul></td>
    </tr>
    <tr>
    <td rowspan="2">actualBlockLenQueryOptional（aclIntArray*）</td>
    <td rowspan="2">输入</td>
    <td rowspan="2">每个 query block 内实际压缩的有效 seq 长度。<br>用于部分压缩场景（如末尾不完整块或仅压缩有效 token）。</td>
    <td>可选输入：<ul><li>BNSD 场景：shape 为 [B, Xblocks]。</li><li>TND 场景：shape 为 [TotalBlockNum_Q]（各 batch 的 Xblocks 堆叠）。</li><li>每个元素取值范围 [0, blockShapeX]。</li><li>当 actualBlockLen = 0 时：对应 block 的 q_cmp 填 0 向量，不会被 topK 选中。</li><li>当 actualBlockLen > 0 时：仅对前 actualBlockLen 个 token 取均值。</li></ul></td>
    <td rowspan="2">INT64</td>
    <td rowspan="2">-</td>
    <td rowspan="2">1</td>
    <td rowspan="2">-</td>
    </tr>
    <tr>
    <td>如不配置（传 nullptr）：对 query 进行完整压缩（使用完整 blockShapeX 长度）。</td>
    </tr>
    <tr>
    <td rowspan="2">actualBlockLenKeyOptional（aclIntArray*）</td>
    <td rowspan="2">输入</td>
    <td rowspan="2">每个 key block 内实际压缩的有效 seq 长度。<br>用于部分压缩场景（如末尾不完整块或仅压缩有效 token）。</td>
    <td>可选输入：<ul><li>BNSD 场景：shape 为 [B, Yblocks]。</li><li>TND 场景：shape 为 [TotalBlockNum_KV]（各 batch 的 Yblocks 堆叠）。</li><li>每个元素取值范围 [0, blockShapeY]。</li><li>当 actualBlockLen = 0 时：对应 block 的 k_cmp 填 0 向量，不会被 topK 选中。</li><li>当 actualBlockLen > 0 时：仅对前 actualBlockLen 个 token 取均值。</li></ul></td>
    <td rowspan="2">INT64</td>
    <td rowspan="2">-</td>
    <td rowspan="2">1</td>
    <td rowspan="2">-</td>
    </tr>
    <tr>
    <td>如不配置（传 nullptr）：对 key 进行完整压缩（使用完整 blockShapeY 长度）。</td>
    </tr>
    <tr>
    <td>queryLayout（char*）</td>
    <td>输入</td>
    <td>query 的数据排布格式。指示输入张量在内存中的具体排布。</td>
    <td>当前仅支持 “TND”、“BNSD”，queryLayout 与 keyLayout 需要保持一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>keyLayout（char*）</td>
    <td>输入</td>
    <td>key 的数据排布格式。指示输入张量在内存中的具体排布。</td>
    <td>当前仅支持 “TND”、“BNSD”，queryLayout 与 keyLayout 需要保持一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyHeads（int64_t）</td>
    <td>输入</td>
    <td>key 的注意力头数。用于支持 GQA（分组查询注意力）机制下的头数比例映射。</td>
    <td>当前仅支持 MHA，numKeyHeads 必须等于 numHeads。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue（double）</td>
    <td>输入</td>
    <td>缩放系数，即公式中的 scale。用于注意力分数的归一化处理。</td>
    <td>一般设置为 1 / sqrt(D)。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparsity（double）</td>
    <td>输入</td>
    <td>稀疏度保留比例。指定公式中 attn_score 中需要保留的块位置占全部块位置的比例。</td>
    <td>取值范围 (0.0, 1.0)。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>blockSparseMaskOut（aclTensor*）</td>
    <td>输出</td>
    <td>块状稀疏掩码输出，表示根据 Q/K 内容自适应生成的稀疏 pattern。可直接作为 BSA 算子的 blockSparseMaskOptional 输入。</td>
    <td>不支持空 Tensor。<br>shape 为 [B, N, Xblocks, Yblocks]：<ul><li>Xblocks = ceilDiv(maxQSeqlen, blockShapeX)。</li><li>Yblocks = ceilDiv(maxKSeqlen, blockShapeY)。</li><li>值为 1 表示该 block 参与注意力计算，值为 0 表示不参与。</li></ul></td>
    <td>INT8</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize（uint64_t*）</td>
    <td>输出</td>
    <td>返回需要在 Device 侧申请的 workspace 大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor（aclOpExecutor**）</td>
    <td>输出</td>
    <td>返回 op 执行器，包含了算子计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
</tbody>
</table>

## 约束说明

* 该接口与 PyTorch 配合使用时，需要保证 CANN 相关包与 PyTorch 相关包的版本匹配。
* actualSeqLengthsOptional 在 queryLayout 为 “TND” 时必选；actualSeqLengthsKOptional 在 keyLayout 为 “TND” 时必选。
* 根据算子支持的输入 Layout，query 张量 Shape 中对应的 head 维度大小记为 N1，key 张量 Shape 中对应的 head 维度大小记为 N2。必须满足 N1 = N2（仅支持 MHA）。
* headDim = 128。
* blockShapeX 和 blockShapeY 必须为 64 的倍数。
* query 和 key 压缩后，对应的 Xblocks 和 Yblocks 需满足 Xblocks * Yblocks > 1。
* query 和 key 的数据类型必须一致，仅支持 FLOAT16 和 BFLOAT16。
* blockSparseMaskOut 数据类型为 INT8（二值：0 或 1）。
* postBlockShapeOptional 当前不支持，必须传入 nullptr。
* actualBlockLenQueryOptional / actualBlockLenKeyOptional 若非 null，每个元素取值范围 [0, blockShapeX] / [0, blockShapeY]；为 null 时完整压缩。

## 调用说明

| 调用方式           | 调用样例                                                                                                          | 说明                                                                                                                  |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_bsa_select_block_mask](examples/test_aclnn_bsa_select_block_mask.cpp)                     | 只支持MHA和格式为TND\BNSD的场景，通过[aclnnBSASelectBlockMask](docs/aclnnBSASelectBlockMask.md)接口方式调用BSASelectBlockMask算子。                   |
