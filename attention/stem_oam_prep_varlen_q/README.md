# StemOamPrepVarlenQ

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- **算子功能**：推理场景，Stem OAM block-sparse attention 中 Q 侧预处理计算。将变长 Q tensor 从 paged 存储格式转化为按 stem block (stemBlockSize token) 分组的 flattened qFlat 输出，供后续 OAM score 计算。主要计算过程如下：
    - 首先对 Q tensor 进行 Scale Fusion，将 per-token scale factor 应用到 Q 上
    - 然后根据 cuSeqLensQ 进行 De-page Varlen，从 paged 存储中提取变长 Q tensor
    - 对 Q 进行 Weighted Group Sum（自然顺序，不翻转 stride 维度）
    - Flatten 输出为 qFlat：shape 为 [batch, H_q, max_Qb, kflat_dim]
    - Cast 输出为 BFLOAT16 格式
- **计算公式**：

    阶段1 Scale Fusion

    $$q\_scale[b, h, pos] = qscale[b, h, pos]$$

    阶段2 De-page Varlen

    $$Q\_dense[b] = Cast(q[cu\_seqlens\_q[b]:cu\_seqlens\_q[b]+q\_len[b], :, :], \text{FP32})$$

    阶段3 Weighted Group Sum (自然顺序，NO flip)

    $$Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_blocks[b,h,qb,r,g,:] \times q\_scale[b,h,position(qb,r,g)]$$

    阶段4 Flatten

    $$qflat[b, h, qb, g \times D : (g+1) \times D] = Q\_group\_sum[b, h, qb, g, :]$$

    阶段5 Cast输出

    $$qflat\_out = qflat.to(\text{BF16})$$

- **关键特性**：Q 侧 stride 维度为自然顺序（g ∈ [0, S-1]），不翻转（与 K 侧处理不同）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1427px"><colgroup>
<col style="width: 150px">
<col style="width: 100px">
<col style="width: 650px">
<col style="width: 180px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>q</td>
    <td>输入</td>
    <td>Device侧的aclTensor，变长 Q tensor，所有 batch 的 token 拼接存储。shape为[total_tokens, H_q, D]，其中 D 必须等于 stemBlockSize（推荐默认 128）。</td>
    <td>FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qScale</td>
    <td>可选输入</td>
    <td>Device侧的aclTensor，Q 的 per-token scale factor，shape为[total_tokens, H_q]。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qSeqLens</td>
    <td>输入</td>
    <td>Device侧的aclTensor，每个 batch 的 Q 序列长度，shape为[batch]。长度等于 batch，每个值≥0，不支持空Array。batch最大值为1024。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cuSeqLensQ</td>
    <td>输入</td>
    <td>Device侧的aclTensor，Q 的累积序列长度偏移量，用于 varlen 索引。shape为[batch+1]。cuSeqLensQ[0]必须为0，cuSeqLensQ[batch]必须等于total_tokens，单调递增。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stemBlockSize</td>
    <td>属性</td>
    <td>stem block 大小，控制每个 stem block 的 token 数量，决定 Q Processing 的分组粒度。推荐默认值128。必须是32的倍数，最大值为256，必须满足 stemStride≤stemBlockSize。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stemStride</td>
    <td>属性</td>
    <td>stem stride 大小，控制 stem block 内 stride group 的 token 数量，决定 qFlat 的维度粒度。推荐默认值16。必须是16的倍数，最大值为64，必须满足 stemStride≤stemBlockSize。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>qFlat</td>
    <td>输出</td>
    <td>Device侧的aclTensor，flattened Q 输出，供 OAM score 计算使用。shape为[batch, H_q, max_Qb, kflat_dim]。其中 max_Qb=ceil(max(qSeqLens)/stemBlockSize)，kflat_dim=stemStride×D（默认 16×128=2048）。数据类型类型固定为BFLOAT16。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- **确定性说明**：aclnnStemOamPrepVarlenQ 默认确定性实现。
- **架构支持**：仅支持 arch35 (Ascend 950PR/Atlas A3) 架构，不支持 arch22。
- **shape格式字段含义及约束**
    - total_tokens：total_tokens表示所有 batch 的 token 总数
    - H_q：H_q表示 Q 多头数，取值范围：1~128
    - stemBlockSize：stem block 大小，取值范围：32~256，为32的倍数
    - stemStride：stem stride 大小，取值范围：16~64，为16的倍数，stemBlockSize / stemStride结果必须为整数
    - D：Q 矩阵最后一维，必须等于 stemBlockSize（推荐默认 128）
    - max_Qb：最大 Q block 数，max_Qb=ceil(max(qSeqLens)/stemBlockSize)
    - kflat_dim：qFlat 输出维度，kflat_dim=stemStride×D（默认 16×128=2048）
- **stride维度约束**：Q 侧 stride 维度为自然顺序（g ∈ [0, S-1]），不翻转（与 K 侧处理不同）
- **空值处理**：当 qSeqLens 中某 batch 值为 0 时，该 batch 对应的 qFlat 输出填充为零
- **无条件 Group Sum**：当 qScale 传入空指针时，Kernel 内部 Q Processing 执行无权重 Group Sum (Σ_r Q_blocks[:, r, :, :])，而非加权 Group Sum (Σ_r Q_blocks×qScale)。
- **FP8输入约束**：FP8输入时，qScale 最后一维需要能覆盖 max(qSeqLens) 个位置（padding 到 stemBlockSize 对齐）

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_stem_oam_prep_varlen_q](./examples/test_aclnn_stem_oam_prep_varlen_q.cpp) | 通过[aclnnStemOamPrepVarlenQ](./docs/aclnnStemOamPrepVarlenQ.md)调用StemOamPrepVarlenQ算子 |
| 图模式 | [test_geir_stem_oam_prep_varlen_q](./examples/test_geir_stem_oam_prep_varlen_q.cpp) | 通过[算子IR](./op_graph/stem_oam_prep_varlen_q_proto.h)调用StemOamPrepVarlenQ算子 |
| torch接口 | [test_torch_stem_oam_prep_varlen_q](./examples/test_torch_stem_oam_prep_varlen_q.py) | 通过`torch.ops.cann_ops_transformer.npu_stem_oam_prep_varlen_q`调用StemOamPrepVarlenQ算子 |
