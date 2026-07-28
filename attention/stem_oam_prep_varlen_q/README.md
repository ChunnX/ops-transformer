# StemOamPrepVarlenQ

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 仅支持 arch35 (DAV_3510) 架构芯片。

## 功能说明

- 算子功能：完成 Stem OAM block-sparse attention 中 Q 侧预处理计算。将变长 Q tensor 从 paged 存储格式转化为按 stem block (128 token) 分组的 flattened qFlat 输出，供后续 OAM score 计算。

- 计算公式：

  阶段1 Scale Fusion:
  $$q\_scale[b, h, pos] = qscale[b, h, pos]$$

  阶段2 De-page Varlen:
  $$Q\_dense[b] = Cast(q[cu\_seqlens\_q[b]:cu\_seqlens\_q[b]+q\_len[b], :, :], \text{FP32})$$

  阶段3 Weighted Group Sum (自然顺序，NO flip):
  $$Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_blocks[b,h,qb,r,g,:] \times q\_scale[b,h,position(qb,r,g)]$$

  阶段4 Flatten:
  $$qflat[b, h, qb, g \times D : (g+1) \times D] = Q\_group\_sum[b, h, qb, g, :]$$

  阶段5 Cast输出:
  $$qflat\_out = qflat.to(\text{BF16})$$

- 关键特性：Q 侧 stride 维度为自然顺序（不翻转），与 K 侧的翻转处理不同。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|-----------|------|----------|----------|
| q | 输入 | 表示变长Q tensor，所有batch的token拼接存储。shape为[total_tokens, H_q, D]，D=128。 | FLOAT8_E4M3FN | ND |
| qSeqLens | 输入 | 表示每个batch的Q序列长度。shape为[batch]，取值范围(0, 1024]。 | INT64 | - |
| cuSeqLensQ | 输入 | 表示Q的累积序列长度偏移量，用于varlen索引。shape为[batch+1]，cuSeqLensQ[0]必须为0，cuSeqLensQ[batch]必须等于total_tokens，单调递增。 | INT64 | - |
| qScale | 输入 | 表示Q的per-token scale factor。q为FP8时必填。shape为[total_tokens, H_q]。 | FLOAT | ND |
| stemBlockSize | 属性 | 表示stem block大小，对应公式中B。仅支持128。 | INT64 | - |
| stemStride | 属性 | 表示stem stride大小，对应公式中S。仅支持16。 | INT64 | - |
| qFlat | 输出 | 表示flattened Q输出，供OAM score计算使用。shape为[batch, H_q, max_Qb, kflat_dim]，其中max_Qb=ceil(max(qSeqLens)/stemBlockSize)，kflat_dim=stemStride×D=2048。 | BFLOAT16 | ND |

## 约束说明

- 确定性计算：默认确定性实现。
- q的最后一维D必须等于128。
- stemBlockSize仅支持128，stemStride仅支持16。派生值：R = stemBlockSize / stemStride = 8, kflat_dim = stemStride × D = 2048。
- q仅支持FLOAT8_E4M3FN数据类型，qScale必填且数据类型为FLOAT，qFlat输出固定为BFLOAT16。
- qSeqLens长度取值范围为(0, 1024]。
- 当qSeqLens中某batch值为0时，该batch对应的qFlat输出填充为零。
- Q侧stride维度为自然顺序（g ∈ [0, S-1]），不翻转（与K侧处理不同）。
- 支持空Tensor输入：当q为空或qSeqLens长度为0时，直接返回。

## 调用说明

| 调用方式  | 样例代码 | 说明 |
| --------- | -------- | ---- |
| aclnn接口 | - | 通过[aclnnStemOamPrepVarlenQ](./docs/aclnnStemOamPrepVarlenQ.md)调用StemOamPrepVarlenQ算子 |
| PyTorch API | - | 通过[stem_oam_prep_varlen_q](../../torch_extension/cann_ops_transformer/docs/zh/stem_oam_prep_varlen_q.md)调用StemOamPrepVarlenQ算子 |
