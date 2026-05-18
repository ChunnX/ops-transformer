# aclnnFlashAttentionScoreV4 约束补充说明

本文档记录了arch35代码实现中发现、且参数表格中未明确说明的约束项。

## 一、约束分类说明

| 分类 | 说明 | 示例 |
|------|------|------|
| A类 | 参数表格已明确说明的约束 | sinkOptional数据类型FLOAT32、shape[N1]已在表格第250-253行说明 |
| B类 | 表格未说明的组合约束或行为后果 | queryRope和keyRope必须同时存在/不存在，表格未说明此组合要求 |
| C类 | 文档与代码不一致 | dScaleV shape分块大小文档写256，代码实际512 |

**注意：A类约束不在本文档记录范围，代码检查是配合表格约束的实现。**

## 二、B类约束（表格未明确说明）

以下约束在代码中有检查，但参数表格中未明确说明：

### 2.1 需补充约束总览

| 序号 | 约束类型 | 约束内容 | 代码位置 |
|------|----------|----------|----------|
| 1 | Rope组合 | queryRope和keyRope必须同时存在或不存在 | tiling_regbase.cpp:234-237 |
| 2 | MLA约束 | MLA concat仅支持dSize=128, dSizeRope=64 | tiling_regbase.cpp:523-526 |
| 3 | GQA约束 | N1必须是N2的整数倍 | tiling_regbase.cpp:263-265 |
| 4 | Batch约束 | query/key/value的batch必须相等 | tiling_regbase.cpp:384-394,452-458等 |
| 5 | TND约束 | actualSeqQLen和actualSeqKvLen长度相等 | tiling_regbase.cpp:324-325 |
| 6 | TND约束 | T必须大于等于seqLen总和 | tiling_regbase.cpp:329-335 |
| 7 | AttenMask压缩 | 压缩模式shape必须为[2048,2048] | tiling_regbase.cpp:699-701 |
| 8 | AttenMask压缩 | Prefix压缩模式shape必须为[3072,2048] | tiling_regbase.cpp:703-708 |
| 9 | SparseMode行为 | sparseMode=2时要求S1<=S2 | tiling_varlen.cpp:319-325 |
| 10 | SparseMode行为 | sparseMode=6时prefix必须传入 | tiling_regbase.cpp:1632-1636 |
| 11 | Band约束 | preTokens和nextTokens均<0时报错 | tiling_regbase.cpp:1268-1272 |
| 12 | Band约束 | preTokens和nextTokens必须有交集 | tiling_regbase.cpp:1285-1303 |
| 13 | StartIdx约束 | kvStartIdx-qStartIdx必须在int32范围 | tiling_regbase.cpp:1713-1715 |
| 14 | HIFLOAT8组合 | 不支持可选输入+仅BSND布局+特定shape组合 | tiling_regbase.cpp:511-521 |
| 15 | PSE组合约束 | pseType=2/3时Sq必须等于Skv | tiling_regbase.cpp:635-640 |
| 16 | 参数范围行为 | preTokens/nextTokens超int32自动调整到边界值 | tiling_regbase.cpp:141-158 |

### 2.2 详细说明

#### B-1: queryRope和keyRope必须同时存在或不存在
**表格现状**: 分别说明了queryRope和keyRope的shape，但未说明必须同时传入。
```cpp
// tiling_regbase.cpp:234-237
if (hasQueryRope ^ hasKeyRope) {
    OP_LOGE(opName, "query_rope and key_rope should be present or absent at the same time.");
    return false;
}
```

#### B-2: MLA concat特定参数要求
**表格现状**: 未说明MLA场景的特定参数要求。
```cpp
// tiling_regbase.cpp:523-526
if (hasRope && (dSize != 128 || dSizeRope != 64)) {
    OPS_REPORT_VECTOR_INNER_ERR(opName, "MLA concat only support dSize=128, dSizeRope=64.");
    return ge::GRAPH_FAILED;
}
```

#### B-3: N1必须是N2的整数倍（GQA约束）
**表格现状**: 未说明N1和N2的数量关系约束。
```cpp
// tiling_regbase.cpp:263-265
OP_CHECK_IF(n1Size % n2Size != 0,
           OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld]"),
           return false);
```

#### B-4: query/key/value的batch必须相等
**表格现状**: 约束说明第568行有"batchsize必须相等"，但表格参数说明未明确。
```cpp
// tiling_regbase.cpp:384-394 (BSH布局)
OP_CHECK_IF(bSize != keyShape.GetDim(0),
           OPS_REPORT_VECTOR_INNER_ERR(opName, "query batchsize[%ld] should be same as key batchsize[%ld]."),
           return false);
```

#### B-5: TND actualSeqQLen和actualSeqKvLen长度相等
**表格现状**: 未明确说明两者长度必须相等。
```cpp
// tiling_regbase.cpp:324-325
OP_CHECK_IF(actualSeqQLen != actualSeqKVLen,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "VarLen scene, q is not equal kv."), return false);
```

#### B-6: TND T必须大于等于seqLen总和
**表格现状**: 未明确T与seqLen总和的关系。
```cpp
// tiling_regbase.cpp:329-335
OP_CHECK_IF(t1Size < accumS1 || t2Size < accumS2,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "Query T need larger than sum of seqLen."),
            return false);
```

#### B-7: AttenMask压缩模式shape必须为[2048,2048]
**表格现状**: 表格shape列为[S,S]，未说明压缩模式需要特定shape。
```cpp
// tiling_regbase.cpp:699-701
if (shape != ATTEN_MASK_COMPRESS_LIMIT || shape != ATTEN_MASK_COMPRESS_LIMIT) {  // 2048
    OP_LOGE(context_, "In the attenmask compression, please set the atten_mask_shape to [2048,2048].");
    return false;
}
```

#### B-8: Prefix AttenMask压缩shape必须为[3072,2048]
**表格现状**: 未说明prefix压缩模式的特定shape要求。
```cpp
// tiling_regbase.cpp:703-708
if (attenMaskCompressMode == PREFIX_MODE && shape != [3072,2048]) {
    OP_LOGE(context_, "In the prefix attenmask compression, please set the atten_mask_shape to [3072,2048].");
    return false;
}
```

#### B-9: sparseMode=2时要求S1<=S2
**表格现状**: 表格只列出sparseMode支持0~6，未说明各模式的具体行为约束。
```cpp
// tiling_varlen.cpp:319-325
for (int64_t i = 0L; i < bSize; ++i) {
    if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
        OP_LOGE(context_, "Batch[%ld] s1 > s2, exist invalid row.");
        return false;
    }
}
```

#### B-10: sparseMode=6时prefix必须传入
**表格现状**: 约束说明第625行有说明，表格参数说明未明确。
```cpp
// tiling_regbase.cpp:1632-1636
if (prefixN == nullptr && sparseMode == PREFIX_COMPRESS) {
    OP_LOGE(context_, "prefixN must be provided when sparseMode=6 (PREFIX_COMPRESS).");
    return SparseEnum::ALL;
}
```

#### B-11: preTokens和nextTokens均<0时报错
**表格现状**: 未说明此边界情况的处理。
```cpp
// tiling_regbase.cpp:1268-1272
if (preTokens < 0 && nextTokens < 0) {
    OP_LOGE(context_, "PreTokens and nextTokens config error, there is no valid data block.");
    return false;
}
```

#### B-12: preTokens和nextTokens必须有交集（band模式）
**表格现状**: 约束说明第599行有说明，但未说明代码具体检查逻辑。
```cpp
// tiling_regbase.cpp:1285-1303
// 检查preTokens和nextTokens之间是否有有效数据块
```

#### B-13: kvStartIdx-qStartIdx必须在int32范围
**表格现状**: 未说明此范围约束。
```cpp
// tiling_regbase.cpp:1713-1715
OP_CHECK_IF(kvStartIdx - qStartIdx > INT32_MAX || kvStartIdx - qStartIdx < INT32_MIN,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "kvStartIdx - qStartIdx should in int32 range."),
            return ge::GRAPH_FAILED);
```

#### B-14: HIFLOAT8组合约束
**表格现状**: 表格只说明HIFLOAT8支持，未说明其特殊组合约束。
```cpp
// tiling_regbase.cpp:511-521
OP_CHECK_IF(inputDtype == DT_HIFLOAT8 && 
            (hasAttenMask || hasPse || hasDropOut || hasRope || layout != BSND || 特定shape),
            OPS_REPORT_VECTOR_INNER_ERR(opName, "HIFLOAT8约束..."), return ge::GRAPH_FAILED);
```

#### B-15: pseType=2/3时Sq必须等于Skv
**表格现状**: 约束说明第594行有说明。
```cpp
// tiling_regbase.cpp:635-640
OP_CHECK_IF(s1Size != s2Size,
           OPS_REPORT_VECTOR_INNER_ERR(opName, "pseType=2 or 3 only support Sq == Skv."),
           return false);
```

#### B-16: preTokens/nextTokens超int32范围自动调整
**表格现状**: 未说明此自动调整行为（warning而非error）。
```cpp
// tiling_regbase.cpp:141-158
if (preTokens > std::numeric_limits<int32_t>::max()) {
    OP_LOGW(context_, "preTokens will be reset to max int value.");
    preTokens = std::numeric_limits<int32_t>::max();
}
```

## 三、C类约束（文档与代码不一致）

| 序号 | 问题 | 代码实际值 | 文档描述 | 处理建议 |
|------|------|------------|----------|----------|
| 1 | dScaleV shape分块大小 | ceil(S2/512) | Ceil(Skv/256) | 修正文档表格第239行 |

```cpp
// tiling_regbase.cpp:866-870
OP_CHECK_IF(dimValue2 != (s2Size + QUANT_V_BLOCK_SIZE - 1) / QUANT_V_BLOCK_SIZE,  // QUANT_V_BLOCK_SIZE=512
            OPS_REPORT_VECTOR_INNER_ERR(opName, "only support [B, N2, ceil(S2/512), 1]"),
            return false);
```

## 四、A类约束（表格已说明，仅供参考）

以下约束在参数表格中已明确说明，代码检查是配合表格的实现：

| 参数 | 表格说明位置 | 代码检查位置 |
|------|--------------|--------------|
| sinkOptional数据类型FLOAT32 | 第250行 | tiling_regbase.cpp:902-905 |
| sinkOptional shape [N1] | 第252行 | tiling_regbase.cpp:918-923 |
| attenMaskOptional数据类型BOOL/UINT8 | 第190行 | tiling_regbase.cpp:720-723 |
| attenMaskOptional维度2/4维 | 第192行shape列 | tiling_regbase.cpp:730-740 |
| dropMaskOptional数据类型UINT8 | 第180行 | tiling_regbase.cpp:762-765 |
| dScaleQ/K/V数据类型FLOAT32 | 第220/230/240行 | tiling_regbase.cpp:807-870 |
| softmaxMax/Sum数据类型FLOAT | 第440/450行 | GetShapeAttrsInfo输出检查 |
| query/key/value数据类型一致 | 第129/139/149行"数据类型一致" | tiling_regbase.cpp:84-98 |
| realShiftOptional数据类型/shape | 第160/162行 | tiling_regbase.cpp:614-642 |
| queryRopeOptional数据类型FLOAT16/BF16 | 第200行 | tiling_regbase.cpp:234-260 |

## 五、总结

| 分类 | 数量 | 处理状态 |
|------|------|----------|
| B类（需补充到约束说明） | 16项 | 已在上次修改中补充到文档 |
| C类（需修正文档） | 1项 | dScaleV shape分块大小需修正 |
| A类（表格已说明） | 多项 | 无需补充到约束说明部分 |

**关键结论**: 参数表格中已明确说明的约束（数据类型、shape、维度等）不应重复补充到约束说明部分，约束说明部分应聚焦于表格未说明的组合约束、行为后果、范围边界处理等。