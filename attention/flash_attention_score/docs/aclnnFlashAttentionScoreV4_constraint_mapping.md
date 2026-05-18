# aclnnFlashAttentionScoreV4 约束补充说明

本文档记录了arch35代码实现中的约束检查逻辑与文档的对应关系分析。

## 一、约束分类说明

根据分析结果，约束可分为以下三类：

| 分类 | 说明 | 处理方式 |
|------|------|----------|
| A类 | 参数表格已说明的约束 | 代码有检查，文档表格已说明，无需补充到约束说明部分 |
| B类 | 约束说明部分缺失的约束 | 需要补充到文档"约束说明"部分 |
| C类 | 文档与代码不一致 | 需要修正文档或代码 |

## 二、A类约束（参数表格已说明）

以下约束在参数说明表格中已有明确说明，代码中的检查是配合表格约束的实现：

| 序号 | 约束内容 | 表格位置 | 代码位置 |
|------|----------|----------|----------|
| 1 | query/key/value数据类型一致 | 第129、139、149行"数据类型与xxx一致" | AnalyzeDtype():84-98 |
| 2 | realShiftOptional支持FLOAT16/BF16/FLOAT32 | 第160行 | AnalyzePseOptionalInput():614-642 |
| 3 | realShiftOptional维度为4维或[B,N]/[N] | 第162行shape列 | AnalyzePseOptionalInput():577-628 |
| 4 | attenMaskOptional支持BOOL/UINT8 | 第190行 | AnalyzeAttenOptionalInput():720-723 |
| 5 | attenMaskOptional维度为2或4维 | 第192行shape列 | AnalyzeAttenOptionalInput():730-740 |
| 6 | dropMaskOptional支持UINT8 | 第180行 | AnalyzeDropOptionalInput():762-765 |
| 7 | dScaleQOptional支持FLOAT32 | 第220行 | AnalyzeFp8OptionalInput():807-810 |
| 8 | dScaleKOptional支持FLOAT32 | 第230行 | AnalyzeFp8OptionalInput():830-833 |
| 9 | dScaleVOptional支持FLOAT32 | 第240行 | AnalyzeFp8OptionalInput():854-857 |
| 10 | sinkOptional支持FLOAT32 | 第250行 | AnalyzeSinkOptionalInput():902-905 |
| 11 | sinkOptional shape为[N1] | 第252行 | AnalyzeSinkOptionalInput():918-923 |
| 12 | queryRopeOptional支持FLOAT16/BF16 | 第200行 | AnalyzeLayout():234-260 |
| 13 | keyRopeOptional支持FLOAT16/BF16 | 第210行 | AnalyzeLayout():234-260 |
| 14 | softmaxMaxOut数据类型FLOAT | 第440行 | GetShapeAttrsInfo():输出检查 |
| 15 | softmaxSumOut数据类型FLOAT | 第450行 | GetShapeAttrsInfo():输出检查 |

## 三、B类约束（约束说明部分缺失）

以下约束在代码中有检查，但文档"约束说明"部分未明确说明，需要补充：

### 3.1 需补充的约束总览

| 序号 | 约束类型 | 约束内容 | 代码位置 |
|------|----------|----------|----------|
| 1 | GQA约束 | N1必须是N2的整数倍 | tiling_regbase.cpp:263-265 |
| 2 | Rope组合约束 | queryRope和keyRope必须同时存在或不存在 | tiling_regbase.cpp:234-237 |
| 3 | MLA约束 | MLA concat仅支持dSize=128, dSizeRope=64 | tiling_regbase.cpp:523-526 |
| 4 | TND约束 | actualSeqQLen和actualSeqKvLen长度必须相等 | tiling_regbase.cpp:324-325 |
| 5 | TND约束 | T必须大于等于seqLen总和 | tiling_regbase.cpp:329-335 |
| 6 | AttenMask压缩 | 压缩模式shape必须为[2048,2048] | tiling_regbase.cpp:699-701 |
| 7 | AttenMask压缩 | Prefix压缩模式shape必须为[3072,2048] | tiling_regbase.cpp:703-708 |
| 8 | SparseMode约束 | sparseMode=2时要求S1<=S2 | tiling_varlen.cpp:319-325 |
| 9 | SparseMode约束 | sparseMode=6时prefixOptional必传 | tiling_regbase.cpp:1632-1636 |
| 10 | Band约束 | preTokens和nextTokens均<0时报错 | tiling_regbase.cpp:1268-1272 |
| 11 | StartIdx约束 | kvStartIdx-qStartIdx必须在int32范围 | tiling_regbase.cpp:1713-1715 |
| 12 | HIFLOAT8约束 | 仅支持BSND布局、无可选输入、特定shape | tiling_regbase.cpp:511-521 |

### 3.2 详细说明

#### B-1: GQA约束（N1是N2的整数倍）
```cpp
// tiling_regbase.cpp:263-265
OP_CHECK_IF(n1Size % n2Size != 0,
           OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld]"),
           return false);
```
**补充位置**: 约束说明"输入query、key、value的"部分

#### B-2: queryRope和keyRope必须同时存在或不存在
```cpp
// tiling_regbase.cpp:234-237
if (hasQueryRope ^ hasKeyRope) {
    OP_LOGE(opName, "query_rope and key_rope should be present or absent at the same time.");
    return false;
}
```
**补充位置**: 约束说明第572-573行后

#### B-3: MLA concat特定约束
```cpp
// tiling_regbase.cpp:523-526
if (hasRope && (dSize != 128 || dSizeRope != 64)) {
    OPS_REPORT_VECTOR_INNER_ERR(opName, "MLA concat only support dSize=128, dSizeRope=64.");
    return ge::GRAPH_FAILED;
}
```
**补充位置**: 约束说明Rope相关部分

#### B-4: TND VarLen batch数量相等
```cpp
// tiling_regbase.cpp:324-325
OP_CHECK_IF(actualSeqQLen != actualSeqKVLen,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "VarLen scene, q is not equal kv."), return false);
```
**补充位置**: 约束说明"actualSeqQLenOptional和actualSeqKvLenOptional约束"部分

#### B-5: TND T必须大于seqLen总和
```cpp
// tiling_regbase.cpp:329-335
OP_CHECK_IF(t1Size < accumS1 || t2Size < accumS2,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "Query T(%ld) and key T(%ld) need larger than sum of seqLen."),
            return false);
```
**补充位置**: 约束说明TND相关部分

#### B-6: AttenMask压缩shape约束
```cpp
// tiling_regbase.cpp:699-701
if ((shape != ATTEN_MASK_COMPRESS_LIMIT) || (shape != ATTEN_MASK_COMPRESS_LIMIT)) {
    OP_LOGE(context_, "In the attenmask compression, please set the atten_mask_shape to [2048,2048].");
    return false;
}
```
**补充位置**: 约束说明"attenMaskOptional约束"部分

#### B-7: Prefix AttenMask压缩shape约束
```cpp
// tiling_regbase.cpp:703-708
if (attenMaskCompressMode == PREFIX_MODE && shape != [3072,2048]) {
    OP_LOGE(context_, "In the prefix attenmask compression, please set the atten_mask_shape to [3072,2048].");
    return false;
}
```
**补充位置**: 约束说明"attenMaskOptional约束"部分

#### B-8: sparseMode=2要求S1<=S2
```cpp
// tiling_varlen.cpp:319-325
for (int64_t i = 0L; i < bSize; ++i) {
    if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
        OP_LOGE(context_, "Batch[%ld] s1[%ld] is larger than s2[%ld], exist invalid row.");
        return false;
    }
}
```
**补充位置**: 约束说明"sparseMode约束"部分

#### B-9: sparseMode=6时prefixOptional必传
```cpp
// tiling_regbase.cpp:1632-1636
if (prefixN == nullptr) {
    if (sparseMode == PREFIX_COMPRESS) {
        OP_LOGE(context_, "prefixN must be provided when sparseMode=6 (PREFIX_COMPRESS).");
        failReason << "prefixN must be provided when sparseMode=6";
    }
    return SparseEnum::ALL;
}
```
**补充位置**: 约束说明"prefixOptional约束"部分（文档第625行已有，但可强化说明）

#### B-10: preTokens和nextTokens均<0报错
```cpp
// tiling_regbase.cpp:1268-1272
if (preTokens < 0 && nextTokens < 0) {
    OP_LOGE(context_, "PreTokens[%ld] and nextTokens[%ld] config error, there is no valid data block.");
    return false;
}
```
**补充位置**: 约束说明"preTokens和nextTokens约束"部分

#### B-11: kvStartIdx-qStartIdx范围约束
```cpp
// tiling_regbase.cpp:1713-1715
OP_CHECK_IF(kvStartIdx - qStartIdx > INT32_MAX || kvStartIdx - qStartIdx < INT32_MIN,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "kvStartIdx - qStartIdx should in int32 range."),
            return ge::GRAPH_FAILED);
```
**补充位置**: 约束说明"qStartIdxOptional和kvStartIdxOptional约束"部分

#### B-12: HIFLOAT8特殊约束
```cpp
// tiling_regbase.cpp:511-521
OP_CHECK_IF((inputDtype == DT_HIFLOAT8) && 
            (hasAttenMask || hasPse || hasDropOut || hasRope || layout != BSND || 特定shape),
            OPS_REPORT_VECTOR_INNER_ERR(opName, "HIFLOAT8约束..."), return ge::GRAPH_FAILED);
```
**补充位置**: 约束说明"HIFLOAT8数据类型约束"部分（已有，可细化）

## 四、C类约束（文档与代码不一致）

| 序号 | 问题 | 代码实际值 | 文档描述 | 处理 |
|------|------|------------|----------|------|
| 1 | dScaleV shape分块大小 | ceil(S2/512) | Ceil(Skv/256) | 修正文档 |
| 2 | PSE batch范围 | 1或bSize | 表格未明确 | 补充说明 |
| 3 | DropMask最小size | ceil(B*N1*S1*S2/8) | 表格未明确 | 补充说明 |

### C-1: dScaleV shape分块大小不一致

**代码位置**: `tiling_regbase.cpp:866-870`
```cpp
OP_CHECK_IF(dimValue2 != (s2Size + QUANT_V_BLOCK_SIZE - 1) / QUANT_V_BLOCK_SIZE,  // QUANT_V_BLOCK_SIZE=512
            OPS_REPORT_VECTOR_INNER_ERR(opName, "only support [B, N2, ceil(S2/512), 1]"),
            return false);
```

**文档位置**: 参数表格第239行
> 输入shape为[B,N2,Ceil(Skv/256),1]

**处理**: 需修正文档为 `[B,N2,Ceil(Skv/512),1]`

## 五、行为说明类（非约束检查）

以下为代码中的行为处理，不属于约束检查范畴：

| 序号 | 内容 | 代码位置 | 说明 |
|------|------|----------|------|
| 1 | preTokens/nextTokens超int32自动调整 | tiling_regbase.cpp:141-158 | Warning日志+自动修正，非报错 |
| 2 | innerPrecise自动使能条件 | 文档已说明 | 功能说明，非约束检查 |
| 3 | seed/offset生效条件 | 文档已说明 | 功能说明 |
| 4 | sparseMode=1/2/3/5时preTokens不生效 | 文档已说明 | 功能说明 |

## 六、总结

| 分类 | 数量 | 处理建议 |
|------|------|----------|
| A类（表格已说明） | 15项 | 无需补充到约束说明部分 |
| B类（约束说明缺失） | 12项 | 补充到文档约束说明部分 |
| C类（文档代码不一致） | 3项 | 修正文档 |
| 行为说明类 | 4项 | 已在文档中说明 |

**建议**: 
- B类约束已在上次修改中补充到约束说明部分
- C类不一致问题需修正文档（特别是dScaleV的shape描述）