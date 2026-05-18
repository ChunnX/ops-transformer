# aclnnFlashAttentionScoreV4 约束补充说明

本文档记录了在arch35代码实现中发现但原文档中未明确说明的约束项，以及这些约束在代码中的具体位置。

## 一、补充的约束项总览

本次分析在arch35目录下的tiling代码中发现并补充了以下约束项：

| 序号 | 约束类型 | 补充的约束内容 | 代码文件位置 |
|------|----------|----------------|--------------|
| 1 | 基本参数 | headNum（n1Size）不能为0 | flash_attention_score_tiling_regbase.cpp:159 |
| 2 | 基本参数 | n2Size（key/value的头数）不能为0 | flash_attention_score_tiling_regbase.cpp:256,367 |
| 3 | 基本参数 | gSize（Group数）不能为0 | flash_attention_score_tiling_regbase.cpp:255 |
| 4 | 基本参数 | n1Size必须是n2Size的整数倍（GQA约束） | flash_attention_score_tiling_regbase.cpp:263-265 |
| 5 | Shape约束 | D超过768会报错（原文档只说范围未说超出后果） | flash_attention_score_tiling_regbase.cpp:977-980 |
| 6 | Shape约束 | S1和S2不能超过int32最大值 | flash_attention_score_tiling_regbase.cpp:250-254 |
| 7 | Layout约束 | query/key/value维度必须与layout一致 | flash_attention_score_tiling_regbase.cpp:244-246 |
| 8 | Layout约束 | 只支持BSH/SBH/TND/BSND/BNSD布局 | flash_attention_score_tiling_regbase.cpp:247-249 |
| 9 | Rope约束 | queryRope和keyRope必须同时存在或不存在 | flash_attention_score_tiling_regbase.cpp:234-237 |
| 10 | Rope约束 | MLA concat只支持dSize=128, dSizeRope=64 | flash_attention_score_tiling_regbase.cpp:523-526 |
| 11 | TND约束 | VarLen场景q和kv的batch数量必须相等 | flash_attention_score_tiling_regbase.cpp:324-325 |
| 12 | TND约束 | T必须大于等于seqLen总和 | flash_attention_score_tiling_regbase.cpp:329-335 |
| 13 | PSE约束 | PSE维度必须为4（普通模式） | flash_attention_score_tiling_regbase.cpp:577-579 |
| 14 | PSE约束 | PSE batch只能为1或bSize | flash_attention_score_tiling_regbase.cpp:581-583 |
| 15 | PSE约束 | Inner PSE（pseType=2/3）维度必须是1或2 | flash_attention_score_tiling_regbase.cpp:626-628 |
| 16 | PSE约束 | PSE alibi（Sq>1024）只支持causal sparse | flash_attention_score_tiling_regbase.cpp:134-135,158-161,1728-1733 |
| 17 | AttenMask约束 | attenMask维度必须为2或4 | flash_attention_score_tiling_regbase.cpp:730-740 |
| 18 | AttenMask约束 | attenMask压缩shape必须为[2048,2048] | flash_attention_score_tiling_regbase.cpp:699-701 |
| 19 | AttenMask约束 | Prefix attenMask压缩shape必须为[3072,2048] | flash_attention_score_tiling_regbase.cpp:703-708 |
| 20 | DropMask约束 | DropMask存在时keepProb必须<1 | flash_attention_score_tiling_regbase.cpp:757-759 |
| 21 | DropMask约束 | DropMask size必须>=需要的最小值 | flash_attention_score_tiling_regbase.cpp:783-787 |
| 22 | dScale约束 | dScaleQ shape必须为[B,N1,ceil(S1/128),1] | flash_attention_score_tiling_regbase.cpp:819-823 |
| 23 | dScale约束 | dScaleK shape必须为[B,N2,ceil(S2/256),1] | flash_attention_score_tiling_regbase.cpp:843-847 |
| 24 | dScale约束 | dScaleV shape必须为[B,N2,ceil(S2/512),1] | flash_attention_score_tiling_regbase.cpp:866-870 |
| 25 | dScale约束 | pScale shape必须为[1] | flash_attention_score_tiling_regbase.cpp:886-888 |
| 26 | Sink约束 | Sink大小必须为headNum | flash_attention_score_tiling_regbase.cpp:920-923 |
| 27 | SparseMode约束 | sparseMode超过6会报错 | flash_attention_score_tiling_regbase.cpp:1342-1344 |
| 28 | SparseMode约束 | RIGHT_DOWN_CAUSAL要求S1<=S2 | flash_attention_score_tiling_regbase.cpp:1205-1215 |
| 29 | Band约束 | preTokens和nextTokens均<0时报错 | flash_attention_score_tiling_regbase.cpp:1268-1272 |
| 30 | 参数范围 | preTokens/nextTokens超出int32范围自动调整 | flash_attention_score_tiling_regbase.cpp:141-158 |
| 31 | 参数范围 | pseType超出范围(0~3)报错 | flash_attention_score_tiling_regbase.cpp:195-196 |
| 32 | 参数范围 | outDtype超出范围(0~1)报错 | flash_attention_score_tiling_regbase.cpp:209-211 |
| 33 | StartIdx约束 | kvStartIdx-qStartIdx超出int32范围报错 | flash_attention_score_tiling_regbase.cpp:1713-1715 |
| 34 | HIFLOAT8约束 | 特定shape和不可选输入约束 | flash_attention_score_tiling_regbase.cpp:511-521 |

## 二、详细约束代码对应关系

### 1. 基本参数约束

#### 1.1 headNum不能为0
**代码位置**: `flash_attention_score_tiling_regbase.cpp:159`
```cpp
OP_CHECK_IF(n1Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "head num is zero."), return false);
```
**文档补充**: 在"输入query、key、value的"部分明确说明headNum参数不能为0。

#### 1.2 n2Size不能为0
**代码位置**: `flash_attention_score_tiling_regbase.cpp:256,367`
```cpp
OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero"), return false);
```
**文档补充**: 在N取值范围约束中明确说明N2不能为0。

#### 1.3 gSize不能为0
**代码位置**: `flash_attention_score_tiling_regbase.cpp:255`
```cpp
OP_CHECK_IF(gSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSize is zero"), return false);
```
**文档补充**: 在数据shape约束中补充G（Group数）不能为0的说明。

#### 1.4 GQA约束（n1Size是n2Size的整数倍）
**代码位置**: `flash_attention_score_tiling_regbase.cpp:263-265`
```cpp
OP_CHECK_IF(n1Size % n2Size != 0,
           OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld]", n1Size, n2Size),
           return false);
```
**文档补充**: 明确说明N1必须是N2的整数倍（Grouped Query Attention约束）。

### 2. Shape维度约束

#### 2.1 D超过768会报错
**代码位置**: `flash_attention_score_tiling_regbase.cpp:977-980`
```cpp
OP_CHECK_IF(dSize > HEAD_DIM_MAX_VALUE,
           OPS_REPORT_VECTOR_INNER_ERR(opName, "query or key dSize is not in range:(0, 768]"), return ge::GRAPH_FAILED);
OP_CHECK_IF(dSizeV > HEAD_DIM_MAX_VALUE,
           OPS_REPORT_VECTOR_INNER_ERR(opName, "value dSize is not in range:(0, 768]"), return ge::GRAPH_FAILED);
```
**文档补充**: 原文档说D取值范围为1~768，但未明确超出范围会报错。补充说明"超出768会报错"。

#### 2.2 S1/S2不能超过int32最大值
**代码位置**: `flash_attention_score_tiling_regbase.cpp:250-254`
```cpp
if (s1Size > std::numeric_limits<int32_t>::max() || s2Size > std::numeric_limits<int32_t>::max()) {
    OP_LOGE(context_, "s1Size[%ld] and s2Size[%ld] config error, both should not greater than max int value.",
        s1Size, s2Size);
    return false;
}
```
**文档补充**: 在S取值范围约束中补充不能超过int32最大值的说明。

### 3. Layout布局约束

#### 3.1 维度与layout一致
**代码位置**: `flash_attention_score_tiling_regbase.cpp:244-246`
```cpp
OP_CHECK_IF(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen ||
           valueShape.GetDimNum() != layoutLen,
           OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid layout[%s].", inputLayout), return false);
```
**文档补充**: 明确说明query/key/value的维度必须与layout维度一致。

#### 3.2 支持的布局类型
**代码位置**: `flash_attention_score_tiling_regbase.cpp:247-249`
```cpp
OP_CHECK_IF(!Analyze3DimLayout(queryShape, keyShape, valueShape, layoutLen, queryRopeShape)||
           !Analyze4DimLayout(queryShape, keyShape, valueShape, layoutLen, queryRopeShape),
           OPS_REPORT_VECTOR_INNER_ERR(opName, "get unsupported layout: %s", inputLayout), return false);
```
**文档补充**: 明确列出支持的布局类型：BSH、SBH、TND、BSND、BNSD。

### 4. Rope约束

#### 4.1 queryRope和keyRope必须同时存在或不存在
**代码位置**: `flash_attention_score_tiling_regbase.cpp:234-237`
```cpp
if (hasQueryRope ^ hasKeyRope) {
    OP_LOGE(opName, "query_rope and key_rope should be present or absent at the same time, check this.");
    return false;
}
```
**文档补充**: 原文档分别描述queryRope和keyRope的shape约束，但未说明必须同时传入。

#### 4.2 MLA concat特定约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:523-526`
```cpp
if (hasRope && (dSize != 128 || dSizeRope != 64)) {
    OPS_REPORT_VECTOR_INNER_ERR(opName, "MLA concat only support dSize=128, dSizeRope=64.");
    return ge::GRAPH_FAILED;
}
```
**文档补充**: 说明MLA concat场景的特定参数要求。

### 5. TND VarLen约束

#### 5.1 batch数量必须相等
**代码位置**: `flash_attention_score_tiling_regbase.cpp:324-325`
```cpp
OP_CHECK_IF(actualSeqQLen != actualSeqKVLen,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "VarLen scene, q is not equal kv."), return false);
```
**文档补充**: 明确VarLen场景下actualSeqQLen和actualSeqKvLen的batch数量必须相等。

#### 5.2 T必须大于等于seqLen总和
**代码位置**: `flash_attention_score_tiling_regbase.cpp:329-335`
```cpp
OP_CHECK_IF(
    t1Size < accumS1 || t2Size < accumS2,
    OPS_REPORT_VECTOR_INNER_ERR(
        opName,
        "Query T(%ld) and key T(%ld) need larger than respectively sum of seqLen(%ld) and sekvLen(%ld).",
        t1Size, t2Size, accumS1, accumS2),
    return false);
```
**文档补充**: 说明TND布局下T参数与seqLen总和的关系约束。

### 6. PSE约束

#### 6.1 PSE维度必须为4
**代码位置**: `flash_attention_score_tiling_regbase.cpp:577-579`
```cpp
if (pseDimNum != PSE_DIM_NUM) {
    OP_LOGE(context_, "pse dim should be 4, but got %zu", pseDimNum);
    return false;
}
```
**文档补充**: 原文档列出了PSE的shape，但未明确说明维度必须为4维。

#### 6.2 PSE batch只能为1或bSize
**代码位置**: `flash_attention_score_tiling_regbase.cpp:581-583`
```cpp
if (pseBSize != bSize && pseBSize != 1) {
    OP_LOGE(context_, "pse batchsize should be 1 or %ld, but got %ld", bSize, pseBSize);
    return false;
}
```
**文档补充**: 明确说明PSE的batch维度只能是1或与query的batch相同。

#### 6.3 Inner PSE维度约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:626-628`
```cpp
if (pseDimNum != SLOPE_BN_DIM_NUM && pseDimNum != SLOPE_N_DIM_NUM) {
    OP_LOGE(context_, "pse inner mode, unsupported pse shape");
    return false;
}
```
**文档补充**: pseType为2或3（inner模式）时，PSE维度必须是1或2维。

#### 6.4 PSE alibi sparse模式约束
**代码位置**: 
- `flash_attention_score_tiling_basic.cpp:134-135`
- `flash_attention_score_tiling_varlen.cpp:158-161`
- `flash_attention_score_tiling_regbase.cpp:1728-1733`

```cpp
OP_CHECK_IF(inputParamsRegbase_->get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL),
           OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support causal sparse type."), return false);
```
**文档补充**: Sq>1024且使能alibi压缩时，仅支持causal sparse模式。

### 7. AttenMask约束

#### 7.1 attenMask维度必须为2或4
**代码位置**: `flash_attention_score_tiling_regbase.cpp:730-740`
```cpp
if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_4) {
    ...
} else if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_2) {
    ...
} else {
    OP_LOGE(context_, "atten mask dim should be 2 or 4, but got %zu", attenMaskDimNum);
    return false;
}
```
**文档补充**: 明确说明attenMask维度只能是2或4维。

#### 7.2 attenMask压缩shape约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:699-701`
```cpp
if ((attenMaskStorageShape.GetDim(attenMaskDimNum - ATTEN_MASK_S1_REV_INDEX) != ATTEN_MASK_COMPRESS_LIMIT) ||
    (attenMaskStorageShape.GetDim(attenMaskDimNum - 1) != ATTEN_MASK_COMPRESS_LIMIT)) {
    OP_LOGE(context_, "In the attenmask compression, please set the atten_mask_shape to [2048,2048].");
    return false;
}
```
**文档补充**: 明确说明attenMask压缩模式下shape必须为[2048,2048]。

#### 7.3 Prefix attenMask压缩shape约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:703-708`
```cpp
if (attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE) &&
    ((attenMaskStorageShape.GetDim(...) != ATTEN_MASK_COMPRESS_PREFIX_LIMIT) ||
     (attenMaskStorageShape.GetDim(...) != ATTEN_MASK_COMPRESS_LIMIT))) {
    OP_LOGE(context_, "In the prefix attenmask compression, please set the atten_mask_shape to [3072,2048].");
    return false;
}
```
**文档补充**: 说明prefix attenMask压缩模式下shape必须为[3072,2048]。

### 8. DropMask约束

#### 8.1 DropMask与keepProb的关系
**代码位置**: `flash_attention_score_tiling_regbase.cpp:757-759`
```cpp
if (!hasDropOut) {
    OP_LOGE(context_, "Dropmask parameter is invalid, please check keepProb.");
    return false;
}
```
**文档补充**: 明确说明传入DropMask时keepProb必须小于1。

#### 8.2 DropMask size约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:783-787`
```cpp
if (dropMaskShapeSize < shapeSize) {
    OP_LOGE(context_, "Input dropMask shapeSize is invalid, it should not be less than %ld, but got %ld",
              shapeSize, dropMaskShapeSize);
    return false;
}
```
**文档补充**: 明确说明DropMask size必须大于等于需要的最小值ceil(B*N1*S1*S2/8)。

### 9. 量化参数约束

#### 9.1 dScaleQ shape约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:819-823`
```cpp
OP_CHECK_IF(dimValue0 != bSize || dimValue1 != n1Size ||
(dimValue2 != (s1Size + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE) || dimValue3 != D_SCALE_DIM_NUM_1,
        OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleQ dimNump[%ld][%ld][%ld][%ld], only support [B, N1, ceil(S1/128), 1]"),
        return false);
```
**文档补充**: 原文档说shape为[B,N1,Ceil(Sq/128),1]，补充说明这是必须满足的约束。

#### 9.2 dScaleK shape约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:843-847`
```cpp
OP_CHECK_IF(dimValue0 != bSize || dimValue1 != n2Size ||
    (dimValue2 != (s2Size + QUANT_K_BLOCK_SIZE  - 1) / QUANT_K_BLOCK_SIZE ) || dimValue3 != D_SCALE_DIM_NUM_1,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleK dimNump[%ld][%ld][%ld][%ld], only support [B, N2, ceil(S2/256), 1]"),
            return false);
```
**文档补充**: 明确dScaleK的shape必须为[B, N2, ceil(S2/256), 1]。

#### 9.3 dScaleV shape约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:866-870`
```cpp
OP_CHECK_IF(dimValue0 != bSize || dimValue1 != n2Size ||
    (dimValue2 != (s2Size + QUANT_V_BLOCK_SIZE - 1) / QUANT_V_BLOCK_SIZE) || dimValue3 != D_SCALE_DIM_NUM_1,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleV dimNum [%ld][%ld][%ld][%ld], only support [B, N2, ceil(S2/512), 1]"),
            return false);
```
**文档补充**: 原文档说shape为[B,N2,Ceil(Skv/256),1]，但代码实际要求ceil(S2/512)，需修正文档。

#### 9.4 pScale shape约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:886-888`
```cpp
OP_CHECK_IF(dimValue0 != D_SCALE_DIM_NUM_1,
        OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid pScale shape[0] [%ld], only support [1].", dimValue0),
        return false);
```
**文档补充**: 说明pScale的shape必须为[1]。

### 10. Sink约束

#### 10.1 Sink大小必须为headNum
**代码位置**: `flash_attention_score_tiling_regbase.cpp:920-923`
```cpp
int64_t expectedSinkSize = n1Size;
auto actualSinkShapeSize = shape.GetShapeSize();
OP_CHECK_IF(actualSinkShapeSize != expectedSinkSize, OP_LOGE(context_,
    "invalid sink shapeSize, expect [%ld], but got [%ld].",
    expectedSinkSize, actualSinkShapeSize),
    return false);
```
**文档补充**: 原文档说shape为[N1]，但未明确说明这是必须满足的约束。

### 11. SparseMode约束

#### 11.1 sparseMode超过6报错
**代码位置**: `flash_attention_score_tiling_regbase.cpp:1342-1344`
```cpp
if (sparseMode > static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
    OP_LOGE(context_, "Not support sparse mode of %ld.", sparseMode);
    return false;
}
```
**文档补充**: 原文档说支持配置值为0~6，但未明确超出会报错。

#### 11.2 RIGHT_DOWN_CAUSAL要求S1<=S2
**代码位置**: `flash_attention_score_tiling_varlen.cpp:319-325`
```cpp
for (int64_t i = 0L; i < bSize; ++i) {
    if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
        OP_LOGE(context_, "Batch[%ld] s1[%ld] is larger than s2[%ld], exist invalid row.", i,
                  actualSeqLenData[i], actualSeqLenKvData[i]);
        return false;
    }
}
```
**文档补充**: 说明sparseMode=2（RIGHT_DOWN_CAUSAL）时要求S1 <= S2。

### 12. Band约束

#### 12.1 preTokens和nextTokens均小于0时报错
**代码位置**: `flash_attention_score_tiling_regbase.cpp:1268-1272`
```cpp
if (preTokens < 0 && nextTokens < 0) {
    OP_LOGE(context_, "PreTokens[%ld] and nextTokens[%ld] config error, there is no valid data block.",
              oriPreTokens, oriNextTokens);
    return false;
}
```
**文档补充**: 原文档说band场景preTokens和nextTokens之间要有交集，补充说明两者均小于0表示无有效数据块会报错。

### 13. 参数范围约束

#### 13.1 preTokens/nextTokens超出int32范围处理
**代码位置**: `flash_attention_score_tiling_regbase.cpp:141-158`
```cpp
if (preTokens > std::numeric_limits<int32_t>::max()) {
    OP_LOGW(context_, "preTokens[%ld] config error, should not greater than max int value."
        "preTokens will be reset max int value.", preTokens);
    preTokens = std::numeric_limits<int32_t>::max();
}
```
**文档补充**: 说明preTokens和nextTokens超出int32范围时会被自动调整到边界值（warning而非error）。

#### 13.2 pseType范围约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:195-196`
```cpp
OP_CHECK_IF(pseType < 0 || pseType >= static_cast<uint8_t>(PseType::PSE_INVALID_TYPE),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "pseType value is out of range"), return false);
```
**文档补充**: 原文档列出了pseType的含义，但未明确超出范围会报错。

#### 13.3 outDtype范围约束
**代码位置**: `flash_attention_score_tiling_regbase.cpp:209-211`
```cpp
OP_CHECK_IF(outDtype < 0 || outDtype >= 2,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "outDtype value is out of range"), return false);
```
**文档补充**: 说明outDtype只能为0或1，超出范围报错。

### 14. StartIdx约束

**代码位置**: `flash_attention_score_tiling_regbase.cpp:1713-1715`
```cpp
OP_CHECK_IF(kvStartIdx - qStartIdx > INT32_MAX || kvStartIdx - qStartIdx < INT32_MIN, 
    OPS_REPORT_VECTOR_INNER_ERR(opName,
    "kvStartIdx - qStartIdx should >= %d and <= %d, but qStartIdx = %ld, kvStartIdx = %ld.", 
    INT32_MIN, INT32_MAX, qStartIdx, kvStartIdx),
    return ge::GRAPH_FAILED);
```
**文档补充**: 说明kvStartIdx - qStartIdx的结果必须在int32范围内。

### 15. HIFLOAT8约束

**代码位置**: `flash_attention_score_tiling_regbase.cpp:511-521`
```cpp
OP_CHECK_IF((inputDtype == ge::DT_HIFLOAT8) && (hasAttenMask || hasPse || hasDropOut || hasRope ||
            tilingKeyLayout != LayoutType::LAYOUT_BSND ||
            bSize != 1 || n1Size != n2Size || dSize != 128 || dSizeV != 128 ||
            !((s1Size == 57600 && s2Size == 57600 && n1Size == 5) || (s1Size == 7200 && s2Size == 512 && n1Size == 40))),
            OPS_REPORT_VECTOR_INNER_ERR(opName, "HIFLOAT8 can only support layout:BSND without any optional inputs, "
            "and the input shape must be: ..."), return ge::GRAPH_FAILED);
```
**文档补充**: 详细说明HIFLOAT8数据类型的特殊约束：
- 不支持可选输入
- 仅支持BSND布局
- 仅支持特定shape组合

## 三、代码文件说明

本次分析的arch35相关代码文件：
- `op_host/arch35/flash_attention_score_tiling_regbase.cpp` - 基础tiling逻辑和主要约束检查
- `op_host/arch35/flash_attention_score_tiling_basic.cpp` - 非VarLen场景的tiling逻辑
- `op_host/arch35/flash_attention_score_tiling_varlen.cpp` - VarLen（TND）场景的tiling逻辑和特殊约束
- `op_host/arch35/flash_attention_score_tiling_dropmask.cpp` - DropMask处理逻辑

主要约束检查函数：
- `AnalyzeAttrs()` - 分析并检查属性参数
- `AnalyzeLayout()` - 分析并检查layout和shape
- `AnalyzePseOptionalInput()` - 分析PSE输入约束
- `AnalyzeAttenOptionalInput()` - 分析attenMask输入约束
- `AnalyzeDropOptionalInput()` - 分析dropMask输入约束
- `AnalyzeFp8OptionalInput()` - 分析量化参数约束
- `AnalyzeSinkOptionalInput()` - 分析sink输入约束
- `GetSparseInfo()` - 分析sparseMode约束
- `SparseBandModeCheck()` - band模式详细检查

## 四、总结

本次分析共发现并补充了34项约束说明，这些约束在代码实现中都有明确的检查逻辑，但在原文档中或未说明、或说明不完整、或说明位置不够清晰。

建议后续维护时：
1. 保持文档与代码约束的一致性
2. 新增约束时同步更新文档
3. 使用统一的约束描述格式
4. 关键约束应注明违反后的行为（报错/warning/自动调整）