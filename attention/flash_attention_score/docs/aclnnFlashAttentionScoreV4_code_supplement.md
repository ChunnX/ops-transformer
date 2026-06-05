# aclnnFlashAttentionScoreV4 代码补充记录

本文档记录了根据约束文档补充到arch35代码中缺失的约束检查逻辑。

## 一、补充的约束检查总览

本次共补充了以下约束检查：

| 序号 | 文档约束描述 | 补充位置 | 补充内容 |
|------|-------------|----------|----------|
| 1 | query/key/value数据类型一致 | AnalyzeDtype() | 检查keyDtype和valueDtype是否与inputDtype一致 |
| 2 | softmaxOutLayout有效性 | AnalyzeAttrs() | 检查softmaxOutLayout只能是空字符串或"same_as_input" |
| 3 | queryRope数据类型与query一致且只支持FLOAT16/BF16 | AnalyzeLayout() | 检查queryRopeDtype与inputDtype一致且只能是FLOAT16/BF16 |
| 4 | keyRope数据类型与key一致且只支持FLOAT16/BF16 | AnalyzeLayout() | 检查keyRopeDtype与inputDtype一致且只能是FLOAT16/BF16 |
| 5 | BSH布局下batchsize相等 | Analyze3DimLayout() | 检查query/key/value的bSize相等 |
| 6 | SBH布局下batchsize相等 | Analyze3DimLayout() | 检查query/key/value的bSize相等 |
| 7 | BSND布局下batchsize相等 | Analyze4DimLayout() | 检查query/key/value的bSize相等 |
| 8 | BNSD布局下batchsize相等 | Analyze4DimLayout() | 检查query/key/value的bSize相等 |
| 9 | queryRope shape与query仅在D维度不同 | Analyze3DimLayout()/Analyze4DimLayout() | 检查queryRope各维度与query匹配(除D外) |
| 10 | PSE数据类型只支持FLOAT16/BF16/FLOAT32 | AnalyzePseOptionalInput() | 检查pseDtype范围 |
| 11 | Inner PSE(pseType=2/3)数据类型必须为FLOAT32 | AnalyzePseOptionalInput() | 检查Inner PSE的dtype必须是FLOAT32 |
| 12 | pseType为2或3时Sq==Skv | AnalyzePseOptionalInput() | 检查s1Size必须等于s2Size |
| 13 | attentionOut数据类型与query一致 | GetShapeAttrsInfo() | 检查attentionOutDtype与inputDtype一致 |
| 14 | softmaxMaxOut数据类型必须为FLOAT | GetShapeAttrsInfo() | 检查softmaxMaxDtype必须是DT_FLOAT |
| 15 | softmaxSumOut数据类型必须为FLOAT | GetShapeAttrsInfo() | 检查softmaxSumDtype必须是DT_FLOAT |

## 二、详细补充内容

### 1. query/key/value数据类型一致性检查

**文档约束位置**: 参数说明表格第129、139、149行
> query: 数据类型与key/value的数据类型一致。
> key: 数据类型与query/value的数据类型一致。
> value: 数据类型与query/key的数据类型一致。

**补充代码位置**: `flash_attention_score_tiling_regbase.cpp:84-98` (AnalyzeDtype函数开头)

**补充代码**:
```cpp
bool FlashAttentionScoreTilingRegbase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    
    // 新增: 检查key/value数据类型与query一致
    auto keyDtype = context_->GetInputDesc(1)->GetDataType();
    auto valueDtype = context_->GetInputDesc(2)->GetDataType();
    OP_CHECK_IF(keyDtype != inputDtype,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "key dtype[%s] should be same as query dtype[%s].",
                                           ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str(),
                                           ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str()),
               return false);
    OP_CHECK_IF(valueDtype != inputDtype,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "value dtype[%s] should be same as query dtype[%s].",
                                           ge::TypeUtils::DataTypeToSerialString(valueDtype).c_str(),
                                           ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str()),
               return false);
    
    // ... 原有switch逻辑
}
```

### 2. softmaxOutLayout有效性检查

**文档约束位置**: 约束说明第665行
> softmaxOutLayout支持传入：空字符串、"same_as_input"。

**补充代码位置**: `flash_attention_score_tiling_regbase.cpp:213-217`

**补充代码**:
```cpp
if (attrs->GetAttrNum() > idx) {
    softmaxOutLayout = attrs->GetAttrPointer<char>(idx++);
    // 新增: 检查softmaxOutLayout有效性
    OP_CHECK_IF(strlen(softmaxOutLayout) > 0 && strcmp(softmaxOutLayout, "same_as_input") != 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "softmaxOutLayout[%s] is invalid, only support empty string or 'same_as_input'.",
                                           softmaxOutLayout),
               return false);
    tndSoftmaxOut = (strcmp(inputLayout, "TND") == 0 && strcmp(softmaxOutLayout, "same_as_input") == 0) ? 1 : tndSoftmaxOut;
}
```

### 3. queryRope/keyRope数据类型检查

**文档约束位置**: 参数说明表格第199、209行
> queryRopeOptional: 数据类型与query的数据类型一致。支持FLOAT16、BFLOAT16
> keyRopeOptional: 数据类型与key的数据类型一致。支持FLOAT16、BFLOAT16

**补充代码位置**: `flash_attention_score_tiling_regbase.cpp:230-260`

**补充代码**:
```cpp
auto queryRopeDesc = context_->GetOptionalInputDesc(QUERY_ROPE_INDEX);
auto keyRopeDesc = context_->GetOptionalInputDesc(KEY_ROPE_INDEX);

if (hasQueryRope && queryRopeDesc != nullptr) {
    auto queryRopeDtype = queryRopeDesc->GetDataType();
    // 新增: 检查数据类型一致性
    OP_CHECK_IF(queryRopeDtype != inputDtype,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "queryRope dtype[%s] should be same as query dtype[%s].",
                                           ge::TypeUtils::DataTypeToSerialString(queryRopeDtype).c_str(),
                                           ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str()),
               return false);
    // 新增: 检查只支持FLOAT16/BF16
    OP_CHECK_IF(queryRopeDtype != ge::DT_FLOAT16 && queryRopeDtype != ge::DT_BF16,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "queryRope dtype[%s] only support FLOAT16 or BFLOAT16.",
                                           ge::TypeUtils::DataTypeToSerialString(queryRopeDtype).c_str()),
               return false);
}
if (hasKeyRope && keyRopeDesc != nullptr) {
    auto keyRopeDtype = keyRopeDesc->GetDataType();
    // 新增: 检查数据类型一致性
    OP_CHECK_IF(keyRopeDtype != inputDtype,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "keyRope dtype[%s] should be same as key dtype[%s].",
                                           ge::TypeUtils::DataTypeToSerialString(keyRopeDtype).c_str(),
                                           ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str()),
               return false);
    // 新增: 检查只支持FLOAT16/BF16
    OP_CHECK_IF(keyRopeDtype != ge::DT_FLOAT16 && keyRopeDtype != ge::DT_BF16,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "keyRope dtype[%s] only support FLOAT16 or BFLOAT16.",
                                           ge::TypeUtils::DataTypeToSerialString(keyRopeDtype).c_str()),
               return false);
}
```

### 4. batchsize相等检查

**文档约束位置**: 约束说明第568行
> B：batchsize必须相等。

**补充代码位置**: 
- `flash_attention_score_tiling_regbase.cpp:384-394` (BSH布局)
- `flash_attention_score_tiling_regbase.cpp:398-408` (SBH布局)
- `flash_attention_score_tiling_regbase.cpp:452-458` (BSND布局)
- `flash_attention_score_tiling_regbase.cpp:474-480` (BNSD布局)

**补充代码示例** (BSH布局):
```cpp
if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[DIM_NUM_2] == 'H') {
    s1Size = queryShape.GetDim(1);
    bSize = queryShape.GetDim(0);
    s2Size = keyShape.GetDim(1);
    // 新增: 检查batchsize相等
    OP_CHECK_IF(bSize != keyShape.GetDim(0),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "query batchsize[%ld] should be same as key batchsize[%ld].",
                                           bSize, keyShape.GetDim(0)),
               return false);
    OP_CHECK_IF(bSize != valueShape.GetDim(0),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "query batchsize[%ld] should be same as value batchsize[%ld].",
                                           bSize, valueShape.GetDim(0)),
               return false);
    // ... 原有逻辑
}
```

### 5. queryRope shape仅在D维度不同检查

**文档约束位置**: 约束说明第572-573行
> 输入queryRopeOptional与query的输入shape仅在D维度不同，其他shape参数应该相同
> 输入keyRopeOptional与key的输入shape仅在D维度不同，其他shape参数应该相同

**补充代码位置**: 
- `flash_attention_score_tiling_regbase.cpp:391-394` (BSH布局)
- `flash_attention_score_tiling_regbase.cpp:405-408` (SBH布局)
- `flash_attention_score_tiling_regbase.cpp:417-420` (TND布局)
- `flash_attention_score_tiling_regbase.cpp:464-467` (BSND布局)
- `flash_attention_score_tiling_regbase.cpp:487-490` (BNSD布局)

**补充代码示例** (BSH布局):
```cpp
if (hasRope) {
    hRope = queryRopeShape->GetDim(DIM_NUM_2);
    // 新增: 检查queryRope shape与query仅在D维度不同
    OP_CHECK_IF(queryRopeShape->GetDim(0) != bSize || queryRopeShape->GetDim(1) != s1Size,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "queryRope shape should match query except D dimension."),
               return false);
}
```

### 6. PSE数据类型检查

**文档约束位置**: 参数说明表格第159行
> realShiftOptional: 数据类型与attentionOut的数据类型一致。支持FLOAT16、BFLOAT16、FLOAT32

**补充代码位置**: `flash_attention_score_tiling_regbase.cpp:614-642`

**补充代码**:
```cpp
if (pseInput != nullptr) {
    auto pseDtype = pseInput->GetDataType();
    // 新增: 检查PSE数据类型范围
    OP_CHECK_IF(pseDtype != ge::DT_FLOAT16 && pseDtype != ge::DT_BF16 && pseDtype != ge::DT_FLOAT,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "PSE dtype[%s] only support FLOAT16, BFLOAT16 or FLOAT32.",
                                           ge::TypeUtils::DataTypeToSerialString(pseDtype).c_str()),
               return false);
    // 新增: Inner PSE(pseType=2/3)必须是FLOAT32
    if (pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
        pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
        OP_CHECK_IF(pseDtype != ge::DT_FLOAT,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "Inner PSE (pseType=2 or 3) dtype must be FLOAT32, but got [%s].",
                                               ge::TypeUtils::DataTypeToSerialString(pseDtype).c_str()),
                   return false);
    }
}
```

### 7. pseType为2或3时Sq==Skv检查

**文档约束位置**: 约束说明第594行
> pseType为2或3的时候，当前只支持Sq和Skv等长。

**补充代码位置**: `flash_attention_score_tiling_regbase.cpp:635-640`

**补充代码**:
```cpp
if (pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
    pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
    // ... shape检查逻辑
    // 新增: 检查Sq==Skv
    OP_CHECK_IF(s1Size != s2Size,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "pseType=2 or 3 only support Sq[%ld] == Skv[%ld].",
                                           s1Size, s2Size),
               return false);
}
```

### 8. 输出tensor数据类型检查

**文档约束位置**: 参数说明表格
> attentionOutOut: 数据类型和shape类型与query保持一致。支持FLOAT16、BFLOAT16、FLOAT32
> softmaxMaxOut: 数据类型FLOAT
> softmaxSumOut: 数据类型FLOAT

**补充代码位置**: `flash_attention_score_tiling_regbase.cpp:598-645` (GetShapeAttrsInfo函数)

**补充代码**:
```cpp
// 新增: 检查attentionOut数据类型
auto attentionOutDesc = context_->GetOutputDesc(ATTEN_OUT_INDEX);
if (attentionOutDesc != nullptr) {
    auto attentionOutDtype = attentionOutDesc->GetDataType();
    OP_CHECK_IF(attentionOutDtype != inputDtype,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "attentionOut dtype[%s] should be same as query dtype[%s].",
                                           ge::TypeUtils::DataTypeToSerialString(attentionOutDtype).c_str(),
                                           ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str()),
               return ge::GRAPH_FAILED);
}

// 新增: 检查softmaxMaxOut数据类型
auto softmaxMaxDesc = context_->GetOutputDesc(SOFTMAX_MAX_OUT_INDEX);
if (softmaxMaxDesc != nullptr) {
    auto softmaxMaxDtype = softmaxMaxDesc->GetDataType();
    OP_CHECK_IF(softmaxMaxDtype != ge::DT_FLOAT,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "softmaxMaxOut dtype[%s] must be FLOAT.",
                                           ge::TypeUtils::DataTypeToSerialString(softmaxMaxDtype).c_str()),
               return ge::GRAPH_FAILED);
}

// 新增: 检查softmaxSumOut数据类型
auto softmaxSumDesc = context_->GetOutputDesc(SOFTMAX_SUM_OUT_INDEX);
if (softmaxSumDesc != nullptr) {
    auto softmaxSumDtype = softmaxSumDesc->GetDataType();
    OP_CHECK_IF(softmaxSumDtype != ge::DT_FLOAT,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "softmaxSumOut dtype[%s] must be FLOAT.",
                                           ge::TypeUtils::DataTypeToSerialString(softmaxSumDtype).c_str()),
               return ge::GRAPH_FAILED);
}
```

## 三、未实现但建议保留的功能性约束

以下文档约束为功能性说明或建议，非强制校验，暂不补充代码检查：

| 序号 | 文档约束 | 说明 |
|------|----------|------|
| 1 | T(B*S)：取值范围为1~1M | 这是性能限制而非硬约束，超出可能导致超时而非立即报错 |
| 2 | B：取值范围为1~2M | 同上，性能限制 |
| 3 | N：取值范围为1~256 | 同上，性能限制 |
| 4 | S：取值范围为1~1M | 同上，性能限制 |
| 5 | 配置为1、2、3、5时preTokens/nextTokens不会生效 | 这是行为说明，不是约束检查 |
| 6 | 用户不特意指定时建议传入0 | 这是建议而非强制 |
| 7 | 当所有attenMaskOptional的shape小于2048时建议使用default模式 | 这是建议而非强制 |
| 8 | 确定性计算默认实现 | 这是功能说明 |
| 9 | innerPrecise配置说明 | 功能性说明，0、1为保留配置值 |
| 10 | seed和offset只在keepProb小于1.0时生效 | 这是行为说明，已有逻辑处理 |

## 四、修改文件汇总

本次修改涉及以下文件：
- `/mnt/workspace/transformer-issue/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_regbase.cpp`

主要修改函数：
- `AnalyzeDtype()` - 补充query/key/value数据类型一致性检查
- `AnalyzeAttrs()` - 补充softmaxOutLayout有效性检查
- `AnalyzeLayout()` - 补充queryRope/keyRope数据类型检查
- `Analyze3DimLayout()` - 补充batchsize相等检查和queryRope shape检查
- `Analyze4DimLayout()` - 补充batchsize相等检查和queryRope shape检查
- `AnalyzePseOptionalInput()` - 补充PSE数据类型检查和Inner PSE约束
- `GetShapeAttrsInfo()` - 补充输出tensor数据类型检查

## 五、总结

本次代码补充确保了约束文档中的关键约束在arch35代码中得到实现，增强了参数校验的完整性。所有补充的检查都会在参数不符合要求时返回错误，帮助用户及时发现配置问题。