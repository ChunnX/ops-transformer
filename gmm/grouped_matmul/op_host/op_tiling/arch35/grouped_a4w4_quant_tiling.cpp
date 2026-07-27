/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_a4w4_quant_tiling.cpp
 * \brief A5 A4W4 (INT4xINT4) quant tiling. Plan: baseM=128/baseN=256/ubCalSize=4096.
 *        Inherits GroupedQmmBasicApiTiling; reuses inputParams_/basicTiling_/mList_/kList_/nList_/aicoreParams_.
 *        Tiling + dispatch only (no kernel prologue/epilogue).
 *
 * TODO (post kernel impl): PROLOGUE_MIN_BYTES_950/A4W4_VEC_BASE_M_950 measured values; sync ID.
 */

#include "grouped_a4w4_quant_tiling.h"
#include <algorithm>
#include "err/ops_err.h"
#include "../../grouped_matmul_host_util.h"
#include "../../../op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"

namespace optiling {

using namespace GroupedMatmul;

namespace {
// local constants (align GmmConstant; avoid pulling GroupedQmmTiling helper)
constexpr uint32_t DB_SIZE = 2U;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint32_t DATA_SIZE_L0C = 4U;  // int8xint8 -> int32 L0C accumulator
constexpr uint64_t GMM_MAX_GROUP_LIST_SIZE = 1024UL;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr float EFFECTIVE_TASK_RATIO = 0.95f;

inline uint64_t CeilDiv(uint64_t a, uint64_t b) { return (a + b - 1UL) / b; }
inline uint64_t CeilAlign(uint64_t a, uint64_t b) { return CeilDiv(a, b) * b; }
inline uint64_t GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) {
    return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}
// AlignUp in grouped_matmul_tiling.cpp is `static inline` (internal linkage), invisible here; mirror it locally.
template <typename T>
static inline auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) { return 0; }
    if (num1 < 0) { return -(-num1 / num2) * num2; }
    return (num1 + num2 - 1) / num2 * num2;
}
}  // namespace

GroupedA4W4QuantTiling::GroupedA4W4QuantTiling(gert::TilingContext *context)
    : GroupedQmmBasicApiTiling(context) {}

void GroupedA4W4QuantTiling::Reset()
{
    a4w4Tiling_ = A4W4BasicTiling{};
    int8WeightWs_ = 0;
    int8XWs_ = 0;
    mmOutWs_ = 0;
    perTokenScaleFillWs_ = 0;
}

bool GroupedA4W4QuantTiling::IsCapable()
{
    return inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4;
}

ge::graphStatus GroupedA4W4QuantTiling::GetPlatformInfo()
{
    OP_CHECK_IF(GroupedQmmBasicApiTiling::GetPlatformInfo() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "A4W4 base GetPlatformInfo failed."),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context_->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
    // AiCoreParams (TilingBaseClass) has no aivNum field; read aivNum from compileInfo directly.
    uint32_t aivNum = compileInfoPtr->aivNum;
    OP_CHECK_IF(aicoreParams_.aicNum == 0 || aivNum != 2UL * aicoreParams_.aicNum,
                OP_LOGE(context_->GetNodeName(), "A4W4 mix-core needs aic:aiv=1:2, got aic=%lu aiv=%u",
                        aicoreParams_.aicNum, aivNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(aicoreParams_.ubSize == 0,
                OP_LOGE(context_->GetNodeName(), "ubSize is 0."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedA4W4QuantTiling::GetShapeAttrsInfo()
{
    // 1) dtype
    auto xDesc = context_->GetDynamicInputDesc(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wDesc);
    inputParams_.bDtype = wDesc->GetDataType();
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    inputParams_.cDtype = yDesc->GetDataType();
    auto scaleDesc = context_->GetDynamicInputDesc(SCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    auto perTokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.aQuantMode = (perTokenScaleDesc != nullptr) ? QuantMode::PERTOKEN_MODE : QuantMode::DEFAULT;
    if (inputParams_.aQuantMode == QuantMode::PERTOKEN_MODE) {
        inputParams_.perTokenScaleDtype = perTokenScaleDesc->GetDataType();
    }
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    // 2) attrs
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_ACT_TYPE + 1,
                    OP_LOGE(context_->GetNodeName(), "A4W4: attr num < %lu, got %zu.",
                            ATTR_INDEX_ACT_TYPE + 1, attrs->GetAttrNum()),
                    return ge::GRAPH_FAILED);
        const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
        const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
        const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
        const int64_t *actTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);
        inputParams_.transB = (transposeWeightPtr != nullptr) && *transposeWeightPtr;
        inputParams_.transA = (transposeXPtr != nullptr) && *transposeXPtr;
        inputParams_.groupType = (groupTypePtr != nullptr) ? static_cast<int8_t>(*groupTypePtr) : inputParams_.groupType;
        inputParams_.groupListType = (groupListTypePtr != nullptr) ? static_cast<uint8_t>(*groupListTypePtr) : inputParams_.groupListType;
        int64_t actTypeVal = (actTypePtr != nullptr) ? *actTypePtr : 0L;
        OP_CHECK_IF(actTypeVal != 0L,
                    OP_LOGE(context_->GetNodeName(), "A4W4: actType must be 0, got %ld.", actTypeVal),
                    return ge::GRAPH_FAILED);
    } else {
        inputParams_.groupType = static_cast<int8_t>(GroupedMatmul::SPLIT_M);
        inputParams_.groupListType = 0U;
        inputParams_.transA = false;
        inputParams_.transB = false;
    }
    OP_CHECK_IF(inputParams_.groupType != GroupedMatmul::SPLIT_M,
                OP_LOGE(context_->GetNodeName(), "A4W4: groupType must be 0(SPLIT_M), got %d.",
                        static_cast<int32_t>(inputParams_.groupType)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputParams_.transA, OP_LOGE(context_->GetNodeName(), "A4W4: transA not supported."),
                return ge::GRAPH_FAILED);
    inputParams_.isSingleX = (context_->GetDynamicInputDesc(X_INDEX, 1) == nullptr);
    inputParams_.isSingleW = (context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) == nullptr);
    inputParams_.isSingleY = (context_->GetOutputDesc(1) == nullptr);
    // 3) SetMKN + SetGroupNum
    auto xShapePtr = context_->GetDynamicInputShape(X_INDEX, 0);
    auto wShapePtr = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShapePtr);
    OP_CHECK_IF(!SetMKN(xShapePtr->GetStorageShape(), wShapePtr->GetStorageShape()),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "A4W4 SetMKN failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "A4W4 SetGroupNum failed."),
                return ge::GRAPH_FAILED);
    // 4) AnalyzeA4W4 + CheckA4W4Params
    OP_CHECK_IF(!AnalyzeA4W4(), OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "A4W4 AnalyzeA4W4 failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckA4W4Params(), OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "A4W4 CheckA4W4Params failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool GroupedA4W4QuantTiling::SetMKN(const gert::Shape &xShape, const gert::Shape &wShape)
{
    uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    OP_CHECK_IF(xDimNum < MIN_ND_DIM || wDimNum < MIN_ND_DIM,
                OP_LOGE(context_->GetNodeName(), "A4W4: x/weight dim must be >= 2."), return false);
    OP_CHECK_IF(!GroupedQmmBasicApiTiling::SetMKN(xShape, wShape),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "A4W4 base SetMKN failed."),
                return false);
    a4w4Tiling_.m = inputParams_.mSize;
    a4w4Tiling_.n = inputParams_.nSize;
    a4w4Tiling_.k = inputParams_.kSize;
    a4w4Tiling_.groupType = inputParams_.groupType;
    a4w4Tiling_.groupListType = inputParams_.groupListType;
    return true;
}

bool GroupedA4W4QuantTiling::SetGroupNum(uint32_t groupListIndex)
{
    OP_CHECK_IF(!GroupedQmmBasicApiTiling::SetGroupNum(groupListIndex),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "A4W4 base SetGroupNum failed."),
                return false);
    a4w4Tiling_.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    return true;
}

bool GroupedA4W4QuantTiling::SetMKNList()
{
    // Base's mList_/kList_/nList_ is private (inaccessible to derived).
    // Write to our own shadow arrays (mimic base GroupedQmmTiling::SetMKNList).
    if (inputParams_.groupType == GroupedMatmul::SPLIT_M) {
        mList_[0] = -1;  // M 动态
        kList_[0] = static_cast<int32_t>(inputParams_.kSize);
        nList_[0] = static_cast<int32_t>(inputParams_.nSize);
    } else {
        mList_[0] = static_cast<int32_t>(inputParams_.mSize);
        kList_[0] = -1;
        nList_[0] = static_cast<int32_t>(inputParams_.nSize);
    }
    return true;
}

bool GroupedA4W4QuantTiling::AnalyzeA4W4()
{
    auto scaleShapePtr = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    uint32_t scaleDimNum = static_cast<uint32_t>(scaleShapePtr->GetStorageShape().GetDimNum());
    if (scaleDimNum == 3U) {
        inputParams_.bQuantMode = QuantMode::PERGROUP_MODE;
        uint64_t g = static_cast<uint64_t>(scaleShapePtr->GetStorageShape().GetDim(1));
        OP_CHECK_IF(g == 0 || inputParams_.kSize % g != 0,
                    OP_LOGE(context_->GetNodeName(), "A4W4 pergroup: k(%lu) must be divisible by G(%lu).",
                            inputParams_.kSize, g),
                    return false);
        // PlanA: groupSize(=k/G) 必须为 128 倍数（baseK 对齐粒度，c0=32 倍数，L0B 约束）。
        // 详见 GMM_A5_A4W4_PERGROUPSIZE_CONSTRAINT.md §3。PlanB(放宽到 32 倍数)待验证，未启用。
        OP_CHECK_IF((inputParams_.kSize / g) % A4W4_QUANT_GROUP_SIZE != 0,
                    OP_LOGE(context_->GetNodeName(), "A4W4 pergroup: k/G(%lu) must be multiple of %u.",
                            inputParams_.kSize / g, A4W4_QUANT_GROUP_SIZE),
                    return false);
        a4w4Tiling_.quantGroupNum = static_cast<uint32_t>(g);
    } else if (scaleDimNum == 2U) {
        inputParams_.bQuantMode = QuantMode::PERCHANNEL_MODE;
        a4w4Tiling_.quantGroupNum = 1U;
    } else {
        OP_LOGE(context_->GetNodeName(), "A4W4: scale dim should be 2 or 3, got %u.", scaleDimNum);
        return false;
    }
    a4w4Tiling_.isPerTokenQuant = (inputParams_.aQuantMode == QuantMode::PERTOKEN_MODE) ? 1U : 0U;
    return true;
}

bool GroupedA4W4QuantTiling::CheckA4W4Params()
{
    OP_CHECK_IF(inputParams_.aDtype != ge::DT_INT4 || inputParams_.bDtype != ge::DT_INT4,
                OP_LOGE(context_->GetNodeName(), "A4W4: x/weight must be INT4."), return false);
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_UINT64,
                OP_LOGE(context_->GetNodeName(), "A4W4: scale must be UINT64, got %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                return false);
    OP_CHECK_IF(inputParams_.aQuantMode == QuantMode::PERTOKEN_MODE && inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "A4W4: perTokenScale must be FLOAT."), return false);
    OP_CHECK_IF(inputParams_.cDtype != ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "A4W4: y must be FP16/BF16."), return false);
    OP_CHECK_IF(inputParams_.nSize % 8UL != 0,
                OP_LOGE(context_->GetNodeName(), "A4W4: n(%lu) must be multiple of 8.", inputParams_.nSize),
                return false);
    OP_CHECK_IF(inputParams_.kSize == 0UL,
                OP_LOGE(context_->GetNodeName(), "A4W4: k must be > 0, got k=%lu.", inputParams_.kSize),
                return false);
    OP_CHECK_IF(inputParams_.kSize % 32UL != 0,
                OP_LOGE(context_->GetNodeName(), "A4W4: k(%lu) must be multiple of 32 (int8 C0_SIZE, L0B alignment).",
                        inputParams_.kSize),
                return false);
    OP_CHECK_IF(!inputParams_.isSingleX,
                OP_LOGE(context_->GetNodeName(), "A4W4 only supports single x."), return false);
    OP_CHECK_IF(!inputParams_.isSingleW,
                OP_LOGE(context_->GetNodeName(), "A4W4 only supports single weight."), return false);
    OP_CHECK_IF(!inputParams_.isSingleY,
                OP_LOGE(context_->GetNodeName(), "A4W4 only supports single y."), return false);
    // auto offsetDesc = context_->GetOptionalInputDesc(OFFSET_INDEX);
    // OP_CHECK_IF(offsetDesc != nullptr,
    //             OP_LOGE(context_->GetNodeName(), "A4W4: offset must be nullptr (symmetric quant)."), return false);
    // auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    // OP_CHECK_IF(biasDesc != nullptr,
    //             OP_LOGE(context_->GetNodeName(), "A4W4: bias must be nullptr."), return false);
    // auto antiquantScaleDesc = context_->GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX);
    // OP_CHECK_IF(antiquantScaleDesc != nullptr,
    //             OP_LOGE(context_->GetNodeName(), "A4W4: antiquantScale must be nullptr."), return false);
    // auto antiquantOffsetDesc = context_->GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
    // OP_CHECK_IF(antiquantOffsetDesc != nullptr,
    //             OP_LOGE(context_->GetNodeName(), "A4W4: antiquantOffset must be nullptr."), return false);
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        uint64_t innerAxis = inputParams_.transB ? inputParams_.kSize : inputParams_.nSize;
        OP_CHECK_IF(innerAxis % 64UL != 0,
                    OP_LOGE(context_->GetNodeName(), "A4W4: NZ weight inner axis(%lu) must be multiple of 64 (int4 32B).",
                            innerAxis),
                    return false);
    }
    return true;
}

uint32_t GroupedA4W4QuantTiling::FindBestSingleN() const
{
    if (a4w4Tiling_.quantGroupNum != 1U || inputParams_.groupType != GroupedMatmul::SPLIT_M) {
        return a4w4Tiling_.baseN;
    }
    if (!inputParams_.isSingleX || !inputParams_.isSingleW) {
        return a4w4Tiling_.baseN;
    }
    uint64_t totalM = a4w4Tiling_.m;
    if (totalM < static_cast<uint64_t>(a4w4Tiling_.baseM)) {
        return a4w4Tiling_.baseN;
    }
    int32_t mDim = static_cast<int32_t>(CeilDiv(totalM, static_cast<uint64_t>(a4w4Tiling_.baseM)));
    int32_t nDim = static_cast<int32_t>(CeilDiv(inputParams_.nSize, static_cast<uint64_t>(a4w4Tiling_.baseN)));
    int64_t taskNum = static_cast<int64_t>(mDim) * nDim * static_cast<int64_t>(a4w4Tiling_.groupNum);
    int64_t usedCore = static_cast<int64_t>(aicoreParams_.aicNum);
    int64_t taskNumPerCore = CeilDiv(taskNum, usedCore);
    if (taskNumPerCore <= 1) {
        return a4w4Tiling_.baseN;
    }
    for (uint32_t i = 1; i <= static_cast<uint32_t>(usedCore); ++i) {
        uint32_t candidate = static_cast<uint32_t>(CeilDiv(inputParams_.nSize, static_cast<uint64_t>(i)));
        if (candidate % a4w4Tiling_.baseN != 0) {
            continue;
        }
        int32_t curNDim = static_cast<int32_t>(CeilDiv(inputParams_.nSize, static_cast<uint64_t>(candidate)));
        int64_t curTaskNum = static_cast<int64_t>(mDim) * curNDim * static_cast<int64_t>(a4w4Tiling_.groupNum);
        float ratio = static_cast<float>(curTaskNum) / static_cast<float>(AlignUp(curTaskNum, usedCore));
        if (ratio >= EFFECTIVE_TASK_RATIO) {
            return candidate;
        }
    }
    return a4w4Tiling_.baseN;
}

ge::graphStatus GroupedA4W4QuantTiling::DoOpTiling()
{
    CalBasicBlockA4W4();
    OP_CHECK_IF(CalUbDivideA4W4() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "CalUbDivideA4W4 failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SetMKNList(), OP_LOGE(context_->GetNodeName(), "SetMKNList failed."),
                return ge::GRAPH_FAILED);
    a4w4Tiling_.singleN = FindBestSingleN();
    return ge::GRAPH_SUCCESS;
}

void GroupedA4W4QuantTiling::CalBasicBlockA4W4()
{
    a4w4Tiling_.baseM = A4W4_BASE_M_950;  // 128
    a4w4Tiling_.baseN = A4W4_BASE_N_950;  // 256
    // baseK：pergroup 固定 128（group 内 groupSize/128 次 Mmad + L0C int32 累加，group 边界 FixPipe）；
    //        perchannel min(K,128)（K 方向 iterate + L0C 累加，避免大 K 超 L0A/L0B）
    if (inputParams_.bQuantMode == QuantMode::PERGROUP_MODE) {
        a4w4Tiling_.baseK = A4W4_BASE_K_950;  // 128
    } else {
        a4w4Tiling_.baseK = std::min(static_cast<uint32_t>(inputParams_.kSize), A4W4_BASE_K_950);
    }
    a4w4Tiling_.usedCoreNum = static_cast<uint32_t>(aicoreParams_.aicNum);
}

ge::graphStatus GroupedA4W4QuantTiling::CalUbDivideA4W4()
{
    uint32_t ubSize = static_cast<uint32_t>(aicoreParams_.ubSize);  // 950: 248KB
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context_->GetNodeName(), "ubSize is 0."), return ge::GRAPH_FAILED);

    // ubCalSize = epilogue 单次计算元素数 = vecBaseM × alignBaseN（元素，非字节）
    uint32_t alignBaseN = AlignUp(a4w4Tiling_.baseN, 8U);           // 256（N%8 对齐，checker 已保证）
    uint32_t ubCalSize = A4W4_VEC_BASE_M_950 * alignBaseN;          // 16×256 = 4096 元素
    a4w4Tiling_.ubCalSize = ubCalSize;

    // epilogue UB(字节) = A4W4_EPILOGUE_UB_COEFF·ubCalSize + 8·baseM
    //   vecInQueue(half,DB2)=4·ubCalSize, vecOutQueue(half,DB2)=4·ubCalSize,
    //   tmpBuff 4×float=16·ubCalSize, perTokenScaleInQueue(float,DB2)=8·baseM
    uint32_t epilogueBytes = A4W4_EPILOGUE_UB_COEFF * ubCalSize + 8U * a4w4Tiling_.baseM;  // 99328
    OP_CHECK_IF(epilogueBytes >= ubSize,
                OP_LOGE(context_->GetNodeName(), "A4W4 epilogue UB overflow: %u >= %u.", epilogueBytes, ubSize),
                return ge::GRAPH_FAILED);

    // ubRestBytes 给 prologue（32B 对齐）
    uint32_t ubRestBytes = ubSize - epilogueBytes;                              // 154624
    ubRestBytes = ubRestBytes / UB_BLOCK_UNIT_SIZE * UB_BLOCK_UNIT_SIZE;        // 154592（32B 对齐）
    OP_CHECK_IF(ubRestBytes < PROLOGUE_MIN_BYTES_950,
                OP_LOGE(context_->GetNodeName(),
                        "A4W4 UB not enough for prologue: ubRestBytes=%u < %u. Consider smaller baseN.",
                        ubRestBytes, PROLOGUE_MIN_BYTES_950),
                return ge::GRAPH_FAILED);
    a4w4Tiling_.ubRestBytes = ubRestBytes;
    return ge::GRAPH_SUCCESS;
}

void GroupedA4W4QuantTiling::InitCommonL1TilingFields()
{
    a4w4Tiling_.stepM = 1UL;
    a4w4Tiling_.stepN = 1UL;
    a4w4Tiling_.singleCoreM = std::min(inputParams_.mSize, static_cast<uint64_t>(a4w4Tiling_.baseM));
    a4w4Tiling_.singleCoreN = std::min(inputParams_.nSize, static_cast<uint64_t>(a4w4Tiling_.baseN));
    a4w4Tiling_.singleCoreK = inputParams_.kSize;
    a4w4Tiling_.iterateOrder = 0U;
    a4w4Tiling_.dbL0c = (static_cast<uint64_t>(a4w4Tiling_.baseM) * a4w4Tiling_.baseN *
                         DATA_SIZE_L0C * DB_SIZE <= aicoreParams_.l0cSize) ? DB_SIZE : 1U;
}

ge::graphStatus GroupedA4W4QuantTiling::CalcLeftL1Size(uint64_t &leftL1Size) const
{
    leftL1Size = aicoreParams_.l1Size;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedA4W4QuantTiling::CalL1Tiling()
{
    InitCommonL1TilingFields();
    if (inputParams_.kSize == 0UL) { return ge::GRAPH_SUCCESS; }
    uint64_t leftL1Size = 0;
    OP_CHECK_IF(CalcLeftL1Size(leftL1Size) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalcLeftL1Size failed."), return ge::GRAPH_FAILED);
    return CalL1Depth(leftL1Size);
}

ge::graphStatus GroupedA4W4QuantTiling::CalL1Depth(uint64_t leftL1Size)
{
    // L0A/L0B 容量校验（双缓冲 dbL0A=dbL0B=2）：baseM·baseK·1·2 ≤ L0A, baseN·baseK·1·2 ≤ L0B
    OP_CHECK_IF(static_cast<uint64_t>(a4w4Tiling_.baseM) * a4w4Tiling_.baseK * DB_SIZE > aicoreParams_.l0aSize,
                OP_LOGE(context_->GetNodeName(), "A4W4 L0A overflow: need=%lu cap=%lu.",
                        static_cast<uint64_t>(a4w4Tiling_.baseM) * a4w4Tiling_.baseK * DB_SIZE, aicoreParams_.l0aSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(a4w4Tiling_.baseN) * a4w4Tiling_.baseK * DB_SIZE > aicoreParams_.l0bSize,
                OP_LOGE(context_->GetNodeName(), "A4W4 L0B overflow: need=%lu cap=%lu.",
                        static_cast<uint64_t>(a4w4Tiling_.baseN) * a4w4Tiling_.baseK * DB_SIZE, aicoreParams_.l0bSize),
                return ge::GRAPH_FAILED);

    // L1 depth 反推：对齐 block_mmad_a8w8_fixpipe_quant.h 的 fixed-tile 物理布局。
    // block L1 布局为 L1_BUFFER_NUM=2 stage，每 stage = AlignUp(A,32)+AlignUp(B,32)+scale（A+B 共享一个 stage，
    // 每 half L1 放一个 stage），约束 GetL1StageBytes(stepKa·baseK, stepKb·baseK) ≤ l1Size/2。
    // 物理足迹公式 SOURCE: ops-tensor include/blaze/gemm/utils/a8w4_fixed_tile_resource.h (逐字一致，合并 identical)。
    // 修正原 floorEven(leftL1/2/线性baseA) 在 NZ 对齐+scale+2 stage 下超 L1 ~4KB 的问题（@950 原 stepKa=8/stepKb=4 超）。
    constexpr uint64_t CUBE_BLOCK = 16UL;
    constexpr uint64_t INT8_C0 = 32UL;
    constexpr uint64_t DATA_BLOCK_BYTES = 32UL;
    constexpr uint64_t L1_BUFFER_NUM = 2UL;
    constexpr uint64_t elemBytes = 1UL;                          // int4->int8 后 cube 按 int8
    constexpr uint64_t scaleElemBytes = sizeof(uint64_t);        // weight scale uint64
    auto AlignUpL = [](uint64_t v, uint64_t a) -> uint64_t { return (v + a - 1UL) / a * a; };
    // transA=false: AlignUp(baseM,16)·AlignUp(kAL1,32); transB=false: AlignUp(kBL1,32)·AlignUp(baseN,16)
    auto GetAL1Bytes = [&](uint64_t kAL1) -> uint64_t {
        return AlignUpL(a4w4Tiling_.baseM, CUBE_BLOCK) * AlignUpL(kAL1, INT8_C0) * elemBytes; };
    auto GetBL1Bytes = [&](uint64_t kBL1) -> uint64_t {
        return AlignUpL(kBL1, INT8_C0) * AlignUpL(a4w4Tiling_.baseN, CUBE_BLOCK) * elemBytes; };
    auto GetScaleL1Bytes = [&]() -> uint64_t {
        return AlignUpL(a4w4Tiling_.baseN * scaleElemBytes, DATA_BLOCK_BYTES); };
    auto GetL1StageBytes = [&](uint64_t kAL1, uint64_t kBL1) -> uint64_t {
        return AlignUpL(GetAL1Bytes(kAL1), DATA_BLOCK_BYTES) +
               AlignUpL(GetBL1Bytes(kBL1), DATA_BLOCK_BYTES) + GetScaleL1Bytes(); };  // isBias=false

    const uint64_t halfL1 = leftL1Size / L1_BUFFER_NUM;
    const uint64_t baseAPhys = GetAL1Bytes(a4w4Tiling_.baseK);   // 单 baseK 份 A 物理大小
    const uint64_t baseBPhys = GetBL1Bytes(a4w4Tiling_.baseK);
    auto floorEven = [](uint64_t v) -> uint64_t { return v / static_cast<uint64_t>(DB_SIZE) * static_cast<uint64_t>(DB_SIZE); };
    // 候选上界（沿原对半分思想），再降档满足 stage ≤ halfL1
    uint64_t depthA1 = floorEven(halfL1 / baseAPhys);
    uint64_t depthB1 = floorEven(halfL1 / baseBPhys);
    while (depthA1 >= DB_SIZE && depthB1 >= DB_SIZE) {
        uint64_t stepKa = depthA1 / DB_SIZE;
        uint64_t stepKb = depthB1 / DB_SIZE;
        if (GetL1StageBytes(stepKa * a4w4Tiling_.baseK, stepKb * a4w4Tiling_.baseK) <= halfL1) {
            break;
        }
        if (baseAPhys * depthA1 >= baseBPhys * depthB1) {  // 降占用大的一侧
            depthA1 -= DB_SIZE;
        } else {
            depthB1 -= DB_SIZE;
        }
    }
    OP_CHECK_IF(depthA1 < DB_SIZE || depthB1 < DB_SIZE,
                OP_LOGE(context_->GetNodeName(),
                        "A4W4 L1 overflow: cannot fit one fixed-tile stage. baseM=%u baseN=%u baseK=%u halfL1=%lu.",
                        a4w4Tiling_.baseM, a4w4Tiling_.baseN, a4w4Tiling_.baseK, halfL1),
                return ge::GRAPH_FAILED);
    a4w4Tiling_.depthA1 = depthA1;
    a4w4Tiling_.depthB1 = depthB1;
    a4w4Tiling_.stepKa = depthA1 / DB_SIZE;
    a4w4Tiling_.stepKb = depthB1 / DB_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedA4W4QuantTiling::DoLibApiTiling()
{
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedA4W4QuantTiling::CalWorkspaceA4W4()
{
    // int4->int8 预转换：weight 与 x 均落 workspace（体积翻倍 int4 0.5B -> int8 1B）
    int8WeightWs_ = static_cast<uint64_t>(a4w4Tiling_.groupNum) * a4w4Tiling_.k * a4w4Tiling_.n * sizeof(int8_t);
    int8XWs_ = static_cast<uint64_t>(a4w4Tiling_.m) * a4w4Tiling_.k * sizeof(int8_t);  // m=totalM (single x)
    mmOutWs_ = static_cast<uint64_t>(A4W4_MMOUT_PIPELINE) * a4w4Tiling_.baseM * a4w4Tiling_.baseN *
               a4w4Tiling_.usedCoreNum * sizeof(uint16_t);
    // F3: perTokenScale 可选兜底。isPerTokenQuant=0 (无 perTokenScale 输入) 时构造全 1 段 (totalM·float),
    //     epilogue 正常 mul (乘 1 等价跳过), 保持 A8W4 epilogue 逐字同步不改。
    perTokenScaleFillWs_ = (a4w4Tiling_.isPerTokenQuant == 0U)
        ? static_cast<uint64_t>(a4w4Tiling_.m) * sizeof(float) : 0UL;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedA4W4QuantTiling::GetWorkspaceSize()
{
    OP_CHECK_IF(CalWorkspaceA4W4() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "CalWorkspaceA4W4 failed."),
                return ge::GRAPH_FAILED);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    // 布局: [sys 16MB][int8WeightWs][int8XWs][mmOutWs][perTokenScaleFillWs?]  (?=isPerTokenQuant=0)
    workspaces[0] = SYS_WORKSPACE_SIZE + int8WeightWs_ + int8XWs_ + mmOutWs_ + perTokenScaleFillWs_;
    OP_LOGI(context_->GetNodeName(), "A4W4 workspace: int8Weight=%lu int8X=%lu mmOut=%lu perTokenFill=%lu total=%lu",
            int8WeightWs_, int8XWs_, mmOutWs_, perTokenScaleFillWs_, workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedA4W4QuantTiling::GetTilingKey() const
{
    constexpr uint64_t KERNEL_A4W4_MIX = 3UL;
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB ? 1 : 0), 0UL, KERNEL_A4W4_MIX);
}

ge::graphStatus GroupedA4W4QuantTiling::PostTiling()
{
    using namespace GroupedMatmulTilingData;
    auto &p = tilingData_.gmmA4W4Params;
    p.m = static_cast<uint32_t>(a4w4Tiling_.m);
    p.n = static_cast<uint32_t>(a4w4Tiling_.n);
    p.k = static_cast<uint32_t>(a4w4Tiling_.k);
    p.groupNum = a4w4Tiling_.groupNum;
    p.coreNum = a4w4Tiling_.usedCoreNum;
    p.baseM = a4w4Tiling_.baseM;
    p.baseN = a4w4Tiling_.baseN;
    p.baseK = a4w4Tiling_.baseK;
    p.ubCalSize = a4w4Tiling_.ubCalSize;
    p.ubRestBytes = a4w4Tiling_.ubRestBytes;
    p.quantGroupNum = a4w4Tiling_.quantGroupNum;
    p.singleN = a4w4Tiling_.singleN;
    p.isPerTokenQuant = a4w4Tiling_.isPerTokenQuant;
    p.isA4W4Optimize = a4w4Tiling_.isA4W4Optimize;
    p.groupType = a4w4Tiling_.groupType;
    p.groupListType = a4w4Tiling_.groupListType;
    // gmmArray (mList_/kList_/nList_ inherited)
    errno_t retM = memcpy_s(tilingData_.gmmArray.mList, sizeof(tilingData_.gmmArray.mList), mList_, sizeof(mList_));
    OP_CHECK_IF(retM != EOK, OP_LOGE(context_->GetNodeName(), "memcpy_s mList failed, ret=%d", retM),
                return ge::GRAPH_FAILED);
    errno_t retK = memcpy_s(tilingData_.gmmArray.kList, sizeof(tilingData_.gmmArray.kList), kList_, sizeof(kList_));
    OP_CHECK_IF(retK != EOK, OP_LOGE(context_->GetNodeName(), "memcpy_s kList failed, ret=%d", retK),
                return ge::GRAPH_FAILED);
    errno_t retN = memcpy_s(tilingData_.gmmArray.nList, sizeof(tilingData_.gmmArray.nList), nList_, sizeof(nList_));
    OP_CHECK_IF(retN != EOK, OP_LOGE(context_->GetNodeName(), "memcpy_s nList failed, ret=%d", retN),
                return ge::GRAPH_FAILED);
    // mmTilingData(TCubeTiling)
    auto &mm = tilingData_.mmTilingData;
    mm.M = inputParams_.mSize;
    mm.N = inputParams_.nSize;
    mm.Ka = inputParams_.kSize;
    mm.Kb = inputParams_.kSize;
    mm.usedCoreNum = aicoreParams_.aicNum;
    mm.baseM = a4w4Tiling_.baseM;
    mm.baseN = a4w4Tiling_.baseN;
    mm.baseK = a4w4Tiling_.baseK;
    mm.singleCoreM = a4w4Tiling_.singleCoreM;
    mm.singleCoreN = a4w4Tiling_.singleCoreN;
    mm.singleCoreK = a4w4Tiling_.singleCoreK;
    mm.depthA1 = a4w4Tiling_.depthA1;
    mm.depthB1 = a4w4Tiling_.depthB1;
    mm.stepM = a4w4Tiling_.stepM;
    mm.stepN = a4w4Tiling_.stepN;
    mm.stepKa = a4w4Tiling_.stepKa;
    mm.stepKb = a4w4Tiling_.stepKb;
    mm.isBias = 0;
    mm.iterateOrder = a4w4Tiling_.iterateOrder;
    mm.dbL0A = 2;
    mm.dbL0B = 2;
    mm.dbL0C = a4w4Tiling_.dbL0c;
    OP_LOGI(context_->GetNodeName(),
            "A4W4 tiling: baseM=%u baseN=%u baseK=%u ubCalSize=%u ubRestBytes=%u quantGroupNum=%u isPerToken=%u "
            "depthA1=%lu depthB1=%lu",
            p.baseM, p.baseN, p.baseK, p.ubCalSize, p.ubRestBytes, p.quantGroupNum, p.isPerTokenQuant,
            a4w4Tiling_.depthA1, a4w4Tiling_.depthB1);
    return SaveTilingDataToContext(tilingData_);
}

}  // namespace optiling
