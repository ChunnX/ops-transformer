/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_quant_basic_api_matmul_tiling.cpp
 * \brief
 */
#include "grouped_quant_basic_api_matmul_tiling.h"
#include <limits>
#include "blaze/gemm/utils/a8w4_fixed_tile_resource.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h"
using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmul;
using namespace optiling::GmmConstant;
using GMMQuantBasicApiTilingData = GroupedMatmulTilingData::GMMQuantBasicApiTilingData;
namespace optiling {
GroupedQmmBasicApiTiling::GroupedQmmBasicApiTiling(gert::TilingContext *context)
    : GroupedQmmTiling(context)
{
    Reset();
}

bool GroupedQmmBasicApiTiling::IsCapable()
{
    // MX 量化：scale 为 FLOAT8_E8M0（IsMicroScaling）；支持 mxfp8 与 mxfp4
    if (!IsMicroScaling()) {
        return false;
    }
    const bool isMxFp8 =
        inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2;
    const bool isMxFp4 =
        inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2;
    return isMxFp8 || isMxFp4;
}

void GroupedQmmBasicApiTiling::Reset()
{
    tilingData_ = GMMQuantBasicApiTilingData();
}

ge::graphStatus GroupedQmmBasicApiTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmTiling::GetShapeAttrsInfo();
}

ge::graphStatus GroupedQmmBasicApiTiling::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.activeType = inputParams_.actType;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    tilingData_.gmmQuantParams.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    tilingData_.gmmQuantParams.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    tilingData_.gmmQuantParams.groupType = static_cast<int8_t>(inputParams_.groupType);
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmQuantParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias);
    OP_LOGD(inputParams_.opName, "%ld", LogQuantParams(tilingData_.gmmQuantParams));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedQmmBasicApiTiling::DoLibApiTiling()
{
    GroupedQmmTiling::CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"),
                return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.m = inputParams_.mSize;
    tilingData_.mmTilingData.n = inputParams_.nSize;
    tilingData_.mmTilingData.k = inputParams_.kSize;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.kAL1 = basicTiling_.stepKa * basicTiling_.baseK;
    tilingData_.mmTilingData.kBL1 = basicTiling_.stepKb * basicTiling_.baseK;
    tilingData_.mmTilingData.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        tilingData_.mmTilingData.scaleKAL1 = std::min(
            std::max(basicTiling_.scaleFactorA * basicTiling_.stepKa, basicTiling_.scaleFactorB * basicTiling_.stepKb) *
                basicTiling_.baseK,
            inputParams_.kSize);
        tilingData_.mmTilingData.scaleKBL1 = tilingData_.mmTilingData.scaleKAL1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedQmmBasicApiTiling::PostTiling()
{
    return SaveTilingDataToContext(tilingData_);
}

ge::graphStatus GroupedQmmBasicApiTiling::CalL1Tiling()
{
    InitCommonL1TilingFields();
    uint64_t leftL1Size = 0;
    OP_CHECK_IF(CalcLeftL1Size(leftL1Size) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalcLeftL1Size failed"), return ge::GRAPH_FAILED);
    return CalL1Depth(leftL1Size);
}

ge::graphStatus GroupedQmmBasicApiTiling::CalL1Depth(uint64_t leftL1Size)
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);

    uint64_t baseScaleASize = 0;
    uint64_t baseScaleBSize = 0;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        CalcAlignedMxBaseScaleSize(baseScaleASize, baseScaleBSize);
    }
    uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
    OP_CHECK_IF(leftL1Size < baseL1Size,
                OP_LOGE(context_->GetNodeName(), "L1 space overflow. Free L1Size : %lu, used space: %lu", leftL1Size,
                        baseL1Size),
                return ge::GRAPH_FAILED);
    uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL); // 求A+B和的平均depth
    // 根据一条指令带宽要求的数据量求取A,B各自的depth
    basicTiling_.depthA1 = GetDepthWithHighBW(std::min(inputParams_.mSize, basicTiling_.baseM));
    basicTiling_.depthB1 = GetDepthWithHighBW(std::min(inputParams_.nSize, basicTiling_.baseN));
    // 如果按照满足带宽的L1数据量超过了L1Size，进行下调整到平均depth;适配mx低阶api scaleKAL1=scaleKBL1的约束
    if (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize +
            std::max(basicTiling_.depthA1, basicTiling_.depthB1) * (baseScaleASize + baseScaleBSize) >
        leftL1Size) {
        basicTiling_.depthA1 = depthInit;
        basicTiling_.depthB1 = depthInit;
    }
    // 用剩余L1空间对内轴ND非对齐场景进行调整depth
    ModifyDepthForUnalign(leftL1Size, baseASize, baseBSize, baseScaleASize + baseScaleBSize);
    CalStepKs();
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        return CalScaleFactors();
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedQmmBasicApiTiling::GetDepthWithHighBW(uint64_t mnL1) const
{
    // 只需要满足读GM数据大于64KB即可获得较高的带宽，不一定要把L1用满，同时减少MTE2头开销
    uint64_t baseKSize = GetSizeWithDataType(basicTiling_.baseK, inputParams_.aDtype);
    uint64_t depth =
        CeilAlign(CeilDiv(MTE2_MIN_LOAD_SIZE_V120, mnL1), static_cast<uint64_t>(GmmConstant::BASIC_BLOCK_SIZE_256)) /
        baseKSize * DB_SIZE;
    uint64_t pow2Depth = POWER_OF_TWO;
    while (pow2Depth < depth) {
        pow2Depth *= POWER_OF_TWO;
    }
    // 对齐2次幂或者实际最大depth大小
    return std::min(pow2Depth, CeilDiv(inputParams_.kSize, basicTiling_.baseK) * DB_SIZE);
}

void GroupedQmmBasicApiTiling::ModifyDepthForUnalign(uint64_t leftL1Size, uint64_t baseASize, uint64_t baseBSize,
                                                     uint64_t baseScaleABSize)
{
    // 只调整K轴非对齐场景
    if (inputParams_.kSize % GmmConstant::BASIC_BLOCK_SIZE_128 == 0) {
        return;
    }
    // m，n在内轴且ND时，修改stepk无法改变ND2NZ小包数量
    if (inputParams_.transA && (!inputParams_.transB || inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ)) {
        return;
    }
    if (!inputParams_.transA) {
        if (basicTiling_.depthA1 <= basicTiling_.depthB1) {
            uint64_t leftASize = leftL1Size - basicTiling_.depthB1 * baseBSize - basicTiling_.depthB1 * baseScaleABSize;
            while (basicTiling_.depthA1 * POWER_OF_TWO * baseASize <= leftASize) {
                basicTiling_.depthA1 *= POWER_OF_TWO;
            }
            if (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize +
                    std::max(basicTiling_.depthA1, basicTiling_.depthB1) * baseScaleABSize >
                leftL1Size) {
                basicTiling_.depthA1 = basicTiling_.depthB1;
            }
        } else if (inputParams_.transB && inputParams_.bFormat == ge::FORMAT_ND) {
            uint64_t leftBSize = leftL1Size - basicTiling_.depthA1 * baseASize - basicTiling_.depthA1 * baseScaleABSize;
            while (basicTiling_.depthB1 * POWER_OF_TWO * baseBSize <= leftBSize) {
                basicTiling_.depthB1 *= POWER_OF_TWO;
            }
            if (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize +
                    std::max(basicTiling_.depthA1, basicTiling_.depthB1) * baseScaleABSize >
                leftL1Size) {
                basicTiling_.depthB1 = basicTiling_.depthA1;
            }
        }
    } else { // transA = true, transB = true, 仅考虑B depth
        while ((basicTiling_.depthA1 * baseASize -
                std::max(basicTiling_.depthA1, basicTiling_.depthB1 * POWER_OF_TWO) * baseScaleABSize) < leftL1Size) {
            basicTiling_.depthB1 *= POWER_OF_TWO;
        }
    }
}

ge::graphStatus GroupedQmmBasicApiTiling::CalScaleFactors()
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);
    uint64_t baseScaleASize = GetSizeWithDataType(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseM,
                                                  inputParams_.perTokenScaleDtype);
    uint64_t baseScaleBSize =
        GetSizeWithDataType(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseN, inputParams_.scaleDtype);
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t baseBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t leftL1Size =
        aicoreParams_.l1Size - (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit = static_cast<uint32_t>(
        leftL1Size / (std::max(basicTiling_.depthA1, basicTiling_.depthB1) * (baseScaleASize + baseScaleBSize)));
    OP_CHECK_IF(
        scaleInit == 0,
        OP_LOGE(context_->GetNodeName(),
                "When m(%lu)/n(%lu)/k(%lu)/groupNum(%lu) in mx quant mode, scaleFactor should not be equal to 0.",
                inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupNum),
        return ge::GRAPH_FAILED);
    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    uint32_t scaleFactorAMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleASize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorBMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleBSize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorA =
        static_cast<uint32_t>(CeilDiv(inputParams_.kSize, basicTiling_.stepKa * basicTiling_.baseK));
    uint32_t scaleFactorB =
        static_cast<uint32_t>(CeilDiv(inputParams_.kSize, basicTiling_.stepKb * basicTiling_.baseK));
    basicTiling_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    basicTiling_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    basicTiling_.scaleFactorA = std::min(scaleFactorAMax, basicTiling_.scaleFactorA);
    basicTiling_.scaleFactorB = std::min(scaleFactorBMax, basicTiling_.scaleFactorB);

    // 来自L1 size 的约束
    if (basicTiling_.scaleFactorA > scaleInit && basicTiling_.scaleFactorB > scaleInit) { // 非scalek全载，ka/kb倍数
        if (basicTiling_.depthA1 >= basicTiling_.depthB1) {
            basicTiling_.scaleFactorA = scaleInit;
            basicTiling_.scaleFactorB = scaleInit * basicTiling_.depthA1 / basicTiling_.depthB1;
        } else {
            basicTiling_.scaleFactorA = scaleInit * basicTiling_.depthB1 / basicTiling_.depthA1;
            basicTiling_.scaleFactorB = scaleInit;
        }
    }
    return ge::GRAPH_SUCCESS;
}

GroupedA8W4BasicApiTiling::GroupedA8W4BasicApiTiling(gert::TilingContext *context)
    : GroupedQmmTiling(context)
{
    Reset();
}

void GroupedA8W4BasicApiTiling::Reset()
{
    hasOffset_ = false;
    specialWeightFormat_ = false;
    weightPackedInt32_ = false;
    expectedTokenNum_ = 0;
    userWorkspaceSize_ = 0;
    dequantMode_ = DequantMode::SYMMETRIC_PER_GROUP;
    tilingData_ = GroupedMatmulTilingData::GMMA8W4BasicApiTilingData();
}

bool GroupedA8W4BasicApiTiling::IsCapable()
{
    return inputParams_.aDtype == ge::DT_INT8 &&
           (inputParams_.bDtype == ge::DT_INT4 || inputParams_.bDtype == ge::DT_INT32) &&
           inputParams_.scaleDtype == ge::DT_UINT64;
}

ge::graphStatus GroupedA8W4BasicApiTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    Reset();
    return GroupedQmmTiling::GetShapeAttrsInfo();
}

bool GroupedA8W4BasicApiTiling::AnalyzeDtype()
{
    auto xDesc = context_->GetDynamicInputDesc(X_INDEX, 0);
    auto weightDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    auto outDesc = context_->GetOutputDesc(Y_INDEX);
    auto scaleDesc = context_->GetDynamicInputDesc(SCALE_INDEX, 0);
    auto perTokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(xDesc == nullptr || weightDesc == nullptr || outDesc == nullptr || scaleDesc == nullptr ||
                    perTokenScaleDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi requires x, weight, scale, perTokenScale and out."),
                return false);

    inputParams_.aDtype = xDesc->GetDataType();
    inputParams_.bDtype = weightDesc->GetDataType();
    inputParams_.cDtype = outDesc->GetDataType();
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    inputParams_.perTokenScaleDtype = perTokenScaleDesc->GetDataType();
    inputParams_.aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(weightDesc->GetStorageFormat()));
    // Match the legacy A8W4 classification: standard FRACTAL_NZ selects NZ;
    // every base format accepted by the API checker uses the ND layout path.
    if (inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ) {
        inputParams_.bFormat = ge::FORMAT_ND;
    }

    OP_CHECK_IF(inputParams_.aDtype != ge::DT_INT8 ||
                    (inputParams_.bDtype != ge::DT_INT4 && inputParams_.bDtype != ge::DT_INT32),
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi expects x INT8 and weight INT4/packed INT32."),
                return false);
    weightPackedInt32_ = inputParams_.bDtype == ge::DT_INT32;
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_UINT64,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi expects scale UINT64."), return false);
    OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi expects perTokenScale FLOAT."), return false);
    OP_CHECK_IF(inputParams_.cDtype != ge::DT_BF16 && inputParams_.cDtype != ge::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi expects out BF16 or FLOAT16."), return false);
    OP_CHECK_IF(inputParams_.aFormat != ge::FORMAT_ND,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi expects x in ND format."), return false);
    OP_CHECK_IF(inputParams_.bFormat != ge::FORMAT_ND && inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi expects weight in ND or FRACTAL_NZ format."),
                return false);

    auto biasShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    inputParams_.hasBias = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    auto biasDesc = context_->GetDynamicInputDesc(BIAS_INDEX, 0);
    OP_CHECK_IF(!inputParams_.hasBias || biasDesc == nullptr || biasDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi requires bias FLOAT with shape [E,N]."),
                return false);
    inputParams_.biasDtype = biasDesc->GetDataType();

    auto offsetShape = context_->GetDynamicInputShape(OFFSET_INDEX, 0);
    hasOffset_ = offsetShape != nullptr && offsetShape->GetStorageShape().GetShapeSize() != 0;
    if (hasOffset_) {
        auto offsetDesc = context_->GetDynamicInputDesc(OFFSET_INDEX, 0);
        OP_CHECK_IF(offsetDesc == nullptr || offsetDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "Asymmetric A8W4 BasicApi expects offset FLOAT."), return false);
    }
    dequantMode_ =
        hasOffset_ ? DequantMode::ASYMMETRIC_PER_CHANNEL : DequantMode::SYMMETRIC_PER_GROUP;
    OP_CHECK_IF(hasOffset_ && inputParams_.cDtype == ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "BF16 A8W4 BasicApi does not support a non-null offset."),
                return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const int64_t *splitItemPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SPLIT_ITEM);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
    const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
    const int64_t *actTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);
    const auto tuningConfigPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG);

    inputParams_.splitItem = splitItemPtr != nullptr ? static_cast<int8_t>(*splitItemPtr) : inputParams_.splitItem;
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    inputParams_.transA = transposeXPtr != nullptr ? *transposeXPtr : false;
    inputParams_.groupType =
        groupTypePtr != nullptr ? static_cast<int8_t>(*groupTypePtr) : inputParams_.groupType;
    inputParams_.actType = actTypePtr != nullptr ? static_cast<int8_t>(*actTypePtr) : inputParams_.actType;
    // The documented A8W4 path does not inspect this attribute and always treats groupList as count mode.
    inputParams_.groupListType = 1;
    if (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > 0) {
        const auto tuningConfig = reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData());
        OP_CHECK_IF(tuningConfig[0] < 0 || static_cast<uint64_t>(tuningConfig[0]) > UINT32_MAX,
                    OP_LOGE(context_->GetNodeName(), "tuningConfig[0] must be in [0, UINT32_MAX]."), return false);
        expectedTokenNum_ = static_cast<uint32_t>(tuningConfig[0]);
        specialWeightFormat_ = tuningConfigPtr->GetSize() > 1 && tuningConfig[1] == 1;
    }
    inputParams_.isSingleX = context_->GetDynamicInputDesc(X_INDEX, 1) == nullptr;
    inputParams_.isSingleW = context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) == nullptr;
    inputParams_.isSingleY = inputParams_.splitItem == 2 || inputParams_.splitItem == 3;

    OP_CHECK_IF(inputParams_.groupType != SPLIT_M || inputParams_.actType != 0 || inputParams_.transA ||
                    inputParams_.transB,
                OP_LOGE(context_->GetNodeName(),
                        "A8W4 BasicApi expects groupType=0, actType=0 and no transpose."),
                return false);
    OP_CHECK_IF(specialWeightFormat_ && inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context_->GetNodeName(),
                        "tuningConfig[1]=1 expects [E,N,K] weight converted to FRACTAL_NZ."),
                return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::CheckTensorLists() const
{
    OP_CHECK_IF(context_->GetDynamicInputDesc(X_INDEX, 1) != nullptr ||
                    context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) != nullptr ||
                    context_->GetDynamicInputDesc(SCALE_INDEX, 1) != nullptr,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi only supports length-1 x/weight/scale TensorLists."),
                return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::CheckCommonShapes(const gert::Shape &xShape, const gert::Shape &wShape) const
{
    OP_CHECK_IF(xShape.GetDimNum() != 2 || wShape.GetDimNum() != 3,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi expects 2D x and 3D weight."), return false);
    OP_CHECK_IF(inputParams_.mSize == 0 || inputParams_.groupNum == 0,
                OP_LOGE(context_->GetNodeName(), "M and groupNum must be positive."), return false);
    OP_CHECK_IF(static_cast<uint64_t>(wShape.GetDim(0)) != inputParams_.groupNum,
                OP_LOGE(context_->GetNodeName(), "weight axis 0 must equal groupNum."), return false);
    OP_CHECK_IF(inputParams_.kSize == 0, OP_LOGE(context_->GetNodeName(), "K must be positive."), return false);
    if (dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP) {
        OP_CHECK_IF(inputParams_.kSize > 18432UL || inputParams_.kSize % QUANT_GROUP_SIZE != 0,
                    OP_LOGE(context_->GetNodeName(),
                            "Symmetric per-group K must not exceed 18432 and must be divisible by 256."),
                    return false);
    }
    OP_CHECK_IF(inputParams_.nSize == 0 || inputParams_.nSize % 8UL != 0,
                OP_LOGE(context_->GetNodeName(), "N must be positive and divisible by 8."), return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::SetLogicalMkn(const gert::Shape &xShape, const gert::Shape &wShape)
{
    auto outStorageShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(outStorageShape == nullptr || xShape.GetDimNum() != 2 || wShape.GetDimNum() != 3,
                OP_LOGE(context_->GetNodeName(), "Invalid x/weight/out shape."), return false);
    const gert::Shape &outShape = outStorageShape->GetOriginShape();
    OP_CHECK_IF(outShape.GetDimNum() != 2,
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi expects out [M,N]."), return false);

    inputParams_.mSize = static_cast<uint64_t>(xShape.GetDim(0));
    inputParams_.kSize = static_cast<uint64_t>(xShape.GetDim(1));
    inputParams_.nSize = static_cast<uint64_t>(outShape.GetDim(1));
    OP_CHECK_IF(static_cast<uint64_t>(outShape.GetDim(0)) != inputParams_.mSize,
                OP_LOGE(context_->GetNodeName(), "out axis 0 must equal M."), return false);

    // OriginShape remains the logical INT4 shape even when storage dtype is packed INT32.
    const uint64_t logicalK = static_cast<uint64_t>(wShape.GetDim(specialWeightFormat_ ? 2 : 1));
    const uint64_t logicalN = static_cast<uint64_t>(wShape.GetDim(specialWeightFormat_ ? 1 : 2));
    OP_CHECK_IF(logicalK != inputParams_.kSize || logicalN != inputParams_.nSize,
                OP_LOGE(context_->GetNodeName(),
                        "weight logical shape mismatches x/out; expected K=%lu,N=%lu, actual K=%lu,N=%lu.",
                        inputParams_.kSize, inputParams_.nSize, logicalK, logicalN),
                return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::CheckBiasShape() const
{
    auto biasShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    OP_CHECK_IF(!inputParams_.hasBias || biasShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "bias is required."), return false);
    const gert::Shape &shape = biasShape->GetOriginShape();
    OP_CHECK_IF(shape.GetDimNum() != 2 || static_cast<uint64_t>(shape.GetDim(0)) != inputParams_.groupNum ||
                    static_cast<uint64_t>(shape.GetDim(1)) != inputParams_.nSize,
                OP_LOGE(context_->GetNodeName(), "bias shape must be [groupNum, N]."), return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::CheckPerTokenScaleShape() const
{
    auto perTokenScaleShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(perTokenScaleShape == nullptr, OP_LOGE(context_->GetNodeName(), "perTokenScale shape is nullptr."),
                return false);
    const gert::Shape &shape = perTokenScaleShape->GetOriginShape();
    OP_CHECK_IF(shape.GetDimNum() != 1 || static_cast<uint64_t>(shape.GetDim(0)) != inputParams_.mSize,
                OP_LOGE(context_->GetNodeName(), "perTokenScale shape must be [M]."), return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::CheckAntiquantInputsEmpty() const
{
    auto antiquantScaleShape = context_->GetDynamicInputShape(ANTIQUANT_SCALE_INDEX, 0);
    auto antiquantOffsetShape = context_->GetDynamicInputShape(ANTIQUANT_OFFSET_INDEX, 0);
    const bool hasAntiquantScale =
        antiquantScaleShape != nullptr && antiquantScaleShape->GetStorageShape().GetShapeSize() != 0;
    const bool hasAntiquantOffset =
        antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetShapeSize() != 0;
    OP_CHECK_IF(hasAntiquantScale || hasAntiquantOffset,
                OP_LOGE(context_->GetNodeName(),
                        "A8W4 BasicApi requires antiquantScale and antiquantOffset to be empty."),
                return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::CheckWeightStorageShape(const gert::StorageShape &weightShape)
{
    uint64_t scalarSlotNum = inputParams_.groupNum * inputParams_.kSize * inputParams_.nSize;
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        const uint64_t alignedK = specialWeightFormat_ ?
                                      CeilAlign(inputParams_.kSize, 64UL) :
                                      CeilAlign(inputParams_.kSize, 16UL);
        const uint64_t alignedN = specialWeightFormat_ ?
                                      CeilAlign(inputParams_.nSize, 16UL) :
                                      CeilAlign(inputParams_.nSize, 64UL);
        scalarSlotNum = inputParams_.groupNum * alignedK * alignedN;
    }
    const uint64_t actualStorageElements =
        static_cast<uint64_t>(weightShape.GetStorageShape().GetShapeSize());
    // aclnnGroupedMatmulV5 normalizes a packed INT32 carrier to logical INT4 before Host tiling.
    // For ND weights the 8:1 physical storage shape is intentionally retained, so dtype alone is
    // insufficient to recover the carrier representation at this point.
    if (!weightPackedInt32_ && inputParams_.bDtype == ge::DT_INT4 && scalarSlotNum % 8UL == 0 &&
        actualStorageElements == scalarSlotNum / 8UL) {
        weightPackedInt32_ = true;
    }
    OP_CHECK_IF(weightPackedInt32_ && scalarSlotNum % 8UL != 0,
                OP_LOGE(context_->GetNodeName(), "Packed INT32 weight requires the physical INT4 slot count "
                                                     "to be divisible by 8."),
                return false);
    const uint64_t expectedStorageElements = weightPackedInt32_ ? scalarSlotNum / 8UL : scalarSlotNum;
    OP_CHECK_IF(actualStorageElements != expectedStorageElements,
                OP_LOGE(context_->GetNodeName(),
                        "Weight storage shape mismatch: expected %lu physical elements for logical INT4, got %lu.",
                        expectedStorageElements, actualStorageElements),
                return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::CheckScaleAndOffsetShapes()
{
    auto scaleShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleShape == nullptr, OP_LOGE(context_->GetNodeName(), "scale shape is nullptr."), return false);
    const gert::Shape &shape = scaleShape->GetOriginShape();
    const uint64_t expectedScaleGroups =
        dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? inputParams_.kSize / QUANT_GROUP_SIZE : 1UL;
    OP_CHECK_IF(shape.GetDimNum() != 3 || static_cast<uint64_t>(shape.GetDim(0)) != inputParams_.groupNum ||
                    static_cast<uint64_t>(shape.GetDim(1)) != expectedScaleGroups ||
                    static_cast<uint64_t>(shape.GetDim(2)) != inputParams_.nSize,
                OP_LOGE(context_->GetNodeName(), "scale shape must be [groupNum, quantGroupNum, N]."), return false);

    if (!hasOffset_) {
        return true;
    }
    auto offsetShape = context_->GetDynamicInputShape(OFFSET_INDEX, 0);
    OP_CHECK_IF(offsetShape == nullptr, OP_LOGE(context_->GetNodeName(), "offset shape is nullptr."), return false);
    const gert::Shape &offset = offsetShape->GetOriginShape();
    OP_CHECK_IF(offset.GetDimNum() != 3 || static_cast<uint64_t>(offset.GetDim(0)) != inputParams_.groupNum ||
                    offset.GetDim(1) != 1 || static_cast<uint64_t>(offset.GetDim(2)) != inputParams_.nSize,
                OP_LOGE(context_->GetNodeName(), "offset shape must be [groupNum, 1, N]."), return false);
    return true;
}

bool GroupedA8W4BasicApiTiling::AnalyzeInputs()
{
    OP_CHECK_IF(!CheckTensorLists(), OP_LOGE(context_->GetNodeName(), "CheckTensorLists failed."), return false);
    auto xStorageShape = context_->GetDynamicInputShape(X_INDEX, 0);
    auto weightStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(xStorageShape == nullptr || weightStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "x or weight shape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    const gert::Shape &weightShape = weightStorageShape->GetOriginShape();

    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(context_->GetNodeName(), "SetGroupNum failed."),
                return false);
    OP_CHECK_IF(!SetLogicalMkn(xShape, weightShape), OP_LOGE(context_->GetNodeName(), "SetLogicalMkn failed."),
                return false);
    OP_CHECK_IF(expectedTokenNum_ > inputParams_.mSize,
                OP_LOGE(context_->GetNodeName(), "tuningConfig[0] must not exceed M."), return false);
    OP_CHECK_IF(!CheckCommonShapes(xShape, weightShape) || !CheckWeightStorageShape(*weightStorageShape) ||
                    !CheckBiasShape() || !CheckPerTokenScaleShape() || !CheckAntiquantInputsEmpty() ||
                    !CheckScaleAndOffsetShapes(),
                OP_LOGE(context_->GetNodeName(), "A8W4 BasicApi shape validation failed."), return false);

    inputParams_.aQuantMode = QuantMode::PERTOKEN_MODE;
    inputParams_.bQuantMode = dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ?
                                  QuantMode::PERGROUP_MODE :
                                  QuantMode::PERCHANNEL_MODE;
    inputParams_.kernelType = 0UL;
    return true;
}

ge::graphStatus GroupedA8W4BasicApiTiling::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.activeType = inputParams_.actType;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    tilingData_.gmmQuantParams.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    tilingData_.gmmQuantParams.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    tilingData_.gmmQuantParams.groupType = inputParams_.groupType;
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmQuantParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias);
    tilingData_.a8w4Params.quantGroupSize =
        dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? QUANT_GROUP_SIZE : static_cast<uint32_t>(inputParams_.kSize);
    tilingData_.a8w4Params.quantGroupNum =
        dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ?
            static_cast<uint32_t>(inputParams_.kSize / QUANT_GROUP_SIZE) :
            1U;
    tilingData_.a8w4Params.expectedTokenNum = expectedTokenNum_;
    tilingData_.a8w4Params.coreNum = static_cast<uint32_t>(aicoreParams_.aicNum);
    tilingData_.a8w4Params.dequantMode = static_cast<uint8_t>(dequantMode_);
    tilingData_.a8w4Params.hasOffset = static_cast<uint8_t>(hasOffset_);
    tilingData_.a8w4Params.specialWeightFormat = static_cast<uint8_t>(specialWeightFormat_);
    tilingData_.a8w4Params.weightPackedInt32 = static_cast<uint8_t>(weightPackedInt32_);
    // Both modes expand the complete W4 weight to W8 in user workspace before
    // entering the ops-tensor A8W4 kernel.
    tilingData_.a8w4Params.enableWeightPreprocess = 1U;
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedA8W4BasicApiTiling::GetTilingKey() const
{
    const uint64_t weightFormat =
        inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ ? WQGMM_FRACTAL_NZ : WQGMM_ND;
    const uint64_t cQuantType =
        dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? WQGMM_PER_GROUP : WQGMM_PER_CHANNEL;
    return GET_TPL_TILING_KEY(
        weightFormat,
        WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST,
        cQuantType,
        WQGMM_NONE,
        WQGMM_NO_TRANS,
        WQGMM_NO_TRANS,
        WQGMM_A8W4_FIXED_TEMPLATE,
        WQGMM_NOT_SINGLE_MULTI_SINGLE,
        WQGMM_VDEFAULT,
        WQGMM_MULTI_SCALE_DEQUANT);
}

uint64_t GroupedA8W4BasicApiTiling::GetRequiredL1Size(uint64_t kL1) const
{
    return Blaze::Gemm::A8W4FixedTileResource::GetRequiredL1Bytes(
        basicTiling_.baseM, basicTiling_.baseN, kL1, kL1);
}

ge::graphStatus GroupedA8W4BasicApiTiling::ValidateFixedTileResources(uint64_t kL1) const
{
    using Resource = Blaze::Gemm::A8W4FixedTileResource;
    const uint64_t l1StageBytes = Resource::GetL1StageBytes(
        basicTiling_.baseM, basicTiling_.baseN, kL1, kL1);
    const uint64_t l1HalfBytes = aicoreParams_.l1Size / Resource::L1_BUFFER_NUM;
    const uint64_t l0aBytes = Resource::L0_BUFFER_NUM *
        Resource::GetL0ABytes(basicTiling_.baseM, basicTiling_.baseK);
    const uint64_t l0bBytes = Resource::L0_BUFFER_NUM *
        Resource::GetL0BBytes(basicTiling_.baseN, basicTiling_.baseK);
    const uint64_t l0cBytes = Resource::GetL0CBytes(basicTiling_.baseM, basicTiling_.baseN);
    const uint64_t ubBytes = dequantMode_ == DequantMode::ASYMMETRIC_PER_CHANNEL ?
        Resource::GetEpilogueUbBytes(basicTiling_.baseM, basicTiling_.baseN) : 0UL;

    OP_CHECK_IF(l1StageBytes > l1HalfBytes || l0aBytes > aicoreParams_.l0aSize ||
                    l0bBytes > aicoreParams_.l0bSize || l0cBytes > aicoreParams_.l0cSize ||
                    ubBytes > aicoreParams_.ubSize,
                OP_LOGE(context_->GetNodeName(),
                        "A8W4 fixed tile resource overflow: baseM=%lu, baseN=%lu, baseK=%lu, kL1=%lu, "
                        "L1 stage=%lu/%lu, L0A=%lu/%lu, L0B=%lu/%lu, L0C=%lu/%lu, UB=%lu/%lu.",
                        basicTiling_.baseM, basicTiling_.baseN, basicTiling_.baseK, kL1,
                        l1StageBytes, l1HalfBytes, l0aBytes, aicoreParams_.l0aSize,
                        l0bBytes, aicoreParams_.l0bSize, l0cBytes, aicoreParams_.l0cSize,
                        ubBytes, aicoreParams_.ubSize),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(),
            "A8W4 fixed tile resources: baseM=%lu, baseN=%lu, baseK=%lu, kL1=%lu, "
            "L1 stage=%lu/%lu, L0A=%lu/%lu, L0B=%lu/%lu, L0C=%lu/%lu, UB=%lu/%lu.",
            basicTiling_.baseM, basicTiling_.baseN, basicTiling_.baseK, kL1,
            l1StageBytes, l1HalfBytes, l0aBytes, aicoreParams_.l0aSize,
            l0bBytes, aicoreParams_.l0bSize, l0cBytes, aicoreParams_.l0cSize,
            ubBytes, aicoreParams_.ubSize);
    return ge::GRAPH_SUCCESS;
}

void GroupedA8W4BasicApiTiling::SetBasicBlock()
{
    const uint64_t preferredBaseM =
        dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? A8W4_BASE_M : A8W4_ASYM_BASE_M;
    basicTiling_.baseM =
        CeilAlign(std::min(inputParams_.mSize, preferredBaseM), static_cast<uint64_t>(CUBE_BLOCK));
    basicTiling_.baseN =
        CeilAlign(std::min(inputParams_.nSize, static_cast<uint64_t>(A8W4_BASE_N)),
                  static_cast<uint64_t>(CUBE_BLOCK));
    basicTiling_.baseK = A8W4_BASE_K;
}

ge::graphStatus GroupedA8W4BasicApiTiling::DoLibApiTiling()
{
    SetBasicBlock();
    uint64_t kL1 = dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ?
                       A8W4_PER_GROUP_K_L1 :
                       A8W4_PER_CHANNEL_K_L1;
    const uint64_t alignedK = CeilAlign(inputParams_.kSize, static_cast<uint64_t>(QUANT_GROUP_SIZE));
    kL1 = std::min(kL1, alignedK);
    OP_CHECK_IF(ValidateFixedTileResources(kL1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "A8W4 fixed tile resource validation failed."),
                return ge::GRAPH_FAILED);

    tilingData_.mmTilingData.m = static_cast<uint32_t>(inputParams_.mSize);
    tilingData_.mmTilingData.n = static_cast<uint32_t>(inputParams_.nSize);
    tilingData_.mmTilingData.k = static_cast<uint32_t>(inputParams_.kSize);
    tilingData_.mmTilingData.baseM = static_cast<uint32_t>(basicTiling_.baseM);
    tilingData_.mmTilingData.baseN = static_cast<uint32_t>(basicTiling_.baseN);
    tilingData_.mmTilingData.baseK = A8W4_BASE_K;
    tilingData_.mmTilingData.kAL1 = static_cast<uint32_t>(kL1);
    tilingData_.mmTilingData.kBL1 = static_cast<uint32_t>(kL1);
    tilingData_.mmTilingData.scaleKAL1 = static_cast<uint32_t>(kL1);
    tilingData_.mmTilingData.scaleKBL1 = static_cast<uint32_t>(kL1);
    // V5 bias is the offline correction for an unsigned W4 transform. This path
    // sign-extends W4 directly, so it must not be injected into Cube.
    tilingData_.mmTilingData.isBias = 0U;
    const uint64_t l0cTileBytes = basicTiling_.baseM * basicTiling_.baseN * sizeof(int32_t);
    tilingData_.mmTilingData.dbL0C =
        l0cTileBytes * 2UL <= aicoreParams_.l0cSize ? 2U : 1U;

    {
        constexpr uint64_t WORKSPACE_ALIGN = 512UL;
        constexpr uint64_t NZ_K_ALIGN = 16UL;
        constexpr uint64_t NZ_N_ALIGN = 32UL;
        constexpr uint64_t UINT64_MAX_VALUE = std::numeric_limits<uint64_t>::max();
        auto checkedMul = [](uint64_t lhs, uint64_t rhs, uint64_t &result) {
            if (rhs != 0UL && lhs > UINT64_MAX_VALUE / rhs) {
                return false;
            }
            result = lhs * rhs;
            return true;
        };
        auto checkedAdd = [](uint64_t lhs, uint64_t rhs, uint64_t &result) {
            if (lhs > UINT64_MAX_VALUE - rhs) {
                return false;
            }
            result = lhs + rhs;
            return true;
        };
        auto checkedAlign = [&](uint64_t value, uint64_t &result) {
            if (value > UINT64_MAX_VALUE - (WORKSPACE_ALIGN - 1UL)) {
                return false;
            }
            result = CeilAlign(value, WORKSPACE_ALIGN);
            return true;
        };
        uint64_t expandedK = inputParams_.kSize;
        uint64_t expandedN = inputParams_.nSize;
        if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
            OP_CHECK_IF(expandedK > UINT64_MAX_VALUE - (NZ_K_ALIGN - 1UL) ||
                            expandedN > UINT64_MAX_VALUE - (NZ_N_ALIGN - 1UL),
                        OP_LOGE(context_->GetNodeName(), "A8W4 expanded NZ alignment overflow."),
                        return ge::GRAPH_FAILED);
            expandedK = CeilAlign(expandedK, NZ_K_ALIGN);
            expandedN = CeilAlign(expandedN, NZ_N_ALIGN);
        }
        uint64_t expertStride = 0UL;
        uint64_t expandedBytes = 0UL;
        uint64_t expandedSize = 0UL;
        uint64_t tileElements = 0UL;
        uint64_t tileBytes = 0UL;
        uint64_t tileStride = 0UL;
        uint64_t tileSize = 0UL;
        uint64_t rowSumBytes = 0UL;
        uint64_t rowSumSize = 0UL;
        uint64_t tileOffset = 0UL;
        uint64_t rowSumOffset = 0UL;
        uint64_t totalSize = 0UL;
        OP_CHECK_IF(!checkedMul(expandedK, expandedN, expertStride) ||
                        !checkedMul(expertStride, inputParams_.groupNum, expandedBytes) ||
                        !checkedAlign(expandedBytes, expandedSize),
                    OP_LOGE(context_->GetNodeName(), "A8W4 expanded weight workspace overflow."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!checkedMul(basicTiling_.baseM, basicTiling_.baseN, tileElements) ||
                        !checkedMul(tileElements, sizeof(uint16_t), tileBytes) ||
                        !checkedAlign(tileBytes, tileStride) ||
                        !checkedMul(tileStride, aicoreParams_.aicNum, tileSize),
                    OP_LOGE(context_->GetNodeName(), "A8W4 per-core tile workspace overflow."),
                    return ge::GRAPH_FAILED);
        const uint64_t rowSumElements =
            dequantMode_ == DequantMode::ASYMMETRIC_PER_CHANNEL ? inputParams_.mSize : 0UL;
        OP_CHECK_IF(!checkedMul(rowSumElements, sizeof(float), rowSumBytes) ||
                        !checkedAlign(rowSumBytes, rowSumSize) ||
                        !checkedAdd(expandedSize, tileSize, rowSumOffset) ||
                        !checkedAdd(rowSumOffset, rowSumSize, totalSize),
                    OP_LOGE(context_->GetNodeName(), "A8W4 row-sum workspace overflow."),
                    return ge::GRAPH_FAILED);
        tileOffset = expandedSize;
        userWorkspaceSize_ = totalSize;
        tilingData_.a8w4Params.expandedWeightOffsetBytes = 0UL;
        tilingData_.a8w4Params.expandedWeightSizeBytes = expandedSize;
        tilingData_.a8w4Params.expandedWeightStrideBytes = expertStride;
        tilingData_.a8w4Params.tileWorkspaceOffsetBytes = tileOffset;
        tilingData_.a8w4Params.tileWorkspaceSizeBytes = tileSize;
        tilingData_.a8w4Params.tileWorkspaceStrideBytes = tileStride;
        tilingData_.a8w4Params.rowSumOffsetBytes = rowSumOffset;
        tilingData_.a8w4Params.rowSumSizeBytes = rowSumSize;
        tilingData_.a8w4Params.userWorkspaceSizeBytes = totalSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedA8W4BasicApiTiling::PostTiling()
{
    // if (dequantMode_ == DequantMode::ASYMMETRIC_PER_CHANNEL) {
    //     // The per-channel kernel expands W4 to W8 and computes the X row sum
    //     // on all vector cores before a full-core SyncAll. Batch scheduling
    //     // is required to guarantee that every participating core is resident;
    //     // otherwise an early wave can wait forever for a core that has not
    //     // been scheduled yet.
    //     context_->SetScheduleMode(1);
    // }
    context_->SetScheduleMode(1);
    return SaveTilingDataToContext(tilingData_);
}

ge::graphStatus GroupedA8W4BasicApiTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(userWorkspaceSize_ > std::numeric_limits<uint64_t>::max() - SYS_WORKSPACE_SIZES,
                OP_LOGE(context_->GetNodeName(), "A8W4 total workspace size overflow."),
                return ge::GRAPH_FAILED);
    // GetUserWorkspace() skips the first 16 MiB reserved by the runtime. The
    // complete expanded W8 weight is placed after that system area.
    uint64_t workspaceSize = SYS_WORKSPACE_SIZES + userWorkspaceSize_;
    OP_CHECK_IF(workspaceSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
                OP_LOGE(context_->GetNodeName(), "A8W4 workspace does not fit size_t."),
                return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(workspaceSize);
    return ge::GRAPH_SUCCESS;
}

#ifdef ASCEND_OPTILING_UT
extern "C" __attribute__((visibility("default"))) ge::graphStatus
TilingGroupedA8W4BasicApiForTest(gert::TilingContext *context)
{
    GroupedA8W4BasicApiTiling tiling(context);
    return tiling.DoTiling();
}
#endif

} // namespace optiling
