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
 * \file grouped_matmul_finalize_routing_quant_tiling.cpp
 * \brief
 */

#include "grouped_matmul_finalize_routing_quant_tiling.h"
#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
using namespace Ops::Transformer::OpTiling;
using namespace optiling::GroupedMatmulFinalizeRoutingArch35TilingConstant;
using namespace optiling::GmmConstant;
using namespace GMMFinalizeRoutingArch35Tiling;

namespace optiling {

void GroupedMatmulFinalizeRoutingQuantTiling::Reset()
{
    tilingData_ = GMMFinalizeRoutingTilingData();
    OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                         context_->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
    return;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "Attrs is nullptr."), return false);
    OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_TUNING_CONFIG + 1,
                OP_LOGE(context_->GetNodeName(), "The num of attrs should be greater than %u, actual is %zu",
                        ATTR_INDEX_TUNING_CONFIG + 1, attrs->GetAttrNum()),
                return false);
    const float *shareInputWeightPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_SHARE_INPUT_WEIGHT);
    const uint32_t *shareInputOffsetPtr = attrs->GetAttrPointer<uint32_t>(ATTR_INDEX_SHARE_INPUT_OFFSET);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_W);
    const uint8_t *groupListTypePtr = attrs->GetAttrPointer<uint8_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
    const uint32_t *outputBSPtr = attrs->GetAttrPointer<uint32_t>(ATTR_INDEX_OUTPUT_BS);

    OP_CHECK_IF(shareInputWeightPtr == nullptr, OP_LOGE(context_->GetNodeName(), "Attr residualScale is nullptr."),
                return false);
    OP_CHECK_IF(outputBSPtr == nullptr, OP_LOGE(context_->GetNodeName(), "Attr batch is nullptr."), return false);

    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    sharedInputOffset_ = shareInputOffsetPtr != nullptr ? *shareInputOffsetPtr : sharedInputOffset_;
    sharedInputWeight_ = *shareInputWeightPtr;
    outputBs_ = *outputBSPtr;

    OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                OP_LOGE(context_->GetNodeName(), "Attr groupListType must be 0 or 1, actual is %d.",
                        inputParams_.groupListType),
                return false);

    OP_CHECK_IF(sharedInputOffset_ > outputBs_,
                OP_LOGE(context_->GetNodeName(), "Attr sharedInputOffset (%lu) out of batch(%lu).", sharedInputOffset_,
                        outputBs_),
                return false);

    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : inputParams_.scaleDtype;
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto outDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(outDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input outDesc is nullptr."), return false);
    inputParams_.cDtype = outDesc->GetDataType();
    auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    inputParams_.hasBias = !(biasStorageShape == nullptr || biasStorageShape->GetStorageShape().GetShapeSize() == 0);
    auto biasDesc = context_->GetDynamicInputDesc(BIAS_INDEX, 0);
    OP_CHECK_IF(inputParams_.hasBias && biasDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "Bias from tensor is not nullptr, but bias from desc is nullptr."),
                return false);
    inputParams_.biasDtype = inputParams_.hasBias ? biasDesc->GetDataType() : inputParams_.biasDtype;

    return CheckDtype();
}

bool GroupedMatmulFinalizeRoutingQuantTiling::IsFp4Dtype(ge::DataType dtype)
{
    return (dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_FLOAT4_E2M1);
}

bool GroupedMatmulFinalizeRoutingQuantTiling::IsFp8Dtype(ge::DataType dtype)
{
    return (dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2);
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckDtype()
{
    bool a8w8 = IsFp8Dtype(inputParams_.aDtype) && IsFp8Dtype(inputParams_.bDtype);
    bool a4w4 = IsFp4Dtype(inputParams_.aDtype) && IsFp4Dtype(inputParams_.bDtype);
    if (a8w8 || a4w4) {
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 ||
                        inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(context_->GetNodeName(),
                            "With DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2/DT_FLOAT4_E1M2/DT_FLOAT4_E2M1 inputs, \
the expected dtype of scale and pertokenScale should be DT_FLOAT8_E8M0, but actual dtype is %s, %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                    return false);
    } else {
        OP_LOGE(context_->GetNodeName(), "Quant case with x dtype %s and weight dtype %s is not supported.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str());
        return false;
    }
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckShapeForMxQuant(const gert::Shape &xShape, const gert::Shape &wShape,
                                                                   const gert::Shape &pertokenScaleShape,
                                                                   const gert::Shape &scaleShape,
                                                                   const gert::Shape &yShape)
{
    auto xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(xDimNum != DIM_NUM_X,
                OP_LOGE(context_->GetNodeName(), "The dimension of x must be %u, actual is %zu", DIM_NUM_X, xDimNum),
                return false);

    auto wDimNum = wShape.GetDimNum();
    OP_CHECK_IF(
        wDimNum != DIM_NUM_WEIGHT,
        OP_LOGE(context_->GetNodeName(), "The dimension of w must be %u, actual is %zu", DIM_NUM_WEIGHT, wDimNum),
        return false);

    auto scaleDimNum = scaleShape.GetDimNum();
    OP_CHECK_IF(scaleDimNum != DIM_NUM_SCALE,
                OP_LOGE(context_->GetNodeName(), "The dimension of scale must be %u, actual is %zu", DIM_NUM_SCALE,
                        scaleDimNum),
                return false);

    auto pertokenScaleDimNum = pertokenScaleShape.GetDimNum();
    OP_CHECK_IF(pertokenScaleDimNum != DIM_NUM_PERTOKENSCALE,
                OP_LOGE(context_->GetNodeName(), "The dimension of pertokenScale must be %u, actual is %zu",
                        DIM_NUM_PERTOKENSCALE, pertokenScaleDimNum),
                return false);

    auto yDimNum = yShape.GetDimNum();
    OP_CHECK_IF(yDimNum != DIM_NUM_Y,
                OP_LOGE(context_->GetNodeName(), "The dimension of y must be %u, actual is %zu", DIM_NUM_Y, yDimNum),
                return false);

    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "Input xStorageShape is nullptr."),
                return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
   
    auto wStorageShape = context_->GetInputShape(W_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "Input wStorageShape is nullptr."),
                return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    
    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "Input scaleStorageShape is nullptr."),
                return false);
    const gert::Shape &scaleShape = scaleStorageShape->GetOriginShape();
    
    auto pertokenScaleStorageShape = context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX);
    OP_CHECK_IF(pertokenScaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "Input pertokenScaleStorageShape is nullptr."), return false);
    const gert::Shape &pertokenScaleShape = pertokenScaleStorageShape->GetOriginShape();
    
    auto yStorageShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(yStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "Output yStorageShape is nullptr."),
                return false);
    const gert::Shape &yShape = yStorageShape->GetOriginShape();

    OP_CHECK_IF(!CheckShapeForMxQuant(xShape, wShape, pertokenScaleShape, scaleShape, yShape),
                OP_LOGE(context_->GetNodeName(), "CheckShapeForMxQuant failed."), return false);

    auto sharedInputDesc = context_->GetOptionalInputDesc(SHARE_INPUT_INDEX);
    sharedInputLen_ = sharedInputDesc != nullptr ?
                          context_->GetOptionalInputShape(SHARE_INPUT_INDEX)->GetStorageShape()[0] :
                          sharedInputLen_;

    OP_CHECK_IF(
        sharedInputLen_ > outputBs_,
        OP_LOGE(context_->GetNodeName(), "Input shared_input_len (%lu) out of batch(%lu).", sharedInputLen_, outputBs_),
        return false);

    OP_CHECK_IF(sharedInputOffset_ + sharedInputLen_ > outputBs_,
                OP_LOGE(context_->GetNodeName(), "SharedInputOffset + sharedInputLen (%lu) out of batch(%lu).",
                        sharedInputOffset_ + sharedInputLen_, outputBs_),
                return false);

    auto LogitDesc = context_->GetOptionalInputDesc(LOGIT_INDEX);
    OP_CHECK_IF(LogitDesc == nullptr, OP_LOGE(context_->GetNodeName(), "LogitDesc is nullptr."), return false);

    auto rowIndexDesc = context_->GetOptionalInputDesc(ROW_INDEX_INDEX);
    OP_CHECK_IF(rowIndexDesc == nullptr, OP_LOGE(context_->GetNodeName(), "RowIndexDesc is nullptr."), return false);
    rowIndex_ = context_->GetOptionalInputShape(ROW_INDEX_INDEX)->GetStorageShape()[0];

    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(context_->GetNodeName(), "SetGroupNum failed."), return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(context_->GetNodeName(), "SetMKN failed."), return false);
    OP_CHECK_IF(!SetQuantModeForGMMFinalizeRouting(),
                OP_LOGE(context_->GetNodeName(), "SetQuantModeForGMMFinalizeRouting failed."), return false);
    OP_CHECK_IF(rowIndex_ > inputParams_.mSize,
                OP_LOGE(context_->GetNodeName(), "Input rowIndex (%lu) out of M (%lu).", rowIndex_, inputParams_.mSize),
                return false);

    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::SetQuantModeForGMMFinalizeRouting()
{
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        return true;
    } else {
        OP_LOGE(inputParams_.opName, "The expected dtype of scale should be DT_FLOAT8_E8M0");
        return false;
    }
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DoOpTiling()
{
    tilingData_.gmmFinalizeRoutingDataParams.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    tilingData_.gmmFinalizeRoutingDataParams.batch = static_cast<uint32_t>(outputBs_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset = static_cast<uint32_t>(sharedInputOffset_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen = static_cast<uint32_t>(sharedInputLen_);
    tilingData_.gmmFinalizeRoutingDataParams.residualScale = static_cast<float>(sharedInputWeight_);
    tilingData_.gmmFinalizeRoutingDataParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.biasDtype = static_cast<uint32_t>(ge::DataType::DT_INT32);
    tilingData_.gmmFinalizeRoutingDataParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmFinalizeRoutingDataParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias ? 1 : 0);

    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulFinalizeRoutingQuantTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB));
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DoLibApiTiling()
{
    CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"),
                return ge::GRAPH_FAILED);
    tilingData_.matmulTiling.M = inputParams_.mSize;
    tilingData_.matmulTiling.N = inputParams_.nSize;
    tilingData_.matmulTiling.Ka = inputParams_.kSize;
    tilingData_.matmulTiling.Kb = inputParams_.kSize;
    tilingData_.matmulTiling.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.singleCoreM = basicTiling_.singleCoreM;
    tilingData_.matmulTiling.singleCoreN = basicTiling_.singleCoreN;
    tilingData_.matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_.matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_.matmulTiling.stepM = basicTiling_.stepM;
    tilingData_.matmulTiling.stepN = basicTiling_.stepN;
    tilingData_.matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_.matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_.matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_.matmulTiling.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.matmulTiling.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.matmulTiling.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }
    PrintMatmulParams();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::PostTiling()
{
    auto tilingDataSize = sizeof(GMMFinalizeRoutingTilingData);
    context_->SetBlockDim(aicoreParams_.aicNum);
    OP_CHECK_IF(tilingDataSize % sizeof(uint64_t) != 0,
                OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8", tilingDataSize),
                return ge::GRAPH_FAILED);
    error_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(inputParams_.opName, "Memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

void GroupedMatmulFinalizeRoutingQuantTiling::PrintMatmulParams()
{
    int32_t enable = AlogCheckDebugLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
        return;
    }
    std::ostringstream oss;
    oss << "GMM matmul tiling: M = " << tilingData_.matmulTiling.M << ", N = " << tilingData_.matmulTiling.N
        << ", Ka = " << tilingData_.matmulTiling.Ka << ", Kb = " << tilingData_.matmulTiling.Kb
        << ", usedCoreNum = " << tilingData_.matmulTiling.usedCoreNum << ", baseM = " << tilingData_.matmulTiling.baseM
        << ", baseN = " << tilingData_.matmulTiling.baseN << ", baseK = " << tilingData_.matmulTiling.baseK;
    OP_LOGD(context_->GetNodeName(), "%s", oss.str().c_str());
}

void GroupedMatmulFinalizeRoutingQuantTiling::PrintQuantParams()
{
    int32_t enable = AlogCheckDebugLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
        return;
    }

    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << tilingData_.gmmFinalizeRoutingDataParams.groupNum
        << ", groupListType = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.groupListType)
        << ", batch = " << tilingData_.gmmFinalizeRoutingDataParams.batch
        << ", sharedInputOffset = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset
        << ", sharedInputLen = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen
        << ", residualScale = " << tilingData_.gmmFinalizeRoutingDataParams.residualScale
        << ", aQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.aQuantMode
        << ", bQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.bQuantMode
        << ", hasBias = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.hasBias);
    OP_LOGD(context_->GetNodeName(), "%s", oss.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulFinalizeRouting, GroupedMatmulFinalizeRoutingQuantTiling, 1);
} // namespace optiling