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
 * \file quant_grouped_matmul_inplace_add_basic_api_tiling.cpp
 * \brief
 */
#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/qgmm_inplace_add_tiling_key.h"
#include "quant_grouped_matmul_inplace_add_basic_api_tiling.h"

using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmul;
using namespace optiling::GmmConstant;
namespace optiling {

QGMMInplaceAddBasicApiTiling::QGMMInplaceAddBasicApiTiling(gert::TilingContext *context)
    : GroupedQmmBasicApiTiling(context)
{
    GroupedQmmBasicApiTiling::Reset();
}

bool QGMMInplaceAddBasicApiTiling::IsCapable()
{
    // T-T
    return inputParams_.aDtype == ge::DT_HIFLOAT8 && AnalyzeInputs();
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);  // 通路保证非负数
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    }
    inputParams_.transA = true;
    return true;
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "scaleDesc is nullptr."), return false);
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context_->GetNodeName(), "yDesc is nullptr."), return false);
    inputParams_.cDtype = yDesc->GetDataType();
    return CheckDtype();
}

bool QGMMInplaceAddBasicApiTiling::CheckDtype()
{
    OP_CHECK_IF(inputParams_.cDtype != ge::DT_FLOAT,
                OP_LOGE(inputParams_.opName, "Input yRef dtype should be DT_FLOAT, actual dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
                return false);
    if (inputParams_.aDtype == ge::DT_HIFLOAT8 && inputParams_.bDtype == ge::DT_HIFLOAT8) {
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT,
                    OP_LOGE(inputParams_.opName,
                            "With DT_HIFLOAT8 inputs, scale2 dtype should be DT_FLOAT, actual dtype is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                    return false);
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                    OP_LOGE(inputParams_.opName,
                            "With DT_HIFLOAT8 inputs, scale1 dtype should be DT_FLOAT, actual dtype is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                    return false);
    } else {
        OP_LOGE(inputParams_.opName, "Quant case with x1 dtype %s and x2 dtype %s is not supported.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str());
        return false;
    }
    return true;
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();

    auto wStorageShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();

    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "scaleStorageShape is nullptr."),
                return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(
        scaleDimNum < 1,
        OP_LOGE(inputParams_.opName, "The dimension of scale1 should be positive integer, actual is %zu", scaleDimNum),
        return false);

    auto x1ScaleStorageShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "x1ScaleStorageShape is nullptr."),
                return false);
    const gert::Shape &x1ScaleShape = x1ScaleStorageShape->GetOriginShape();

    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);

    // T-T 模式：scale1 和 scale2 均为 (g,) 或 (g,1)
    OP_CHECK_IF(!CheckShapeForTTQuant(x1ScaleShape, wScaleShape),
                OP_LOGE(inputParams_.opName, "CheckShapeForPertensorQuant failed."), return false);

    OP_CHECK_IF(!SetQuantModeForQGmmInplaceAdd(), OP_LOGE(inputParams_.opName, "SetQuantModeForQGmmInplaceAdd failed."),
                return false);
    SetKernelType();
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    return true;
}

bool QGMMInplaceAddBasicApiTiling::SetQuantModeForQGmmInplaceAdd()
{
    inputParams_.bQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    inputParams_.aQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    return true;
}

bool QGMMInplaceAddBasicApiTiling::CheckShapeForTTQuant(const gert::Shape &x1ScaleShape,
                                                        const gert::Shape &x2ScaleShape)
{
    // pertensor-pertensor 模式：scale1 和 scale2 均为 (g,) 或 (g, 1)
    // 校验 scale1 (x1Scale)
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != 1 && x1ScaleDimNum != 2,
                OP_LOGE(inputParams_.opName,
                        "The dimension of scale1 should be 1 or 2 in pertensor-pertensor mode, but \
actual is %zu",
                        x1ScaleDimNum),
                return false);
    auto firstDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    if (x1ScaleDimNum == 1) {
        OP_CHECK_IF(firstDim != inputParams_.groupNum,
                    OP_LOGE(inputParams_.opName,
                            "In pertensor-pertensor mode, the expected shape of scale1 is (%lu,) or \
(%lu, 1), but the actual is (%lu,).",
                            inputParams_.groupNum, inputParams_.groupNum, firstDim),
                    return false);
    } else {
        auto lastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(1));
        OP_CHECK_IF(firstDim != inputParams_.groupNum || lastDim != 1,
                    OP_LOGE(inputParams_.opName,
                            "In pertensor-pertensor mode, the expected shape of scale1 is (%lu,) or \
(%lu, 1), but the actual is (%lu, %lu).",
                            inputParams_.groupNum, inputParams_.groupNum, firstDim, lastDim),
                    return false);
    }

    // 校验 scale2 (x2Scale)
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != 1 && x2ScaleDimNum != 2,
                OP_LOGE(inputParams_.opName,
                        "The dimension of scale2 should be 1 or 2 in pertensor-pertensor mode, but \
actual is %zu",
                        x2ScaleDimNum),
                return false);
    firstDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    if (x2ScaleDimNum == 1) {
        OP_CHECK_IF(firstDim != inputParams_.groupNum,
                    OP_LOGE(inputParams_.opName,
                            "In pertensor-pertensor mode, the expected shape of scale2 is (%lu,) or \
(%lu, 1), but the actual is (%lu,).",
                            inputParams_.groupNum, inputParams_.groupNum, firstDim),
                    return false);
    } else {
        auto lastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(1));
        OP_CHECK_IF(firstDim != inputParams_.groupNum || lastDim != 1,
                    OP_LOGE(inputParams_.opName,
                            "In pertensor-pertensor mode, the expected shape of scale2 is (%lu,) or \
(%lu, 1), but the actual is (%lu, %lu).",
                            inputParams_.groupNum, inputParams_.groupNum, firstDim, lastDim),
                    return false);
    }
    return true;
}

bool QGMMInplaceAddBasicApiTiling::CheckCoreNum() const
{
    auto aicNum = context_->GetCompileInfo<GMMCompileInfo>()->aicNum;
    auto aivNum = context_->GetCompileInfo<GMMCompileInfo>()->aivNum;
    OP_CHECK_IF(aicNum == 0, OP_LOGE(inputParams_.opName, "aicNum should be positive integer, actual is %u.", aicNum),
                return false);
    OP_CHECK_IF(
        aivNum != GmmConstant::CORE_RATIO * aicNum,
        OP_LOGE(inputParams_.opName, "aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.", aicNum, aivNum),
        return false);
    return true;
}

ge::graphStatus QGMMInplaceAddBasicApiTiling::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    tilingData_.gmmQuantParams.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    tilingData_.gmmQuantParams.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    tilingData_.gmmQuantParams.groupType = static_cast<int8_t>(inputParams_.groupType);
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmQuantParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias);
    GroupedQmmBasicApiTiling::PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QGMMInplaceAddBasicApiTiling::DoLibApiTiling()
{
    GroupedQmmTiling::CalBasicBlock();
    OP_CHECK_IF(GroupedQmmBasicApiTiling::CalL1Tiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"), return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.m = inputParams_.mSize;
    tilingData_.mmTilingData.n = inputParams_.nSize;
    tilingData_.mmTilingData.k = inputParams_.kSize;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.kAL1 = basicTiling_.stepKa * basicTiling_.baseK;
    tilingData_.mmTilingData.kBL1 = basicTiling_.stepKb * basicTiling_.baseK;
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    return ge::GRAPH_SUCCESS;
}

uint64_t QGMMInplaceAddBasicApiTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA),
                              static_cast<uint64_t>(inputParams_.kernelType));
}

}  // namespace optiling