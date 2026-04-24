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
 * \file quant_grouped_matmul_inplace_add_host_utils.cpp
 * \brief
 */
#include "../quant_grouped_matmul_inplace_add_host_utils.h"

#include <graph/utils/type_utils.h>

#include "log/log.h"

using namespace optiling;
using namespace optiling::GmmConstant;

namespace QuantGroupedMatmulInplaceAdd {

bool AnalyzeAttrsForInplaceAdd(gert::TilingContext *context, GQmmInputInfo &inputParams)
{
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        inputParams.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams.groupListType;
    }
    inputParams.transA = true;
    return true;
}

bool AnalyzeDtypeForInplaceAdd(gert::TilingContext *context, GQmmInputInfo &inputParams)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context->GetNodeName(), "xDesc is nullptr."), return false);
    inputParams.aDtype = xDesc->GetDataType();
    auto wDesc = context->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context->GetNodeName(), "wDesc is nullptr."), return false);
    inputParams.bDtype = wDesc->GetDataType();
    auto scaleDesc = context->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context->GetNodeName(), "scaleDesc is nullptr."), return false);
    inputParams.scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams.perTokenScaleDtype;
    auto yDesc = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context->GetNodeName(), "yDesc is nullptr."), return false);
    inputParams.cDtype = yDesc->GetDataType();
    return true;
}

bool CheckDtypeForInplaceAdd(const GQmmInputInfo &inputParams, bool allowFp8)
{
    OP_CHECK_IF(inputParams.cDtype != ge::DT_FLOAT,
                OP_LOGE(inputParams.opName, "Input yRef dtype should be DT_FLOAT, actual dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams.cDtype).c_str()),
                return false);
    if (inputParams.aDtype == ge::DT_HIFLOAT8 && inputParams.bDtype == ge::DT_HIFLOAT8) {
        OP_CHECK_IF(inputParams.scaleDtype != ge::DT_FLOAT,
                    OP_LOGE(inputParams.opName,
                            "With DT_HIFLOAT8 inputs, scale2 dtype should be DT_FLOAT, actual dtype is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype).c_str()),
                    return false);
        OP_CHECK_IF(inputParams.perTokenScaleDtype != ge::DT_FLOAT,
                    OP_LOGE(inputParams.opName,
                            "With DT_HIFLOAT8 inputs, scale1 dtype should be DT_FLOAT, actual dtype is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype).c_str()),
                    return false);
        return true;
    }
    if (allowFp8 && (inputParams.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams.aDtype == ge::DT_FLOAT8_E5M2) &&
        (inputParams.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams.bDtype == ge::DT_FLOAT8_E5M2)) {
        OP_CHECK_IF(inputParams.scaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(inputParams.opName,
                            "With DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 inputs, scale2 dtype should be DT_FLOAT8_E8M0, "
                            "actual dtype is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype).c_str()),
                    return false);
        OP_CHECK_IF(inputParams.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(inputParams.opName,
                            "With DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 inputs, scale1 dtype should be DT_FLOAT8_E8M0, "
                            "actual dtype is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype).c_str()),
                    return false);
        return true;
    }
    OP_LOGE(inputParams.opName, "Quant case with x1 dtype %s and x2 dtype %s is not supported.",
            ge::TypeUtils::DataTypeToSerialString(inputParams.aDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.bDtype).c_str());
    return false;
}

bool CheckCoreNumForInplaceAdd(gert::TilingContext *context, const GQmmInputInfo &inputParams)
{
    auto aicNum = context->GetCompileInfo<GMMCompileInfo>()->aicNum;
    auto aivNum = context->GetCompileInfo<GMMCompileInfo>()->aivNum;
    OP_CHECK_IF(aicNum == 0, OP_LOGE(inputParams.opName, "aicNum should be positive integer, actual is %u.", aicNum),
                return false);
    OP_CHECK_IF(
        aivNum != GmmConstant::CORE_RATIO * aicNum,
        OP_LOGE(inputParams.opName, "aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.", aicNum, aivNum),
        return false);
    return true;
}

bool CheckShapeForTTQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                          const GQmmInputInfo &inputParams)
{
    // pertensor-pertensor 模式：scale1 和 scale2 均为 (g,) 或 (g, 1)
    // 校验 scale1 (x1Scale)
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != 1 && x1ScaleDimNum != 2,
                OP_LOGE(inputParams.opName,
                        "The dimension of scale1 should be 1 or 2 in pertensor-pertensor mode, but \
actual is %zu",
                        x1ScaleDimNum),
                return false);
    auto firstDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    if (x1ScaleDimNum == 1) {
        OP_CHECK_IF(firstDim != inputParams.groupNum,
                    OP_LOGE(inputParams.opName,
                            "In pertensor-pertensor mode, the expected shape of scale1 is (%lu,) or \
(%lu, 1), but the actual is (%lu,).",
                            inputParams.groupNum, inputParams.groupNum, firstDim),
                    return false);
    } else {
        auto lastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(1));
        OP_CHECK_IF(firstDim != inputParams.groupNum || lastDim != 1,
                    OP_LOGE(inputParams.opName,
                            "In pertensor-pertensor mode, the expected shape of scale1 is (%lu,) or \
(%lu, 1), but the actual is (%lu, %lu).",
                            inputParams.groupNum, inputParams.groupNum, firstDim, lastDim),
                    return false);
    }

    // 校验 scale2 (x2Scale)
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != 1 && x2ScaleDimNum != 2,
                OP_LOGE(inputParams.opName,
                        "The dimension of scale2 should be 1 or 2 in pertensor-pertensor mode, but \
actual is %zu",
                        x2ScaleDimNum),
                return false);
    firstDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    if (x2ScaleDimNum == 1) {
        OP_CHECK_IF(firstDim != inputParams.groupNum,
                    OP_LOGE(inputParams.opName,
                            "In pertensor-pertensor mode, the expected shape of scale2 is (%lu,) or \
(%lu, 1), but the actual is (%lu,).",
                            inputParams.groupNum, inputParams.groupNum, firstDim),
                    return false);
    } else {
        auto lastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(1));
        OP_CHECK_IF(firstDim != inputParams.groupNum || lastDim != 1,
                    OP_LOGE(inputParams.opName,
                            "In pertensor-pertensor mode, the expected shape of scale2 is (%lu,) or \
(%lu, 1), but the actual is (%lu, %lu).",
                            inputParams.groupNum, inputParams.groupNum, firstDim, lastDim),
                    return false);
    }
    return true;
}

bool CheckShapeForMxQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                          const GQmmInputInfo &inputParams)
{
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != MXFP_TYPE_K_SCALE_DIM_NUM,
                OP_LOGE(inputParams.opName,
                        "The dimension of scale2 should be 3 in mx quant mode, but actual is %zu", x2ScaleDimNum),
                return false);
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
                OP_LOGE(inputParams.opName,
                        "The dim num of scale1 should be 3 in mx quant mode, but \
actual is %zu",
                        x1ScaleDimNum),
                return false);
    auto xScaleLastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - 1));
    auto xScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto xScaleMDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleLastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - 1));
    auto wScaleNDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto expectedKDimValue = inputParams.kSize / MXFP_BASEK_FACTOR + inputParams.groupNum;
    OP_CHECK_IF(xScaleLastDim != MXFP_MULTI_BASE_SIZE || xScaleKDim != expectedKDimValue ||
                    xScaleMDim != inputParams.mSize,
                OP_LOGE(inputParams.opName,
                        "In mx quant mode, the expected shape of scale1 is \
(%lu,%lu,%lu), but the actual is (%lu,%lu,%lu).",
                        expectedKDimValue, inputParams.mSize, MXFP_MULTI_BASE_SIZE, xScaleKDim, xScaleMDim,
                        xScaleLastDim),
                return false);
    OP_CHECK_IF(wScaleLastDim != MXFP_MULTI_BASE_SIZE || wScaleKDim != expectedKDimValue ||
                    wScaleNDim != inputParams.nSize,
                OP_LOGE(inputParams.opName,
                        "In mx quant mode, the expected shape of scale2 is (%lu,%lu,%lu), \
but the actual is (%lu,%lu,%lu).",
                        expectedKDimValue, inputParams.nSize, MXFP_MULTI_BASE_SIZE, wScaleKDim, wScaleNDim,
                        wScaleLastDim),
                return false);
    return true;
}

bool CheckShapeForTCQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                          const GQmmInputInfo &inputParams)
{
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != 1 && x1ScaleDimNum != 2, // Max dim num in T-C quant: 2
                OP_LOGE(inputParams.opName,
                        "The dimension of scale1 should be 1 or 2 in T-C quant mode, but \
actual is %zu",
                        x1ScaleDimNum),
                return false);
    auto lastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - 1));
    auto firstDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    if (x1ScaleDimNum == 1) {
        OP_CHECK_IF(firstDim != inputParams.groupNum,
                    OP_LOGE(inputParams.opName,
                            "In T-C quant mode, the expected shape of scale1 is (%lu, ) or (%lu, 1), \
but the actual is (%lu, ).",
                            inputParams.groupNum, inputParams.groupNum, firstDim),
                    return false);
    } else {
        OP_CHECK_IF(firstDim != inputParams.groupNum || lastDim != 1,
                    OP_LOGE(inputParams.opName,
                            "In T-C quant mode, the expected shape of scale1 is (%lu, ) or (%lu, 1), \
but the actual is (%lu, %lu).",
                            inputParams.groupNum, inputParams.groupNum, firstDim, lastDim),
                    return false);
    }

    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != 2, // Max dim num in T-C quant: 2
                OP_LOGE(inputParams.opName,
                        "The dimension of scale2 should be 2 in T-C quant mode, but \
actual is %zu",
                        x2ScaleDimNum),
                return false);
    lastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - 1));
    firstDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    OP_CHECK_IF(firstDim != inputParams.groupNum || lastDim != inputParams.nSize,
                OP_LOGE(inputParams.opName,
                        "In T-C quant mode, the expected shape of scale2 is (%lu, %lu), \
but the actual is (%lu, %lu).",
                        inputParams.groupNum, inputParams.nSize, firstDim, lastDim),
                return false);

    return true;
}

} // namespace QuantGroupedMatmulInplaceAdd
