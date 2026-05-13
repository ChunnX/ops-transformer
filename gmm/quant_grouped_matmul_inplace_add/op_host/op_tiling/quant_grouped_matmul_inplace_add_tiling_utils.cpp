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
 * \file quant_grouped_matmul_inplace_add_tiling_utils.cpp
 * \brief
 */
#include "quant_grouped_matmul_inplace_add_tiling_utils.h"

#include <graph/utils/type_utils.h>

#include "../../op_kernel/arch35/qgmm_inplace_add_tiling_key.h"
#include "../quant_grouped_matmul_inplace_add_host_utils.h"
#include "log/log.h"

using namespace optiling;
using namespace optiling::GmmConstant;

namespace QuantGroupedMatmulInplaceAdd {

bool AnalyzeAttrsForInplaceAdd(gert::TilingContext *context, GQmmInputInfo &inputParams)
{
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
        if (groupListTypePtr != nullptr) {
            OP_CHECK_IF(*groupListTypePtr != 0 && *groupListTypePtr != 1,
                        OP_LOGE(context->GetNodeName(), "GroupListType must be 0 or 1, but actual value is %ld.",
                                *groupListTypePtr),
                        return false);
            inputParams.groupListType = static_cast<int8_t>(*groupListTypePtr);
        }
        const int64_t *groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_SIZE);
        if (groupSizePtr != nullptr) {
            OP_CHECK_IF(*groupSizePtr != 0,
                        OP_LOGE(context->GetNodeName(), "GroupSize must be 0, but actual value is %ld.", *groupSizePtr),
                        return false);
        }
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

bool CheckDtypeForInplaceAdd(const GQmmInputInfo &inputParams)
{
    OP_CHECK_IF(inputParams.cDtype != ge::DT_FLOAT,
                OP_LOGE(inputParams.opName, "Input yRef dtype should be DT_FLOAT, actual dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams.cDtype).c_str()),
                return false);

    bool isHif8 = inputParams.aDtype == ge::DT_HIFLOAT8 && inputParams.bDtype == ge::DT_HIFLOAT8;
    bool isFp8 = (inputParams.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams.aDtype == ge::DT_FLOAT8_E5M2) &&
                 (inputParams.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams.bDtype == ge::DT_FLOAT8_E5M2);

    if (isHif8) {
        OP_CHECK_IF(
            inputParams.scaleDtype != ge::DT_FLOAT,
            OP_LOGE(inputParams.opName, "With DT_HIFLOAT8 inputs, scale2 dtype should be DT_FLOAT, actual dtype is %s.",
                    ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype).c_str()),
            return false);
        OP_CHECK_IF(
            inputParams.perTokenScaleDtype != ge::DT_FLOAT,
            OP_LOGE(inputParams.opName, "With DT_HIFLOAT8 inputs, scale1 dtype should be DT_FLOAT, actual dtype is %s.",
                    ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype).c_str()),
            return false);
        return true;
    }

    if (isFp8) {
        OP_CHECK_IF(inputParams.scaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(inputParams.opName,
                            "With DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 inputs, scale2 dtype should be DT_FLOAT8_E8M0, \
actual dtype is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype).c_str()),
                    return false);
        OP_CHECK_IF(inputParams.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(inputParams.opName,
                            "With DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 inputs, scale1 dtype should be DT_FLOAT8_E8M0, \
actual dtype is %s.",
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

bool CheckShapeForHif8Quant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                            const GQmmInputInfo &inputParams)
{
    // T-T / T-C 合并校验，与 aclnn CheckHif8QuantParamsShape 对齐：
    //   scale1 (x1Scale/perToken): 1D 或 2D；firstDim == groupNum；若 2D 则 lastDim == 1
    //   scale2 (x2Scale):          1D 或 2D；firstDim == groupNum；若 2D 则 lastDim ∈ {1, nSize}
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != 1 && x1ScaleDimNum != 2,
                OP_LOGE(inputParams.opName,
                        "In T-T/T-C mode, the dimension of scale1 should be 1 or 2, but actual is %zu", x1ScaleDimNum),
                return false);
    auto x1FirstDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    OP_CHECK_IF(x1FirstDim != inputParams.groupNum,
                OP_LOGE(inputParams.opName,
                        "In T-T/T-C mode, the first dim of scale1 must equal groupNum[%lu], but actual is %lu.",
                        inputParams.groupNum, x1FirstDim),
                return false);
    if (x1ScaleDimNum == 2) {
        auto x1LastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(1));
        OP_CHECK_IF(x1LastDim != 1,
                    OP_LOGE(inputParams.opName,
                            "In T-T/T-C mode, the last dim of scale1 should be 1, but actual is %lu.", x1LastDim),
                    return false);
    }

    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(
        x2ScaleDimNum != 1 && x2ScaleDimNum != 2,
        OP_LOGE(inputParams.opName, "The dimension of scale2 should be 1 or 2, but actual is %zu", x2ScaleDimNum),
        return false);
    auto x2FirstDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    OP_CHECK_IF(x2FirstDim != inputParams.groupNum,
                OP_LOGE(inputParams.opName,
                        "In T-T/T-C mode, the first dim of scale2 must equal groupNum[%lu], but actual is %lu.",
                        inputParams.groupNum, x2FirstDim),
                return false);
    if (x2ScaleDimNum == 2) {
        auto x2LastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(1));
        OP_CHECK_IF(x2LastDim != 1 && x2LastDim != inputParams.nSize,
                    OP_LOGE(inputParams.opName,
                            "In T-T/T-C mode, the last dim of scale2 should be 1 or n[%lu], but actual is %lu.",
                            inputParams.nSize, x2LastDim),
                    return false);
    }
    return true;
}

bool CheckShapeForMxQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                          const GQmmInputInfo &inputParams)
{
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != MXFP_TYPE_K_SCALE_DIM_NUM,
                OP_LOGE(inputParams.opName, "The dimension of scale2 should be 3 in mx quant mode, but actual is %zu",
                        x2ScaleDimNum),
                return false);
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
                OP_LOGE(inputParams.opName, "The dim num of scale1 should be 3 in mx quant mode, but actual is %zu",
                        x1ScaleDimNum),
                return false);
    auto xScaleLastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - 1));
    auto xScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto xScaleMDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleLastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - 1));
    auto wScaleNDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto expectedKDimValue = inputParams.kSize / MXFP_BASEK_FACTOR + inputParams.groupNum;
    OP_CHECK_IF(
        xScaleLastDim != MXFP_MULTI_BASE_SIZE || xScaleKDim != expectedKDimValue || xScaleMDim != inputParams.mSize,
        OP_LOGE(
            inputParams.opName,
            "In mx quant mode, the expected shape of scale1 is ( %lu, %lu, %lu ), but the actual is ( %lu, %lu, %lu ).",
            expectedKDimValue, inputParams.mSize, MXFP_MULTI_BASE_SIZE, xScaleKDim, xScaleMDim, xScaleLastDim),
        return false);
    OP_CHECK_IF(
        wScaleLastDim != MXFP_MULTI_BASE_SIZE || wScaleKDim != expectedKDimValue || wScaleNDim != inputParams.nSize,
        OP_LOGE(
            inputParams.opName,
            "In mx quant mode, the expected shape of scale2 is ( %lu, %lu, %lu ), but the actual is ( %lu, %lu, %lu ).",
            expectedKDimValue, inputParams.nSize, MXFP_MULTI_BASE_SIZE, wScaleKDim, wScaleNDim, wScaleLastDim),
        return false);
    return true;
}

uint64_t GetTilingKeyForInplaceAdd(const GQmmInputInfo &inputParams)
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams.transB), static_cast<uint64_t>(inputParams.transA),
                              static_cast<uint64_t>(inputParams.kernelType));
}

}  // namespace QuantGroupedMatmulInplaceAdd
