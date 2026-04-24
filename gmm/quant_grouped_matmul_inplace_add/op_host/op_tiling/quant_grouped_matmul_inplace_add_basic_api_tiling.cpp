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
#include "quant_grouped_matmul_inplace_add_tiling_utils.h"

using namespace Ops::Transformer::OpTiling;
using namespace QuantGroupedMatmulInplaceAdd;
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
    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    if (scaleStorageShape == nullptr) {
        return false;
    }
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();

    return inputParams_.aDtype == ge::DT_HIFLOAT8 &&
           (scaleDimNum == 1 || (scaleDimNum == 2 && static_cast<uint64_t>(wScaleShape.GetDim(1)) == 1));
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeAttrs()
{
    return AnalyzeAttrsForInplaceAdd(context_, inputParams_);
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeDtype()
{
    return AnalyzeDtypeForInplaceAdd(context_, inputParams_) && CheckDtypeForInplaceAdd(inputParams_);
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

    // HIFLOAT8 T-T 分支：IsCapable 已保证 wScale 为 (g,) 或 (g,1)，复用合并后的 T-T/T-C 校验
    OP_CHECK_IF(!CheckShapeForHif8Quant(x1ScaleShape, wScaleShape, inputParams_),
                OP_LOGE(inputParams_.opName, "CheckShapeForHif8Quant failed."), return false);

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

bool QGMMInplaceAddBasicApiTiling::CheckCoreNum() const
{
    return CheckCoreNumForInplaceAdd(context_, inputParams_);
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