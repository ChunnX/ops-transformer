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
 * \file softmax_lse_checker.cpp
 * \brief Implementation of SoftmaxLseChecker for SoftmaxLse参数组.
 */

#include "softmax_lse_checker.h"
#include "log/log.h"

namespace optiling {

using std::string;
using std::to_string;
using std::vector;

ge::graphStatus SoftmaxLseChecker::CheckSinglePara(const QSMLATilingInfo &info) const
{
    const auto &lse = info.opParamInfo.softmaxLse;
    if (lse.desc != nullptr) {
        if (CheckDtypeSupport(lse.desc, SOFTMAX_LSE_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLseChecker::CheckParaExistence(const QSMLATilingInfo &info) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLseChecker::CheckCrossFeature(const QSMLATilingInfo &info) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLseChecker::CheckMultiParaConsistency(const QSMLATilingInfo &info) const
{
    const auto &lse = info.opParamInfo.softmaxLse;
    if (lse.shape == nullptr)
        return ge::GRAPH_SUCCESS;

    if (!info.returnSoftmaxLse) {
        OP_CHECK_IF(lse.shape->GetStorageShape().GetShapeSize() != 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        info.opName, "softmax_lse", to_string(lse.shape->GetStorageShape().GetShapeSize()).c_str(),
                        "softmax_lse shapeSize should be 1 when return_softmax_lse=false"),
                    return ge::GRAPH_FAILED);
    } else {
        const auto &lseShape = lse.shape->GetStorageShape();
        if (info.qLayout == QSMLALayout::BSND) {
            OP_CHECK_IF(lseShape.GetDimNum() != DIM_NUM_FOUR,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(info.opName, "softmax_lse",
                                                                 to_string(lseShape.GetDimNum()).c_str(),
                                                                 "softmax_lse should be 4D when layout_q=BSND"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(lseShape.GetDim(DIM_0) != static_cast<int64_t>(info.bSize),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            info.opName, "softmax_lse", GetShapeStr(lseShape).c_str(),
                            ("softmax_lse B should equal q B(" + to_string(info.bSize) + ")").c_str()),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(lseShape.GetDimNum() != DIM_NUM_THREE,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(info.opName, "softmax_lse",
                                                                 to_string(lseShape.GetDimNum()).c_str(),
                                                                 "softmax_lse should be 3D when layout_q=TND"),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
