/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file softmax_lse_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "softmax_lse_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// CheckSinglePara
ge::graphStatus SoftmaxLSEChecker::CheckSingleDtype(const FaTilingInfo &faInfo)
{
    // SoftmaxLse only supports output FP32
    if (faInfo.softmaxLseFlag) {
        OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.lseOut.desc, SOFTMAX_LSE_NAME),
                    OP_LOGE(faInfo.opName, "SoftmaxLse only support dtype FP32, but got %s",
                            DataTypeToSerialString(faInfo.opParamInfo.lseOut.desc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckParaExistence
ge::graphStatus SoftmaxLSEChecker::CheckExistenceShapeAndDesc(const FaTilingInfo &faInfo)
{
    // When softmaxLseFlag is true, both the SoftmaxLse shape and the lseOut tensor must not be null.
    if (faInfo.softmaxLseFlag) {
        const gert::StorageShape *lseShape = faInfo.opParamInfo.lseOut.shape;
        OP_CHECK_IF(lseShape == nullptr,
                    OP_LOGE(faInfo.opName, "SoftmaxLse shape is nullptr but softmaxLseFlag is true!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(faInfo.opParamInfo.lseOut.desc == nullptr,
                    OP_LOGE(faInfo.opName, "Desc of lseOut tensor is nullptr!"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckMultiPara
ge::graphStatus SoftmaxLSEChecker::CheckMultiParaDimAndShape(const FaTilingInfo &faInfo)
{
    // When softmaxLseFlag is true and emptyTensorFlag is false:
    // -If the LayOut is TND, the shape must have 3 dimensions (matching TN1).
    // -Otherwise, the shape must have 4 dimensions (matching [B,N,S1,1]).
    if (faInfo.softmaxLseFlag && (!faInfo.emptyTensorFlag)) {
        const gert::StorageShape *lseShape = faInfo.opParamInfo.lseOut.shape;
        std::shared_ptr<FaTilingShapeCompare> softmaxLseShapeCmp_ = nullptr;
        if (faInfo.outLayout == FaLayout::TND || faInfo.outLayout == FaLayout::NTD) {
            OP_CHECK_IF(lseShape->GetStorageShape().GetDimNum() != DIM_NUM_3,
                        OP_LOGE(faInfo.opName, "TND/NTD SoftmaxLse shape dim(%zu) should be 3!",
                                lseShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(lseShape->GetStorageShape().GetDim(DIM_NUM_0) != faInfo.qTSize ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_1) != faInfo.n1Size ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_2) != SHAPE_PARAMS_CONST,
                        OP_LOGE(faInfo.opName, "SoftmaxLse shape[%ld, %ld, %ld] does not match TN1!",
                                lseShape->GetStorageShape().GetDim(DIM_NUM_0),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_1),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_2)),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(lseShape->GetStorageShape().GetDimNum() != DIM_NUM_4,
                        OP_LOGE(faInfo.opName, "SoftmaxLse shape dim(%zu) should be 4!",
                                lseShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(lseShape->GetStorageShape().GetDim(DIM_NUM_0) != faInfo.bSize ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_1) != faInfo.n1Size ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_2) != faInfo.s1Size ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_3) != SHAPE_PARAMS_CONST,
                        OP_LOGE(faInfo.opName, "SoftmaxLse shape[%ld, %ld, %ld, %ld] does not match [B, N, Q_S, 1]!",
                                lseShape->GetStorageShape().GetDim(DIM_NUM_0),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_1),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_2),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_3)),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

// enableAntiQuant 相关校验函数
ge::graphStatus SoftmaxLSEChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSingleDtype(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckExistenceShapeAndDesc(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckMultiParaDimAndShape(faInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
