/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mask_checker.cpp
 * \brief Checker for mask_mode, attn_mask, win_left, win_right parameters (文档约束: Mask参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "../flash_attn_tiling_constants.h"
#include "mask_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;

ge::graphStatus MaskChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    const std::vector<int32_t> maskModeList = {MASK_MODE_ALL_MASK, MASK_MODE_RIGHT_DOWN, MASK_MODE_BAND};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(faInfo.maskMode, maskModeList),
                OP_LOGE(faInfo.opName, "mask_mode only supports 1/3/4, but got %d", faInfo.maskMode),
                return ge::GRAPH_FAILED);

    auto &attnMaskTensor = faInfo.opParamInfo.attnMask.tensor;
    if (attnMaskTensor != nullptr) {
        const gert::CompileTimeTensorDesc *attnMaskDesc = faInfo.opParamInfo.attnMask.desc;
        OP_CHECK_IF(attnMaskDesc == nullptr,
            OP_LOGE(faInfo.opName, "attn_mask desc is null pointer!"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(attnMaskDesc->GetDataType() != ge::DT_INT8,
            OP_LOGE(faInfo.opName, "attn_mask dtype must be INT8, but got %s",
                    DataTypeToSerialString(attnMaskDesc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);

        uint32_t dimNum = attnMaskTensor->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum != 2,
            OP_LOGE(faInfo.opName, "attn_mask dim num must be 2, but got %u", dimNum),
            return ge::GRAPH_FAILED);

        int64_t dim0 = attnMaskTensor->GetStorageShape().GetDim(0);
        int64_t dim1 = attnMaskTensor->GetStorageShape().GetDim(1);
        OP_CHECK_IF(dim0 != 2048 || dim1 != 2048,
            OP_LOGE(faInfo.opName, "attn_mask shape must be (2048, 2048), but got (%ld, %ld)", dim0, dim1),
            return ge::GRAPH_FAILED);
    }

    if (faInfo.maskMode == MASK_MODE_BAND) {
        auto &opParamInfo = faInfo.opParamInfo;
        if (opParamInfo.preToken != nullptr && opParamInfo.nextToken != nullptr) {
            int64_t winLeft = *opParamInfo.preToken;
            int64_t winRight = *opParamInfo.nextToken;
            OP_CHECK_IF(winLeft > winRight,
                OP_LOGE(faInfo.opName, "win_left(%ld) should not be greater than win_right(%ld) in band mode.",
                        winLeft, winRight),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    auto &attnMaskTensor = faInfo.opParamInfo.attnMask.tensor;
    auto &opParamInfo = faInfo.opParamInfo;
    bool hasWinLeft = (opParamInfo.preToken != nullptr && *opParamInfo.preToken != -1);
    bool hasWinRight = (opParamInfo.nextToken != nullptr && *opParamInfo.nextToken != -1);

    if (faInfo.maskMode == MASK_MODE_ALL_MASK) {
        OP_CHECK_IF(attnMaskTensor != nullptr,
            OP_LOGE(faInfo.opName, "attn_mask should not be provided when mask_mode=1 (all mask mode)."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(hasWinLeft || hasWinRight,
            OP_LOGE(faInfo.opName, "win_left and win_right should not be provided when mask_mode=1 (all mask mode)."),
            return ge::GRAPH_FAILED);
    }

    if (faInfo.maskMode == MASK_MODE_RIGHT_DOWN) {
        OP_CHECK_IF(attnMaskTensor == nullptr,
            OP_LOGE(faInfo.opName, "attn_mask must be provided when mask_mode=3 (causal mode)."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(hasWinLeft || hasWinRight,
            OP_LOGE(faInfo.opName, "win_left and win_right should not be provided when mask_mode=3 (causal mode)."),
            return ge::GRAPH_FAILED);
    }

    if (faInfo.maskMode == MASK_MODE_BAND) {
        OP_CHECK_IF(attnMaskTensor == nullptr,
            OP_LOGE(faInfo.opName, "attn_mask must be provided when mask_mode=4 (band mode)."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling