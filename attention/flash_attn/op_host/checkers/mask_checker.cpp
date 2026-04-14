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
 * \file mask_checker.cpp
 * \brief Checker for mask_mode parameter (文档约束: Mask参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "mask_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus MaskChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    const std::vector<int32_t> maskModeList = {1, 3, 4};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(faInfo.sparseMode, maskModeList),
                OP_LOGE(faInfo.opName, "mask_mode only supports 1/3/4, but got %d", faInfo.sparseMode),
                return ge::GRAPH_FAILED);

    if (faInfo.sparseMode == SPARSE_MODE_BAND) {
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

} // namespace optiling
