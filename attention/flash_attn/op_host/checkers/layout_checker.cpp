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
 * \file layout_checker.cpp
 * \brief Checker for layout parameters (文档约束: layout匹配关系)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "layout_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus LayoutChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    const std::vector<FaLayout> supportedQLayouts = {FaLayout::BNSD, FaLayout::BSND, FaLayout::TND};
    const std::vector<FaLayout> supportedKvLayouts = {FaLayout::BNSD,    FaLayout::BSND,    FaLayout::TND,
                                                      FaLayout::PA_BBND, FaLayout::PA_BNBD, FaLayout::PA_Nz};
    const std::vector<FaLayout> supportedOutLayouts = {FaLayout::BNSD, FaLayout::BSND, FaLayout::TND};

    OP_CHECK_IF(std::find(supportedQLayouts.begin(), supportedQLayouts.end(), faInfo.qLayout) ==
                    supportedQLayouts.end(),
                OP_LOGE(faInfo.opName, "layout_q only supports BNSD/BSND/TND, but got %s",
                        LayoutToSerialString(faInfo.qLayout).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), faInfo.kvLayout) ==
                    supportedKvLayouts.end(),
                OP_LOGE(faInfo.opName, "layout_kv only supports BNSD/BSND/TND/PA_BBND/PA_BNBD/PA_Nz, but got %s",
                        LayoutToSerialString(faInfo.kvLayout).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), faInfo.outLayout) ==
                    supportedOutLayouts.end(),
                OP_LOGE(faInfo.opName, "layout_out only supports BNSD/BSND/TND, but got %s",
                        LayoutToSerialString(faInfo.outLayout).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayoutChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (faInfo.qLayout == FaLayout::BNSD) {
        const std::vector<FaLayout> supportedKvLayouts = {FaLayout::BNSD, FaLayout::PA_BBND, FaLayout::PA_BNBD,
                                                          FaLayout::PA_Nz};
        const std::vector<FaLayout> supportedOutLayouts = {FaLayout::BNSD, FaLayout::BSND};

        OP_CHECK_IF(std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), faInfo.kvLayout) ==
                        supportedKvLayouts.end(),
                    OP_LOGE(faInfo.opName,
                            "When layout_q is BNSD, layout_kv must be one of BNSD/PA_BBND/PA_BNBD/PA_Nz, but got %s",
                            LayoutToSerialString(faInfo.kvLayout).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), faInfo.outLayout) ==
                        supportedOutLayouts.end(),
                    OP_LOGE(faInfo.opName, "When layout_q is BNSD, layout_out must be one of BNSD/BSND, but got %s",
                            LayoutToSerialString(faInfo.outLayout).c_str()),
                    return ge::GRAPH_FAILED);
    } else if (faInfo.qLayout == FaLayout::BSND) {
        const std::vector<FaLayout> supportedKvLayouts = {FaLayout::BSND, FaLayout::PA_BBND, FaLayout::PA_BNBD,
                                                          FaLayout::PA_Nz};
        const std::vector<FaLayout> supportedOutLayouts = {FaLayout::BSND};

        OP_CHECK_IF(std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), faInfo.kvLayout) ==
                        supportedKvLayouts.end(),
                    OP_LOGE(faInfo.opName,
                            "When layout_q is BSND, layout_kv must be one of BSND/PA_BBND/PA_BNBD/PA_Nz, but got %s",
                            LayoutToSerialString(faInfo.kvLayout).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), faInfo.outLayout) ==
                        supportedOutLayouts.end(),
                    OP_LOGE(faInfo.opName, "When layout_q is BSND, layout_out must be BSND, but got %s",
                            LayoutToSerialString(faInfo.outLayout).c_str()),
                    return ge::GRAPH_FAILED);
    } else if (faInfo.qLayout == FaLayout::TND) {
        const std::vector<FaLayout> supportedKvLayouts = {FaLayout::TND, FaLayout::PA_BBND, FaLayout::PA_BNBD,
                                                          FaLayout::PA_Nz};
        const std::vector<FaLayout> supportedOutLayouts = {FaLayout::TND};

        OP_CHECK_IF(std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), faInfo.kvLayout) ==
                        supportedKvLayouts.end(),
                    OP_LOGE(faInfo.opName,
                            "When layout_q is TND, layout_kv must be one of TND/PA_BBND/PA_BNBD/PA_Nz, but got %s",
                            LayoutToSerialString(faInfo.kvLayout).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), faInfo.outLayout) ==
                        supportedOutLayouts.end(),
                    OP_LOGE(faInfo.opName, "When layout_q is TND, layout_out must be TND, but got %s",
                            LayoutToSerialString(faInfo.outLayout).c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling