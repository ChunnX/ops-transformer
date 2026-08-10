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
 * \file mask_checker.cpp
 * \brief Implementation of MaskChecker for Mask参数组.
 */

#include "mask_checker.h"
#include "log/log.h"

namespace optiling {

using std::to_string;
using std::vector;

ge::graphStatus MaskChecker::CheckOriMaskMode(const QSMLATilingInfo &info) const
{
    const vector<uint64_t> supportList = {MASK_MODE_NO_MASK, MASK_MODE_CAUSAL, MASK_MODE_SLIDING_WINDOW};
    if (CheckValueSupport(info.oriMaskMode, supportList, "ori_mask_mode") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckCmpMaskMode(const QSMLATilingInfo &info) const
{
    if (info.templateMode == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(info.cmpMaskMode != MASK_MODE_NO_MASK,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "cmp_mask_mode",
                                                          to_string(info.cmpMaskMode).c_str(),
                                                          "cmp_mask_mode only support 0 in SWA mode"),
                    return ge::GRAPH_FAILED);
    } else {
        const vector<uint64_t> supportList = {MASK_MODE_NO_MASK, MASK_MODE_CAUSAL};
        if (CheckValueSupport(info.cmpMaskMode, supportList, "cmp_mask_mode") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckOriWinLeftRight(const QSMLATilingInfo &info) const
{
    OP_CHECK_IF(info.oriWinLeft != WIN_LEFT_RIGHT_UNLIMITED && info.oriWinLeft < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "ori_win_left", to_string(info.oriWinLeft).c_str(),
                                                      "ori_win_left only support -1 or non-negative"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.oriWinRight != WIN_LEFT_RIGHT_UNLIMITED && info.oriWinRight < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "ori_win_right", to_string(info.oriWinRight).c_str(),
                                                      "ori_win_right only support -1 or non-negative"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSinglePara(const QSMLATilingInfo &info) const
{
    if (CheckOriMaskMode(info) != ge::GRAPH_SUCCESS || CheckCmpMaskMode(info) != ge::GRAPH_SUCCESS ||
        CheckOriWinLeftRight(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const QSMLATilingInfo &info) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckCrossFeature(const QSMLATilingInfo &info) const
{
    if (info.templateMode == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        OP_CHECK_IF(info.cmpMaskMode != MASK_MODE_NO_MASK || info.oriMaskMode != MASK_MODE_NO_MASK,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        info.opName, "ori_mask_mode and cmp_mask_mode",
                        (to_string(info.oriMaskMode) + " and " + to_string(info.cmpMaskMode)).c_str(),
                        "In ori-only sparse scenario, both ori_mask_mode and cmp_mask_mode must be 0"),
                    return ge::GRAPH_FAILED);
    }

    if (info.templateMode == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        const vector<uint64_t> swaSupport = {MASK_MODE_NO_MASK, MASK_MODE_CAUSAL, MASK_MODE_SLIDING_WINDOW};
        if (CheckValueSupport(info.oriMaskMode, swaSupport, "ori_mask_mode in SWA") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (info.templateMode == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(info.cmpMaskMode != MASK_MODE_NO_MASK,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        info.opName, "cmp_mask_mode",
                        (to_string(info.cmpMaskMode)).c_str(),
                        "In SWA mode, cmp_mask_mode only support 0"),
                    return ge::GRAPH_FAILED);
    }

    if (info.oriMaskMode != MASK_MODE_SLIDING_WINDOW) {
        OP_CHECK_IF(info.oriWinLeft >= 0 || info.oriWinRight >= 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        info.opName, "ori_win_left and ori_win_right",
                        (to_string(info.oriWinLeft) + " and " + to_string(info.oriWinRight)).c_str(),
                        "ori_win_left/ori_win_right can be >= 0 only when ori_mask_mode=4"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMultiParaConsistency(const QSMLATilingInfo &info) const
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
