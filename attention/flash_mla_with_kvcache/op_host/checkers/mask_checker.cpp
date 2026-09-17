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
 * \brief Checker for mask_mode, attn_mask parameters (文档约束: Mask参数组)
 *
 * Modes 3 and 4 use the shared [2048, 2048] upper-triangular mask template.
 * Mode 4 applies a right-aligned inclusive window. -1 means unbounded.
 */

#include <map>
#include <numeric>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_with_kvcache_tiling_info.h"
#include "mask_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;

ge::graphStatus MaskChecker::CheckSingleParaMaskMode(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    const std::vector<int64_t> maskModeList = {static_cast<int64_t>(MaskMode::NO_MASK),
                                               static_cast<int64_t>(MaskMode::CAUSAL),
                                               static_cast<int64_t>(MaskMode::BAND)};
    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != CheckValueSupport(static_cast<int64_t>(faInfo.maskMode), maskModeList),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "mask_mode", std::to_string(faInfo.maskMode).c_str(),
                                              "mask_mode must be 0 (NO_MASK), 3 (CAUSAL) or 4 (BAND)"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSingleParaAttnMask(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    auto &attnMaskTensor = faInfo.opParamInfo.attnMask.tensor;
    if (attnMaskTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *attnMaskDesc = faInfo.opParamInfo.attnMask.desc;
    OP_CHECK_IF(attnMaskDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "TensorDesc of attn_mask"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(attnMaskDesc->GetDataType() != ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "attn_mask",
                                          Ops::Base::ToString(attnMaskDesc->GetDataType()).c_str(), "INT8"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(attnMaskDesc, ATTN_MASK_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = attnMaskTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "attn_mask", (std::to_string(dimNum) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);

    int64_t dim0 = attnMaskTensor->GetStorageShape().GetDim(0);
    int64_t dim1 = attnMaskTensor->GetStorageShape().GetDim(1);
    OP_CHECK_IF(
        dim0 != SPARSE_OPTIMIZE_ATTENTION_SIZE || dim1 != SPARSE_OPTIMIZE_ATTENTION_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "attn_mask",
                                              ("[" + std::to_string(dim0) + ", " + std::to_string(dim1) + "]").c_str(),
                                              "The shape of attn_mask must be [2048, 2048]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if (CheckSingleParaMaskMode(faInfo) != ge::GRAPH_SUCCESS || CheckSingleParaAttnMask(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    auto &attnMaskTensor = faInfo.opParamInfo.attnMask.tensor;
    OP_CHECK_IF(faInfo.winLeft < -1 || faInfo.winLeft > MASK_MODE_INT_MAX || faInfo.winRight < -1 ||
                    faInfo.winRight > MASK_MODE_INT_MAX,
                OP_LOGE(faInfo.opName, "win_left and win_right must be in [-1, INT32_MAX]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        faInfo.maskMode != static_cast<int64_t>(MaskMode::BAND) && (faInfo.winLeft != -1 || faInfo.winRight != -1),
        OP_LOGE(faInfo.opName, "win_left and win_right must be -1 for mask_mode 0 or 3."), return ge::GRAPH_FAILED);

    if (faInfo.maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) {
        OP_CHECK_IF(attnMaskTensor != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "attn_mask", "not empty",
                                                          "When mask_mode=0 (no mask mode), attn_mask must be empty"),
                    return ge::GRAPH_FAILED);
    }
    if (faInfo.maskMode == static_cast<int64_t>(MaskMode::CAUSAL) ||
        faInfo.maskMode == static_cast<int64_t>(MaskMode::BAND)) {
        OP_CHECK_IF(attnMaskTensor == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "attn_mask", "empty",
                                                          "When mask_mode=3 or 4, attn_mask must be provided"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla_with_kvcache
} // namespace optiling
