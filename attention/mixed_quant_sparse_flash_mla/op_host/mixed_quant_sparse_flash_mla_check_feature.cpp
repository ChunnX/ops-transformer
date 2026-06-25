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
 * \file mixed_quant_sparse_flash_mla_check_feature.cpp
 * \brief
 */

#include "mixed_quant_sparse_flash_mla_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

ge::graphStatus QSMLATilingCheck::CheckFeatureWinKV() const
{
    OP_CHECK_IF(oriWinLeft_ != 127, // 127:当前不泛化
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "oriWinLeft_",
            std::to_string(oriWinLeft_).c_str(), "oriWinLeft_ only support 127"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(oriWinRight_ != 0, // 0:当前不泛化
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "oriWinRight_",
            std::to_string(oriWinRight_).c_str(), "oriWinRight_ only support 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquantShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
        OP_LOGE(opName_, "batch_size should be greater than 0, but got %u", bSize_),
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "batch_size",
            std::to_string(bSize_).c_str(), "batch_size should be greater than 0"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == QSMLALayout::TND),
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "T_size of query",
            std::to_string(qTSize_).c_str(), "T_size of query should be greater than 0"),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "kv_head_num",
            std::to_string(n2Size_).c_str(), "kv_head_num only support 1"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0,
        OP_LOGE_WITH_INVALID_VALUES_WITH_REASON(opName_, "q_head_num and kv_head_num",
            std::to_string(n1Size_) + " and " + std::to_string(n2Size_),
            "q_head_num must be divisible by kv_head_num"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(dSize_ != 512, // 512:当前不泛化
        OP_LOGE(opName_, "Head dim of input q only support 512, but got %u", dSize_),
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "Head dim of input q",
            std::to_string(dSize_).c_str(), "Head dim of input q only support 512"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(dSizeV_ != 512, // 512:当前不泛化
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "dSizeV",
            std::to_string(dSizeV_).c_str(), "dSizeV only support 512"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(dSizeVInput_ != KV_INPUT_DIM_LIMIT, // 608:当前不泛化
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "dSizeVInput",
            std::to_string(dSizeVInput_).c_str(), "dSizeVInput only support " + std::to_string(KV_INPUT_DIM_LIMIT)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQ;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
        OP_LOGE_WITH_INVALID_FORMAT(opName_, "layoutQuery",
            layoutQuery.c_str(), "BSND/TND"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquantDtype() const
{
    OP_CHECK_IF(qType_ != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "query",
            QSMLADataTypeToSerialString(qType_).c_str(),
            "query dtype only support " + QSMLADataTypeToSerialString(ge::DT_BF16) + " and " +
            QSMLADataTypeToSerialString(ge::DT_FLOAT16)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquantAttr() const
{
    // 1 for fp8_e4m3fn per-tile and 10 for hif8 per-tensor
    OP_CHECK_IF(*opParamInfo_.kvQuantMode != 1 && *opParamInfo_.kvQuantMode != 10,
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "kv_quant_mode_",
            std::to_string(*opParamInfo_.kvQuantMode).c_str(),
            "kv_quant_mode_ only support 1 and 10"),
        return ge::GRAPH_FAILED);

    if (*opParamInfo_.kvQuantMode == 1) {
        OP_CHECK_IF(opParamInfo_.oriKv.desc->GetDataType() == ge::DT_HIFLOAT8, // 前面已校验dtype在fp8 e4m3和hif8之间
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "oriKv", "DT_HIFLOAT8",
                "oriKv dtype only support DT_FLOAT8_E4M3FN"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(opParamInfo_.oriKv.desc->GetDataType() == ge::DT_FLOAT8_E4M3FN, // 前面已校验dtype在fp8 e4m3和hif8之间
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "oriKv", "DT_FLOAT8_E4M3FN",
                "oriKv dtype only support DT_HIFLOAT8"),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(*opParamInfo_.ropeHeadDim != 64, // 64:当前不泛化
        OP_LOGE_WITH_INVALID_VALUE_WITH_REASON(opName_, "rope_head_dim",
            std::to_string(*opParamInfo_.ropeHeadDim).c_str(),
            "rope_head_dim only support 64"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquantPa() const
{
    if (kvLayout_ != QSMLALayout::PA_BBND) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(oriBlockSize_ <= 0 || oriBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OP_LOGE(opName_, "when page attention is enabled, oriBlockSize_(%u) should be in range (0, %u].",
        oriBlockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

    if (cmpBlockSize_ != 0) {
        OP_CHECK_IF(cmpBlockSize_ <= 0 || cmpBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
            OP_LOGE(opName_, "when page attention is enabled, cmpBlockSize_(%u) should be in range (0, %u].",
            cmpBlockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureAntiquantAttr() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureWinKV() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeature() const
{
    return CheckFeatureAntiquant();
}

}