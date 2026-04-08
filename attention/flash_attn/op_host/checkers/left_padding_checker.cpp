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
 * \file left_padding_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "left_padding_checker.h"

namespace optilingForBuildPKG {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// Utility functions for common checkers.
ge::graphStatus LeftPaddingChecker::CheckShapeSupport(const gert::Tensor *tensor,
                                                      const std::vector<int64_t> &expectShapeList) const
{
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectShapeList.begin(), expectShapeList.end(), tensor->GetShapeSize()) == expectShapeList.end()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// CheckSinglePara
ge::graphStatus LeftPaddingChecker::CheckShapeAndDim(const FaTilingInfo &faInfo)
{
    // When left-padding is enabled for Query and Key/Value,
    // the Shape size and Dim number corresponding to the Padding size must both be 1.
    if (faInfo.qPaddingSizeFlag) {
        const std::vector<int64_t> querypaddingsizeShapeNumList = {SHAPE_NUM_ONE};
        OP_CHECK_IF(ge::GRAPH_SUCCESS !=
                        CheckShapeSupport(faInfo.opParamInfo.queryPaddingSize.tensor, querypaddingsizeShapeNumList),
                    OP_LOGE(faInfo.opName, "The shape size of query paddingsize is not 1!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(faInfo.opParamInfo.queryPaddingSize.tensor->GetStorageShape().GetDimNum() != DIM_NUM_1,
                    OP_LOGE(faInfo.opName, "The dim number of query paddingsize is not 1!"), return ge::GRAPH_FAILED);
    }

    if (faInfo.kvPaddingSizeFlag) {
        const std::vector<int64_t> kvpaddingsizeShapeNumList = {SHAPE_NUM_ONE};
        OP_CHECK_IF(
            ge::GRAPH_SUCCESS != CheckShapeSupport(faInfo.opParamInfo.kvPaddingSize.tensor, kvpaddingsizeShapeNumList),
            OP_LOGE(faInfo.opName, "The shape size of kv paddingsize is not 1!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(faInfo.opParamInfo.kvPaddingSize.tensor->GetStorageShape().GetDimNum() != DIM_NUM_1,
                    OP_LOGE(faInfo.opName, "The dim number of kv paddingsize is not 1!"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// CheckExistence
ge::graphStatus LeftPaddingChecker::CheckExistenceDesc(const FaTilingInfo &faInfo)
{
    if (faInfo.qPaddingSizeFlag) {
        OP_CHECK_IF(faInfo.opParamInfo.queryPaddingSize.desc == nullptr,
                    OP_LOGE(faInfo.opName, "Desc of tensor query_padding_size is nullptr!"), return ge::GRAPH_FAILED);
    }

    if (faInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(faInfo.opParamInfo.kvPaddingSize.desc == nullptr,
                    OP_LOGE(faInfo.opName, "Desc of tensor kv_padding_size is nullptr!"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus LeftPaddingChecker::CheckFeatureActualLen(const FaTilingInfo &faInfo)
{
    // When left-padding is enabled for Query and Key/Value, the actual sequence length information must be provided.
    if (faInfo.qPaddingSizeFlag && !faInfo.isMaxWorkspace) {
        OP_CHECK_IF(faInfo.opParamInfo.actualSeqLengthsQ.tensor == nullptr ||
                        faInfo.opParamInfo.actualSeqLengthsQ.tensor->GetShapeSize() == 0U,
                    OP_LOGE(faInfo.opName,
                            "When query_padding_size exists, the query's actual sequence lengths are required!"),
                    return ge::GRAPH_FAILED);
    }

    if (faInfo.kvPaddingSizeFlag && !faInfo.isMaxWorkspace) {
        OP_CHECK_IF(faInfo.opParamInfo.actualSeqLengths.tensor == nullptr ||
                        faInfo.opParamInfo.actualSeqLengths.tensor->GetShapeSize() == 0U,
                    OP_LOGE(faInfo.opName,
                            "When kv_padding_size exists, the key/value's actual sequence lengths are required!"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeatureLayout(const FaTilingInfo &faInfo)
{
    // When left-padding is enabled for Query and Key/Value, BSH_BNSD/BSND_BNSD/TND/NTD/NTD_TND/TND_NTD scenarios are not supported.
    std::string layoutStr(faInfo.opParamInfo.layOut);
    if (faInfo.qPaddingSizeFlag || faInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(layoutStr == "BSH_BNSD",
                    OP_LOGE(faInfo.opName, "QueryLeftPadding illegal condition:input layout BSH_BNSD!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(layoutStr == "BSND_BNSD",
                    OP_LOGE(faInfo.opName, "QueryLeftPadding illegal condition:input layout BSND_BNSD!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(layoutStr == "TND",
                    OP_LOGE(faInfo.opName, "QueryLeftPadding illegal condition:input layout TND!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(layoutStr == "NTD",
                    OP_LOGE(faInfo.opName, "QueryLeftPadding illegal condition:input layout NTD!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(layoutStr == "NTD_TND",
                    OP_LOGE(faInfo.opName, "QueryLeftPadding illegal condition:input layout NTD_TND!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(layoutStr == "TND_NTD",
                    OP_LOGE(faInfo.opName, "QueryLeftPadding illegal condition:input layout TND_NTD!"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeatureAlibiPse(const FaTilingInfo &faInfo)
{
    // When left-padding is enabled for Query and Key/Value, PseType = 2/3 is not supported.
    if (faInfo.qPaddingSizeFlag || faInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(faInfo.enableAlibiPse,
                    OP_LOGE(faInfo.opName, "When pseType = 2/3, left padding is not supported!"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeaturePageAttention(const FaTilingInfo &faInfo)
{
    // When left-padding is enabled for Query and Key/Value, PageAttention scenarios are not supported.
    if (faInfo.qPaddingSizeFlag || faInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(faInfo.pageAttentionFlag,
                    OP_LOGE(faInfo.opName, "When page attention is used, left padding is not supported!"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckShapeAndDim(faInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckExistenceDesc(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureActualLen(faInfo) || ge::GRAPH_SUCCESS != CheckFeatureLayout(faInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureAlibiPse(faInfo) || ge::GRAPH_SUCCESS != CheckFeaturePageAttention(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace optilingForBuildPKG
