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
 * \file fa_checker.cpp
 * \brief Main checker for npu_flash_attn parameters
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "fa_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;

ge::graphStatus FAChecker::Init(const FaTilingInfo &faInfo)
{
    if (faInfo.inputQType != faInfo.inputKvType) {
        enableAntiQuant_ = true;
    } else if (faInfo.inputQType == ge::DT_FLOAT16 || faInfo.inputQType == ge::DT_BF16) {
        enableNonQuant_ = true;
    } else {
        enableFullQuant_ = true;
    }

    actualSeqLenChecker_ = std::make_unique<ActualSeqLenChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    emptyTensorChecker_ = std::make_unique<EmptyTensorChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    layoutChecker_ = std::make_unique<LayoutChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    maskChecker_ = std::make_unique<MaskChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    metadataChecker_ = std::make_unique<MetadataChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    pagedAttentionChecker_ =
        std::make_unique<PagedAttentionChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    sinksChecker_ = std::make_unique<SinksChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    shapeChecker_ = std::make_unique<ShapeChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    softmaxLSEChecker_ = std::make_unique<SoftmaxLSEChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FAChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != shapeChecker_->CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != layoutChecker_->CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != maskChecker_->CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != sinksChecker_->CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != metadataChecker_->CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FAChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != shapeChecker_->CheckParaExistence(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckParaExistence(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckParaExistence(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckParaExistence(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FAChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != shapeChecker_->CheckFeature(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckFeature(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FAChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != shapeChecker_->CheckMultiPara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != layoutChecker_->CheckMultiPara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckMultiPara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckMultiPara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckMultiPara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FAChecker::Process(const FaTilingInfo &faInfo)
{
    if (emptyTensorChecker_->IsEmptyTensorScenario(faInfo)) {
        OP_LOGI(faInfo.opName, "Empty tensor scenario detected, skip normal validation.");
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckParaExistence(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckFeature(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckMultiPara(faInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling