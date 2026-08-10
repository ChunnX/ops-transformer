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
 * \file qsmla_checker.cpp
 * \brief Implementation of QSMLAChecker orchestrator.
 *
 *        Process() runs the four-phase check pipeline by delegating to each
 *        sub-checker in order. The templateMode is computed once and stored
 *        in a mutable copy of QSMLATilingInfo so all sub-checkers can read it.
 */

#include "qsmla_checker.h"
#include "log/log.h"

namespace optiling {

ge::graphStatus QSMLAChecker::Process(const QSMLATilingInfo &info)
{
    // Compute template mode from input combination
    QSMLATemplateMode templateMode;
    const auto &p = info.opParamInfo;
    if (p.cmpKv.tensor == nullptr) {
        if (p.oriSparseIndices.tensor != nullptr) {
            templateMode = QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE;
        } else {
            templateMode = QSMLATemplateMode::SWA_TEMPLATE_MODE;
        }
    } else if (p.cmpSparseIndices.tensor != nullptr) {
        if (p.oriSparseIndices.tensor != nullptr) {
            templateMode = QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE;
        } else {
            templateMode = QSMLATemplateMode::CSA_TEMPLATE_MODE;
        }
    } else {
        templateMode = QSMLATemplateMode::HCA_TEMPLATE_MODE;
    }

    // Store templateMode in a mutable copy so sub-checkers can read it
    QSMLATilingInfo mutableInfo = info;
    mutableInfo.templateMode = templateMode;

    // Instantiate sub-checkers
    commonChecker_ = std::make_unique<CommonChecker>();
    maskChecker_ = std::make_unique<MaskChecker>();
    seqlensChecker_ = std::make_unique<SeqLensChecker>();
    sparseCompressChecker_ = std::make_unique<SparseCompressChecker>();
    pagedAttentionChecker_ = std::make_unique<PagedAttentionChecker>();
    sinksChecker_ = std::make_unique<SinksChecker>();
    softmaxLseChecker_ = std::make_unique<SoftmaxLseChecker>();
    // Phase 1: CheckSinglePara — dtype / shape / format per tensor, value per attribute
    if (commonChecker_->CheckSinglePara(mutableInfo) != ge::GRAPH_SUCCESS ||
        maskChecker_->CheckSinglePara(mutableInfo) != ge::GRAPH_SUCCESS ||
        seqlensChecker_->CheckSinglePara(mutableInfo) != ge::GRAPH_SUCCESS ||
        sparseCompressChecker_->CheckSinglePara(mutableInfo) != ge::GRAPH_SUCCESS ||
        pagedAttentionChecker_->CheckSinglePara(mutableInfo) != ge::GRAPH_SUCCESS ||
        sinksChecker_->CheckSinglePara(mutableInfo) != ge::GRAPH_SUCCESS ||
        softmaxLseChecker_->CheckSinglePara(mutableInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Phase 2: CheckParaExistence — required / optional / forbidden rules
    if (commonChecker_->CheckParaExistence(mutableInfo) != ge::GRAPH_SUCCESS ||
        maskChecker_->CheckParaExistence(mutableInfo) != ge::GRAPH_SUCCESS ||
        seqlensChecker_->CheckParaExistence(mutableInfo) != ge::GRAPH_SUCCESS ||
        sparseCompressChecker_->CheckParaExistence(mutableInfo) != ge::GRAPH_SUCCESS ||
        pagedAttentionChecker_->CheckParaExistence(mutableInfo) != ge::GRAPH_SUCCESS ||
        sinksChecker_->CheckParaExistence(mutableInfo) != ge::GRAPH_SUCCESS ||
        softmaxLseChecker_->CheckParaExistence(mutableInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Phase 3: CheckCrossFeature — cross-parameter feature constraints
    if (commonChecker_->CheckCrossFeature(mutableInfo) != ge::GRAPH_SUCCESS ||
        maskChecker_->CheckCrossFeature(mutableInfo) != ge::GRAPH_SUCCESS ||
        seqlensChecker_->CheckCrossFeature(mutableInfo) != ge::GRAPH_SUCCESS ||
        sparseCompressChecker_->CheckCrossFeature(mutableInfo) != ge::GRAPH_SUCCESS ||
        pagedAttentionChecker_->CheckCrossFeature(mutableInfo) != ge::GRAPH_SUCCESS ||
        sinksChecker_->CheckCrossFeature(mutableInfo) != ge::GRAPH_SUCCESS ||
        softmaxLseChecker_->CheckCrossFeature(mutableInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Phase 4: CheckMultiParaConsistency — multi-parameter shape/dtype matching
    if (commonChecker_->CheckMultiParaConsistency(mutableInfo) != ge::GRAPH_SUCCESS ||
        maskChecker_->CheckMultiParaConsistency(mutableInfo) != ge::GRAPH_SUCCESS ||
        seqlensChecker_->CheckMultiParaConsistency(mutableInfo) != ge::GRAPH_SUCCESS ||
        sparseCompressChecker_->CheckMultiParaConsistency(mutableInfo) != ge::GRAPH_SUCCESS ||
        pagedAttentionChecker_->CheckMultiParaConsistency(mutableInfo) != ge::GRAPH_SUCCESS ||
        sinksChecker_->CheckMultiParaConsistency(mutableInfo) != ge::GRAPH_SUCCESS ||
        softmaxLseChecker_->CheckMultiParaConsistency(mutableInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
