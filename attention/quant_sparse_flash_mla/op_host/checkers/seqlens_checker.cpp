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
 * \file seqlens_checker.cpp
 * \brief Implementation of SeqLensChecker for SeqLens参数组.
 */

#include "seqlens_checker.h"
#include "log/log.h"

namespace optiling {

using std::string;
using std::to_string;
using std::vector;

ge::graphStatus SeqLensChecker::CheckCuSeqLensQ(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.cuSeqLensQ;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, CU_SEQLENS_Q_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, CU_SEQLENS_Q_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, CU_SEQLENS_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_ONE};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, CU_SEQLENS_Q_NAME);
}

ge::graphStatus SeqLensChecker::CheckCuSeqLensOriKv(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.cuSeqLensOriKv;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, CU_SEQLENS_ORI_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, CU_SEQLENS_ORI_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, CU_SEQLENS_ORI_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_ONE};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, CU_SEQLENS_ORI_KV_NAME);
}

ge::graphStatus SeqLensChecker::CheckCuSeqLensCmpKv(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.cuSeqLensCmpKv;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, CU_SEQLENS_CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, CU_SEQLENS_CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, CU_SEQLENS_CMP_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_ONE};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, CU_SEQLENS_CMP_KV_NAME);
}

ge::graphStatus SeqLensChecker::CheckSequsedQ(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.seqUsedQ;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, SEQUSED_Q_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, SEQUSED_Q_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, SEQUSED_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_ONE};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, SEQUSED_Q_NAME);
}

ge::graphStatus SeqLensChecker::CheckSequsedOriKv(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.sequsedOriKv;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, SEQUSED_ORI_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, SEQUSED_ORI_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, SEQUSED_ORI_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_ONE};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, SEQUSED_ORI_KV_NAME);
}

ge::graphStatus SeqLensChecker::CheckSequsedCmpKv(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.sequsedCmpKv;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, SEQUSED_CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, SEQUSED_CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, SEQUSED_CMP_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_ONE};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, SEQUSED_CMP_KV_NAME);
}

ge::graphStatus SeqLensChecker::CheckSinglePara(const QSMLATilingInfo &info) const
{
    if (CheckCuSeqLensQ(info) != ge::GRAPH_SUCCESS || CheckCuSeqLensOriKv(info) != ge::GRAPH_SUCCESS ||
        CheckCuSeqLensCmpKv(info) != ge::GRAPH_SUCCESS || CheckSequsedQ(info) != ge::GRAPH_SUCCESS ||
        CheckSequsedOriKv(info) != ge::GRAPH_SUCCESS || CheckSequsedCmpKv(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLensChecker::CheckParaExistence(const QSMLATilingInfo &info) const
{
    if (info.qLayout == QSMLALayout::TND) {
        OP_CHECK_IF(info.opParamInfo.cuSeqLensQ.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_q",
                                                             "cu_seqlens_q must be provided when layout_q is TND"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(info.opParamInfo.cuSeqLensQ.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_q",
                                                             "cu_seqlens_q is not supported when layout_q is not TND"),
                    return ge::GRAPH_FAILED);
    }

    if (info.kvLayout == QSMLALayout::TND) {
        OP_CHECK_IF(info.opParamInfo.cuSeqLensOriKv.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "cu_seqlens_ori_kv", "cu_seqlens_ori_kv must be provided when layout_kv is TND"),
                    return ge::GRAPH_FAILED);
        if (info.opParamInfo.cmpKv.tensor != nullptr) {
            OP_CHECK_IF(info.opParamInfo.cuSeqLensCmpKv.tensor == nullptr,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            info.opName, "cu_seqlens_cmp_kv",
                            "cu_seqlens_cmp_kv must be provided when layout_kv is TND and cmp_kv is provided"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(
            info.opParamInfo.cuSeqLensOriKv.tensor != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_ori_kv",
                                                     "cu_seqlens_ori_kv is not supported when layout_kv is not TND"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            info.opParamInfo.cuSeqLensCmpKv.tensor != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_cmp_kv",
                                                     "cu_seqlens_cmp_kv is not supported when layout_kv is not TND"),
            return ge::GRAPH_FAILED);
    }

    if (info.kvLayout == QSMLALayout::PA_BBND && info.templateMode != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        info.templateMode != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        OP_CHECK_IF(info.opParamInfo.sequsedOriKv.tensor == nullptr && info.opParamInfo.oriTopkLength.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "seqused_ori_kv",
                        "seqused_ori_kv must be provided when layout_kv is PA_BBND (or ori_topk_length)"),
                    return ge::GRAPH_FAILED);
    }

    if (info.kvLayout == QSMLALayout::PA_BBND && info.opParamInfo.cmpKv.tensor != nullptr) {
        OP_CHECK_IF(info.opParamInfo.sequsedCmpKv.tensor == nullptr && info.opParamInfo.cmpTopkLength.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "seqused_cmp_kv",
                                                             "seqused_cmp_kv must be provided when layout_kv is "
                                                             "PA_BBND and cmp_kv is provided (or cmp_topk_length)"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLensChecker::CheckCrossFeature(const QSMLATilingInfo &info) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLensChecker::CheckMultiParaConsistency(const QSMLATilingInfo &info) const
{
    auto checkCuSeqLen = [&](const gert::Tensor *tensor, const string &name) -> ge::graphStatus {
        if (tensor == nullptr)
            return ge::GRAPH_SUCCESS;
        OP_CHECK_IF(tensor->GetShapeSize() != static_cast<int64_t>(info.bSize + 1),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        info.opName, name.c_str(), to_string(tensor->GetShapeSize()).c_str(),
                        (name + " shapeSize should be B+1(" + to_string(info.bSize + 1) + ")").c_str()),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    auto checkSeqused = [&](const gert::Tensor *tensor, const string &name) -> ge::graphStatus {
        if (tensor == nullptr)
            return ge::GRAPH_SUCCESS;
        OP_CHECK_IF(tensor->GetShapeSize() != static_cast<int64_t>(info.bSize),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        info.opName, name.c_str(), to_string(tensor->GetShapeSize()).c_str(),
                        (name + " shapeSize should be B(" + to_string(info.bSize) + ")").c_str()),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    if (checkCuSeqLen(info.opParamInfo.cuSeqLensQ.tensor, CU_SEQLENS_Q_NAME) != ge::GRAPH_SUCCESS ||
        checkCuSeqLen(info.opParamInfo.cuSeqLensOriKv.tensor, CU_SEQLENS_ORI_KV_NAME) != ge::GRAPH_SUCCESS ||
        checkCuSeqLen(info.opParamInfo.cuSeqLensCmpKv.tensor, CU_SEQLENS_CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        checkSeqused(info.opParamInfo.seqUsedQ.tensor, SEQUSED_Q_NAME) != ge::GRAPH_SUCCESS ||
        checkSeqused(info.opParamInfo.sequsedOriKv.tensor, SEQUSED_ORI_KV_NAME) != ge::GRAPH_SUCCESS ||
        checkSeqused(info.opParamInfo.sequsedCmpKv.tensor, SEQUSED_CMP_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
