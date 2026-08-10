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
 * \file sparse_compress_checker.cpp
 * \brief Implementation of SparseCompressChecker for 稀疏压缩参数组.
 */

#include "sparse_compress_checker.h"
#include "log/log.h"

namespace optiling {

using std::string;
using std::to_string;
using std::vector;

// =========================================================================
// CheckSinglePara
// =========================================================================
ge::graphStatus SparseCompressChecker::CheckCmpRatio(const QSMLATilingInfo &info) const
{
    OP_CHECK_IF(info.cmpRatio < CMP_RATIO_MIN || info.cmpRatio > CMP_RATIO_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "cmp_ratio", to_string(info.cmpRatio).c_str(),
                                                      ("cmp_ratio should be in range [" + to_string(CMP_RATIO_MIN) +
                                                       ", " + to_string(CMP_RATIO_MAX) + "]")
                                                          .c_str()),
                return ge::GRAPH_FAILED);
    if (info.templateMode == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(info.cmpRatio != CMP_RATIO_MIN,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "cmp_ratio", to_string(info.cmpRatio).c_str(),
                                                          "cmp_ratio must be 1 in SWA mode"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckCmpResidualKv(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.cmpResidualKv;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, CMP_RESIDUAL_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, CMP_RESIDUAL_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, CMP_RESIDUAL_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_ONE};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, CMP_RESIDUAL_KV_NAME);
}

ge::graphStatus SparseCompressChecker::CheckOriSparseIndices(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.oriSparseIndices;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, ORI_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, ORI_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, ORI_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = t.tensor->GetStorageShape();
    if (info.qLayout == QSMLALayout::BSND) {
        OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_FOUR,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(info.opName, "ori_sparse_indices",
                                                             to_string(shape.GetDimNum()).c_str(),
                                                             "When layout_q is BSND, ori_sparse_indices should be 4D"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_THREE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(info.opName, "ori_sparse_indices",
                                                             to_string(shape.GetDimNum()).c_str(),
                                                             "When layout_q is TND, ori_sparse_indices should be 3D"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckCmpSparseIndices(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.cmpSparseIndices;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, CMP_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, CMP_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, CMP_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = t.tensor->GetStorageShape();
    if (info.qLayout == QSMLALayout::BSND) {
        OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_FOUR,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(info.opName, "cmp_sparse_indices",
                                                             to_string(shape.GetDimNum()).c_str(),
                                                             "When layout_q is BSND, cmp_sparse_indices should be 4D"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_THREE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(info.opName, "cmp_sparse_indices",
                                                             to_string(shape.GetDimNum()).c_str(),
                                                             "When layout_q is TND, cmp_sparse_indices should be 3D"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckOriTopkLength(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.oriTopkLength;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, ORI_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, ORI_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, ORI_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims =
        (info.qLayout == QSMLALayout::TND) ? vector<size_t>{DIM_NUM_TWO} : vector<size_t>{DIM_NUM_THREE};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, ORI_TOPK_LENGTH_NAME);
}

ge::graphStatus SparseCompressChecker::CheckCmpTopkLength(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.cmpTopkLength;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, CMP_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, CMP_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, CMP_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims =
        (info.qLayout == QSMLALayout::TND) ? vector<size_t>{DIM_NUM_TWO} : vector<size_t>{DIM_NUM_THREE};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, CMP_TOPK_LENGTH_NAME);
}

ge::graphStatus SparseCompressChecker::CheckTopkValueMode(const QSMLATilingInfo &info) const
{
    if (info.templateMode == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(info.topkValueMode != TOPK_VALUE_MODE_DEFAULT,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "topk_value_mode",
                                                          to_string(info.topkValueMode).c_str(),
                                                          "topk_value_mode must be 1 in SWA mode"),
                    return ge::GRAPH_FAILED);
    } else {
        const vector<int64_t> supportList = {1, 2};
        if (CheckValueSupport(info.topkValueMode, supportList, "topk_value_mode") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckSinglePara(const QSMLATilingInfo &info) const
{
    if (CheckCmpRatio(info) != ge::GRAPH_SUCCESS || CheckCmpResidualKv(info) != ge::GRAPH_SUCCESS ||
        CheckOriSparseIndices(info) != ge::GRAPH_SUCCESS || CheckCmpSparseIndices(info) != ge::GRAPH_SUCCESS ||
        CheckOriTopkLength(info) != ge::GRAPH_SUCCESS || CheckCmpTopkLength(info) != ge::GRAPH_SUCCESS ||
        CheckTopkValueMode(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =========================================================================
// CheckParaExistence
// =========================================================================
ge::graphStatus SparseCompressChecker::CheckCmpResidualKvExistence(const QSMLATilingInfo &info) const
{
    if (info.opParamInfo.cmpKv.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (info.cmpMaskMode == MASK_MODE_CAUSAL && info.cmpRatio != CMP_RATIO_MIN) {
        OP_CHECK_IF(
            info.opParamInfo.cmpResidualKv.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                info.opName, "cmp_residual_kv", "cmp_residual_kv is required when cmp_mask_mode=3 and cmp_ratio!=1"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckOriTopkLengthExistence(const QSMLATilingInfo &info) const
{
    const auto &p = info.opParamInfo;
    if (info.oriMaskMode == MASK_MODE_NO_MASK && p.oriSparseIndices.tensor != nullptr) {
        OP_CHECK_IF(p.oriTopkLength.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "ori_topk_length",
                        "ori_topk_length must be provided when ori_mask_mode=0 and ori_sparse_indices is provided"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(p.oriTopkLength.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "ori_topk_length",
                        "ori_topk_length is not supported when ori_mask_mode!=0 or ori_sparse_indices is None"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckCmpTopkLengthExistence(const QSMLATilingInfo &info) const
{
    const auto &p = info.opParamInfo;
    if (p.cmpKv.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (info.cmpMaskMode == MASK_MODE_NO_MASK && p.cmpSparseIndices.tensor != nullptr) {
        OP_CHECK_IF(p.cmpTopkLength.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "cmp_topk_length",
                        "cmp_topk_length must be provided when cmp_mask_mode=0 and cmp_sparse_indices is provided"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(p.cmpTopkLength.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "cmp_topk_length",
                        "cmp_topk_length is not supported when cmp_mask_mode!=0 or cmp_sparse_indices is None"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckParaExistence(const QSMLATilingInfo &info) const
{
    if (CheckCmpResidualKvExistence(info) != ge::GRAPH_SUCCESS ||
        CheckOriTopkLengthExistence(info) != ge::GRAPH_SUCCESS ||
        CheckCmpTopkLengthExistence(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (info.opParamInfo.cmpKv.tensor == nullptr && info.opParamInfo.cmpSparseIndices.tensor != nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cmp_sparse_indices",
                                                 "cmp_sparse_indices is not supported when cmp_kv is None");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =========================================================================
// CheckCrossFeature
// =========================================================================
ge::graphStatus SparseCompressChecker::CheckCrossFeature(const QSMLATilingInfo &info) const
{
    return ge::GRAPH_SUCCESS;
}

// =========================================================================
// CheckMultiParaConsistency
// =========================================================================
ge::graphStatus SparseCompressChecker::CheckSparseIndicesShapeMatchQ(const QSMLATilingInfo &info) const
{
    auto checkOne = [&](const gert::Tensor *tensor, const string &name) -> ge::graphStatus {
        if (tensor == nullptr)
            return ge::GRAPH_SUCCESS;
        const auto &shape = tensor->GetStorageShape();
        if (info.qLayout == QSMLALayout::BSND) {
            OP_CHECK_IF(shape.GetDim(DIM_0) != static_cast<int64_t>(info.bSize),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            info.opName, name.c_str(), GetShapeStr(shape).c_str(),
                            (name + " B should equal q B(" + to_string(info.bSize) + ")").c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(shape.GetDim(DIM_1) != static_cast<int64_t>(info.s1Size),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            info.opName, name.c_str(), GetShapeStr(shape).c_str(),
                            (name + " S1 should equal q S1(" + to_string(info.s1Size) + ")").c_str()),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(shape.GetDim(DIM_0) != static_cast<int64_t>(info.qTSize),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            info.opName, name.c_str(), GetShapeStr(shape).c_str(),
                            (name + " T should equal q T(" + to_string(info.qTSize) + ")").c_str()),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    };
    if (checkOne(info.opParamInfo.oriSparseIndices.tensor, ORI_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS ||
        checkOne(info.opParamInfo.cmpSparseIndices.tensor, CMP_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckTopkLengthShapeMatchQ(const QSMLATilingInfo &info) const
{
    auto checkOne = [&](const gert::Tensor *tensor, const string &name) -> ge::graphStatus {
        if (tensor == nullptr)
            return ge::GRAPH_SUCCESS;
        const auto &shape = tensor->GetStorageShape();
        if (info.qLayout == QSMLALayout::BSND) {
            OP_CHECK_IF(shape.GetDim(DIM_0) != static_cast<int64_t>(info.bSize),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            info.opName, name.c_str(), GetShapeStr(shape).c_str(),
                            (name + " B should equal q B(" + to_string(info.bSize) + ")").c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(shape.GetDim(DIM_1) != static_cast<int64_t>(info.s1Size),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            info.opName, name.c_str(), GetShapeStr(shape).c_str(),
                            (name + " S1 should equal q S1(" + to_string(info.s1Size) + ")").c_str()),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(shape.GetDim(DIM_0) != static_cast<int64_t>(info.qTSize),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            info.opName, name.c_str(), GetShapeStr(shape).c_str(),
                            (name + " T should equal q T(" + to_string(info.qTSize) + ")").c_str()),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    };
    if (checkOne(info.opParamInfo.oriTopkLength.tensor, ORI_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS ||
        checkOne(info.opParamInfo.cmpTopkLength.tensor, CMP_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckN2Consistency(const QSMLATilingInfo &info) const
{
    auto checkN2 = [&](const gert::Tensor *tensor, const string &name) -> ge::graphStatus {
        if (tensor == nullptr)
            return ge::GRAPH_SUCCESS;
        const auto &shape = tensor->GetStorageShape();
        int64_t n2 = (info.qLayout == QSMLALayout::BSND) ? shape.GetDim(DIM_2) : shape.GetDim(DIM_1);
        OP_CHECK_IF(static_cast<uint32_t>(n2) != info.n2Size,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        info.opName, (name + " N").c_str(), to_string(n2).c_str(),
                        (name + " N should equal kv_head_num(" + to_string(info.n2Size) + ")").c_str()),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };
    if (checkN2(info.opParamInfo.oriSparseIndices.tensor, ORI_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS ||
        checkN2(info.opParamInfo.cmpSparseIndices.tensor, CMP_SPARSE_INDICES_NAME) != ge::GRAPH_SUCCESS ||
        checkN2(info.opParamInfo.oriTopkLength.tensor, ORI_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS ||
        checkN2(info.opParamInfo.cmpTopkLength.tensor, CMP_TOPK_LENGTH_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (info.opParamInfo.cmpResidualKv.tensor != nullptr) {
        OP_CHECK_IF(info.opParamInfo.cmpResidualKv.tensor->GetShapeSize() != static_cast<int64_t>(info.bSize),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        info.opName, "cmp_residual_kv",
                        to_string(info.opParamInfo.cmpResidualKv.tensor->GetShapeSize()).c_str(),
                        ("cmp_residual_kv shapeSize should be B(" + to_string(info.bSize) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressChecker::CheckMultiParaConsistency(const QSMLATilingInfo &info) const
{
    if (CheckSparseIndicesShapeMatchQ(info) != ge::GRAPH_SUCCESS ||
        CheckTopkLengthShapeMatchQ(info) != ge::GRAPH_SUCCESS || CheckN2Consistency(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
