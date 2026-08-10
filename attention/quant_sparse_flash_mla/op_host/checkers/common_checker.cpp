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
 * \file common_checker.cpp
 * \brief Implementation of CommonChecker for 公共参数组.
 */

#include "common_checker.h"
#include "log/log.h"

namespace optiling {

using std::string;
using std::to_string;
using std::vector;

// =========================================================================
// CheckSinglePara: dtype / shape / format per tensor, value per attribute
// =========================================================================

ge::graphStatus CommonChecker::CheckQuery(const QSMLATilingInfo &info) const
{
    const auto &q = info.opParamInfo.q;
    OP_CHECK_IF(q.desc == nullptr || q.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "q", "q desc or shape is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(q.shape->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(info.opName, "q", "0", "Any dim of q cannot be 0"),
                return ge::GRAPH_FAILED);

    if (CheckDtypeSupport(q.desc, QUERY_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(q.desc, QUERY_NAME) != ge::GRAPH_SUCCESS ||
        CheckDimNumInLayoutSupport(info.qLayout, q.shape, QUERY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckOriKv(const QSMLATilingInfo &info) const
{
    const auto &oriKv = info.opParamInfo.oriKv;
    OP_CHECK_IF(oriKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "ori_kv", "ori_kv cannot be nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(oriKv.tensor->GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(info.opName, "ori_kv", "0", "Any dim of ori_kv cannot be 0"),
                return ge::GRAPH_FAILED);

    if (CheckDtypeSupport(oriKv.desc, ORI_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(oriKv.desc, ORI_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDimNumInLayoutSupport(info.kvLayout, &oriKv.tensor->GetShape(), ORI_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t kvD = GetAxisNum(oriKv.tensor->GetStorageShape(), QSMLAAxis::D, info.kvLayout);
    OP_CHECK_IF(kvD != HEAD_DIM_512,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    info.opName, "ori_kv", GetShapeStr(oriKv.tensor->GetStorageShape()).c_str(),
                    ("Dimension of ori_kv only support 512, but got " + to_string(kvD)).c_str()),
                return ge::GRAPH_FAILED);

    int64_t kvN = GetAxisNum(oriKv.tensor->GetStorageShape(), QSMLAAxis::N, info.kvLayout);
    OP_CHECK_IF(static_cast<uint32_t>(kvN) != info.n2Size,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    info.opName, "ori_kv N", to_string(kvN).c_str(),
                    ("N of ori_kv should be equal to kv_head_num " + to_string(info.n2Size)).c_str()),
                return ge::GRAPH_FAILED);

    if (info.kvLayout == QSMLALayout::PA_BBND) {
        OP_CHECK_IF(info.oriBlockSize <= 0 || info.oriBlockSize > static_cast<int32_t>(MAX_BLOCK_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        info.opName, "ori_block_size", to_string(info.oriBlockSize).c_str(),
                        ("ori_block_size should be in range (0, " + to_string(MAX_BLOCK_SIZE) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckCmpKv(const QSMLATilingInfo &info) const
{
    const auto &cmpKv = info.opParamInfo.cmpKv;
    if (cmpKv.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(cmpKv.tensor->GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(info.opName, "cmp_kv", "0", "Any dim of cmp_kv cannot be 0"),
                return ge::GRAPH_FAILED);

    if (CheckDtypeSupport(cmpKv.desc, CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(cmpKv.desc, CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDimNumInLayoutSupport(info.kvLayout, &cmpKv.tensor->GetShape(), CMP_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t kvD = GetAxisNum(cmpKv.tensor->GetStorageShape(), QSMLAAxis::D, info.kvLayout);
    OP_CHECK_IF(kvD != HEAD_DIM_512,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    info.opName, "cmp_kv", GetShapeStr(cmpKv.tensor->GetStorageShape()).c_str(),
                    ("Dimension of cmp_kv only support 512, but got " + to_string(kvD)).c_str()),
                return ge::GRAPH_FAILED);

    int64_t kvN = GetAxisNum(cmpKv.tensor->GetStorageShape(), QSMLAAxis::N, info.kvLayout);
    OP_CHECK_IF(static_cast<uint32_t>(kvN) != info.n2Size,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    info.opName, "cmp_kv N", to_string(kvN).c_str(),
                    ("N of cmp_kv should be equal to kv_head_num " + to_string(info.n2Size)).c_str()),
                return ge::GRAPH_FAILED);

    if (info.kvLayout == QSMLALayout::PA_BBND) {
        OP_CHECK_IF(info.cmpBlockSize <= 0 || info.cmpBlockSize > static_cast<int32_t>(MAX_BLOCK_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        info.opName, "cmp_block_size", to_string(info.cmpBlockSize).c_str(),
                        ("cmp_block_size should be in range (0, " + to_string(MAX_BLOCK_SIZE) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDescale(const QSMLATilingInfo &info) const
{
    const auto &qDescale = info.opParamInfo.qDescale;
    const auto &oriKvDescale = info.opParamInfo.oriKvDescale;
    const auto &cmpKvDescale = info.opParamInfo.cmpKvDescale;

    auto checkOne = [&](const gert::CompileTimeTensorDesc *desc, const gert::Tensor *tensor,
                        const string &name) -> ge::graphStatus {
        if (tensor == nullptr)
            return ge::GRAPH_SUCCESS;
        if (CheckDtypeSupport(desc, name) != ge::GRAPH_SUCCESS || CheckFormatSupport(desc, name) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        const vector<size_t> expectDims = {DIM_NUM_ONE};
        return CheckDimNumSupport(&tensor->GetShape(), expectDims, name);
    };
    if (checkOne(qDescale.desc, qDescale.tensor, Q_DESCALE_NAME) != ge::GRAPH_SUCCESS ||
        checkOne(oriKvDescale.desc, oriKvDescale.tensor, ORI_KV_DESCALE_NAME) != ge::GRAPH_SUCCESS ||
        checkOne(cmpKvDescale.desc, cmpKvDescale.tensor, CMP_KV_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMetadata(const QSMLATilingInfo &info) const
{
    const auto &meta = info.opParamInfo.metadata;
    OP_CHECK_IF(
        meta.tensor == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "metadata", "metadata is required but got nullptr"),
        return ge::GRAPH_FAILED);

    if (CheckDtypeSupport(meta.desc, METADATA_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(meta.desc, METADATA_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_ONE};
    if (CheckDimNumSupport(&meta.tensor->GetShape(), expectDims, METADATA_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(meta.tensor->GetShapeSize() != METADATA_SIZE,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    info.opName, "metadata", to_string(meta.tensor->GetShapeSize()).c_str(),
                    ("metadata shapeSize should be " + to_string(METADATA_SIZE)).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAttentionOut(const QSMLATilingInfo &info) const
{
    const auto &out = info.opParamInfo.attnOut;
    OP_CHECK_IF(out.desc == nullptr || out.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attention_out",
                                                         "attention_out desc or shape is nullptr"),
                return ge::GRAPH_FAILED);

    if (CheckDtypeSupport(out.desc, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(out.desc, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS ||
        CheckDimNumInLayoutSupport(info.outLayout, out.shape, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckQuantModeAttr(const QSMLATilingInfo &info) const
{
    OP_CHECK_IF(info.quantMode != QUANT_MODE_PER_TENSOR_HIFLOAT8,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "quant_mode", to_string(info.quantMode).c_str(),
                                                      "quant_mode only support 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckLayoutAttr(const QSMLATilingInfo &info) const
{
    const vector<string> supportList = {LAYOUT_BSND, LAYOUT_TND};
    string layoutQ(info.opParamInfo.layoutQ);
    OP_CHECK_IF(std::find(supportList.begin(), supportList.end(), layoutQ) == supportList.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "layout_q", layoutQ.c_str(),
                                                      "layout_q only support BSND or TND"),
                return ge::GRAPH_FAILED);

    string layoutKv(info.opParamInfo.layoutKv);
    const vector<string> supportListKv = {LAYOUT_BSND, LAYOUT_TND, LAYOUT_PA_BBND};
    OP_CHECK_IF(std::find(supportListKv.begin(), supportListKv.end(), layoutKv) == supportListKv.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "layout_kv", layoutKv.c_str(),
                                                      "layout_kv only support BSND, TND or PA_BBND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAxis(const QSMLATilingInfo &info) const
{
    OP_CHECK_IF(info.bSize <= 0 || info.bSize > B_LIMIT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    info.opName, "batch_size", to_string(info.bSize).c_str(),
                    ("batch_size should be in range (0, " + to_string(B_LIMIT) + "]").c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(info.n1Size < Q_HEAD_NUM_MIN || info.n1Size > Q_HEAD_NUM_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "q_head_num", to_string(info.n1Size).c_str(),
                                                      ("q_head_num should be in range [" + to_string(Q_HEAD_NUM_MIN) +
                                                       ", " + to_string(Q_HEAD_NUM_MAX) + "]")
                                                          .c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(info.n2Size != KV_HEAD_NUM_LIMIT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "kv_head_num", to_string(info.n2Size).c_str(),
                                                      "kv_head_num only support 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        info.n1Size % info.n2Size != 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "q_head_num and kv_head_num",
                                               (to_string(info.n1Size) + " and " + to_string(info.n2Size)).c_str(),
                                               "q_head_num must be divisible by kv_head_num"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(info.qkHeadDim != HEAD_DIM_512,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "head_dim", to_string(info.qkHeadDim).c_str(),
                                                      "head_dim only support 512"),
                return ge::GRAPH_FAILED);

    if (info.qLayout == QSMLALayout::TND) {
        OP_CHECK_IF(info.qTSize <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "T of query", to_string(info.qTSize).c_str(),
                                                          "T of query should be > 0 when layout is TND"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSinglePara(const QSMLATilingInfo &info) const
{
    if (CheckQuery(info) != ge::GRAPH_SUCCESS || CheckOriKv(info) != ge::GRAPH_SUCCESS ||
        CheckCmpKv(info) != ge::GRAPH_SUCCESS || CheckDescale(info) != ge::GRAPH_SUCCESS ||
        CheckMetadata(info) != ge::GRAPH_SUCCESS || CheckAttentionOut(info) != ge::GRAPH_SUCCESS ||
        CheckQuantModeAttr(info) != ge::GRAPH_SUCCESS || CheckLayoutAttr(info) != ge::GRAPH_SUCCESS ||
        CheckAxis(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =========================================================================
// CheckParaExistence
// =========================================================================
ge::graphStatus CommonChecker::CheckRequiredExistence(const QSMLATilingInfo &info) const
{
    const auto &p = info.opParamInfo;
    OP_CHECK_IF(p.q.desc == nullptr || p.q.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "q", "q is required"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(p.oriKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "ori_kv", "ori_kv is required"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(p.attnOut.desc == nullptr || p.attnOut.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attention_out", "attention_out is required"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(p.metadata.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "metadata", "metadata is required"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(p.qDescale.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "q_descale", "q_descale is required"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(p.oriKvDescale.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "ori_kv_descale", "ori_kv_descale is required"),
                return ge::GRAPH_FAILED);
    if (p.cmpKv.tensor != nullptr) {
        OP_CHECK_IF(p.cmpKvDescale.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cmp_kv_descale",
                                                             "cmp_kv_descale is required when cmp_kv is provided"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(p.cmpKvDescale.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "cmp_kv_descale", "cmp_kv_descale is not supported when cmp_kv is not provided"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckParaExistence(const QSMLATilingInfo &info) const
{
    return CheckRequiredExistence(info);
}

// =========================================================================
// CheckCrossFeature
// =========================================================================
ge::graphStatus CommonChecker::CheckLayoutQKvConsistency(const QSMLATilingInfo &info) const
{
    if (info.kvLayout != QSMLALayout::PA_BBND) {
        OP_CHECK_IF(info.qLayout != info.kvLayout,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        info.opName, "layout_q and layout_kv",
                        (LayoutToSerialString(info.qLayout) + " and " + LayoutToSerialString(info.kvLayout)).c_str(),
                        "When kv layout is not PA_BBND, q layout must equal kv layout"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckQuantModeDtype(const QSMLATilingInfo &info) const
{
    OP_CHECK_IF(info.qType != ge::DT_HIFLOAT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(info.opName, "q", DataTypeToSerialString(info.qType).c_str(),
                                                      "When quant_mode=1, q dtype must be HIFLOAT8"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(info.oriKvType != ge::DT_HIFLOAT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(info.opName, "ori_kv",
                                                      DataTypeToSerialString(info.oriKvType).c_str(),
                                                      "When quant_mode=1, ori_kv dtype must be HIFLOAT8"),
                return ge::GRAPH_FAILED);

    if (info.opParamInfo.cmpKv.desc != nullptr) {
        OP_CHECK_IF(info.cmpKvType != ge::DT_HIFLOAT8,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(info.opName, "cmp_kv",
                                                          DataTypeToSerialString(info.cmpKvType).c_str(),
                                                          "When quant_mode=1, cmp_kv dtype must be HIFLOAT8"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckOriCmpKvDtypeConsistency(const QSMLATilingInfo &info) const
{
    if (info.opParamInfo.cmpKv.tensor != nullptr) {
        OP_CHECK_IF(
            info.cmpKvType != info.oriKvType,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                info.opName, "ori_kv and cmp_kv",
                (DataTypeToSerialString(info.oriKvType) + " and " + DataTypeToSerialString(info.cmpKvType)).c_str(),
                "ori_kv dtype and cmp_kv dtype must be the same"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKvContiguous(const QSMLATilingInfo &info) const
{
    auto validateContiguous = [&](const std::vector<int64_t> &strides, const gert::Shape &shape, QSMLALayout kvLayout,
                                  const string &name) -> ge::graphStatus {
        if (strides.empty())
            return ge::GRAPH_SUCCESS;
        std::vector<int64_t> expected;
        if (kvLayout == QSMLALayout::BSND) {
            expected = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3), shape.GetDim(2) * shape.GetDim(3),
                        shape.GetDim(3), 1};
        } else if (kvLayout == QSMLALayout::TND) {
            expected = {shape.GetDim(1) * shape.GetDim(2), shape.GetDim(2), 1};
        } else if (kvLayout == QSMLALayout::PA_BBND) {
            expected = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3), shape.GetDim(2) * shape.GetDim(3),
                        shape.GetDim(3), 1};
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (kvLayout == QSMLALayout::PA_BBND && i == 0)
                continue;
            OP_CHECK_IF(
                strides[i] != expected[i],
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, name.c_str(),
                                                         (name + " stride[" + to_string(i) + "] expected " +
                                                          to_string(expected[i]) + ", got " + to_string(strides[i]))
                                                             .c_str()),
                return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    };
    if (validateContiguous(info.oriKvStrides, info.oriKvStorageShape, info.kvLayout, ORI_KV_NAME) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (validateContiguous(info.cmpKvStrides, info.cmpKvStorageShape, info.kvLayout, CMP_KV_NAME) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckCrossFeature(const QSMLATilingInfo &info) const
{
    if (CheckLayoutQKvConsistency(info) != ge::GRAPH_SUCCESS || CheckQuantModeDtype(info) != ge::GRAPH_SUCCESS ||
        CheckOriCmpKvDtypeConsistency(info) != ge::GRAPH_SUCCESS || CheckKvContiguous(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =========================================================================
// CheckMultiParaConsistency
// =========================================================================
ge::graphStatus CommonChecker::CheckQueryOutShapeConsistency(const QSMLATilingInfo &info) const
{
    const auto &qShape = info.opParamInfo.q.shape->GetStorageShape();
    const auto &outShape = info.opParamInfo.attnOut.shape->GetStorageShape();
    OP_CHECK_IF(qShape.GetDimNum() != outShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(info.opName, "q and attention_out",
                                                         to_string(qShape.GetDimNum()).c_str(),
                                                         "q dim num must equal attention_out dim num"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < qShape.GetDimNum(); ++i) {
        OP_CHECK_IF(qShape.GetDim(i) != outShape.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        info.opName, "q and attention_out",
                        (GetShapeStr(qShape) + " and " + GetShapeStr(outShape)).c_str(),
                        ("q axis " + to_string(i) + " must equal attention_out axis " + to_string(i)).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDescaleShapeConsistency(const QSMLATilingInfo &info) const
{
    auto checkOne = [&](const gert::Tensor *tensor, const string &name) -> ge::graphStatus {
        if (tensor == nullptr)
            return ge::GRAPH_SUCCESS;
        OP_CHECK_IF(tensor->GetShapeSize() != 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        info.opName, name.c_str(), to_string(tensor->GetShapeSize()).c_str(),
                        (name + " shapeSize should be 1 when quant_mode=1").c_str()),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };
    if (checkOne(info.opParamInfo.qDescale.tensor, Q_DESCALE_NAME) != ge::GRAPH_SUCCESS ||
        checkOne(info.opParamInfo.oriKvDescale.tensor, ORI_KV_DESCALE_NAME) != ge::GRAPH_SUCCESS ||
        checkOne(info.opParamInfo.cmpKvDescale.tensor, CMP_KV_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiParaConsistency(const QSMLATilingInfo &info) const
{
    if (CheckQueryOutShapeConsistency(info) != ge::GRAPH_SUCCESS ||
        CheckDescaleShapeConsistency(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
