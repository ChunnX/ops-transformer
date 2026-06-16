/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_CHECKER_H
#define OP_API_INC_QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_CHECKER_H

#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_log.h"
#include "log/log.h"

#define OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(aclnnName, tensor, supportList, retExpr)                               \
    do {                                                                                                              \
        if (!QkvRmsNormRopeCacheWithKScaleCheck::CheckType(tensor->GetDataType(), supportList)) {                     \
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(                                                                    \
                aclnnName, #tensor, op::ToString(tensor->GetDataType()).GetString(),                                  \
                "The dtype of " + std::string(#tensor) + " must be one of " + op::ToString(supportList).GetString()); \
            retExpr;                                                                                                  \
        }                                                                                                             \
    } while (0)

namespace QkvRmsNormRopeCacheWithKScaleCheck {

inline bool CheckType(op::DataType dtype, const std::initializer_list<op::DataType> &supportList)
{
    for (const auto &supported : supportList) {
        if (dtype == supported) {
            return true;
        }
    }
    return false;
}

struct QkvRmsNormRopeCacheWithKScaleParams {
    // Required inputs
    const aclTensor *qkv{nullptr};
    const aclTensor *qGamma{nullptr};
    const aclTensor *kGamma{nullptr};
    const aclTensor *cosSin{nullptr};
    const aclTensor *slotMapping{nullptr};
    aclTensor *kCache{nullptr};
    aclTensor *vCache{nullptr};
    aclTensor *kScaleCache{nullptr};
    const aclTensor *cuSeqLen{nullptr};
    // Optional inputs (required in first phase, but declared as optional in def)
    const aclTensor *optionalQRotation{nullptr};
    const aclTensor *optionalKRotation{nullptr};
    const aclTensor *optionalVScale{nullptr};
    // Attributes
    const aclIntArray *headNums{nullptr};
    float epsilon{1e-6f};
    // Outputs
    aclTensor *qOut{nullptr};
    aclTensor *qScale{nullptr};
    // Workspace/executor
    uint64_t *workspaceSize{nullptr};
    aclOpExecutor **executor{nullptr};
};

class QkvRmsNormRopeCacheWithKScaleChecker {
public:
    // ACLNN operator name for log messages
    static constexpr const char *ACLNN_NAME = "aclnnQkvRmsNormRopeCacheWithKScale";

    // Constants
    static constexpr uint64_t HEAD_NUMS_SIZE = 3;
    static constexpr int64_t HEAD_DIM_FAST_PATH = 128;

    // Dtype support lists
    inline static const std::initializer_list<op::DataType> QKV_TYPE_LIST = {op::DataType::DT_BF16};
    inline static const std::initializer_list<op::DataType> GAMMA_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::initializer_list<op::DataType> COS_SIN_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::initializer_list<op::DataType> SLOT_MAPPING_TYPE_LIST = {op::DataType::DT_INT32};
    inline static const std::initializer_list<op::DataType> CACHE_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
    inline static const std::initializer_list<op::DataType> K_SCALE_CACHE_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::initializer_list<op::DataType> CU_SEQ_LEN_TYPE_LIST = {op::DataType::DT_INT32};
    inline static const std::initializer_list<op::DataType> ROTATION_TYPE_LIST = {op::DataType::DT_BF16};
    inline static const std::initializer_list<op::DataType> V_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::initializer_list<op::DataType> Q_OUT_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
    inline static const std::initializer_list<op::DataType> Q_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT};

    // Main entry point
    aclnnStatus CheckParams(const QkvRmsNormRopeCacheWithKScaleParams &p)
    {
        params_ = p;
        aclnnStatus ret = CheckNotNull();
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckEmptyTensor();
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeValid();
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckShape();
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckFormat();
        if (ret != ACLNN_SUCCESS) { return ret; }
        return ACLNN_SUCCESS;
    }

private:
    QkvRmsNormRopeCacheWithKScaleParams params_;

    // Helper: shape to string
    static std::string ShapeToString(const aclTensor *tensor)
    {
        std::string result;
        const auto &shape = tensor->GetViewShape();
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            if (i != 0) { result += ", "; }
            result += std::to_string(shape.GetDim(i));
        }
        return result;
    }

    // Helper: check if tensor has expected shape
    static bool IsShape(const aclTensor *tensor, const std::vector<int64_t> &expectedShape)
    {
        const auto &shape = tensor->GetViewShape();
        if (shape.GetDimNum() != expectedShape.size()) {
            return false;
        }
        for (size_t i = 0; i < expectedShape.size(); ++i) {
            if (shape.GetDim(i) != expectedShape[i]) {
                return false;
            }
        }
        return true;
    }

    // 1. Null pointer check
    aclnnStatus CheckNotNull()
    {
        if (params_.qkv == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qkv", "nullptr", "qkv must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.qGamma == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qGamma", "nullptr", "qGamma must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.kGamma == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "kGamma", "nullptr", "kGamma must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.cosSin == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "cosSin", "nullptr", "cosSin must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.slotMapping == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "slotMapping", "nullptr",
                "slotMapping must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.kCache == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "kCache", "nullptr", "kCache must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.vCache == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "vCache", "nullptr", "vCache must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.kScaleCache == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "kScaleCache", "nullptr",
                "kScaleCache must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.cuSeqLen == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "cuSeqLen", "nullptr",
                "cuSeqLen must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.optionalQRotation == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "optionalQRotation", "nullptr",
                "optionalQRotation must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.optionalKRotation == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "optionalKRotation", "nullptr",
                "optionalKRotation must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.optionalVScale == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "optionalVScale", "nullptr",
                "optionalVScale must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.headNums == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "headNums", "nullptr",
                "headNums must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.qOut == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qOut", "nullptr", "qOut must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.qScale == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qScale", "nullptr", "qScale must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.workspaceSize == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "workspaceSize", "nullptr",
                "workspaceSize must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.executor == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "executor", "nullptr",
                "executor must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        return ACLNN_SUCCESS;
    }

    // 2. Dtype check
    aclnnStatus CheckDtypeValid()
    {
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.qkv, QKV_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.qGamma, GAMMA_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.kGamma, GAMMA_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.cosSin, COS_SIN_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.slotMapping, SLOT_MAPPING_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.kCache, CACHE_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.vCache, CACHE_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.kScaleCache, K_SCALE_CACHE_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.cuSeqLen, CU_SEQ_LEN_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.optionalQRotation, ROTATION_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.optionalKRotation, ROTATION_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.optionalVScale, V_SCALE_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.qOut, Q_OUT_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, params_.qScale, Q_SCALE_TYPE_LIST,
            return ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }

    // 3. Shape check
    aclnnStatus ParseHeadNums(int64_t &nq, int64_t &nk, int64_t &nv)
    {
        if (params_.headNums->Size() != static_cast<int64_t>(HEAD_NUMS_SIZE)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "headNums",
                std::to_string(params_.headNums->Size()), "headNums must contain exactly 3 elements");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t *headNumsData = params_.headNums->GetData();
        if (headNumsData == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "headNums", "nullptr", "headNums data is null");
            return ACLNN_ERR_PARAM_INVALID;
        }

        nq = headNumsData[0];
        nk = headNumsData[1];
        nv = headNumsData[2];
        if (nq <= 0 || nk <= 0 || nv <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "headNums",
                std::to_string(nq) + ", " + std::to_string(nk) + ", " + std::to_string(nv),
                "headNums values must be positive");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (nv != nk) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "Nv, Nk",
                std::to_string(nv) + ", " + std::to_string(nk),
                "First phase only supports Nv == Nk");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckShape()
    {
        int64_t nq = 0;
        int64_t nk = 0;
        int64_t nv = 0;
        CHECK_RET(ParseHeadNums(nq, nk, nv) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

        // qkv shape check: [Nq+Nk+Nv, T, D]
        const auto &qkvShape = params_.qkv->GetViewShape();
        uint64_t qkvDimNum = qkvShape.GetDimNum();
        if (unlikely(qkvDimNum != 3)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "qkv",
                std::to_string(qkvDimNum), "qkv must be a 3D tensor [Nq+Nk+Nv, T, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t hqkv = qkvShape.GetDim(0);
        const int64_t t = qkvShape.GetDim(1);
        const int64_t d = qkvShape.GetDim(2);
        if (unlikely(hqkv != nq + nk + nv)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "qkv and headNums",
                "qkv.shape[0]=" + std::to_string(hqkv) + ", Nq+Nk+Nv=" + std::to_string(nq + nk + nv),
                "qkv shape[0] must equal Nq+Nk+Nv from headNums");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(d != HEAD_DIM_FAST_PATH)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "D",
                std::to_string(d), "First phase only supports D=128");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(d % 2 != 0)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "D",
                std::to_string(d), "D must be even for RoPE");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // gamma shape check: [D]
        if (!IsShape(params_.qGamma, {d}) || !IsShape(params_.kGamma, {d})) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "qGamma, kGamma",
                ShapeToString(params_.qGamma) + ", " + ShapeToString(params_.kGamma),
                "qGamma and kGamma must have shape [D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // cosSin shape check: [*, D]
        const auto &cosSinShape = params_.cosSin->GetViewShape();
        if (unlikely(cosSinShape.GetDimNum() != 2 || cosSinShape.GetDim(1) != d)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "cosSin",
                ShapeToString(params_.cosSin), "cosSin must have shape [MaxSeqLen, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // slotMapping shape check: [T]
        if (!IsShape(params_.slotMapping, {t})) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "slotMapping",
                ShapeToString(params_.slotMapping), "slotMapping must have shape [T]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // cuSeqLen shape check: 1D, len >= 2
        const auto &cuSeqLenShape = params_.cuSeqLen->GetViewShape();
        if (unlikely(cuSeqLenShape.GetDimNum() != 1 || cuSeqLenShape.GetDim(0) < 2)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "cuSeqLen",
                ShapeToString(params_.cuSeqLen), "cuSeqLen must be a 1D tensor with length >= 2");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // rotation shape check: [D, D]
        if (!IsShape(params_.optionalQRotation, {d, d}) ||
            !IsShape(params_.optionalKRotation, {d, d})) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "optionalQRotation, optionalKRotation",
                ShapeToString(params_.optionalQRotation) + ", " + ShapeToString(params_.optionalKRotation),
                "optionalQRotation and optionalKRotation must have shape [D, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // vScale shape check: [Nv]
        if (!IsShape(params_.optionalVScale, {nv})) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "optionalVScale",
                ShapeToString(params_.optionalVScale), "optionalVScale must have shape [Nv]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // kCache shape check: [BlockNum, Nk, BlockSize, D]
        const auto &kCacheShape = params_.kCache->GetViewShape();
        if (unlikely(kCacheShape.GetDimNum() != 4 || kCacheShape.GetDim(1) != nk ||
            kCacheShape.GetDim(3) != d)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "kCache",
                ShapeToString(params_.kCache),
                "kCache must have shape [BlockNum, Nk, BlockSize, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // vCache shape check: [BlockNum, Nv, BlockSize, D]
        const auto &vCacheShape = params_.vCache->GetViewShape();
        if (unlikely(vCacheShape.GetDimNum() != 4 || vCacheShape.GetDim(1) != nv ||
            vCacheShape.GetDim(3) != d)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "vCache",
                ShapeToString(params_.vCache),
                "vCache must have shape [BlockNum, Nv, BlockSize, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // kScaleCache shape check: [BlockNum, Nk, BlockSize, 1]
        const auto &kScaleCacheShape = params_.kScaleCache->GetViewShape();
        if (unlikely(kScaleCacheShape.GetDimNum() != 4 || kScaleCacheShape.GetDim(1) != nk ||
            kScaleCacheShape.GetDim(3) != 1)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "kScaleCache",
                ShapeToString(params_.kScaleCache),
                "kScaleCache must have shape [BlockNum, Nk, BlockSize, 1]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // Block consistency: kScaleCache BlockNum/BlockSize must match kCache
        if (unlikely(kCacheShape.GetDim(0) != kScaleCacheShape.GetDim(0) ||
            kCacheShape.GetDim(2) != kScaleCacheShape.GetDim(2))) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "kCache and kScaleCache",
                "kCache=[" + ShapeToString(params_.kCache) +
                "], kScaleCache=[" + ShapeToString(params_.kScaleCache) + "]",
                "kScaleCache BlockNum/BlockSize must match kCache");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // Block consistency: vCache BlockNum/BlockSize must match kCache
        if (unlikely(vCacheShape.GetDim(0) != kCacheShape.GetDim(0) ||
            vCacheShape.GetDim(2) != kCacheShape.GetDim(2))) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "vCache and kCache",
                "vCache=[" + ShapeToString(params_.vCache) +
                "], kCache=[" + ShapeToString(params_.kCache) + "]",
                "vCache BlockNum/BlockSize must match kCache");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // Output shape check: qOut [Nq, T, D], qScale [Nq, T]
        if (!IsShape(params_.qOut, {nq, t, d}) || !IsShape(params_.qScale, {nq, t})) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "qOut, qScale",
                ShapeToString(params_.qOut) + ", " + ShapeToString(params_.qScale),
                "qOut must have shape [Nq, T, D], qScale must have shape [Nq, T]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        return ACLNN_SUCCESS;
    }

    // 4. Format check
    aclnnStatus CheckFormat()
    {
        if (unlikely(op::IsPrivateFormat(params_.qkv->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "qkv",
                op::ToString(params_.qkv->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.qGamma->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "qGamma",
                op::ToString(params_.qGamma->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.kGamma->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "kGamma",
                op::ToString(params_.kGamma->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.cosSin->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "cosSin",
                op::ToString(params_.cosSin->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.slotMapping->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "slotMapping",
                op::ToString(params_.slotMapping->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.kCache->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "kCache",
                op::ToString(params_.kCache->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.vCache->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "vCache",
                op::ToString(params_.vCache->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.kScaleCache->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "kScaleCache",
                op::ToString(params_.kScaleCache->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.cuSeqLen->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "cuSeqLen",
                op::ToString(params_.cuSeqLen->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.optionalQRotation->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "optionalQRotation",
                op::ToString(params_.optionalQRotation->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.optionalKRotation->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "optionalKRotation",
                op::ToString(params_.optionalKRotation->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.optionalVScale->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "optionalVScale",
                op::ToString(params_.optionalVScale->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.qOut->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "qOut",
                op::ToString(params_.qOut->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.qScale->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "qScale",
                op::ToString(params_.qScale->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    // 5. Empty tensor check
    aclnnStatus CheckEmptyTensor()
    {
        int64_t qkvShapeSize = params_.qkv->GetViewShape().GetShapeSize();
        int64_t kCacheShapeSize = params_.kCache->GetViewShape().GetShapeSize();
        int64_t vCacheShapeSize = params_.vCache->GetViewShape().GetShapeSize();
        if (qkvShapeSize == 0 || kCacheShapeSize == 0 || vCacheShapeSize == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qkv/kCache/vCache", "empty tensor",
                "do not support empty tensors");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
};

} // namespace QkvRmsNormRopeCacheWithKScaleCheck

#endif
