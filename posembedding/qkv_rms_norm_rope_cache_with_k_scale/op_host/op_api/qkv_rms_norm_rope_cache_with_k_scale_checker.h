/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CHECKER_H
#define OP_API_INC_QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CHECKER_H

#include "log/log.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_log.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace QkvRmsNormRopeCacheWithKScaleCheck {

constexpr float DEFAULT_EPSILON = 1e-6f;

inline bool CheckType(op::DataType dtype, const std::vector<op::DataType> &supportList)
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
    const aclTensor *actualSeqLens{nullptr};
    // Optional inputs (required in first phase, but declared as optional in def)
    const aclTensor *optionalQRotation{nullptr};
    const aclTensor *optionalKRotation{nullptr};
    const aclTensor *optionalVScale{nullptr};
    // Attributes
    const aclIntArray *headNums{nullptr};
    const char *inputLayout{nullptr};
    float epsilon{DEFAULT_EPSILON};
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
    static constexpr int64_t SUPPORTED_HEAD_DIM = 128;
    static constexpr uint64_t RANK_1D = 1;
    static constexpr uint64_t RANK_2D = 2;
    static constexpr uint64_t RANK_3D = 3;
    static constexpr uint64_t RANK_4D = 4;
    static constexpr uint64_t HEAD_NUMS_NQ_INDEX = 0;
    static constexpr uint64_t HEAD_NUMS_NK_INDEX = 1;
    static constexpr uint64_t HEAD_NUMS_NV_INDEX = 2;
    static constexpr uint64_t QKV_NTD_HEAD_INDEX = 0;
    static constexpr uint64_t QKV_NTD_TOKEN_INDEX = 1;
    static constexpr uint64_t QKV_TND_TOKEN_INDEX = 0;
    static constexpr uint64_t QKV_TND_HEAD_INDEX = 1;
    static constexpr uint64_t QKV_HEAD_DIM_INDEX = 2;
    static constexpr uint64_t COS_SIN_HEAD_DIM_INDEX = 1;
    static constexpr uint64_t VECTOR_LENGTH_INDEX = 0;
    static constexpr uint64_t CACHE_BLOCK_NUM_INDEX = 0;
    static constexpr uint64_t CACHE_HEAD_INDEX = 1;
    static constexpr uint64_t CACHE_BLOCK_SIZE_INDEX = 2;
    static constexpr uint64_t CACHE_HEAD_DIM_INDEX = 3;
    static constexpr int64_t MIN_CU_SEQ_LEN_SIZE = 2;
    static constexpr int64_t ACTUAL_SEQ_LENS_BATCH_DIFF = 1;
    static constexpr int64_t K_SCALE_CACHE_SCALE_DIM = 1;
    static constexpr const char *INPUT_LAYOUT_NTD = "NTD";
    static constexpr const char *INPUT_LAYOUT_TND = "TND";

    // Dtype support lists
    inline static const std::vector<op::DataType> QKV_TYPE_LIST = {op::DataType::DT_BF16};
    inline static const std::vector<op::DataType> GAMMA_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::vector<op::DataType> COS_SIN_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::vector<op::DataType> SLOT_MAPPING_TYPE_LIST = {op::DataType::DT_INT32};
    inline static const std::vector<op::DataType> CACHE_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
    inline static const std::vector<op::DataType> K_SCALE_CACHE_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::vector<op::DataType> CU_SEQ_LEN_TYPE_LIST = {op::DataType::DT_INT32};
    inline static const std::vector<op::DataType> ACTUAL_SEQ_LENS_TYPE_LIST = {op::DataType::DT_INT32};
    inline static const std::vector<op::DataType> ROTATION_TYPE_LIST = {op::DataType::DT_BF16};
    inline static const std::vector<op::DataType> V_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::vector<op::DataType> Q_OUT_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
    inline static const std::vector<op::DataType> Q_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT};

    // Main entry point
    aclnnStatus CheckParams(const QkvRmsNormRopeCacheWithKScaleParams &p)
    {
        params_ = p;
        aclnnStatus ret = CheckNotNull();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckEmptyTensor();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckDtypeValid();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckShape();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckFormat();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        return ACLNN_SUCCESS;
    }

private:
    QkvRmsNormRopeCacheWithKScaleParams params_;

    struct PointerNullRule {
        const char *paramName;
        const void *ptr;
        const char *reason;
    };

    struct TensorDtypeRule {
        const char *paramName;
        const aclTensor *tensor;
        const std::vector<op::DataType> *supportList;
    };

    struct TensorRule {
        const char *paramName;
        const aclTensor *tensor;
    };

    // Helper: shape to string
    static std::string ShapeToString(const aclTensor *tensor)
    {
        std::string result;
        const auto &shape = tensor->GetViewShape();
        for (uint64_t i = 0; i < shape.GetDimNum(); ++i) {
            if (i != 0) {
                result += ", ";
            }
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
        for (uint64_t i = 0; i < static_cast<uint64_t>(expectedShape.size()); ++i) {
            if (shape.GetDim(i) != expectedShape[i]) {
                return false;
            }
        }
        return true;
    }

    static std::string DataTypeListToString(const std::vector<op::DataType> &supportList)
    {
        std::string result;
        for (uint64_t i = 0; i < static_cast<uint64_t>(supportList.size()); ++i) {
            if (i != 0) {
                result += ", ";
            }
            result += op::ToString(supportList[i]).GetString();
        }
        return result;
    }

    static aclnnStatus CheckDtypeSupported(const std::string &paramName, const aclTensor *tensor,
                                           const std::vector<op::DataType> &supportList)
    {
        if (!CheckType(tensor->GetDataType(), supportList)) {
            const std::string incorrectDtype = op::ToString(tensor->GetDataType()).GetString();
            const std::string reason =
                "the dtype of " + paramName + " must be one of " + DataTypeListToString(supportList);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ACLNN_NAME, paramName.c_str(), incorrectDtype.c_str(),
                                                  reason.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckFormatValid(const TensorRule &rule)
    {
        if (unlikely(op::IsPrivateFormat(rule.tensor->GetStorageFormat()))) {
            const std::string incorrectFormat = op::ToString(rule.tensor->GetStorageFormat()).GetString();
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, rule.paramName, incorrectFormat.c_str(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckTensorNotEmpty(const TensorRule &rule)
    {
        if (rule.tensor->GetViewShape().GetShapeSize() == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, rule.paramName, "empty tensor",
                                                  "tensor can not be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckExactShape(const char *paramName, const aclTensor *tensor,
                                       const std::vector<int64_t> &expectedShape, const char *reason)
    {
        if (!IsShape(tensor, expectedShape)) {
            const std::string incorrectShape = ShapeToString(tensor);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, paramName, incorrectShape.c_str(), reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    // 1. Null pointer check
    aclnnStatus CheckNotNull()
    {
        const PointerNullRule rules[] = {
            {"qkv", params_.qkv, "qkv can not be nullptr"},
            {"qGamma", params_.qGamma, "qGamma can not be nullptr"},
            {"kGamma", params_.kGamma, "kGamma can not be nullptr"},
            {"cosSin", params_.cosSin, "cosSin can not be nullptr"},
            {"slotMapping", params_.slotMapping, "slotMapping can not be nullptr"},
            {"kCache", params_.kCache, "kCache can not be nullptr"},
            {"vCache", params_.vCache, "vCache can not be nullptr"},
            {"kScaleCache", params_.kScaleCache, "kScaleCache can not be nullptr"},
            {"cuSeqLen", params_.cuSeqLen, "cuSeqLen can not be nullptr"},
            {"actualSeqLens", params_.actualSeqLens, "actualSeqLens can not be nullptr"},
            {"optionalQRotation", params_.optionalQRotation, "optionalQRotation can not be nullptr"},
            {"optionalKRotation", params_.optionalKRotation, "optionalKRotation can not be nullptr"},
            {"optionalVScale", params_.optionalVScale, "optionalVScale can not be nullptr"},
            {"headNums", params_.headNums, "headNums can not be nullptr"},
            {"inputLayout", params_.inputLayout, "inputLayout can not be nullptr"},
            {"qOut", params_.qOut, "qOut can not be nullptr"},
            {"qScale", params_.qScale, "qScale can not be nullptr"},
            {"workspaceSize", params_.workspaceSize, "workspaceSize can not be nullptr"},
            {"executor", params_.executor, "executor can not be nullptr"},
        };
        for (const auto &rule : rules) {
            if (rule.ptr == nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, rule.paramName, "nullptr", rule.reason);
                return ACLNN_ERR_PARAM_NULLPTR;
            }
        }
        return ACLNN_SUCCESS;
    }

    // 2. Dtype check
    aclnnStatus CheckDtypeValid()
    {
        const TensorDtypeRule rules[] = {
            {"qkv", params_.qkv, &QKV_TYPE_LIST},
            {"qGamma", params_.qGamma, &GAMMA_TYPE_LIST},
            {"kGamma", params_.kGamma, &GAMMA_TYPE_LIST},
            {"cosSin", params_.cosSin, &COS_SIN_TYPE_LIST},
            {"slotMapping", params_.slotMapping, &SLOT_MAPPING_TYPE_LIST},
            {"kCache", params_.kCache, &CACHE_TYPE_LIST},
            {"vCache", params_.vCache, &CACHE_TYPE_LIST},
            {"kScaleCache", params_.kScaleCache, &K_SCALE_CACHE_TYPE_LIST},
            {"cuSeqLen", params_.cuSeqLen, &CU_SEQ_LEN_TYPE_LIST},
            {"actualSeqLens", params_.actualSeqLens, &ACTUAL_SEQ_LENS_TYPE_LIST},
            {"optionalQRotation", params_.optionalQRotation, &ROTATION_TYPE_LIST},
            {"optionalKRotation", params_.optionalKRotation, &ROTATION_TYPE_LIST},
            {"optionalVScale", params_.optionalVScale, &V_SCALE_TYPE_LIST},
            {"qOut", params_.qOut, &Q_OUT_TYPE_LIST},
            {"qScale", params_.qScale, &Q_SCALE_TYPE_LIST},
        };
        for (const auto &rule : rules) {
            aclnnStatus ret = CheckDtypeSupported(rule.paramName, rule.tensor, *rule.supportList);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
        return ACLNN_SUCCESS;
    }

    // 3. Shape check
    aclnnStatus ParseHeadNums(int64_t &nq, int64_t &nk, int64_t &nv)
    {
        if (params_.headNums->Size() != static_cast<int64_t>(HEAD_NUMS_SIZE)) {
            const std::string incorrectSize = std::to_string(params_.headNums->Size());
            const std::string correctSize = std::to_string(HEAD_NUMS_SIZE);
            OP_LOGE_FOR_INVALID_LISTSIZE(ACLNN_NAME, "headNums", incorrectSize.c_str(), correctSize.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t *headNumsData = params_.headNums->GetData();
        if (headNumsData == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "headNums", "nullptr",
                                                  "headNums data can not be nullptr");
            return ACLNN_ERR_PARAM_INVALID;
        }

        nq = headNumsData[HEAD_NUMS_NQ_INDEX];
        nk = headNumsData[HEAD_NUMS_NK_INDEX];
        nv = headNumsData[HEAD_NUMS_NV_INDEX];
        if (nq <= 0 || nk <= 0 || nv <= 0) {
            const std::string incorrectValue =
                std::to_string(nq) + ", " + std::to_string(nk) + ", " + std::to_string(nv);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "headNums", incorrectValue.c_str(),
                                                  "headNums values must be positive");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (nv != nk) {
            const std::string incorrectValue = std::to_string(nv) + ", " + std::to_string(nk);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "Nv, Nk", incorrectValue.c_str(),
                                                  "nv must be equal to nk");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckInputLayout(bool &isTnd) const
    {
        const bool isNtd = std::strcmp(params_.inputLayout, INPUT_LAYOUT_NTD) == 0;
        isTnd = std::strcmp(params_.inputLayout, INPUT_LAYOUT_TND) == 0;
        if (!isNtd && !isTnd) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "inputLayout", params_.inputLayout,
                                                  "inputLayout must be NTD or TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckQkvShape(int64_t nq, int64_t nk, int64_t nv, bool isTnd, int64_t &t, int64_t &d) const
    {
        const auto &qkvShape = params_.qkv->GetViewShape();
        uint64_t qkvDimNum = qkvShape.GetDimNum();
        if (unlikely(qkvDimNum != RANK_3D)) {
            const std::string incorrectDim = std::to_string(qkvDimNum) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "qkv", incorrectDim.c_str(),
                                                     "qkv must be a 3D tensor [Nq+Nk+Nv, T, D] or [T, Nq+Nk+Nv, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t hqkv = qkvShape.GetDim(isTnd ? QKV_TND_HEAD_INDEX : QKV_NTD_HEAD_INDEX);
        t = qkvShape.GetDim(isTnd ? QKV_TND_TOKEN_INDEX : QKV_NTD_TOKEN_INDEX);
        d = qkvShape.GetDim(QKV_HEAD_DIM_INDEX);
        if (unlikely(hqkv != nq + nk + nv)) {
            const std::string incorrectValues = std::to_string(hqkv) + ", " + std::to_string(nq + nk + nv);
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(ACLNN_NAME, "qkv, headNums", incorrectValues.c_str(),
                                                   "qkv logical N dimension must equal Nq+Nk+Nv from headNums");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(d != SUPPORTED_HEAD_DIM)) {
            const std::string incorrectValue = std::to_string(d);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "D", incorrectValue.c_str(),
                                                  "head dimension must be 128");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckNonCacheShapes(int64_t t, int64_t d, int64_t nv) const
    {
        if (!IsShape(params_.qGamma, {d}) || !IsShape(params_.kGamma, {d})) {
            const std::string incorrectShapes = ShapeToString(params_.qGamma) + ", " + ShapeToString(params_.kGamma);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "qGamma, kGamma", incorrectShapes.c_str(),
                                                   "qGamma and kGamma must have shape [D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const auto &cosSinShape = params_.cosSin->GetViewShape();
        if (unlikely(cosSinShape.GetDimNum() != RANK_2D || cosSinShape.GetDim(COS_SIN_HEAD_DIM_INDEX) != d)) {
            const std::string incorrectShape = ShapeToString(params_.cosSin);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "cosSin", incorrectShape.c_str(),
                                                  "cosSin must have shape [MaxSeqLen, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        aclnnStatus ret = CheckExactShape("slotMapping", params_.slotMapping, {t}, "slotMapping must have shape [T]");
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        const auto &cuSeqLenShape = params_.cuSeqLen->GetViewShape();
        if (unlikely(cuSeqLenShape.GetDimNum() != RANK_1D ||
                     cuSeqLenShape.GetDim(VECTOR_LENGTH_INDEX) < MIN_CU_SEQ_LEN_SIZE)) {
            const std::string incorrectShape = ShapeToString(params_.cuSeqLen);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "cuSeqLen", incorrectShape.c_str(),
                                                  "cuSeqLen must be a 1D tensor with length >= 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const auto &actualSeqLensShape = params_.actualSeqLens->GetViewShape();
        if (unlikely(actualSeqLensShape.GetDimNum() != RANK_1D ||
                     actualSeqLensShape.GetDim(VECTOR_LENGTH_INDEX) !=
                         cuSeqLenShape.GetDim(VECTOR_LENGTH_INDEX) - ACTUAL_SEQ_LENS_BATCH_DIFF)) {
            const std::string incorrectShape = ShapeToString(params_.actualSeqLens);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "actualSeqLens", incorrectShape.c_str(),
                                                  "actualSeqLens must have shape [cuSeqLen.shape[0] - 1]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (!IsShape(params_.optionalQRotation, {d, d}) || !IsShape(params_.optionalKRotation, {d, d})) {
            const std::string incorrectShapes =
                ShapeToString(params_.optionalQRotation) + ", " + ShapeToString(params_.optionalKRotation);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "optionalQRotation, optionalKRotation",
                                                   incorrectShapes.c_str(),
                                                   "optionalQRotation and optionalKRotation must have shape [D, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        return CheckExactShape("optionalVScale", params_.optionalVScale, {nv}, "optionalVScale must have shape [Nv]");
    }

    aclnnStatus CheckCacheShapes(int64_t nk, int64_t nv, int64_t d) const
    {
        const auto &kCacheShape = params_.kCache->GetViewShape();
        if (unlikely(kCacheShape.GetDimNum() != RANK_4D || kCacheShape.GetDim(CACHE_HEAD_INDEX) != nk ||
                     kCacheShape.GetDim(CACHE_HEAD_DIM_INDEX) != d)) {
            const std::string incorrectShape = ShapeToString(params_.kCache);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "kCache", incorrectShape.c_str(),
                                                  "kCache must have shape [BlockNum, Nk, BlockSize, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const auto &vCacheShape = params_.vCache->GetViewShape();
        if (unlikely(vCacheShape.GetDimNum() != RANK_4D || vCacheShape.GetDim(CACHE_HEAD_INDEX) != nv ||
                     vCacheShape.GetDim(CACHE_HEAD_DIM_INDEX) != d)) {
            const std::string incorrectShape = ShapeToString(params_.vCache);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "vCache", incorrectShape.c_str(),
                                                  "vCache must have shape [BlockNum, Nv, BlockSize, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const auto &kScaleCacheShape = params_.kScaleCache->GetViewShape();
        if (unlikely(kScaleCacheShape.GetDimNum() != RANK_4D ||
                     kScaleCacheShape.GetDim(CACHE_HEAD_INDEX) != nk ||
                     kScaleCacheShape.GetDim(CACHE_HEAD_DIM_INDEX) != K_SCALE_CACHE_SCALE_DIM)) {
            const std::string incorrectShape = ShapeToString(params_.kScaleCache);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "kScaleCache", incorrectShape.c_str(),
                                                  "kScaleCache must have shape [BlockNum, Nk, BlockSize, 1]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (unlikely(kCacheShape.GetDim(CACHE_BLOCK_NUM_INDEX) !=
                         kScaleCacheShape.GetDim(CACHE_BLOCK_NUM_INDEX) ||
                     kCacheShape.GetDim(CACHE_BLOCK_SIZE_INDEX) !=
                         kScaleCacheShape.GetDim(CACHE_BLOCK_SIZE_INDEX))) {
            const std::string incorrectShapes = "kCache=[" + ShapeToString(params_.kCache) + "], kScaleCache=[" +
                                                ShapeToString(params_.kScaleCache) + "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "kCache and kScaleCache", incorrectShapes.c_str(),
                                                   "kScaleCache BlockNum/BlockSize must match kCache");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (unlikely(vCacheShape.GetDim(CACHE_BLOCK_NUM_INDEX) != kCacheShape.GetDim(CACHE_BLOCK_NUM_INDEX) ||
                     vCacheShape.GetDim(CACHE_BLOCK_SIZE_INDEX) != kCacheShape.GetDim(CACHE_BLOCK_SIZE_INDEX))) {
            const std::string incorrectShapes =
                "vCache=[" + ShapeToString(params_.vCache) + "], kCache=[" + ShapeToString(params_.kCache) + "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "vCache and kCache", incorrectShapes.c_str(),
                                                   "vCache BlockNum/BlockSize must match kCache");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckOutputShapes(int64_t nq, int64_t t, int64_t d, bool isTnd) const
    {
        const std::vector<int64_t> expectedQOutShape =
            isTnd ? std::vector<int64_t>{t, nq, d} : std::vector<int64_t>{nq, t, d};
        const std::vector<int64_t> expectedQScaleShape =
            isTnd ? std::vector<int64_t>{t, nq} : std::vector<int64_t>{nq, t};
        if (!IsShape(params_.qOut, expectedQOutShape) || !IsShape(params_.qScale, expectedQScaleShape)) {
            const std::string incorrectShapes = ShapeToString(params_.qOut) + ", " + ShapeToString(params_.qScale);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "qOut, qScale", incorrectShapes.c_str(),
                                                   "qOut/qScale must keep the same N/T layout as qkv inputLayout");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckShape()
    {
        int64_t nq = 0;
        int64_t nk = 0;
        int64_t nv = 0;
        aclnnStatus ret = ParseHeadNums(nq, nk, nv);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        bool isTnd = false;
        ret = CheckInputLayout(isTnd);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        int64_t t = 0;
        int64_t d = 0;
        ret = CheckQkvShape(nq, nk, nv, isTnd, t, d);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        ret = CheckNonCacheShapes(t, d, nv);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        ret = CheckCacheShapes(nk, nv, d);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        return CheckOutputShapes(nq, t, d, isTnd);
    }

    // 4. Format check
    aclnnStatus CheckFormat()
    {
        const TensorRule rules[] = {
            {"qkv", params_.qkv},
            {"qGamma", params_.qGamma},
            {"kGamma", params_.kGamma},
            {"cosSin", params_.cosSin},
            {"slotMapping", params_.slotMapping},
            {"kCache", params_.kCache},
            {"vCache", params_.vCache},
            {"kScaleCache", params_.kScaleCache},
            {"cuSeqLen", params_.cuSeqLen},
            {"actualSeqLens", params_.actualSeqLens},
            {"optionalQRotation", params_.optionalQRotation},
            {"optionalKRotation", params_.optionalKRotation},
            {"optionalVScale", params_.optionalVScale},
            {"qOut", params_.qOut},
            {"qScale", params_.qScale},
        };
        for (const auto &rule : rules) {
            aclnnStatus ret = CheckFormatValid(rule);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
        return ACLNN_SUCCESS;
    }

    // 5. Empty tensor check
    aclnnStatus CheckEmptyTensor()
    {
        const TensorRule rules[] = {
            {"qkv", params_.qkv},
            {"qGamma", params_.qGamma},
            {"kGamma", params_.kGamma},
            {"cosSin", params_.cosSin},
            {"slotMapping", params_.slotMapping},
            {"kCache", params_.kCache},
            {"vCache", params_.vCache},
            {"kScaleCache", params_.kScaleCache},
            {"cuSeqLen", params_.cuSeqLen},
            {"actualSeqLens", params_.actualSeqLens},
            {"optionalQRotation", params_.optionalQRotation},
            {"optionalKRotation", params_.optionalKRotation},
            {"optionalVScale", params_.optionalVScale},
            {"qOut", params_.qOut},
            {"qScale", params_.qScale},
        };
        for (const auto &rule : rules) {
            aclnnStatus ret = CheckTensorNotEmpty(rule);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
        return ACLNN_SUCCESS;
    }
};

} // namespace QkvRmsNormRopeCacheWithKScaleCheck

#endif
