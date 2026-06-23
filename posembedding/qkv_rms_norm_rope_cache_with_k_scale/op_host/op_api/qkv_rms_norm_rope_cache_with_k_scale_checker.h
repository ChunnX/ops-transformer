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

#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_log.h"
#include "base/err_msg.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace QkvRmsNormRopeCacheWithKScaleCheck {

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
        for (uint64_t i = 0; i < shape.GetDimNum(); ++i) {
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
        for (uint64_t i = 0; i < static_cast<uint64_t>(expectedShape.size()); ++i) {
            if (shape.GetDim(i) != expectedShape[i]) {
                return false;
            }
        }
        return true;
    }

    static void LogInvalidValue(const std::string &paramName, const std::string &incorrectValue,
                                const std::string &reason)
    {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID,
                               "Parameter %s of operator %s has incorrect value %s. Reason: %s.",
                               paramName.c_str(), ACLNN_NAME, incorrectValue.c_str(), reason.c_str());
        const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_value", "reason"};
        const std::vector<const char *> msgValue = {paramName.c_str(), ACLNN_NAME, incorrectValue.c_str(),
                                                    reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ0026", msgKey, msgValue);
    }

    static void LogInvalidValues(const std::string &paramNames, const std::string &incorrectValues,
                                 const std::string &reason)
    {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID,
                               "Parameters %s of operator %s have incorrect values %s. Reason: %s.",
                               paramNames.c_str(), ACLNN_NAME, incorrectValues.c_str(), reason.c_str());
        const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_values", "reason"};
        const std::vector<const char *> msgValue = {paramNames.c_str(), ACLNN_NAME, incorrectValues.c_str(),
                                                    reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ0027", msgKey, msgValue);
    }

    static void LogInvalidListSize(const std::string &paramName, const std::string &incorrectSize,
                                   const std::string &correctSize)
    {
        OP_LOGE_WITHOUT_REPORT(
            ACLNN_ERR_PARAM_INVALID,
            "Parameter %s of operator %s has incorrect element nums %s. The correct one should be %s.",
            paramName.c_str(), ACLNN_NAME, incorrectSize.c_str(), correctSize.c_str());
        const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_size", "correct_size"};
        const std::vector<const char *> msgValue = {paramName.c_str(), ACLNN_NAME, incorrectSize.c_str(),
                                                    correctSize.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ0025", msgKey, msgValue);
    }

    static void LogInvalidDtype(const std::string &paramName, const std::string &incorrectDtype,
                                const std::string &reason)
    {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID,
                               "Parameter %s of operator %s has incorrect dtype %s. Reason: %s.",
                               paramName.c_str(), ACLNN_NAME, incorrectDtype.c_str(), reason.c_str());
        const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_dtype", "reason"};
        const std::vector<const char *> msgValue = {paramName.c_str(), ACLNN_NAME, incorrectDtype.c_str(),
                                                    reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ0020", msgKey, msgValue);
    }

    static void LogInvalidShape(const std::string &paramName, const std::string &incorrectShape,
                                const std::string &reason)
    {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID,
                               "Parameter %s of operator %s has incorrect shape [%s]. Reason: %s.",
                               paramName.c_str(), ACLNN_NAME, incorrectShape.c_str(), reason.c_str());
        const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_shape", "reason"};
        const std::vector<const char *> msgValue = {paramName.c_str(), ACLNN_NAME, incorrectShape.c_str(),
                                                    reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ0009", msgKey, msgValue);
    }

    static void LogInvalidShapes(const std::string &paramNames, const std::string &incorrectShapes,
                                 const std::string &reason)
    {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID,
                               "Parameters %s of operator %s have incorrect shapes %s. Reason: %s.",
                               paramNames.c_str(), ACLNN_NAME, incorrectShapes.c_str(), reason.c_str());
        const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_shapes", "reason"};
        const std::vector<const char *> msgValue = {paramNames.c_str(), ACLNN_NAME, incorrectShapes.c_str(),
                                                    reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ0010", msgKey, msgValue);
    }

    static void LogInvalidShapeDim(const std::string &paramName, const std::string &incorrectDim,
                                   const std::string &reason)
    {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID,
                               "Parameter %s of operator %s has incorrect shape dim %s. Reason: %s.",
                               paramName.c_str(), ACLNN_NAME, incorrectDim.c_str(), reason.c_str());
        const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_dim", "reason"};
        const std::vector<const char *> msgValue = {paramName.c_str(), ACLNN_NAME, incorrectDim.c_str(),
                                                    reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ0012", msgKey, msgValue);
    }

    static void LogInvalidFormat(const std::string &paramName, const std::string &incorrectFormat,
                                 const std::string &correctFormat)
    {
        OP_LOGE_WITHOUT_REPORT(
            ACLNN_ERR_PARAM_INVALID,
            "Parameter %s of operator %s has incorrect format %s. The correct one should be %s.",
            paramName.c_str(), ACLNN_NAME, incorrectFormat.c_str(), correctFormat.c_str());
        const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_format", "correct_format"};
        const std::vector<const char *> msgValue = {paramName.c_str(), ACLNN_NAME, incorrectFormat.c_str(),
                                                    correctFormat.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ0017", msgKey, msgValue);
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
            LogInvalidDtype(paramName, op::ToString(tensor->GetDataType()).GetString(),
                            "the dtype of " + paramName + " must be one of " +
                            DataTypeListToString(supportList));
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    // 1. Null pointer check
    aclnnStatus CheckNotNull()
    {
        if (params_.qkv == nullptr) {
            LogInvalidValue("qkv", "nullptr", "qkv can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.qGamma == nullptr) {
            LogInvalidValue("qGamma", "nullptr", "qGamma can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.kGamma == nullptr) {
            LogInvalidValue("kGamma", "nullptr", "kGamma can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.cosSin == nullptr) {
            LogInvalidValue("cosSin", "nullptr", "cosSin can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.slotMapping == nullptr) {
            LogInvalidValue("slotMapping", "nullptr",
                "slotMapping can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.kCache == nullptr) {
            LogInvalidValue("kCache", "nullptr", "kCache can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.vCache == nullptr) {
            LogInvalidValue("vCache", "nullptr", "vCache can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.kScaleCache == nullptr) {
            LogInvalidValue("kScaleCache", "nullptr",
                "kScaleCache can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.cuSeqLen == nullptr) {
            LogInvalidValue("cuSeqLen", "nullptr",
                "cuSeqLen can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.actualSeqLens == nullptr) {
            LogInvalidValue("actualSeqLens", "nullptr",
                "actualSeqLens can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.optionalQRotation == nullptr) {
            LogInvalidValue("optionalQRotation", "nullptr",
                "optionalQRotation can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.optionalKRotation == nullptr) {
            LogInvalidValue("optionalKRotation", "nullptr",
                "optionalKRotation can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.optionalVScale == nullptr) {
            LogInvalidValue("optionalVScale", "nullptr",
                "optionalVScale can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.headNums == nullptr) {
            LogInvalidValue("headNums", "nullptr",
                "headNums can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.inputLayout == nullptr) {
            LogInvalidValue("inputLayout", "nullptr",
                "inputLayout can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.qOut == nullptr) {
            LogInvalidValue("qOut", "nullptr", "qOut can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.qScale == nullptr) {
            LogInvalidValue("qScale", "nullptr", "qScale can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.workspaceSize == nullptr) {
            LogInvalidValue("workspaceSize", "nullptr",
                "workspaceSize can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (params_.executor == nullptr) {
            LogInvalidValue("executor", "nullptr",
                "executor can not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        return ACLNN_SUCCESS;
    }

    // 2. Dtype check
    aclnnStatus CheckDtypeValid()
    {
        aclnnStatus ret = CheckDtypeSupported("qkv", params_.qkv, QKV_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("qGamma", params_.qGamma, GAMMA_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("kGamma", params_.kGamma, GAMMA_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("cosSin", params_.cosSin, COS_SIN_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("slotMapping", params_.slotMapping, SLOT_MAPPING_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("kCache", params_.kCache, CACHE_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("vCache", params_.vCache, CACHE_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("kScaleCache", params_.kScaleCache, K_SCALE_CACHE_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("cuSeqLen", params_.cuSeqLen, CU_SEQ_LEN_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("actualSeqLens", params_.actualSeqLens, ACTUAL_SEQ_LENS_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("optionalQRotation", params_.optionalQRotation, ROTATION_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("optionalKRotation", params_.optionalKRotation, ROTATION_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("optionalVScale", params_.optionalVScale, V_SCALE_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("qOut", params_.qOut, Q_OUT_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        ret = CheckDtypeSupported("qScale", params_.qScale, Q_SCALE_TYPE_LIST);
        if (ret != ACLNN_SUCCESS) { return ret; }
        return ACLNN_SUCCESS;
    }

    // 3. Shape check
    aclnnStatus ParseHeadNums(int64_t &nq, int64_t &nk, int64_t &nv)
    {
        if (params_.headNums->Size() != static_cast<int64_t>(HEAD_NUMS_SIZE)) {
            LogInvalidListSize("headNums", std::to_string(params_.headNums->Size()),
                               std::to_string(HEAD_NUMS_SIZE));
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t *headNumsData = params_.headNums->GetData();
        if (headNumsData == nullptr) {
            LogInvalidValue("headNums", "nullptr", "headNums data can not be nullptr");
            return ACLNN_ERR_PARAM_INVALID;
        }

        nq = headNumsData[0];
        nk = headNumsData[1];
        nv = headNumsData[2];
        if (nq <= 0 || nk <= 0 || nv <= 0) {
            LogInvalidValue("headNums",
                std::to_string(nq) + ", " + std::to_string(nk) + ", " + std::to_string(nv),
                "headNums values must be positive");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (nv != nk) {
            LogInvalidValue("Nv, Nk",
                std::to_string(nv) + ", " + std::to_string(nk),
                "nv must be equal to nk");
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
        const bool isNtd = std::strcmp(params_.inputLayout, INPUT_LAYOUT_NTD) == 0;
        const bool isTnd = std::strcmp(params_.inputLayout, INPUT_LAYOUT_TND) == 0;
        if (!isNtd && !isTnd) {
            LogInvalidValue("inputLayout", params_.inputLayout,
                "inputLayout must be NTD or TND");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // qkv shape check: NTD [Nq+Nk+Nv, T, D], TND [T, Nq+Nk+Nv, D]
        const auto &qkvShape = params_.qkv->GetViewShape();
        uint64_t qkvDimNum = qkvShape.GetDimNum();
        if (unlikely(qkvDimNum != 3)) {
            LogInvalidShapeDim("qkv",
                std::to_string(qkvDimNum) + "D", "qkv must be a 3D tensor [Nq+Nk+Nv, T, D] or [T, Nq+Nk+Nv, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t hqkv = qkvShape.GetDim(isTnd ? 1 : 0);
        const int64_t t = qkvShape.GetDim(isTnd ? 0 : 1);
        const int64_t d = qkvShape.GetDim(2);
        if (unlikely(hqkv != nq + nk + nv)) {
            LogInvalidValues("qkv, headNums",
                std::to_string(hqkv) + ", " + std::to_string(nq + nk + nv),
                "qkv logical N dimension must equal Nq+Nk+Nv from headNums");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(d != HEAD_DIM_FAST_PATH)) {
            LogInvalidValue("D",
                std::to_string(d), "head dimension must be 128");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(d % 2 != 0)) {
            LogInvalidValue("D",
                std::to_string(d), "head dimension must be even for RoPE");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // gamma shape check: [D]
        if (!IsShape(params_.qGamma, {d}) || !IsShape(params_.kGamma, {d})) {
            LogInvalidShapes("qGamma, kGamma",
                ShapeToString(params_.qGamma) + ", " + ShapeToString(params_.kGamma),
                "qGamma and kGamma must have shape [D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // cosSin shape check: [*, D]
        const auto &cosSinShape = params_.cosSin->GetViewShape();
        if (unlikely(cosSinShape.GetDimNum() != 2 || cosSinShape.GetDim(1) != d)) {
            LogInvalidShape("cosSin",
                ShapeToString(params_.cosSin), "cosSin must have shape [MaxSeqLen, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // slotMapping shape check: [T]
        if (!IsShape(params_.slotMapping, {t})) {
            LogInvalidShape("slotMapping",
                ShapeToString(params_.slotMapping), "slotMapping must have shape [T]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // cuSeqLen shape check: 1D, len >= 2
        const auto &cuSeqLenShape = params_.cuSeqLen->GetViewShape();
        if (unlikely(cuSeqLenShape.GetDimNum() != 1 || cuSeqLenShape.GetDim(0) < 2)) {
            LogInvalidShape("cuSeqLen",
                ShapeToString(params_.cuSeqLen), "cuSeqLen must be a 1D tensor with length >= 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const auto &actualSeqLensShape = params_.actualSeqLens->GetViewShape();
        if (unlikely(actualSeqLensShape.GetDimNum() != 1 ||
            actualSeqLensShape.GetDim(0) != cuSeqLenShape.GetDim(0) - 1)) {
            LogInvalidShape("actualSeqLens",
                ShapeToString(params_.actualSeqLens), "actualSeqLens must have shape [cuSeqLen.shape[0] - 1]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // rotation shape check: [D, D]
        if (!IsShape(params_.optionalQRotation, {d, d}) ||
            !IsShape(params_.optionalKRotation, {d, d})) {
            LogInvalidShapes("optionalQRotation, optionalKRotation",
                ShapeToString(params_.optionalQRotation) + ", " + ShapeToString(params_.optionalKRotation),
                "optionalQRotation and optionalKRotation must have shape [D, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // vScale shape check: [Nv]
        if (!IsShape(params_.optionalVScale, {nv})) {
            LogInvalidShape("optionalVScale",
                ShapeToString(params_.optionalVScale), "optionalVScale must have shape [Nv]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // kCache shape check: [BlockNum, Nk, BlockSize, D]
        const auto &kCacheShape = params_.kCache->GetViewShape();
        if (unlikely(kCacheShape.GetDimNum() != 4 || kCacheShape.GetDim(1) != nk ||
            kCacheShape.GetDim(3) != d)) {
            LogInvalidShape("kCache",
                ShapeToString(params_.kCache),
                "kCache must have shape [BlockNum, Nk, BlockSize, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // vCache shape check: [BlockNum, Nv, BlockSize, D]
        const auto &vCacheShape = params_.vCache->GetViewShape();
        if (unlikely(vCacheShape.GetDimNum() != 4 || vCacheShape.GetDim(1) != nv ||
            vCacheShape.GetDim(3) != d)) {
            LogInvalidShape("vCache",
                ShapeToString(params_.vCache),
                "vCache must have shape [BlockNum, Nv, BlockSize, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // kScaleCache shape check: [BlockNum, Nk, BlockSize, 1]
        const auto &kScaleCacheShape = params_.kScaleCache->GetViewShape();
        if (unlikely(kScaleCacheShape.GetDimNum() != 4 || kScaleCacheShape.GetDim(1) != nk ||
            kScaleCacheShape.GetDim(3) != 1)) {
            LogInvalidShape("kScaleCache",
                ShapeToString(params_.kScaleCache),
                "kScaleCache must have shape [BlockNum, Nk, BlockSize, 1]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // Block consistency: kScaleCache BlockNum/BlockSize must match kCache
        if (unlikely(kCacheShape.GetDim(0) != kScaleCacheShape.GetDim(0) ||
            kCacheShape.GetDim(2) != kScaleCacheShape.GetDim(2))) {
            LogInvalidShapes("kCache and kScaleCache",
                "kCache=[" + ShapeToString(params_.kCache) +
                "], kScaleCache=[" + ShapeToString(params_.kScaleCache) + "]",
                "kScaleCache BlockNum/BlockSize must match kCache");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // Block consistency: vCache BlockNum/BlockSize must match kCache
        if (unlikely(vCacheShape.GetDim(0) != kCacheShape.GetDim(0) ||
            vCacheShape.GetDim(2) != kCacheShape.GetDim(2))) {
            LogInvalidShapes("vCache and kCache",
                "vCache=[" + ShapeToString(params_.vCache) +
                "], kCache=[" + ShapeToString(params_.kCache) + "]",
                "vCache BlockNum/BlockSize must match kCache");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const std::vector<int64_t> expectedQOutShape = isTnd ? std::vector<int64_t>{t, nq, d} :
                                                               std::vector<int64_t>{nq, t, d};
        const std::vector<int64_t> expectedQScaleShape = isTnd ? std::vector<int64_t>{t, nq} :
                                                                 std::vector<int64_t>{nq, t};
        if (!IsShape(params_.qOut, expectedQOutShape) || !IsShape(params_.qScale, expectedQScaleShape)) {
            LogInvalidShapes("qOut, qScale",
                ShapeToString(params_.qOut) + ", " + ShapeToString(params_.qScale),
                "qOut/qScale must keep the same N/T layout as qkv inputLayout");
            return ACLNN_ERR_PARAM_INVALID;
        }

        return ACLNN_SUCCESS;
    }

    // 4. Format check
    aclnnStatus CheckFormat()
    {
        if (unlikely(op::IsPrivateFormat(params_.qkv->GetStorageFormat()))) {
            LogInvalidFormat("qkv",
                op::ToString(params_.qkv->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.qGamma->GetStorageFormat()))) {
            LogInvalidFormat("qGamma",
                op::ToString(params_.qGamma->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.kGamma->GetStorageFormat()))) {
            LogInvalidFormat("kGamma",
                op::ToString(params_.kGamma->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.cosSin->GetStorageFormat()))) {
            LogInvalidFormat("cosSin",
                op::ToString(params_.cosSin->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.slotMapping->GetStorageFormat()))) {
            LogInvalidFormat("slotMapping",
                op::ToString(params_.slotMapping->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.kCache->GetStorageFormat()))) {
            LogInvalidFormat("kCache",
                op::ToString(params_.kCache->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.vCache->GetStorageFormat()))) {
            LogInvalidFormat("vCache",
                op::ToString(params_.vCache->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.kScaleCache->GetStorageFormat()))) {
            LogInvalidFormat("kScaleCache",
                op::ToString(params_.kScaleCache->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.cuSeqLen->GetStorageFormat()))) {
            LogInvalidFormat("cuSeqLen",
                op::ToString(params_.cuSeqLen->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.actualSeqLens->GetStorageFormat()))) {
            LogInvalidFormat("actualSeqLens",
                op::ToString(params_.actualSeqLens->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.optionalQRotation->GetStorageFormat()))) {
            LogInvalidFormat("optionalQRotation",
                op::ToString(params_.optionalQRotation->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.optionalKRotation->GetStorageFormat()))) {
            LogInvalidFormat("optionalKRotation",
                op::ToString(params_.optionalKRotation->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.optionalVScale->GetStorageFormat()))) {
            LogInvalidFormat("optionalVScale",
                op::ToString(params_.optionalVScale->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.qOut->GetStorageFormat()))) {
            LogInvalidFormat("qOut",
                op::ToString(params_.qOut->GetStorageFormat()).GetString(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(op::IsPrivateFormat(params_.qScale->GetStorageFormat()))) {
            LogInvalidFormat("qScale",
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
        int64_t actualSeqLensShapeSize = params_.actualSeqLens->GetViewShape().GetShapeSize();
        if (qkvShapeSize == 0 || kCacheShapeSize == 0 || vCacheShapeSize == 0 || actualSeqLensShapeSize == 0) {
            LogInvalidValue("qkv/kCache/vCache/actualSeqLens", "empty tensor",
                "input tensors can not be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
};

} // namespace QkvRmsNormRopeCacheWithKScaleCheck

#endif
