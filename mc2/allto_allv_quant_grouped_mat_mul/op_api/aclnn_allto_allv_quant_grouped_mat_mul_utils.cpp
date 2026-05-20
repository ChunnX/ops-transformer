/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include "securec.h"
#include "acl/acl.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_allto_allv_quant_grouped_mat_mul_utils.h"

#ifdef BUILD_OPEN_PROJECT
#include "version/hcomm_version.h"
#define HCCL_CHANNEL_SUPPORT_VERSION 90000000
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include "common/op_api/mc2_context.h"
#endif
#endif

#include "mc2_comm_utils.h"

namespace AlltoAllvQuantGroupedMatMulUtils {
using namespace op;

bool CheckNullStatus(const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                     const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                     const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
                     bool permuteOutFlag, const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Optional output flag does not match optional output ptr.");
        return false;
    }
    return true;
}

bool CheckNotNull(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY,
                 const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, int64_t gmmXQuantMode,
                 int64_t gmmWeightQuantMode)
{
    if (gmmX == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmX should not be null.");
        return false;
    }
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmWeight should not be null.");
        return false;
    }
    if (gmmY == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmY should not be null.");
        return false;
    }
    if (gmmXScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXScale should not be null.");
        return false;
    }
    if (gmmWeightScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeightScale should not be null.");
        return false;
    }
    if ((gmmXQuantMode != static_cast<int64_t>(QuantModeType::PERTENSOR_QUANT)) &&
        (gmmXQuantMode != static_cast<int64_t>(QuantModeType::MX_QUANT))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXQuantMode should be 1(pertensor quant) or 6(mx quant), "
            "but actual is %ld.", gmmXQuantMode);
        return false;
    }
    return true;
}

bool CheckGmmWeightValid(const aclTensor *gmmWeight) {
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "In AlltoAllvQuantGroupedMatmul, input gmmWeight should not be null.");
        return false;
    }
    OP_CHECK_WRONG_DIMENSION(gmmWeight, THREE_DIMS, return false);
    return true;
}

bool CheckMmWeightValid(const aclTensor *mmWeightOptional) {
    if (mmWeightOptional == nullptr) {
        return false;
    }
    OP_CHECK_WRONG_DIMENSION(mmWeightOptional, TWO_DIMS, return false);
    return true;
}

const aclTensor *TransGmmWeightTensor(const aclTensor *gmmWeight)
{
    uint64_t storageShapeDimNum = gmmWeight->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = gmmWeight->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = gmmWeight->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = gmmWeight->GetViewShape().GetDim(i);
    }
    viewDim[1] = gmmWeight->GetViewShape().GetDim(2);
    viewDim[2] = gmmWeight->GetViewShape().GetDim(1);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclnnStatus dtypeRet = aclGetDataType(gmmWeight, &dataType);
    if (dtypeRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclGetDataType failed for gmmWeight, ret=%d", dtypeRet);
        return nullptr;
    }
    auto transStride = gmmWeight->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    stride[1] = transStride[2];
    stride[2] = transStride[1];

    auto offset = gmmWeight->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, gmmWeight->GetTensor()->GetAddr());
}

const aclTensor *TransGmmWeightScaleTensor(const aclTensor *gmmWeightScale)
{
    if (gmmWeightScale->GetViewShape().GetDimNum() < 4) {
        return gmmWeightScale;
    }
    
    uint64_t storageShapeDimNum = gmmWeightScale->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = gmmWeightScale->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = gmmWeightScale->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = gmmWeightScale->GetViewShape().GetDim(i);
    }
    viewDim[1] = gmmWeightScale->GetViewShape().GetDim(2);
    viewDim[2] = gmmWeightScale->GetViewShape().GetDim(1);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclnnStatus dtypeRet = aclGetDataType(gmmWeightScale, &dataType);
    if (dtypeRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclGetDataType failed for gmmWeightScale, ret=%d", dtypeRet);
        return nullptr;
    }
    auto transStride = gmmWeightScale->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    stride[1] = transStride[2];
    stride[2] = transStride[1];

    auto offset = gmmWeightScale->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, gmmWeightScale->GetTensor()->GetAddr());
}

const aclTensor *TransMmWeightOptionalTensor(const aclTensor *mmWeightOptional)
{
    uint64_t storageShapeDimNum = mmWeightOptional->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = mmWeightOptional->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = mmWeightOptional->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = mmWeightOptional->GetViewShape().GetDim(i);
    }
    viewDim[0] = mmWeightOptional->GetViewShape().GetDim(1);
    viewDim[1] = mmWeightOptional->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclnnStatus dtypeRet = aclGetDataType(mmWeightOptional, &dataType);
    if (dtypeRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclGetDataType failed for mmWeightOptional, ret=%d", dtypeRet);
        return nullptr;
    }
    auto transStride = mmWeightOptional->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    stride[0] = transStride[1];
    stride[1] = transStride[0];

    auto offset = mmWeightOptional->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, mmWeightOptional->GetTensor()->GetAddr());
}

const aclTensor *TransMmWeightScaleTensor(const aclTensor *mmWeightScale)
{
    if (mmWeightScale->GetViewShape().GetDimNum() < 3) {
        return mmWeightScale;
    }
    
    uint64_t storageShapeDimNum = mmWeightScale->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = mmWeightScale->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = mmWeightScale->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = mmWeightScale->GetViewShape().GetDim(i);
    }
    viewDim[0] = mmWeightScale->GetViewShape().GetDim(1);
    viewDim[1] = mmWeightScale->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclnnStatus dtypeRet = aclGetDataType(mmWeightScale, &dataType);
    if (dtypeRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclGetDataType failed for mmWeightScale, ret=%d", dtypeRet);
        return nullptr;
    }
    auto transStride = mmWeightScale->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    stride[0] = transStride[1];
    stride[1] = transStride[0];

    auto offset = mmWeightScale->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, mmWeightScale->GetTensor()->GetAddr());
}

bool IsTransposeTwoDims(const aclTensor *tensor, int64_t firstDimIndex, int64_t secondDimIndex)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() < 2) {
        return false;
    }
    int64_t dimNum = tensor->GetViewShape().GetDimNum();
    int64_t dim1 = firstDimIndex < 0 ? dimNum + firstDimIndex : firstDimIndex;
    int64_t dim2 = secondDimIndex < 0 ? dimNum + secondDimIndex : secondDimIndex;
    if (dim1 < 0 || dim1 >= dimNum || dim2 < 0 || dim2 >= dimNum || dim1 == dim2) {
        return false;
    }
    int64_t shape1 = tensor->GetViewShape().GetDim(dim1);
    int64_t shape2 = tensor->GetViewShape().GetDim(dim2);
    int64_t stride1 = tensor->GetViewStrides()[dim1];
    int64_t stride2 = tensor->GetViewStrides()[dim2];
    if (shape1 == 1 && shape2 == 1) {
        return false;
    }
    if (stride1 == 1) {
        return (stride2 != shape1);
    }
    if (stride2 == 1) {
        return true;
    }
    return (stride2 != shape1 * stride1);
}

aclnnStatus CheckWeightScaleTransposeConsistency(const aclTensor *gmmWeight,
    const aclTensor *gmmWeightScale, const aclTensor *mmWeightOptional, const aclTensor *mmWeightScaleOptional)
{
    if (gmmWeight != nullptr && gmmWeightScale != nullptr && gmmWeightScale->GetViewShape().GetDimNum() >= 4) {
        bool gmmWeightNotContiguous = IsTransposeTwoDims(gmmWeight, -1, -2);
        bool gmmWeightScaleNotContiguous = IsTransposeTwoDims(gmmWeightScale, -2, -3);
        if (gmmWeightNotContiguous != gmmWeightScaleNotContiguous) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "gmmWeight and gmmWeightScale transpose state must be consistent. "
                    "gmmWeight is %s, but gmmWeightScale is %s.",
                    gmmWeightNotContiguous ? "transposed(non-contiguous)" : "not transposed(contiguous)",
                    gmmWeightScaleNotContiguous ? "transposed(non-contiguous)" : "not transposed(contiguous)");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (mmWeightOptional != nullptr && mmWeightScaleOptional != nullptr &&
        mmWeightScaleOptional->GetViewShape().GetDimNum() >= 3) {
        bool mmWeightNotContiguous = IsTransposeTwoDims(mmWeightOptional, -1, -2);
        bool mmWeightScaleNotContiguous = IsTransposeTwoDims(mmWeightScaleOptional, -2, -3);
        if (mmWeightNotContiguous != mmWeightScaleNotContiguous) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "mmWeight and mmWeightScale transpose state must be consistent. "
                    "mmWeight is %s, but mmWeightScale is %s.",
                    mmWeightNotContiguous ? "transposed(non-contiguous)" : "not transposed(contiguous)",
                    mmWeightScaleNotContiguous ? "transposed(non-contiguous)" : "not transposed(contiguous)");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                       const aclTensor *gmmWeightScale, const aclTensor *sendCountsTensorOptional,
                       const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                       const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                       const aclTensor *mmWeightScaleOptional, int64_t gmmXQuantMode,
                       int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                       const char *group, int64_t epWorldSize, bool permuteOutFlag, const aclTensor *gmmY,
                       const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    CHECK_RET(CheckNullStatus(sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                              mmXScaleOptional, mmWeightScaleOptional, permuteOutFlag, mmYOptional, permuteOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, gmmY, gmmXScale, gmmWeightScale, gmmXQuantMode, gmmWeightQuantMode),
              ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("AlltoAllvQuantGroupedMatmul checkParams success");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckAndHandleCommMode(const char *group, const char *commModeStr, uint8_t &commModeEnum) {
    uint32_t rankSize = 0;
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    auto getRankSizeRet = Mc2Aclnn::Mc2Context::GetMc2RankSize(group, rankSize);
    CHECK_RET(getRankSizeRet == ACLNN_SUCCESS, getRankSizeRet);
#endif

    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (std::strcmp(commModeStr, "aicpu") == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_AICPU;
        } else if (std::strcmp(commModeStr, "ccu") == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_CCU;
        } else if (std::strcmp(commModeStr, "default") == 0) {
            if (rankSize <= 8) {
                commModeEnum = Mc2Comm::COMM_MODE_CCU;
            } else {
                commModeEnum = Mc2Comm::COMM_MODE_AICPU;
            }
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "commMode only support 'default', 'ccu', 'aicpu'.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

}