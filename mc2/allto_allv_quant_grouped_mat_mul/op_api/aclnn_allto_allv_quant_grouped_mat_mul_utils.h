/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_ALLTO_ALLV_QUANT_GROUPED_MAT_MUL_UTILS_H_
#define ACLNN_ALLTO_ALLV_QUANT_GROUPED_MAT_MUL_UTILS_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

namespace AlltoAllvQuantGroupedMatMulUtils {

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
};

constexpr size_t TWO_DIMS = 2U;
constexpr size_t THREE_DIMS = 3U;

bool CheckNullStatus(const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                     const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                     const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
                     bool permuteOutFlag, const aclTensor *mmYOptional, const aclTensor *permuteOutOptional);

bool CheckNotNull(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY,
                 const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, int64_t gmmXQuantMode,
                 int64_t gmmWeightQuantMode);

bool CheckGmmWeightValid(const aclTensor *gmmWeight);

bool CheckMmWeightValid(const aclTensor *mmWeightOptional);

const aclTensor *TransGmmWeightTensor(const aclTensor *gmmWeight);

const aclTensor *TransGmmWeightScaleTensor(const aclTensor *gmmWeightScale);

const aclTensor *TransMmWeightOptionalTensor(const aclTensor *mmWeightOptional);

const aclTensor *TransMmWeightScaleTensor(const aclTensor *mmWeightScale);

bool IsTransposeTwoDims(const aclTensor *tensor, int64_t firstDimIndex, int64_t secondDimIndex);

aclnnStatus CheckWeightScaleTransposeConsistency(const aclTensor *gmmWeight,
    const aclTensor *gmmWeightScale, const aclTensor *mmWeightOptional, const aclTensor *mmWeightScaleOptional);

aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                       const aclTensor *gmmWeightScale, const aclTensor *sendCountsTensorOptional,
                       const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                       const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                       const aclTensor *mmWeightScaleOptional, int64_t gmmXQuantMode,
                       int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                       const char *group, int64_t epWorldSize, bool permuteOutFlag, const aclTensor *gmmY,
                       const aclTensor *mmYOptional, const aclTensor *permuteOutOptional);

aclnnStatus CheckAndHandleCommMode(const char *group, const char *commModeStr, uint8_t &commModeEnum);

}

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t groupSize, int64_t yDtype, int64_t mmDtype, const char *commMode,
    aclTensor *gmmY, aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, AlltoAllvQuantGroupedMatMulUtils::NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

#endif