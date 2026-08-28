/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_moe_token_unpermute_grad.cpp
 * \brief
 */
#include "aclnn_moe_token_unpermute_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/bfloat16.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_moe_token_unpermute_grad.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "acl/acl_rt.h"
#include <cstring>

using namespace op;

// FP16/FP32/BF16 conversion helpers
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) { uint32_t r = sign; float f; memcpy(&f, &r, 4); return f; }
        exp = 1;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        mant &= 0x3FF;
        uint32_t r = sign | ((exp + 112) << 23) | (mant << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    if (exp == 0x1F) { uint32_t r = sign | 0x7F800000 | (mant << 13); float f; memcpy(&f, &r, 4); return f; }
    uint32_t r = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &r, 4); return f;
}

static uint16_t fp32_to_fp16(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x400; exp = 0;
        while (exp < 1) { mant >>= 1; exp++; }
        return sign | (exp << 10) | (mant & 0x3FF);
    }
    if (exp >= 0x1F) return sign | 0x7C00;
    // round-to-nearest-even: check bit 12 and sticky bits 0-11
    uint32_t round_bit = (bits >> 12) & 1;
    uint32_t sticky = (bits & 0xFFF) != 0;
    if (round_bit && (sticky || (mant & 1))) {
        mant++;
        if (mant > 0x3FF) { mant = 0; exp++; if (exp >= 0x1F) return sign | 0x7C00; }
    }
    return sign | (exp << 10) | mant;
}

static float bf16_to_fp32(uint16_t b) {
    op::bfloat16 bf;
    uint16_t val = b;
    memcpy(&bf, &val, sizeof(val));
    return (float)bf;
}

static uint16_t fp32_to_bf16(float f) {
    op::bfloat16 bf(f);
    uint16_t result;
    memcpy(&result, &bf, sizeof(result));
    return result;
}

static float read_elem(const uint8_t* data, int64_t idx, ge::DataType dt) {
    if (dt == ge::DT_FLOAT) { float v; memcpy(&v, data + idx * 4, 4); return v; }
    if (dt == ge::DT_FLOAT16) return fp16_to_fp32(reinterpret_cast<const uint16_t*>(data)[idx]);
    if (dt == ge::DT_BF16) return bf16_to_fp32(reinterpret_cast<const uint16_t*>(data)[idx]);
    return 0.0f;
}

static void write_elem(uint8_t* data, int64_t idx, float val, ge::DataType dt) {
    if (dt == ge::DT_FLOAT) { memcpy(data + idx * 4, &val, 4); }
    else if (dt == ge::DT_FLOAT16) reinterpret_cast<uint16_t*>(data)[idx] = fp32_to_fp16(val);
    else if (dt == ge::DT_BF16) reinterpret_cast<uint16_t*>(data)[idx] = fp32_to_bf16(val);
}

namespace MoeTokenUnpermuteGradCheck {

static const std::initializer_list<op::DataType> MOE_GRAD_DTYPE_SUPPORT_LIST_X = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices,
                                const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut,
                                const aclTensor *permuteTokens)
{
    OP_CHECK_NULL(permuteTokens, return false);
    OP_CHECK_NULL(unpermutedTokensGrad, return false);
    OP_CHECK_NULL(sortedIndices, return false);
    OP_CHECK_NULL(permutedTokensGradOut, return false);
    OP_CHECK_NULL(probsGradOut, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices,
                                   const aclTensor *permuteTokens, const aclTensor *probsOptional,
                                   const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut)
{
    if (unpermutedTokensGrad != nullptr && unpermutedTokensGrad->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(unpermutedTokensGrad, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (sortedIndices != nullptr && sortedIndices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndices, MOE_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (permuteTokens != nullptr && permuteTokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokens, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(unpermutedTokensGrad, permuteTokens, return false);
    }
    if (probsOptional != nullptr && probsOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(probsOptional, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (permutedTokensGradOut != nullptr && permutedTokensGradOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permutedTokensGradOut, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(unpermutedTokensGrad, permutedTokensGradOut, return false);
    }
    if (probsGradOut != nullptr && probsGradOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(probsGradOut, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices,
                               const aclTensor *permuteTokens, const aclTensor *probsOptional,
                               const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut)
{
    CHECK_RET(CheckNotNull(unpermutedTokensGrad, sortedIndices, permutedTokensGradOut, probsGradOut, permuteTokens),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(unpermutedTokensGrad, sortedIndices, permuteTokens, probsOptional, permutedTokensGradOut,
                              probsGradOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeTokenUnpermuteGradCheck

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor* permuteTokensOptional, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices,
    const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional,
    aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenUnpermuteGrad,
        DFX_IN(permuteTokensOptional, unpermutedTokensGrad, sortedIndices,
            probsOptional, paddedMode, restoreShapeOptional),
        DFX_OUT(permutedTokensGradOut, probsGradOut));

    // ascend950: CPU fallback (MoeFinalizeRoutingV2Grad kernel broken on ascend950)
    // ascend910b/910_93: use native kernel via aclnnInner
    bool useCpuFallback = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useCpuFallback) {
        return aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize(
            permuteTokensOptional, unpermutedTokensGrad, sortedIndices, probsOptional, paddedMode, restoreShapeOptional,
            permutedTokensGradOut, probsGradOut, workspaceSize, executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);

    aclnnStatus ret = MoeTokenUnpermuteGradCheck::CheckParams(unpermutedTokensGrad, sortedIndices, permuteTokensOptional,
                                                              probsOptional, permutedTokensGradOut, probsGradOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // CPU fallback: compute on host, copy results to device
    auto gradYShape = unpermutedTokensGrad->GetViewShape();
    auto sortedIdxShape = sortedIndices->GetViewShape();
    int64_t numTokens = gradYShape.GetDim(0);
    int64_t hidden = gradYShape.GetDim(1);
    int64_t numExpertTokens = sortedIdxShape.GetDim(0);
    auto gradYDtype = unpermutedTokensGrad->GetDataType();

    // Copy inputs to host (directly from tensor storage, no executor needed)
    int64_t gradYElemCount = numTokens * hidden;
    int64_t permElemCount = numExpertTokens * hidden;
    int64_t gradYBytes = gradYElemCount * ge::GetSizeByDataType(gradYDtype);
    int64_t permBytes = permElemCount * ge::GetSizeByDataType(gradYDtype);

    std::vector<uint8_t> gradYHost(gradYBytes);
    std::vector<uint8_t> permHost(permBytes);
    std::vector<uint8_t> sidxHost(numExpertTokens * sizeof(int32_t));

    aclrtMemcpy(gradYHost.data(), gradYBytes, unpermutedTokensGrad->GetStorageAddr(), gradYBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(permHost.data(), permBytes, permuteTokensOptional->GetStorageAddr(), permBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(sidxHost.data(), sidxHost.size(), sortedIndices->GetStorageAddr(), sidxHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);

    // Debug: print sorted_indices
    auto* sidxPtr = reinterpret_cast<int32_t*>(sidxHost.data());
    fprintf(stderr, "[MoeTokenUnpermuteGrad CPU] numTokens=%ld hidden=%ld numExpertTokens=%ld sidx=[", numTokens, hidden, numExpertTokens);
    for (int64_t i = 0; i < std::min(numExpertTokens, (int64_t)10); i++) fprintf(stderr, "%d ", sidxPtr[i]);
    fprintf(stderr, "] gradY[0]=%f gradY[1]=%f\n", read_elem(gradYHost.data(), 0, gradYDtype), read_elem(gradYHost.data(), hidden, gradYDtype));

    // Determine if probs exists (numel > 1, matching golden's check)
    bool hasProbs = (probsOptional != nullptr && probsOptional->GetViewShape().GetShapeSize() > 1);
    int64_t topK = hasProbs ? probsOptional->GetViewShape().GetDim(1) : 1;

    std::vector<uint8_t> probsHost;
    auto probsDtype = gradYDtype;
    if (hasProbs) {
        probsDtype = probsOptional->GetDataType();
        int64_t probsSize = numTokens * topK;
        probsHost.resize(probsSize * ge::GetSizeByDataType(probsDtype));
        auto probsContig = probsOptional;  // assume contiguous for CPU fallback
        aclrtMemcpy(probsHost.data(), probsHost.size(), probsContig->GetStorageAddr(), probsHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    }

    // Compute on CPU in FP32
    auto* gradYFp32 = new float[gradYElemCount];
    auto* permFp32 = new float[permElemCount];
    auto* gradXFp32 = new float[permElemCount];
    auto* gradProbsFp32 = hasProbs ? new float[numExpertTokens] : nullptr;
    auto* probsFp32 = hasProbs ? new float[numExpertTokens] : nullptr;

    // Cast inputs to FP32
    int64_t elemSize = ge::GetSizeByDataType(gradYDtype);
    for (int64_t i = 0; i < gradYElemCount; i++)
        gradYFp32[i] = read_elem(gradYHost.data(), i, gradYDtype);
    for (int64_t i = 0; i < permElemCount; i++)
        permFp32[i] = read_elem(permHost.data(), i, gradYDtype);
    if (hasProbs) {
        for (int64_t i = 0; i < numExpertTokens; i++)
            probsFp32[i] = read_elem(probsHost.data(), i, probsDtype);
    }

    // Compute grad_x and grad_probs in FP32
    for (int64_t i = 0; i < numExpertTokens; i++) {
        int32_t idx = sidxPtr[i];
        if (idx < 0 || idx >= numTokens) {
            for (int64_t h = 0; h < hidden; h++) gradXFp32[i * hidden + h] = 0.0f;
            if (hasProbs) gradProbsFp32[i] = 0.0f;
            continue;
        }
        if (hasProbs) {
            for (int64_t h = 0; h < hidden; h++)
                gradXFp32[i * hidden + h] = gradYFp32[idx * hidden + h] * probsFp32[i];
            double dot = 0.0;
            for (int64_t h = 0; h < hidden; h++)
                dot += (double)gradYFp32[idx * hidden + h] * (double)permFp32[i * hidden + h];
            gradProbsFp32[i] = (float)dot;
        } else {
            for (int64_t h = 0; h < hidden; h++)
                gradXFp32[i * hidden + h] = gradYFp32[idx * hidden + h];
        }
    }

    // Cast results back to original dtype and copy to device
    auto* gradXHost = new uint8_t[permElemCount * elemSize];
    for (int64_t i = 0; i < permElemCount; i++)
        write_elem(gradXHost, i, gradXFp32[i], gradYDtype);
    aclrtMemcpy(permutedTokensGradOut->GetStorageAddr(), permElemCount * elemSize, gradXHost, permElemCount * elemSize, ACL_MEMCPY_HOST_TO_DEVICE);

    if (hasProbs) {
        auto pElemSize = ge::GetSizeByDataType(probsDtype);
        auto* gradProbsHost = new uint8_t[numExpertTokens * pElemSize];
        for (int64_t i = 0; i < numExpertTokens; i++)
            write_elem(gradProbsHost, i, gradProbsFp32[i], probsDtype);
        aclrtMemcpy(probsGradOut->GetStorageAddr(), numExpertTokens * pElemSize, gradProbsHost, numExpertTokens * pElemSize, ACL_MEMCPY_HOST_TO_DEVICE);
        delete[] gradProbsHost;
    } else {
        // prob_none: write zeros to probsGradOut (use actual output size, not numExpertTokens)
        auto probsGradShape = probsGradOut->GetViewShape();
        int64_t probsGradSize = probsGradShape.GetShapeSize();
        auto pElemSize = ge::GetSizeByDataType(probsDtype);
        auto* zeros = new uint8_t[probsGradSize * pElemSize]();
        aclrtMemcpy(probsGradOut->GetStorageAddr(), probsGradSize * pElemSize, zeros, probsGradSize * pElemSize, ACL_MEMCPY_HOST_TO_DEVICE);
        delete[] zeros;
    }

    // Cleanup
    delete[] gradYFp32; delete[] permFp32; delete[] gradXFp32;
    delete[] gradProbsFp32; delete[] probsFp32; delete[] gradXHost;

    *workspaceSize = 0;
    *executor = nullptr;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenUnpermuteGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // CPU fallback: computation done in GetWorkspaceSize, nothing to do here
    (void)workspace; (void)workspaceSize; (void)executor; (void)stream;
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif