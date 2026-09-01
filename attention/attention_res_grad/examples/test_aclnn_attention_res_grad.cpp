/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_res_grad.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Test shapes: B = 2, N = 4, H = 64, N1 = N + 1 = 5
    const int64_t B  = 2;
    const int64_t N  = 4;
    const int64_t H  = 64;
    const int64_t N1 = N + 1;

    std::vector<int64_t> prefixSumShape      = {B, H};
    std::vector<int64_t> blockResidualShape  = {B, N, H};
    std::vector<int64_t> projWeightShape     = {1, H};
    std::vector<int64_t> normWeightShape     = {H};
    std::vector<int64_t> gradOutputShape     = {B, H};
    std::vector<int64_t> invRmsShape         = {B, N1};
    std::vector<int64_t> probsShape          = {B, N1};

    std::vector<uint16_t> prefixSumData     (GetShapeSize(prefixSumShape),     0x3C00); // FP16 1.0
    std::vector<uint16_t> blockResidualData (GetShapeSize(blockResidualShape),  0x3C00);
    std::vector<uint16_t> projWeightData    (GetShapeSize(projWeightShape),     0x3C00);
    std::vector<uint16_t> normWeightData    (GetShapeSize(normWeightShape),     0x3C00);
    std::vector<uint16_t> gradOutputData    (GetShapeSize(gradOutputShape),     0x3C00);
    std::vector<uint16_t> invRmsData        (GetShapeSize(invRmsShape),         0x3C00);
    std::vector<uint16_t> probsData         (GetShapeSize(probsShape),          0x3C00);

    aclTensor *prefix_sum = nullptr, *block_residual = nullptr, *proj_weight = nullptr, *norm_weight = nullptr;
    aclTensor *grad_output = nullptr, *inv_rms = nullptr, *probs = nullptr;
    void *d_prefix_sum = nullptr, *d_block_residual = nullptr, *d_proj_weight = nullptr, *d_norm_weight = nullptr;
    void *d_grad_output = nullptr, *d_inv_rms = nullptr, *d_probs = nullptr;

    ret = CreateAclTensor(prefixSumData, prefixSumShape, &d_prefix_sum, aclDataType::ACL_FLOAT16, &prefix_sum); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockResidualData, blockResidualShape, &d_block_residual, aclDataType::ACL_FLOAT16, &block_residual); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(projWeightData, projWeightShape, &d_proj_weight, aclDataType::ACL_FLOAT16, &proj_weight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normWeightData, normWeightShape, &d_norm_weight, aclDataType::ACL_FLOAT16, &norm_weight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradOutputData, gradOutputShape, &d_grad_output, aclDataType::ACL_FLOAT16, &grad_output); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invRmsData, invRmsShape, &d_inv_rms, aclDataType::ACL_FLOAT16, &inv_rms); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsData, probsShape, &d_probs, aclDataType::ACL_FLOAT16, &probs); CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *grad_prefix_sum = nullptr, *grad_block_residual = nullptr, *grad_proj_weight = nullptr, *grad_norm_weight = nullptr;
    void *d_grad_prefix_sum = nullptr, *d_grad_block_residual = nullptr, *d_grad_proj_weight = nullptr, *d_grad_norm_weight = nullptr;

    std::vector<uint16_t> gradPrefixSumData    (GetShapeSize(prefixSumShape),     0);
    std::vector<uint16_t> gradBlockResidualData(GetShapeSize(blockResidualShape), 0);
    std::vector<uint16_t> gradProjWeightData   (GetShapeSize(projWeightShape),    0);
    std::vector<uint16_t> gradNormWeightData   (GetShapeSize(normWeightShape),    0);

    ret = CreateAclTensor(gradPrefixSumData, prefixSumShape, &d_grad_prefix_sum, aclDataType::ACL_FLOAT16, &grad_prefix_sum); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradBlockResidualData, blockResidualShape, &d_grad_block_residual, aclDataType::ACL_FLOAT16, &grad_block_residual); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradProjWeightData, projWeightShape, &d_grad_proj_weight, aclDataType::ACL_FLOAT16, &grad_proj_weight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormWeightData, normWeightShape, &d_grad_norm_weight, aclDataType::ACL_FLOAT16, &grad_norm_weight); CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnAttentionResGradGetWorkspaceSize(
        prefix_sum, block_residual, proj_weight, norm_weight,
        grad_output, inv_rms, probs,
        grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionResGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnAttentionResGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionResGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("AttentionResGrad example ran successfully!\n");

    // clean up inputs
    aclDestroyTensor(prefix_sum); aclDestroyTensor(block_residual); aclDestroyTensor(proj_weight);
    aclDestroyTensor(norm_weight); aclDestroyTensor(grad_output); aclDestroyTensor(inv_rms);
    aclDestroyTensor(probs);
    aclrtFree(d_prefix_sum); aclrtFree(d_block_residual); aclrtFree(d_proj_weight);
    aclrtFree(d_norm_weight); aclrtFree(d_grad_output); aclrtFree(d_inv_rms);
    aclrtFree(d_probs);

    // clean up outputs
    aclDestroyTensor(grad_prefix_sum); aclDestroyTensor(grad_block_residual); aclDestroyTensor(grad_proj_weight);
    aclDestroyTensor(grad_norm_weight);
    aclrtFree(d_grad_prefix_sum); aclrtFree(d_grad_block_residual); aclrtFree(d_grad_proj_weight);
    aclrtFree(d_grad_norm_weight);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
