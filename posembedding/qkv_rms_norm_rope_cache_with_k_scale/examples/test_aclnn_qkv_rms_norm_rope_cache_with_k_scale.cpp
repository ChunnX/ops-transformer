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
#include "acl/acl.h"
#include "aclnn_qkv_rms_norm_rope_cache_with_k_scale.h"

int main() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to initialize ACL" << std::endl;
        return -1;
    }

    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to set device" << std::endl;
        aclFinalize();
        return -1;
    }

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to create stream" << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        return -1;
    }

    const int64_t T = 128;
    const int64_t Nq = 16;
    const int64_t Nk = 2;
    const int64_t Nv = 2;
    const int64_t D = 128;
    const int64_t BlockNum = 32;
    const int64_t BlockSize = 16;
    const int64_t Batch = 1;
    const int64_t MaxSeqLen = 128;
    const char *qkvLayout = "TND";
    const float epsilon = 1e-6f;

    std::vector<int64_t> qkvShape = {T, Nq + Nk + Nv, D};
    std::vector<int64_t> qGammaShape = {D};
    std::vector<int64_t> kGammaShape = {D};
    std::vector<int64_t> cosSinShape = {MaxSeqLen, D};
    std::vector<int64_t> slotMappingShape = {T};
    std::vector<int64_t> kCacheShape = {BlockNum, Nk, BlockSize, D};
    std::vector<int64_t> vCacheShape = {BlockNum, Nv, BlockSize, D};
    std::vector<int64_t> kScaleCacheShape = {BlockNum, Nk, BlockSize, 1};
    std::vector<int64_t> cuSeqLenShape = {Batch + 1};
    std::vector<int64_t> actualSeqLensShape = {Batch};
    std::vector<int64_t> rotationShape = {D, D};
    std::vector<int64_t> vScaleShape = {Nv};

    std::vector<int64_t> qOutShape = {T, Nq, D};
    std::vector<int64_t> qScaleShape = {T, Nq};

    aclTensor *qkv = aclCreateTensor(qkvShape.data(), qkvShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, qkvShape.data(), qkvShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *qGamma = aclCreateTensor(qGammaShape.data(), qGammaShape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, qGammaShape.data(), qGammaShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *kGamma = aclCreateTensor(kGammaShape.data(), kGammaShape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, kGammaShape.data(), kGammaShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *cosSin = aclCreateTensor(cosSinShape.data(), cosSinShape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, cosSinShape.data(), cosSinShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *slotMapping = aclCreateTensor(slotMappingShape.data(), slotMappingShape.size(), ACL_INT32, nullptr, 0, ACL_FORMAT_ND, slotMappingShape.data(), slotMappingShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *kCache = aclCreateTensor(kCacheShape.data(), kCacheShape.size(), ACL_FLOAT8_E4M3FN, nullptr, 0, ACL_FORMAT_ND, kCacheShape.data(), kCacheShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *vCache = aclCreateTensor(vCacheShape.data(), vCacheShape.size(), ACL_FLOAT8_E4M3FN, nullptr, 0, ACL_FORMAT_ND, vCacheShape.data(), vCacheShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *kScaleCache = aclCreateTensor(kScaleCacheShape.data(), kScaleCacheShape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, kScaleCacheShape.data(), kScaleCacheShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *cuSeqLen = aclCreateTensor(cuSeqLenShape.data(), cuSeqLenShape.size(), ACL_INT32, nullptr, 0, ACL_FORMAT_ND, cuSeqLenShape.data(), cuSeqLenShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *actualSeqLens = aclCreateTensor(actualSeqLensShape.data(), actualSeqLensShape.size(), ACL_INT32, nullptr, 0, ACL_FORMAT_ND, actualSeqLensShape.data(), actualSeqLensShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *rotation = aclCreateTensor(rotationShape.data(), rotationShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, rotationShape.data(), rotationShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *vScale = aclCreateTensor(vScaleShape.data(), vScaleShape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, vScaleShape.data(), vScaleShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);

    std::vector<int64_t> headNumsVec = {Nq, Nk, Nv};
    aclIntArray *headNums = aclCreateIntArray(headNumsVec.data(), headNumsVec.size());

    aclTensor *qOut = aclCreateTensor(qOutShape.data(), qOutShape.size(), ACL_FLOAT8_E4M3FN, nullptr, 0, ACL_FORMAT_ND, qOutShape.data(), qOutShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    aclTensor *qScale = aclCreateTensor(qScaleShape.data(), qScaleShape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, qScaleShape.data(), qScaleShape.size(), ACL_MEMTYPE_HIGH_BANDWIDTH);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus status = aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
        qkv, qGamma, kGamma, cosSin, slotMapping,
        kCache, vCache, kScaleCache, cuSeqLen, actualSeqLens,
        rotation, vScale, headNums, qkvLayout, epsilon,
        qOut, qScale,
        &workspaceSize, &executor);

    if (status != ACLNN_SUCCESS) {
        std::cerr << "Failed to get workspace size" << std::endl;
        goto cleanup;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Failed to allocate workspace" << std::endl;
            goto cleanup;
        }
    }

    status = aclnnQkvRmsNormRopeCacheWithKScale(workspace, workspaceSize, executor, stream);
    if (status != ACLNN_SUCCESS) {
        std::cerr << "Failed to execute operator" << std::endl;
        if (workspace) aclrtFree(workspace);
        goto cleanup;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to synchronize stream" << std::endl;
    }

    if (workspace) aclrtFree(workspace);
    std::cout << "Operator executed successfully" << std::endl;

cleanup:
    aclDestroyTensor(qkv);
    aclDestroyTensor(qGamma);
    aclDestroyTensor(kGamma);
    aclDestroyTensor(cosSin);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(kCache);
    aclDestroyTensor(vCache);
    aclDestroyTensor(kScaleCache);
    aclDestroyTensor(cuSeqLen);
    aclDestroyTensor(actualSeqLens);
    aclDestroyTensor(rotation);
    aclDestroyTensor(vScale);
    aclDestroyIntArray(headNums);
    aclDestroyTensor(qOut);
    aclDestroyTensor(qScale);
    if (executor) aclDestroyAclOpExecutor(executor);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
