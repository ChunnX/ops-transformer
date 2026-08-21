/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_prepare.h"

#define CHECK_RET(condition, message)           \
    do {                                        \
        if (!(condition)) {                     \
            std::printf("%s\n", (message));    \
            return 1;                           \
        }                                       \
    } while (0)

namespace {

int64_t GetElementCount(const std::vector<int64_t> &shape)
{
    int64_t count = 1;
    for (const int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

template <typename T>
int CreateTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                 void **deviceAddress, aclTensor **tensor)
{
    const size_t bytes = static_cast<size_t>(GetElementCount(shape)) * sizeof(T);
    aclError ret = aclrtMalloc(deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddress, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddress);
    return *tensor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed");
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed");

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed");

    const std::vector<int64_t> blockResShape = {1, 2, 4};
    const std::vector<int64_t> validBlocksShape = {1};
    const std::vector<int64_t> pseudoQueryShape = {2, 4};
    const std::vector<int64_t> numeratorShape = {2, 1, 4};
    const std::vector<int64_t> statsShape = {2, 1};

    const std::vector<float> blockResData = {
        1.0F, 2.0F, 3.0F, 4.0F,
        2.0F, 0.0F, -1.0F, 1.0F,
    };
    const std::vector<int32_t> validBlocksData = {2};
    const std::vector<float> pseudoQueryData = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
    };
    std::vector<float> numeratorData(GetElementCount(numeratorShape), 0.0F);
    std::vector<float> logitMaxData(GetElementCount(statsShape), 0.0F);
    std::vector<float> expSumData(GetElementCount(statsShape), 0.0F);

    void *blockResAddress = nullptr;
    void *validBlocksAddress = nullptr;
    void *pseudoQueryAddress = nullptr;
    void *numeratorAddress = nullptr;
    void *logitMaxAddress = nullptr;
    void *expSumAddress = nullptr;
    aclTensor *blockRes = nullptr;
    aclTensor *validBlocks = nullptr;
    aclTensor *pseudoQuery = nullptr;
    aclTensor *numerator = nullptr;
    aclTensor *logitMax = nullptr;
    aclTensor *expSum = nullptr;

    ret = CreateTensor(blockResData, blockResShape, ACL_FLOAT, &blockResAddress, &blockRes);
    CHECK_RET(ret == ACL_SUCCESS, "create blockRes failed");
    ret = CreateTensor(validBlocksData, validBlocksShape, ACL_INT32, &validBlocksAddress, &validBlocks);
    CHECK_RET(ret == ACL_SUCCESS, "create validBlocks failed");
    ret = CreateTensor(pseudoQueryData, pseudoQueryShape, ACL_FLOAT, &pseudoQueryAddress, &pseudoQuery);
    CHECK_RET(ret == ACL_SUCCESS, "create pseudoQuery failed");
    ret = CreateTensor(numeratorData, numeratorShape, ACL_FLOAT, &numeratorAddress, &numerator);
    CHECK_RET(ret == ACL_SUCCESS, "create numerator failed");
    ret = CreateTensor(logitMaxData, statsShape, ACL_FLOAT, &logitMaxAddress, &logitMax);
    CHECK_RET(ret == ACL_SUCCESS, "create logitMax failed");
    ret = CreateTensor(expSumData, statsShape, ACL_FLOAT, &expSumAddress, &expSum);
    CHECK_RET(ret == ACL_SUCCESS, "create expSum failed");

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBlockAttnResPrepareGetWorkspaceSize(blockRes, validBlocks, pseudoQuery, numerator, logitMax,
                                                    expSum, 1.0e-6F, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnBlockAttnResPrepareGetWorkspaceSize failed");

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "allocate workspace failed");
    }
    ret = aclnnBlockAttnResPrepare(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnBlockAttnResPrepare failed");
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed");

    ret = aclrtMemcpy(numeratorData.data(), numeratorData.size() * sizeof(float), numeratorAddress,
                      numeratorData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "copy numerator failed");
    ret = aclrtMemcpy(logitMaxData.data(), logitMaxData.size() * sizeof(float), logitMaxAddress,
                      logitMaxData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "copy logitMax failed");
    ret = aclrtMemcpy(expSumData.data(), expSumData.size() * sizeof(float), expSumAddress,
                      expSumData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "copy expSum failed");

    for (size_t i = 0; i < numeratorData.size(); ++i) {
        std::printf("numerator[%zu] = %.6f\n", i, numeratorData[i]);
    }
    for (size_t i = 0; i < logitMaxData.size(); ++i) {
        std::printf("logit_max[%zu] = %.6f, exp_sum[%zu] = %.6f\n",
                    i, logitMaxData[i], i, expSumData[i]);
    }

    aclDestroyTensor(blockRes);
    aclDestroyTensor(validBlocks);
    aclDestroyTensor(pseudoQuery);
    aclDestroyTensor(numerator);
    aclDestroyTensor(logitMax);
    aclDestroyTensor(expSum);
    aclrtFree(blockResAddress);
    aclrtFree(validBlocksAddress);
    aclrtFree(pseudoQueryAddress);
    aclrtFree(numeratorAddress);
    aclrtFree(logitMaxAddress);
    aclrtFree(expSumAddress);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
