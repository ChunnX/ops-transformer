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
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_lightning_indexer.h"

using namespace std;

namespace {
#define CHECK_RET(cond) ((cond) ? true : (false))
#define LOG_PRINT(message, ...) \
    do { \
        (void)printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    return aclrtCreateStream(stream);
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    std::vector<int64_t> qShape = {1, 2, 8, 128};
    std::vector<int64_t> kShape = {1, 16, 1, 128};
    std::vector<int64_t> wShape = {1, 2, 8};
    std::vector<int64_t> idxShape = {1, 2, 8};
    std::vector<float> qHost(GetShapeSize(qShape), 1.0f);
    std::vector<float> kHost(GetShapeSize(kShape), 1.0f);
    std::vector<float> wHost(GetShapeSize(wShape), 1.0f);
    std::vector<int32_t> idxHost(GetShapeSize(idxShape), 0);
    void *qAddr = nullptr;
    void *kAddr = nullptr;
    void *wAddr = nullptr;
    void *idxAddr = nullptr;
    void *valAddr = nullptr;
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *w = nullptr;
    aclTensor *idx = nullptr;
    aclTensor *val = nullptr;
    CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q);
    CreateAclTensor(kHost, kShape, &kAddr, aclDataType::ACL_FLOAT16, &k);
    CreateAclTensor(wHost, wShape, &wAddr, aclDataType::ACL_FLOAT16, &w);
    CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx);
    std::vector<float> valHost(GetShapeSize(idxShape), 0.0f);
    CreateAclTensor(valHost, idxShape, &valAddr, aclDataType::ACL_FLOAT, &val);

    char layoutQ[] = "BSND";
    char layoutK[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGenericLightningIndexerGetWorkspaceSize(q, k, w, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                       nullptr, 8, 1, layoutQ, layoutK, 3, 0, true, false, false, -1,
                                                       true, 0, 0, 1.0f, -1, idx, val, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnGenericLightningIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnGenericLightningIndexer(workspace, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnGenericLightningIndexer failed. ERROR: %d\n", ret);
        return ret;
    }
    aclrtSynchronizeStream(stream);
    LOG_PRINT("GenericLightningIndexer example finished.\n");
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(w);
    aclDestroyTensor(idx);
    aclDestroyTensor(val);
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(wAddr);
    aclrtFree(idxAddr);
    aclrtFree(valAddr);
    if (workspace) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
