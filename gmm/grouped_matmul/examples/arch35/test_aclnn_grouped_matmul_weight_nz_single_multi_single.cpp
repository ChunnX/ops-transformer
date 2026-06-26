/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#if __has_include("aclnnop/aclnn_grouped_matmul_weight_nz.h")
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"
#else
#include "../../op_api/aclnn_grouped_matmul_weight_nz.h"
#endif

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1L, std::multiplies<int64_t>());
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostDataList,
                        const std::vector<std::vector<int64_t>> &shapes, std::vector<void *> &deviceAddrs,
                        aclDataType dataType, aclTensorList **tensorList)
{
    std::vector<aclTensor *> tensors(shapes.size(), nullptr);
    deviceAddrs.assign(shapes.size(), nullptr);
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto ret = CreateAclTensor(hostDataList[i], shapes[i], &deviceAddrs[i], dataType, &tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensorList = aclCreateTensorList(tensors.data(), tensors.size());
    CHECK_RET(*tensorList != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int CreateNzInt8Weight(const std::vector<int8_t> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                       aclTensor **tensor)
{
    constexpr int64_t kBlock = 16L;
    constexpr int64_t nBlock = 32L;
    const int64_t group = shape[0];
    const int64_t k = shape[1];
    const int64_t n = shape[2];
    const std::vector<int64_t> storageShape = {group, (k + nBlock - 1) / nBlock, (n + kBlock - 1) / kBlock, kBlock,
                                               nBlock};
    const std::vector<int64_t> strides = {k * n, 1L, k};
    const auto size = hostData.size() * sizeof(int8_t);

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_INT8, strides.data(), 0, ACL_FORMAT_FRACTAL_NZ,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor for NZ weight failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int CreateNzInt8WeightList(const std::vector<std::vector<int8_t>> &hostDataList,
                           const std::vector<std::vector<int64_t>> &shapes, std::vector<void *> &deviceAddrs,
                           aclTensorList **tensorList)
{
    std::vector<aclTensor *> tensors(shapes.size(), nullptr);
    deviceAddrs.assign(shapes.size(), nullptr);
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto ret = CreateNzInt8Weight(hostDataList[i], shapes[i], &deviceAddrs[i], &tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensorList = aclCreateTensorList(tensors.data(), tensors.size());
    CHECK_RET(*tensorList != nullptr, LOG_PRINT("aclCreateTensorList for NZ weight failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

void FreeDeviceAddrs(std::vector<void *> &deviceAddrs)
{
    for (void *addr : deviceAddrs) {
        if (addr != nullptr) {
            (void)aclrtFree(addr);
        }
    }
    deviceAddrs.clear();
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

int RunGroupedMatmulWeightNzSingleMultiSingle(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    constexpr int64_t m = 64L;
    constexpr int64_t k = 128L;
    constexpr int64_t n = 64L;
    constexpr int64_t groupNum = 2L;

    const std::vector<std::vector<int64_t>> xShape = {{m, k}};
    const std::vector<std::vector<int64_t>> weightShape = {{1L, k, n}, {1L, k, n}};
    const std::vector<std::vector<int64_t>> outShape = {{m, n}};
    const std::vector<int64_t> groupListShape = {groupNum};
    const std::vector<int64_t> groupListData = {32L, 64L};

    std::vector<std::vector<int8_t>> xHostData = {std::vector<int8_t>(m * k, 1)};
    std::vector<std::vector<int8_t>> weightHostData = {std::vector<int8_t>(k * n, 1),
                                                       std::vector<int8_t>(k * n, 2)};
    std::vector<std::vector<int32_t>> outHostData = {std::vector<int32_t>(m * n, 0)};

    std::vector<void *> xDeviceAddrs;
    std::vector<void *> weightDeviceAddrs;
    std::vector<void *> outDeviceAddrs;
    void *groupListDeviceAddr = nullptr;
    aclTensorList *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *out = nullptr;
    aclTensor *groupedList = nullptr;

    ret = CreateAclTensorList(xHostData, xShape, xDeviceAddrs, ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> xTensorPtr(x, aclDestroyTensorList);
    ret = CreateNzInt8WeightList(weightHostData, weightShape, weightDeviceAddrs, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorPtr(weight,
                                                                                           aclDestroyTensorList);
    ret = CreateAclTensorList(outHostData, outShape, outDeviceAddrs, ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> outTensorPtr(out, aclDestroyTensorList);
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, ACL_INT64, &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupedList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);

    ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
        x, weight, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, groupedList, nullptr, nullptr, nullptr, 3L, 0L,
        0L, 0L, nullptr, 0L, out, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnGroupedMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<int32_t> resultData(m * n, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddrs[0],
                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < 10; ++i) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    FreeDeviceAddrs(xDeviceAddrs);
    FreeDeviceAddrs(weightDeviceAddrs);
    FreeDeviceAddrs(outDeviceAddrs);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = RunGroupedMatmulWeightNzSingleMultiSingle(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("GroupedMatmul WeightNz single-multi-single demo failed. ERROR: %d\n", ret); return ret);
    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
