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
 * \file test_aclnn_engram_fetch.cpp
 * \brief engram_fetch 算子测试用例
 */

#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_engram_fetch.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

constexpr int DEV_NUM = 4;
static std::mutex barrierMtx;
static std::condition_variable barrierCv;
static int barrierCount = 0;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor, aclFormat format = aclFormat::ACL_FORMAT_ND)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, format,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int LaunchOneThreadEngramFetch(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    // Test parameters
    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t numTokens = 8;
    int64_t outputDtype = 1;  // BF16

    // Shapes
    // bufferHandle: int32 ND, buffer descriptor (flat int32 array)
    std::vector<int64_t> bufferHandleShape = {numEntriesPerRank * hiddenSize};
    // indices: int32 ND, 1D, contains token ids to fetch
    std::vector<int64_t> indicesShape = {numTokens};
    // fetched: BF16 ND, 2D [numTokens, hiddenSize]
    std::vector<int64_t> fetchedShape = {numTokens, hiddenSize};

    // Host data
    long long bufferHandleSize = GetShapeSize(bufferHandleShape);
    long long indicesSize = GetShapeSize(indicesShape);
    long long fetchedSize = GetShapeSize(fetchedShape);

    std::vector<int32_t> bufferHandleHostData(bufferHandleSize, 0);
    // Fill indices with sequential token ids: 0, 1, 2, ..., numTokens-1
    std::vector<int32_t> indicesHostData(indicesSize);
    for (int64_t i = 0; i < indicesSize; i++) {
        indicesHostData[i] = static_cast<int32_t>(i);
    }
    std::vector<uint16_t> fetchedHostData(fetchedSize, 0);  // BF16 output, init to 0

    // Device memory and tensors
    void *bufferHandleDeviceAddr = nullptr;
    void *indicesDeviceAddr = nullptr;
    void *fetchedDeviceAddr = nullptr;
    aclTensor *bufferHandle = nullptr;
    aclTensor *indices = nullptr;
    aclTensor *fetched = nullptr;

    // Create input tensors
    ret = CreateAclTensor(bufferHandleHostData, bufferHandleShape, &bufferHandleDeviceAddr,
                          aclDataType::ACL_INT32, &bufferHandle);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr,
                          aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create output tensor (BF16)
    ret = CreateAclTensor(fetchedHostData, fetchedShape, &fetchedDeviceAddr,
                          aclDataType::ACL_BF16, &fetched);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    LOG_PRINT("[INFO] rank = %d, Before aclnnEngramFetchGetWorkspaceSize\n", args.rankId);

    // Phase 1: Get workspace size
    ret = aclnnEngramFetchGetWorkspaceSize(
        bufferHandle, indices, fetched,
        hiddenSize, numEntriesPerRank, hcomName, outputDtype,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnEngramFetchGetWorkspaceSize failed. ret = %d\n", ret); return ret);

    LOG_PRINT("[INFO] rank = %d, workspaceSize = %lu\n", args.rankId, workspaceSize);

    // Allocate workspace
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d\n", ret); return ret);
    }

    LOG_PRINT("[INFO] rank = %d, Before aclnnEngramFetch\n", args.rankId);

    // Phase 2: Execute
    ret = aclnnEngramFetch(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnEngramFetch failed. ret = %d\n", ret); return ret);

    // Sync
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 30000);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret);

    LOG_PRINT("[INFO] rank = %d, aclnnEngramFetch execute successfully.\n", args.rankId);

    // Copy output back
    std::vector<uint16_t> fetchedResult(fetchedSize);
    ret = aclrtMemcpy(fetchedResult.data(), fetchedSize * sizeof(uint16_t),
                      fetchedDeviceAddr, fetchedSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtMemcpy fetched failed. ret = %d\n", ret); return ret);

    // Print first few output values
    LOG_PRINT("[INFO] rank=%d, fetched first 10 values (bf16 hex):\n", args.rankId);
    for (int64_t i = 0; i < 10 && i < fetchedSize; i++) {
        LOG_PRINT("  [%lld] 0x%04x\n", static_cast<long long>(i), fetchedResult[i]);
    }

    // Cleanup
    if (bufferHandle != nullptr) {
        aclDestroyTensor(bufferHandle);
        aclrtFree(bufferHandleDeviceAddr);
    }
    if (indices != nullptr) {
        aclDestroyTensor(indices);
        aclrtFree(indicesDeviceAddr);
    }
    if (fetched != nullptr) {
        aclDestroyTensor(fetched);
        aclrtFree(fetchedDeviceAddr);
    }
    if (workspaceSize > 0 && workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    // Barrier: wait for all ranks before cleanup
    {
        std::unique_lock<std::mutex> lock(barrierMtx);
        barrierCount++;
        if (barrierCount >= DEV_NUM) {
            barrierCv.notify_all();
        } else {
            barrierCv.wait(lock, [&]() { return barrierCount >= DEV_NUM; });
        }
    }

    ret = HcclCommDestroy(args.hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d\n", ret); return ret);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d\n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d\n", ret); return ret);
    ret = aclrtDestroyContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d\n", ret); return ret);

    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
    }

    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }

    // Initialize HCCL communication domain
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d\n", ret); return ret);

    Args args[DEV_NUM];
    // Launch threads
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneThreadEngramFetch,
            std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}
