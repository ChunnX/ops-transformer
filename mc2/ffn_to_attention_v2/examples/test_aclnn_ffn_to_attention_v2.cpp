/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_ffn_to_attention_v2.cpp
 * \brief ACLNN example for FFNToAttentionV2 on Atlas A5.
 */

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_to_attention_v2.h"
#include "hccl/hccl.h"
#include "../../common/op_api/mc2_context.h"

#define CHECK_RET(cond, returnExpr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            returnExpr;                                                                                                \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream ffnToAttentionStream;
    aclrtContext context;
};

constexpr uint32_t WORLD_SIZE = 16;
constexpr uint32_t ATTENTION_WORKER_NUM = 8;
constexpr uint32_t DEVICE_NUM = WORLD_SIZE;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int LaunchOneProcessFFNToAttentionV2(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] rank = %u, hcomName = %s, stream = %p, context = %p\n", args.rankId, hcomName,
              args.ffnToAttentionStream, args.context);

    uint64_t hcclBufferSize = 0;
    aclTensor *mc2Context = nullptr;
    ret = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(hcomName, "FFNToAttentionV2", hcclBufferSize, mc2Context);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] GetMc2ContextTensor failed. ret = %d\n", ret); return ret);

    constexpr int64_t microBatchNum = 1;
    constexpr int64_t tokenNum = 8;
    constexpr int64_t hiddenSize = 7168;
    constexpr int64_t routedExpertNum = 7;
    constexpr int64_t sharedExpertNum = 1;
    constexpr int64_t expertNumPerToken = routedExpertNum + sharedExpertNum;
    int64_t tokenInfoShapeData[] = {microBatchNum, tokenNum, expertNumPerToken};
    int64_t tokenDataShapeData[] = {microBatchNum, tokenNum, expertNumPerToken, hiddenSize};

    void *xDeviceAddr = nullptr;
    void *sessionIdsDeviceAddr = nullptr;
    void *microBatchIdsDeviceAddr = nullptr;
    void *tokenIdsDeviceAddr = nullptr;
    void *expertOffsetsDeviceAddr = nullptr;
    void *actualTokenNumDeviceAddr = nullptr;
    void *attnRankTableDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *sessionIds = nullptr;
    aclTensor *microBatchIds = nullptr;
    aclTensor *tokenIds = nullptr;
    aclTensor *expertOffsets = nullptr;
    aclTensor *actualTokenNum = nullptr;
    aclTensor *attnRankTable = nullptr;
    aclIntArray *tokenInfoTableShape = aclCreateIntArray(tokenInfoShapeData, 3);
    aclIntArray *tokenDataShape = aclCreateIntArray(tokenDataShapeData, 4);

    const std::vector<int64_t> xShape{tokenNum, hiddenSize};
    const std::vector<int64_t> metadataShape{tokenNum};
    const std::vector<int64_t> actualTokenNumShape{1};
    const std::vector<int64_t> attnRankTableShape{ATTENTION_WORKER_NUM};

    std::vector<int16_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<int32_t> sessionIdsHostData(GetShapeSize(metadataShape), 0);
    std::vector<int32_t> microBatchIdsHostData(GetShapeSize(metadataShape), 0);
    std::vector<int32_t> tokenIdsHostData(GetShapeSize(metadataShape), 0);
    std::vector<int32_t> expertOffsetsHostData(GetShapeSize(metadataShape), 0);
    std::vector<int64_t> actualTokenNumHostData(GetShapeSize(actualTokenNumShape), tokenNum);
    std::vector<int32_t> attnRankTableHostData(GetShapeSize(attnRankTableShape), 0);

    for (int32_t i = 0; i < tokenNum; ++i) {
        sessionIdsHostData[i] = i % ATTENTION_WORKER_NUM;
        tokenIdsHostData[i] = i % tokenNum;
        expertOffsetsHostData[i] = i % expertNumPerToken;
    }
    for (int32_t i = 0; i < ATTENTION_WORKER_NUM; ++i) {
        attnRankTableHostData[i] = i;
    }

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sessionIdsHostData, metadataShape, &sessionIdsDeviceAddr, ACL_INT32, &sessionIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(microBatchIdsHostData, metadataShape, &microBatchIdsDeviceAddr, ACL_INT32, &microBatchIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tokenIdsHostData, metadataShape, &tokenIdsDeviceAddr, ACL_INT32, &tokenIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertOffsetsHostData, metadataShape, &expertOffsetsDeviceAddr, ACL_INT32, &expertOffsets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actualTokenNumHostData, actualTokenNumShape, &actualTokenNumDeviceAddr, ACL_INT64,
                          &actualTokenNum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnRankTableHostData, attnRankTableShape, &attnRankTableDeviceAddr, ACL_INT32,
                          &attnRankTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    ret = aclnnFFNToAttentionV2GetWorkspaceSize(
        mc2Context, x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum, attnRankTable, hcomName,
        WORLD_SIZE, tokenInfoTableShape, tokenDataShape, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnFFNToAttentionV2GetWorkspaceSize failed. ret = %d\n", ret); return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d\n", ret); return ret);
    }

    if (args.rankId >= ATTENTION_WORKER_NUM) {
        ret = aclnnFFNToAttentionV2(workspaceAddr, workspaceSize, executor, args.ffnToAttentionStream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] aclnnFFNToAttentionV2 failed. ret = %d\n", ret); return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.ffnToAttentionStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret);
        LOG_PRINT("[INFO] device_%u aclnnFFNToAttentionV2 executed successfully.\n", args.rankId);
    } else {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        LOG_PRINT("[INFO] device_%u is an Attention worker, sleeping for 10 seconds.\n", args.rankId);
    }

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (mc2Context != nullptr) {
        aclDestroyTensor(mc2Context);
    }
    for (aclTensor *tensor : {x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum, attnRankTable}) {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }
    if (tokenInfoTableShape != nullptr) {
        aclDestroyIntArray(tokenInfoTableShape);
    }
    if (tokenDataShape != nullptr) {
        aclDestroyIntArray(tokenDataShape);
    }
    for (void *deviceAddr : {xDeviceAddr, sessionIdsDeviceAddr, microBatchIdsDeviceAddr, tokenIdsDeviceAddr,
                             expertOffsetsDeviceAddr, actualTokenNumDeviceAddr, attnRankTableDeviceAddr}) {
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.ffnToAttentionStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return ACL_SUCCESS;
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

    aclrtStream streams[DEVICE_NUM];
    aclrtContext contexts[DEVICE_NUM];
    for (uint32_t rankId = 0; rankId < DEVICE_NUM; ++rankId) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&contexts[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&streams[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int32_t deviceId = 0; deviceId < WORLD_SIZE; ++deviceId) {
        devices[deviceId] = deviceId;
    }
    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d\n", ret); return ret);

    Args args[DEVICE_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEVICE_NUM);
    for (uint32_t rankId = 0; rankId < DEVICE_NUM; ++rankId) {
        args[rankId] = {rankId, comms[rankId], streams[rankId], contexts[rankId]};
        threads[rankId].reset(
            new (std::nothrow) std::thread(&LaunchOneProcessFFNToAttentionV2, std::ref(args[rankId])));
    }
    for (auto &thread : threads) {
        thread->join();
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");
    return ACL_SUCCESS;
}
