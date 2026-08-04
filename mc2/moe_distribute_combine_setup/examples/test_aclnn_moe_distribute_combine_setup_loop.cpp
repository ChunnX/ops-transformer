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
 * \file test_aclnn_moe_distribute_combine_setup_loop.cpp
 * \brief 同 EP 通信域循环执行 CombineSetup/Teardown N 次（中间不销毁通信域）
 *
 * 用法: ./test_aclnn_moe_distribute_combine_setup_loop [loop_count]
 *       loop_count 默认 3
 */

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_moe_distribute_combine_setup.h"
#include "aclnnop/aclnn_moe_distribute_combine_teardown.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

constexpr int DEV_NUM = 2;
constexpr int DEFAULT_LOOP_COUNT = 3;

template <typename Func>
class Guard {
public:
    explicit Guard(Func &func) : func_(func)
    {
    }
    ~Guard()
    {
        func_();
    }

private:
    Func &func_;
};

int64_t AlignN(int64_t x, int64_t n)
{
    return (x + n - 1) / n * n;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    HcclComm hcclEpComm;
    aclrtStream combinesetupstream;
    aclrtStream combineteardownstream;
    aclrtContext context;
    int loopCount;
};

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void DestroyTensor(aclTensor *tensor)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
}

void FreeDeviceAddr(void *deviceAddr)
{
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

int RunOneIteration(Args &args, const char *hcomEpName, aclTensor *expandX, aclTensor *expertIds,
                    aclTensor *assistInfoForCombineOut, aclTensor *quantExpandXOut, aclTensor *commCmdInfoOutforCombine,
                    aclTensor *expertScales, aclTensor *expandIdx, aclTensor *xOut, int64_t epWorldSize,
                    int64_t epRankId, int64_t moeExpertNum, int64_t expertSharedType, int64_t sharedExpertNum,
                    int64_t sharedExpertRankNum, int64_t globalBS, int64_t commQuantMode, int64_t commType,
                    int64_t timeOut, int iter)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    int ret = 0;

    ret = aclnnMoeDistributeCombineSetupGetWorkspaceSize(
        expandX, expertIds, assistInfoForCombineOut, hcomEpName, epWorldSize, epRankId, moeExpertNum, expertSharedType,
        sharedExpertNum, sharedExpertRankNum, globalBS, commQuantMode, commType, nullptr, quantExpandXOut,
        commCmdInfoOutforCombine, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] iter=%d aclnnMoeDistributeCombineSetupGetWorkspaceSize failed. ret = %d\n", iter, ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] iter=%d CombineSetup aclrtMalloc failed. ret = %d\n", iter, ret);
                  return ret);
    }

    ret = aclnnMoeDistributeCombineSetup(workspaceAddr, workspaceSize, executor, args.combinesetupstream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] iter=%d aclnnMoeDistributeCombineSetup failed. ret = %d\n", iter, ret);
              FreeDeviceAddr(workspaceAddr); return ret);

    ret = aclrtSynchronizeStreamWithTimeout(args.combinesetupstream, timeOut);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] iter=%d CombineSetup sync failed. ret = %d\n", iter, ret);
              FreeDeviceAddr(workspaceAddr); return ret);
    FreeDeviceAddr(workspaceAddr);
    workspaceAddr = nullptr;
    workspaceSize = 0;

    ret = aclnnMoeDistributeCombineTeardownGetWorkspaceSize(
        expandX, quantExpandXOut, expertIds, expandIdx, expertScales, commCmdInfoOutforCombine, nullptr, nullptr,
        hcomEpName, epWorldSize, epRankId, moeExpertNum, expertSharedType, sharedExpertNum, sharedExpertRankNum,
        globalBS, commQuantMode, commType, nullptr, xOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] iter=%d aclnnMoeDistributeCombineTeardownGetWorkspaceSize failed. ret = %d\n", iter,
                        ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] iter=%d CombineTeardown aclrtMalloc failed. ret = %d\n", iter, ret); return ret);
    }

    ret = aclnnMoeDistributeCombineTeardown(workspaceAddr, workspaceSize, executor, args.combineteardownstream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] iter=%d aclnnMoeDistributeCombineTeardown failed. ret = %d\n", iter, ret);
              FreeDeviceAddr(workspaceAddr); return ret);

    ret = aclrtSynchronizeStreamWithTimeout(args.combineteardownstream, timeOut);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] iter=%d CombineTeardown sync failed. ret = %d\n", iter, ret);
              FreeDeviceAddr(workspaceAddr); return ret);
    FreeDeviceAddr(workspaceAddr);

    LOG_PRINT("[INFO] device_%d iter=%d CombineSetup+Teardown ok (same EP domain reused).\n", args.rankId, iter);
    return 0;
}

int LaunchOneProcess(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return ret);

    // 仅在全部循环结束后销毁通信域，验证同域多轮复用
    auto destroyFunc = [&args]() {
        std::cout << "== begin to destroy " << std::endl;
        HcclCommDestroy(args.hcclEpComm);
        aclrtDestroyStream(args.combinesetupstream);
        aclrtDestroyStream(args.combineteardownstream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
    };
    auto guard = Guard<decltype(destroyFunc)>(destroyFunc);

    int64_t bs = 16;
    int64_t h = 4096;
    int64_t k = 6;
    int64_t expertSharedType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 32;
    int64_t commQuantMode = 0;
    int64_t tpWorldSize = 1;
    int64_t epWorldSize = DEV_NUM;
    int64_t commType = 2;
    int64_t timeOut = 100000000;

    int64_t globalBS = bs * epWorldSize;
    int64_t localExpertNum;
    int64_t localToken;

    if (args.epRankId < sharedExpertRankNum) {
        localExpertNum = 1;
        localToken = globalBS / sharedExpertRankNum;
    } else {
        localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
        localToken = globalBS * (localExpertNum < k ? localExpertNum : k);
    }

    uint64_t assistInfoForCombineOutSize = localToken * 128;

    std::vector<int64_t> expandXShape{tpWorldSize * localToken, h};
    std::vector<int64_t> expertIdsShape{bs, k};
    std::vector<int64_t> expertScalesShape{bs, k};
    std::vector<int64_t> expandIdxShape{bs * k};
    std::vector<int64_t> assistInfoForCombineOutShape{static_cast<int64_t>(assistInfoForCombineOutSize)};
    std::vector<int64_t> quantExpandXOutShape{tpWorldSize * localToken,
                                              AlignN(AlignN(h, 32) + AlignN(h, 8) / 8 * sizeof(float), 512)};
    std::vector<int64_t> commCmdInfoOutShapeforcombine{(localToken + epWorldSize) * 16};
    std::vector<int64_t> xOutShape{bs, h};

    std::vector<int16_t> expandXHostData(GetShapeSize(expandXShape), 0);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t tokenId = 0; tokenId < expertIdsShape[0]; tokenId++) {
        for (int32_t kId = 0; kId < expertIdsShape[1]; kId++) {
            expertIdsHostData.push_back(kId);
        }
    }
    std::vector<float> expertScalesHostData(GetShapeSize(expertScalesShape), 0);
    std::vector<int32_t> expandIdxHostData(GetShapeSize(expandIdxShape), 0);
    std::vector<int32_t> assistInfoForCombineOutHostData(GetShapeSize(assistInfoForCombineOutShape), 0);
    std::vector<int8_t> quantExpandXOutHostData(GetShapeSize(quantExpandXOutShape), 0);
    std::vector<int32_t> commCmdInfoOutforCombineHostData(GetShapeSize(commCmdInfoOutShapeforcombine), 0);
    std::vector<float> xOutHostData(GetShapeSize(xOutShape), 0);

    void *expandXDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *assistInfoForCombineOutDeviceAddr = nullptr;
    void *quantExpandXOutDeviceAddr = nullptr;
    void *commCmdInfoOutforCombineDeviceAddr = nullptr;
    void *xOutDeviceAddr = nullptr;

    aclTensor *expandX = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *expertScales = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *assistInfoForCombineOut = nullptr;
    aclTensor *quantExpandXOut = nullptr;
    aclTensor *commCmdInfoOutforCombine = nullptr;
    aclTensor *xOut = nullptr;

    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, aclDataType::ACL_FLOAT16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(assistInfoForCombineOutHostData, assistInfoForCombineOutShape,
                          &assistInfoForCombineOutDeviceAddr, aclDataType::ACL_INT32, &assistInfoForCombineOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(quantExpandXOutHostData, quantExpandXOutShape, &quantExpandXOutDeviceAddr,
                          aclDataType::ACL_INT8, &quantExpandXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(commCmdInfoOutforCombineHostData, commCmdInfoOutShapeforcombine,
                          &commCmdInfoOutforCombineDeviceAddr, aclDataType::ACL_INT32, &commCmdInfoOutforCombine);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_FLOAT16, &xOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("[INFO] device_%d reuse EP domain for %d iterations, group=%s\n", args.rankId, args.loopCount,
              hcomEpName);

    for (int iter = 0; iter < args.loopCount; ++iter) {
        ret = RunOneIteration(args, hcomEpName, expandX, expertIds, assistInfoForCombineOut, quantExpandXOut,
                              commCmdInfoOutforCombine, expertScales, expandIdx, xOut, epWorldSize, args.epRankId,
                              moeExpertNum, expertSharedType, sharedExpertNum, sharedExpertRankNum, globalBS,
                              commQuantMode, commType, timeOut, iter);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    LOG_PRINT("[INFO] device_%d all %d iterations passed without HcclCommDestroy between runs.\n", args.rankId,
              args.loopCount);

    DestroyTensor(expandX);
    DestroyTensor(expertIds);
    DestroyTensor(expertScales);
    DestroyTensor(expandIdx);
    DestroyTensor(assistInfoForCombineOut);
    DestroyTensor(quantExpandXOut);
    DestroyTensor(commCmdInfoOutforCombine);
    DestroyTensor(xOut);

    FreeDeviceAddr(expandXDeviceAddr);
    FreeDeviceAddr(expertIdsDeviceAddr);
    FreeDeviceAddr(expertScalesDeviceAddr);
    FreeDeviceAddr(expandIdxDeviceAddr);
    FreeDeviceAddr(assistInfoForCombineOutDeviceAddr);
    FreeDeviceAddr(quantExpandXOutDeviceAddr);
    FreeDeviceAddr(commCmdInfoOutforCombineDeviceAddr);
    FreeDeviceAddr(xOutDeviceAddr);

    return 0;
}

int main(int argc, char *argv[])
{
    int loopCount = DEFAULT_LOOP_COUNT;
    if (argc >= 2) {
        loopCount = std::atoi(argv[1]);
        if (loopCount <= 0) {
            LOG_PRINT("[ERROR] loop_count must be > 0, got %d\n", loopCount);
            return -1;
        }
    }

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);

    aclrtStream combineSetupStream[DEV_NUM];
    aclrtStream combineTeardownStream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&combineSetupStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&combineTeardownStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }

    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].epRankId = rankId;
        args[rankId].hcclEpComm = comms[rankId];
        args[rankId].combinesetupstream = combineSetupStream[rankId];
        args[rankId].combineteardownstream = combineTeardownStream[rankId];
        args[rankId].context = context[rankId];
        args[rankId].loopCount = loopCount;
        threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneProcess, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
