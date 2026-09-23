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
 * \file test_aclnn_moe_distribute_combine_v2.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <cstdio>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
#include "aclnnop/aclnn_moe_distribute_combine_v2.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    HcclComm hcclEpComm;
    aclrtStream dispatchV2Stream;
    aclrtStream combineV2Stream;
    aclrtContext context;
};

const uint32_t EP_WORLD_SIZE = 2;
const uint32_t DEV_NUM = EP_WORLD_SIZE;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

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

int launchOneThreadDispatchV2AndCombineV2_A5(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, context = %p\n",
              args.rankId, hcomEpName, args.dispatchV2Stream, args.combineV2Stream, args.context);

    // 设置场景
    int64_t BS = 8;
    int64_t H = 7168;
    int64_t K = 1;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = EP_WORLD_SIZE - sharedExpertRankNum;
    int64_t quantMode = 0;
    int64_t globalBS = BS * EP_WORLD_SIZE;
    int64_t expertTokenNumsType = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t localExpertNum;
    int64_t A;
    if (args.epRankId < sharedExpertRankNum) {
        // 共享专家卡
        localExpertNum = 1;
        A = globalBS / sharedExpertRankNum;
    } else {
        // Moe专家卡
        localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
        A = globalBS * (localExpertNum < K ? localExpertNum : K);
    }
    std::string commAlg = "";

    /* 根据当前场景，构造device侧输入输出变量*/
    // 声明device侧输入输出变量
    void *xDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;
    void *expandXDeviceAddr = nullptr;
    void *dynamicScalesDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;
    void *epRecvCountsDeviceAddr = nullptr;
    void *tpRecvCountsDeviceAddr = nullptr;
    void *expandScalesDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *scales = nullptr;
    aclTensor *expertScales = nullptr;
    aclTensor *expandX = nullptr;
    aclTensor *dynamicScales = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *expertTokenNums = nullptr;
    aclTensor *epRecvCounts = nullptr;
    aclTensor *tpRecvCounts = nullptr;
    aclTensor *expandScales = nullptr;

    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{BS, H};
    std::vector<int64_t> expertIdsShape{BS, K};
    std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
    std::vector<int64_t> expertScalesShape{BS, K};
    std::vector<int64_t> expandXShape{A, H};
    std::vector<int64_t> dynamicScalesShape{A};
    std::vector<int64_t> expandIdxShape{A * 128};
    std::vector<int64_t> expertTokenNumsShape{localExpertNum};
    std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE};
    std::vector<int64_t> tpRecvCountsShape{1};
    std::vector<int64_t> expandScalesShape{A};

    long long xShapeSize = GetShapeSize(xShape);
    long long scalesShapeSize = GetShapeSize(scalesShape);
    long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
    long long expandXShapeSize = GetShapeSize(expandXShape);
    long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
    long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
    long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    long long epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
    long long tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
    long long expandScalesShapeSize = GetShapeSize(expandScalesShape);

    // 构造host侧变量
    std::vector<op::fp16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
        // 每个token发给moe专家{0, 1, ... k - 1}
        for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
            expertIdsHostData.push_back(k_id);
        }
    }
    std::vector<float> scalesHostData(scalesShapeSize, 0);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
    std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
    std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
    std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

    // 构造device侧变量
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr,
                          (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &dynamicScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr,
                          aclDataType::ACL_INT64, &expertTokenNums);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32,
                          &epRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32,
                          &tpRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expandScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    /* 声明算子执行必需变量 */
    uint64_t dispatchV2WorkspaceSize = 0;
    aclOpExecutor *dispatchV2Executor = nullptr;
    void *dispatchV2WorkspaceAddr = nullptr;

    uint64_t combineV2WorkspaceSize = 0;
    aclOpExecutor *combineV2Executor = nullptr;
    void *combineV2WorkspaceAddr = nullptr;

    /* 依次执行dispatchV2及combineV2算子 */
    // 调用dispatchV2算子第一阶段接口
    ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(
        x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr, expertScales, hcomEpName, EP_WORLD_SIZE,
        args.epRankId, moeExpertNum, "", 0, 0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode,
        globalBS, expertTokenNumsType, commAlg.c_str(), expandX, dynamicScales, expandIdx, expertTokenNums,
        epRecvCounts, tpRecvCounts, expandScales, &dispatchV2WorkspaceSize, &dispatchV2Executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2GetWorkspaceSize failed. ret = %d\n", ret);
              return ret);
    // 根据dispatchV2算子第一阶段接口计算出的workspaceSize申请device内存
    if (dispatchV2WorkspaceSize > 0) {
        ret = aclrtMalloc(&dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }
    // 调用dispatchV2算子第二阶段接口
    ret = aclnnMoeDistributeDispatchV2(dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, dispatchV2Executor,
                                       args.dispatchV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2 failed. ret = %d\n", ret);
              return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret);
              return ret);

    // 调用combineV2算子第一阶段接口
    ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(
        expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, "", 0, 0, expertShardType,
        sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode, groupListType, commAlg.c_str(), x,
        &combineV2WorkspaceSize, &combineV2Executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2GetWorkspaceSize failed. ret = %d\n", ret);
              return ret);
    // 根据combineV2算子第一阶段接口计算出的workspaceSize申请device内存
    if (combineV2WorkspaceSize > 0) {
        ret = aclrtMalloc(&combineV2WorkspaceAddr, combineV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }
    // 调用combineV2算子第二阶段接口
    ret = aclnnMoeDistributeCombineV2(combineV2WorkspaceAddr, combineV2WorkspaceSize, combineV2Executor,
                                      args.combineV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2 failed. ret = %d\n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret);
              return ret);

    LOG_PRINT("[INFO] device_%d aclnnMoeDistributedispatchV2 and aclnnMoeDistributecombineV2 execute successfully.\n",
              args.rankId);

    // 释放device资源
    if (dispatchV2WorkspaceSize > 0) {
        aclrtFree(dispatchV2WorkspaceAddr);
    }
    if (combineV2WorkspaceSize > 0) {
        aclrtFree(combineV2WorkspaceAddr);
    }
    DestroyTensor(x);
    DestroyTensor(expertIds);
    DestroyTensor(scales);
    DestroyTensor(expertScales);
    DestroyTensor(expandX);
    DestroyTensor(dynamicScales);
    DestroyTensor(expandIdx);
    DestroyTensor(expertTokenNums);
    DestroyTensor(epRecvCounts);
    DestroyTensor(tpRecvCounts);
    DestroyTensor(expandScales);

    FreeDeviceAddr(xDeviceAddr);
    FreeDeviceAddr(expertIdsDeviceAddr);
    FreeDeviceAddr(scalesDeviceAddr);
    FreeDeviceAddr(expertScalesDeviceAddr);
    FreeDeviceAddr(expandXDeviceAddr);
    FreeDeviceAddr(dynamicScalesDeviceAddr);
    FreeDeviceAddr(expandIdxDeviceAddr);
    FreeDeviceAddr(expertTokenNumsDeviceAddr);
    FreeDeviceAddr(epRecvCountsDeviceAddr);
    FreeDeviceAddr(tpRecvCountsDeviceAddr);
    FreeDeviceAddr(expandScalesDeviceAddr);

    HcclCommDestroy(args.hcclEpComm);
    aclrtDestroyStream(args.dispatchV2Stream);
    aclrtDestroyStream(args.combineV2Stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

int run_example_on_A5()
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

    aclrtStream dispatchV2Stream[DEV_NUM];
    aclrtStream combineV2Stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&dispatchV2Stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&combineV2Stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
    }

    int32_t devicesEp[EP_WORLD_SIZE];
    for (uint32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
        devicesEp[epId] = epId;
    }
    // 初始化ep通信域.
    HcclComm commsEp[EP_WORLD_SIZE];
    ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep failed. ret = %d\n", ret); return ret);

    Args args[DEV_NUM];
    // 各线程调用各卡执行算子
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        uint32_t epRankId = rankId;

        args[rankId].rankId = rankId;
        args[rankId].epRankId = epRankId;
        args[rankId].hcclEpComm = commsEp[epRankId];
        args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
        args[rankId].combineV2Stream = combineV2Stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow)
                                  std::thread(&launchOneThreadDispatchV2AndCombineV2_A5, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");
    return 0;
}

int main(int argc, char *argv[])
{
    LOG_PRINT("Example on <Atlas A5> will be executed!\n");
    run_example_on_A5();

    return 0;
}
