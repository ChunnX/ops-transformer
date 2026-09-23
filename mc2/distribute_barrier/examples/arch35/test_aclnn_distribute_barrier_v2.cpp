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
 * \file test_aclnn_distribute_barrier_v2.cpp
 * \brief
 */
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_moe_distribute_dispatch_v3.h"
#include "aclnnop/aclnn_distribute_barrier_v2.h"
#include "aclnnop/aclnn_moe_distribute_combine_v3.h"

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
    uint32_t tpRankId;
    HcclComm hcclEpComm;
    HcclComm hcclEpBarrierComm;
    HcclComm hcclTpComm;
    aclrtStream dispatchStream;
    aclrtStream barrierStream;
    aclrtStream combineStream;
    aclrtContext context;
};

constexpr uint32_t EP_WORLD_SIZE = 2;
constexpr uint32_t TP_WORLD_SIZE = 1;
constexpr uint32_t DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

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

int LaunchOneProcessDispatchAndCombine(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
    char hcomEpBarrierName[128] = {0};
    ret = HcclGetCommName(args.hcclEpBarrierComm, hcomEpBarrierName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpBarrierCommName failed, ret %d\n", ret); return -1);
    char hcomTpName[128] = {0};

    int64_t Bs = 8;
    int64_t H = 7168;
    int64_t K = 2;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 2;
    int64_t quantMode = 0;
    int64_t globalBs = Bs * EP_WORLD_SIZE;
    int64_t expertTokenNumsType = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupList_type = 1;
    int64_t localExpertNum;
    int64_t A;
    if (args.epRankId < sharedExpertRankNum) {
        localExpertNum = 1;
        A = globalBs / sharedExpertRankNum;
    } else {
        localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
        A = globalBs * (localExpertNum < K ? localExpertNum : K);
    }

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

    void *elasticInfoDeviceAddr = nullptr;
    void *timeOutDeviceAddr = nullptr;

    void *xOutDeviceAddr = nullptr;
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

    aclTensor *elasticInfo = nullptr;
    aclTensor *timeOut = nullptr;

    aclTensor *xOut = nullptr;

    std::vector<int64_t> xShape{Bs, H};
    std::vector<int64_t> expertIdsShape{Bs, K};
    std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
    std::vector<int64_t> expertScalesShape{Bs, K};
    std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
    std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
    std::vector<int64_t> expandIdxShape{A * 128};
    std::vector<int64_t> expertTokenNumsShape{localExpertNum};
    std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
    std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE * localExpertNum};
    std::vector<int64_t> expandScalesShape{A};

    std::vector<int64_t> elasticInfoShape{4 + EP_WORLD_SIZE * 2};
    std::vector<int64_t> timeOutShape{1};
    std::vector<int64_t> xOutShape{Bs, H};

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t scalesShapeSize = GetShapeSize(scalesShape);
    int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
    int64_t expandXShapeSize = GetShapeSize(expandXShape);
    int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
    int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
    int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
    int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
    int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);
    int64_t timeOutShapeSize = GetShapeSize(timeOutShape);
    int64_t xOutShapeSize = GetShapeSize(xOutShape);

    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
        for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
            expertIdsHostData.push_back(k_id);
        }
    }

    std::vector<float> scalesHostData(scalesShapeSize, 0.1);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);
    std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
    std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
    std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

    int32_t isElastic = 0;
    int32_t rankNumAfterElastic = 2;
    int32_t sharedExpertRankNumAfterElastic = sharedExpertRankNum;
    int32_t moeExpertNumAfterElastic = rankNumAfterElastic - sharedExpertRankNumAfterElastic;
    std::unordered_set<int16_t> availableRank{0, 1, /*2, 3, 4, 5,*/ 6, 7};
    std::vector<int32_t> elasticInfoHostData{isElastic,
                                             rankNumAfterElastic,
                                             sharedExpertRankNumAfterElastic,
                                             moeExpertNumAfterElastic,
                                             0,
                                             1,
                                             -1,
                                             -1,
                                             -1,
                                             -1,
                                             2,
                                             3,
                                             0,
                                             1,
                                             6,
                                             7,
                                             -1,
                                             -1,
                                             -1,
                                             -1};
    std::vector<int32_t> timeOutHostData(timeOutShapeSize, 1000000);
    std::vector<int16_t> xOutHostData(xOutShapeSize, 0);

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
    ret = CreateAclTensor(elasticInfoHostData, elasticInfoShape, &elasticInfoDeviceAddr, aclDataType::ACL_INT32,
                          &elasticInfo);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(timeOutHostData, timeOutShape, &timeOutDeviceAddr, aclDataType::ACL_INT32, &timeOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    uint64_t dispatchWorkspaceSize = 0;
    aclOpExecutor *dispatchExecutor = nullptr;
    void *dispatchWorkspaceAddr = nullptr;

    uint64_t barrierWorkspaceSize = 0;
    aclOpExecutor *barrierExecutor = nullptr;
    void *barrierWorkspaceAddr = nullptr;

    uint64_t combineWorkspaceSize = 0;
    aclOpExecutor *combineExecutor = nullptr;
    void *combineWorkspaceAddr = nullptr;

    /**************************************** 调用dispatch warm up********************************************/
    ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(
        x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr, expertScales, nullptr, hcomEpName, EP_WORLD_SIZE,
        args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, nullptr, 0, 0, 0, expandX, dynamicScales,
        expandIdx, expertTokenNums, epRecvCounts, tpRecvCounts, expandScales, &dispatchWorkspaceSize,
        &dispatchExecutor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    if (dispatchWorkspaceSize > 0) {
        ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtMalloc workspace failed. ret = %d \n", ret);
                  return ret);
    }
    // 调用第二阶段接口
    ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize, dispatchExecutor,
                                       args.dispatchStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);
              return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] warm up aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);

    /**************************************** 调用dispatch ********************************************/
    if (availableRank.find(args.rankId) != availableRank.end()) {
        // 调用第一阶段接口
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(
            x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr, expertScales, nullptr, hcomEpName, EP_WORLD_SIZE,
            args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId, expertShardType, sharedExpertNum,
            sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, nullptr, 0, 0, 0, expandX, dynamicScales,
            expandIdx, expertTokenNums, epRecvCounts, tpRecvCounts, expandScales, &dispatchWorkspaceSize,
            &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret);
                  return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);
                      return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize, dispatchExecutor,
                                           args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);
                  return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                  return ret);
    }
    /**************************************** 调用barrier warm up********************************************/
    ret = aclnnDistributeBarrierV2GetWorkspaceSize(expandX, nullptr, nullptr, hcomEpBarrierName, EP_WORLD_SIZE,
                                                   &barrierWorkspaceSize, &barrierExecutor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] warm up aclnnDistributeBarrierV2GetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    if (barrierWorkspaceSize > 0) {
        ret = aclrtMalloc(&barrierWorkspaceAddr, barrierWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtMalloc workspace failed. ret = %d \n", ret);
                  return ret);
    }
    // 调用第二阶段接口
    ret = aclnnDistributeBarrierV2(barrierWorkspaceAddr, barrierWorkspaceSize, barrierExecutor, args.barrierStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclnnDistributeBarrierV2 failed. ret = %d \n", ret);
              return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.barrierStream, 10000);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] warm up aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);

    /**************************************** 调用barrier********************************************/
    if (availableRank.find(args.rankId) != availableRank.end()) {
        ret = aclnnDistributeBarrierV2GetWorkspaceSize(expandX, timeOut, elasticInfo, hcomEpBarrierName, EP_WORLD_SIZE,
                                                       &barrierWorkspaceSize, &barrierExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] aclnnDistributeBarrierV2GetWorkspaceSize failed. ret = %d \n", ret);
                  return ret);

        if (barrierWorkspaceSize > 0) {
            ret = aclrtMalloc(&barrierWorkspaceAddr, barrierWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);
                      return ret);
        }

        // 调用第二阶段接口
        ret = aclnnDistributeBarrierV2(barrierWorkspaceAddr, barrierWorkspaceSize, barrierExecutor, args.barrierStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnDistributeBarrierV2 failed. ret = %d \n", ret);
                  return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.barrierStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                  return ret);
    }
    /**************************************** 调用combine ********************************************/
    // 调用第一阶段接口
    if (availableRank.find(args.rankId) != availableRank.end()) {
        ret = aclnnMoeDistributeCombineV3GetWorkspaceSize(
            expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, elasticInfo, nullptr, nullptr, nullptr, nullptr, hcomEpName, EP_WORLD_SIZE, args.epRankId,
            moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId, expertShardType, sharedExpertNum,
            sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupList_type, nullptr, 0, 0, 0, xOut,
            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3GetWorkspaceSize failed. ret = %d \n", ret);
                  return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);
                      return ret);
        }

        // 调用第二阶段接口
        ret = aclnnMoeDistributeCombineV3(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor,
                                          args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3 failed. ret = %d \n", ret);
                  return ret);
        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                  return ret);
        LOG_PRINT(
            "[INFO] device_%d aclnnMoeDistributeDispatchV3, aclnnDistributeBarrierV2 and aclnnMoeDistributeCombineV3 \
                    execute successfully.\n",
            args.rankId);
    }

    // 释放device资源
    if (dispatchWorkspaceSize > 0) {
        aclrtFree(dispatchWorkspaceAddr);
    }
    if (combineWorkspaceSize > 0) {
        aclrtFree(combineWorkspaceAddr);
    }
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (scales != nullptr) {
        aclDestroyTensor(scales);
    }
    if (expertScales != nullptr) {
        aclDestroyTensor(expertScales);
    }
    if (expandX != nullptr) {
        aclDestroyTensor(expandX);
    }
    if (dynamicScales != nullptr) {
        aclDestroyTensor(dynamicScales);
    }
    if (expandIdx != nullptr) {
        aclDestroyTensor(expandIdx);
    }
    if (expertTokenNums != nullptr) {
        aclDestroyTensor(expertTokenNums);
    }
    if (epRecvCounts != nullptr) {
        aclDestroyTensor(epRecvCounts);
    }
    if (tpRecvCounts != nullptr) {
        aclDestroyTensor(tpRecvCounts);
    }
    if (expandScales != nullptr) {
        aclDestroyTensor(expandScales);
    }
    if (elasticInfo != nullptr) {
        aclDestroyTensor(elasticInfo);
    }
    if (timeOut != nullptr) {
        aclDestroyTensor(elasticInfo);
    }
    if (xOut != nullptr) {
        aclDestroyTensor(xOut);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }
    if (expertScalesDeviceAddr != nullptr) {
        aclrtFree(expertScalesDeviceAddr);
    }
    if (expandXDeviceAddr != nullptr) {
        aclrtFree(expandXDeviceAddr);
    }
    if (dynamicScalesDeviceAddr != nullptr) {
        aclrtFree(dynamicScalesDeviceAddr);
    }
    if (expandIdxDeviceAddr != nullptr) {
        aclrtFree(expandIdxDeviceAddr);
    }
    if (expertTokenNumsDeviceAddr != nullptr) {
        aclrtFree(expertTokenNumsDeviceAddr);
    }
    if (epRecvCountsDeviceAddr != nullptr) {
        aclrtFree(epRecvCountsDeviceAddr);
    }
    if (expandScalesDeviceAddr != nullptr) {
        aclrtFree(expandScalesDeviceAddr);
    }
    if (tpRecvCountsDeviceAddr != nullptr) {
        aclrtFree(tpRecvCountsDeviceAddr);
    }
    if (elasticInfoDeviceAddr != nullptr) {
        aclrtFree(elasticInfoDeviceAddr);
    }
    if (timeOutDeviceAddr != nullptr) {
        aclrtFree(timeOutDeviceAddr);
    }
    if (xOutDeviceAddr != nullptr) {
        aclrtFree(xOutDeviceAddr);
    }

    HcclCommDestroy(args.hcclEpComm);
    HcclCommDestroy(args.hcclEpBarrierComm);
    HcclCommDestroy(args.hcclTpComm);
    aclrtDestroyStream(args.dispatchStream);
    aclrtDestroyStream(args.combineStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtInit failed, ret = %d\n", ret); return ret);

    aclrtStream dispatchStream[DEV_NUM];
    aclrtStream barrierStream[DEV_NUM];
    aclrtStream combineStream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&dispatchStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&barrierStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&combineStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
    }

    int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (uint32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        for (uint32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
        }
    }

    HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (uint32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret);
                  return ret);
    }

    HcclComm commsEpBarrier[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (uint32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEpBarrier[tpId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll epBarrier %d failed, ret %d\n", tpId, ret);
                  return ret);
    }

    int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
    for (uint32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
        for (uint32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
        }
    }

    HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
    for (uint32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
        ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll tp %d failed, ret %d\n", epId, ret);
                  return ret);
    }

    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        uint32_t epRankId = rankId / TP_WORLD_SIZE;
        uint32_t tpRankId = rankId % TP_WORLD_SIZE;

        args[rankId].rankId = rankId;
        args[rankId].epRankId = epRankId;
        args[rankId].tpRankId = tpRankId;
        args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
        args[rankId].hcclEpBarrierComm = commsEpBarrier[tpRankId][epRankId];
        args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
        args[rankId].dispatchStream = dispatchStream[rankId];
        args[rankId].barrierStream = barrierStream[rankId];
        args[rankId].combineStream = combineStream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow)
                                  std::thread(&LaunchOneProcessDispatchAndCombine, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");

    return 0;
}
