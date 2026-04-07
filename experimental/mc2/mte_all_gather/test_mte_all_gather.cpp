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
 * \file test_mte_all_gather.cpp
 * \brief AllGather <<<>>>测试样例
 *        通过MTE通信拿到其它卡数据，按rank顺序拼接到output（不做Dequant和ReduceSum）
 */
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <getopt.h>
#include <fstream>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include "hccl/hccl.h"
#include "acl/acl.h"
#include "hccl/hcom.h"
#include "securec.h"
#include "tiling/hccl/hccl_tiling.h"
#include "./kernel/mte_all_gather_tiling_data.h"
#include "./mte_all_gather.h"

using namespace std;
using namespace AscendC;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int64_t BLOCK_NUM = 64;
constexpr int64_t DEV_NUM = 4;
constexpr const char* PATH_PREFIX = "./golden/mteallgather_";
int streamWithTimeout = 10000;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateDeviceTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    return ACL_SUCCESS;
}

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);

int CreatTilingDataAndContext(const char* hcomName, aclrtStream stream, void **deviceTilingAddr, void **deviceContextAddr,
                               int64_t bs, int64_t hiddenSize, int64_t hcclBufferSize)
{
    MteAllGatherTilingData *tilingData = new MteAllGatherTilingData();
    if (tilingData == nullptr) {
        return -1;
    }
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    tilingData->mc2InitTiling = mc2InitTiling;
    tilingData->mc2CcTiling = mc2CcTiling;
    tilingData->mteAllGatherTilingInfo.bs = bs;
    tilingData->mteAllGatherTilingInfo.hiddenSize = hiddenSize;
    tilingData->mteAllGatherTilingInfo.aivNum = BLOCK_NUM;
    tilingData->mteAllGatherTilingInfo.totalWinSize = hcclBufferSize;

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 8, "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(3);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    int tilingSize = sizeof(struct MteAllGatherTilingData);
    aclrtMalloc(deviceTilingAddr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(*deviceTilingAddr, tilingSize, tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);

    HcclComm commHandle;
    HcomGetCommHandleByGroup(hcomName, &commHandle);
    void *mc2Context = nullptr;
    HcclAllocComResourceByTiling(commHandle, stream, tilingData, &mc2Context);
    if (mc2Context == nullptr) {
        return -1;
    }

    int64_t mc2ContextSize = 576;
    aclrtMalloc(deviceContextAddr, mc2ContextSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(*deviceContextAddr, mc2ContextSize, mc2Context, mc2ContextSize, ACL_MEMCPY_HOST_TO_DEVICE);

    delete tilingData;
    return ACL_SUCCESS;
}

struct Args {
    uint32_t rankId;
    uint32_t rankDim;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int AllocDeviceMemory(Args& args, const char* hcomName, void** xDevice, void** workspace,
                       void** tiling, void** mc2Context, int64_t bs, int64_t hiddenSize, int64_t hcclBufferSize,
                       int64_t workSpaceSize)
{
    CreatTilingDataAndContext(hcomName, args.stream, tiling, mc2Context, bs, hiddenSize, hcclBufferSize);

    std::vector<int64_t> xShape = {bs, hiddenSize};

    if (workSpaceSize > 0) {
        aclrtMalloc(workspace, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    long long xShapeSize = GetShapeSize(xShape);
    std::vector<int8_t> xHostData(xShapeSize, 0);
    CreateDeviceTensor(xHostData, xShape, xDevice);

    return ACL_SUCCESS;
}

int freeDeviceMemory(Args& args, void *xDeviceAddr, void *workspaceAddr,
                      void *tilingAddr, void *mc2ContextAddr, void *outputDeviceAddr, int64_t workSpaceSize)
{
    if (xDeviceAddr != nullptr)  aclrtFree(xDeviceAddr);
    if (outputDeviceAddr != nullptr) aclrtFree(outputDeviceAddr);
    if (mc2ContextAddr != nullptr) aclrtFree(mc2ContextAddr);
    if (tilingAddr != nullptr) aclrtFree(tilingAddr);
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtResetDevice(args.rankId);
    aclrtDestroyContext(args.context);
    return ACL_SUCCESS;
}

int LaunchOneThreadMteAllGather(Args &args)
{
    aclrtSetCurrentContext(args.context);
    char hcomName[128] = {0};
    HcclGetCommName(args.hcclComm, hcomName);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    int64_t bs = 16;
    int64_t hiddenSize = 8192;
    int64_t hcclBufferSize = 200;
    int64_t workSpaceSize = 256 * 1024 * 1024;

    void *xDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    void *tilingAddr = nullptr;
    void *mc2ContextAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    AllocDeviceMemory(args, hcomName, &xDeviceAddr, &workspaceAddr,
                      &tilingAddr, &mc2ContextAddr, bs, hiddenSize, hcclBufferSize, workSpaceSize);

    std::vector<int64_t> outputShape = {args.rankDim, bs, hiddenSize};
    long long outputShapeSize = GetShapeSize(outputShape) * sizeof(int8_t);
    aclrtMalloc(&outputDeviceAddr, outputShapeSize, ACL_MEM_MALLOC_HUGE_FIRST);

    mte_all_gather_demo(BLOCK_NUM, args.stream, (uint8_t*)xDeviceAddr,
                            (uint8_t*)outputDeviceAddr, (uint8_t*)workspaceAddr, (uint8_t*)mc2ContextAddr, (uint8_t*)tilingAddr);
    aclrtSynchronizeStreamWithTimeout(args.stream, streamWithTimeout);
    LOG_PRINT("[INFO] device_%d quant_all_gather execute successfully.\n", args.rankId);

    int8_t *output_host;
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), outputShapeSize);
    aclrtMemcpy(output_host, outputShapeSize, outputDeviceAddr, outputShapeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::string outputFile = std::string(PATH_PREFIX) + "output_npu_" + std::to_string(args.rankId) + ".bin";
    aclrtFreeHost(output_host);
    freeDeviceMemory(args, xDeviceAddr, workspaceAddr, tilingAddr, mc2ContextAddr, outputDeviceAddr, workSpaceSize);
    return 0;
}

int main(int argc, char *argv[])
{
    aclInit(nullptr);
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        aclrtSetDevice(rankId);
        aclrtCreateContext(&context[rankId], rankId);
        aclrtCreateStream(&stream[rankId]);
    }
    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }
    HcclComm comms[DEV_NUM];
    HcclCommInitAll(DEV_NUM, devices, comms);

    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].rankDim = DEV_NUM;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadMteAllGather, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
